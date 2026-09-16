# Feature store (stage 6)

Feast registers the stage-5 feature table for **offline retrieval only**, so the training phase
can pull a point-in-time-correct training set instead of reading a parquet by hand.

```
feature_repo/
  feature_store.yaml   project, local provider, file offline store, online_store: null
  entities.py          the cliente entity, named from the spec
  features.py          FileSource + four feature views, typed from the parquet
  data/registry.db     built by feast apply; gitignored
src/features/feast_types.py   the one Arrow -> Feast type table, shared by both sides
```

`entity_name`, `verify_sample` and `event_timestamp_timezone` live in
`config/features/default.yaml` and are validated by `FeatureSpec`, so neither the repo
definitions nor the verification script repeats them.

## There is no load step

With a file offline store, `data/processed/features.parquet` **is** the offline store. `feast
apply` registers metadata pointing at it — no data is copied. `feast materialize` loads the
*online* store, which this stage deliberately does not have, so it is never run.

`online_store: null` is set explicitly rather than left to the provider default, which would give
a local sqlite online store nobody asked for.

## Feature views

Grouped so a consumer can pull only what it needs. The grouping lives in
`config/features/default.yaml`. `FeatureSpec` rejects a column claimed by two views or by a
reserved role, but it never sees the built table, so it cannot notice an *omission* — that half
is checked by `test_the_feature_views_cover_the_table_exactly`, which compares the views against
the parquet's actual columns. Both halves are needed: without the second, deleting a column from
a view validates cleanly, still lands in the parquet, and is silently dropped from every
consumer built off `spec.feature_views`.

| View | Columns | Holds |
| --- | ---: | --- |
| `credito_solicitud` | 8 | what the applicant declares on the form |
| `credito_bureau` | 11 | the DataCrédito block, an origination-time pull |
| `credito_derivadas` | 8 | ratios, bands and contrast flags |
| `credito_faltantes` | 8 | the `falta_*` indicators |

`ttl=timedelta(days=0)` — unlimited. A non-zero TTL would silently drop loans older than the
window from every point-in-time join, which on a 15-month book is most of them.

## Types

Feast's type system has no Categorical, and its file store reads through pyarrow, so
`build_feature_table` writes the four categorical columns as strings. Nothing is lost: the fixed
vocabulary lives in the spec and is enforced by `clean()` and the pandera schema, not by parquet
dictionary encoding.

| Arrow (in the parquet) | Feast |
| --- | --- |
| `int64` | `Int64` |
| `double` | `Float64` |
| `bool` | `Bool` |
| `string` | `String` |
| `timestamp[ns, tz=…]` | `UnixTimestamp` |

Types are read from the parquet rather than retyped by hand, and `feast_type()` **raises** on
anything outside that table — an upstream dtype change fails the build instead of quietly
reshaping the store. The mapping lives in `src/features/feast_types.py` so the views are built
from it and checked against it.

### The registry can go stale

`feast apply` snapshots the parquet's dtypes into `registry.db`. `make features` alone
regenerates the parquet and leaves the registry behind, and Feast's file offline store does
**not** validate what it reads against the registered schema: a column retyped `int64 → double`
retrieves happily and comes back as the new type while the registry still claims the old one. A
column *removed* from the parquet is loud (`KeyError`), a retyped one is silent.

So `stale_fields()` runs first in `make feast-verify`, which now depends on `feast-apply`, and it
exits non-zero naming the drifted fields rather than reporting a green point-in-time check
against a registry that no longer describes its own data.

Nulls are preserved as nulls. 16 of the 37 columns carry them, and they carry signal: the
`falta_*` indicators exist precisely because absence is predictive.

## Event timestamps are UTC

Feast requires timezone-aware event timestamps; the source carries no zone, so
`build_feature_table` localises `fecha_prestamo` to UTC (configurable via
`event_timestamp_timezone`, which `FeatureSpec` now checks against the IANA database — a typo
used to survive validation and surface as `UnknownTimeZoneError` halfway through the pipeline).
Point-in-time joins depend only on ordering, which localising cannot change. Left naive, Feast's
dask offline store fails with a dtype comparison error inside its TTL filter.

A naive `entity_df` is safe too: Feast defaults naive timestamps to UTC, which is the same
convention the table was written under, so naive and aware queries return the same rows.

## Pulling a training set

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")
entity_df = pd.DataFrame({
    "cliente_id": ["CLI-0000001", "CLI-0000002"],
    "event_timestamp": pd.to_datetime(["2025-01-07 14:40", "2025-01-09 11:18"], utc=True),
})
training = store.get_historical_features(
    entity_df=entity_df,
    features=["credito_solicitud:capital_prestado", "credito_bureau:puntaje_datacredito"],
).to_df()
```

The label is joined from outside the store: `Pago_atiempo` never entered it, so it cannot come
back out.

> **Join it on `cliente_id`, never by position.** Feast returns the rows sorted by event
> timestamp and resets the index, so the result is *not* positionally aligned with the
> `entity_df` that was passed in. On a 300-loan shuffled sample, 299 of 300 rows come back in a
> different position than they went in, and because the index is reset there is nothing to
> notice. Pairing features with labels by position therefore mislabels almost the whole training
> set, silently. `test_retrieval_is_not_positionally_aligned_with_the_entity_df` pins this, and
> `make feast-verify` checks every retrieved value against the source loan-by-loan — counting
> non-nulls would not catch a scrambled answer.

Two more Feast behaviours worth knowing before building an `entity_df`: identical
`(cliente_id, event_timestamp)` pairs are **deduplicated**, and rows whose timestamp predates
the loan are **dropped entirely** rather than returned as nulls. Both mean the result can be
shorter than the `entity_df` — another reason to join on the key.

## Point-in-time correctness

`make feast-verify` asks for the same `verify_sample` loans at three moments:

```
  un día antes del desembolso:   0 filas,   0 con features
  en el desembolso          : 200 filas, 200 con features
  30 días después           : 200 filas, 200 con features
```

Zero features before origination is the property worth having. A store that answered there would
be handing a model information the applicant's file did not contain at decision time. Feast
expresses "nothing known yet" by dropping the row rather than returning nulls; the test accepts
either, because what matters is that no value crosses back over its own event timestamp.

That last sentence is asserted over the **whole book**, not a sample:
`test_nothing_crosses_back_over_its_own_event_timestamp` joins all 10 763 retrieved rows against
their source and requires `fecha_prestamo <= event_timestamp` for every one. The boundary is
inclusive — a query at exactly the origination timestamp returns the loan, one nanosecond
earlier returns nothing.

`ttl=timedelta(days=0)` is genuinely unlimited rather than a zero-width window: Feast's dask
offline store guards the TTL filter with `if feature_view.ttl and ttl.total_seconds() != 0`, so a
zero TTL leaves only the `event_timestamp <= entity_timestamp` bound. The `entity_df` timestamp
range is carried as retrieval *metadata* and never used to filter the source, so no feature row
is dropped for being older than the earliest row in the query.

## Known limitation

One ID per row means the entity is a loan, not a borrower. Point-in-time joins are correct, but
there is no borrower history to aggregate — no prior loans, no prior defaults, no time since last
application — until the business exports the real client identifier.
