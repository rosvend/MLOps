# Feature store (stage 6)

Feast registers the stage-5 feature table for **offline retrieval only**, so the training phase
can pull a point-in-time-correct training set instead of reading a parquet by hand.

```
feature_repo/
  feature_store.yaml   project, local provider, file offline store, online_store: null
  entities.py          the cliente entity
  features.py          FileSource + four feature views, typed from the parquet
  data/registry.db     built by feast apply; gitignored
```

## There is no load step

With a file offline store, `data/processed/features.parquet` **is** the offline store. `feast
apply` registers metadata pointing at it — no data is copied. `feast materialize` loads the
*online* store, which this stage deliberately does not have, so it is never run.

`online_store: null` is set explicitly rather than left to the provider default, which would give
a local sqlite online store nobody asked for.

## Feature views

Grouped so a consumer can pull only what it needs. The grouping lives in
`config/features/default.yaml`, and `FeatureSpec` validates that the views cover every feature
column exactly once — none duplicated, none forgotten, none reserved.

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
reshaping the store. A test asserts every registered `Field` matches the parquet under this
mapping.

Nulls are preserved as nulls. 16 of the 37 columns carry them, and they carry signal: the
`falta_*` indicators exist precisely because absence is predictive.

## Event timestamps are UTC

Feast requires timezone-aware event timestamps; the source carries no zone, so
`build_feature_table` localises `fecha_prestamo` to UTC (configurable via
`event_timestamp_timezone`). Point-in-time joins depend only on ordering, which localising cannot
change. Left naive, Feast's dask offline store fails with a dtype comparison error inside its TTL
filter.

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

## Point-in-time correctness

`make feast-verify` asks for the same 200 loans at three moments:

```
  un día antes del desembolso:   0 filas,   0 con features
  en el desembolso          : 200 filas, 200 con features
  30 días después           : 200 filas, 200 con features
```

Zero features before origination is the property worth having. A store that answered there would
be handing a model information the applicant's file did not contain at decision time. Feast
expresses "nothing known yet" by dropping the row rather than returning nulls; the test accepts
either, because what matters is that no value crosses back over its own event timestamp.

## Known limitation

One ID per row means the entity is a loan, not a borrower. Point-in-time joins are correct, but
there is no borrower history to aggregate — no prior loans, no prior defaults, no time since last
application — until the business exports the real client identifier.
