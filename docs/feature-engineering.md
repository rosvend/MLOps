# Feature engineering (stage 5)

Everything the pipeline does is declared in `config/features/default.yaml` and validated by
`FeatureSpec` in `src/features/spec.py`. No column list, bound, vocabulary or threshold is a
Python literal, so a run records the exact spec that produced it and an experiment can vary it
with a Hydra override rather than a code change.

## Column roles

Four roles are reserved and never reach a model. Everything else is a feature.

| Role | Column | Why it is not a feature |
| --- | --- | --- |
| Entity key | `cliente_id` | A join key. Surrogate (`CLI-0000001`…) until the business exports the real client ID. |
| Event timestamp | `fecha_prestamo` | It *is* the vintage. Feast needs it for point-in-time joins; a model must not read it. |
| Target | `Pago_atiempo` | — |
| Withheld | `puntaje` | Leakage: 87 % share the maximum value and none of them defaulted. |
| Withheld | `mes_prestamo` | Censoring control derived from the event timestamp. |

`fecha_prestamo` was previously *not* reserved, so it reached the model view and the fitted
transforms coerced it to a number — 1.7362608e+18 nanoseconds, correlating **1.0000** with the
loan date. That is the vintage leak `mes_prestamo` was withheld to prevent, arriving through the
back door. The highest correlation between any engineered column and the loan date is now 0.12.

## Two layers, and why they are separate

**Row-independent** (`clean` → `derive` → `contract`). Sentinel nulling, unit scaling, ratios,
bands and missingness indicators. A loan produces identical values alone or inside a portfolio.
This is what `make features` materialises.

**Fitted** (`src/features/engineering.py`). A p99 cap and a median are statistics of a sample,
so they are fitted on the training rows and applied to the test rows.

The split is the important design decision here:

> The feature store holds only the row-independent layer. Materialising winsorised or imputed
> values would bake statistics taken over the whole book into the store, and every future
> train/test split would silently inherit them. The fitted transforms stay inside the model
> pipeline, refitted on each training split.

## What the fitted layer does

Order matters: **winsorise → impute → log1p**. Imputing before the log keeps the filled value on
the same scale as the median it came from; doing it after put raw pesos (2 900 000) next to log
values (~15) in the same column.

Only columns with a real tail are touched. `edad_cliente` (skew 0.27) and `puntaje_datacredito`
(−0.71) are left alone.

| Column | skew before | skew after |
| --- | ---: | ---: |
| `dti` | 97.5 | 0.52 |
| `pti` | 96.7 | 1.85 |
| `monto_sobre_ingreso` | 95.3 | 0.87 |
| `saldo_mora` | 40.6 | 0.00 |
| `total_otros_prestamos` | 38.5 | −3.77 |
| `saldo_total` | 20.2 | −2.47 |
| `ratio_ingreso_declarado_bureau` | 19.6 | 2.66 |
| `salario_cliente` | 19.5 | 0.42 |

The ratios are why this layer exists: `dti`'s largest value sat 1739× above its own 99th
percentile, `pti`'s 363×, `monto_sobre_ingreso`'s 248×. A few columns overshoot into mild
negative skew, which is the ordinary cost of a log on a zero-inflated column and far less
damaging than the tail it removes.

Imputation is only safe because `falta_<column>` still records the fact. All eight indicators
carry signal, lifting the default rate 1.18× (`tendencia_ingresos`) to 1.54× (`edad_cliente`)
over the 4.75 % base.

Encoding needs **no fitted state**: `clean()` freezes the category sets, so a batch containing no
`tipo_credito == "Otro"` still emits that column. That is what stops the feature space drifting
between training and serving.

Standardisation is deliberately absent. Winsorise/log/encode is representation; scaling is
model-specific and belongs in the model's own pipeline.

35 contract columns become 46 engineered ones, with no nulls remaining.

## Running it

```bash
make features                                                   # -> data/processed/features.parquet
uv run python -m src.pipelines.features features.engineering.winsor_quantile=0.95
```

The output is keyed by `cliente_id` with `fecha_prestamo` as the event timestamp: 10 763 loans ×
37 columns, which is what stage 6 will register with Feast.

## Known limitation

One ID per row means there are no repeat borrowers to find. The surrogate key exists so stage 6
can be built; borrower history — prior loans, prior defaults, time since last loan — has to wait
for the real client identifier.
