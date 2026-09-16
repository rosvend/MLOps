# Feature engineering (stage 5)

`src/features/engineering.py`. Everything before this stage is a pure function of one row.
These transforms are not: a p99 cap and a median are statistics of a sample, so they are
fitted on the training rows and applied to the test rows.

## What gets done, and why

Measured on all 10 763 loans. Only columns with a genuine tail are touched:

| Column | skew | max / p99 | treatment |
| --- | ---: | ---: | --- |
| `dti` | 97.5 | 1739× | winsorise p99, log1p |
| `pti` | 96.7 | 363× | winsorise p99, log1p |
| `monto_sobre_ingreso` | 95.3 | 248× | winsorise p99, log1p |
| `saldo_mora` | 40.6 | — | winsorise p99, log1p |
| `total_otros_prestamos` | 38.5 | 378× | winsorise p99, log1p |
| `saldo_total` | 20.2 | 14× | winsorise p99, log1p |
| `ratio_ingreso_declarado_bureau` | 19.6 | 11× | winsorise p99, log1p |
| `salario_cliente` | 19.5 | 29× | winsorise p99, log1p |
| `edad_cliente` | 0.27 | 1.0× | left alone |
| `puntaje_datacredito` | −0.71 | 1.1× | left alone |

The ratio columns are the reason this stage exists: a `dti` whose largest value sits 1739×
above the 99th percentile will dominate any fit that is not protected from it.

Which column gets which treatment is a **frozen list**, not a rule evaluated at fit time.
Selecting by measured skew would let the output schema differ between folds.

## Order of operations

1. **Winsorise** at the training p99.
2. **Impute** with the training median — before the log, so the filled value is on the same
   scale as the median it came from.
3. **log1p** the monetary and ratio columns.
4. **One-hot encode** the categoricals.

Imputation is only safe because `falta_<column>` still records that the value was missing;
all eight indicators carry signal, lifting the default rate 1.18× to 1.54×.

Encoding needs **no fitted state**. `clean()` freezes the category sets, so a batch that
happens to contain no `tipo_credito == "Otro"` still emits that column. That is what stops the
feature space drifting between training and serving.

Standardisation is deliberately **not** here. Winsorise/log/encode is representation; scaling
is model-specific and belongs in the model's own pipeline — a gradient booster does not want it.

36 contract columns become 47 engineered ones.

## Results

5-fold stratified CV, shuffled, seed 0 — the same split the heuristic's published 0.676 uses:

| Model | AUC | Gini |
| --- | ---: | ---: |
| Heuristic scorecard (baseline) | 0.6760 | 0.3520 |
| **Logistic regression** on engineered features | **0.6895** | **0.3790** |
| Gradient boosting on engineered features | 0.6541 | 0.3081 |

## The result that matters more

Train on the oldest 75 % of vintages, test on the newest 25 %:

| Model | AUC | Gini |
| --- | ---: | ---: |
| Logistic regression | 0.6216 | 0.2432 |
| **Heuristic scorecard** | **0.6468** | **0.2937** |

**Out of time, the heuristic still wins.** Shuffled CV says the learned model is ahead by
0.027 Gini; an out-of-time split says it is behind by 0.050. Shuffled folds let a model see
loans from the same months it is scoring, and the learned model uses that. The heuristic
cannot, because its constants are frozen.

Two things follow:

- **Out-of-time is the split to report from here on.** Shuffled CV flatters any model fitted on
  a book with vintage structure, and this book has it.
- **Stage 5 does not yet clear the bar.** The transforms are correct and the leakage discipline
  holds, but better inputs alone did not beat eight hand-written rules where it counts. The gap
  is not in the features; it is that nothing has been done about vintage yet — the loan book
  spans 15 months and recent vintages are censored, which `mes_prestamo` marks and no model
  currently accounts for.

Gradient boosting doing worse than logistic regression on 10 763 loans with 511 defaults is the
ordinary result at this sample size, not a bug.
