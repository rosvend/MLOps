# Why is the loan-default model weak?

Is the champion's out-of-time ROC-AUC of 0.664 and PR-AUC of 0.129 a ceiling set by the data (about 410 defaults), or something feature engineering and modelling missed?

| Finding | Evidence |
| --- | --- |
| Mostly a data ceiling | Best single feature AUC about 0.63; linear, L1, boosted trees and forest all land at ROC 0.70 ± 0.01 in time-ordered folds. |
| Scores are blurry | Any one number has about ±0.03 ROC and ±0.04 PR uncertainty. |
| Labels may be immature | Default rate falls with origination month for terms <= 12 months. |
| One lead | A non-linear `plazo_meses` gains +0.009 PR-AUC, 1 pass out of 29 comparisons. |

**Data used.** The newest 25 % of months (the TEST window, June 2025 onward) was dropped in `experiments/signal/data.py` before any look at the data. Everything here uses the 7,713 loans (410 defaults, 5.3 %) of 2024-11 to 2025-05. Nothing was computed on TEST, and nothing in the project was changed. Code is in `experiments/signal/`, raw tables in `docs/investigation/results/*.csv`.

**Method.** Two validation schemes with the same folds for every model: time-ordered (train on earlier months, validate on one later month, 4 folds) and stratified 5-fold CV repeated 3 times. All transformers are fitted inside each training fold. A gain counts as confirmed only if the paired difference exceeds 2 standard errors, is positive in every time fold, holds in stratified CV, and its paired bootstrap interval excludes zero. 29 candidates were compared on 2 metrics and 2 schemes (116 paired tests), so one or two false positives are expected by chance.

## 1. Noise ceiling

| Champion, development months | ROC-AUC | PR-AUC |
| --- | --- | --- |
| Time-ordered, pooled (237 defaults) [95 % CI] | 0.702 [0.668, 0.739] | 0.151 [0.115, 0.198] |
| Time-ordered, per-fold mean, sd | 0.703, 0.050 | 0.160, 0.074 |
| Stratified CV, pooled (410 defaults) [95 % CI] | 0.682 [0.654, 0.709] | 0.152 [0.124, 0.190] |
| Random ranking | 0.5 | 0.053 |

![ROC by time fold](plots/time_folds_roc.png)

A one-month fold holds only 47 to 69 defaults. Two near-identical models differ by ±0.003 ROC, genuinely different ones by ±0.01 to ±0.03, so **a gain below about +0.01 cannot be told from noise; +0.02 or more can.** In the learning curve, 10 % of the data gives ROC 0.62, 50 % gives 0.67 and 100 % gives 0.68. It rises about +0.01 per doubling, so more data helps a little.

![Learning curve](plots/learning_curve.png)

## 2. Label maturity

Term (`plazo_meses`) is the clearest risk driver and is not monotone: 11 to 12 months is the safest.

| Term (months) | loans | default % |
| --- | ---: | ---: |
| <= 6 | 2452 | 6.0 |
| 7-10 | 1945 | 5.3 |
| 11-12 | 2597 | 3.2 |
| 13-24 | 508 | 7.5 |
| > 24 | 211 | 18.5 |

![Default rate by month and term](plots/maturity_month_term.png)

For terms <= 12 months (6,994 loans) the default rate falls with origination month: -0.078 per month in log-odds, 95 % CI [-0.146, -0.016]. For all terms together it is flat (-0.021, CI [-0.077, 0.036]) because long-term, riskier loans became a larger share of recent months. The drop is concentrated in 2025-05, where even loans of 6 months or less show 3.0 %, so term ending alone does not explain it.

If defaults accrue with loan age, TEST labels (5.3 % to 3.3 %) are incomplete, every out-of-time score is depressed, and ranking can be distorted too. This is a hypothesis: the data has no default date and no extraction date, so it cannot be proved here. A fix needs those dates from the bank, a fixed observation window, and training and evaluation on mature vintages only.

## 3. Signal audit and additions

| Feature | Univariate AUC [95 % CI] | Same sign in months (of 7) |
| --- | --- | --- |
| `puntaje_datacredito` (lower is riskier) | 0.366 [0.336, 0.397] | 7 |
| `huella_consulta` | 0.614 [0.587, 0.640] | 7 |
| `edad_cliente` (younger is riskier) | 0.424 [0.390, 0.455] | 6 |
| `ratio_ingreso_declarado_bureau` | 0.552 [0.525, 0.578] | 7 |
| `capital_prestado` | 0.550 [0.516, 0.579] | 5 |

![Univariate AUC](plots/signal_univariate.png)

Signal is spread thin across many weak features. Winsorising, log1p and imputation leave every univariate AUC unchanged (differences below 0.0005) because they are monotone. Only two missingness flags have a lift whose interval excludes 1 (`falta_promedio_ingresos_datacredito` 1.27x, `falta_tendencia_ingresos` 1.24x); the "1.18x to 1.54x" quoted in [Feature engineering](../feature-engineering.md) is not statistically supported for the six rare flags.

Differences are against the champion (time-ordered ROC 0.7025, PR 0.1596).

| Variant | time ROC diff | time PR diff (folds won) | Verdict |
| --- | ---: | ---: | --- |
| Raw columns only | -0.003 | +0.009 (4/4) | no real difference |
| No `falta_*` flags | -0.003 | +0.006 (3/4) | no real difference |
| No ratios / age bands | -0.001 | +0.004 (3/4) | no real difference |
| No `plazo_meses` | -0.014 | -0.011 (0/4) | term matters |
| **WOE bins of `plazo_meses`** | +0.009 | +0.009 (4/4) | **PR yes, ROC no** |
| WOE on 12 columns, kept | +0.006 | +0.012 (3/4) | no |
| WOE only, replacing originals | -0.016 | -0.026 | worse |
| 4 interactions | -0.001 | +0.002 | no |
| Target encoding of 2 categoricals | -0.003 | -0.001 | no |

The ratios, bands and flags neither add nor remove measurable signal. The only addition passing the rule is a non-linear term (+0.009 PR in time folds, +0.010 stratified; ROC gain inside bootstrap noise). One pass in 29 is what chance can produce, so treat it as a well-motivated lead, not proof.

## 4. Imbalance, calibration and model families

| Change | Result |
| --- | --- |
| Class weights, over/undersampling, SMOTE | No ranking gain; undersampling and SMOTE 100 % hurt (ROC -0.015 to -0.027 for undersampling). |
| Platt or isotonic calibration | No measurable change. The champion is already calibrated (ECE 0.0088, mean predicted 5.9 % vs observed 5.4 %). Brier skill over the base rate is only 3 %. |
| Re-tuning C from 0.03 to 1 | Within ±0.003 ROC. |
| Shallow GBMs, XGBoost depth 2, random forest | About +0.02 in shuffled CV but not in time-ordered folds, repeating the project finding that CV flatters trees. |

![Calibration](plots/calibration.png)

## Conclusion and recommendation

1. **Label maturity first.** Get default and extraction dates, define a fixed window, and evaluate on mature vintages. It changes what "better" means.
2. **Try one change**: add a WOE or banded `plazo_meses` to the champion pipeline, everything else frozen. Compare on rolling-origin folds over all vintages and on the TEST window once, with paired differences and a bootstrap interval, as one pre-registered hypothesis.
3. **Report intervals** and time-ordered folds in model selection, and stop reading 0.01 differences.

Not worth doing: resampling, class weights, calibration, more interactions, target encoding, L1 selection, re-tuning C, or swapping to GBMs. Do not promote the term change on this investigation alone.

## Caveats and unverified points

- The extraction date and the exact definition of `Pago_atiempo` are not in the data; the censoring explanation is a hypothesis.
- Only 4 time folds, so the 2-standard-error rule is rough. Stratified repeats share rows, so their standard errors are optimistic.
- The C and WOE rows were scored on the folds used to pick them, so they are slightly optimistic.
- Only four monotone-constraint variants were tried, with directions set from domain sense.
