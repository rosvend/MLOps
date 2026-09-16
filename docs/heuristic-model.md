# Heuristic model — points scorecard

A rule-based baseline built straight from `notebooks/eda.ipynb`. It is not trained: every band
and every point is read off the default rates the EDA measured. Its job is to give the pipeline
something end-to-end to run against, and to set the bar any learned model has to clear.

## How it scores

One pure function per EDA finding, each looking at a single application and returning points.
The score is their sum; higher means riskier. A loan is flagged when the score reaches the
threshold in `config/model/heuristic.yaml`.

```python
score  = sum(rule(record) for rule in RULES)
flagged = score >= threshold
```

Points come from one formula, not from judgement:

```text
points = round((band default rate / portfolio default rate - 1) * 4)
```

So a band that doubles the portfolio's 4.75 % costs +4, and one that halves it pays −2. The base
rate and the ×4 multiplier are recorded in `config/model/heuristic.yaml` alongside the weights
they generated, so the scorecard can be regenerated from the repo rather than from this page.

Band cut-points live in the same file and are frozen. They are never recomputed per batch, so one
application scores the same alone as it does inside a portfolio.

An application with nothing filled in scores **5** — above the threshold of 4 — because absence
itself charges points in three rules. That is deliberate: maximum uncertainty goes to manual
review.

A missing *column* is a different thing from a missing value, and `score_frame` raises on it.
The single-record `score()` cannot make that distinction — a dict is not a schema — so a
serving endpoint must validate its payload against `CreditoFeaturesSchema` before scoring, or
a typo'd field name will read as "unknown" and quietly add points.

## The rules

Rates measured on all 10 763 loans; portfolio base rate 4.75 %.

### `bureau_score_band` — the strongest single signal

| `puntaje_datacredito` | n | default | lift | points |
| --- | ---: | ---: | ---: | ---: |
| < 770 | 3 585 | 7.17 % | 1.51 | **+2** |
| 770 – 812 | 3 528 | 4.11 % | 0.87 | **−1** |
| ≥ 813 | 3 491 | 2.81 % | 0.59 | **−2** |
| missing | 159 | 6.92 % | 1.46 | **+2** |

### `inquiry_band` — what the score misses

Recent bureau inquiries stay predictive inside every score band, so the two are not the same
information.

| `huella_consulta` | n | default | lift | points |
| --- | ---: | ---: | ---: | ---: |
| 0 – 3 | 5 303 | 3.41 % | 0.72 | **−1** |
| 4 – 6 | 3 447 | 5.22 % | 1.10 | **0** |
| 7 or more | 2 013 | 7.45 % | 1.57 | **+2** |

### `age_band` — the only applicant-form variable that competes with the bureau

| `rango_edad` | n | default | lift | points |
| --- | ---: | ---: | ---: | ---: |
| 18-25 | 568 | 8.80 % | 1.85 | **+3** |
| 26-35 | 2 723 | 6.21 % | 1.31 | **+1** |
| 36-45 | 3 176 | 4.03 % | 0.85 | **−1** |
| 46-55 | 2 216 | 3.43 % | 0.72 | **−1** |
| 56-65 | 1 582 | 4.17 % | 0.88 | **0** |
| 66+ | 348 | 3.16 % | 0.67 | **−1** |
| missing | 150 | 7.33 % | 1.54 | **+2** |

### `young_independent` — an interaction, not an effect

Averaged over all ages, employment type looks irrelevant (Cramér V 0.027). Split by age it is
real under 36 (p = 0.0002) and gone afterwards (p = 0.085).

| under 36 | n | default |
| --- | ---: | ---: |
| Independiente | 906 | 9.38 % |
| Empleado | 2 385 | 5.62 % |

Because `age_band` already charges for being young, the points here measure only the extra:
9.38 / 5.62 = 1.67 → **+3**, and **0** for everyone else.

### `income_gap_quartile` — the gap matters, the income does not

`salario_cliente` alone has no association with default (AUC 0.505). The ratio of declared income
to the bureau's estimate does, and monotonically.

| `ratio_ingreso_declarado_bureau` | n | default | lift | points |
| --- | ---: | ---: | ---: | ---: |
| < 1.081 | 1 952 | 3.02 % | 0.64 | **−1** |
| 1.081 – 1.807 | 1 952 | 3.84 % | 0.81 | **−1** |
| 1.807 – 3.594 | 1 951 | 4.87 % | 1.03 | **0** |
| ≥ 3.594 | 1 952 | 5.79 % | 1.22 | **+1** |
| missing | 2 956 | — | — | **0** |

### `decreasing_income_trend` — stacks with over-declaration

| `tendencia_ingresos` | n | default | lift | points |
| --- | ---: | ---: | ---: | ---: |
| Decreciente | 1 291 | 6.27 % | 1.32 | **+1** |
| Estable | 1 188 | 4.63 % | 0.98 | **0** |
| Creciente | 5 294 | 3.91 % | 0.82 | **−1** |
| missing | 2 990 | — | — | **0** |

### `high_amount_long_term` — the one lever the bank sets itself

| `capital_prestado` ≥ 3 084 840 and `plazo_meses` ≥ 13 | n | default | lift | points |
| --- | ---: | ---: | ---: | ---: |
| yes | 626 | 11.02 % | 2.32 | **+5** |
| no | 10 137 | 4.36 % | 0.92 | **0** |

### `missing_bureau_income` — absence is information

The three missing counts above are not the same set. The income-gap ratio is also null when
`salario_cliente` is itself a sentinel (2 956), and `tendencia_ingresos` carries its own
missingness (2 990), while the bureau income block is missing for 2 937. So roughly 19 records
with no income-gap ratio and ~53 with no income trend are *not* charged by this rule.

| bureau income block | n | default | lift | points |
| --- | ---: | ---: | ---: | ---: |
| missing | 2 937 | 5.72 % | 1.20 | **+1** |
| present | 7 826 | 4.38 % | 0.92 | **0** |

## What was deliberately left out

| Variable | Why |
| --- | --- |
| `puntaje` | Leakage. 87 % of loans share the maximum value and none of them defaulted, so it is computed after the outcome is known. Using it would inflate every metric into fiction. |
| `tiene_mora_bureau` | 36.4 % default against 4.6 %, but on 55 loans, and it is unconfirmed whether `saldo_mora` is observed at origination or afterwards. Must be settled with the business before it can be used. |
| `pti`, `dti`, `cuota_supera_salario`, `salario_cliente` | No measured signal (AUC 0.505–0.511, Cramér V 0.000). Declared repayment capacity does not separate good from bad in this portfolio. |
| `mes_prestamo` | A vintage-censoring control, not a business predictor. Including it would let the model "predict" that recent loans default less simply because they have had less time to. |
| `saldo_total` / `saldo_principal` | ρ = 0.95 with each other, and the EDA flags their scale as implausible for Colombian pesos — possibly reported in thousands. Needs confirmation from the data provider. |

## Results

Scored over all 10 763 loans, at the configured threshold of 4:

| Metric | Value | EDA benchmark |
| --- | ---: | --- |
| AUC | 0.676 | — |
| **Gini** | **0.352** | 0.248 — bureau score alone |
| Top band default rate | 12.33 % | 11.1 % — worst manual segment |
| Bottom band default rate | 1.75 % | 1.4 % — best manual segment |
| Band lift | 7.06× | 7.7× — best/worst manual segment |
| Flagged share | 15.7 % | — |
| Precision | 10.28 % (2.17× base) | — |
| Recall | 34.1 % | — |

Gini clears the bureau score comfortably (+42 %). The 7.06× spread sits just under the EDA's
7.7×, but the two are not the same measurement: the 7.7× compares the best and worst of 34
hand-picked segments with n ≥ 100, while 7.06× is a split over the whole book.

The split yields nine bands, not ten. A 21-point integer scale over 10 763 loans means decile
edges land inside large tied groups, and loans with the same score share a band rather than
being separated by their position in the file. Splitting ties by position made the reported
lift move between 6.83× and 7.24× on the same data simply by re-sorting the rows.

Every rule earns its place — removing any one of them costs Gini:

| Rule removed | Gini | Δ |
| --- | ---: | ---: |
| `high_amount_long_term` | 0.3262 | −0.0262 |
| `bureau_score_band` | 0.3276 | −0.0248 |
| `inquiry_band` | 0.3301 | −0.0223 |
| `income_gap_quartile` | 0.3377 | −0.0147 |
| `decreasing_income_trend` | 0.3416 | −0.0108 |
| `age_band` | 0.3461 | −0.0063 |
| `missing_bureau_income` | 0.3463 | −0.0061 |
| `young_independent` | 0.3466 | −0.0058 |

## As a scikit-learn estimator

`src/models/estimator.py` splits the job the way scikit-learn splits it: one class ranks,
another turns a rank into a probability.

```python
credit_pipeline().fit(X_train, y_train)   # prepare -> score -> calibrate
pipe.predict_proba(X)[:, 1]               # probability of default
pipe.predict(X)                           # flagged at the configured threshold
cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
```

`y` is the default indicator: `True` means the loan defaulted.

**`HeuristicModel` is a ranker, not a probability model.** `decision_function` returns the
raw integer score. It deliberately has no `predict_proba`: the points are not a probability, and
rescaling them into `[0, 1]` would only make them look like one.

**Calibration never sees the rows it scores.** `calibrated_model()` wraps the ranker in
`CalibratedClassifierCV(method="isotonic", cv=5)`, which fits one calibrator per fold on the
other folds and averages them. Fitting isotonic on the same rows it then reports on was both
optimistic and unusable:

| | Isotonic fitted in-sample | Cross-fitted |
| --- | ---: | ---: |
| Train → test log-loss gap | 0.01715 | **0.00555** |
| Highest probability of default | **1.0000** | 0.5464 |

The second row is the one that mattered. A 21-point scale leaves small pure bands at the top,
and isotonic handed them a probability of exactly 1.0 — a claim that this applicant will
certainly default, which no credit model can make and which no amount of data supports here.

`score()` is overridden to AUC. `ClassifierMixin` gives accuracy, and at a 4.75 % base rate a
model that never flags anyone scores 95.25 %, so the inherited metric would rank this scorecard
below doing nothing.

Feature metadata is established and checked by `validate_data(..., skip_check_array=True)`, so
`n_features_in_` and `feature_names_in_` are sklearn's to maintain rather than hand-written, and
predicting on a different feature space raises instead of scoring something wrong. The rules
address columns by name, so `skip_check_array` keeps the frame a frame.

### Out-of-fold results

5-fold stratified cross-validation over all 10 763 loans:

| Metric | Out-of-fold | In-sample |
| --- | ---: | ---: |
| AUC | 0.676 | 0.676 |
| **Gini** | **0.352** | 0.352 |

They match, which is what frozen rules should do — nothing is fitted to the scored rows, so the
ranking carries no optimism. **This does not retire the in-sample caveat below.** The band
cut-points and the weights were chosen by reading this same dataset before the code existed;
cross-validation cannot detect that, because the choice happened outside the estimator. Only an
out-of-time split, or a fresh vintage, can.

### On `check_estimator`

It reports success for both classes, and that result is empty: the DataFrame-only input tag makes
it skip its entire suite, so exactly one check runs. Rather than bank a hollow pass, the
applicable checks are enumerated and run directly — 24 against the scorecard and 16 against the
preparer, including `check_dataframe_column_names_consistency`,
`check_n_features_in_after_fitting`, `check_estimators_unfitted` and
`check_no_attributes_set_in_init`. A test guards the count so the suite cannot silently collapse
back to a skip.

DataFrame-only is a safety property, not a limitation: the rules pay points for a *missing*
value, so accepting an unnamed array would make every rule take its "absent" branch and score a
blank application as high-risk rather than raising.

## Limitations

- **In-sample.** Bands and points were measured on the same 10 763 loans they are scored against,
  so these numbers are optimistic. A held-out split belongs with the first trained model, not here.
- **Vintage censoring.** The observed 4.75 % is a floor, not the portfolio's final default rate —
  recent loans have had less time to deteriorate. Every rate above inherits that bias.
- **Additive double counting.** Correlated signals each charge their full points. The EDA showed
  no single variable separates the classes, so combining weak signals is the right move, but an
  additive sum overstates it. A learned model handles this properly.
- **The score is not monotone everywhere.** Scores of −1 (4.75 %) and 0 (2.71 %) sit out of order.
  It is a local wobble in a 21-point scale; fixing it would mean fitting to this dataset, which is
  exactly what this baseline is not for.
- **Not calibrated.** The score ranks risk; it is not a probability of default.
