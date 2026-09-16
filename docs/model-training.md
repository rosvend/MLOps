# Model training and champion selection (stage 7)

Three candidate families are tuned with Optuna, every run is tracked in MLflow, and a champion
is chosen programmatically against the held-out window.

```
src/models/training_spec.py   shape of config/training/default.yaml
src/models/dataset.py         Feast retrieval, label join, out-of-time split
src/models/metrics.py         summarize_classification
src/models/pipelines.py       build_model
src/models/champion.py        comparison table and selection
src/pipelines/train.py        make train
```

## Protocol

**Tuning** uses stratified 5-fold cross-validation **inside the training window only**, scored on
average precision. **Selection** uses a single held-out window of the newest vintages, read once,
after all tuning is finished.

```
2024-11 ................ 2025-05 | 2025-06 ............ 2026-04
[ 7 713 loans, 5.32 % default  ] [ 3 050 loans, 3.31 % ]
   tune: StratifiedKFold(5)        champion: read once
```

The held-out window has a lower default rate than the training window. Part of that is genuine
drift and part is the newest vintages not having matured, so every out-of-time number is
depressed — equally, for all candidates.

**Every transformer lives inside the sklearn `Pipeline`**, so each fold refits its own winsor
caps, medians and encodings. Nothing is fitted before the split. Two tests pin this: fold
estimators must carry *different* fitted caps, and no fold's cap may equal the whole-book value.

## Results

25 Optuna trials per family, 5 folds each. Thresholded metrics are taken at the operating point
the business actually uses — the 15.7 % of applications sent to manual review — so they compare
like for like between a probability model and a ranking one; 0.5 would mean different things to
each.

| model | CV mean ± std | **PR-AUC** | ROC-AUC | **Gini** | F1 | precision | recall | Brier | fit s | ms / 1k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **logistic** | 0.1606 ± 0.0250 | **0.1293** | 0.6636 | **0.3272** | 0.1172 | 0.0710 | 0.3366 | 0.0319 | 0.17 | 12.0 |
| xgboost | **0.1865** ± 0.0137 | 0.0908 | 0.6525 | 0.3050 | 0.1172 | 0.0710 | 0.3366 | 0.0416 | 0.46 | 13.2 |
| lightgbm | 0.1810 ± 0.0113 | 0.1020 | 0.6398 | 0.2796 | 0.1138 | 0.0689 | 0.3267 | 0.1307 | 0.38 | 17.0 |
| heuristic (incumbent) | — | 0.0585 | 0.6619 | 0.3237 | 0.1022 | 0.0610 | 0.3168 | — | 0.005 | 23.0 |

Brier is withheld for the heuristic: it emits integer points, not probabilities, and scoring
those against Brier would compare points to probabilities and report a meaningless number.

![Out-of-time PR-AUC and Gini by model](plots/model_comparison_performance.png)

## The result that justifies the protocol

**XGBoost has the best cross-validation score and the worst out-of-time PR-AUC of the three
learned models.** CV 0.1865 against logistic's 0.1606 — and then 0.0908 out of time against
logistic's 0.1293.

Had the champion been selected on cross-validation, as is conventional, XGBoost would have won
and it would have been the wrong model. The boosters have the capacity to fit vintage-specific
structure in the tuning folds, and that structure does not survive into the next six months.
Consistency would not have caught it either: XGBoost and LightGBM have *lower* fold variance than
logistic (0.0137 and 0.0113 against 0.0250). Only the held-out window separates them.

![Cross-validation score vs out-of-time score, per tuned model](plots/model_comparison_cv_vs_oot.png)

Regenerate both from the current `reports/champion.json` with `uv run python scripts/plot_model_comparison.py`.

## Champion: logistic regression

`C = 0.0869`, `class_weight = None`, with standardisation, on the 46 engineered features.

- **Performance.** Best on the leading metric by a wide margin: PR-AUC 0.1293 against 0.1020 for
  the next best learned model and 0.0585 for the incumbent — **2.2× the incumbent** at
  concentrating defaults in the flagged queue. It also takes the Gini, 0.3272 against 0.3237,
  though only narrowly, and this is the first model in the project to beat the heuristic on both.
- **Consistency.** Its fold variance is the highest of the three (±0.0250), which is the one
  honest mark against it. It is outweighed by the out-of-time result: the models with tighter
  folds are the ones that generalised worst, so low CV variance here is a sign of consistently
  fitting the same vintage structure, not of robustness.
- **Scalability.** Fastest of the learned models to fit (0.17 s against 0.46 s and 0.38 s) and to
  score (12.0 ms per 1 000 rows). It is also linear and inspectable, which matters when a declined
  applicant is entitled to a reason.

The incumbent heuristic is not retired by this. It stays the reference: it still gets within
0.004 Gini of a tuned model while being a set of frozen rules with no fitted state, and it remains
the fallback if the learned model drifts.

## Honest limitations

- **One held-out window.** The champion is chosen on a single 3 050-loan slice. Rolling-origin
  evaluation over several cut-points would be a stronger basis and is the obvious next refinement.
- **Absolute performance is modest.** Gini 0.33 is a usable but not strong scorecard. The gap
  between CV and out-of-time says that some of what these models learn is vintage-specific.
- **Not calibrated.** The champion's `predict_proba` is whatever logistic regression produces; it
  has not been calibrated against observed default rates. Brier 0.0319 is reported, not defended.
- **The maturity problem is unresolved.** Recent vintages are censored, which depresses the
  held-out metrics. A maturity-adjusted label would be a better target than the raw one.

## Running it

```bash
make train                                                        # 25 trials x 5 folds x 3 families, ~2.5 min
uv run python -m src.pipelines.train training.n_trials=50
uv run python -m src.pipelines.train training.selection.primary_metric=gini
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLflow holds one parent run per family and one child run per trial — 80 runs for the default
configuration — with parameters, CV mean and standard deviation, all out-of-time metrics, feature
importances and the fitted model. `reports/champion.json` carries the decision.
