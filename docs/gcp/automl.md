# Vertex AI AutoML Tabular

AutoML trained a classifier on the same loan file as the custom pipeline, with no model code on our
side. This page says what it did, what we ran, and why its own headline numbers cannot be compared
with the local champion.

```
config/gcp_automl/default.yaml   project, budget, column types, batch settings
src/automl/spec.py               typed loader (rejects any budget above 1000)
src/automl/train.py              create the dataset and submit the training job
src/automl/status.py             print the state of a training pipeline
src/automl/evaluate.py           build the TEST input, parse scores, compute project metrics
src/automl/report.py             batch prediction on TEST + Vertex metrics -> reports/gcp/automl_metrics.json
```

## What AutoML does

You give it a table and a target column. It then, on Google's machines, tries many feature
transformations, model families (mostly gradient-boosted trees and neural nets) and hyperparameters,
and ends with an ensemble of the best ones. You never see the code or the trained pipeline, only a
model you can predict with.

- **Node hour.** The unit AutoML bills and budgets in: one machine working for one hour. The budget
  is `budget_milli_node_hours = 1000` (1 node hour, the minimum). It is a cap on the search, not on
  the wall clock: our job ran 03:55 to 06:10 UTC (about 2h15) because setup, model building and
  evaluation are not counted against the search budget.
- **Predefined split.** By default AutoML splits rows randomly, so a model trains on loans from the
  same months it is evaluated on. Our file has a `split` column (TRAIN / VALIDATE / TEST by
  origination month, TEST newest), passed as `predefined_split_column_name="split"`. AutoML tunes on
  VALIDATE and reports on TEST; TEST never influences training.
- **Column types.** Booleans and `tipo_credito` are declared `categorical` in the config, the rest
  `numeric` or `categorical` as listed. The label and `split` are not features.
- **Objective.** `maximize-au-prc`, because defaults are about 5 % of loans.

## Commands run

```bash
export GOOGLE_CLOUD_QUOTA_PROJECT=mlops-roysandoval
uv run pytest tests/automl -q
uv run --group gcp python -m src.automl.train                      # dataset + AutoMLTabularTrainingJob.run(sync=False)
uv run --group gcp python -m src.automl.status 4953124313457754112
uv run --group gcp python -m src.automl.report <model> [<batch job>]   # Model.batch_predict on TEST rows only
```

Key SDK calls: `aiplatform.TabularDataset.create(gcs_source=...)`,
`AutoMLTabularTrainingJob(optimization_prediction_type="classification", optimization_objective="maximize-au-prc", column_specs=...)`,
`.run(target_column="defaulted", predefined_split_column_name="split", budget_milli_node_hours=1000, disable_early_stopping=False, sync=False)`,
`Model.batch_predict(instances_format="csv", predictions_format="jsonl", machine_type="n1-standard-2")`.
No endpoint was created.

## Resources (us-central1, project `mlops-roysandoval`)

| Resource | Id |
|---|---|
| Dataset | `9098751464817295360` |
| Training pipeline `loans-automl-v1` | `4953124313457754112` (SUCCEEDED) |
| Model | `5603841880623153152` |
| Batch prediction job | `3654697838077542400` (13:07 to 13:29 UTC, 2,166 rows scored) |
| Batch output | `gs://mlops-roysandoval-data/automl/batch/output/` |

## Result

Vertex's own evaluation (model evaluation `2081481660607332385`):

| auPRC | auROC | log loss | rows | positives |
|---|---|---|---|---|
| 0.970 | 0.979 | 0.169 | 2,166 | 63 |

These are **not** comparable with the local champion. Two reasons:

1. **Averaging.** The evaluation payload's confusion matrix is `[[2097, 6], [57, 6]]` (rows are the
   actual class 0 and 1). At threshold 0.5 its `truePositiveCount` is 2,103 = 2,097 + 6: the counts
   of both classes are pooled, i.e. micro-averaged. Pooling is dominated by the 95 % majority class,
   so precision, recall and F1 all equal about 0.971 even though only 6 of 63 defaulters are caught.
   That the headline auPRC/auROC use the same pooling is consistent with the payload but **unverified**:
   the documentation we found does not state it. Our recomputation below, on the identical 2,166
   rows, gives PR-AUC 0.235 and ROC-AUC 0.815 for the default class alone, nowhere near 0.97.
2. **Rows.** The evaluation covers 2,166 rows, not the 3,050 of TEST. The batch job explains it:
   AutoML refused to score 884 TEST rows (successfulCount 2,166) with the error "Failed numeric
   transformation" on `promedio_ingresos_datacredito`, `saldo_principal`, `salario_cliente` and
   `edad_cliente`. The rows it dropped are exactly those with an empty numeric cell
   (`promedio_ingresos_datacredito` is empty in 95 % of dropped rows and 0 % of kept ones). The
   2,166 coincidence with the evaluation count strongly suggests the evaluation dropped the same
   rows (not checked row by row). Whether training also dropped them is **unverified**; if so the
   model learned only from complete-case loans, which is unlike the real population, where the
   local pipeline imputes and adds `falta_*` flags.

Recomputed with `src/models/metrics.py` (`summarize_classification`), flagging the top 15.7 % by
score, as in `src/pipelines/train.py`:

| Model / view on TEST | rows | PR-AUC | ROC-AUC | Gini | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| AutoML, rows it scored only | 2,166 (63 defaults) | 0.235 | 0.815 | 0.629 | 0.164 | 0.097 | 0.524 |
| AutoML, whole TEST, unscored rows scored 0 | 3,050 (101 defaults) | 0.159 | 0.595 | 0.190 | 0.138 | 0.084 | 0.396 |
| Local logistic (champion) | 3,050 | 0.129 | 0.664 | 0.327 | 0.117 | 0.071 | 0.337 |
| Heuristic baseline | 3,050 | 0.059 | - | - | - | - | - |

How to read it: the only row comparable with the champion is the whole-TEST one, because it is the
same loans and the same flagging rule. There AutoML ranks slightly better at the top (PR-AUC 0.159
vs 0.129, recall 0.40 vs 0.34) but worse overall (Gini 0.19 vs 0.33), and the unscored-as-zero
rule is a harsh choice that mixes model quality with its refusal to score. The first row looks
better only because the dropped rows were the hard ones; it is a different, easier population.
Neither is "AutoML beats the champion". Caveats: only 101 defaults, so every number is noisy;
the training set differs slightly (VALIDATE is a slice of the training window, and AutoML may have
dropped incomplete rows).

A cleaner follow-up, not run: impute the missing numeric cells before sending the file to AutoML
(the same imputation the local pipeline fits on TRAIN) so all 3,050 rows are scored.

## Cost

No billing export is available here, so this is an estimate, not a measured figure.

- Training: at most 1 node hour at about $21.25 (the budget is a hard cap), so up to about $21.
- Batch prediction: 1 `n1-standard-2` replica, about 18 minutes running, under $1.
- Cloud Storage: a few MB, negligible.

Check the real number in Billing, Reports, filtered to project `mlops-roysandoval` and the
Vertex AI service.

## Version 2: imputed data

**What changed.** v1 could not score 884 TEST loans because of empty numeric cells. v2 trains on a
copy of the same file with no empty cell (`config/gcp_automl/imputed.yaml`, built by
`src/automl/prepare.py`, logic in `src/automl/impute.py`). Numeric columns get the median of the
TRAIN rows only; categorical and boolean columns get the label `missing`; the `falta_*` flags, the
label and `split` are unchanged. VALIDATE and TEST never contribute a statistic (unit-tested), so
there is no leakage. Everything else is the same: `maximize-au-prc`, predefined split, budget 1000
milli-node-hours.

```bash
uv run --group gcp python -m src.automl.prepare
uv run --group gcp python -m src.automl.train config/gcp_automl/imputed.yaml reports/gcp/automl_v2_state.json
uv run --group gcp python -m src.automl.report config/gcp_automl/imputed.yaml <model> reports/gcp/automl_v2_metrics.json
```

| Resource | Id |
|---|---|
| Dataset | `4558560090474414080` |
| Training pipeline `loans-automl-v2-imputed` | `3237129710127284224` (14:35 to 16:48 UTC) |
| Model | `7535886120765095936` |
| Batch prediction job | `4669028101026480128` (20:21 to 20:35 UTC, successfulCount 3,050) |
| Batch output | `gs://mlops-roysandoval-data/automl/batch/output/prediction-loans-automl-v2-imputed-*` |

**Result.** All 3,050 TEST loans (101 defaults) were scored, flagging the top 15.7 %:

| Model on TEST | PR-AUC | ROC-AUC | Gini | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| AutoML v2 (imputed) | 0.215 | 0.790 | 0.581 | 0.169 | 0.102 | 0.485 |
| AutoML v1, unscored as 0 | 0.159 | 0.595 | 0.190 | 0.138 | 0.084 | 0.396 |
| Local logistic champion | 0.129 | 0.664 | 0.327 | 0.117 | 0.071 | 0.337 |
| Heuristic baseline | 0.059 | | | | | |

Vertex's own evaluation of v2 now covers all 3,050 rows (its confusion matrix sums to 3,050), but
still reports auPRC 0.973, auROC 0.977, log loss 0.148 while catching only 7 of 101 defaulters at
0.5. It is the same pooled-over-both-classes figure as in v1 (pooling is unverified for the headline
numbers, see above) and is not comparable; our recomputation is the one to quote.

**Reading it.** v2 ranks defaulters better than the champion on every metric above, and much better
than v1, which confirms that v1's weak showing came from the unscored rows rather than the model.
Honest caveats: TEST has only 101 defaults, so a few loans move PR-AUC a lot and the gap to the
champion has not been shown to be significant here (the paired bootstrap intervals are on the results
page). The champion is a hand-tuned, interpretable logistic model; AutoML is a black-box ensemble,
and it trained on slightly different data (VALIDATE is a slice of the training window; imputed copy).
One run, one seed: we did not repeat the training.

**Cost (estimate; no billing export read).** Training at most 1 node hour, about $21.25 (hard cap);
batch prediction about 14 minutes on one `n1-standard-2`, under $1. v2 adds about $22 to the
roughly $22 of v1.

### What v2 relies on

AutoML publishes a global feature importance with each model evaluation (share of the total absolute
attribution; read from the model's evaluation in the Vertex API):

| Feature | Share |
|---|---|
| `huella_consulta` (recent credit inquiries) | 15.7 % |
| `plazo_meses` (loan term) | 12.7 % |
| `puntaje_datacredito` (bureau score) | 12.6 % |
| `tendencia_ingresos`, `rango_edad`, `cant_creditosvigentes`, `cuota_pactada` | 4 % to 5 % each |

All are inputs known at the moment of the decision. The model never sees the label, the `split` column or
an identifier (35 input columns, checked in the training pipeline's configuration). The loan term is the
non-linear effect that the [investigation](../investigation/model-signal.md) found a linear model cannot
use: default risk is lowest around 11 to 12 months and highest above 24. So the likely explanation for the
gap is real structure that AutoML's ensemble captures and our linear model does not. That is a plausible
explanation, not a proof, and it comes from one test window.
