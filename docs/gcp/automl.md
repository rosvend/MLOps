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
