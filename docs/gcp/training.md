# Training on Vertex AI

Two ways of training on Google Cloud are compared with the local Python run, all on the **same
held-out loans**.

## The split

AutoML splits randomly unless told otherwise, which would let it train on the months it is
scored on. A `split` column pins every loan to a set by origination month.

```mermaid
flowchart LR
    subgraph months["Loans by origination month, oldest to newest"]
        direction LR
        tr["TRAIN<br/>5,528 loans<br/>5.3 % default"] --> va["VALIDATE<br/>2,185 loans<br/>5.3 % default"] --> te["TEST<br/>3,050 loans<br/>3.3 % default"]
    end
```

- **TRAIN + VALIDATE** (7,713 loans) are the same loans the local run trains on.
- **VALIDATE** is the newest slice of the training window. AutoML needs a validation set, and
  this keeps it from tuning on TEST.
- **TEST** is the newest 25 % and is read once, for the final comparison.

The test default rate (3.3 %) is lower than in training (5.3 %). Recent loans may not have had
time to default yet, or the book may have changed. Either way it affects every model equally.

The file is built by `make export-gcp-dataset` and has the model features, a `defaulted` label
(1 = default) and the `split` column. Identifiers, the origination date and the leaky `puntaje`
column are excluded.

## Custom pipeline on Vertex Pipelines

The same three models as the local run train in parallel, then a final step compares them.
Hyperparameters were copied from the local tuning runs, so nothing is re-tuned in the cloud.
See [Custom pipeline run](custom-pipeline.md) for the steps and the commands.

| Model | PR-AUC | Gini | ROC-AUC | F1 |
| --- | --- | --- | --- | --- |
| **Logistic regression** | **0.129** | **0.327** | 0.664 | 0.117 |
| LightGBM | 0.102 | 0.280 | 0.640 | 0.114 |
| XGBoost | 0.091 | 0.305 | 0.653 | 0.117 |
| Heuristic baseline (local) | 0.059 | | | |

Logistic regression on Vertex reproduces the local champion (PR-AUC 0.129, Gini 0.327), so the
cloud pipeline and the local one agree. The gradient-boosted models tune better in
cross-validation but generalise worse to the newest months, the same finding as locally.

## AutoML Tabular

AutoML searches model types and settings itself, within a training budget of **1 node-hour**
(about $21 at list price, the minimum it allows). It uses the same split column. Details, commands
and resource ids are on the [AutoML run](automl.md) page.

AutoML's own evaluation reports PR-AUC 0.970 and ROC-AUC 0.979. **Those are not comparable.** They
appear to pool both classes, so the 96 % of loans that do not default dominate, and they cover
only 2,166 of the 3,050 TEST loans: AutoML refused to score any row with an empty numeric cell.
The comparison below recomputes everything the same way for every model.

| Model | Loans scored | PR-AUC | ROC-AUC | Gini | F1 | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic champion (local and Vertex) | 3,050 | 0.129 | 0.664 | 0.327 | 0.117 | 0.337 |
| AutoML v2, imputed data (fully scored) | 3,050 | 0.215 | 0.790 | 0.581 | 0.169 | 0.485 |
| AutoML v1, unscored loans given score 0 | 3,050 | 0.159 | 0.595 | 0.190 | 0.138 | 0.396 |
| AutoML, scored loans only (not comparable) | 2,166 | 0.235 | 0.815 | 0.629 | 0.164 | 0.524 |
| AutoML, Vertex's own figures (not comparable) | 2,166 | 0.970 | 0.979 | | | |

Read the rows in pairs. AutoML v1 (third row) looks mixed against the champion: a higher PR-AUC (0.159 against
0.129) but a worse Gini (0.190 against 0.327), because it could not score 884 loans and they were given a
score of 0, a harsh choice. AutoML v2 (second row) removes that problem and is the one to compare.

The empty-cell problem was the practical lesson: the local pipeline imputes missing values and
adds "was missing" flags, while AutoML v1 could not score such rows. v2 fixed it by imputing
before upload (medians fitted on TRAIN only) and paying for a second training run. All 3,050 loans
are now scored and AutoML v2 beats the logistic champion on every ranking metric in the table
(PR-AUC 0.215 against 0.129, Gini 0.581 against 0.327). The paired bootstrap on the
[results page](results.md) says the PR-AUC gap is distinguishable from noise, but it is one window with
101 defaults that has been used before, so read it as strong evidence and not as proof. Vertex's own figures for v2 (PR-AUC 0.973)
are still pooled over both classes and not comparable. Details on the [AutoML run](automl.md) page.

## Local versus cloud

| | Local Python | Vertex custom pipeline | AutoML |
| --- | --- | --- | --- |
| Control over the model | Full | Full | Limited |
| Setup effort | Low | High (image, pipeline, IAM) | Medium |
| Wall-clock time | About 2.5 minutes for tuning | Dominated by VM start-up per step | Dominated by the training budget |
| Cost | Free | A few cents | About $22 (estimate; check Billing) |
| Reproducibility | Depends on the machine | Container plus config | Managed |
