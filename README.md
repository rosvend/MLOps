# MLOps

End-to-end MLOps pipeline, from training to monitoring, for loan applications in a Colombian bank.

## Tech stack

Feast (feature store) · MLflow (experiment tracking & monitoring) · W&B (experiment tracking) ·
DVC (data versioning) · Pydantic + Hydra · Great Expectations · Evidently AI (monitoring & data
drift) · FastAPI · Docker

## Layout

```text
config/                 Hydra config groups: data_source/ and model/
data/raw/               BD_creditos.csv, 10 763 loans (DVC takes over later)
notebooks/eda.ipynb     the exploratory analysis and its findings
src/data/               DataSource port, CSV adapter, pandera contract
src/features/           cleaning, derived features, and the leakage contract
src/models/             scorecard rules, its sklearn estimator, and the metrics
src/pipelines/          prepare_features() / prepare_labelled(): read -> clean -> derive -> validate
                        score():   prepare -> score -> evaluate
tests/                  pytest, on a small hand-written fixture
docs/heuristic-model.md every rule, the EDA rate behind it, and the limitations
feature_repo/           Feast offline feature store: entity, four feature views
```

## Getting started

```bash
make install
make test
make eda
make score
```

## Baseline model

`make score` runs the heuristic scorecard over the portfolio. It is a rule-based baseline read
straight off the EDA — one pure function per finding, points summed, no fitting — and it exists to
set the bar a trained model has to clear: **Gini 0.352 in-sample** against the bureau score's
0.248, with the riskiest band defaulting at 12.3 % and the safest at 1.8 %. The bands and points
were measured on the same loans they are scored against, so these numbers are optimistic — a
held-out split belongs with the first trained model. Full rule table, exclusions and caveats in
[`docs/heuristic-model.md`](docs/heuristic-model.md).

The scorecard is a scikit-learn classifier, so it drops into a `Pipeline` and
`cross_val_score` alongside any trained model that follows. `HeuristicModel` ranks;
`credit_pipeline()` adds an isotonic calibration fitted out-of-fold, so a probability of
default never comes from a calibration that saw the row it is scoring. Out-of-fold Gini is
0.352, the same as in-sample — frozen rules do not overfit, though the bands were still
chosen on this dataset.

Thresholds and weights are config, not code: `python -m src.pipelines.score model.threshold=5`,
or `--multirun model.threshold=3,4,5,6` to sweep. Every run records the exact scorecard that
produced it under `outputs/`.

`make feast-apply` registers the feature table with Feast for offline retrieval, and
`make feast-verify` proves the point-in-time join holds — zero features returned before a
loan was originated. Details in [`docs/feature-store.md`](docs/feature-store.md).

`make train` tunes logistic regression, XGBoost and LightGBM with Optuna, tracks every run
in MLflow and picks a champion on a held-out window of the newest vintages. Details and the
results table in [`docs/model-training.md`](docs/model-training.md).

## Swapping the data source

`src/pipelines/prepare.py` is the single seam the rest of the pipeline attaches to. It takes any
object with a `read() -> DataFrame` method, so moving from the raw CSV to a cloud database means
adding one adapter next to `src/data/csv_source.py`, registering it in `src/data/factory.py`, and
pointing `config/data_source/` at it. No other module changes.

```yaml
# config/data_source/csv.yaml
type: csv
path: data/raw/BD_creditos.csv
separator: ";"
```

Select it with `data_source=csv`, or point a run at another adapter without editing a file.
