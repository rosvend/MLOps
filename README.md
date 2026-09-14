# MLOps

End-to-end MLOps pipeline, from training to monitoring, for loan applications in a Colombian bank.

## Tech stack

Feast (feature store) · MLflow (experiment tracking & monitoring) · W&B (experiment tracking) ·
DVC (data versioning) · Pydantic + Hydra · Great Expectations · Evidently AI (monitoring & data
drift) · FastAPI · Docker

## Layout

```text
config/config.yaml      where the data lives and how it is read
data/raw/               BD_creditos.csv, 10 763 loans (DVC takes over later)
notebooks/eda.ipynb     the exploratory analysis and its findings
src/data/               DataSource port, CSV adapter, pandera contract
src/features/           cleaning and derived features
src/models/             heuristic points scorecard and its metrics
src/pipelines/          prepare(): read -> clean -> derive -> validate
                        score():   prepare -> score -> evaluate
tests/                  pytest, on a small hand-written fixture
docs/heuristic-model.md every rule, the EDA rate behind it, and the limitations
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
set the bar a trained model has to clear: **Gini 0.352** against the bureau score's 0.248, with the
riskiest decile defaulting at 11.4 % and the safest at 1.7 %. Full rule table, exclusions and
caveats in [`docs/heuristic-model.md`](docs/heuristic-model.md).

## Swapping the data source

`src/pipelines/prepare.py` is the single seam the rest of the pipeline attaches to. It takes any
object with a `read() -> DataFrame` method, so moving from the raw CSV to a cloud database means
adding one adapter next to `src/data/csv_source.py`, registering it in `src/data/factory.py`, and
pointing `config/config.yaml` at it. No other module changes.

```yaml
data_source:
  type: csv
  path: data/raw/BD_creditos.csv
  separator: ";"
```
