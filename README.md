# MLOps — credit scoring

End-to-end MLOps pipeline for loan default risk: raw CSV → feature store → tuned model →
batch API → drift monitoring, for a Colombian bank's loan book.

## Tech stack

pandas + pandera (prep & validation) · scikit-learn (`FeatureEngineer`, the champion
pipeline) · Feast (offline feature store) · Optuna (hyperparameter search) · XGBoost /
LightGBM (candidates) · MLflow (experiment tracking) · FastAPI + Docker (batch serving) ·
Evidently AI (drift) · Hydra + pydantic (config, everywhere) · uv (dependency management)

## Pipeline

| Stage | What | Where |
| --- | --- | --- |
| 1–4 | Ingest, clean, derive features, validate (pandera contract) | `src/data/`, `src/features/`, `src/pipelines/prepare.py` |
| 5 | Fitted transforms (winsorise, impute, encode) — refit per fold, never globally | `src/features/engineering.py` |
| 6 | Offline feature store, point-in-time correct | `feature_repo/`, `src/pipelines/features.py` |
| 7 | Tune 3 candidates + the heuristic baseline, select on a held-out out-of-time window | `src/pipelines/train.py` |
| 8 | Export the champion, serve it behind a batch API, containerise | `src/pipelines/export_champion.py`, `src/api/` |
| 9 | Drift detection, request logging, alerting | `src/monitoring/` |

Full detail per stage in `docs/`: [architecture](docs/architecture.md) ·
[feature engineering](docs/feature-engineering.md) · [feature store](docs/feature-store.md) ·
[model training](docs/model-training.md) · [serving](docs/serving.md) ·
[monitoring](docs/monitoring.md).

## Run it end to end

```bash
make install
make features feast-apply          # build the feature table + offline store
make train                         # tune, track in MLflow, select a champion (~2.5 min)
make export-champion export-reference
docker compose up --build          # batch API at :8000, monitoring endpoint included
make monitor                       # offline drift + model-quality report
```

```bash
curl -X POST localhost:8000/predict/batch -H 'Content-Type: application/json' -d '{
  "records": [{"application_id":"APP-1","tipo_credito":"9","capital_prestado":4200000,
    "plazo_meses":24,"edad_cliente":23,"tipo_laboral":"Independiente","salario_cliente":1200000,
    "total_otros_prestamos":3500000,"cuota_pactada":390000,"puntaje_datacredito":680,
    "cant_creditosvigentes":7,"huella_consulta":11,"tendencia_ingresos":"Decreciente"}]}'
```

## Results

Out-of-time (train on the oldest 75% of vintages, test on the newest):

| Model | Gini | PR-AUC |
| --- | ---: | ---: |
| Heuristic scorecard (8 rules, frozen, no fitting) | 0.324 | 0.059 |
| **Champion — logistic regression** (Optuna-tuned) | **0.327** | **0.129** |

The champion is the first model in the project to beat the hand-built baseline on both
metrics — 2.2× the incumbent on PR-AUC, which is what matters at a 4.75% default rate. It
was chosen on the held-out window, not cross-validation: the best CV model (XGBoost,
0.187 CV mean) was the *worst* out-of-time (0.091 PR-AUC) — a reminder that this book has
real vintage structure a shuffled fold cannot see.

Full comparison and the reasoning behind the selection protocol in
[`docs/model-training.md`](docs/model-training.md).
