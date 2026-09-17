# MLOps — credit scoring

End-to-end MLOps pipeline for loan default risk: raw CSV → feature store → tuned model →
batch API → drift monitoring, for a Colombian bank's loan book.

## Tech stack

pandas + pandera (prep & validation) · scikit-learn (`FeatureEngineer`, the champion
pipeline) · Feast (offline feature store) · Optuna (hyperparameter search) · XGBoost /
LightGBM (candidates) · MLflow (experiment tracking) · FastAPI + Docker (batch serving) ·
Evidently AI (drift) · Hydra + pydantic (config, everywhere) · uv (dependency management)

## Pipeline

![Pipeline: raw loan data flows through prep/validation, feature engineering and an offline Feast feature store, into Optuna/MLflow training that exports the best model, which is served behind a FastAPI batch API and watched by Evidently-based drift monitoring that signals back into training.](docs/plots/pipeline_diagram.png)

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

Out-of-time (train on the oldest 75% of vintages, test on the newest) comparison across all
four candidates — the heuristic baseline, logistic regression, XGBoost and LightGBM:

![PR-AUC and Gini per model, out of time](docs/plots/model_comparison_performance.png)
![CV mean vs. out-of-time PR-AUC per tuned candidate](docs/plots/model_comparison_cv_vs_oot.png)
![Confusion matrices at each model's own operating threshold](docs/plots/confusion_matrices.png)
![ROC curves for all four models](docs/plots/roc_curves.png)
![Precision-recall curves for all four models](docs/plots/pr_curves.png)
![Normalized comparison across six metrics, parallel coordinates](docs/plots/parallel_coordinates.png)

The best model until now is the Optuna-tuned logistic regression — 0.129 PR-AUC and 0.327
Gini out of time, 2.2× the heuristic baseline on PR-AUC (the metric that matters at a 4.75%
default rate). It won on the held-out vintage window, not cross-validation: XGBoost scored
best in CV (0.187) but worst out of time (0.091), a reminder that this book has real vintage
structure a shuffled fold can't see.

Full comparison and the reasoning behind the selection protocol in
[`docs/model-training.md`](docs/model-training.md).
