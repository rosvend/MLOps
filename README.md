<div align="center">

<br/>

# MLOps Pipeline — Loan default 
**A production-grade, end-to-end MLOps system for bank loan predictions**

Reproducible experiments · Automated CI/CD · Containerized serving · Live drift monitoring

<br/>

[![CI/CD](https://github.com/MOHD-OMER/mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/MOHD-OMER/mlops-pipeline/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/docker/v/omer022/mlops-news-classifier?label=DockerHub&logo=docker&color=2496ED)](https://hub.docker.com/r/omer022/mlops-news-classifier)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

<br/>

[Overview](#-overview) · [Architecture](#-architecture) · [Pipeline Stages](#-pipeline-stages) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Results](#-results) · [Configuration](#-configuration) · [Extending](#-extending-the-pipeline)

<br/>

</div>

---

## Architecture

<p align="center">
  <img src="docs/plots/pipeline_diagram.png" alt="Pipeline: raw loan data flows through prep/validation, feature engineering and an offline Feast feature store, into Optuna/MLflow training that exports the best model, which is served behind a FastAPI batch API and watched by Evidently-based drift monitoring that signals back into training." width="100%">
</p>

Full detail per stage in `docs/`: [architecture](docs/architecture.md) ·
[feature engineering](docs/feature-engineering.md) · [feature store](docs/feature-store.md) ·
[model training](docs/model-training.md) · [serving](docs/serving.md) ·
[monitoring](docs/monitoring.md).


---

## 🚀 Quick Start 

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- uv
- Git & DVC

### 1. Clone & install

```bash
git clone https://github.com/rosvend/mlops.git
cd mlops-pipeline
uv sync
```
### 2. Initialise DVC

```bash
dvc init
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "chore: track raw data with DVC"
```
### 3. Run the full pipeline

```bash
make install
make features feast-apply          # build the feature table + offline store
make train                         # tune, track in MLflow, select a champion (~2.5 min)
make export-champion export-reference
docker compose up --build          # batch API at :8000, monitoring endpoint included
make monitor                       # offline drift + model-quality report
```
### 4. Test the FastAPI endpoint

```bash
curl -X POST localhost:8000/predict/batch -H 'Content-Type: application/json' -d '{
  "records": [{"application_id":"APP-1","tipo_credito":"9","capital_prestado":4200000,
    "plazo_meses":24,"edad_cliente":23,"tipo_laboral":"Independiente","salario_cliente":1200000,
    "total_otros_prestamos":3500000,"cuota_pactada":390000,"puntaje_datacredito":680,
    "cant_creditosvigentes":7,"huella_consulta":11,"tendencia_ingresos":"Decreciente"}]}'
```
#### API Reference

> **Coming soon**: Interactive docs auto-generated at [http://localhost:8000/docs](http://localhost:8000/docs).

## Results

Out-of-time (train on the oldest 75% of vintages, test on the newest) comparison across all
four candidates — the heuristic baseline, logistic regression, XGBoost and LightGBM:

<p align="center">
  <img src="docs/plots/model_comparison_performance.png" alt="PR-AUC and Gini per model, out of time" width="85%">
</p>
<p align="center">
  <sub><b>PR-AUC and Gini per model, out of time</b> — the headline comparison across all four candidates</sub>
</p>

<p align="center">
  <img src="docs/plots/confusion_matrices.png" alt="Confusion matrices at each model's own operating threshold" width="85%">
</p>
<p align="center">
  <sub><b>Confusion matrices</b>, each model at its own operating threshold</sub>
</p>

<p align="center">
  <img src="docs/plots/roc_curves.png" alt="ROC curves for all four models" height="240">
  <img src="docs/plots/pr_curves.png" alt="Precision-recall curves for all four models" height="240">
</p>
<p align="center">
  <sub><b>ROC curves</b></sub> &nbsp;·&nbsp;
  <sub><b>Precision-recall curves</b> — the metric the selection is actually made on</sub>
</p>

<p align="center">
  <img src="docs/plots/model_comparison_cv_vs_oot.png" alt="CV mean vs out-of-time PR-AUC per tuned candidate" height="240">
  <img src="docs/plots/parallel_coordinates.png" alt="Six metrics normalized, parallel coordinates" height="240">
</p>
<p align="center">
  <sub><b>CV mean vs. out-of-time PR-AUC</b> — tuning picked a different model than selection did</sub> &nbsp;·&nbsp;
  <sub><b>Six metrics side by side</b>, normalized, parallel coordinates</sub>
</p>

The best model until now is the Optuna-tuned logistic regression — 0.129 PR-AUC and 0.327
Gini out of time, 2.2× the heuristic baseline on PR-AUC (the metric that matters at a 4.75%
default rate). It won on the held-out vintage window, not cross-validation: XGBoost scored
best in CV (0.187) but worst out of time (0.091), a reminder that this book has real vintage
structure a shuffled fold can't see.

Full comparison and the reasoning behind the selection protocol in
[`docs/model-training.md`](docs/model-training.md).
