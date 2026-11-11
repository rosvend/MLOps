# Custom training pipeline on Vertex AI

A pipeline that trains the project's three candidates (logistic, XGBoost, LightGBM) on Google
Cloud, scores each once on the out-of-time TEST window, and compares them with the local champion.

```
src/gcp/spec.py       typed loader for config/gcp_pipeline/default.yaml
src/gcp/data.py       split file -> train (TRAIN+VALIDATE) / test (TEST)
src/gcp/evaluate.py   build the model, fit, score -> metrics dict
src/gcp/compare.py    best candidate + difference to the local champion
src/gcp/storage.py    read/write gs:// objects
src/gcp/tasks.py      what each pipeline step runs
src/gcp/pipeline.py   the KFP pipeline definition
src/gcp/submit.py     compile, run on Vertex, copy the result locally
docker/Dockerfile.train, docker/cloudbuild.train.yaml, docker/requirements-train.txt
```

All the functions are pure and unit-tested without network (`tests/gcp/`). Only `storage.py`,
`tasks.py` and `submit.py` touch the cloud.

## The pipeline, step by step

| Step | What it does | Why |
|---|---|---|
| `train-candidate` x3 (parallel) | Reads `loans_split.csv`, trains one model on TRAIN+VALIDATE, scores TEST once, writes `metrics/<name>.json` | One step per model keeps them independent and parallel; each fails or retries alone |
| `compare-candidates` | Reads the three JSONs, picks the best by PR-AUC, computes differences to the local champion, writes `custom_pipeline_metrics.json` | One file with the answer |

Choices worth knowing:

- **No hyperparameter search here.** The Optuna search already ran locally. The tuned values for
  each model were copied from the local MLflow run into `config/gcp_pipeline/default.yaml`
  (`candidates.*.params`); the estimator and fixed settings still come from
  `config/training/default.yaml`. Same code (`build_model`), same metrics (`summarize_classification`).
- **No data leakage.** TEST is only scored, never fitted on. Imputation, winsor caps and scaling
  live inside the sklearn pipeline, so they are fitted on the training rows only.
- **Threshold.** F1, precision and recall use the same rule as the local evaluation: flag the
  15.7 % highest-scored loans of the TEST window.
- **Serving artifacts.** Each candidate step also saves the fitted model so the cloud model can be
  deployed instead of the local one (`src/gcp/artifacts.py`). Layout, per candidate:
  `<run_dir>/models/<name>/champion.joblib`, `champion.meta.json` and `reference_data.parquet`.
  They have the same format as `make export-champion` / `make export-reference`: the threshold is
  the 84.3 % score quantile (flagged share 0.157) on the training rows, frozen at fit; the meta
  carries the Vertex run id (`vertex_pipeline_run_id`) and the git sha (a pipeline parameter set by
  `submit.py`) instead of the MLflow ids; the reference is the training features plus
  `probability_default`, `review_flag` and `defaulted`, with the nullable dtypes restored.
- **Experiments.** The job is submitted with the experiment `loan-default` (`experiment` in the
  config), and after the run `src/gcp/experiments.py` logs one experiment run per candidate
  (`<name>-<run id>`) with its tuned params and all metrics, so they compare side by side in
  Vertex AI > Experiments.
- **Versions.** The joblib is a pickled sklearn pipeline, so `docker/requirements-train.txt` pins
  scikit-learn, numpy, pandas and joblib to the versions the `serving` group resolves to in
  `uv.lock` (1.9.0, 2.5.2, 2.3.3, 1.5.3); `pyarrow` is pinned to write the parquet.

## GCP services involved

- **Cloud Storage (GCS)**: the bucket `gs://mlops-roysandoval-data` holds the input CSV and every
  pipeline output (`pipeline-runs/<run>/`).
- **Artifact Registry**: a private store for container images. Repo `mlops` (created in
  `infra/main.tf`) holds the training image.
- **Cloud Build**: builds the Docker image in the cloud (no local Docker needed) and pushes it to Artifact Registry.
- **Vertex AI Pipelines**: runs the KFP pipeline. Each step becomes a short-lived container
  started on a VM (`e2-highmem-2`, 2 vCPU, 16 GB, chosen by Vertex from the 2 CPU / 8 GB request in
  the config) which is deleted when the step ends. No GPU.

IAM: Vertex's pipeline service agent
(`service-<project-number>@gcp-sa-aiplatform-cc.iam.gserviceaccount.com`) must be allowed to pull
the image, so `infra/main.tf` grants it `roles/artifactregistry.reader` on that one repository.
The first run failed with a permission error until this was added. Storage access worked with the
default compute service account.

## Commands run

```bash
export GOOGLE_CLOUD_QUOTA_PROJECT=mlops-roysandoval
uv run pytest tests/gcp -q                 # unit tests, no network
cd infra && tofu plan && tofu apply        # Artifact Registry repo + reader binding
make build-train-image                     # gcloud builds submit --config docker/cloudbuild.train.yaml .
make gcp-pipeline                          # uv run python -m src.gcp.submit
```

`.gcloudignore` keeps the build upload to `src/`, `config/` and `docker/` (anchored with leading
slashes: an unanchored `models` also dropped `src/models` and broke the first image).

## Result

Run `run-20261111-040709`, Vertex pipeline job `loan-default-custom-20261110230712`. Metrics:
`gs://mlops-roysandoval-data/pipeline-runs/run-20261111-040709/custom_pipeline_metrics.json`, copied to
`reports/gcp/custom_pipeline_metrics.json`. Train 7,713 loans, TEST 3,050.

| Model | PR-AUC | ROC-AUC | Gini | F1 | Precision | Recall | Brier | Fit s | Predict ms/1k |
|---|---|---|---|---|---|---|---|---|---|
| logistic | 0.1293 | 0.6636 | 0.3272 | 0.1172 | 0.0710 | 0.3366 | 0.0319 | 0.73 | 78.9 |
| xgboost | 0.0908 | 0.6525 | 0.3050 | 0.1172 | 0.0710 | 0.3366 | 0.0416 | 5.44 | 85.9 |
| lightgbm | 0.1020 | 0.6398 | 0.2796 | 0.1138 | 0.0689 | 0.3267 | 0.1307 | 5.76 | 83.5 |

Against the local champion (logistic, PR-AUC 0.129, Gini 0.327; baseline heuristic PR-AUC 0.059):

- Best on Vertex: **logistic**, a difference of +0.0003 PR-AUC and +0.0002 Gini, i.e. the same
  model with the same score. The cloud run reproduces the local numbers.
- XGBoost and LightGBM are below the champion (PR-AUC -0.038 and -0.027), as they were locally.
- The TEST window is the same by construction. The training set is TRAIN+VALIDATE; VALIDATE is
  the newest slice of the same pre-TEST window the local run trained on, so the training rows
  are in practice the same 7,713 loans and the results match.
- Latency is higher than locally (about 80 vs 13 ms per 1k rows) because it was measured on a
  small shared cloud VM right after container start; it is not a model property.

## Latest run (with artifacts)

Run `run-20261111-141822`, job `loan-default-custom-run-20261111-141822`, succeeded in about 6
minutes. Same metrics as above (logistic PR-AUC 0.1293). Logistic artifacts:

- `gs://mlops-roysandoval-data/pipeline-runs/run-20261111-141822/models/logistic/champion.joblib`
- `gs://mlops-roysandoval-data/pipeline-runs/run-20261111-141822/models/logistic/champion.meta.json`
- `gs://mlops-roysandoval-data/pipeline-runs/run-20261111-141822/models/logistic/reference_data.parquet`

Checked after downloading them: the model loads with joblib, PR-AUC on the 3,050 TEST rows is
0.12928, the threshold is 0.082282 (equal to the local one), the meta keys and features match the
local file (only the run ids differ) and the reference has the same 38 columns and dtypes.

Experiment `loan-default`: Vertex AI > Experiments, region us-central1 (project
`mlops-roysandoval`); it holds the pipeline run plus the `logistic-`, `xgboost-` and
`lightgbm-run-20261111-141822` runs. The first submit of this version failed before creating a job
(`PipelineJob.run` has no `experiment` argument; `submit(experiment=...)` is used instead). Logging
the candidate runs was interrupted by a power loss and was done afterwards from the saved metrics.
Creating the experiment also created an empty Vertex Tensorboard instance, which costs nothing
until it holds data.

## Cost

Estimated from list prices (billing data lags by a day): three Vertex pipeline runs at $0.03 (two
failed early, on IAM and on the image) plus a few minutes of `e2-highmem-2` (about $0.12/h), three
Cloud Build runs of about 2 minutes (inside the 120 free minutes/day), and a few MB in Artifact
Registry and GCS. Total well under **$0.25**, far inside the $3 cap.
