# Design decisions

Each choice, with the alternative and the reason. They are also the raw material for comparing
cloud providers.

## CI/CD: GitHub Actions, not Jenkins

| | GitHub Actions | Jenkins |
| --- | --- | --- |
| Runs on | GitHub's hosted runners | A server you operate |
| Pipeline file | YAML in `.github/workflows/` | `Jenkinsfile` (Groovy) |
| Cloud login | Workload Identity Federation, no keys | Stored credentials |
| Fits | Code already on GitHub | Teams with on-premise servers |

## Serving: Cloud Run, not a Vertex endpoint

| | Cloud Run | Vertex endpoint |
| --- | --- | --- |
| What it is | Serverless runner for any container | Managed model-serving service |
| Idle cost | Scales to zero | Keeps at least one replica running (check current pricing) |
| This API | Runs as-is, including `/health` and the drift endpoint | Expects a prediction contract; extra routes are awkward |
| Feature logic | Lives in the container, so training and serving share code | The same, if you use a custom container |
| Monitoring | Evidently, which we run ourselves | Integrates with Vertex Model Monitoring |
| Fits | Light models behind a custom API | Registry-to-endpoint flows, GPUs, large models |

Both are common. Cloud Run fits this workload; a Vertex endpoint is a reasonable alternative
when a team wants a fully managed platform.

## Prediction log: BigQuery, not Cloud SQL

| | BigQuery | Cloud SQL (Postgres) |
| --- | --- | --- |
| Operation | Serverless, IAM only | An instance that bills while it runs, plus secrets |
| Writes | Append-friendly | Row-oriented, transactional |
| Analytics | Native | Needs export or extra tooling |

A prediction log is append-only and read in bulk for drift checks, which suits BigQuery. The log
sits behind an interface, so the local SQLite version and the BigQuery version are swappable by
config.

## Experiment tracking: MLflow

MLflow works on any cloud and has a model registry, so the same tracking code runs locally and in
the cloud. Vertex Experiments is the managed alternative and ties you to Google Cloud.

## Infrastructure: OpenTofu

Resources are declared in files, reviewed with `plan`, and removed with `destroy`. The state file
is local for this project. A team would keep it in a shared bucket with locking.

## Training in the cloud: when it pays off

For this dataset (about 10,000 rows) the cloud adds time and cost without improving the model.
Each pipeline step waits for a fresh virtual machine, so most of the wall-clock time is
start-up. Cloud training earns its place with large data, scheduled retraining and the need for
reproducible, auditable runs.

## AutoML: price versus control

AutoML tabular training is billed per node-hour (about $21.25 at list price) with a 1 node-hour
minimum budget. It is convenient, but costs more than the whole custom pipeline and offers less
control over the model.
