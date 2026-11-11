# Model Registry

## What it is

The Vertex AI Model Registry is the catalogue of the models we trust enough to keep. Each entry
says which files make the model, which container serves it, which code and run produced it, and
how good it was on the test window. Training jobs come and go; the registry is where a model
gets a name, a version number and a status.

## How it differs from the other trackers

| Tool | Answers | Granularity |
| --- | --- | --- |
| MLflow (local) | "What did each trial and run do?" | every run, every trial |
| Vertex Experiments | the same question, but on GCP | every run |
| Model Registry | "Which model versions exist and which one is live?" | only models worth keeping |

Experiment tracking is a lab notebook, the registry is the shelf of finished products. A run
that is not good enough never gets registered.

## Versions and aliases

All our models share one entry, `loan-default-scorer`. Every registration adds a new
**version** (1, 2, 3...); versions are never overwritten. An **alias** is a movable label on a
version:

- `challenger`: the latest registered candidate, waiting to be judged.
- `champion`: the version we consider the best. Moving it is the promotion.

Aliases are unique per model, so moving `champion` to version 3 removes it from version 2.

## What gets registered

Only the files and metadata: the artifact folder `<run_dir>/models/<candidate>/`, the FastAPI
scoring image, labels (git sha, run id, candidate) and a description with the test metrics.
The model is **never deployed to a Vertex endpoint**: our container exposes `/predict/batch`,
not Vertex's prediction contract, so the registry is a catalogue and serving stays on Cloud Run.

## Reading it in the console

Google Cloud console, project `mlops-roysandoval`: Vertex AI, Model Registry (region
us-central1). Open `loan-default-scorer` to see the versions, their aliases, labels and
description.

## Commands

```bash
# register a candidate of a pipeline run (adds a version, alias challenger)
make register-model RUN_DIR=gs://mlops-roysandoval-data/pipeline-runs/<run> CANDIDATE=logistic

# move the champion alias to a version
uv run python -m src.gcp.register --promote 3

# inspect
gcloud ai models list --region us-central1 --project mlops-roysandoval
gcloud ai models list-version <MODEL_ID> --region us-central1 --project mlops-roysandoval
```

Everything is configured in `config/gcp_registry/default.yaml` (names, image, routes, aliases).
