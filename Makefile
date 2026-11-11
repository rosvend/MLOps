.PHONY: install test eda features feast-apply feast-verify score train export-champion export-reference export-gcp-dataset build-train-image gcp-pipeline docs-build docs-serve monitor serve tunnel

# Application Default Credentials may carry another project's quota project; bill this one.
export GOOGLE_CLOUD_QUOTA_PROJECT ?= mlops-roysandoval

install:
	uv sync

test:
	uv run pytest -q

eda:
	uv run jupyter lab notebooks/eda.ipynb

features:
	uv run python -m src.pipelines.features

feast-apply: features
	uv run feast -c feature_repo apply

feast-verify: feast-apply
	uv run python -m src.pipelines.feature_store_check

train: feast-apply
	uv run python -m src.pipelines.train

export-champion:
	uv run python -m src.pipelines.export_champion

export-reference: export-champion
	uv run python -m src.pipelines.export_reference

export-gcp-dataset:
	uv run python -m src.pipelines.export_gcp_dataset

build-train-image:
	gcloud builds submit --project mlops-roysandoval --region us-central1 --config docker/cloudbuild.train.yaml .

gcp-pipeline:
	uv run python -m src.gcp.submit

docs-build:
	uv run --group docs zensical build --clean

docs-serve:
	uv run --group docs zensical serve

monitor: export-reference
	uv run python -m src.pipelines.monitor

serve: export-champion export-reference
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

score:
	uv run python -m src.pipelines.score

tunnel:
	./scripts/setup_tunnel.sh
