.PHONY: install test eda features feast-apply feast-verify score train export-champion serve tunnel

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

serve: export-champion
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

score:
	uv run python -m src.pipelines.score

tunnel:
	./scripts/setup_tunnel.sh
