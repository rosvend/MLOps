.PHONY: install test eda features feast-apply feast-verify score train

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

score:
	uv run python -m src.pipelines.score
