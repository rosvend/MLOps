.PHONY: install test eda features score

install:
	uv sync

test:
	uv run pytest -q

eda:
	uv run jupyter lab notebooks/eda.ipynb

features:
	uv run python -m src.pipelines.features

score:
	uv run python -m src.pipelines.score
