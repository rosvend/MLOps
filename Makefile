.PHONY: install test eda score

install:
	uv sync

test:
	uv run pytest -q

eda:
	uv run jupyter lab notebooks/eda.ipynb

score:
	uv run python -m src.pipelines.score
