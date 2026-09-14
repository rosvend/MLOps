from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data.factory import build_source
from src.data.source import DataSource
from src.models.evaluate import evaluate
from src.models.heuristic import score_frame
from src.pipelines.prepare import prepare


def score_portfolio(source: DataSource, threshold: int) -> tuple[pd.Series, dict[str, float]]:
    prepared = prepare(source)
    scores = score_frame(prepared)
    return scores, evaluate(scores, ~prepared["Pago_atiempo"], threshold)


def score_portfolio_from_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[pd.Series, dict[str, float]]:
    config = load_config(config_path)
    return score_portfolio(build_source(config.data_source), config.model.threshold)


def main() -> None:
    scores, metrics = score_portfolio_from_config()
    print(f"Créditos puntuados: {len(scores):,}  |  puntaje de {scores.min()} a {scores.max()}\n")
    for nombre, valor in metrics.items():
        print(f"{nombre:20} {valor:8.3f}")


if __name__ == "__main__":
    main()
