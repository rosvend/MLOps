"""Score the portfolio and report how well the scorecard ranks risk.

    uv run python -m src.pipelines.score
    uv run python -m src.pipelines.score model.threshold=5
    uv run python -m src.pipelines.score --multirun model.threshold=3,4,5,6
"""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.config import Config, from_dict, load_config
from src.data.factory import build_source
from src.data.source import DataSource
from src.features.contract import TARGET
from src.models.evaluate import evaluate
from src.models.heuristic import score_frame
from src.models.scorecard import Scorecard
from src.pipelines.prepare import prepare_labelled

_log = logging.getLogger(__name__)


def score_portfolio(
    source: DataSource, threshold: int, scorecard: Scorecard | None = None
) -> tuple[pd.Series, dict[str, float]]:
    prepared = prepare_labelled(source)
    scores = score_frame(prepared, scorecard)
    return scores, evaluate(scores, ~prepared[TARGET], threshold)


def score_portfolio_from_config(
    config: Config | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    config = config or load_config()
    return score_portfolio(build_source(config.data_source), config.model.threshold, config.model)


def _report(scores: pd.Series, metrics: dict[str, float]) -> None:
    print(f"Créditos puntuados: {len(scores):,}  |  puntaje de {scores.min()} a {scores.max()}\n")
    for nombre, valor in metrics.items():
        print(f"{nombre:20} {valor:8.3f}" if pd.notna(valor) else f"{nombre:20}   no disponible")


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    config = from_dict(OmegaConf.to_container(cfg, resolve=True))
    scores, metrics = score_portfolio_from_config(config)
    _log.info("Umbral %d, scorecard base_rate %.4f", config.model.threshold, config.model.base_rate)
    _report(scores, metrics)


if __name__ == "__main__":
    main()
