"""Stage 5 entrypoint: materialise the feature table a feature store can register.

    uv run python -m src.pipelines.features
    uv run python -m src.pipelines.features features.engineering.winsor_quantile=0.95

What lands here is deliberately only what is a pure function of one row: cleaning,
derived ratios, bands and missingness indicators, keyed by the entity and the event
timestamp.

The fitted transforms in src/features/engineering.py are deliberately NOT applied.
Winsorising and imputation learn statistics from a sample; materialising their output
would bake statistics taken over the whole book into the store, and every future
train/test split would silently inherit them. They belong inside the model pipeline,
refitted on each training split.
"""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.config import Config, from_dict
from src.data.factory import build_source
from src.data.source import DataSource
from src.features.contract import feature_names
from src.features.spec import FeatureSpec
from src.pipelines.prepare import prepare_features

_log = logging.getLogger(__name__)


def build_feature_table(source: DataSource, spec: FeatureSpec) -> pd.DataFrame:
    """Entity key, event timestamp, then every row-independent feature."""
    prepared = prepare_features(source, spec)
    columnas = [spec.entity_key, spec.event_timestamp, *feature_names(prepared.columns, spec)]
    return prepared[columnas].copy()


def materialise(source: DataSource, spec: FeatureSpec, destino: str | Path) -> pd.DataFrame:
    tabla = build_feature_table(source, spec)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(destino, index=False)
    _log.info("Feature table: %d filas x %d columnas -> %s", len(tabla), tabla.shape[1], destino)
    return tabla


def materialise_from_config(config: Config) -> pd.DataFrame:
    return materialise(build_source(config.data_source), config.features, config.features.output_path)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    config = from_dict(OmegaConf.to_container(cfg, resolve=True))
    tabla = materialise_from_config(config)
    print(f"Feature table: {len(tabla):,} créditos x {tabla.shape[1]} columnas")
    print(f"Clave: {config.features.entity_key}  |  timestamp: {config.features.event_timestamp}")
    print(f"Escrito en {config.features.output_path}")


if __name__ == "__main__":
    main()
