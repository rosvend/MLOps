"""Hydra composes the config, pydantic validates it.

Hydra owns config groups, CLI overrides and --multirun sweeps; pydantic stays the
typed contract, so a wrong shape fails at load rather than deep inside a pipeline.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from src.features.spec import FeatureSpec
from src.models.scorecard import Scorecard

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG_NAME = "config"


class DataSourceConfig(BaseModel):
    """Storage-agnostic: `type` picks the adapter, the rest is passed to its constructor."""

    model_config = ConfigDict(extra="allow")

    type: str

    @property
    def options(self) -> dict:
        return self.model_dump(exclude={"type"})


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source: DataSourceConfig
    features: FeatureSpec
    model: Scorecard


def from_dict(raw: dict) -> Config:
    return Config(**raw)


def load_config(
    overrides: list[str] | None = None,
    config_dir: str | Path = CONFIG_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
) -> Config:
    """Compose API rather than @hydra.main, so tests and notebooks can load config too.

    Reuses an existing Hydra session when there is one: initialize_config_dir raises if
    called from inside a @hydra.main entry point.
    """
    if GlobalHydra.instance().is_initialized():
        cfg = compose(config_name=config_name, overrides=overrides or [])
    else:
        with initialize_config_dir(version_base=None, config_dir=str(Path(config_dir).resolve())):
            cfg = compose(config_name=config_name, overrides=overrides or [])
    return from_dict(OmegaConf.to_container(cfg, resolve=True))
