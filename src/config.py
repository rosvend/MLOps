from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


class DataSourceConfig(BaseModel):
    """Storage-agnostic: `type` picks the adapter, the rest is passed to its constructor."""

    model_config = ConfigDict(extra="allow")

    type: str

    @property
    def options(self) -> dict:
        return self.model_dump(exclude={"type"})


class Config(BaseModel):
    data_source: DataSourceConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    return Config(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
