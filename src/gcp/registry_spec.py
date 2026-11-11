"""Shape of config/gcp_registry/default.yaml, loaded standalone like ServingSpec."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "gcp_registry" / "default.yaml"


class Serving(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_repo: str
    image_tag: str
    predict_route: str
    health_route: str
    port: int

    @property
    def image_uri(self) -> str:
        return f"{self.image_repo}:{self.image_tag}"


class Artifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models_dir: str
    meta_file: str


class LabelKeys(BaseModel):
    model_config = ConfigDict(extra="forbid")

    git_sha: str
    run_id: str
    candidate: str


class Aliases(BaseModel):
    model_config = ConfigDict(extra="forbid")

    challenger: str
    champion: str


class RegistrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    region: str
    display_name: str
    description: str
    serving: Serving
    artifacts: Artifacts
    labels: LabelKeys
    aliases: Aliases
    description_metrics: list[str]


@lru_cache(maxsize=1)
def default_registry_spec() -> RegistrySpec:
    return RegistrySpec(**yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")))
