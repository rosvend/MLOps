"""Shape of config/gcp_gate/default.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

GATE_PATH = Path(__file__).resolve().parents[2] / "config" / "gcp_gate" / "default.yaml"


class Bootstrap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resamples: int = Field(gt=0)
    confidence: float = Field(gt=0, lt=1)
    seed: int


class Promotion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_pr_auc_gain: float
    bootstrap: Bootstrap


class Sanity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flagged_share_ratio: tuple[float, float]


class GateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    champion_uri: str
    evaluation_uri: str
    limitation: str
    promotion: Promotion
    sanity: Sanity


@lru_cache(maxsize=1)
def default_gate_spec() -> GateSpec:
    return GateSpec(**yaml.safe_load(GATE_PATH.read_text(encoding="utf-8")))
