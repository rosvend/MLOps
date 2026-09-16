"""Shape of config/training/default.yaml.

Search spaces are declared as records rather than Python, so a sweep is a config edit and
every run records exactly the space it searched. No eval anywhere.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

TRAINING_PATH = Path(__file__).resolve().parents[2] / "config" / "training" / "default.yaml"
SERVING_PATH = Path(__file__).resolve().parents[2] / "config" / "serving" / "default.yaml"


class Distribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    step: int | None = None
    log: bool = False
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _shape_matches_type(self):
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("una distribución categorical necesita choices")
        elif self.low is None or self.high is None:
            raise ValueError(f"una distribución {self.type} necesita low y high")
        elif self.low >= self.high:
            raise ValueError(f"low debe ser menor que high: {self.low} >= {self.high}")
        return self

    def suggest(self, trial, nombre: str):
        if self.type == "categorical":
            return trial.suggest_categorical(nombre, self.choices)
        if self.type == "int":
            return trial.suggest_int(nombre, int(self.low), int(self.high), step=self.step or 1)
        return trial.suggest_float(nombre, self.low, self.high, log=self.log)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimator: str
    scale: bool = False
    fixed: dict[str, Any] = {}
    search: dict[str, Distribution] = {}


class Baseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimator: str


class Split(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str
    train_fraction: float

    @model_validator(mode="after")
    def _leaves_both_sides_populated(self):
        if not 0.3 <= self.train_fraction <= 0.95:
            raise ValueError(f"split.train_fraction fuera de rango: {self.train_fraction}")
        return self


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_metric: str
    report_metrics: list[str]
    cv_scoring: str
    flagged_share: float

    @model_validator(mode="after")
    def _flagged_share_is_a_share(self):
        if not 0.0 < self.flagged_share < 1.0:
            raise ValueError(f"selection.flagged_share fuera de rango: {self.flagged_share}")
        return self

    @model_validator(mode="after")
    def _metric_names_exist(self):
        """A typo here would score every candidate -inf and crown whichever came first."""
        from src.models.metrics import METRICAS

        desconocidas = [m for m in self.report_metrics if m not in METRICAS]
        if desconocidas:
            raise ValueError(
                f"selection.report_metrics no reconoce {desconocidas}; disponibles {list(METRICAS)}"
            )
        return self

    @model_validator(mode="after")
    def _primary_is_reported(self):
        if self.primary_metric not in self.report_metrics:
            raise ValueError(
                f"selection.primary_metric {self.primary_metric!r} no está en report_metrics"
            )
        return self


class TrainingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str
    tracking_uri: str
    artifact_location: str = "mlartifacts"
    seed: int
    n_trials: int
    cv_folds: int
    split: Split
    selection: Selection
    candidates: dict[str, Candidate]
    baseline: Baseline


@lru_cache(maxsize=1)
def default_training_spec() -> TrainingSpec:
    return TrainingSpec(**yaml.safe_load(TRAINING_PATH.read_text(encoding="utf-8")))


class ServingSpec(BaseModel):
    """Where the scoring artifact lives and how large a batch may be."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    meta_path: str
    max_batch_size: int
    default_entity_id: str

    @model_validator(mode="after")
    def _batch_is_bounded(self):
        if not 0 < self.max_batch_size <= 100_000:
            raise ValueError(f"serving.max_batch_size fuera de rango: {self.max_batch_size}")
        return self


@lru_cache(maxsize=1)
def default_serving_spec() -> ServingSpec:
    """Plain YAML, no Hydra: the API composes nothing and should not ship a composer."""
    return ServingSpec(**yaml.safe_load(SERVING_PATH.read_text(encoding="utf-8")))
