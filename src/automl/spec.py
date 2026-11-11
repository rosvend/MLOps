"""Shape of config/gcp_automl/default.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

AUTOML_PATH = Path(__file__).resolve().parents[2] / "config" / "gcp_automl" / "default.yaml"
MAX_BUDGET_MILLI_NODE_HOURS = 1000


class BatchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    machine_type: str
    replicas: int
    input_uri: str
    output_prefix: str
    key_column: str


class ImputationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_uri: str
    train_value: str
    missing_label: str


class AutomlSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    region: str
    dataset_uri: str
    dataset_display_name: str
    job_display_name: str
    model_display_name: str
    label_column: str
    split_column: str
    test_value: str
    flagged_share: float
    optimization_objective: str
    budget_milli_node_hours: int
    early_stopping: bool
    batch: BatchSpec
    imputation: ImputationSpec | None = None
    categorical: list[str]
    numeric: list[str]

    @model_validator(mode="after")
    def _budget_is_capped(self):
        if not 1000 <= self.budget_milli_node_hours <= MAX_BUDGET_MILLI_NODE_HOURS:
            raise ValueError(f"budget_milli_node_hours fuera de rango: {self.budget_milli_node_hours}")
        return self

    @model_validator(mode="after")
    def _columns_are_unambiguous(self):
        todas = self.categorical + self.numeric
        if len(set(todas)) != len(todas):
            raise ValueError("una columna no puede ser categorical y numeric a la vez")
        if {self.label_column, self.split_column} & set(todas):
            raise ValueError("la etiqueta y el split no pueden ser features")
        return self

    @property
    def column_specs(self) -> dict[str, str]:
        return {c: "categorical" for c in self.categorical} | {c: "numeric" for c in self.numeric}


def load_automl_spec(path: Path = AUTOML_PATH) -> AutomlSpec:
    return AutomlSpec(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def default_automl_spec() -> AutomlSpec:
    return load_automl_spec()
