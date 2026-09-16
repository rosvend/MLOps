"""How monitoring behaves, loaded from config/monitoring/default.yaml.

Same plain-YAML-cached pattern as src/features/spec.py and
src/models/training_spec.py::default_serving_spec - no Hydra, so the API can load it
without importing a composer it never uses.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

SPEC_PATH = Path(__file__).resolve().parents[2] / "config" / "monitoring" / "default.yaml"


class MonitoringSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_path: str
    predictions_db_path: str

    drift_share_threshold: float
    flagged_share_threshold: float

    numeric_sample_size_cutoff: int
    numeric_small_sample_stattest: str
    numeric_large_sample_stattest: str
    categorical_stattest: str
    alpha: float

    min_current_rows: int

    webhook_url: str | None = None

    offline_report_html: str
    offline_report_json: str
    live_report_html: str
    live_report_json: str

    @model_validator(mode="after")
    def _shares_are_shares(self):
        for nombre in ("drift_share_threshold", "flagged_share_threshold", "alpha"):
            valor = getattr(self, nombre)
            if not 0.0 < valor < 1.0:
                raise ValueError(f"monitoring.{nombre} fuera de rango: {valor}")
        return self

    @model_validator(mode="after")
    def _positive_sizes(self):
        if self.numeric_sample_size_cutoff <= 0:
            raise ValueError("monitoring.numeric_sample_size_cutoff debe ser positivo")
        if self.min_current_rows <= 0:
            raise ValueError("monitoring.min_current_rows debe ser positivo")
        return self


@lru_cache(maxsize=1)
def default_monitoring_spec() -> MonitoringSpec:
    return MonitoringSpec(**yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8")))
