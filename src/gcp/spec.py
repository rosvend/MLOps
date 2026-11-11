"""Shape of config/gcp_pipeline/default.yaml. Nothing about the cloud setup lives in Python."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from src.models.training_spec import GcpDatasetSpec

_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = _ROOT / "config" / "gcp_pipeline" / "default.yaml"
DATASET_PATH = _ROOT / "config" / "gcp_dataset" / "default.yaml"


class CandidateParams(BaseModel):
    """Tuned hyperparameters for a candidate defined in config/training/default.yaml."""

    model_config = ConfigDict(extra="forbid")

    params: dict = {}


class PipelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    region: str
    bucket: str
    dataset_uri: str
    runs_prefix: str
    pipeline_name: str
    image: str
    cpu: str
    memory: str
    seed: int
    reference: dict
    candidates: dict[str, CandidateParams]

    def run_dir(self, run_id: str) -> str:
        return f"gs://{self.bucket}/{self.runs_prefix}/{run_id}"


@lru_cache(maxsize=1)
def default_pipeline_spec() -> PipelineSpec:
    return PipelineSpec(**yaml.safe_load(PIPELINE_PATH.read_text(encoding="utf-8")))


def default_dataset_spec() -> GcpDatasetSpec:
    return GcpDatasetSpec(**yaml.safe_load(DATASET_PATH.read_text(encoding="utf-8")))
