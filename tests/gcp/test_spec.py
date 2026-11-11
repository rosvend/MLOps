from src.gcp.spec import default_pipeline_spec
from src.models.training_spec import default_training_spec


def test_default_spec_loads_and_candidates_exist_in_training_config():
    spec = default_pipeline_spec()
    assert set(spec.candidates) <= set(default_training_spec().candidates)
    assert spec.dataset_uri.startswith("gs://") and spec.image


def test_run_dir_is_under_the_bucket():
    assert default_pipeline_spec().run_dir("r1").startswith("gs://")
