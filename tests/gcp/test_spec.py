from src.gcp.spec import default_pipeline_spec
from src.models.training_spec import default_training_spec


def test_default_spec_loads_and_candidates_exist_in_training_config():
    spec = default_pipeline_spec()
    assert set(spec.candidates) <= set(default_training_spec().candidates)
    assert spec.dataset_uri.startswith("gs://") and spec.image


def test_run_dir_is_under_the_bucket():
    assert default_pipeline_spec().run_dir("r1").startswith("gs://")


def test_pipeline_runs_as_the_dedicated_service_account():
    assert default_pipeline_spec().service_account.startswith("vertex-pipeline@")


def test_with_image_tag_swaps_only_the_tag():
    spec = default_pipeline_spec()
    tagged = spec.with_image_tag("abc123")
    assert tagged.image == spec.image.rsplit(":", 1)[0] + ":abc123"
    assert spec.image.endswith(":latest")
