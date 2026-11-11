import yaml

from src.gcp.pipeline import compile_pipeline
from src.gcp.spec import default_pipeline_spec


def test_one_train_step_per_candidate_then_a_compare_step(tmp_path):
    compile_pipeline(str(tmp_path / "p.yaml"))
    tasks = next(yaml.safe_load_all((tmp_path / "p.yaml").read_text()))["root"]["dag"]["tasks"]
    trains = [t for t in tasks if t.startswith("train-candidate")]
    assert len(trains) == len(default_pipeline_spec().candidates)
    assert set(tasks["compare-candidates"]["dependentTasks"]) == set(trains)
