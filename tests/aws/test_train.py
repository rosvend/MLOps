import json
import tarfile

from src.aws.train import find_dataset, save_outputs


def test_find_dataset(tmp_path):
    (tmp_path / "loans_split.csv").write_text("a\n1\n")
    assert find_dataset(tmp_path).name == "loans_split.csv"


def test_save_outputs_writes_metrics_and_model(tmp_path):
    save_outputs({"name": "x", "pr_auc": 0.1}, {"m": 1}, tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text())["pr_auc"] == 0.1
    assert (tmp_path / "model.joblib").exists()
