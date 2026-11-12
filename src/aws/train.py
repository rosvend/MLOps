"""SageMaker training entry point: one candidate, same project code as the Vertex pipeline."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

CHANNEL = Path("/opt/ml/input/data/train")
MODEL_DIR = Path("/opt/ml/model")
HYPERPARAMS = Path("/opt/ml/input/config/hyperparameters.json")


def find_dataset(channel: Path) -> Path:
    return next(channel.glob("*.csv"))


def save_outputs(result: dict, model, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    joblib.dump(model, model_dir / "model.joblib")


def main() -> None:
    from src.gcp.data import split_frame
    from src.gcp.evaluate import build_candidate, train_and_evaluate
    from src.gcp.spec import default_dataset_spec, default_pipeline_spec
    from src.models.training_spec import default_training_spec

    name = json.loads(HYPERPARAMS.read_text())["candidate"]
    spec, training = default_pipeline_spec(), default_training_spec()
    data = split_frame(pd.read_csv(find_dataset(CHANNEL)), default_dataset_spec())
    model = build_candidate(training.candidates[name], spec.candidates[name].params, spec.seed)
    result = train_and_evaluate(name, model, data, training.selection.flagged_share)
    save_outputs(result, model, MODEL_DIR)
    print(json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())
