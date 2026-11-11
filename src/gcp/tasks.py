"""What each pipeline step does, as plain functions the KFP components just call."""

from src.gcp.compare import compare
from src.gcp.data import split_frame
from src.gcp.evaluate import build_candidate, train_and_evaluate
from src.gcp.spec import default_dataset_spec, default_pipeline_spec
from src.gcp.storage import read_csv, read_json, write_json
from src.models.training_spec import default_training_spec


def metrics_uri(run_dir: str, name: str) -> str:
    return f"{run_dir}/metrics/{name}.json"


def run_candidate(name: str, run_dir: str, client=None) -> dict:
    """Load the split file, train one candidate on TRAIN+VALIDATE, score TEST, save the metrics."""
    spec, training = default_pipeline_spec(), default_training_spec()
    data = split_frame(read_csv(spec.dataset_uri, client), default_dataset_spec())
    model = build_candidate(training.candidates[name], spec.candidates[name].params, spec.seed)
    result = train_and_evaluate(name, model, data, training.selection.flagged_share)
    write_json(metrics_uri(run_dir, name), result, client)
    return result


def run_compare(run_dir: str, names: list[str], reference: dict, metric: str, client=None) -> dict:
    """Gather every candidate's metrics, pick the best, and save one summary file."""
    results = [read_json(metrics_uri(run_dir, n), client) for n in names]
    summary = {"candidates": results, "comparison": compare(results, reference, metric)}
    write_json(f"{run_dir}/custom_pipeline_metrics.json", summary, client)
    return summary
