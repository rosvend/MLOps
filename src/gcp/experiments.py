"""Log each candidate's parameters and metrics as a run of a Vertex AI Experiment."""


def metrics_for_logging(result: dict) -> dict[str, float]:
    """Vertex metrics are flat numbers: drop text, spread the confusion matrix."""
    flat = {}
    for key, value in result.items():
        if isinstance(value, dict):
            flat.update({f"cm_{k}": float(v) for k, v in value.items()})
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[key] = float(value)
    return flat


def log_candidates(aip, experiment: str, run_id: str, results: list[dict], params: dict,
                   project: str, region: str) -> None:
    aip.init(project=project, location=region, experiment=experiment)
    for result in results:
        name = result["name"]
        aip.start_run(f"{name}-{run_id}")
        aip.log_params(params[name])
        aip.log_metrics(metrics_for_logging(result))
        aip.end_run()
