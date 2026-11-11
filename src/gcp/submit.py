"""Compile the pipeline and run it on Vertex AI.

    uv run python -m src.gcp.submit [--image-tag TAG]

Prints a final `run_dir=gs://...` line so CI can pick the run up.
"""

import argparse
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform, storage

from src.gcp.experiments import log_candidates
from src.gcp.pipeline import compile_pipeline
from src.gcp.spec import default_pipeline_spec
from src.gcp.storage import read_json
from src.gcp.tasks import metrics_uri

LOCAL_REPORT = Path(__file__).resolve().parents[2] / "reports" / "gcp" / "custom_pipeline_metrics.json"


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-tag", help="training image tag to run (default: the one in the config)")
    return parser.parse_args(argv)


def submit_job(vertex, spec, template: str, run_id: str, run_dir: str, sha: str) -> None:
    """Run the compiled pipeline as the dedicated service account and wait for it."""
    job = vertex.PipelineJob(
        display_name=f"{spec.pipeline_name}-{run_id}",
        template_path=template,
        parameter_values={"run_dir": run_dir, "git_sha": sha},
    )
    job.submit(experiment=spec.experiment, service_account=spec.service_account)
    job.wait()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = default_pipeline_spec()
    if args.image_tag:
        spec = spec.with_image_tag(args.image_tag)
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    run_dir = spec.run_dir(run_id)

    aiplatform.init(project=spec.project, location=spec.region)
    with tempfile.TemporaryDirectory() as tmp:
        template = compile_pipeline(str(Path(tmp) / "pipeline.yaml"), spec)
        submit_job(aiplatform, spec, template, run_id, run_dir, git_sha())

    results = [read_json(metrics_uri(run_dir, n)) for n in spec.candidates]
    params = {n: c.params for n, c in spec.candidates.items()}
    log_candidates(aiplatform, spec.experiment, run_id, results, params, spec.project, spec.region)

    bucket, _, obj = f"{run_dir}/custom_pipeline_metrics.json".removeprefix("gs://").partition("/")
    LOCAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket).blob(obj).download_to_filename(str(LOCAL_REPORT))
    print(f"Metrics: {run_dir}/custom_pipeline_metrics.json -> {LOCAL_REPORT}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
