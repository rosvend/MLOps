"""Compile the pipeline and run it on Vertex AI.

    uv run python -m src.gcp.submit
"""

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


def main() -> None:
    spec = default_pipeline_spec()
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    run_dir = spec.run_dir(run_id)

    aiplatform.init(project=spec.project, location=spec.region)
    with tempfile.TemporaryDirectory() as tmp:
        template = compile_pipeline(str(Path(tmp) / "pipeline.yaml"), spec)
        job = aiplatform.PipelineJob(
            display_name=f"{spec.pipeline_name}-{run_id}",
            template_path=template,
            parameter_values={"run_dir": run_dir, "git_sha": git_sha()},
        )
        job.submit(experiment=spec.experiment)
        job.wait()

    results = [read_json(metrics_uri(run_dir, n)) for n in spec.candidates]
    params = {n: c.params for n, c in spec.candidates.items()}
    log_candidates(aiplatform, spec.experiment, run_id, results, params, spec.project, spec.region)

    bucket, _, obj = f"{run_dir}/custom_pipeline_metrics.json".removeprefix("gs://").partition("/")
    LOCAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket).blob(obj).download_to_filename(str(LOCAL_REPORT))
    print(f"Metrics: {run_dir}/custom_pipeline_metrics.json -> {LOCAL_REPORT}")


if __name__ == "__main__":
    main()
