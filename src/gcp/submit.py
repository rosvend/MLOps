"""Compile the pipeline and run it on Vertex AI.

    uv run python -m src.gcp.submit
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform, storage

from src.gcp.pipeline import compile_pipeline
from src.gcp.spec import default_pipeline_spec

LOCAL_REPORT = Path(__file__).resolve().parents[2] / "reports" / "gcp" / "custom_pipeline_metrics.json"


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
            parameter_values={"run_dir": run_dir},
        )
        job.run(sync=True)

    bucket, _, obj = f"{run_dir}/custom_pipeline_metrics.json".removeprefix("gs://").partition("/")
    LOCAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket).blob(obj).download_to_filename(str(LOCAL_REPORT))
    print(f"Metrics: {run_dir}/custom_pipeline_metrics.json -> {LOCAL_REPORT}")


if __name__ == "__main__":
    main()
