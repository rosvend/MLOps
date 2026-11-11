"""Submit the AutoML Tabular training job.

    GOOGLE_CLOUD_QUOTA_PROJECT=mlops-roysandoval uv run --group gcp python -m src.automl.train
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.automl.spec import AutomlSpec, default_automl_spec, load_automl_spec


def submit(spec: AutomlSpec, sdk):
    """Create the dataset and start training; the predefined split keeps TEST out of training."""
    sdk.init(project=spec.project, location=spec.region)
    dataset = sdk.TabularDataset.create(
        display_name=spec.dataset_display_name, gcs_source=spec.dataset_uri
    )
    job = sdk.AutoMLTabularTrainingJob(
        display_name=spec.job_display_name,
        optimization_prediction_type="classification",
        optimization_objective=spec.optimization_objective,
        column_specs=spec.column_specs,
    )
    model = job.run(
        dataset=dataset,
        target_column=spec.label_column,
        predefined_split_column_name=spec.split_column,
        budget_milli_node_hours=spec.budget_milli_node_hours,
        model_display_name=spec.model_display_name,
        disable_early_stopping=not spec.early_stopping,
        sync=False,
    )
    return dataset, job, model


def main() -> None:
    import os
    import time

    from google.cloud import aiplatform

    spec = load_automl_spec(Path(sys.argv[1])) if len(sys.argv) > 1 else default_automl_spec()
    state_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    submitted = datetime.now(timezone.utc).isoformat()
    dataset, job, _ = submit(spec, aiplatform)
    while job._gca_resource is None:  # the pipeline is created in a background thread
        time.sleep(3)
    state = {
        "job_display_name": spec.job_display_name,
        "dataset": dataset.resource_name,
        "training_pipeline": job.resource_name,
        "console_url": f"https://console.cloud.google.com/vertex-ai/locations/{spec.region}"
        f"/training/{job.resource_name.rsplit('/', 1)[-1]}?project={spec.project}",
        "submitted_at_utc": submitted,
        "budget_milli_node_hours": spec.budget_milli_node_hours,
    }
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2), flush=True)
    os._exit(0)  # training runs server-side; do not wait for it


if __name__ == "__main__":
    main()
