"""Submit the AutoML Tabular training job.

    GOOGLE_CLOUD_QUOTA_PROJECT=mlops-roysandoval uv run --group gcp python -m src.automl.train
"""

from src.automl.spec import AutomlSpec, default_automl_spec


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
    from google.cloud import aiplatform

    spec = default_automl_spec()
    dataset, job, model = submit(spec, aiplatform)
    print("dataset:", dataset.resource_name, flush=True)
    model.wait()  # the SDK logs the console URL on start; this keeps the process alive until training ends
    print("pipeline:", job.resource_name, flush=True)
    print("model:", model.resource_name, flush=True)


if __name__ == "__main__":
    main()
