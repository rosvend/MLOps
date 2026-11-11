"""Print the state of a training pipeline: `python -m src.automl.status <pipeline_id>`."""

import sys

from google.cloud import aiplatform

from src.automl.spec import default_automl_spec


def main(pipeline_id: str) -> None:
    spec = default_automl_spec()
    aiplatform.init(project=spec.project, location=spec.region)
    p = aiplatform.AutoMLTabularTrainingJob.get(pipeline_id)
    print(p.state.name, "| model:", getattr(p._gca_resource.model_to_upload, "name", None))


if __name__ == "__main__":
    main(sys.argv[1])
