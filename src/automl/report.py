"""After training: Vertex's own metrics, a TEST-only batch prediction, and the comparable metrics.

    GOOGLE_CLOUD_QUOTA_PROJECT=mlops-roysandoval uv run --group gcp python -m src.automl.report <config.yaml> <model> <report.json> [<batch job>]
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
from google.cloud import aiplatform, storage
from google.protobuf.json_format import MessageToDict

from src.automl.evaluate import batch_input, evaluate_scores, fill_unscored, parse_predictions
from src.automl.spec import load_automl_spec

LOCAL_CSV = Path(__file__).resolve().parents[2] / "data" / "gcp" / "loans_split.csv"


def _blob(uri: str, client):
    bucket, _, path = uri.removeprefix("gs://").partition("/")
    return client.bucket(bucket), path


def _download(uri: str, client) -> bytes:
    bucket, path = _blob(uri, client)
    return bucket.blob(path).download_as_bytes()


def vertex_metrics(model) -> dict:
    ev = model.get_model_evaluation()
    m = MessageToDict(ev._gca_resource._pb)
    return {"name": ev.resource_name, "metrics": m.get("metrics", {})}


def main(config: str, model_name: str, report_path: str, batch_job: str | None = None) -> None:
    spec = load_automl_spec(Path(config))
    aiplatform.init(project=spec.project, location=spec.region)
    model = aiplatform.Model(model_name)
    gcs = storage.Client(project=spec.project)

    # the file the model was trained on: the imputed copy for v2, the raw split file for v1
    source = io.BytesIO(_download(spec.dataset_uri, gcs)) if spec.imputation else LOCAL_CSV
    feats, y = batch_input(pd.read_csv(source), spec)
    bucket, path = _blob(spec.batch.input_uri, gcs)
    bucket.blob(path).upload_from_string(feats.to_csv(index=False), content_type="text/csv")

    job = aiplatform.BatchPredictionJob(batch_job) if batch_job else model.batch_predict(
        job_display_name=f"{spec.job_display_name}-test",
        gcs_source=spec.batch.input_uri,
        instances_format="csv",
        gcs_destination_prefix=spec.batch.output_prefix,
        predictions_format="jsonl",
        machine_type=spec.batch.machine_type,
        starting_replica_count=spec.batch.replicas,
        max_replica_count=spec.batch.replicas,
    )
    out_bucket, out_prefix = _blob(job.output_info.gcs_output_directory, gcs)
    lines = [
        line
        for b in out_bucket.list_blobs(prefix=out_prefix)
        if b.name.rsplit("/", 1)[-1].startswith("prediction.results")
        for line in b.download_as_text().splitlines()
        if line.strip()
    ]
    scores = parse_predictions(lines, spec)
    # AutoML rejects rows with an empty numeric cell, so only part of TEST gets a score
    y = y.reset_index(drop=True)
    y_scored = y.loc[scores.index].to_numpy()
    full = fill_unscored(scores, len(y))
    result = {
        "model": model_name,
        "batch_job": job.resource_name,
        "test_rows": len(y),
        "test_positive_rate": float(y.mean()),
        "vertex_own_evaluation": vertex_metrics(model),
        "scored_rows": int(len(scores)),
        "scored_positives": int(y_scored.sum()),
        "coverage": len(scores) / len(y),
        "scored_subset_only": evaluate_scores(y_scored, scores.to_numpy(), spec.flagged_share),
        "full_window_unscored_as_zero": evaluate_scores(y.to_numpy(), full.to_numpy(), spec.flagged_share),
        "flagged_share": spec.flagged_share,
    }
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("scored_rows", "scored_positives", "scored_subset_only", "full_window_unscored_as_zero")}, indent=2, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:5])
