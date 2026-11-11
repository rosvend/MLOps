import io
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.gcp.artifacts import (
    artifact_uris,
    build_meta,
    build_reference,
    freeze_threshold,
    save_artifacts,
)
from tests.gcp.test_storage import FakeClient


def test_threshold_is_the_training_score_quantile():
    scores = np.linspace(0, 1, 1001)
    assert abs(freeze_threshold(scores, 0.2) - 0.8) < 1e-9


def test_artifact_uris_follow_the_fixed_layout():
    u = artifact_uris("gs://b/pipeline-runs/r1", "logistic")
    base = "gs://b/pipeline-runs/r1/models/logistic"
    assert u == {
        "model": f"{base}/champion.joblib",
        "meta": f"{base}/champion.meta.json",
        "reference": f"{base}/reference_data.parquet",
    }


def test_meta_has_the_keys_the_api_reads_and_only_metrics_from_the_result():
    result = {"name": "m", "pr_auc": 0.1, "gini": 0.3, "confusion_matrix": {"tp": 1},
              "threshold": 0.5, "fit_seconds": 1.0, "predict_ms_per_1k": 2.0,
              "n_train": 5, "n_test": 5}
    meta = build_meta("m", 0.08, 0.157, ["a", "b"], result, "gs://b/pipeline-runs/r1", "abc123", "calib")
    assert set(meta) == {"model", "threshold", "flagged_share_at_fit", "calibrated_on", "features",
                         "metrics", "vertex_pipeline_run_id", "git_sha"}
    assert meta["vertex_pipeline_run_id"] == "r1" and meta["git_sha"] == "abc123"
    assert meta["metrics"] == {"pr_auc": 0.1, "gini": 0.3, "confusion_matrix": {"tp": 1}}
    assert meta["threshold"] == 0.08


def test_reference_adds_predictions_flag_and_label_and_restores_dtypes():
    X = pd.DataFrame({
        "n": [1.0, 2.0, 3.0], "gap": [1.0, np.nan, 3.0], "flag": [1.0, np.nan, 0.0],
        "falta": [True, False, True], "txt": ["a", None, "b"],
    })
    ref = build_reference(X, np.array([0.1, 0.2, 0.3]), 0.2, pd.Series([0, 1, 0]))
    assert list(ref.columns) == [*X.columns, "probability_default", "review_flag", "defaulted"]
    assert ref["review_flag"].tolist() == [False, True, True]
    assert ref["defaulted"].dtype == bool
    assert str(ref["n"].dtype) == "int64" and str(ref["gap"].dtype) == "Int64"
    assert str(ref["flag"].dtype) == "boolean" and str(ref["falta"].dtype) == "bool"
    assert str(ref["txt"].dtype) == "string"


def test_save_artifacts_writes_the_three_files():
    c = FakeClient()
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    result = {"pr_auc": 0.5}
    save_artifacts("lr", model, X, y, result, "gs://b/pipeline-runs/r1", 0.5, "sha", "calib", c)

    assert isinstance(joblib.load(io.BytesIO(c.store["b/pipeline-runs/r1/models/lr/champion.joblib"])),
                      LogisticRegression)
    meta = json.loads(c.store["b/pipeline-runs/r1/models/lr/champion.meta.json"])
    assert meta["features"] == ["a"] and meta["model"] == "lr"
    ref = pd.read_parquet(io.BytesIO(c.store["b/pipeline-runs/r1/models/lr/reference_data.parquet"]))
    assert len(ref) == 4 and "probability_default" in ref
    proba = model.predict_proba(X)[:, 1]
    assert abs(meta["threshold"] - np.quantile(proba, 0.5)) < 1e-12
