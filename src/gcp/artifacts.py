"""Serving artifacts for one trained candidate, in the same format `make export-champion` writes."""

import io

import joblib
import numpy as np
import pandas as pd

from src.gcp.storage import write_bytes, write_json

_NOT_METRICS = {"name", "threshold", "fit_seconds", "predict_ms_per_1k", "n_train", "n_test"}


def artifact_uris(run_dir: str, name: str) -> dict[str, str]:
    base = f"{run_dir}/models/{name}"
    return {
        "model": f"{base}/champion.joblib",
        "meta": f"{base}/champion.meta.json",
        "reference": f"{base}/reference_data.parquet",
    }


def freeze_threshold(train_scores, flagged_share: float) -> float:
    """Cut-point from the training window only; fixed from here on."""
    return float(np.quantile(train_scores, 1 - flagged_share))


def build_meta(
    name: str, threshold: float, flagged_share: float, features: list[str], result: dict,
    run_dir: str, git_sha: str, calibrated_on: str,
) -> dict:
    return {
        "model": name,
        "threshold": threshold,
        "flagged_share_at_fit": flagged_share,
        "calibrated_on": calibrated_on,
        "features": features,
        "metrics": {k: v for k, v in result.items() if k not in _NOT_METRICS},
        "vertex_pipeline_run_id": run_dir.rstrip("/").rsplit("/", 1)[-1],
        "git_sha": git_sha,
    }


def _restore_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """The CSV loses the nullable dtypes the local reference has; put them back."""
    out = X.convert_dtypes()
    for col in out.columns:
        kind, has_na = str(out[col].dtype), out[col].isna().any()
        if kind == "Int64" and not has_na:
            out[col] = out[col].astype("int64")
        elif kind == "boolean" and not has_na:
            out[col] = out[col].astype(bool)
        elif kind == "Int64" and set(out[col].dropna().unique()) <= {0, 1}:
            out[col] = out[col].astype("boolean")
    return out


def build_reference(X_train: pd.DataFrame, proba, threshold: float, y_train) -> pd.DataFrame:
    ref = _restore_dtypes(X_train)
    ref["probability_default"] = proba
    ref["review_flag"] = proba >= threshold
    ref["defaulted"] = np.asarray(y_train).astype(bool)
    return ref


def save_artifacts(
    name: str, model, X_train: pd.DataFrame, y_train, result: dict, run_dir: str,
    flagged_share: float, git_sha: str, calibrated_on: str, client=None,
) -> dict:
    proba = model.predict_proba(X_train)[:, 1]
    threshold = freeze_threshold(proba, flagged_share)
    meta = build_meta(
        name, threshold, flagged_share, list(X_train.columns), result, run_dir, git_sha, calibrated_on
    )
    reference = build_reference(X_train, proba, threshold, y_train)

    uris = artifact_uris(run_dir, name)
    model_bytes, ref_bytes = io.BytesIO(), io.BytesIO()
    joblib.dump(model, model_bytes)
    reference.to_parquet(ref_bytes, index=False)
    write_bytes(uris["model"], model_bytes.getvalue(), client=client)
    write_json(uris["meta"], meta, client)
    write_bytes(uris["reference"], ref_bytes.getvalue(), client=client)
    return meta
