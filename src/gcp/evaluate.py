"""Fit one model on the training window and score the TEST window once."""

import time
import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from src.features.spec import FeatureSpec
from src.gcp.data import SplitData
from src.models.metrics import summarize_classification
from src.models.pipelines import build_model
from src.models.training_spec import Candidate


def build_candidate(
    cand: Candidate, params: dict[str, Any], seed: int, features: FeatureSpec | None = None
) -> BaseEstimator:
    """The same pipeline the local run builds: fixed settings plus the tuned params."""
    return build_model(
        cand.estimator, cand.fixed | params, scale=cand.scale, spec=features, seed=seed
    )


def train_and_evaluate(name: str, model: BaseEstimator, data: SplitData, flagged_share: float) -> dict:
    """Metrics dict; the threshold flags the same share of loans as the local evaluation."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data.X_train, data.y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    scores = model.predict_proba(data.X_test)[:, 1]
    predict_ms = (time.perf_counter() - start) * 1_000_000 / max(len(data.X_test), 1)

    threshold = float(np.quantile(scores, 1 - flagged_share))
    metrics = summarize_classification(data.y_test, scores, threshold=threshold)
    return {
        "name": name,
        **metrics,
        "threshold": threshold,
        "fit_seconds": fit_seconds,
        "predict_ms_per_1k": predict_ms,
        "n_train": len(data.X_train),
        "n_test": len(data.X_test),
    }
