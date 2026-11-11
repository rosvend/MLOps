"""Score the TEST window with the AutoML model and measure it like the local champion."""

import json

import numpy as np
import pandas as pd

from src.automl.spec import AutomlSpec
from src.models.metrics import summarize_classification


def batch_input(df: pd.DataFrame, spec: AutomlSpec) -> tuple[pd.DataFrame, pd.Series]:
    """TEST rows with a row key and no label/split, plus the labels kept locally."""
    test = df[df[spec.split_column] == spec.test_value].reset_index(drop=True)
    y = test[spec.label_column]
    feats = test.drop(columns=[spec.label_column, spec.split_column])
    feats.insert(0, spec.batch.key_column, range(len(feats)))
    return feats, y


def parse_predictions(lines, spec: AutomlSpec) -> pd.Series:
    """P(default) per row key from the batch-prediction jsonl output."""
    scores = {}
    for line in lines:
        rec = json.loads(line)
        by_class = dict(zip(rec["prediction"]["classes"], rec["prediction"]["scores"]))
        scores[int(rec["instance"][spec.batch.key_column])] = float(by_class["1"])
    return pd.Series(scores).sort_index()


def evaluate_scores(y_true, y_score, flagged_share: float) -> dict:
    """Same operating point as src/pipelines/train.py: flag the top `flagged_share` by score."""
    umbral = float(np.quantile(y_score, 1 - flagged_share))
    return summarize_classification(y_true, y_score, threshold=umbral, probabilistic=True)


def fill_unscored(scores: pd.Series, n_rows: int) -> pd.Series:
    """Rows AutoML refused to score are never flagged: score 0 over the whole window."""
    return scores.reindex(range(n_rows), fill_value=0.0)
