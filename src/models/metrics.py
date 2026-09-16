"""Classification metrics for a credit scorecard.

PR-AUC leads: at a 4.75 % default rate the negatives dominate ROC, so average precision
is what reflects the review queue - of the applications flagged, how many default. Gini
is reported alongside because it is the industry convention this project already quotes.

src/models/evaluate.py stays as the scorecard-oriented decile view; this module is the
probability-oriented one, and reuses its gini conversion rather than re-deriving it.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.evaluate import gini_from_auc

METRICAS = ("pr_auc", "roc_auc", "gini", "f1", "precision", "recall", "brier")


def summarize_classification(
    y_true, y_score, threshold: float = 0.5, probabilistic: bool = True
) -> dict[str, float | dict[str, int]]:
    """Every metric this stage reports, from true labels and a continuous score.

    `probabilistic` says whether y_score is a calibrated probability. Brier is only
    defined for one: scoring a raw ranking against it would compare points to
    probabilities and report a number that means nothing.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=float)
    flagged = y_score >= threshold

    roc = float(roc_auc_score(y_true, y_score))
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[False, True]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": roc,
        "gini": gini_from_auc(roc),
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)) if probabilistic else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
