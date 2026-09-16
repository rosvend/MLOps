"""Portfolio metrics. `defaulted=True` means the loan defaulted, not that it was paid."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

DECILES = 10
# Below this, a "decile" is a handful of loans and its rate is noise, not a measurement.
DECILE_MINIMO = 200

_log = logging.getLogger(__name__)


def auc(scores: pd.Series, defaulted: pd.Series) -> float:
    """Probability that a defaulted loan scores above a performing one; nan on one class."""
    en_mora, al_dia = scores[defaulted], scores[~defaulted]
    if en_mora.empty or al_dia.empty:
        _log.warning(
            "AUC no calculable: el lote trae una sola clase (%d en mora, %d al día)",
            len(en_mora),
            len(al_dia),
        )
        return float("nan")
    estadistico = stats.mannwhitneyu(en_mora, al_dia).statistic
    return float(estadistico / (len(en_mora) * len(al_dia)))


def gini(scores: pd.Series, defaulted: pd.Series) -> float:
    return 2 * auc(scores, defaulted) - 1


def decile_rates(scores: pd.Series, defaulted: pd.Series) -> pd.Series:
    """Ties are broken by position, the usual convention for banded scorecards."""
    if len(scores) < 2:
        return pd.Series([defaulted.mean()] if len(scores) else [], dtype=float)
    bins = min(DECILES, len(scores))
    deciles = pd.qcut(scores.rank(method="first"), bins, labels=False)
    return defaulted.groupby(deciles).mean()


def precision_recall(scores: pd.Series, defaulted: pd.Series, threshold: int) -> dict[str, float]:
    flagged = scores >= threshold
    aciertos = int((flagged & defaulted).sum())
    return {
        "precision": aciertos / int(flagged.sum()) if flagged.any() else 0.0,
        "recall": aciertos / int(defaulted.sum()) if defaulted.any() else 0.0,
        "flagged_share": float(flagged.mean()),
    }


def _decile_metrics(scores: pd.Series, defaulted: pd.Series) -> dict[str, float]:
    nan = float("nan")
    if len(scores) < DECILE_MINIMO:
        _log.warning(
            "Métricas por decil omitidas: %d créditos, mínimo %d", len(scores), DECILE_MINIMO
        )
        return {"top_decile_rate": nan, "bottom_decile_rate": nan, "decile_lift": nan}
    tasas = decile_rates(scores, defaulted)
    peor, mejor = float(tasas.iloc[-1]), float(tasas.iloc[0])
    if not mejor:
        # A clean bottom decile makes the ratio undefined; inf would read as a real number.
        _log.warning("Lift por decil no calculable: el decil más seguro no trae incumplimientos")
        return {"top_decile_rate": peor, "bottom_decile_rate": mejor, "decile_lift": nan}
    return {"top_decile_rate": peor, "bottom_decile_rate": mejor, "decile_lift": peor / mejor}


def evaluate(scores: pd.Series, defaulted: pd.Series, threshold: int) -> dict[str, float]:
    return {
        "auc": auc(scores, defaulted),
        "gini": gini(scores, defaulted),
        **_decile_metrics(scores, defaulted),
        **precision_recall(scores, defaulted, threshold),
    }
