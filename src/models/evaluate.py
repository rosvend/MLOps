"""Portfolio metrics. `defaulted=True` means the loan defaulted, not that it was paid."""

import logging

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


def gini_from_auc(valor: float) -> float:
    return 2 * valor - 1


def gini(scores: pd.Series, defaulted: pd.Series) -> float:
    return gini_from_auc(auc(scores, defaulted))


def decile_rates(scores: pd.Series, defaulted: pd.Series, deciles: int = DECILES) -> pd.Series:
    """Equal scores share a band: the score is the decision, not the row's position.

    A 21-point integer scale over thousands of loans means decile edges fall inside huge
    tied groups, so splitting them by position would make the metric depend on file order.
    Ties collapse instead, which can yield fewer than `deciles` bands.
    """
    if len(scores) < 2:
        return pd.Series([defaulted.mean()] if len(scores) else [], dtype=float)
    bandas = pd.qcut(
        scores.rank(method="average"), min(deciles, len(scores)), labels=False, duplicates="drop"
    )
    if bandas.isna().all():
        # Every loan scored the same: one band, not none.
        return pd.Series([defaulted.mean()], dtype=float)
    return defaulted.groupby(bandas).mean()


def precision_recall(scores: pd.Series, defaulted: pd.Series, threshold: int) -> dict[str, float]:
    flagged = scores >= threshold
    aciertos = int((flagged & defaulted).sum())
    return {
        "precision": aciertos / int(flagged.sum()) if flagged.any() else 0.0,
        "recall": aciertos / int(defaulted.sum()) if defaulted.any() else 0.0,
        "flagged_share": float(flagged.mean()),
    }


def _decile_metrics(
    scores: pd.Series,
    defaulted: pd.Series,
    deciles: int = DECILES,
    decile_minimo: int = DECILE_MINIMO,
) -> dict[str, float]:
    nan = float("nan")
    if len(scores) < decile_minimo:
        _log.warning(
            "Métricas por decil omitidas: %d créditos, mínimo %d", len(scores), decile_minimo
        )
        return {"top_decile_rate": nan, "bottom_decile_rate": nan, "decile_lift": nan}
    tasas = decile_rates(scores, defaulted, deciles)
    peor, mejor = float(tasas.iloc[-1]), float(tasas.iloc[0])
    if not mejor:
        # A clean bottom band makes the ratio undefined; inf would read as a real number.
        _log.warning("Lift por decil no calculable: la banda más segura no trae incumplimientos")
        return {"top_decile_rate": peor, "bottom_decile_rate": mejor, "decile_lift": nan}
    return {"top_decile_rate": peor, "bottom_decile_rate": mejor, "decile_lift": peor / mejor}


def evaluate(
    scores: pd.Series,
    defaulted: pd.Series,
    threshold: int,
    deciles: int = DECILES,
    decile_minimo: int = DECILE_MINIMO,
) -> dict[str, float]:
    valor_auc = auc(scores, defaulted)
    return {
        "auc": valor_auc,
        "gini": gini_from_auc(valor_auc),
        **_decile_metrics(scores, defaulted, deciles, decile_minimo),
        **precision_recall(scores, defaulted, threshold),
    }
