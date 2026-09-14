import pandas as pd
from scipy import stats

DECILES = 10


def auc(scores: pd.Series, defaulted: pd.Series) -> float:
    """Probability that a defaulted loan scores above a performing one."""
    en_mora, al_dia = scores[defaulted], scores[~defaulted]
    if en_mora.empty or al_dia.empty:
        return float("nan")
    estadistico = stats.mannwhitneyu(en_mora, al_dia, alternative="two-sided").statistic
    return float(estadistico / (len(en_mora) * len(al_dia)))


def gini(scores: pd.Series, defaulted: pd.Series) -> float:
    return 2 * auc(scores, defaulted) - 1


def decile_rates(scores: pd.Series, defaulted: pd.Series) -> pd.Series:
    """Ties are broken by position, the usual convention for banded scorecards."""
    deciles = pd.qcut(scores.rank(method="first"), DECILES, labels=False)
    return defaulted.groupby(deciles).mean()


def precision_recall(scores: pd.Series, defaulted: pd.Series, threshold: int) -> dict[str, float]:
    flagged = scores >= threshold
    aciertos = int((flagged & defaulted).sum())
    return {
        "precision": aciertos / int(flagged.sum()) if flagged.any() else 0.0,
        "recall": aciertos / int(defaulted.sum()) if defaulted.any() else 0.0,
        "flagged_share": float(flagged.mean()),
    }


def evaluate(scores: pd.Series, defaulted: pd.Series, threshold: int) -> dict[str, float]:
    tasas = decile_rates(scores, defaulted)
    peor, mejor = float(tasas.iloc[-1]), float(tasas.iloc[0])
    return {
        "auc": auc(scores, defaulted),
        "gini": gini(scores, defaulted),
        "top_decile_rate": peor,
        "bottom_decile_rate": mejor,
        "decile_lift": peor / mejor if mejor else float("inf"),
        **precision_recall(scores, defaulted, threshold),
    }
