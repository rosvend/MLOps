import pandas as pd

from src.models import rules
from src.models.rules import REQUIRED_COLUMNS, Record
from src.models.scorecard import Scorecard, default_scorecard

RULES = (
    rules.bureau_score_band,
    rules.inquiry_band,
    rules.age_band,
    rules.young_independent,
    rules.income_gap_quartile,
    rules.decreasing_income_trend,
    rules.high_amount_long_term,
    rules.missing_bureau_income,
)


def score(record: Record, scorecard: Scorecard | None = None) -> int:
    """Higher score = higher risk."""
    scorecard = scorecard or default_scorecard()
    return sum(rule(record, scorecard) for rule in RULES)


def explain(record: Record, scorecard: Scorecard | None = None) -> dict[str, int]:
    """Every point traced back to the rule that charged it."""
    scorecard = scorecard or default_scorecard()
    return {rule.__name__: rule(record, scorecard) for rule in RULES}


def predict(record: Record, threshold: int, scorecard: Scorecard | None = None) -> bool:
    return score(record, scorecard) >= threshold


def score_frame(df: pd.DataFrame, scorecard: Scorecard | None = None) -> pd.Series:
    """Higher score = higher risk. Raises if a rule's input column is absent."""
    faltantes = REQUIRED_COLUMNS - set(df.columns)
    if faltantes:
        raise KeyError(f"Faltan columnas que las reglas puntúan: {', '.join(sorted(faltantes))}")
    scorecard = scorecard or default_scorecard()
    return df.apply(lambda fila: score(fila, scorecard), axis=1)
