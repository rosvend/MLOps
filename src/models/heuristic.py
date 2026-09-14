import pandas as pd

from src.models import rules
from src.models.rules import Record

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


def score(record: Record) -> int:
    return sum(rule(record) for rule in RULES)


def explain(record: Record) -> dict[str, int]:
    """Every point traced back to the rule that charged it."""
    return {rule.__name__: rule(record) for rule in RULES}


def predict(record: Record, threshold: int) -> bool:
    return score(record) >= threshold


def score_frame(df: pd.DataFrame) -> pd.Series:
    return df.apply(score, axis=1)
