"""Leakage-free imputation of the split file: statistics come from TRAIN rows only."""

import pandas as pd


def fit_medians(df: pd.DataFrame, numeric: list[str], split_column: str, train_value: str) -> dict[str, float]:
    train = df[df[split_column] == train_value]
    return {c: float(train[c].median()) for c in numeric}


def impute(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    split_column: str,
    train_value: str,
    missing_label: str,
) -> pd.DataFrame:
    """Median (fit on TRAIN) for numeric columns, a constant label for categorical ones."""
    medians = fit_medians(df, numeric, split_column, train_value)
    out = df.copy()
    for c, m in medians.items():
        out[c] = out[c].fillna(m)
    for c in categorical:
        out[c] = out[c].astype(object).where(out[c].notna(), missing_label)
    return out
