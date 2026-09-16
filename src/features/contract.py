"""Which prepared columns a model is allowed to see.

Keeping this as a list rather than a convention means adding a leaky feature is a
deliberate edit to a reviewed file, not an accident in a notebook.

saldo_mora and tiene_mora_bureau were withheld while it was unconfirmed whether the
bureau balance is observed at origination. The business confirmed an origination-time
pull on 2026-09-16, so they are features now.
"""

import pandas as pd

TARGET = "Pago_atiempo"

PROHIBIDAS = frozenset(
    {
        "puntaje",  # leakage: 87% share the maximum value and none of them defaulted
        "mes_prestamo",  # vintage censoring control, not a predictor
    }
)


def feature_names(columns: pd.Index | list[str]) -> list[str]:
    return [c for c in columns if c != TARGET and c not in PROHIBIDAS]


def features(df: pd.DataFrame) -> pd.DataFrame:
    """The only view a model should ever be fitted or scored on."""
    return df[feature_names(df.columns)].copy()
