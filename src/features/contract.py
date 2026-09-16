"""Which prepared columns a model is allowed to see.

Four roles are reserved and never reach a model: the entity key (a join key, not a
signal), the event timestamp (the vintage itself), the target, and the withheld list.
Everything else is a feature. The roles live in config/features/default.yaml, so adding
a leaky column is a deliberate edit to a reviewed file rather than an accident.
"""

import pandas as pd

from src.features.spec import FeatureSpec, default_spec


def feature_names(columns: pd.Index | list[str], spec: FeatureSpec | None = None) -> list[str]:
    spec = spec or default_spec()
    reservadas = spec.no_son_features
    return [c for c in columns if c not in reservadas]


def features(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """The only view a model should ever be fitted or scored on."""
    return df[feature_names(df.columns, spec)].copy()


def entity_frame(df: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Join key and event timestamp: what a feature store needs for point-in-time joins."""
    spec = spec or default_spec()
    return df[[spec.entity_key, spec.event_timestamp]].copy()
