"""Turn the exported split file into train / test sets. Pure: no I/O."""

from dataclasses import dataclass

import pandas as pd

from src.models.training_spec import GcpDatasetSpec


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def _bools_to_float(features: pd.DataFrame) -> pd.DataFrame:
    """A CSV stores a boolean with gaps as text; the model pipeline wants a number."""
    out = features.copy()
    for col in out.select_dtypes(include="object"):
        values = out[col].dropna()
        if len(values) and values.map(lambda v: isinstance(v, bool)).all():
            out[col] = out[col].astype(float)
    return out


def split_frame(frame: pd.DataFrame, ds: GcpDatasetSpec) -> SplitData:
    """TRAIN + VALIDATE train the model; TEST is only ever scored once, at the end."""
    known = {ds.train_value, ds.validation_value, ds.test_value}
    unknown = set(frame[ds.split_column].unique()) - known
    if unknown:
        raise ValueError(f"valores desconocidos en {ds.split_column}: {sorted(unknown)}")
    is_test = frame[ds.split_column] == ds.test_value
    if not is_test.any() or is_test.all():
        raise ValueError(f"se necesitan filas {ds.test_value} y de entrenamiento")

    features = _bools_to_float(frame.drop(columns=[ds.label_column, ds.split_column]))
    label = frame[ds.label_column].astype(int)
    train, test = ~is_test, is_test
    return SplitData(
        X_train=features[train].reset_index(drop=True),
        y_train=label[train].reset_index(drop=True),
        X_test=features[test].reset_index(drop=True),
        y_test=label[test].reset_index(drop=True),
    )
