import pandas as pd
import pytest

from src.gcp.data import split_frame
from src.models.training_spec import GcpDatasetSpec

DS = GcpDatasetSpec(
    split_column="split", label_column="defaulted", train_value="TRAIN",
    validation_value="VALIDATE", test_value="TEST", validation_fraction=0.15, output_path="x.csv",
)


def _frame(splits):
    return pd.DataFrame(
        {"a": range(len(splits)), "defaulted": [0, 1] * (len(splits) // 2), "split": splits}
    )


def test_train_is_train_plus_validate_and_test_is_test():
    data = split_frame(_frame(["TRAIN"] * 4 + ["VALIDATE"] * 2 + ["TEST"] * 2), DS)
    assert len(data.X_train) == 6
    assert len(data.X_test) == 2
    assert list(data.X_test["a"]) == [6, 7]


def test_label_and_split_never_reach_the_features():
    data = split_frame(_frame(["TRAIN"] * 4 + ["TEST"] * 2), DS)
    for X in (data.X_train, data.X_test):
        assert list(X.columns) == ["a"]


def test_missing_test_rows_is_an_error():
    with pytest.raises(ValueError, match="TEST"):
        split_frame(_frame(["TRAIN"] * 4), DS)


def test_unknown_split_label_is_an_error():
    with pytest.raises(ValueError, match="split"):
        split_frame(_frame(["TRAIN"] * 3 + ["TEST"] * 2 + ["OTHER"]), DS)


def test_boolean_columns_with_gaps_become_numeric_after_csv_round_trip():
    f = _frame(["TRAIN"] * 4 + ["TEST"] * 2)
    f["flag"] = pd.Series([True, False, None, True, False, True], dtype=object)
    f["txt"] = ["x"] * 6
    data = split_frame(f, DS)
    assert pd.api.types.is_float_dtype(data.X_train["flag"])
    assert data.X_train["flag"].isna().sum() == 1
    assert data.X_train["txt"].dtype == object
