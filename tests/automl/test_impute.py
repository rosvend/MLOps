import numpy as np
import pandas as pd

from src.automl.impute import fit_medians, impute


def _frame():
    return pd.DataFrame(
        {
            "n": [1.0, 3.0, np.nan, 1000.0, np.nan],
            "c": ["a", None, "b", "a", None],
            "defaulted": [0, 1, 0, 1, 0],
            "split": ["TRAIN", "TRAIN", "TRAIN", "VALIDATE", "TEST"],
        }
    )


def test_medians_use_train_rows_only():
    # VALIDATE holds 1000.0; a leaky median over all rows would be 3.0 not 2.0
    assert fit_medians(_frame(), ["n"], "split", "TRAIN") == {"n": 2.0}


def test_impute_fills_every_cell_with_train_statistics():
    out = impute(_frame(), numeric=["n"], categorical=["c"], split_column="split",
                 train_value="TRAIN", missing_label="missing")
    assert not out.isna().any().any()
    assert out["n"].tolist() == [1.0, 3.0, 2.0, 1000.0, 2.0]
    assert out["c"].tolist() == ["a", "missing", "b", "a", "missing"]


def test_impute_keeps_label_split_and_does_not_mutate_input():
    f = _frame()
    out = impute(f, ["n"], ["c"], "split", "TRAIN", "missing")
    assert out[["defaulted", "split"]].equals(f[["defaulted", "split"]])
    assert f["n"].isna().sum() == 2


def test_changing_test_rows_does_not_change_the_fitted_statistics():
    f = _frame()
    g = f.copy()
    g.loc[g.split != "TRAIN", "n"] = 99999.0
    assert fit_medians(f, ["n"], "split", "TRAIN") == fit_medians(g, ["n"], "split", "TRAIN")
