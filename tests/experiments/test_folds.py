import numpy as np
import pandas as pd

from experiments.signal.folds import time_folds


def test_time_folds_train_strictly_earlier_than_validation():
    v = pd.Series(["2024-01"] * 3 + ["2024-02"] * 3 + ["2024-03"] * 3 + ["2024-04"] * 3)
    folds = time_folds(v, n_val=2)
    assert len(folds) == 2
    for tr, va in folds:
        assert v.iloc[tr].max() < v.iloc[va].min()
        assert not set(tr) & set(va)
    assert set(v.iloc[folds[0][1]]) == {"2024-03"}
