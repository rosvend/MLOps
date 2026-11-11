import json

import numpy as np
import pandas as pd

from src.automl.evaluate import batch_input, evaluate_scores, parse_predictions
from src.automl.spec import default_automl_spec

SPEC = default_automl_spec()


def _frame():
    return pd.DataFrame(
        {"x": [1, 2, 3, 4], "defaulted": [0, 1, 0, 1], "split": ["TRAIN", "TRAIN", "TEST", "TEST"]}
    )


def test_batch_input_keeps_only_test_rows_without_label_or_split():
    feats, y = batch_input(_frame(), SPEC)
    assert list(feats.columns) == ["row_id", "x"]
    assert feats["x"].tolist() == [3, 4]
    assert y.tolist() == [0, 1]
    assert feats["row_id"].tolist() == [0, 1]


def _line(row_id, p1):
    return json.dumps(
        {"instance": {"row_id": str(row_id)}, "prediction": {"classes": ["0", "1"], "scores": [1 - p1, p1]}}
    )


def test_parse_predictions_returns_default_probability_by_row_id_in_any_order():
    scores = parse_predictions([_line(1, 0.8), _line(0, 0.1)], SPEC)
    assert scores.index.tolist() == [0, 1]
    assert np.allclose(scores.to_numpy(), [0.1, 0.8])


def test_parse_predictions_reads_the_class_one_score_even_if_classes_are_reversed():
    line = json.dumps({"instance": {"row_id": "0"}, "prediction": {"classes": ["1", "0"], "scores": [0.7, 0.3]}})
    assert parse_predictions([line], SPEC).iloc[0] == 0.7


def test_evaluate_scores_flags_the_configured_share_of_rows():
    y = np.array([0] * 8 + [1] * 2)
    s = np.linspace(0, 1, 10)
    m = evaluate_scores(y, s, flagged_share=0.2)
    assert m["pr_auc"] == 1.0
    assert m["recall"] == 1.0 and m["precision"] == 1.0


def test_unscored_rows_get_the_lowest_score_on_the_full_window():
    from src.automl.evaluate import fill_unscored

    scores = pd.Series({0: 0.4, 2: 0.9})
    full = fill_unscored(scores, n_rows=4)
    assert full.tolist() == [0.4, 0.0, 0.9, 0.0]
