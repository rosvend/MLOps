"""The reference parquet must carry what drift detection needs: features, prediction,
decision, and the label - all from the training window, never the held-out one."""

from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (RAIZ / "models" / "reference_data.parquet").exists(),
    reason="run `make export-reference` first",
)


@pytest.fixture(scope="module")
def referencia():
    from src.config import load_config

    return pd.read_parquet(RAIZ / load_config().monitoring.reference_path)


def test_it_carries_the_features_the_prediction_and_the_label(referencia):
    from src.config import load_config

    spec = load_config().features
    for columna in spec.feature_view_columns:
        assert columna in referencia.columns
    assert "probability_default" in referencia.columns
    assert "review_flag" in referencia.columns
    assert "defaulted" in referencia.columns


def test_probabilities_are_probabilities(referencia):
    assert referencia["probability_default"].between(0, 1).all()


def test_the_flagged_share_matches_the_frozen_calibration(referencia):
    """15.7 % by construction: the threshold was fit on this exact window."""
    assert referencia["review_flag"].mean() == pytest.approx(0.157, abs=0.01)


def test_it_is_the_training_window_not_the_full_book(referencia):
    assert len(referencia) < 10_763
