"""The drift detector must find an injected shift and stay quiet without one."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftSummary, run_drift_report
from src.monitoring.spec import default_monitoring_spec


@pytest.fixture
def spec():
    return default_monitoring_spec()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _frames(rng, n=600):
    ref = pd.DataFrame(
        {
            "num_stable": rng.normal(0, 1, n),
            "num_shifted": rng.normal(0, 1, n),
            "cat_stable": rng.choice(["a", "b", "c"], n),
            "cat_shifted": rng.choice(["x", "y"], n),
        }
    )
    cur = ref.copy()
    cur["num_shifted"] = rng.normal(6, 1, n)
    cur["cat_shifted"] = rng.choice(["x", "y"], n, p=[0.98, 0.02])
    return ref, cur


def test_a_shifted_column_is_found(spec, rng):
    ref, cur = _frames(rng)

    resumen = run_drift_report(ref, cur, spec)

    assert isinstance(resumen, DriftSummary)
    assert "num_shifted" in resumen.drifted_features
    assert "cat_shifted" in resumen.drifted_features


def test_a_stable_column_is_not_flagged(spec, rng):
    ref, cur = _frames(rng)

    resumen = run_drift_report(ref, cur, spec)

    assert "num_stable" not in resumen.drifted_features
    assert "cat_stable" not in resumen.drifted_features


def test_drift_share_reflects_the_fraction_of_drifted_columns(spec, rng):
    ref, cur = _frames(rng)

    resumen = run_drift_report(ref, cur, spec)

    assert resumen.drift_share == pytest.approx(2 / 4)


def test_dataset_drift_detected_follows_the_configured_share_threshold(spec, rng):
    ref, cur = _frames(rng)

    resumen = run_drift_report(ref, cur, spec)

    # 2/4 = 0.5 drifted columns, above the default 0.30 threshold.
    assert resumen.dataset_drift_detected is True


def test_identical_data_shows_no_drift(spec, rng):
    ref, _ = _frames(rng)

    resumen = run_drift_report(ref, ref.copy(), spec)

    assert resumen.dataset_drift_detected is False
    assert resumen.drift_share == pytest.approx(0.0)
    assert resumen.drifted_features == []


def test_large_reference_switches_to_the_large_sample_stattest(spec):
    """Confirms the numeric method actually changes at the configured cutoff, not just
    that the code compiles - the cutoff is read from the metric config it produced."""
    rng = np.random.default_rng(1)
    grande = spec.numeric_sample_size_cutoff + 200
    ref = pd.DataFrame({"num": rng.normal(0, 1, grande)})
    cur = pd.DataFrame({"num": rng.normal(0, 1, grande)})

    resumen = run_drift_report(ref, cur, spec)

    assert resumen.methods_used["num"] == spec.numeric_large_sample_stattest


def test_small_reference_uses_the_small_sample_stattest(spec):
    rng = np.random.default_rng(1)
    pequeno = spec.numeric_sample_size_cutoff - 200
    ref = pd.DataFrame({"num": rng.normal(0, 1, pequeno)})
    cur = pd.DataFrame({"num": rng.normal(0, 1, pequeno)})

    resumen = run_drift_report(ref, cur, spec)

    assert resumen.methods_used["num"] == spec.numeric_small_sample_stattest


def test_only_shared_columns_are_compared(spec, rng):
    """A column present in only one frame (e.g. the label, or an id) must not crash the
    report or silently appear in the drift results."""
    ref, cur = _frames(rng)
    ref["solo_en_referencia"] = 1
    cur["solo_en_actual"] = 1

    resumen = run_drift_report(ref, cur, spec)

    assert "solo_en_referencia" not in resumen.drifted_features
    assert "solo_en_actual" not in resumen.drifted_features


def test_saving_the_report_writes_both_files(spec, rng, tmp_path):
    ref, cur = _frames(rng)
    resumen = run_drift_report(ref, cur, spec)

    html = tmp_path / "r.html"
    json_path = tmp_path / "r.json"
    resumen.save(html, json_path)

    assert html.exists() and html.stat().st_size > 0
    assert json_path.exists()
    import json

    datos = json.loads(json_path.read_text())
    assert datos["drift_share"] == pytest.approx(resumen.drift_share)
