"""Isolates each test's predictions log: the real one is a process-lifetime singleton
(same lifecycle as the champion), so tests must not share it or one test's logged
traffic would leak into another's drift check."""

import pytest

from src.monitoring.predictions_log import PredictionsLog


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    log = PredictionsLog(tmp_path / "predictions.db")
    monkeypatch.setattr("src.api.app.load_predictions_log", lambda: log)
    return log
