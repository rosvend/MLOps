"""SQLite in WAL mode: safe under concurrent writes, and what score() actually wrote
must come back out with the same values and dtypes."""

import threading
from pathlib import Path

import pandas as pd
import pytest

from src.monitoring.predictions_log import PredictionsLog


@pytest.fixture
def log(tmp_path):
    return PredictionsLog(tmp_path / "predictions.db")


def test_a_logged_row_round_trips(log):
    log.append(
        [{"application_id": "APP-1", "probability_default": 0.42, "review_flag": True,
          "capital_prestado": 1000000, "tipo_credito": "4"}]
    )

    df = log.read_all()

    assert len(df) == 1
    assert df.iloc[0]["probability_default"] == pytest.approx(0.42)
    assert bool(df.iloc[0]["review_flag"]) is True
    assert df.iloc[0]["tipo_credito"] == "4"


def test_appends_accumulate(log):
    log.append([{"application_id": "APP-1", "probability_default": 0.1, "review_flag": False}])
    log.append([{"application_id": "APP-2", "probability_default": 0.9, "review_flag": True}])

    assert len(log.read_all()) == 2


def test_wal_mode_is_active(log):
    log.append([{"application_id": "APP-1", "probability_default": 0.1, "review_flag": False}])

    assert log.journal_mode() == "wal"


def test_concurrent_appends_do_not_lose_rows(log):
    def escribir(i):
        log.append(
            [{"application_id": f"APP-{i}", "probability_default": i / 100, "review_flag": False}]
        )

    hilos = [threading.Thread(target=escribir, args=(i,)) for i in range(30)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()

    assert len(log.read_all()) == 30


def test_reading_an_empty_log_gives_an_empty_frame(log):
    df = log.read_all()

    assert len(df) == 0
    assert isinstance(df, pd.DataFrame)


def test_the_db_file_is_created_under_the_configured_path(tmp_path):
    destino = tmp_path / "nested" / "predictions.db"
    log = PredictionsLog(destino)

    log.append([{"application_id": "APP-1", "probability_default": 0.5, "review_flag": False}])

    assert destino.exists()
