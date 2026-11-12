"""The BigQuery log behaves like the SQLite one and is picked by config; the client is a
fake, so nothing here touches the network."""

import json

import pandas as pd
import pytest

from src.monitoring import predictions_log as pl
from src.monitoring.bigquery_log import BigQueryPredictionsLog
from src.monitoring.predictions_log import PredictionsLog, build_predictions_log
from src.monitoring.spec import PredictionsLogSpec, default_monitoring_spec


class FakeClient:
    def __init__(self, errors=None):
        self.rows, self.errors, self.queries = [], errors or [], []

    def insert_rows_json(self, table, rows):
        self.rows.extend(rows)
        return self.errors

    def query(self, sql):
        self.queries.append(sql)
        stored = [(pd.Timestamp(r["scored_at"]).to_pydatetime(), r["payload"]) for r in self.rows]
        return type("Job", (), {"result": lambda self_: iter(stored)})()


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def log(client):
    return BigQueryPredictionsLog(client, "proj.mlops_monitoring.predictions")


def test_append_sends_one_json_row_per_record(log, client):
    log.append([{"application_id": "A1", "review_flag": True}, {"application_id": "A2", "review_flag": False}])

    assert len(client.rows) == 2
    assert json.loads(client.rows[0]["payload"]) == {"application_id": "A1", "review_flag": True}
    assert client.rows[0]["scored_at"] == client.rows[1]["scored_at"]


def test_read_all_has_the_same_shape_as_sqlite(log, tmp_path):
    registros = [{"application_id": "A1", "probability_default": 0.42, "review_flag": True}]
    sqlite = PredictionsLog(tmp_path / "p.db")
    sqlite.append(registros)
    log.append(registros)

    df = log.read_all()

    assert list(df.columns) == list(sqlite.read_all().columns)
    assert df.iloc[0]["probability_default"] == pytest.approx(0.42)
    assert bool(df.iloc[0]["review_flag"]) is True


def test_empty_table_gives_an_empty_frame(log):
    df = log.read_all()

    assert isinstance(df, pd.DataFrame) and len(df) == 0


def test_payload_returned_as_dict_is_accepted(client, log):
    client.rows.append({"scored_at": "2026-01-01T00:00:00+00:00", "payload": {"application_id": "A1"}})

    assert log.read_all().iloc[0]["application_id"] == "A1"


def test_insert_errors_raise(log, client):
    client.errors = [{"index": 0, "errors": ["boom"]}]

    with pytest.raises(RuntimeError):
        log.append([{"application_id": "A1"}])


def test_factory_picks_sqlite(tmp_path):
    spec = PredictionsLogSpec(type="sqlite", sqlite_path=str(tmp_path / "p.db"),
                              bigquery_dataset="d", bigquery_table="t")

    assert isinstance(build_predictions_log(spec, root=tmp_path), PredictionsLog)


def test_factory_picks_bigquery(monkeypatch, client):
    spec = PredictionsLogSpec(type="bigquery", sqlite_path="x.db",
                              bigquery_dataset="mlops_monitoring", bigquery_table="predictions")
    client.project = "proj"
    monkeypatch.setattr(pl, "_bigquery_client", lambda: client)

    log = build_predictions_log(spec)

    assert isinstance(log, BigQueryPredictionsLog)
    log.read_all()
    assert "proj.mlops_monitoring.predictions" in client.queries[0]


def test_default_config_is_sqlite(monkeypatch):
    monkeypatch.delenv("PREDICTIONS_LOG_TYPE", raising=False)
    default_monitoring_spec.cache_clear()

    assert default_monitoring_spec().predictions_log.type == "sqlite"


def test_env_var_switches_to_bigquery(monkeypatch):
    monkeypatch.setenv("PREDICTIONS_LOG_TYPE", "bigquery")
    default_monitoring_spec.cache_clear()
    try:
        assert default_monitoring_spec().predictions_log.type == "bigquery"
    finally:
        monkeypatch.delenv("PREDICTIONS_LOG_TYPE")
        default_monitoring_spec.cache_clear()


def test_missing_values_are_sent_as_json_null_not_nan(log, client):
    """BigQuery's JSON column rejects the literal NaN, which rejected the whole insert."""
    import numpy as np

    log.append([{"application_id": "A1", "salario_cliente": float("nan"), "dti": np.float64("nan"),
                 "huella_consulta": float("inf"), "tendencia_ingresos": None, "edad_cliente": 30.0}])

    payload = client.rows[0]["payload"]
    assert "NaN" not in payload and "Infinity" not in payload
    stored = json.loads(payload)
    assert stored["salario_cliente"] is None and stored["dti"] is None and stored["huella_consulta"] is None
    assert stored["edad_cliente"] == 30.0


def test_the_sqlite_log_also_stores_missing_values_as_null(tmp_path):
    local = PredictionsLog(tmp_path / "p.db")
    local.append([{"application_id": "A1", "salario_cliente": float("nan")}])

    raw = local._conn.execute("SELECT payload FROM predictions").fetchone()[0]
    assert "NaN" not in raw
