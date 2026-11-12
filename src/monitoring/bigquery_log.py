"""BigQuery-backed predictions log: durable across Cloud Run instances and restarts,
unlike a SQLite file on an ephemeral disk. Same behaviour as the SQLite log."""

import json

import pandas as pd

from src.monitoring.predictions_log import _jsonable


class BigQueryPredictionsLog:
    """`client` is a google.cloud.bigquery.Client (injected); `table_id` is project.dataset.table."""

    def __init__(self, client, table_id: str):
        self._client = client
        self._table_id = table_id

    def append(self, records: list[dict]) -> None:
        ahora = pd.Timestamp.now(tz="UTC").isoformat()
        filas = [{"scored_at": ahora, "payload": json.dumps(_jsonable(r), default=str, allow_nan=False)} for r in records]
        errores = self._client.insert_rows_json(self._table_id, filas)
        if errores:
            raise RuntimeError(f"BigQuery rechazó filas de {self._table_id}: {errores}")

    def read_all(self) -> pd.DataFrame:
        filas = self._client.query(
            f"SELECT scored_at, payload FROM `{self._table_id}` ORDER BY scored_at"
        ).result()
        registros = [
            {"scored_at": pd.Timestamp(ts).isoformat(), **(json.loads(p) if isinstance(p, str) else p)}
            for ts, p in filas
        ]
        return pd.DataFrame(registros)
