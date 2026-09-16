"""Local, append-only log of what the API has scored - the "current data" drift compares
the reference against.

SQLite in WAL mode: safe under concurrent requests without running a database server,
and the one file the constraint's "lightweight, no heavy DB" explicitly allows.
"""

import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd

_TABLE = "predictions"


class PredictionsLog:
    """One connection per instance, guarded by a lock: sqlite3 connections are not
    thread-safe to share, and a single API process holds exactly one of these."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} "
            "(scored_at TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._conn.commit()

    def journal_mode(self) -> str:
        return self._conn.execute("PRAGMA journal_mode").fetchone()[0]

    def append(self, records: list[dict]) -> None:
        """Each record's values are stored as JSON - the columns a request carries can
        vary (optional bureau fields), and a fixed schema would force a null for every
        column any single request omitted."""
        ahora = pd.Timestamp.now(tz="UTC").isoformat()
        filas = [(ahora, json.dumps(_jsonable(r), default=str)) for r in records]
        with self._lock:
            self._conn.executemany(f"INSERT INTO {_TABLE} (scored_at, payload) VALUES (?, ?)", filas)
            self._conn.commit()

    def read_all(self) -> pd.DataFrame:
        with self._lock:
            filas = self._conn.execute(f"SELECT scored_at, payload FROM {_TABLE}").fetchall()
        if not filas:
            return pd.DataFrame()
        registros = [{"scored_at": ts, **json.loads(payload)} for ts, payload in filas]
        return pd.DataFrame(registros)

    def count(self) -> int:
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]


def _jsonable(record: dict) -> dict:
    return {k: (bool(v) if isinstance(v, bool) else v) for k, v in record.items()}
