from pathlib import Path

import pandas as pd


class CsvDataSource:
    def __init__(self, path: str | Path, separator: str = ";") -> None:
        self._path = Path(path)
        self._separator = separator

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self._path, sep=self._separator, encoding="utf-8-sig")
