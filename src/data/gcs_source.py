from io import BytesIO

import pandas as pd


class GcsDataSource:
    """A CSV object in Cloud Storage, authenticated through Application Default Credentials."""

    def __init__(self, uri: str, separator: str = ";", client=None) -> None:
        bucket, _, objeto = uri.removeprefix("gs://").partition("/")
        if not uri.startswith("gs://") or not bucket or not objeto:
            raise ValueError(f"se esperaba gs://bucket/objeto, no {uri!r}")
        self._bucket, self._object = bucket, objeto
        self._separator = separator
        self._client = client

    def read(self) -> pd.DataFrame:
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        datos = self._client.bucket(self._bucket).blob(self._object).download_as_bytes()
        return pd.read_csv(BytesIO(datos), sep=self._separator, encoding="utf-8-sig")
