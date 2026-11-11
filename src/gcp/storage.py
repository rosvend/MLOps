"""Read and write Cloud Storage objects by gs:// URI. The client is injectable for tests."""

import json
from io import BytesIO

import pandas as pd


def _blob(uri: str, client):
    if not uri.startswith("gs://"):
        raise ValueError(f"se esperaba gs://bucket/objeto, no {uri!r}")
    bucket, _, obj = uri.removeprefix("gs://").partition("/")
    if client is None:
        from google.cloud import storage

        client = storage.Client()
    return client.bucket(bucket).blob(obj)


def read_csv(uri: str, client=None) -> pd.DataFrame:
    return pd.read_csv(BytesIO(_blob(uri, client).download_as_bytes()))


def read_json(uri: str, client=None) -> dict:
    return json.loads(_blob(uri, client).download_as_bytes())


def write_json(uri: str, obj: dict, client=None) -> None:
    _blob(uri, client).upload_from_string(json.dumps(obj, indent=2), content_type="application/json")
