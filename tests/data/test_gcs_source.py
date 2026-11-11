import pytest

from src.config import DataSourceConfig
from src.data.factory import build_source
from src.data.gcs_source import GcsDataSource


class _Blob:
    def __init__(self, store: dict, key: tuple[str, str]) -> None:
        self._store, self._key = store, key

    def download_as_bytes(self) -> bytes:
        return self._store[self._key]


class _Bucket:
    def __init__(self, store: dict, name: str) -> None:
        self._store, self._name = store, name

    def blob(self, path: str) -> _Blob:
        return _Blob(self._store, (self._name, path))


class FakeClient:
    def __init__(self, store: dict) -> None:
        self._store = store

    def bucket(self, name: str) -> _Bucket:
        return _Bucket(self._store, name)


def _client(content: bytes) -> FakeClient:
    return FakeClient({("my-bucket", "raw/loans.csv"): content})


def test_reads_the_object_as_a_dataframe():
    fuente = GcsDataSource("gs://my-bucket/raw/loans.csv", client=_client(b"a;b\n1;2\n"))

    assert fuente.read().to_dict("records") == [{"a": 1, "b": 2}]


def test_honours_the_separator_and_strips_the_byte_order_mark():
    contenido = "﻿a,b\n1,2\n".encode("utf-8")
    fuente = GcsDataSource("gs://my-bucket/raw/loans.csv", separator=",", client=_client(contenido))

    assert fuente.read().columns.tolist() == ["a", "b"]


@pytest.mark.parametrize("uri", ["my-bucket/raw/loans.csv", "gs://my-bucket", "s3://b/k.csv"])
def test_rejects_anything_that_is_not_a_bucket_and_object_uri(uri):
    with pytest.raises(ValueError):
        GcsDataSource(uri, client=_client(b""))


def test_the_factory_builds_it_from_config():
    config = DataSourceConfig(type="gcs", uri="gs://my-bucket/raw/loans.csv", separator=";")

    assert isinstance(build_source(config), GcsDataSource)
