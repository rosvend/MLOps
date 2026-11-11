import pandas as pd
import pytest

from src.gcp.storage import read_csv, read_json, write_json


class FakeBlob:
    def __init__(self, store, key):
        self.store, self.key = store, key

    def download_as_bytes(self):
        return self.store[self.key]

    def upload_from_string(self, data, content_type=None):
        self.store[self.key] = data.encode() if isinstance(data, str) else data


class FakeClient:
    def __init__(self):
        self.store = {}

    def bucket(self, name):
        client = self

        class B:
            def blob(self, obj):
                return FakeBlob(client.store, f"{name}/{obj}")

        return B()


def test_json_round_trip():
    c = FakeClient()
    write_json("gs://b/runs/x/m.json", {"a": 1}, client=c)
    assert read_json("gs://b/runs/x/m.json", client=c) == {"a": 1}


def test_read_csv():
    c = FakeClient()
    c.store["b/d.csv"] = b"a,b\n1,2\n"
    assert read_csv("gs://b/d.csv", client=c).equals(pd.DataFrame({"a": [1], "b": [2]}))


def test_rejects_non_gcs_uri():
    with pytest.raises(ValueError):
        read_json("/tmp/x.json", client=FakeClient())
