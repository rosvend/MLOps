import json

import pytest

from src.gcp.register import label_value, promote, register
from src.gcp.registry_spec import default_registry_spec

RUN = "gs://b/pipeline-runs/run-20260101-000000"
META = {
    "model": "logistic",
    "git_sha": "ABC123.dirty",
    "metrics": {"test": {"pr_auc": 0.13, "gini": 0.33, "roc_auc": 0.66}},
}


class FakeStorage:
    def __init__(self):
        self.reads = []

    def bucket(self, name):
        outer = self

        class B:
            def blob(self, obj):
                outer.reads.append(f"{name}/{obj}")

                class Blob:
                    def download_as_bytes(self_inner):
                        return json.dumps(META).encode()

                return Blob()

        return B()


class FakeModel:
    def __init__(self, name, version, aliases=()):
        self.resource_name = f"projects/p/locations/l/models/{name}"
        self.version_id = version
        self.aliases = list(aliases)


class FakeRegistry:
    versions = {}

    def __init__(self, model):
        self.model = model

    def list_versions(self):
        return [type("V", (), {"version_id": v, "version_aliases": a})() for v, a in self.versions.items()]

    def add_version_aliases(self, new_aliases, version):
        self.versions[version] = list(self.versions[version]) + new_aliases

    def remove_version_aliases(self, target_aliases, version):
        self.versions[version] = [a for a in self.versions[version] if a not in target_aliases]


class FakeVertex:
    """Stands in for the google.cloud.aiplatform module."""

    def __init__(self, existing=None):
        self.existing = existing
        self.uploaded = None
        FakeRegistry.versions = {"1": []} if existing else {}
        self.ModelRegistry = FakeRegistry
        outer = self

        class Model:
            @staticmethod
            def list(filter):
                outer.filter = filter
                return [outer.existing] if outer.existing else []

            @staticmethod
            def upload(**kwargs):
                outer.uploaded = kwargs
                FakeRegistry.versions[str(len(FakeRegistry.versions) + 1)] = []
                return FakeModel("1", str(len(FakeRegistry.versions)))

        self.Model = Model


def test_label_value_is_a_valid_gcp_label():
    v = label_value("Run_2026.01 ABC")
    assert v == "run_2026-01-abc" and len(label_value("x" * 99)) == 63


def test_first_registration_creates_the_model_with_serving_and_labels():
    vertex, spec = FakeVertex(), default_registry_spec()
    register(RUN, "logistic", vertex=vertex, storage_client=FakeStorage())
    up = vertex.uploaded
    assert up["display_name"] == spec.display_name
    assert up["artifact_uri"] == f"{RUN}/models/logistic/"
    assert up["serving_container_image_uri"] == spec.serving.image_uri
    assert up["serving_container_predict_route"] == spec.serving.predict_route
    assert up["serving_container_health_route"] == spec.serving.health_route
    assert up["serving_container_ports"] == [spec.serving.port]
    assert up["labels"] == {
        spec.labels.git_sha: "abc123-dirty",
        spec.labels.run_id: "run-20260101-000000",
        spec.labels.candidate: "logistic",
    }
    assert "pr_auc=0.13" in up["version_description"] and "test" in up["version_description"]
    assert not up.get("parent_model")


def test_existing_model_gets_a_new_version_not_a_second_model():
    parent = FakeModel("1", "1")
    vertex = FakeVertex(existing=parent)
    register(RUN, "logistic", vertex=vertex, storage_client=FakeStorage())
    assert vertex.uploaded["parent_model"] == parent.resource_name
    assert vertex.uploaded["is_default_version"] is False
    assert default_registry_spec().display_name in vertex.filter


def test_register_moves_the_challenger_alias_to_the_new_version():
    vertex = FakeVertex(existing=FakeModel("1", "1"))
    FakeRegistry.versions = {"1": ["challenger"]}
    version = register(RUN, "logistic", vertex=vertex, storage_client=FakeStorage())
    assert version.version_id == "2"
    assert FakeRegistry.versions == {"1": [], "2": ["challenger"]}


def test_promote_moves_champion_alias():
    vertex = FakeVertex(existing=FakeModel("1", "1"))
    FakeRegistry.versions = {"1": ["champion"], "2": ["challenger"]}
    promote("2", vertex=vertex)
    assert FakeRegistry.versions == {"1": [], "2": ["challenger", "champion"]}


def test_register_never_creates_endpoints():
    vertex = FakeVertex()
    register(RUN, "logistic", vertex=vertex, storage_client=FakeStorage())
    assert not hasattr(vertex, "Endpoint")
