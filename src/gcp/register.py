"""Register a pipeline candidate in the Vertex AI Model Registry; never deploys it.

    uv run python -m src.gcp.register RUN_DIR CANDIDATE
    uv run python -m src.gcp.register --promote VERSION_ID
"""

import argparse
import re

from src.gcp.registry_spec import RegistrySpec, default_registry_spec
from src.gcp.storage import read_json


def label_value(text: str) -> str:
    """Lowercase letters, digits, dashes and underscores, at most 63 characters."""
    return re.sub(r"[^a-z0-9_-]", "-", str(text).lower())[:63]


def _find_parent(vertex, spec: RegistrySpec):
    found = vertex.Model.list(filter=f'display_name="{spec.display_name}"')
    return found[0] if found else None


def _move_alias(vertex, parent_name: str, alias: str, version_id: str) -> None:
    registry = vertex.ModelRegistry(model=parent_name)
    for v in registry.list_versions():
        if alias in (v.version_aliases or []) and v.version_id != version_id:
            registry.remove_version_aliases([alias], v.version_id)
    registry.add_version_aliases([alias], version_id)


def _description(spec: RegistrySpec, meta: dict) -> str:
    metrics = meta.get("metrics", {})
    test = metrics.get("test", metrics)
    shown = ", ".join(f"{k}={test[k]:.4g}" for k in spec.description_metrics if k in test)
    return f"{meta.get('model', '')} | test metrics: {shown} | threshold={meta.get('threshold')}"


def register(run_dir: str, candidate: str, spec: RegistrySpec | None = None, vertex=None, storage_client=None):
    spec = spec or default_registry_spec()
    if vertex is None:
        from google.cloud import aiplatform as vertex

        vertex.init(project=spec.project, location=spec.region)
    model_uri = f"{run_dir.rstrip('/')}/{spec.artifacts.models_dir}/{candidate}/"
    meta = read_json(model_uri + spec.artifacts.meta_file, client=storage_client)
    parent = _find_parent(vertex, spec)
    model = vertex.Model.upload(
        display_name=spec.display_name,
        description=spec.description,
        artifact_uri=model_uri,
        serving_container_image_uri=spec.serving.image_uri,
        serving_container_predict_route=spec.serving.predict_route,
        serving_container_health_route=spec.serving.health_route,
        serving_container_ports=[spec.serving.port],
        labels={
            spec.labels.git_sha: label_value(meta.get("git_sha", "unknown")),
            spec.labels.run_id: label_value(run_dir.rstrip("/").rsplit("/", 1)[-1]),
            spec.labels.candidate: label_value(candidate),
        },
        version_description=_description(spec, meta),
        parent_model=parent.resource_name if parent else None,
        is_default_version=parent is None,
    )
    _move_alias(vertex, model.resource_name, spec.aliases.challenger, model.version_id)
    return model


def promote(version_id: str, alias: str | None = None, spec: RegistrySpec | None = None, vertex=None) -> None:
    spec = spec or default_registry_spec()
    if vertex is None:
        from google.cloud import aiplatform as vertex

        vertex.init(project=spec.project, location=spec.region)
    parent = _find_parent(vertex, spec)
    if parent is None:
        raise LookupError(f"no hay modelo registrado como {spec.display_name!r}")
    _move_alias(vertex, parent.resource_name, alias or spec.aliases.champion, version_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("candidate", nargs="?")
    parser.add_argument("--promote", metavar="VERSION_ID")
    args = parser.parse_args()
    if args.promote:
        promote(args.promote)
        print(f"alias champion -> version {args.promote}")
        return
    if not (args.run_dir and args.candidate):
        parser.error("se necesita RUN_DIR y CANDIDATE, o --promote VERSION_ID")
    model = register(args.run_dir, args.candidate)
    print(f"{model.resource_name} version {model.version_id} (alias challenger)")


if __name__ == "__main__":
    main()
