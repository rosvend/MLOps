"""Fail the build when the champion's recorded metrics fall below the configured minimums.

    make check-metrics
"""

import json
import math
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "ci" / "metric_thresholds.yaml"


class MetricsCheckError(Exception):
    pass


def check(meta: dict, minimums: dict[str, float]) -> None:
    metrics = meta.get("metrics")
    if not isinstance(metrics, dict):
        raise MetricsCheckError("metadata has no 'metrics' mapping")

    problems = []
    for name, minimum in minimums.items():
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
            problems.append(f"{name}: missing or not a number ({value!r})")
        elif value < minimum:
            problems.append(f"{name}: {value:.4f} is below the minimum {minimum}")
    if problems:
        raise MetricsCheckError("; ".join(problems))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    config_path = Path(args[0]) if args else DEFAULT_CONFIG
    config = yaml.safe_load(config_path.read_text())
    meta_path = Path(config["meta_path"])
    if not meta_path.is_absolute():
        meta_path = config_path.resolve().parents[2] / meta_path

    try:
        meta = json.loads(meta_path.read_text())
        check(meta, config["minimums"])
    except (OSError, json.JSONDecodeError, MetricsCheckError) as exc:
        print(f"Metrics check FAILED: {exc}")
        return 1
    shown = ", ".join(f"{name}={meta['metrics'][name]:.4f}" for name in config["minimums"])
    print(f"Metrics check passed: {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
