"""Pick the best candidate and measure it against a reference model."""

import math


def _score(result: dict, metric: str) -> float:
    value = result.get(metric, float("nan"))
    return float("-inf") if value is None or math.isnan(value) else value


def compare(results: list[dict], reference: dict, metric: str) -> dict:
    """Best by the primary metric (NaN cannot win) and each candidate's delta to the reference."""
    best = max(results, key=lambda r: _score(r, metric))
    deltas = {
        r["name"]: {k: r[k] - v for k, v in reference.items() if k != "name" and k in r}
        for r in results
    }
    return {"best": best["name"], "metric": metric, "reference": reference, "delta_vs_reference": deltas}
