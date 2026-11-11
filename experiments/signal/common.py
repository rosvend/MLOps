"""Shared setup: development data (cached), fold sets and the pipelines under comparison."""

import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.signal.data import load_development
from experiments.signal.folds import stratified_folds, time_folds
from src.features.engineering import FeatureEngineer

CACHE = Path(os.environ.get("SIGNAL_CACHE", "/tmp/signal_dev.parquet"))
OUT = Path(__file__).resolve().parents[2] / "docs" / "investigation"
CHAMPION_C = 0.08686007279808698  # from reports/champion.json
SEED = 0


def get_dev():
    """X, y, vintage of TRAIN+VALIDATE months. The TEST months were dropped before caching."""
    if not CACHE.exists():
        d = load_development()
        d.X.assign(y=d.y, mes=d.vintage).to_parquet(CACHE)
    t = pd.read_parquet(CACHE)
    return t.drop(columns=["y", "mes"]), t["y"].astype(int), t["mes"]


def folds_for(y, vintage):
    return {"time": time_folds(vintage, 4), "strat": stratified_folds(y, 5, 3, SEED)}


def logistic(C=CHAMPION_C, class_weight=None, pre=(), post=(), model=None):
    """Champion recipe (engineer -> scale -> logistic) with optional steps around the engineer."""
    steps = [(f"pre{i}", s) for i, s in enumerate(pre)] + [("features", FeatureEngineer())]
    steps += [(f"post{i}", s) for i, s in enumerate(post)] + [("scale", StandardScaler())]
    steps.append(("model", model or LogisticRegression(C=C, class_weight=class_weight, max_iter=2000)))
    return Pipeline(steps)
