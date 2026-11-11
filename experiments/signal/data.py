"""Development data only: the TEST window is dropped the moment the book is loaded."""

from dataclasses import dataclass

import pandas as pd

from src.features.contract import features
from src.models.dataset import vintage_cutoff


@dataclass(frozen=True)
class Development:
    X: pd.DataFrame
    y: pd.Series
    vintage: pd.Series
    corte: str


def development_only(prepared: pd.DataFrame, train_fraction: float, target: str = "Pago_atiempo") -> Development:
    """Keep loans originated before the out-of-time cut; newer months never leave this function."""
    corte = vintage_cutoff(prepared["mes_prestamo"], train_fraction)
    dev = prepared[prepared["mes_prestamo"] < corte].reset_index(drop=True)
    return Development(
        X=features(dev),
        y=(~dev[target]).astype(int).rename("defaulted"),
        vintage=dev["mes_prestamo"],
        corte=corte,
    )


def load_development() -> Development:
    from src.config import load_config
    from src.data.factory import build_source
    from src.pipelines.prepare import prepare_labelled

    cfg = load_config()
    prepared = prepare_labelled(build_source(cfg.data_source))
    return development_only(prepared, cfg.training.split.train_fraction)
