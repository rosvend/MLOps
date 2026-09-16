"""Training data: features from Feast, label joined from outside it, split by vintage.

The label deliberately never entered the feature store, so it is joined here on the
entity key. The split is by origination month: the newest vintages are held out and read
once, after all tuning, because shuffled folds would let a model see loans from the very
months it is being asked to score.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.source import DataSource
from src.features.spec import FeatureSpec
from src.models.training_spec import Split
from src.pipelines.prepare import prepare_labelled

_log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2] / "feature_repo"


@dataclass(frozen=True)
class Dataset:
    """Features, label and the vintage each loan belongs to."""

    X: pd.DataFrame
    y: pd.Series
    vintage: pd.Series

    def __len__(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class OutOfTime:
    train: Dataset
    test: Dataset
    corte: str

    def describe(self) -> str:
        return (
            f"train {len(self.train):,} loans (< {self.corte}, {self.train.y.mean():.2%} default) "
            f"-> held out {len(self.test):,} (>= {self.corte}, {self.test.y.mean():.2%})"
        )


def load_from_feast(source: DataSource, spec: FeatureSpec) -> Dataset:
    """Point-in-time retrieval from the offline store, each loan at its own timestamp."""
    from feast import FeatureStore

    etiquetado = prepare_labelled(source)
    entity_df = pd.DataFrame(
        {
            spec.entity_key: etiquetado[spec.entity_key],
            "event_timestamp": pd.to_datetime(etiquetado[spec.event_timestamp]).dt.tz_localize(
                spec.event_timestamp_timezone
            ),
        }
    )
    refs = [f"{v}:{c}" for v, cols in spec.feature_views.items() for c in cols]
    store = FeatureStore(repo_path=str(REPO))
    recuperado = store.get_historical_features(entity_df=entity_df, features=refs).to_df()
    _log.info("Feast devolvió %d x %d", *recuperado.shape)

    # The label lives outside the store; join it back on the entity key.
    etiquetas = etiquetado.set_index(spec.entity_key)
    alineado = recuperado.set_index(spec.entity_key).loc[etiquetas.index]
    X = alineado[spec.feature_view_columns].reset_index(drop=True)
    y = (~etiquetas[spec.target]).reset_index(drop=True).rename("defaulted")
    vintage = etiquetas["mes_prestamo"].reset_index(drop=True)
    return Dataset(X=X, y=y, vintage=vintage)


def split_out_of_time(datos: Dataset, split: Split) -> OutOfTime:
    """Oldest vintages train; every test loan is originated no earlier than every train loan."""
    meses = sorted(datos.vintage.unique())
    acumulado = datos.vintage.value_counts().reindex(meses).cumsum() / len(datos)
    corte = next(
        (m for m in meses if acumulado[m] >= split.train_fraction), meses[-1]
    )
    es_train = datos.vintage < corte
    if not es_train.any() or es_train.all():
        raise ValueError(f"el corte {corte} deja un lado vacío")

    def _sub(mascara):
        return Dataset(
            X=datos.X[mascara].reset_index(drop=True),
            y=datos.y[mascara].reset_index(drop=True),
            vintage=datos.vintage[mascara].reset_index(drop=True),
        )

    return OutOfTime(train=_sub(es_train), test=_sub(~es_train), corte=corte)
