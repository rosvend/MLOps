"""Export the labelled loans with an explicit time-based split column for Vertex AI.

    make export-gcp-dataset

AutoML splits randomly unless told otherwise, which would let it train on months it is
scored on. The split column pins TRAIN / VALIDATE / TEST by origination month, with TEST
identical to the local champion's held-out window so both are compared on the same loans.
"""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.config import Config, from_dict
from src.data.factory import build_source
from src.features.spec import FeatureSpec
from src.models.dataset import vintage_cutoff
from src.models.training_spec import GcpDatasetSpec, Split
from src.pipelines.prepare import prepare_labelled

_log = logging.getLogger(__name__)
RAIZ = Path(__file__).resolve().parents[2]


def assign_split(vintage: pd.Series, split: Split, gcp: GcpDatasetSpec) -> pd.Series:
    """TEST is the newest vintages; VALIDATE the newest slice of what remains; the rest TRAIN."""
    corte_test = vintage_cutoff(vintage, split.train_fraction)
    es_test = vintage >= corte_test
    corte_val = vintage_cutoff(vintage[~es_test], 1 - gcp.validation_fraction)
    etiquetas = pd.Series(gcp.train_value, index=vintage.index)
    etiquetas[(vintage >= corte_val) & ~es_test] = gcp.validation_value
    etiquetas[es_test] = gcp.test_value
    return etiquetas


def build_dataset(
    etiquetado: pd.DataFrame, features: FeatureSpec, split: Split, gcp: GcpDatasetSpec
) -> pd.DataFrame:
    """Model features, the default label and the split. Identifiers and leaky columns stay out."""
    salida = etiquetado[features.feature_view_columns].copy()
    salida[gcp.label_column] = (~etiquetado[features.target]).astype(int)
    salida[gcp.split_column] = assign_split(etiquetado[split.by], split, gcp)
    return salida.reset_index(drop=True)


def export(config: Config) -> Path:
    etiquetado = prepare_labelled(build_source(config.data_source), config.features)
    gcp = config.gcp_dataset
    datos = build_dataset(etiquetado, config.features, config.training.split, gcp)

    destino = RAIZ / gcp.output_path
    destino.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(destino, index=False)

    resumen = datos.groupby(gcp.split_column)[gcp.label_column].agg(["size", "mean"])
    print(resumen.rename(columns={"size": "loans", "mean": "default_rate"}).to_string())
    print(f"Artefacto: {destino} ({destino.stat().st_size / 1024:.1f} KB)")
    return destino


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    export(from_dict(OmegaConf.to_container(cfg, resolve=True)))


if __name__ == "__main__":
    main()
