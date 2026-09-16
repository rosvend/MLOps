"""Export the training window as the drift baseline everything else compares against.

    make export-reference

Reference is the training distribution, not the full book - the textbook definition of
a drift baseline, and the one that cannot silently include the future. Requires
`make export-champion` to have run first: the reference carries the champion's own
predictions, computed the same way a live request's would be.
"""

import logging
from pathlib import Path

import hydra
import joblib
from omegaconf import DictConfig, OmegaConf

from src.config import Config, from_dict
from src.data.factory import build_source
from src.models.dataset import load_from_feast, split_out_of_time

_log = logging.getLogger(__name__)
RAIZ = Path(__file__).resolve().parents[2]


def export(config: Config) -> Path:
    datos = split_out_of_time(
        load_from_feast(build_source(config.data_source), config.features), config.training.split
    )
    train = datos.train

    modelo_path = RAIZ / config.serving.model_path
    if not modelo_path.exists():
        raise RuntimeError(f"{modelo_path} no existe; corre `make export-champion` primero")
    modelo = joblib.load(modelo_path)
    proba = modelo.predict_proba(train.X)[:, 1]

    import json

    meta = json.loads((RAIZ / config.serving.meta_path).read_text())
    umbral = float(meta["threshold"])

    referencia = train.X.copy()
    referencia["probability_default"] = proba
    referencia["review_flag"] = proba >= umbral
    referencia["defaulted"] = train.y.to_numpy()

    destino = RAIZ / config.monitoring.reference_path
    destino.parent.mkdir(parents=True, exist_ok=True)
    referencia.to_parquet(destino, index=False)

    _log.info("referencia: %d filas x %d columnas -> %s", *referencia.shape, destino)
    print(f"Referencia  : {len(referencia):,} créditos (ventana de entrenamiento, < {datos.corte})")
    print(f"Umbral      : {umbral:.6f}  |  marcado en referencia: {referencia['review_flag'].mean():.2%}")
    print(f"Artefacto   : {destino} ({destino.stat().st_size / 1024:.1f} KB)")
    return destino


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    export(from_dict(OmegaConf.to_container(cfg, resolve=True)))


if __name__ == "__main__":
    main()
