"""Export the champion MLflow run as a self-contained scoring artifact.

    make export-champion

The joblib file alone is not enough to serve with: a probability means nothing without
the cut-point that turns it into a decision, and an artifact nobody can trace back to a
run has no business in an image. Both go in the sidecar.

The threshold is computed once, here, on the training window, and frozen. Recomputing it
per batch would make one applicant's decision depend on who else was scored alongside
them - the batch-dependence this pipeline has refused at every stage.
"""

import json
import logging
import subprocess
from pathlib import Path

import hydra
import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.config import Config, from_dict
from src.data.factory import build_source
from src.models.dataset import load_from_feast, split_out_of_time

_log = logging.getLogger(__name__)
RAIZ = Path(__file__).resolve().parents[2]
REPORTES = RAIZ / "reports"


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RAIZ, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def export(config: Config) -> dict:
    import mlflow

    from src.pipelines.train import _ensure_experiment, _tracking_uri

    registro = json.loads((REPORTES / "champion.json").read_text())
    nombre = registro["champion"]["name"]

    mlflow.set_tracking_uri(_tracking_uri(config.training.tracking_uri))
    _ensure_experiment(config.training)
    model_id = registro["champion"].get("model_id")
    if not model_id:
        raise RuntimeError(
            "reports/champion.json no registra model_id; vuelve a correr `make train` "
            "para que la selección anote qué artefacto exacto ganó"
        )
    try:
        # By id, never by name and creation time: several runs log a model called
        # 'logistic', and picking the newest would ship weights from one run with the
        # metrics and threshold of another.
        modelo = mlflow.sklearn.load_model(f"models:/{model_id}")
    except Exception as exc:
        raise RuntimeError(f"no se pudo cargar el modelo {model_id} de MLflow: {exc}") from exc

    # The cut-point comes from the training window only, and is frozen from here on.
    datos = split_out_of_time(
        load_from_feast(build_source(config.data_source), config.features), config.training.split
    )
    proba_train = modelo.predict_proba(datos.train.X)[:, 1]
    cuota = config.training.selection.flagged_share
    umbral = float(np.quantile(proba_train, 1 - cuota))

    destino = RAIZ / config.serving.model_path
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, destino)

    meta = {
        "model": nombre,
        "threshold": umbral,
        "flagged_share_at_fit": cuota,
        "calibrated_on": f"training window before {datos.corte}",
        "features": list(config.features.feature_view_columns),
        "metrics": registro["champion"]["metrics"],
        "mlflow_model_id": model_id,
        "mlflow_run_id": registro["champion"].get("run_id"),
        "git_sha": _git_sha(),
    }
    (RAIZ / config.serving.meta_path).write_text(json.dumps(meta, indent=2, default=str))

    _log.info("champion %s -> %s (%.1f KB)", nombre, destino, destino.stat().st_size / 1024)
    print(f"Champion    : {nombre}")
    print(f"Umbral      : {umbral:.6f}  ({cuota:.1%} marcado en entrenamiento)")
    print(f"Calibrado en: {meta['calibrated_on']}")
    print(f"Artefacto   : {destino} ({destino.stat().st_size / 1024:.1f} KB)")
    print(f"Metadatos   : {config.serving.meta_path}")
    return meta


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    export(from_dict(OmegaConf.to_container(cfg, resolve=True)))


if __name__ == "__main__":
    main()
