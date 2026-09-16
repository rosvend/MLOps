"""Stage 9 offline report: drift and model-quality, train window vs held-out test window.

    make monitor

Both windows carry labels - unlike live production traffic, which never does; a credit
application's outcome is unknown at scoring time. That is what makes this the only place
in the project honest classification-quality drift can be reported at all. The live
endpoint (src/api/app.py) compares the same reference against logged production traffic
and deliberately stops at feature/prediction/operational drift, with no accuracy claim it
has no label to support.
"""

import logging
from pathlib import Path

import hydra
import joblib
from omegaconf import DictConfig, OmegaConf

from src.config import Config, from_dict
from src.data.factory import build_source
from src.models.dataset import load_from_feast, split_out_of_time
from src.models.metrics import summarize_classification
from src.monitoring.alerts import send_alert
from src.monitoring.drift_detector import run_drift_report

_log = logging.getLogger(__name__)
RAIZ = Path(__file__).resolve().parents[2]


def run(config: Config) -> dict:
    spec = config.monitoring
    datos = split_out_of_time(
        load_from_feast(build_source(config.data_source), config.features), config.training.split
    )
    modelo = joblib.load(RAIZ / config.serving.model_path)

    tren, prueba = datos.train, datos.test
    p_tren = modelo.predict_proba(tren.X)[:, 1]
    p_prueba = modelo.predict_proba(prueba.X)[:, 1]

    resumen_drift = run_drift_report(tren.X, prueba.X, spec)

    umbral = tren.X.pipe(lambda _: __import__("json").loads(
        (RAIZ / config.serving.meta_path).read_text()
    )["threshold"])
    calidad_tren = summarize_classification(tren.y.to_numpy(), p_tren, threshold=umbral)
    calidad_prueba = summarize_classification(prueba.y.to_numpy(), p_prueba, threshold=umbral)

    flagged_tren = float((p_tren >= umbral).mean())
    flagged_prueba = float((p_prueba >= umbral).mean())

    informe = {
        **resumen_drift.to_dict(),
        "model_quality": {
            "train": {k: v for k, v in calidad_tren.items() if k != "confusion_matrix"},
            "test": {k: v for k, v in calidad_prueba.items() if k != "confusion_matrix"},
        },
        "operational": {
            "flagged_share_train": flagged_tren,
            "flagged_share_test": flagged_prueba,
        },
    }

    resumen_drift.save(RAIZ / spec.offline_report_html, RAIZ / spec.offline_report_json)
    # save() writes only the drift keys; the fuller report (model quality, operational)
    # is what this pipeline actually promises, so it overwrites the JSON with everything.
    import json

    (RAIZ / spec.offline_report_json).write_text(json.dumps(informe, indent=2))

    if informe["dataset_drift_detected"]:
        send_alert(
            f"dataset drift detected: {informe['drift_share']:.1%} of features drifted "
            f"(train window vs held-out test)",
            informe,
            spec,
        )
    if flagged_prueba > spec.flagged_share_threshold:
        send_alert(
            f"flagged share {flagged_prueba:.1%} exceeds the {spec.flagged_share_threshold:.0%} "
            f"operational threshold on the held-out test window",
            {"flagged_share": flagged_prueba},
            spec,
        )

    print(f"Drift del dataset : {informe['dataset_drift_detected']}  (share {informe['drift_share']:.2%})")
    print(f"Features con drift: {informe['drifted_features'] or '(ninguna)'}")
    print(f"Gini  train/test  : {calidad_tren['gini']:.4f} / {calidad_prueba['gini']:.4f}")
    print(f"PR-AUC train/test : {calidad_tren['pr_auc']:.4f} / {calidad_prueba['pr_auc']:.4f}")
    print(f"Marcado train/test: {flagged_tren:.2%} / {flagged_prueba:.2%}")
    print(f"Reporte           : {spec.offline_report_html}")
    print(f"Métricas          : {spec.offline_report_json}")
    return informe


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    run(from_dict(OmegaConf.to_container(cfg, resolve=True)))


if __name__ == "__main__":
    main()
