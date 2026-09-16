"""Stage 7: tune candidate models, track every run in MLflow, pick a champion.

    make train
    uv run python -m src.pipelines.train training.n_trials=50
    uv run python -m src.pipelines.train training.selection.primary_metric=gini

Protocol. Hyperparameters are searched with Optuna against stratified K-fold cross
validation *inside the training window only*. The newest vintages are held out and read
once, at the end, to decide the champion - shuffled folds would let a model see loans
from the same months it is scoring, and this book spans 18 vintages.

Every transformer lives inside the sklearn Pipeline, so each fold refits its own
winsor caps, medians and encodings. Nothing is fitted before the split.
"""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import Config, from_dict
from src.data.factory import build_source
from src.features.spec import FeatureSpec
from src.models.champion import (
    CandidateResult,
    as_record,
    comparison_table,
    justification,
    select_champion,
)
from src.models.dataset import Dataset, OutOfTime, load_from_feast, split_out_of_time
from src.models.metrics import summarize_classification
from src.models.pipelines import _resolve, build_model
from src.models.training_spec import Candidate, TrainingSpec

_log = logging.getLogger(__name__)
RAIZ = Path(__file__).resolve().parents[2]
REPORTES = RAIZ / "reports"


def _tracking_uri(configurado: str) -> str:
    """Resolve a relative sqlite path against the project root, so cwd cannot move the store."""
    prefijo = "sqlite:///"
    if configurado.startswith(prefijo):
        ruta = Path(configurado[len(prefijo):])
        if not ruta.is_absolute():
            return f"{prefijo}{RAIZ / ruta}"
    return configurado


def _ensure_experiment(spec: TrainingSpec) -> None:
    """Create the experiment with a project-local artifact root the first time only."""
    if mlflow.get_experiment_by_name(spec.experiment) is None:
        mlflow.create_experiment(
            spec.experiment, artifact_location=(RAIZ / spec.artifact_location).as_uri()
        )
    mlflow.set_experiment(spec.experiment)


def _scores(modelo, X: pd.DataFrame) -> np.ndarray:
    """A probability where there is one, otherwise the raw ranking score."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def _cv(spec: TrainingSpec) -> StratifiedKFold:
    return StratifiedKFold(spec.cv_folds, shuffle=True, random_state=spec.seed)


def _tune(nombre: str, cand: Candidate, train: Dataset, spec: TrainingSpec, features: FeatureSpec):
    """One Optuna study, one MLflow child run per trial."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objetivo(trial: optuna.Trial) -> float:
        params = dict(cand.fixed)
        params |= {k: d.suggest(trial, k) for k, d in cand.search.items()}
        modelo = build_model(
            cand.estimator, params, scale=cand.scale, spec=features, seed=spec.seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            puntajes = cross_val_score(
                modelo, train.X, train.y, cv=_cv(spec), scoring=spec.selection.cv_scoring, n_jobs=1
            )
        with mlflow.start_run(run_name=f"{nombre}-trial-{trial.number}", nested=True):
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
            mlflow.log_metrics(
                {"cv_mean": float(puntajes.mean()), "cv_std": float(puntajes.std())}
            )
        trial.set_user_attr("cv_std", float(puntajes.std()))
        return float(puntajes.mean())

    estudio = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=spec.seed)
    )
    estudio.optimize(objetivo, n_trials=spec.n_trials, show_progress_bar=False)
    return estudio


def _importances(modelo) -> dict[str, float] | None:
    paso = modelo.named_steps["model"]
    nombres = list(modelo.named_steps["features"].get_feature_names_out())
    if hasattr(paso, "feature_importances_"):
        valores = np.asarray(paso.feature_importances_, dtype=float)
    elif hasattr(paso, "coef_"):
        valores = np.abs(np.ravel(paso.coef_)).astype(float)
    else:
        return None
    return dict(sorted(zip(nombres, valores), key=lambda kv: -kv[1]))


def _evaluate(
    nombre: str, modelo, datos: OutOfTime, tuned: bool, flagged_share: float, **extra
) -> CandidateResult:
    """Fit on the training window, then read the held-out window exactly once."""
    inicio = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(datos.train.X, datos.train.y)
    fit_s = time.perf_counter() - inicio

    inicio = time.perf_counter()
    puntajes = _scores(modelo, datos.test.X)
    latencia = (time.perf_counter() - inicio) * 1000 * 1000 / max(len(datos.test), 1)

    # Every candidate is thresholded at the same operating point - the share actually sent
    # to manual review - so the thresholded metrics compare like for like. 0.5 would mean
    # something different for a probability model than for a ranking one.
    probabilistico = hasattr(modelo, "predict_proba")
    umbral = float(np.quantile(puntajes, 1 - flagged_share))
    return CandidateResult(
        name=nombre,
        metrics=summarize_classification(
            datos.test.y, puntajes, threshold=umbral, probabilistic=probabilistico
        ),
        fit_seconds=fit_s,
        predict_ms_per_1k=latencia,
        tuned=tuned,
        **extra,
    )


def run(config: Config) -> dict[str, Any]:
    spec, features = config.training, config.features
    mlflow.set_tracking_uri(_tracking_uri(spec.tracking_uri))
    _ensure_experiment(spec)

    datos = split_out_of_time(load_from_feast(build_source(config.data_source), features), spec.split)
    _log.info(datos.describe())
    print(f"\nSplit: {datos.describe()}\n")

    resultados: list[CandidateResult] = []

    for nombre, cand in spec.candidates.items():
        print(f"Tuning {nombre} ({spec.n_trials} trials x {spec.cv_folds} folds)...", flush=True)
        with mlflow.start_run(run_name=nombre) as parent:
            estudio = _tune(nombre, cand, datos.train, spec, features)
            params = dict(cand.fixed) | estudio.best_params
            modelo = build_model(
                cand.estimator, params, scale=cand.scale, spec=features, seed=spec.seed
            )
            resultado = _evaluate(
                nombre,
                modelo,
                datos,
                tuned=True,
                flagged_share=spec.selection.flagged_share,
                params=estudio.best_params,
                cv_mean=estudio.best_value,
                cv_std=estudio.best_trial.user_attrs.get("cv_std", float("nan")),
                run_id=parent.info.run_id,
            )
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
            mlflow.log_metrics(
                {"cv_mean": resultado.cv_mean, "cv_std": resultado.cv_std}
                | {f"oot_{k}": v for k, v in resultado.metrics.items() if isinstance(v, float)}
                | {"fit_seconds": resultado.fit_seconds}
            )
            if (imp := _importances(modelo)) is not None:
                mlflow.log_dict(imp, "feature_importances.json")
            # cloudpickle, not skops: the pipeline carries custom transformers and pydantic
            # config objects, and skops' allow-list would have to enumerate every stdlib
            # container inside every estimator. These artifacts are produced and read by
            # this project only.
            mlflow.sklearn.log_model(
                modelo,
                name=nombre,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            resultados.append(resultado)
            print(f"  {nombre}: cv {resultado.cv_mean:.4f}, "
                  f"oot {spec.selection.primary_metric} "
                  f"{resultado.score(spec.selection.primary_metric):.4f}", flush=True)

    # The incumbent, untuned: frozen rules, entered so the selection can keep it.
    print("Evaluating the incumbent heuristic (no tuning)...", flush=True)
    with mlflow.start_run(run_name="heuristic") as parent:
        # No FeatureEngineer: the frozen rules read named raw columns, and encoding them
        # would remove exactly the columns they look up.
        base = _resolve(spec.baseline.estimator)()
        resultado = _evaluate(
            "heuristic",
            base,
            datos,
            tuned=False,
            flagged_share=spec.selection.flagged_share,
            run_id=parent.info.run_id,
        )
        mlflow.log_metrics(
            {f"oot_{k}": v for k, v in resultado.metrics.items() if isinstance(v, float)}
        )
        resultados.append(resultado)

    metrica = spec.selection.primary_metric
    tabla = comparison_table(resultados, spec.selection.report_metrics)
    campeon = select_champion(resultados, metrica)
    texto = justification(campeon, resultados, metrica)

    print(f"\n{tabla.round(4).to_string()}\n\n{texto}\n")
    REPORTES.mkdir(exist_ok=True)
    registro = as_record(campeon, tabla, metrica)
    (REPORTES / "champion.json").write_text(json.dumps(registro, indent=2, default=str))
    with mlflow.start_run(run_name="champion-selection"):
        mlflow.log_param("primary_metric", metrica)
        mlflow.log_param("champion", campeon.name)
        mlflow.log_dict(registro, "champion.json")
        mlflow.log_text(texto, "justification.txt")
    return registro


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    run(from_dict(OmegaConf.to_container(cfg, resolve=True)))


if __name__ == "__main__":
    main()
