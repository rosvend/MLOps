"""Six more comparison plots: confusion matrices, ROC/PR curves, a parallel-coordinates
view across every metric, per-feature drift magnitude, and why out-of-time matters.

    uv run python scripts/plot_model_analysis.py

Reloads the three tuned candidates by name from the current MLflow store (exactly one
logged model per name after a `make train` run - verified, not assumed) and the
heuristic module directly, then scores all four on the real held-out test window. Static
PNGs for docs/README, not a dashboard.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this is a standalone
# tool outside src/pipelines/, not a package module - it needs src on the path itself.

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import load_config
from src.data.factory import build_source
from src.models.dataset import load_from_feast, split_out_of_time
from src.models.estimator import HeuristicModel
from src.monitoring.drift_detector import run_drift_report
from src.pipelines.train import _tracking_uri

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "docs" / "plots"

ORDEN = ["heuristic", "logistic", "xgboost", "lightgbm"]
ETIQUETAS = {"heuristic": "Heuristic", "logistic": "Logistic (champion)",
             "xgboost": "XGBoost", "lightgbm": "LightGBM"}
COLOR = {"heuristic": "#2a78d6", "logistic": "#eb6834", "xgboost": "#1baf7a", "lightgbm": "#eda100"}
TEXTO_PRIMARIO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRID = "#d9d8d3"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "text.color": TEXTO_PRIMARIO, "axes.labelcolor": TEXTO_SECUNDARIO,
    "xtick.color": TEXTO_PRIMARIO, "ytick.color": TEXTO_SECUNDARIO,
    "font.family": "sans-serif",
})


def _cargar_modelo_entrenado(nombre: str, experimento: str):
    """The one logged model with this name in the current store - a training run logs
    exactly one per candidate, verified against the live store before relying on it."""
    exp = mlflow.get_experiment_by_name(experimento)
    candidatos = [
        m for m in mlflow.search_logged_models(experiment_ids=[exp.experiment_id], output_format="list")
        if m.name == nombre
    ]
    if len(candidatos) != 1:
        raise RuntimeError(f"esperaba exactamente un modelo '{nombre}' en MLflow, hay {len(candidatos)}")
    return mlflow.sklearn.load_model(f"models:/{candidatos[0].model_id}")


def _puntuar_todos(config):
    """Real out-of-time scores for every candidate, on the same held-out test window
    train.py itself selected the champion against."""
    datos = split_out_of_time(
        load_from_feast(build_source(config.data_source), config.features), config.training.split
    )
    tren, prueba = datos.train, datos.test
    mlflow.set_tracking_uri(_tracking_uri(config.training.tracking_uri))

    modelos = {"heuristic": HeuristicModel().fit(tren.X, tren.y)}
    for nombre in ("logistic", "xgboost", "lightgbm"):
        modelos[nombre] = _cargar_modelo_entrenado(nombre, config.training.experiment)

    puntajes = {"heuristic": modelos["heuristic"].decision_function(prueba.X).astype(float)}
    for nombre in ("logistic", "xgboost", "lightgbm"):
        puntajes[nombre] = modelos[nombre].predict_proba(prueba.X)[:, 1]

    return tren, prueba, modelos, puntajes


def plot_confusion_matrices(y_true: np.ndarray, puntajes: dict, flagged_share: float) -> Path:
    """Each model at its own threshold calibrated to the same 15.7% flagged share on
    this test set - the project's established operating-point convention, not 0.5."""
    fig, ejes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, modelo in zip(ejes, ORDEN):
        s = puntajes[modelo]
        umbral = np.quantile(s, 1 - flagged_share)
        pred = s >= umbral
        cm = confusion_matrix(y_true, pred, labels=[False, True])
        ax.imshow(cm, cmap=_sequential_cmap(COLOR[modelo]), vmin=0)
        for (i, j), v in np.ndenumerate(cm):
            claro = cm.max() and v > cm.max() * 0.6
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                    color="white" if claro else TEXTO_PRIMARIO)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nno default", "Predicted\ndefault"], fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual\nno default", "Actual\ndefault"], fontsize=9)
        ax.set_title(ETIQUETAS[modelo], fontsize=11.5, color=TEXTO_PRIMARIO, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(
        "Confusion matrix, each model at its own 15.7%-flagged threshold",
        fontsize=13.5, fontweight="bold", x=0.02, ha="left",
    )
    fig.text(0.02, -0.02, f"Held-out test window, {len(y_true):,} loans.",
              fontsize=9.5, color=TEXTO_SECUNDARIO)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    destino = SALIDA / "confusion_matrices.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destino


def _sequential_cmap(hex_color: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq", ["#ffffff", hex_color])


def plot_roc_curves(y_true: np.ndarray, puntajes: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=GRID, zorder=1)
    for modelo in ORDEN:
        fpr, tpr, _ = roc_curve(y_true, puntajes[modelo])
        auc = roc_auc_score(y_true, puntajes[modelo])
        ax.plot(fpr, tpr, linewidth=2, color=COLOR[modelo],
                 label=f"{ETIQUETAS[modelo]} (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve, out of time", fontsize=13, fontweight="bold", color=TEXTO_PRIMARIO,
                 pad=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    destino = SALIDA / "roc_curves.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destino


def plot_pr_curves(y_true: np.ndarray, puntajes: dict) -> Path:
    """PR, not just ROC: this project selects on PR-AUC because ROC is misleadingly
    optimistic at a 4.75% base rate - the curve that actually decided the champion."""
    base = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.axhline(base, linestyle="--", linewidth=1, color=GRID, zorder=1)
    for modelo in ORDEN:
        precision, recall, _ = precision_recall_curve(y_true, puntajes[modelo])
        ap = average_precision_score(y_true, puntajes[modelo])
        ax.plot(recall, precision, linewidth=2, color=COLOR[modelo],
                 label=f"{ETIQUETAS[modelo]} (PR-AUC {ap:.3f})")
    ax.annotate(f"random: {base:.3f}", (0.98, base), xytext=(0, 5), textcoords="offset points",
                ha="right", fontsize=9, color=TEXTO_SECUNDARIO)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve, out of time — the metric that chose the champion",
                 fontsize=12.5, fontweight="bold", color=TEXTO_PRIMARIO, pad=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, None)
    fig.tight_layout()
    destino = SALIDA / "pr_curves.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destino


def plot_parallel_coordinates() -> Path:
    """Every model across every metric at once. All six are "higher is better", so no
    axis needs inverting for the reading to stay consistent: outward is always better."""
    filas = {f["model"]: f for f in json.loads((RAIZ / "reports" / "champion.json").read_text())["comparison"]}
    metricas = ["pr_auc", "roc_auc", "gini", "f1", "precision", "recall"]
    etiquetas_eje = ["PR-AUC", "ROC-AUC", "Gini", "F1", "Precision", "Recall"]

    tabla = pd.DataFrame({m: [filas[mod][m] for mod in ORDEN] for m in metricas}, index=ORDEN)
    normalizada = (tabla - tabla.min()) / (tabla.max() - tabla.min())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(metricas))
    for eje in x:
        ax.axvline(eje, color=GRID, linewidth=1, zorder=1)
    for modelo in ORDEN:
        ax.plot(list(x), normalizada.loc[modelo], color=COLOR[modelo], linewidth=2.2,
                 marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1,
                 label=ETIQUETAS[modelo], zorder=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(etiquetas_eje, fontsize=10.5)
    ax.set_yticks([])
    ax.set_title(
        "Every model, every metric — normalised so outward always means better",
        fontsize=12.5, fontweight="bold", color=TEXTO_PRIMARIO, pad=12, loc="left",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=9.5)
    fig.tight_layout()
    destino = SALIDA / "parallel_coordinates.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destino


def plot_feature_drift_magnitude(tren_X: pd.DataFrame, prueba_X: pd.DataFrame, tren_p, prueba_p, config, top_n=12) -> Path:
    """Top drifted features, train vs the held-out test window - the same comparison
    make monitor runs, with the per-column magnitude it computes but does not plot."""
    comp_tren = tren_X.assign(probability_default=tren_p)
    comp_prueba = prueba_X.assign(probability_default=prueba_p)
    resumen = run_drift_report(comp_tren, comp_prueba, config.monitoring)
    d = resumen.snapshot.dict()
    id_a_col = {m["id"]: m["config"].get("column") for m in d["metrics"]}
    id_a_val = {m["id"]: m["value"] for m in d["metrics"]}

    TOPE = 20.0  # a p-value can be arbitrarily close to zero, which would let one
    # column's bar dwarf every other on a linear axis - capped so the chart stays
    # legible; the label says so on any bar that hit it, rather than hiding the cap.
    filas = []
    for prueba_metric in d["tests"]:
        mid = prueba_metric["metric_config"]["metric_id"]
        columna = id_a_col.get(mid)
        if columna is None or columna not in resumen.methods_used:
            continue
        metodo = resumen.methods_used[columna]
        valor = id_a_val[mid]
        # A single "times over its own threshold" score, comparable across p-value
        # methods (invert: smaller p = more extreme) and distance methods (use as is).
        razon = (config.monitoring.alpha / valor) if metodo != "wasserstein" else (valor / config.monitoring.alpha)
        recortada = min(razon, TOPE)
        filas.append((columna, recortada, razon > TOPE, columna in resumen.drifted_features))
    filas.sort(key=lambda f: f[1], reverse=True)
    filas = filas[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    colores = ["#e34948" if drift else "#c9c8c2" for _, _, _, drift in filas]
    barras = ax.barh([f[0] for f in filas], [f[1] for f in filas], color=colores, height=0.6)
    for rect, (_, valor, recortado, _) in zip(barras, filas):
        etiqueta = f"{valor:.0f}×+" if recortado else f"{valor:.1f}×"
        ax.annotate(etiqueta, (rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=9.5,
                    color=TEXTO_PRIMARIO)
    ax.axvline(1.0, color=TEXTO_SECUNDARIO, linewidth=1, linestyle="--")
    ax.annotate("its own drift threshold", (1.0, len(filas) - 0.6), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=TEXTO_SECUNDARIO)
    ax.set_xlabel("Multiples past its own drift threshold (capped at 20×; higher = more drifted)")
    ax.set_title(
        f"Top {top_n} most-drifted features, train vs held-out test", fontsize=12.5,
        fontweight="bold", color=TEXTO_PRIMARIO, pad=10, loc="left",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(0, TOPE * 1.12)
    fig.tight_layout()
    destino = SALIDA / "feature_drift_magnitude.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destino


def plot_why_out_of_time_matters() -> Path:
    """The champion's own quality, train vs test - read straight from the offline
    monitor report. This is the picture behind "out of time", not just the phrase."""
    m = json.loads((RAIZ / "reports" / "evidently_metrics.json").read_text())["model_quality"]
    metricas = [("gini", "Gini"), ("pr_auc", "PR-AUC")]

    fig, ejes = plt.subplots(1, 2, figsize=(8.5, 4))
    for ax, (clave, etiqueta) in zip(ejes, metricas):
        y = [m["train"][clave], m["test"][clave]]
        x = [0, 1]
        ax.plot(x, y, color=GRID, linewidth=2, zorder=1)
        ax.scatter(x, y, s=90, color=["#c9c8c2", COLOR["logistic"]], zorder=3,
                   edgecolor="white", linewidth=1.5)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=11, color=TEXTO_PRIMARIO)
        ax.set_xticks(x); ax.set_xticklabels(["Train\n(< Jun 2025)", "Held-out test\n(>= Jun 2025)"])
        ax.set_title(etiqueta, fontsize=12, color=TEXTO_PRIMARIO, pad=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.yaxis.set_visible(False)
        pad = (max(y) - min(y)) * 0.6 or 0.02
        ax.set_ylim(min(y) - pad, max(y) + pad)
    fig.suptitle(
        "Why out-of-time: the champion's own quality on data it never trained on",
        fontsize=12.5, fontweight="bold", x=0.02, ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    destino = SALIDA / "why_out_of_time_matters.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destino


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    config = load_config()
    tren, prueba, modelos, puntajes = _puntuar_todos(config)
    y_true = prueba.y.to_numpy()  # already the default indicator: True means defaulted

    p_tren_campeon = modelos["logistic"].predict_proba(tren.X)[:, 1]

    destinos = [
        plot_confusion_matrices(y_true, puntajes, config.training.selection.flagged_share),
        plot_roc_curves(y_true, puntajes),
        plot_pr_curves(y_true, puntajes),
        plot_parallel_coordinates(),
        plot_feature_drift_magnitude(tren.X, prueba.X, p_tren_campeon, puntajes["logistic"], config),
        plot_why_out_of_time_matters(),
    ]
    for d in destinos:
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
