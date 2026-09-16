"""Two simple charts comparing the stage-7 candidates, from reports/champion.json.

    uv run python scripts/plot_model_comparison.py

Static PNGs, not a dashboard: this is for the README/docs, not a live monitoring
surface. Colors are the project's validated categorical palette (fixed order, never
reassigned by rank), and each model is direct-labeled on its own axis rather than
relying on a legend.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "docs" / "plots"

# Fixed categorical order, one color per model, never reassigned by rank or value.
ORDEN = ["heuristic", "logistic", "xgboost", "lightgbm"]
ETIQUETAS = {"heuristic": "Heuristic\n(baseline)", "logistic": "Logistic\n(champion)",
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


def _cargar():
    filas = json.loads((RAIZ / "reports" / "champion.json").read_text())["comparison"]
    return {f["model"]: f for f in filas}


def _barras(ax, valores, formato="{:.3f}", titulo=""):
    modelos = [m for m in ORDEN if valores.get(m) is not None]
    y = [valores[m] for m in modelos]
    x = range(len(modelos))
    barras = ax.bar(x, y, width=0.55, color=[COLOR[m] for m in modelos])
    for rect, valor in zip(barras, y):
        ax.annotate(
            formato.format(valor), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10.5,
            color=TEXTO_PRIMARIO,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([ETIQUETAS[m] for m in modelos])
    ax.set_title(titulo, fontsize=12, color=TEXTO_PRIMARIO, pad=10, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(0, max(y) * 1.22)


def plot_performance(datos: dict) -> Path:
    """PR-AUC and Gini, out of time - two panels, one metric each: they sit on
    different scales, so one shared axis would misrepresent both."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(
        "Out-of-time performance by model", fontsize=13.5, fontweight="bold",
        color=TEXTO_PRIMARIO, x=0.02, ha="left",
    )
    _barras(ax1, {m: d["pr_auc"] for m, d in datos.items()}, titulo="PR-AUC (the metric that decided)")
    _barras(ax2, {m: d["gini"] for m, d in datos.items()}, titulo="Gini")
    fig.text(
        0.02, -0.02,
        "Held out: newest loan vintages, never seen during tuning. Higher is better on both.",
        fontsize=9.5, color=TEXTO_SECUNDARIO,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    destino = SALIDA / "model_comparison_performance.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destino


def plot_cv_vs_oot(datos: dict) -> Path:
    """The finding that justified the selection protocol: the best cross-validation
    score was the worst out-of-time score, for the same three tuned candidates."""
    tuneados = [m for m in ("logistic", "xgboost", "lightgbm") if datos[m]["cv_mean"] == datos[m]["cv_mean"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(tuneados))
    ancho = 0.32
    cv = [datos[m]["cv_mean"] for m in tuneados]
    oot = [datos[m]["pr_auc"] for m in tuneados]
    b1 = ax.bar([i - ancho / 2 - 0.02 for i in x], cv, ancho, color="#c9c8c2", label="Cross-validation (tuning)")
    b2 = ax.bar([i + ancho / 2 + 0.02 for i in x], oot, ancho, color=[COLOR[m] for m in tuneados],
                label="Out-of-time (selection)")
    for rects, vals in ((b1, cv), (b2, oot)):
        for rect, valor in zip(rects, vals):
            ax.annotate(f"{valor:.3f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10,
                        color=TEXTO_PRIMARIO)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ETIQUETAS[m].replace("\n", " ") for m in tuneados])
    ax.set_title(
        "XGBoost had the best tuning score and the worst real one", fontsize=12.5,
        fontweight="bold", color=TEXTO_PRIMARIO, pad=12, loc="left",
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(0, max(cv + oot) * 1.25)
    fig.text(0.5, -0.05, "PR-AUC, both bars. The champion was chosen on the right bar, not the left.",
              fontsize=9.5, color=TEXTO_SECUNDARIO, ha="center")
    fig.tight_layout()
    destino = SALIDA / "model_comparison_cv_vs_oot.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destino


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    datos = _cargar()
    a = plot_performance(datos)
    b = plot_cv_vs_oot(datos)
    print(f"wrote {a}")
    print(f"wrote {b}")


if __name__ == "__main__":
    main()
