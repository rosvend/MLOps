"""Figures for the Phase 2 page. Light surface with dark ink, so they read in both site themes."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
PLATFORM_COLOR = {"local": "#2a78d6", "vertex_pipeline": "#eb6834", "automl": "#1baf7a"}
MODEL_COLOR = {"Logistic": "#2a78d6", "XGBoost": "#eb6834", "LightGBM": "#6250d6",
               "Heuristic": "#52514e", "AutoML v1": "#1baf7a", "AutoML v2": "#e87ba4"}
GENERIC = "#8a8984"


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel or "", color=MUTED)
    ax.set_ylabel(ylabel or "", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _fig(w, h, **kw):
    fig, axes = plt.subplots(figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, axes


def _color(platform):
    return PLATFORM_COLOR.get(platform, GENERIC)


def _legend(ax, platforms, used):
    handles = [plt.Line2D([], [], marker="s", ls="", color=_color(k), label=v) for k, v in platforms.items() if k in used]
    handles += [h for h in ax.get_lines() if h.get_label() and not h.get_label().startswith("_")]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK, loc="center")


def metric_bars(rows, platforms, path):
    """PR-AUC / ROC-AUC / Gini / F1 per model and platform, with bootstrap 95 % intervals."""
    ok = [r for r in rows if r["status"] == "ok"][::-1]
    metrics = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("gini", "Gini"), ("f1", "F1 at 15.7 % flagged")]
    fig, axes = _fig(13, 5.2, ncols=4, sharey=True)
    y = np.arange(len(ok))
    names = [f"{r['label']} ({platforms[r['platform']]})" for r in ok]
    for ax, (key, title) in zip(axes, metrics):
        vals = np.array([r[key] for r in ok])
        err = np.array([[r[key] - r[f"{key}_ci"][0], r[f"{key}_ci"][1] - r[key]] for r in ok]).T
        ax.barh(y, vals, xerr=err, color=[_color(r["platform"]) for r in ok], height=0.6,
                error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        _style(ax, title)
        for yi, v in zip(y, vals):
            ax.text(v + 0.005, yi + 0.38, f"{v:.3f}", fontsize=7, color=INK, va="center")
    axes[0].set_yticks(y, names, fontsize=8, color=INK)
    handles = [plt.Line2D([], [], marker="s", ls="", color=_color(k), label=v) for k, v in platforms.items() if k in {r["platform"] for r in ok}]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK, loc="lower center", ncols=3)
    fig.suptitle("Same TEST loans, every model: bar = value, whisker = bootstrap 95 % interval", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def pr_curves(preds, entries, platforms, path):
    """Precision-recall overlay; colour = model, dashed = cloud copy of the same model."""
    fig, ax = _fig(8, 5.5)
    for e in entries:
        if e["id"] not in preds:
            continue
        p, r, _ = precision_recall_curve(preds["y"], preds[e["id"]])
        cloud = e["platform"] == "vertex_pipeline"
        ax.plot(r, p, color=MODEL_COLOR.get(e["label"], GENERIC), lw=1.2 if cloud else 2,
                ls=(0, (4, 3)) if cloud else "-", label=f"{e['label']} ({platforms[e['platform']]})")
    prevalence = preds["y"].mean()
    ax.axhline(prevalence, color=MUTED, lw=1, ls=":", label=f"No skill = default rate {prevalence:.1%}")
    ax.set_ylim(0, 0.5)
    _style(ax, "Precision-recall on the TEST window (dashed cloud lines sit on the local ones)", "Recall", "Precision")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def accuracy_vs_baseline(rows, baseline_acc, platforms, path):
    """Dot plot (not bars, so the shortened axis cannot mislead) of accuracy against 'nobody defaults'."""
    ok = [r for r in rows if r["status"] == "ok"][::-1]
    fig, ax = _fig(8.5, 5)
    y = np.arange(len(ok))
    ax.scatter([r["accuracy"] for r in ok], y, s=70, c=[_color(r["platform"]) for r in ok], zorder=3)
    for yi, r in zip(y, ok):
        ax.text(r["accuracy"] + 0.002, yi, f"{r['accuracy']:.1%}", va="center", fontsize=8, color=INK)
    ax.axvline(baseline_acc, color=INK, lw=1.5, ls="--", label=f"Nobody defaults: {baseline_acc:.1%}")
    ax.set_yticks(y, [f"{r['label']} ({platforms[r['platform']]})" for r in ok], fontsize=8, color=INK)
    ax.set_xlim(min(r["accuracy"] for r in ok) - 0.03, 1.0)
    _style(ax, "Accuracy at 15.7 % flagged: every model is below doing nothing", "Accuracy (axis starts at %.0f %%)" % (ax.get_xlim()[0] * 100))
    _legend(ax, platforms, {r["platform"] for r in ok})
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def cv_vs_oot(cv_rows, test_ci, path):
    """Cross-validated PR-AUC (+/- 1 sd) next to the held-out PR-AUC (bootstrap 95 % CI)."""
    fig, ax = _fig(7.5, 4.8)
    x = np.arange(len(cv_rows))
    w = 0.36
    ax.bar(x - w / 2, [c["cv_mean"] for c in cv_rows], w, yerr=[c["cv_std"] for c in cv_rows], color="#9085e9",
           error_kw={"ecolor": INK, "capsize": 3}, label="Cross-validation, train window (+/- 1 sd)")
    pts = [c["pr_auc"] for c in cv_rows]
    err = np.array([[c["pr_auc"] - test_ci[c["model"]][0], test_ci[c["model"]][1] - c["pr_auc"]] for c in cv_rows]).T
    ax.bar(x + w / 2, pts, w, yerr=err, color="#2a78d6", error_kw={"ecolor": INK, "capsize": 3},
           label="Held-out TEST window (bootstrap 95 % CI)")
    for xi, c in zip(x, cv_rows):
        ax.text(xi, 0.01, f"gap {c['gap']:+.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [c["model"] for c in cv_rows], color=INK)
    _style(ax, "Validation flatters the trees more than the logistic model", None, "PR-AUC")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    ax.set_ylim(0, max(c["cv_mean"] + c["cv_std"] for c in cv_rows) * 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
