"""Publication-quality confusion-matrix heatmap.

Design choices for publication-quality reporting:
  - Two panels: counts (top) and row-normalized (bottom), share x-axis.
  - Off-white background, thin axes spines, Times-style serif text
    (matches common paper body fonts).
  - Annotated cells with adaptive text color (light text on dark cells, vice versa).
  - Sequential, perceptually-uniform colormap (`mako_r`); diagonal stands out
    naturally without being garish.
  - Footer reports overall accuracy and macro-F1.
  - Saved as both PDF (vector) and PNG (preview).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .labels import LEVEL1_LABELS

logger = logging.getLogger("remmd.plot")


# Times New Roman family chain. "Times New Roman" is preferred (and will be
# embedded when the PDF is later viewed on a system that has it); on Linux
# build hosts that lack it, matplotlib falls back to Liberation Serif (a
# metric-identical Red Hat replacement designed as a drop-in for Times New
# Roman) or Nimbus Roman (URW's Times equivalent) -- both keep glyph widths
# and line heights identical to Times New Roman.
_TIMES_FAMILY = [
    "Times New Roman",
    "Times",
    "Liberation Serif",
    "Nimbus Roman",
    "DejaVu Serif",
    "serif",
]

_PUB_RC = {
    "font.family": "serif",
    "font.serif": _TIMES_FAMILY,
    "mathtext.fontset": "stix",   # Times-compatible math glyphs
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,   # embed TrueType fonts for portable PDFs
    "ps.fonttype": 42,
}


def _annotate_heatmap(ax, data: np.ndarray, fmt: str, threshold: float):
    """Manually annotate cells with adaptive text color for readability."""
    nrow, ncol = data.shape
    for i in range(nrow):
        for j in range(ncol):
            val = data[i, j]
            color = "white" if val >= threshold else "#1d1d1d"
            ax.text(
                j + 0.5, i + 0.5,
                fmt.format(val),
                ha="center", va="center",
                color=color, fontsize=10,
            )


def plot_confusion_matrix_l1(
    cm: list[list[int]] | np.ndarray,
    *,
    accuracy: float,
    macro_f1: float,
    out_path_base: Path,
    title: str = "Level-1 Verdict Confusion Matrix",
    model_name: str = "",
    n_samples: int = 0,
) -> tuple[Path, Path]:
    """Render a two-panel figure (counts + row-normalized) and save PNG+PDF.

    Returns (png_path, pdf_path).
    """
    cm = np.asarray(cm, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square 2D; got {cm.shape}")
    n = cm.shape[0]
    # row-normalized
    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    with mpl.rc_context(_PUB_RC):
        fig, axes = plt.subplots(
            1, 2,
            figsize=(11.0, 4.6),
            gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.30},
        )

        # ------ panel A: raw counts ------
        ax = axes[0]
        cmap_counts = sns.color_palette("mako_r", as_cmap=True)
        vmax_c = max(cm.max(), 1)
        sns.heatmap(
            cm, ax=ax,
            cmap=cmap_counts, vmin=0, vmax=vmax_c,
            cbar=True, cbar_kws={"shrink": 0.85, "pad": 0.02, "label": "Count"},
            square=True, linewidths=0.6, linecolor="white",
            xticklabels=LEVEL1_LABELS, yticklabels=LEVEL1_LABELS,
            annot=False,
        )
        _annotate_heatmap(ax, cm.astype(float), "{:.0f}", threshold=0.55 * vmax_c)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("(a) Counts", loc="left", pad=8, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # ------ panel B: row-normalized ------
        ax = axes[1]
        cmap_norm = sns.color_palette("rocket_r", as_cmap=True)
        sns.heatmap(
            cm_norm, ax=ax,
            cmap=cmap_norm, vmin=0.0, vmax=1.0,
            cbar=True,
            cbar_kws={"shrink": 0.85, "pad": 0.02, "label": "Row-normalized"},
            square=True, linewidths=0.6, linecolor="white",
            xticklabels=LEVEL1_LABELS, yticklabels=LEVEL1_LABELS,
            annot=False,
        )
        _annotate_heatmap(ax, cm_norm, "{:.2f}", threshold=0.55)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("")
        ax.set_title("(b) Recall per class (row-normalized)", loc="left", pad=8, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # super-title and footer
        suptitle = title
        if model_name:
            suptitle = f"{title} — {model_name}"
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
        footer = (
            f"Accuracy = {accuracy*100:.2f}%   |   Macro-F1 = {macro_f1*100:.2f}%"
            + (f"   |   N = {n_samples}" if n_samples else "")
        )
        fig.text(0.5, -0.04, footer, ha="center", fontsize=10.5, color="#333333")

        out_path_base.parent.mkdir(parents=True, exist_ok=True)
        png = out_path_base.with_suffix(".png")
        pdf = out_path_base.with_suffix(".pdf")
        plt.savefig(png)
        plt.savefig(pdf)
        plt.close(fig)
    logger.info("saved heatmap: %s and %s", png, pdf)
    return png, pdf


def plot_level2_per_class_bars(
    per_class: dict[str, dict[str, float]],
    *,
    out_path_base: Path,
    title: str = "Level-2 per-class F1 (multi-label)",
    model_name: str = "",
) -> tuple[Path, Path]:
    """Bar chart of L2 per-class precision/recall/F1. Optional companion figure."""
    labels = list(per_class.keys())
    p = np.array([per_class[l]["precision"] for l in labels])
    r = np.array([per_class[l]["recall"] for l in labels])
    f = np.array([per_class[l]["f1"] for l in labels])

    with mpl.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(11.5, 5.2))
        x = np.arange(len(labels))
        w = 0.26
        palette = sns.color_palette("rocket", n_colors=3)
        ax.bar(x - w, p, w, label="Precision", color=palette[0])
        ax.bar(x,     r, w, label="Recall",    color=palette[1])
        ax.bar(x + w, f, w, label="F1",        color=palette[2])
        ax.set_xticks(x)
        # Two-line tick labels: short code on top, full name (line-broken) below
        tick_text = []
        for l in labels:
            parts = l.split(" ", 1)
            code = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            # split long names onto two visual lines for readability
            if len(name) > 14 and " " in name:
                cut = name.rfind(" ", 0, 14)
                if cut == -1:
                    cut = name.find(" ")
                name = name[:cut] + "\n" + name[cut + 1:]
            tick_text.append(f"{code}\n{name}")
        ax.set_xticklabels(tick_text, rotation=0, fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title((f"{title} — {model_name}" if model_name else title),
                     loc="left", fontweight="bold", pad=10)
        ax.legend(loc="upper right", frameon=False)
        ax.spines["bottom"].set_color("#444444")
        ax.tick_params(axis="x", which="both", length=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.subplots_adjust(bottom=0.20, top=0.90)

        out_path_base.parent.mkdir(parents=True, exist_ok=True)
        png = out_path_base.with_suffix(".png")
        pdf = out_path_base.with_suffix(".pdf")
        plt.savefig(png)
        plt.savefig(pdf)
        plt.close(fig)
    return png, pdf
