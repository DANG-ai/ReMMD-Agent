"""Publication-quality plotting for ReMMDBench evaluation metrics.

These plots are intended for direct inclusion in paper reports, so the styling
deliberately follows the typography and colour-palette conventions commonly
used by high-quality AI/NLP figures:

* Serif (Times-style) typography with a clean white background.
* Sequential colormap with strong contrast on the diagonal.
* Cell-by-cell annotations that combine the absolute count and the row-
  normalised percentage.
* High-resolution PNG and a matched vector PDF for every figure.

The functions in this module never raise: any matplotlib / seaborn issue is
caught and reported as a string so a metric run never blocks because of a
plotting problem.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .labels import (
    REALMMDBENCH_TAXONOMY_LABELS,
    REALMMDBENCH_VERDICT_LABELS,
)


_VERDICT_SHORT_LABELS = {
    "True": "True",
    "Mostly True": "Mostly\nTrue",
    "Mixture": "Mixture",
    "Mostly False": "Mostly\nFalse",
    "False": "False",
    "ERROR": "ERROR",
}


_TAXONOMY_SHORT_LABELS = {
    "T1 Fabrication": "T1\nFabrication",
    "T2 Distortion": "T2\nDistortion",
    "T3 Misleading Context": "T3\nMisleading\nContext",
    "V1 Synthetic Visual Content": "V1\nSynthetic\nVisual",
    "V2 Visual Editing": "V2\nVisual\nEditing",
    "C1 Semantic Inconsistency": "C1\nSemantic\nInconsist.",
    "C2 Contextual Inconsistency": "C2\nContextual\nInconsist.",
    "C3 Pragmatic Inconsistency": "C3\nPragmatic\nInconsist.",
}


def _apply_publication_style() -> None:
    """Set a global matplotlib style suitable for publication figures."""

    plt.rcdefaults()
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
                "Times New Roman",
                "Times",
            ],
            "mathtext.fontset": "cm",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 360,
            "figure.dpi": 144,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.unicode_minus": False,
        }
    )


def _save_figure(fig: "plt.Figure", base_path: Path) -> list[str]:
    """Persist a figure to both PNG (raster) and PDF (vector) and return paths."""

    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for suffix in (".png", ".pdf"):
        out = base_path.with_suffix(suffix)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        saved.append(str(out))
    return saved


def _annotate_cells(
    ax: "plt.Axes",
    counts: np.ndarray,
    row_normalized: np.ndarray,
    *,
    color_threshold: float,
) -> None:
    """Draw the absolute counts + row-normalised percentages in each cell."""

    rows, cols = counts.shape
    for r in range(rows):
        for c in range(cols):
            count_value = int(counts[r, c])
            pct_value = float(row_normalized[r, c])
            text_color = "white" if pct_value >= color_threshold else "#111111"
            primary = f"{count_value}"
            secondary = f"{pct_value * 100:.1f}%"
            ax.text(
                c + 0.5,
                r + 0.42,
                primary,
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )
            ax.text(
                c + 0.5,
                r + 0.70,
                secondary,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontstyle="italic",
                alpha=0.95,
            )


def plot_verdict_confusion_matrix(
    confusion_matrix: dict[str, Any],
    output_dir: Path,
    *,
    title: str = "5-way Verdict Confusion Matrix",
    file_stem: str = "verdict_confusion_matrix",
) -> dict[str, Any]:
    """Render the 5-way verdict confusion matrix as a publication heatmap."""

    try:
        _apply_publication_style()
        row_labels: Sequence[str] = confusion_matrix["row_labels"]
        col_labels: Sequence[str] = confusion_matrix["col_labels"]
        counts = np.array(confusion_matrix["counts"], dtype=float)
        row_normalized = np.array(confusion_matrix["row_normalized"], dtype=float)

        x_labels = [_VERDICT_SHORT_LABELS.get(label, label) for label in col_labels]
        y_labels = [_VERDICT_SHORT_LABELS.get(label, label) for label in row_labels]

        fig_width = max(5.0, 0.95 * len(col_labels) + 2.0)
        fig_height = max(4.5, 0.85 * len(row_labels) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        cmap = sns.color_palette("mako_r", as_cmap=True)
        heatmap = ax.imshow(
            row_normalized,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            extent=(0, len(col_labels), len(row_labels), 0),
        )
        _annotate_cells(ax, counts, row_normalized, color_threshold=0.55)

        ax.set_xticks(np.arange(len(col_labels)) + 0.5)
        ax.set_yticks(np.arange(len(row_labels)) + 0.5)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_yticklabels(y_labels, fontsize=10, rotation=0)
        ax.set_xlabel("Predicted label", fontweight="bold")
        ax.set_ylabel("Ground-truth label", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=12)

        ax.set_xticks(np.arange(len(col_labels) + 1), minor=True)
        ax.set_yticks(np.arange(len(row_labels) + 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(axis="x", which="major", length=0, pad=4)
        ax.tick_params(axis="y", which="major", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row-normalised proportion", rotation=270, labelpad=14)
        cbar.outline.set_visible(False)

        fig.tight_layout()
        saved = _save_figure(fig, output_dir / file_stem)
        plt.close(fig)
        return {"status": "ok", "files": saved}
    except Exception as error:  # noqa: BLE001
        plt.close("all")
        return {"status": "error", "error": repr(error)}


def plot_verdict_per_label_bar(
    per_label: list[dict[str, Any]],
    output_dir: Path,
    *,
    title: str = "5-way Verdict Per-class Precision / Recall / F1",
    file_stem: str = "verdict_per_label_bar",
) -> dict[str, Any]:
    """Grouped bar chart of per-class P / R / F1 for the 5-way verdict task."""

    try:
        _apply_publication_style()
        labels = [item["label"] for item in per_label]
        precisions = [item["precision"] for item in per_label]
        recalls = [item["recall"] for item in per_label]
        f1s = [item["f1"] for item in per_label]
        short_labels = [_VERDICT_SHORT_LABELS.get(label, label) for label in labels]

        x = np.arange(len(labels))
        width = 0.26
        palette = sns.color_palette("crest", n_colors=3)
        fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(labels) + 2.0), 4.2))
        ax.bar(x - width, precisions, width, label="Precision", color=palette[0])
        ax.bar(x, recalls, width, label="Recall", color=palette[1])
        ax.bar(x + width, f1s, width, label="F1", color=palette[2])
        for idx, value in enumerate(precisions):
            ax.text(
                idx - width,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222",
            )
        for idx, value in enumerate(recalls):
            ax.text(idx, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color="#222")
        for idx, value in enumerate(f1s):
            ax.text(idx + width, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color="#222")

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle=":", alpha=0.55, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        saved = _save_figure(fig, output_dir / file_stem)
        plt.close(fig)
        return {"status": "ok", "files": saved}
    except Exception as error:  # noqa: BLE001
        plt.close("all")
        return {"status": "error", "error": repr(error)}


def plot_taxonomy_per_label_bar(
    per_label: list[dict[str, Any]],
    output_dir: Path,
    *,
    title: str = "8-way Distortion Taxonomy Per-label Precision / Recall / F1",
    file_stem: str = "taxonomy_per_label_bar",
) -> dict[str, Any]:
    """Grouped bar chart of per-label P / R / F1 for the multi-label task."""

    try:
        _apply_publication_style()
        labels = [item["label"] for item in per_label]
        precisions = [item["precision"] for item in per_label]
        recalls = [item["recall"] for item in per_label]
        f1s = [item["f1"] for item in per_label]
        short_labels = [_TAXONOMY_SHORT_LABELS.get(label, label) for label in labels]

        x = np.arange(len(labels))
        width = 0.26
        palette = sns.color_palette("rocket_r", n_colors=3)
        fig, ax = plt.subplots(figsize=(max(7.0, 1.0 * len(labels) + 2.5), 4.6))
        ax.bar(x - width, precisions, width, label="Precision", color=palette[0])
        ax.bar(x, recalls, width, label="Recall", color=palette[1])
        ax.bar(x + width, f1s, width, label="F1", color=palette[2])
        for idx, value in enumerate(precisions):
            ax.text(idx - width, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5, color="#222")
        for idx, value in enumerate(recalls):
            ax.text(idx, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5, color="#222")
        for idx, value in enumerate(f1s):
            ax.text(idx + width, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7.5, color="#222")

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8.5)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle=":", alpha=0.55, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        saved = _save_figure(fig, output_dir / file_stem)
        plt.close(fig)
        return {"status": "ok", "files": saved}
    except Exception as error:  # noqa: BLE001
        plt.close("all")
        return {"status": "error", "error": repr(error)}


def _build_taxonomy_label_alignment_matrix(
    records: list[dict[str, Any]],
    *,
    ground_truth_key: str = "ground_truth",
    predicted_key: str = "predicted",
) -> dict[str, Any]:
    """Sample-level label alignment between predictions and ground truth.

    For every sample we build a binary indicator vector over the 8 labels for
    both prediction and ground truth, then average ``pred[i] * gt[j]`` across
    samples to get an 8x8 matrix where the diagonal carries the per-label
    co-occurrence (jointly predicted and gold). The matrix is also returned in
    row-normalised form so each row sums to 1 (or 0 when the label is unused).
    """

    from .labels import normalize_taxonomy_labels, is_error_prediction

    labels = list(REALMMDBENCH_TAXONOMY_LABELS)
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    valid = 0
    for record in records:
        pred_raw = record.get(predicted_key)
        gt_raw = record.get(ground_truth_key)
        if is_error_prediction(pred_raw):
            continue
        pred_set = set(normalize_taxonomy_labels(pred_raw))
        gt_set = set(normalize_taxonomy_labels(gt_raw))
        if not gt_set and not pred_set:
            continue
        valid += 1
        for r, gt_label in enumerate(labels):
            for c, pred_label in enumerate(labels):
                if gt_label in gt_set and pred_label in pred_set:
                    matrix[r, c] += 1

    row_norm = np.zeros_like(matrix)
    for r in range(matrix.shape[0]):
        row_sum = matrix[r].sum()
        if row_sum > 0:
            row_norm[r] = matrix[r] / row_sum

    return {
        "labels": labels,
        "counts": matrix.tolist(),
        "row_normalized": row_norm.tolist(),
        "valid_samples": valid,
    }


def plot_taxonomy_label_alignment_heatmap(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    title: str = "Multi-label Distortion Taxonomy Alignment",
    file_stem: str = "taxonomy_label_alignment_heatmap",
    ground_truth_key: str = "ground_truth",
    predicted_key: str = "predicted",
) -> dict[str, Any]:
    """Heatmap of (ground-truth label x predicted label) co-occurrence.

    Because the taxonomy task is multi-label, a strict confusion matrix is not
    well-defined. The alignment heatmap instead shows, for every ground-truth
    label *l*, the row-normalised distribution of predicted labels that
    co-occur with *l* across the samples. The diagonal therefore represents
    correct per-label co-prediction; off-diagonal cells reveal common
    confusions.
    """

    try:
        _apply_publication_style()
        matrix_payload = _build_taxonomy_label_alignment_matrix(
            records,
            ground_truth_key=ground_truth_key,
            predicted_key=predicted_key,
        )
        labels = matrix_payload["labels"]
        counts = np.array(matrix_payload["counts"], dtype=float)
        row_normalized = np.array(matrix_payload["row_normalized"], dtype=float)

        short_labels = [_TAXONOMY_SHORT_LABELS.get(label, label) for label in labels]
        fig, ax = plt.subplots(figsize=(8.8, 7.4))
        cmap = sns.color_palette("rocket_r", as_cmap=True)
        heatmap = ax.imshow(
            row_normalized,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            extent=(0, len(labels), len(labels), 0),
        )
        _annotate_cells(ax, counts, row_normalized, color_threshold=0.5)

        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(short_labels, fontsize=8.5)
        ax.set_yticklabels(short_labels, fontsize=8.5, rotation=0)
        ax.set_xlabel("Predicted label", fontweight="bold")
        ax.set_ylabel("Ground-truth label", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=12)

        ax.set_xticks(np.arange(len(labels) + 1), minor=True)
        ax.set_yticks(np.arange(len(labels) + 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(axis="x", which="major", length=0, pad=4)
        ax.tick_params(axis="y", which="major", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row-normalised co-occurrence", rotation=270, labelpad=14)
        cbar.outline.set_visible(False)

        fig.tight_layout()
        saved = _save_figure(fig, output_dir / file_stem)
        plt.close(fig)
        return {
            "status": "ok",
            "files": saved,
            "valid_samples": matrix_payload["valid_samples"],
            "counts": matrix_payload["counts"],
            "row_normalized": matrix_payload["row_normalized"],
        }
    except Exception as error:  # noqa: BLE001
        plt.close("all")
        return {"status": "error", "error": repr(error)}


__all__ = [
    "plot_verdict_confusion_matrix",
    "plot_verdict_per_label_bar",
    "plot_taxonomy_per_label_bar",
    "plot_taxonomy_label_alignment_heatmap",
]
