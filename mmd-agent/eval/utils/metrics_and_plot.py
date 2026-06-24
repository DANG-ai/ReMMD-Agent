"""指标计算 + 论文报告风格热力图。

模块分成两个部分：
1. 纯 Python 指标计算（5 分类 + 8 标签多标签），不依赖 sklearn。
2. 基于 matplotlib + seaborn 的高质量混淆矩阵热力图，配色 / 字体 / 排版
   均按高质量 AI/NLP 论文图表的常用样式设计。

设计原则（论文图表标准）：
- 字体：DejaVu Serif（系统通用衬线字体，确保跨平台一致）。
- 配色：使用 seaborn 的 'rocket_r' / 'crest' 等渐变色板，颜色饱满、打印友好。
- 注释：每个格子同时显示原始计数和行归一化百分比，避免阅读时再换算。
- 比例：方形画布、辅以 colorbar，比例适合双栏论文。
- 输出：同时保存 PDF（矢量，适合 LaTeX 嵌入）和 PNG（300 dpi，预览）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns


_BASE_FONT_FAMILY = "DejaVu Serif"


def _apply_publication_style() -> None:
    """统一字体 / 字号 / 线宽，适合论文报告中的高质量图表。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [_BASE_FONT_FAMILY, "Times New Roman", "Times", "serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.labelweight": "semibold",
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#222222",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# 5 分类指标
# ---------------------------------------------------------------------------

def compute_verdict_metrics_5cls(
    predictions: list[dict],
    label_order: list[str],
) -> dict:
    """计算 5 分类 verdict 的 accuracy / macro P/R/F1 / weighted F1 / 混淆矩阵 / per-class。"""
    y_true: list[str] = []
    y_pred: list[str] = []
    fallback_count = 0
    for p in predictions:
        gt = p.get("verdict_gt", "")
        pred = p.get("final_verdict", "")
        if gt not in label_order or pred not in label_order:
            continue
        y_true.append(gt)
        y_pred.append(pred)
        if p.get("fallback"):
            fallback_count += 1

    if not y_true:
        return {
            "total_samples": len(predictions),
            "evaluated_samples": 0,
            "fallback_samples": fallback_count,
            "note": "No samples with valid GT/pred for the 5-class metric.",
        }

    confusion = {gt: {pr: 0 for pr in label_order} for gt in label_order}
    for gt, pr in zip(y_true, y_pred):
        confusion[gt][pr] += 1

    per_class: dict[str, dict[str, float]] = {}
    macro_p = macro_r = macro_f1 = 0.0
    total_correct = 0
    for label in label_order:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in label_order if other != label)
        fn = sum(confusion[label][other] for other in label_order if other != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        total_correct += tp

    n_classes = len(label_order)
    macro_p /= n_classes
    macro_r /= n_classes
    macro_f1 /= n_classes
    accuracy = total_correct / len(y_true)

    weighted_p = weighted_r = weighted_f1 = 0.0
    total_support = len(y_true)
    if total_support > 0:
        for label in label_order:
            w = per_class[label]["support"] / total_support
            weighted_p += per_class[label]["precision"] * w
            weighted_r += per_class[label]["recall"] * w
            weighted_f1 += per_class[label]["f1"] * w

    pred_dist = {label: int(y_pred.count(label)) for label in label_order}
    gt_dist = {label: int(y_true.count(label)) for label in label_order}

    confusion_matrix = [[confusion[gt][pr] for pr in label_order] for gt in label_order]

    return {
        "total_samples": len(predictions),
        "evaluated_samples": len(y_true),
        "fallback_samples": fallback_count,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": label_order, "matrix": confusion_matrix},
        "pred_distribution": pred_dist,
        "gt_distribution": gt_dist,
    }


# ---------------------------------------------------------------------------
# 8 标签多标签指标
# ---------------------------------------------------------------------------

def compute_distortion_metrics_8label(
    predictions: list[dict],
    label_order: list[str],
) -> dict:
    """计算 8 标签多标签 distortion 的 precision / recall / macro F1 / exact match。"""
    y_true: list[dict[str, int]] = []
    y_pred: list[dict[str, int]] = []
    for p in predictions:
        gt_vector = p.get("distortion_pred_vs_gt", {}).get("gt_vector")
        pred_vector = p.get("distortion_pred_vs_gt", {}).get("pred_vector")
        if gt_vector is None or pred_vector is None:
            continue
        y_true.append(gt_vector)
        y_pred.append(pred_vector)

    if not y_true:
        return {
            "total_samples": len(predictions),
            "evaluated_samples": 0,
            "note": "No samples with valid GT/pred for the 8-label metric.",
        }

    per_label: dict[str, dict[str, float]] = {}
    macro_p = macro_r = macro_f1 = 0.0
    micro_tp = micro_fp = micro_fn = 0
    exact_match = 0
    hamming_correct = 0
    hamming_total = 0

    for gt_vec, pr_vec in zip(y_true, y_pred):
        if all(gt_vec[label] == pr_vec[label] for label in label_order):
            exact_match += 1
        for label in label_order:
            hamming_total += 1
            if gt_vec[label] == pr_vec[label]:
                hamming_correct += 1

    for label in label_order:
        tp = sum(1 for gt, pr in zip(y_true, y_pred) if gt[label] == 1 and pr[label] == 1)
        fp = sum(1 for gt, pr in zip(y_true, y_pred) if gt[label] == 0 and pr[label] == 1)
        fn = sum(1 for gt, pr in zip(y_true, y_pred) if gt[label] == 1 and pr[label] == 0)
        support = sum(1 for gt in y_true if gt[label] == 1)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    n_labels = len(label_order)
    macro_p /= n_labels
    macro_r /= n_labels
    macro_f1 /= n_labels
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return {
        "total_samples": len(predictions),
        "evaluated_samples": len(y_true),
        "exact_match_ratio": round(exact_match / len(y_true), 4),
        "exact_match_count": int(exact_match),
        "hamming_accuracy": round(hamming_correct / hamming_total, 4) if hamming_total else 0.0,
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "per_label": per_label,
        "labels": label_order,
    }


# ---------------------------------------------------------------------------
# 混淆矩阵热力图（论文图表标准）
# ---------------------------------------------------------------------------

def plot_confusion_matrix_heatmap(
    matrix: list[list[int]] | np.ndarray,
    labels: list[str],
    out_path_pdf: Path,
    out_path_png: Path,
    title: str = "5-Class Verdict Confusion Matrix",
    model_name: str = "",
    normalize: str = "row",
    cmap: str = "rocket_r",
    show_diag_emphasis: bool = True,
) -> None:
    """绘制顶级会议风格的混淆矩阵热力图。

    Parameters
    ----------
    matrix : 二维计数矩阵（rows = ground truth, cols = prediction）。
    labels : 标签列表（行列同序）。
    out_path_pdf / out_path_png : 输出路径。
    title : 图标题。
    model_name : 副标题，注明模型名称。
    normalize : "row" -> 按行（GT）归一化；"col" -> 按列；"none" -> 原始计数。
    cmap : seaborn 调色板名称。'rocket_r' 是顶级会议常用的深蓝-黄渐变。
    show_diag_emphasis : 对角线以更粗边框强调正确预测。
    """
    _apply_publication_style()

    cm = np.asarray(matrix, dtype=float)
    n = cm.shape[0]
    assert cm.shape == (n, n), "confusion matrix must be square"
    assert len(labels) == n, "labels length mismatch"

    if normalize == "row":
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        annot = np.empty_like(cm, dtype=object)
        for i in range(n):
            for j in range(n):
                count = int(cm[i, j])
                pct = cm_norm[i, j] * 100.0
                annot[i, j] = f"{count}\n{pct:.1f}%"
        heat = cm_norm
        cbar_label = "Row-normalized rate"
        vmin, vmax = 0.0, 1.0
    elif normalize == "col":
        col_sum = cm.sum(axis=0, keepdims=True)
        cm_norm = np.divide(cm, col_sum, out=np.zeros_like(cm), where=col_sum > 0)
        annot = np.empty_like(cm, dtype=object)
        for i in range(n):
            for j in range(n):
                count = int(cm[i, j])
                pct = cm_norm[i, j] * 100.0
                annot[i, j] = f"{count}\n{pct:.1f}%"
        heat = cm_norm
        cbar_label = "Column-normalized rate"
        vmin, vmax = 0.0, 1.0
    else:
        annot = np.array([[str(int(v)) for v in row] for row in cm], dtype=object)
        heat = cm
        cbar_label = "Count"
        vmin, vmax = 0.0, float(cm.max() if cm.size else 1.0) or 1.0

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * n + 3.5), max(5.2, 0.9 * n + 2.8)))

    sns.heatmap(
        heat,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.6,
        linecolor="#FFFFFF",
        cbar_kws={"shrink": 0.78, "pad": 0.02, "label": cbar_label, "aspect": 22},
        annot_kws={"fontsize": 11, "fontweight": "semibold"},
        ax=ax,
    )

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.outline.set_linewidth(0.8)
        cbar.outline.set_edgecolor("#222222")
        cbar.ax.tick_params(width=0.8)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels, rotation=0)

    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("Ground-truth label", labelpad=10)

    full_title = title
    if model_name:
        full_title = f"{title}\n\\textit{{Model: {model_name}}}".replace("\\textit", "").replace("{", "").replace("}", "")
        full_title = f"{title}\n(Model: {model_name})"
    ax.set_title(full_title, pad=14, fontweight="bold")

    for i in range(n):
        for j in range(n):
            cell_norm = heat[i, j]
            txt = ax.texts[i * n + j]
            if vmax > 0 and (cell_norm / vmax) > 0.55:
                txt.set_color("white")
            else:
                txt.set_color("#1a1a1a")

    if show_diag_emphasis:
        for i in range(n):
            rect = matplotlib.patches.Rectangle(
                (i, i), 1, 1,
                fill=False,
                edgecolor="#1f77b4",
                linewidth=1.8,
            )
            ax.add_patch(rect)

    plt.tight_layout()
    out_path_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_pdf, format="pdf")
    plt.savefig(out_path_png, format="png")
    plt.close(fig)


def plot_per_label_bar(
    metric_values: dict[str, dict[str, float]],
    label_order: list[str],
    out_path_pdf: Path,
    out_path_png: Path,
    title: str,
    model_name: str = "",
    metric_keys: tuple[str, ...] = ("precision", "recall", "f1"),
    palette: tuple[str, ...] = ("#4C72B0", "#DD8452", "#55A868"),
) -> None:
    """画 per-label 的 P/R/F1 柱状图（用于 8 标签多标签）。

    与混淆矩阵图配套，方便论文中并排展示。
    采用双栏可读尺寸 + 顶部 legend，避免标签重叠。
    """
    _apply_publication_style()

    n_metrics = len(metric_keys)
    n_labels = len(label_order)
    values = np.zeros((n_metrics, n_labels), dtype=float)
    for j, lab in enumerate(label_order):
        stats = metric_values.get(lab, {})
        for i, mk in enumerate(metric_keys):
            values[i, j] = float(stats.get(mk, 0.0))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.95 * n_labels + 3.5), 5.6))
    bar_width = 0.78 / n_metrics
    x = np.arange(n_labels)

    bars_list = []
    for i, mk in enumerate(metric_keys):
        offset = (i - (n_metrics - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values[i],
            width=bar_width,
            label=mk.replace("_", " ").title(),
            color=palette[i % len(palette)],
            edgecolor="white",
            linewidth=0.7,
        )
        bars_list.append(bars)

    for bars in bars_list:
        for rect in bars:
            h = rect.get_height()
            if h > 0.01:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h + 0.015,
                    f"{h:.2f}",
                    ha="center", va="bottom",
                    fontsize=9,
                    color="#1a1a1a",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel("Score", labelpad=8)
    full_title = title if not model_name else f"{title}\n(Model: {model_name})"
    ax.set_title(full_title, pad=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper right",
        ncol=n_metrics,
        frameon=True,
        framealpha=0.92,
        edgecolor="#444444",
        handlelength=1.5,
        fontsize=10,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.subplots_adjust(bottom=0.28, top=0.88, left=0.10, right=0.98)
    out_path_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_pdf, format="pdf")
    plt.savefig(out_path_png, format="png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 终端 + 文本格式化总结报告
# ---------------------------------------------------------------------------

def _format_kv(key: str, value: float | int | str, width: int = 22) -> str:
    return f"  {key:<{width}}: {value}"


def render_summary_report(
    verdict_metrics: dict,
    distortion_metrics: dict,
    model_name: str = "",
    run_tag: str = "",
) -> str:
    """生成一份纯文本的 eval summary，既用于 print 也用于落盘 .txt 文件。"""
    lines: list[str] = []
    sep = "=" * 78
    lines.append(sep)
    lines.append(f"  MMD-Agent  Evaluation Summary")
    if model_name:
        lines.append(f"  Model    : {model_name}")
    if run_tag:
        lines.append(f"  Run Tag  : {run_tag}")
    lines.append(sep)

    lines.append("")
    lines.append("[ 5-Class Verdict ]")
    lines.append("-" * 78)
    if "accuracy" in verdict_metrics:
        lines.append(_format_kv("Total samples", verdict_metrics["total_samples"]))
        lines.append(_format_kv("Evaluated", verdict_metrics["evaluated_samples"]))
        lines.append(_format_kv("Fallback", verdict_metrics.get("fallback_samples", 0)))
        lines.append(_format_kv("Accuracy", f"{verdict_metrics['accuracy']:.4f}"))
        lines.append(_format_kv("Macro-Precision", f"{verdict_metrics['macro_precision']:.4f}"))
        lines.append(_format_kv("Macro-Recall", f"{verdict_metrics['macro_recall']:.4f}"))
        lines.append(_format_kv("Macro-F1", f"{verdict_metrics['macro_f1']:.4f}"))
        lines.append(_format_kv("Weighted-F1", f"{verdict_metrics['weighted_f1']:.4f}"))

        lines.append("")
        lines.append("  Per-class:")
        header = f"    {'Label':<16s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}  {'Supp':>6s}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for label, stats in verdict_metrics["per_class"].items():
            lines.append(
                f"    {label:<16s}  {stats['precision']:>7.4f}  {stats['recall']:>7.4f}  "
                f"{stats['f1']:>7.4f}  {stats['support']:>6d}"
            )

        cm = verdict_metrics["confusion_matrix"]
        lines.append("")
        lines.append("  Confusion Matrix (rows=GT, cols=Pred):")
        header = "    " + " " * 16 + " ".join(f"{l[:14]:>14s}" for l in cm["labels"])
        lines.append(header)
        for label, row in zip(cm["labels"], cm["matrix"]):
            lines.append(f"    {label:<16s}" + " ".join(f"{v:>14d}" for v in row))
    else:
        lines.append("  " + verdict_metrics.get("note", "No 5-class metrics."))

    lines.append("")
    lines.append("[ 8-Label Distortion Taxonomy (multi-label) ]")
    lines.append("-" * 78)
    if "macro_f1" in distortion_metrics:
        lines.append(_format_kv("Total samples", distortion_metrics["total_samples"]))
        lines.append(_format_kv("Evaluated", distortion_metrics["evaluated_samples"]))
        lines.append(_format_kv("Exact-Match Ratio", f"{distortion_metrics['exact_match_ratio']:.4f}"))
        lines.append(_format_kv("Exact-Match Count", distortion_metrics["exact_match_count"]))
        lines.append(_format_kv("Hamming Accuracy", f"{distortion_metrics['hamming_accuracy']:.4f}"))
        lines.append(_format_kv("Macro-Precision", f"{distortion_metrics['macro_precision']:.4f}"))
        lines.append(_format_kv("Macro-Recall", f"{distortion_metrics['macro_recall']:.4f}"))
        lines.append(_format_kv("Macro-F1", f"{distortion_metrics['macro_f1']:.4f}"))
        lines.append(_format_kv("Micro-Precision", f"{distortion_metrics['micro_precision']:.4f}"))
        lines.append(_format_kv("Micro-Recall", f"{distortion_metrics['micro_recall']:.4f}"))
        lines.append(_format_kv("Micro-F1", f"{distortion_metrics['micro_f1']:.4f}"))

        lines.append("")
        lines.append("  Per-label:")
        header = f"    {'Label':<32s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}  {'Supp':>6s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for label in distortion_metrics["labels"]:
            stats = distortion_metrics["per_label"][label]
            lines.append(
                f"    {label:<32s}  {stats['precision']:>7.4f}  {stats['recall']:>7.4f}  "
                f"{stats['f1']:>7.4f}  {stats['support']:>6d}  {stats['tp']:>5d}  {stats['fp']:>5d}  {stats['fn']:>5d}"
            )
    else:
        lines.append("  " + distortion_metrics.get("note", "No 8-label metrics."))

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def save_eval_summary(
    out_dir: Path,
    verdict_metrics: dict,
    distortion_metrics: dict,
    model_name: str,
    run_tag: str,
    verdict_label_order: list[str],
    distortion_label_order: list[str],
) -> dict[str, str]:
    """落盘一次完整的 eval summary：JSON + TXT + 两张图。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = out_dir / "metrics.json"
    metrics_txt_path = out_dir / "eval_summary.txt"
    cm_pdf_path = out_dir / "confusion_matrix_verdict_5cls.pdf"
    cm_png_path = out_dir / "confusion_matrix_verdict_5cls.png"
    bar_pdf_path = out_dir / "per_label_bar_distortion_8label.pdf"
    bar_png_path = out_dir / "per_label_bar_distortion_8label.png"

    combined = {
        "verdict_5cls": verdict_metrics,
        "distortion_8label_multi": distortion_metrics,
        "verdict_label_order": verdict_label_order,
        "distortion_label_order": distortion_label_order,
        "model_name": model_name,
        "run_tag": run_tag,
    }
    metrics_json_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text_report = render_summary_report(
        verdict_metrics, distortion_metrics, model_name=model_name, run_tag=run_tag,
    )
    metrics_txt_path.write_text(text_report, encoding="utf-8")

    if "confusion_matrix" in verdict_metrics:
        cm = verdict_metrics["confusion_matrix"]
        try:
            plot_confusion_matrix_heatmap(
                matrix=cm["matrix"],
                labels=cm["labels"],
                out_path_pdf=cm_pdf_path,
                out_path_png=cm_png_path,
                title="5-Class Verdict — Confusion Matrix on ReMMDBench",
                model_name=model_name,
                normalize="row",
                cmap="rocket_r",
                show_diag_emphasis=True,
            )
        except Exception as exc:  # 绘图失败不应阻断 metrics 落盘
            print(f"[WARN] confusion matrix heatmap failed: {exc}", flush=True)

    if "per_label" in distortion_metrics:
        try:
            plot_per_label_bar(
                metric_values=distortion_metrics["per_label"],
                label_order=distortion_metrics["labels"],
                out_path_pdf=bar_pdf_path,
                out_path_png=bar_png_path,
                title="8-Label Distortion Taxonomy — Per-Label Scores",
                model_name=model_name,
                metric_keys=("precision", "recall", "f1"),
            )
        except Exception as exc:
            print(f"[WARN] per-label bar plot failed: {exc}", flush=True)

    return {
        "metrics_json": str(metrics_json_path),
        "metrics_txt": str(metrics_txt_path),
        "confusion_matrix_pdf": str(cm_pdf_path),
        "confusion_matrix_png": str(cm_png_path),
        "per_label_bar_pdf": str(bar_pdf_path),
        "per_label_bar_png": str(bar_png_path),
        "text_report": text_report,
    }
