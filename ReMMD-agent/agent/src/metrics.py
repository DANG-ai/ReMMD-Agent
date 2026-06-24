"""Evaluation metrics for L1 (5-way) and L2 (multi-label 8).

Level-1 metrics:
  • Accuracy.
  • Macro/Micro/Weighted Precision/Recall/F1.
  • Per-class Precision/Recall/F1 with support.
  • 5x5 confusion matrix.

Level-2 metrics (the 8-way multi-label task):
  • Macro Precision / Recall / F1  — average of per-class scores
                                     (each class weighted equally).
  • Micro Precision / Recall / F1  — global TP/FP/FN aggregation
                                     (each prediction weighted equally).
  • Weighted Precision / Recall / F1 — average weighted by support.
  • Samples Precision / Recall / F1 — average over samples.
  • Hamming loss + Subset (exact-match) accuracy.
  • Per-class Precision / Recall / F1 with support and prediction-positives.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from .labels import LEVEL1_LABELS, LEVEL2_LABELS, normalize_level1, normalize_level2_list


def _collect_pairs(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter to results that have both a parseable prediction and a gold label."""
    paired = []
    skipped = []
    for r in results:
        if r.get("status") != "ok":
            skipped.append({"sample_id": r.get("sample_id"), "reason": r.get("status")})
            continue
        gold_v = normalize_level1((r.get("gold") or {}).get("verdict"))
        pred_v = normalize_level1((r.get("judge") or {}).get("level1_verdict"))
        if gold_v is None or pred_v is None:
            skipped.append({
                "sample_id": r.get("sample_id"),
                "reason": "missing_label",
                "gold_v": gold_v, "pred_v": pred_v,
            })
            continue
        gold_l2 = normalize_level2_list((r.get("gold") or {}).get("distortion_taxonomy", []))
        pred_l2 = normalize_level2_list((r.get("judge") or {}).get("level2_taxonomy", []))
        paired.append({
            "sample_id": r.get("sample_id"),
            "gold_v": gold_v, "pred_v": pred_v,
            "gold_l2": gold_l2, "pred_l2": pred_l2,
        })
    return paired, skipped


def _binarize(labels: list[list[str]]) -> np.ndarray:
    Y = np.zeros((len(labels), len(LEVEL2_LABELS)), dtype=np.int32)
    idx = {l: i for i, l in enumerate(LEVEL2_LABELS)}
    for r, row in enumerate(labels):
        for lbl in row:
            if lbl in idx:
                Y[r, idx[lbl]] = 1
    return Y


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    paired, skipped = _collect_pairs(results)
    n = len(paired)
    out: dict[str, Any] = {
        "n_total_results": len(results),
        "n_eligible_for_eval": n,
        "n_skipped": len(skipped),
        "skipped_reasons": Counter([s.get("reason") for s in skipped]),
        "labels_l1": LEVEL1_LABELS,
        "labels_l2": LEVEL2_LABELS,
    }
    if n == 0:
        out["error"] = "no eligible results"
        return out

    # ---- L1: 5-class single label ----
    y_true = [p["gold_v"] for p in paired]
    y_pred = [p["pred_v"] for p in paired]
    cm = confusion_matrix(y_true, y_pred, labels=LEVEL1_LABELS).tolist()
    out["level1"] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Macro = each class weighted equally (matches the user's required
        # 5-way / 8-way summary metrics).
        "macro_precision": float(precision_score(y_true, y_pred, labels=LEVEL1_LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LEVEL1_LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LEVEL1_LABELS, average="macro", zero_division=0)),
        # Micro = global TP/FP/FN aggregation (≡ accuracy for a single-label
        # 5-way task; included for symmetry with L2 and for correctness checks).
        "micro_precision": float(precision_score(y_true, y_pred, labels=LEVEL1_LABELS, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, labels=LEVEL1_LABELS, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=LEVEL1_LABELS, average="micro", zero_division=0)),
        # Weighted = per-class scores weighted by support.
        "weighted_precision": float(precision_score(y_true, y_pred, labels=LEVEL1_LABELS, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=LEVEL1_LABELS, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LEVEL1_LABELS, average="weighted", zero_division=0)),
        "confusion_matrix": cm,  # rows=true, cols=pred (in LEVEL1_LABELS order)
        "per_class": {},
    }
    p_cls = precision_score(y_true, y_pred, labels=LEVEL1_LABELS, average=None, zero_division=0)
    r_cls = recall_score(y_true, y_pred, labels=LEVEL1_LABELS, average=None, zero_division=0)
    f_cls = f1_score(y_true, y_pred, labels=LEVEL1_LABELS, average=None, zero_division=0)
    support = Counter(y_true)
    for i, lbl in enumerate(LEVEL1_LABELS):
        out["level1"]["per_class"][lbl] = {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f_cls[i]),
            "support": int(support.get(lbl, 0)),
        }

    # ---- L2: multi-label 8-way ----
    Y_true = _binarize([p["gold_l2"] for p in paired])
    Y_pred = _binarize([p["pred_l2"] for p in paired])
    # Subset (exact-match) accuracy: prediction matches gold on EVERY one of
    # the 8 binary labels for that sample.
    exact_match = float((Y_true == Y_pred).all(axis=1).mean())
    out["level2"] = {
        # ----- the user's required summary trio: macro P / R / F1 -----
        "macro_precision": float(precision_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        # Micro / weighted / samples-averaged P/R/F1 round out the
        # multi-label picture.
        "micro_precision": float(precision_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "weighted_precision": float(precision_score(Y_true, Y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(Y_true, Y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(Y_true, Y_pred, average="weighted", zero_division=0)),
        "samples_precision": float(precision_score(Y_true, Y_pred, average="samples", zero_division=0)),
        "samples_recall": float(recall_score(Y_true, Y_pred, average="samples", zero_division=0)),
        "samples_f1": float(f1_score(Y_true, Y_pred, average="samples", zero_division=0)),
        # Hamming loss = fraction of labels (over n_samples * n_classes) that
        # were predicted incorrectly. Lower is better. Together with subset
        # (exact-match) accuracy it pins down multi-label performance.
        "hamming_loss": float(hamming_loss(Y_true, Y_pred)),
        "exact_match": exact_match,
        "subset_accuracy": exact_match,  # alias to make the metric name unambiguous
        "per_class": {},
    }
    p_cls2 = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    r_cls2 = recall_score(Y_true, Y_pred, average=None, zero_division=0)
    f_cls2 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    for i, lbl in enumerate(LEVEL2_LABELS):
        out["level2"]["per_class"][lbl] = {
            "precision": float(p_cls2[i]),
            "recall": float(r_cls2[i]),
            "f1": float(f_cls2[i]),
            "support": int(Y_true[:, i].sum()),
            "pred_positives": int(Y_pred[:, i].sum()),
            "true_positives": int(((Y_true[:, i] == 1) & (Y_pred[:, i] == 1)).sum()),
            "false_positives": int(((Y_true[:, i] == 0) & (Y_pred[:, i] == 1)).sum()),
            "false_negatives": int(((Y_true[:, i] == 1) & (Y_pred[:, i] == 0)).sum()),
        }

    out["pairs"] = paired
    out["skipped"] = skipped
    return out


def save_metrics(metrics: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "metrics.json"
    # avoid Counter (not JSON-serializable directly)
    payload = dict(metrics)
    if "skipped_reasons" in payload and isinstance(payload["skipped_reasons"], Counter):
        payload["skipped_reasons"] = dict(payload["skipped_reasons"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def render_text_summary(metrics: dict[str, Any]) -> str:
    if "error" in metrics:
        return f"ERROR: {metrics['error']}"
    L1 = metrics["level1"]; L2 = metrics["level2"]
    lines = []
    lines.append(f"Eligible samples: {metrics['n_eligible_for_eval']} / {metrics['n_total_results']} (skipped {metrics['n_skipped']})")
    lines.append("")
    lines.append("=== LEVEL-1 (5-way single-label) ===")
    lines.append(f"  Accuracy           : {L1['accuracy']*100:6.2f}%")
    lines.append(f"  Macro    P/R/F1    : {L1['macro_precision']*100:6.2f}% / {L1['macro_recall']*100:6.2f}% / {L1['macro_f1']*100:6.2f}%")
    lines.append(f"  Micro    P/R/F1    : {L1['micro_precision']*100:6.2f}% / {L1['micro_recall']*100:6.2f}% / {L1['micro_f1']*100:6.2f}%")
    lines.append(f"  Weighted P/R/F1    : {L1['weighted_precision']*100:6.2f}% / {L1['weighted_recall']*100:6.2f}% / {L1['weighted_f1']*100:6.2f}%")
    lines.append("  Per-class:")
    for lbl, m in L1["per_class"].items():
        lines.append(f"    {lbl:<13}  P={m['precision']*100:6.2f}  R={m['recall']*100:6.2f}  F1={m['f1']*100:6.2f}  n={m['support']}")
    lines.append("")
    lines.append("=== LEVEL-2 (multi-label 8-way) ===")
    lines.append(f"  Macro    P/R/F1    : {L2['macro_precision']*100:6.2f}% / {L2['macro_recall']*100:6.2f}% / {L2['macro_f1']*100:6.2f}%")
    lines.append(f"  Micro    P/R/F1    : {L2['micro_precision']*100:6.2f}% / {L2['micro_recall']*100:6.2f}% / {L2['micro_f1']*100:6.2f}%")
    lines.append(f"  Weighted P/R/F1    : {L2['weighted_precision']*100:6.2f}% / {L2['weighted_recall']*100:6.2f}% / {L2['weighted_f1']*100:6.2f}%")
    lines.append(f"  Samples  P/R/F1    : {L2['samples_precision']*100:6.2f}% / {L2['samples_recall']*100:6.2f}% / {L2['samples_f1']*100:6.2f}%")
    lines.append(f"  Hamming Loss       : {L2['hamming_loss']*100:6.2f}%")
    lines.append(f"  Subset Accuracy    : {L2['subset_accuracy']*100:6.2f}% (exact-match across all 8 labels)")
    lines.append("  Per-class:")
    for lbl, m in L2["per_class"].items():
        lines.append(
            f"    {lbl:<32}  P={m['precision']*100:6.2f}  R={m['recall']*100:6.2f}  "
            f"F1={m['f1']*100:6.2f}  n={m['support']} (pred+={m['pred_positives']}, "
            f"TP={m.get('true_positives','?')} FP={m.get('false_positives','?')} FN={m.get('false_negatives','?')})"
        )
    return "\n".join(lines)
