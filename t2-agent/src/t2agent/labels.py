"""Label vocabularies, normalization helpers, and metric utilities.

This module is the single source of truth for the two ReMMDBench tasks:

* ``REALMMDBENCH_VERDICT_LABELS``     : ordered five-way single-label verdict.
* ``REALMMDBENCH_TAXONOMY_LABELS``    : eight-way multi-label distortion taxonomy.

The functions in this module are reused by the agent, the runner, the metric
scripts, and the per-sample dumps, so every consumer reads exactly the same
canonical label spelling.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


REALMMDBENCH_VERDICT_LABELS: list[str] = [
    "True",
    "Mostly True",
    "Mixture",
    "Mostly False",
    "False",
]

REALMMDBENCH_TAXONOMY_LABELS: list[str] = [
    "T1 Fabrication",
    "T2 Distortion",
    "T3 Misleading Context",
    "V1 Synthetic Visual Content",
    "V2 Visual Editing",
    "C1 Semantic Inconsistency",
    "C2 Contextual Inconsistency",
    "C3 Pragmatic Inconsistency",
]

NO_SECONDARY_LABEL = "None"


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_CODE_TO_LABEL = {label.split()[0].lower(): label for label in REALMMDBENCH_TAXONOMY_LABELS}

_ALIASES: dict[str, str] = {
    "fabrication": "T1 Fabrication",
    "text fabrication": "T1 Fabrication",
    "textual fabrication": "T1 Fabrication",
    "distortion": "T2 Distortion",
    "text distortion": "T2 Distortion",
    "textual distortion": "T2 Distortion",
    "misleading context": "T3 Misleading Context",
    "text misleading context": "T3 Misleading Context",
    "synthetic visual content": "V1 Synthetic Visual Content",
    "synthetic image": "V1 Synthetic Visual Content",
    "ai generated visual": "V1 Synthetic Visual Content",
    "ai generated image": "V1 Synthetic Visual Content",
    "visual editing": "V2 Visual Editing",
    "image editing": "V2 Visual Editing",
    "edited visual": "V2 Visual Editing",
    "semantic inconsistency": "C1 Semantic Inconsistency",
    "cross modal semantic inconsistency": "C1 Semantic Inconsistency",
    "contextual inconsistency": "C2 Contextual Inconsistency",
    "cross modal contextual inconsistency": "C2 Contextual Inconsistency",
    "pragmatic inconsistency": "C3 Pragmatic Inconsistency",
    "cross modal pragmatic inconsistency": "C3 Pragmatic Inconsistency",
}

_CANONICAL_BY_KEY: dict[str, str] = {
    _key(label): label for label in REALMMDBENCH_TAXONOMY_LABELS
}
_CANONICAL_BY_KEY.update({_key(alias): label for alias, label in _ALIASES.items()})


_VERDICT_ALIASES: dict[str, str] = {
    "true": "True",
    "mostly true": "Mostly True",
    "mostly-true": "Mostly True",
    "mostly_true": "Mostly True",
    "mixture": "Mixture",
    "mixed": "Mixture",
    "half true": "Mixture",
    "half-true": "Mixture",
    "mostly false": "Mostly False",
    "mostly-false": "Mostly False",
    "mostly_false": "Mostly False",
    "false": "False",
}


def normalize_taxonomy_label(label: Any) -> str | None:
    """Return the canonical taxonomy spelling, or ``None`` if not recognized."""

    text = str(label).strip()
    if not text:
        return None
    if text.upper() == "ERROR" or _key(text) in {
        "none",
        "no label",
        "no labels",
        "no distortion",
    }:
        return None

    code_match = re.match(r"^([TVC][1-3])\b", text, flags=re.IGNORECASE)
    if code_match:
        canonical = _CODE_TO_LABEL.get(code_match.group(1).lower())
        if canonical:
            return canonical

    return _CANONICAL_BY_KEY.get(_key(text))


def normalize_taxonomy_labels(labels: Any) -> list[str]:
    """Normalize an arbitrary input into ordered canonical taxonomy labels."""

    if labels is None:
        return []
    if isinstance(labels, str):
        raw_items: Iterable[Any] = re.split(r"[,;|、\n]+", labels)
    elif isinstance(labels, dict):
        raw_items = labels.keys()
    else:
        try:
            raw_items = list(labels)
        except TypeError:
            raw_items = [labels]

    selected: set[str] = set()
    for item in raw_items:
        normalized = normalize_taxonomy_label(item)
        if normalized:
            selected.add(normalized)
    return [label for label in REALMMDBENCH_TAXONOMY_LABELS if label in selected]


def normalize_verdict_label(label: Any) -> str | None:
    """Return the canonical verdict spelling, or ``None`` if not recognized."""

    if label is None:
        return None
    text = str(label).strip()
    if not text or text.upper() == "ERROR":
        return None
    canonical = _VERDICT_ALIASES.get(text.lower())
    if canonical:
        return canonical
    for canonical_label in REALMMDBENCH_VERDICT_LABELS:
        if text.lower() == canonical_label.lower():
            return canonical_label
    return None


def is_error_prediction(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().upper() == "ERROR"
    if isinstance(value, list):
        return any(
            isinstance(item, str) and item.strip().upper() == "ERROR" for item in value
        )
    return False


def format_label_list(labels: Any) -> str:
    normalized = normalize_taxonomy_labels(labels)
    return "; ".join(normalized) if normalized else NO_SECONDARY_LABEL


def labels_equal(predicted: Any, ground_truth: Any) -> bool:
    if is_error_prediction(predicted):
        return False
    return set(normalize_taxonomy_labels(predicted)) == set(
        normalize_taxonomy_labels(ground_truth)
    )


def label_jaccard(predicted: Any, ground_truth: Any) -> float:
    if is_error_prediction(predicted):
        return 0.0
    pred_set = set(normalize_taxonomy_labels(predicted))
    gt_set = set(normalize_taxonomy_labels(ground_truth))
    union = pred_set | gt_set
    if not union:
        return 1.0
    return len(pred_set & gt_set) / len(union)


def multilabel_metrics(
    records: list[dict[str, Any]],
    *,
    label_order: list[str] | None = None,
    ground_truth_key: str = "ground_truth",
    predicted_key: str = "predicted",
) -> dict[str, Any]:
    """Compute exact-match accuracy and per-label precision/recall/F1."""

    labels = label_order or REALMMDBENCH_TAXONOMY_LABELS
    total = len(records)
    error_count = sum(1 for item in records if is_error_prediction(item.get(predicted_key)))
    valid = total - error_count
    exact_correct = sum(
        1
        for item in records
        if labels_equal(item.get(predicted_key), item.get(ground_truth_key))
    )

    per_label: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = fp = fn = support = predicted_count = 0
        for item in records:
            pred_set: set[str]
            if is_error_prediction(item.get(predicted_key)):
                pred_set = set()
            else:
                pred_set = set(normalize_taxonomy_labels(item.get(predicted_key)))
            gt_set = set(normalize_taxonomy_labels(item.get(ground_truth_key)))
            if label in gt_set:
                support += 1
            if label in pred_set:
                predicted_count += 1
            if label in gt_set and label in pred_set:
                tp += 1
            elif label not in gt_set and label in pred_set:
                fp += 1
            elif label in gt_set and label not in pred_set:
                fn += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label.append(
            {
                "label": label,
                "support": support,
                "predicted": predicted_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    macro_precision = (
        sum(item["precision"] for item in per_label) / len(per_label)
        if per_label
        else 0.0
    )
    macro_recall = (
        sum(item["recall"] for item in per_label) / len(per_label) if per_label else 0.0
    )
    macro_f1 = sum(item["f1"] for item in per_label) / len(per_label) if per_label else 0.0
    avg_jaccard = (
        sum(
            label_jaccard(item.get(predicted_key), item.get(ground_truth_key))
            for item in records
        )
        / total
        if total
        else 0.0
    )

    return {
        "total": total,
        "valid": valid,
        "errors": error_count,
        "exact_correct": exact_correct,
        "exact_match_accuracy": round(exact_correct / total, 4) if total else 0.0,
        "valid_exact_match_accuracy": round(exact_correct / valid, 4) if valid else 0.0,
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "average_jaccard": round(avg_jaccard, 4),
        "per_label": per_label,
    }


def verdict_confusion_matrix(
    records: list[dict[str, Any]],
    *,
    ground_truth_key: str = "ground_truth_verdict",
    predicted_key: str = "predicted_verdict",
) -> dict[str, Any]:
    """Build the 5x5 confusion matrix (rows = ground truth, cols = prediction).

    ``ERROR`` predictions and unrecognized labels are bucketed into a dedicated
    final column so the matrix accounts for *all* samples. The matrix is also
    returned in row-normalized form (the per-class recall).
    """

    label_order = list(REALMMDBENCH_VERDICT_LABELS)
    extended_cols = label_order + ["ERROR"]
    size_rows = len(label_order)
    size_cols = len(extended_cols)
    matrix = [[0 for _ in range(size_cols)] for _ in range(size_rows)]

    for record in records:
        gt_raw = record.get(ground_truth_key)
        pred_raw = record.get(predicted_key)
        gt = normalize_verdict_label(gt_raw)
        if gt is None or gt not in label_order:
            continue
        row = label_order.index(gt)
        if is_error_prediction(pred_raw):
            col = size_cols - 1
        else:
            pred = normalize_verdict_label(pred_raw)
            if pred is None or pred not in label_order:
                col = size_cols - 1
            else:
                col = label_order.index(pred)
        matrix[row][col] += 1

    normalized = []
    for row in matrix:
        row_sum = sum(row)
        if row_sum == 0:
            normalized.append([0.0 for _ in row])
        else:
            normalized.append([round(value / row_sum, 4) for value in row])

    return {
        "row_labels": label_order,
        "col_labels": extended_cols,
        "counts": matrix,
        "row_normalized": normalized,
    }


def verdict_metrics(
    records: list[dict[str, Any]],
    *,
    ground_truth_key: str = "ground_truth_verdict",
    predicted_key: str = "predicted_verdict",
) -> dict[str, Any]:
    """Compute accuracy / macro-F1 for the five-way verdict classification."""

    total = len(records)
    label_order = REALMMDBENCH_VERDICT_LABELS
    correct = 0
    errors = 0
    per_label: list[dict[str, Any]] = []

    valid_gt: list[str] = []
    valid_pred: list[str] = []
    for record in records:
        pred_raw = record.get(predicted_key)
        gt_raw = record.get(ground_truth_key)
        if is_error_prediction(pred_raw):
            errors += 1
            continue
        pred = normalize_verdict_label(pred_raw)
        gt = normalize_verdict_label(gt_raw)
        if pred is None or gt is None:
            errors += 1
            continue
        valid_gt.append(gt)
        valid_pred.append(pred)
        if pred == gt:
            correct += 1

    valid_total = len(valid_gt)

    for label in label_order:
        tp = sum(
            1
            for gt, pred in zip(valid_gt, valid_pred)
            if gt == label and pred == label
        )
        fp = sum(
            1
            for gt, pred in zip(valid_gt, valid_pred)
            if gt != label and pred == label
        )
        fn = sum(
            1
            for gt, pred in zip(valid_gt, valid_pred)
            if gt == label and pred != label
        )
        support = sum(1 for gt in valid_gt if gt == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label.append(
            {
                "label": label,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )

    macro_f1 = sum(item["f1"] for item in per_label) / len(per_label) if per_label else 0.0
    macro_precision = (
        sum(item["precision"] for item in per_label) / len(per_label)
        if per_label
        else 0.0
    )
    macro_recall = (
        sum(item["recall"] for item in per_label) / len(per_label)
        if per_label
        else 0.0
    )
    accuracy_all = correct / total if total else 0.0
    accuracy_valid = correct / valid_total if valid_total else 0.0

    cm = verdict_confusion_matrix(
        records,
        ground_truth_key=ground_truth_key,
        predicted_key=predicted_key,
    )

    return {
        "total": total,
        "valid": valid_total,
        "errors": errors,
        "correct": correct,
        "accuracy": round(accuracy_all, 4),
        "valid_accuracy": round(accuracy_valid, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_label": per_label,
        "confusion_matrix": cm,
    }
