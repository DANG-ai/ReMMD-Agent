"""Recompute verdict + taxonomy metrics from a saved ``run_summary.json``.

Usage::

    conda run -n mmd python scripts/calc_metrics.py artifacts/runs/<run>/run_summary.json

The script prints both the five-way verdict metrics and the eight-way
multi-label taxonomy metrics, plus per-label breakdowns. It also rebuilds
the publication-quality plots (5x5 confusion matrix heatmap, multi-label
alignment heatmap, per-class P/R/F1 bar charts) into the run's
``figures/`` sub-directory. It is reusable across all four model providers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2agent.labels import (  # noqa: E402
    REALMMDBENCH_TAXONOMY_LABELS,
    REALMMDBENCH_VERDICT_LABELS,
    format_label_list,
    multilabel_metrics,
    normalize_taxonomy_labels,
    normalize_verdict_label,
    verdict_metrics,
)
from t2agent.metrics_plot import (  # noqa: E402
    plot_taxonomy_label_alignment_heatmap,
    plot_taxonomy_per_label_bar,
    plot_verdict_confusion_matrix,
    plot_verdict_per_label_bar,
)


def _load_records(summary_path: Path) -> list[dict[str, Any]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for rec in data["records"]:
        gt_verdict = normalize_verdict_label(rec.get("ground_truth_verdict")) or rec.get(
            "ground_truth_verdict"
        )
        gt_taxonomy = normalize_taxonomy_labels(rec.get("ground_truth_taxonomy", []))
        pred_taxonomy_raw = rec.get("predicted_taxonomy")
        if isinstance(pred_taxonomy_raw, str) and pred_taxonomy_raw.upper() == "ERROR":
            pred_taxonomy: Any = "ERROR"
        else:
            pred_taxonomy = normalize_taxonomy_labels(pred_taxonomy_raw)
        records.append(
            {
                "index": rec.get("index"),
                "sample_id": rec.get("sample_id"),
                "ground_truth_verdict": gt_verdict,
                "predicted_verdict": rec.get("predicted_verdict"),
                "ground_truth": gt_taxonomy,
                "predicted": pred_taxonomy,
                "ground_truth_taxonomy": gt_taxonomy,
                "predicted_taxonomy": pred_taxonomy,
            }
        )
    return records


def _resolve_provider(summary_path: Path) -> str:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data.get("provider", "model")


def _print_confusion_matrix(matrix: dict[str, Any]) -> None:
    if not matrix:
        return
    col_labels = matrix["col_labels"]
    header = "         | " + " | ".join(f"{label:>10s}" for label in col_labels)
    print(header)
    print("-" * len(header))
    for label, row in zip(matrix["row_labels"], matrix["counts"]):
        cells = " | ".join(f"{v:>10d}" for v in row)
        print(f"{label:>9s} | {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute ReMMDBench metrics from a run.")
    parser.add_argument("summary", type=Path, help="Path to run_summary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the metrics JSON output.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip regenerating the heatmap / bar-chart figures.",
    )
    args = parser.parse_args()

    records = _load_records(args.summary)
    verdict_eval = verdict_metrics(records)
    taxonomy_eval = multilabel_metrics(records)
    provider = _resolve_provider(args.summary)

    run_dir = args.summary.resolve().parent
    figure_paths: dict[str, Any] = {}
    if not args.skip_plots:
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        cm = verdict_eval.get("confusion_matrix")
        if cm:
            figure_paths["verdict_confusion_matrix"] = plot_verdict_confusion_matrix(
                cm,
                figures_dir,
                title=f"{provider}: 5-way Verdict Confusion Matrix",
                file_stem=f"{provider}_verdict_confusion_matrix",
            )
        figure_paths["verdict_per_label_bar"] = plot_verdict_per_label_bar(
            verdict_eval["per_label"],
            figures_dir,
            title=f"{provider}: 5-way Verdict Per-class P/R/F1",
            file_stem=f"{provider}_verdict_per_label_bar",
        )
        figure_paths["taxonomy_per_label_bar"] = plot_taxonomy_per_label_bar(
            taxonomy_eval["per_label"],
            figures_dir,
            title=f"{provider}: 8-way Distortion Taxonomy Per-label P/R/F1",
            file_stem=f"{provider}_taxonomy_per_label_bar",
        )
        figure_paths["taxonomy_label_alignment_heatmap"] = plot_taxonomy_label_alignment_heatmap(
            records,
            figures_dir,
            title=f"{provider}: Multi-label Distortion Taxonomy Alignment",
            file_stem=f"{provider}_taxonomy_label_alignment_heatmap",
        )

    output = {
        "summary_path": str(args.summary.resolve()),
        "provider": provider,
        "verdict_label_order": REALMMDBENCH_VERDICT_LABELS,
        "taxonomy_label_order": REALMMDBENCH_TAXONOMY_LABELS,
        "verdict_metrics": verdict_eval,
        "taxonomy_metrics": taxonomy_eval,
        "figures": figure_paths,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("=" * 60)
    print(f"Five-way verdict metrics (n={verdict_eval['total']}):")
    print(
        f"  accuracy={verdict_eval['accuracy']:.4f} | "
        f"valid_accuracy={verdict_eval['valid_accuracy']:.4f} | "
        f"errors={verdict_eval['errors']}"
    )
    print(
        f"  macro P/R/F1={verdict_eval['macro_precision']:.4f} / "
        f"{verdict_eval['macro_recall']:.4f} / {verdict_eval['macro_f1']:.4f}"
    )
    print("  per-label:")
    for item in verdict_eval["per_label"]:
        print(
            f"    {item['label']:<14}: support={item['support']:<3} "
            f"P={item['precision']:.4f} R={item['recall']:.4f} F1={item['f1']:.4f}"
        )

    print()
    print("  Confusion matrix:")
    _print_confusion_matrix(verdict_eval.get("confusion_matrix") or {})

    print("=" * 60)
    print(f"Eight-way distortion taxonomy metrics (n={taxonomy_eval['total']}):")
    print(
        f"  exact_match={taxonomy_eval['exact_match_accuracy']:.4f} | "
        f"valid_exact_match={taxonomy_eval['valid_exact_match_accuracy']:.4f} | "
        f"errors={taxonomy_eval['errors']}"
    )
    print(
        f"  micro P/R/F1={taxonomy_eval['micro_precision']:.4f} / "
        f"{taxonomy_eval['micro_recall']:.4f} / {taxonomy_eval['micro_f1']:.4f}"
    )
    print(
        f"  macro P/R/F1={taxonomy_eval['macro_precision']:.4f} / "
        f"{taxonomy_eval['macro_recall']:.4f} / {taxonomy_eval['macro_f1']:.4f}"
    )
    print(f"  average Jaccard={taxonomy_eval['average_jaccard']:.4f}")
    print("  per-label:")
    for item in taxonomy_eval["per_label"]:
        print(
            f"    {item['label']:<32}: support={item['support']:<3} pred={item['predicted']:<3} "
            f"P={item['precision']:.4f} R={item['recall']:.4f} F1={item['f1']:.4f}"
        )

    if figure_paths:
        print("=" * 60)
        print("Figures:")
        for tag, payload in figure_paths.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("status") == "ok":
                for path in payload.get("files", []):
                    print(f"  {tag}: {path}")
            else:
                print(f"  {tag}: ERROR {payload.get('error')}")

    print("=" * 60)
    print("First 10 sample predictions:")
    for rec in records[:10]:
        verdict_gt = rec["ground_truth_verdict"]
        verdict_pred = rec["predicted_verdict"]
        gt = format_label_list(rec["ground_truth"])
        pred = rec["predicted"]
        if isinstance(pred, str) and pred.upper() == "ERROR":
            pred_display = "ERROR"
        else:
            pred_display = format_label_list(pred)
        print(
            f"  {rec['index'] + 1:03d}: verdict GT={verdict_gt} pred={verdict_pred} | "
            f"taxonomy GT=[{gt}] pred=[{pred_display}]"
        )


if __name__ == "__main__":
    main()
