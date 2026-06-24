"""Run the unified T2-Agent on ReMMDBench.

One pass over the benchmark produces BOTH outputs per sample:

* ``predicted_verdict``  -- 5-way single-label classification (vs ``annotation.json["verdict"]``).
* ``predicted_taxonomy`` -- 8-way multi-label classification (vs ``annotation.json["distortion_taxonomy"]``).

Per-sample results are written to disk *immediately* after each prediction so a
long-running job can be resumed and inspected at any time. All LLM and tool
calls are logged as JSONL under ``records/<run_name>/llm_calls.jsonl``.

Highlights of this entrypoint:

* **Resumable**: the script discovers any pre-existing ``details/<idx>_*.json``
  files in the target run directory and skips those samples. A long network
  outage or process crash no longer wastes the partial progress.
* **Default concurrency = 10**: every sample triggers many API calls, so we
  fan out 10 samples in parallel by default.
* **Full eval summary**: at the end of every run (and even when the user
  Ctrl-C-s out) we recompute the 5-way verdict metrics + the 8-label
  distortion taxonomy metrics, write both into JSON / Markdown summaries,
  and generate publication-quality confusion-matrix / per-label heatmaps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2agent.agent import PredictionResult, T2Agent  # noqa: E402
from t2agent.config import RuntimeConfig, load_runtime_config  # noqa: E402
from t2agent.data import BenchmarkSample, load_realmmdbench  # noqa: E402
from t2agent.labels import (  # noqa: E402
    REALMMDBENCH_TAXONOMY_LABELS,
    REALMMDBENCH_VERDICT_LABELS,
    format_label_list,
    labels_equal,
    multilabel_metrics,
    normalize_taxonomy_labels,
    normalize_verdict_label,
    verdict_metrics,
)
from t2agent.logging_utils import CallLogger, set_default_logger  # noqa: E402
from t2agent.metrics_plot import (  # noqa: E402
    plot_taxonomy_label_alignment_heatmap,
    plot_taxonomy_per_label_bar,
    plot_verdict_confusion_matrix,
    plot_verdict_per_label_bar,
)


_PRINT_LOCK = Lock()


def _log(msg: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    with _PRINT_LOCK:
        print(f"[{timestamp}] {msg}", flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _safe_sample_id(sample_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_id)[:80]


def _detail_path(output_dir: Path, index: int, sample_id: str) -> Path:
    return (
        output_dir
        / "details"
        / f"{index:03d}_{_safe_sample_id(sample_id)}.json"
    )


def _load_existing_detail(detail_path: Path) -> dict[str, Any] | None:
    """Load a per-sample detail JSON if it exists and looks complete."""

    if not detail_path.exists():
        return None
    try:
        with detail_path.open("r", encoding="utf-8") as handle:
            detail = json.load(handle)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(detail, dict):
        return None
    if "predicted_verdict" not in detail or "predicted_taxonomy" not in detail:
        return None
    return detail


def _detail_to_record(detail: dict[str, Any]) -> dict[str, Any]:
    """Project a detail JSON back into the compact per-sample record dict."""

    keys = (
        "index",
        "sample_id",
        "ground_truth_verdict",
        "ground_truth_taxonomy",
        "predicted_verdict",
        "predicted_taxonomy",
        "verdict_match",
        "taxonomy_match",
        "elapsed_seconds",
        "metadata",
    )
    record = {key: detail.get(key) for key in keys}
    record["resumed"] = True
    return record


def _predict_one(
    config: RuntimeConfig,
    sample: BenchmarkSample,
    output_dir: Path,
    index: int,
    total: int,
    position: int,
    progress: dict[str, int],
    progress_lock: Lock,
) -> dict[str, Any]:
    """Run a single prediction and write per-sample artifacts immediately."""

    _log(f"START [{position}/{total}] index={index} sample={sample.sample_id}")
    started = time.time()
    error_info: str | None = None
    prediction: PredictionResult | None = None

    try:
        agent = T2Agent(config)
        prediction = agent.predict(sample)
    except Exception:  # noqa: BLE001
        error_info = traceback.format_exc()
        _log(
            f"ERROR [{position}/{total}] index={index} sample={sample.sample_id}: "
            f"{error_info.splitlines()[-1]}"
        )

    elapsed = time.time() - started
    gt_verdict = normalize_verdict_label(sample.verdict) or sample.verdict
    gt_taxonomy = normalize_taxonomy_labels(sample.taxonomy_labels)
    predicted_verdict = prediction.predicted_verdict if prediction else "ERROR"
    predicted_taxonomy: list[str] | str = (
        prediction.predicted_taxonomy if prediction else "ERROR"
    )

    verdict_match = (
        predicted_verdict == gt_verdict if prediction is not None else False
    )
    taxonomy_match = (
        labels_equal(predicted_taxonomy, gt_taxonomy) if prediction is not None else False
    )

    record = {
        "index": index,
        "sample_id": sample.sample_id,
        "ground_truth_verdict": gt_verdict,
        "ground_truth_taxonomy": gt_taxonomy,
        "predicted_verdict": predicted_verdict,
        "predicted_taxonomy": predicted_taxonomy,
        "verdict_match": verdict_match,
        "taxonomy_match": taxonomy_match,
        "elapsed_seconds": round(elapsed, 2),
        "metadata": sample.metadata,
        "resumed": False,
    }
    detail = {
        **record,
        "sample_dir": str(sample.sample_dir),
        "image_paths": [str(p) for p in sample.image_paths],
        "prediction": asdict(prediction) if prediction else None,
        "error": error_info,
    }

    detail_path = _detail_path(output_dir, index, sample.sample_id)
    _write_json(detail_path, detail)

    with progress_lock:
        progress["done"] += 1
        done_local = progress["done"]
        total_local = progress["total"]
    verdict_str = "OK" if verdict_match else "MISS"
    taxonomy_str = "OK" if taxonomy_match else "MISS"
    _log(
        f"DONE  [{done_local}/{total_local}] index={index} sample={sample.sample_id} "
        f"verdict={predicted_verdict} ({verdict_str}) "
        f"taxonomy=[{format_label_list(predicted_taxonomy) if prediction else 'ERROR'}] "
        f"({taxonomy_str}) ({elapsed:.1f}s)"
    )
    return record


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _taxonomy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": rec["index"],
            "sample_id": rec["sample_id"],
            "ground_truth": rec["ground_truth_taxonomy"],
            "predicted": rec["predicted_taxonomy"],
        }
        for rec in records
    ]


def _verdict_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": rec["index"],
            "sample_id": rec["sample_id"],
            "ground_truth_verdict": rec["ground_truth_verdict"],
            "predicted_verdict": rec["predicted_verdict"],
        }
        for rec in records
    ]


def _build_markdown(
    records: list[dict[str, Any]],
    verdict_eval: dict[str, Any],
    taxonomy_eval: dict[str, Any],
    config: RuntimeConfig,
    timestamp: str,
    output_dir: Path,
    call_log_path: Path,
    wall_seconds: float,
    figure_paths: dict[str, Any],
) -> str:
    lines = [
        "# Unified T2-Agent Evaluation on ReMMDBench",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Provider: `{config.api.provider}`",
        f"- Model: `{config.api.model}`",
        f"- Base URL: `{config.api.primary_base_url}`",
        f"- Samples: {len(records)}",
        f"- Wall time: {wall_seconds:.1f}s",
        f"- Output dir: `{output_dir}`",
        f"- Call log: `{call_log_path}`",
        "",
        "## Five-way verdict (single-label)",
        "",
        f"- Accuracy (all): {verdict_eval['accuracy']:.4f} ({verdict_eval['correct']}/{verdict_eval['total']})",
        f"- Accuracy (valid only): {verdict_eval['valid_accuracy']:.4f}",
        f"- Macro Precision / Recall / F1: "
        f"{verdict_eval['macro_precision']:.4f} / {verdict_eval['macro_recall']:.4f} / "
        f"{verdict_eval['macro_f1']:.4f}",
        f"- ERROR predictions: {verdict_eval['errors']}",
        "",
        "| Verdict | Support | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in verdict_eval["per_label"]:
        lines.append(
            f"| {item['label']} | {item['support']} | {item['tp']} | {item['fp']} | "
            f"{item['fn']} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )

    cm = verdict_eval.get("confusion_matrix") or {}
    if cm:
        lines.extend(
            [
                "",
                "### 5x5 Confusion Matrix (rows = ground truth, cols = prediction; final col = ERROR)",
                "",
                "| GT \\\\ Pred | " + " | ".join(cm["col_labels"]) + " |",
                "| --- " * (len(cm["col_labels"]) + 1) + "|",
            ]
        )
        for row_label, row in zip(cm["row_labels"], cm["counts"]):
            lines.append(
                "| " + row_label + " | " + " | ".join(str(v) for v in row) + " |"
            )

    lines.extend(
        [
            "",
            "## Eight-way distortion taxonomy (multi-label)",
            "",
            f"- Exact-match accuracy: {taxonomy_eval['exact_match_accuracy']:.4f} "
            f"({taxonomy_eval['exact_correct']}/{taxonomy_eval['total']})",
            f"- Valid exact-match accuracy: {taxonomy_eval['valid_exact_match_accuracy']:.4f}",
            f"- Micro Precision / Recall / F1: "
            f"{taxonomy_eval['micro_precision']:.4f} / {taxonomy_eval['micro_recall']:.4f} / "
            f"{taxonomy_eval['micro_f1']:.4f}",
            f"- Macro Precision / Recall / F1: "
            f"{taxonomy_eval['macro_precision']:.4f} / {taxonomy_eval['macro_recall']:.4f} / "
            f"{taxonomy_eval['macro_f1']:.4f}",
            f"- Average Jaccard: {taxonomy_eval['average_jaccard']:.4f}",
            f"- ERROR predictions: {taxonomy_eval['errors']}",
            "",
            "| Label | Support | Predicted | TP | FP | FN | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in taxonomy_eval["per_label"]:
        lines.append(
            f"| {item['label']} | {item['support']} | {item['predicted']} | "
            f"{item['tp']} | {item['fp']} | {item['fn']} | "
            f"{item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )

    if figure_paths:
        lines.extend(["", "## Generated figures", ""])
        for tag, payload in figure_paths.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("status") == "ok":
                for path in payload.get("files", []):
                    lines.append(f"- `{tag}`: `{path}`")
            else:
                lines.append(f"- `{tag}`: ERROR ({payload.get('error')})")

    lines.extend(
        [
            "",
            "## First 20 per-sample predictions",
            "",
            "| # | Sample | GT Verdict | Pred Verdict | GT Taxonomy | Pred Taxonomy | V | T | Time |",
            "| ---: | --- | --- | --- | --- | --- | :-: | :-: | ---: |",
        ]
    )
    for record in records[:20]:
        verdict_flag = "Y" if record["verdict_match"] else ""
        taxonomy_flag = "Y" if record["taxonomy_match"] else ""
        predicted = record["predicted_taxonomy"]
        if predicted == "ERROR":
            pred_display = "ERROR"
        else:
            pred_display = format_label_list(predicted)
        lines.append(
            f"| {record['index'] + 1} | {record['sample_id'][:30]} | "
            f"{record['ground_truth_verdict']} | {record['predicted_verdict']} | "
            f"{format_label_list(record['ground_truth_taxonomy'])} | {pred_display} | "
            f"{verdict_flag} | {taxonomy_flag} | {record['elapsed_seconds']:.1f}s |"
        )
    return "\n".join(lines)


def _generate_figures(
    output_dir: Path,
    sorted_records: list[dict[str, Any]],
    verdict_eval: dict[str, Any],
    taxonomy_eval: dict[str, Any],
    provider: str,
) -> dict[str, Any]:
    """Create the publication-quality plots that summarise this run."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm = verdict_eval.get("confusion_matrix")
    figures: dict[str, Any] = {}
    if cm:
        figures["verdict_confusion_matrix"] = plot_verdict_confusion_matrix(
            cm,
            figures_dir,
            title=f"{provider}: 5-way Verdict Confusion Matrix",
            file_stem=f"{provider}_verdict_confusion_matrix",
        )
    figures["verdict_per_label_bar"] = plot_verdict_per_label_bar(
        verdict_eval["per_label"],
        figures_dir,
        title=f"{provider}: 5-way Verdict Per-class P/R/F1",
        file_stem=f"{provider}_verdict_per_label_bar",
    )
    figures["taxonomy_per_label_bar"] = plot_taxonomy_per_label_bar(
        taxonomy_eval["per_label"],
        figures_dir,
        title=f"{provider}: 8-way Distortion Taxonomy Per-label P/R/F1",
        file_stem=f"{provider}_taxonomy_per_label_bar",
    )
    figures["taxonomy_label_alignment_heatmap"] = plot_taxonomy_label_alignment_heatmap(
        _taxonomy_records(sorted_records),
        figures_dir,
        title=f"{provider}: Multi-label Distortion Taxonomy Alignment",
        file_stem=f"{provider}_taxonomy_label_alignment_heatmap",
    )
    return figures


def _build_eval_summary_md(
    verdict_eval: dict[str, Any],
    taxonomy_eval: dict[str, Any],
    *,
    provider: str,
    model: str,
    timestamp: str,
    output_dir: Path,
    wall_seconds: float,
    samples_completed: int,
    samples_total: int,
    figure_paths: dict[str, Any],
) -> str:
    cm = verdict_eval.get("confusion_matrix") or {}
    lines = [
        "# ReMMDBench Evaluation Summary",
        "",
        f"- Provider: `{provider}`",
        f"- Model: `{model}`",
        f"- Timestamp: `{timestamp}`",
        f"- Samples evaluated: {samples_completed} / {samples_total}",
        f"- Wall time: {wall_seconds:.1f}s",
        f"- Output dir: `{output_dir}`",
        "",
        "## Five-way Verdict (single-label)",
        "",
        f"- **Accuracy**: {verdict_eval['accuracy']:.4f}",
        f"- **Macro Precision**: {verdict_eval['macro_precision']:.4f}",
        f"- **Macro Recall**: {verdict_eval['macro_recall']:.4f}",
        f"- **Macro F1**: {verdict_eval['macro_f1']:.4f}",
        f"- ERROR predictions: {verdict_eval['errors']}",
        "",
        "### Per-class metrics",
        "",
        "| Verdict | Support | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in verdict_eval["per_label"]:
        lines.append(
            f"| {item['label']} | {item['support']} | "
            f"{item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )

    if cm:
        lines.extend(
            [
                "",
                "### Confusion Matrix (counts; rows = GT, cols = Pred; last col = ERROR)",
                "",
                "| GT \\\\ Pred | " + " | ".join(cm["col_labels"]) + " |",
                "| --- " * (len(cm["col_labels"]) + 1) + "|",
            ]
        )
        for row_label, row in zip(cm["row_labels"], cm["counts"]):
            lines.append(
                "| " + row_label + " | " + " | ".join(str(v) for v in row) + " |"
            )

    lines.extend(
        [
            "",
            "## Eight-way Distortion Taxonomy (multi-label)",
            "",
            f"- **Exact match**: {taxonomy_eval['exact_match_accuracy']:.4f} "
            f"({taxonomy_eval['exact_correct']}/{taxonomy_eval['total']})",
            f"- **Macro Precision**: {taxonomy_eval['macro_precision']:.4f}",
            f"- **Macro Recall**: {taxonomy_eval['macro_recall']:.4f}",
            f"- **Macro F1**: {taxonomy_eval['macro_f1']:.4f}",
            f"- Micro P/R/F1: {taxonomy_eval['micro_precision']:.4f} / "
            f"{taxonomy_eval['micro_recall']:.4f} / {taxonomy_eval['micro_f1']:.4f}",
            f"- Average Jaccard: {taxonomy_eval['average_jaccard']:.4f}",
            f"- ERROR predictions: {taxonomy_eval['errors']}",
            "",
            "### Per-label metrics",
            "",
            "| Label | Support | Predicted | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in taxonomy_eval["per_label"]:
        lines.append(
            f"| {item['label']} | {item['support']} | {item['predicted']} | "
            f"{item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )

    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for tag, payload in figure_paths.items():
            if not isinstance(payload, dict):
                continue
            if payload.get("status") == "ok":
                for path in payload.get("files", []):
                    lines.append(f"- `{tag}`: `{path}`")
            else:
                lines.append(f"- `{tag}`: ERROR ({payload.get('error')})")

    return "\n".join(lines)


def _write_eval_summary(
    output_dir: Path,
    sorted_records: list[dict[str, Any]],
    verdict_eval: dict[str, Any],
    taxonomy_eval: dict[str, Any],
    *,
    provider: str,
    model: str,
    timestamp: str,
    wall_seconds: float,
    samples_total: int,
    figure_paths: dict[str, Any],
) -> dict[str, Any]:
    eval_summary = {
        "provider": provider,
        "model": model,
        "timestamp": timestamp,
        "samples_completed": len(sorted_records),
        "samples_total": samples_total,
        "wall_seconds": round(wall_seconds, 2),
        "verdict_label_order": REALMMDBENCH_VERDICT_LABELS,
        "taxonomy_label_order": REALMMDBENCH_TAXONOMY_LABELS,
        "verdict_metrics": verdict_eval,
        "taxonomy_metrics": taxonomy_eval,
        "figures": figure_paths,
    }
    _write_json(output_dir / "eval_summary.json", eval_summary)
    markdown = _build_eval_summary_md(
        verdict_eval,
        taxonomy_eval,
        provider=provider,
        model=model,
        timestamp=timestamp,
        output_dir=output_dir,
        wall_seconds=wall_seconds,
        samples_completed=len(sorted_records),
        samples_total=samples_total,
        figure_paths=figure_paths,
    )
    (output_dir / "eval_summary.md").write_text(markdown, encoding="utf-8")
    return eval_summary


def _run_summary(
    records: list[dict[str, Any]],
    config: RuntimeConfig,
    timestamp: str,
    output_dir: Path,
    call_log_path: Path,
    wall_seconds: float,
    indexed_total: int,
) -> dict[str, Any]:
    sorted_records = sorted(records, key=lambda item: item["index"])
    verdict_eval = verdict_metrics(_verdict_records(sorted_records))
    taxonomy_eval = multilabel_metrics(_taxonomy_records(sorted_records))

    figures = _generate_figures(
        output_dir,
        sorted_records,
        verdict_eval,
        taxonomy_eval,
        provider=config.api.provider,
    )

    summary = {
        "timestamp": timestamp,
        "provider": config.api.provider,
        "model": config.api.model,
        "primary_base_url": config.api.primary_base_url,
        "config_path": str(Path(config.paths.workspace_root) / "configs"),
        "remmdbench_root": str(config.paths.realmmdbench_root),
        "serper_api_file": str(config.paths.serper_api_file),
        "serper_api_key_index": config.serper.api_key_index,
        "samples_total": indexed_total,
        "samples_completed": len(sorted_records),
        "wall_seconds": round(wall_seconds, 2),
        "call_log": str(call_log_path),
        "verdict_metrics": verdict_eval,
        "taxonomy_metrics": taxonomy_eval,
        "figures": figures,
        "records": sorted_records,
    }
    summary_path = output_dir / "run_summary.json"
    _write_json(summary_path, summary)

    markdown = _build_markdown(
        sorted_records,
        verdict_eval,
        taxonomy_eval,
        config,
        timestamp,
        output_dir,
        call_log_path,
        wall_seconds,
        figures,
    )
    (output_dir / "run_summary.md").write_text(markdown, encoding="utf-8")

    _write_eval_summary(
        output_dir,
        sorted_records,
        verdict_eval,
        taxonomy_eval,
        provider=config.api.provider,
        model=config.api.model,
        timestamp=timestamp,
        wall_seconds=wall_seconds,
        samples_total=indexed_total,
        figure_paths=figures,
    )
    return summary


def _parse_indices(text: str, total: int) -> list[int]:
    if not text.strip():
        return list(range(total))
    indices: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(chunk))
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        if 0 <= idx < total and idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def _resolve_run_dir(
    config: RuntimeConfig,
    explicit_run_name: str,
    timestamp: str,
    resume: bool,
) -> tuple[Path, str]:
    """Return the output directory + run-name for this invocation.

    With ``resume`` enabled and no explicit ``run-name``, we re-use the latest
    existing ``artifacts/runs/<provider>_*`` directory if it has details files
    on disk; otherwise we create a brand new one named ``<provider>_<ts>``.
    """

    runs_root = config.paths.artifacts_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    if explicit_run_name:
        return runs_root / explicit_run_name, explicit_run_name

    if resume:
        prefix = f"{config.api.provider}_"
        candidates = sorted(
            (path for path in runs_root.iterdir() if path.is_dir() and path.name.startswith(prefix)),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            details_dir = candidate / "details"
            if details_dir.exists() and any(details_dir.glob("*.json")):
                return candidate, candidate.name

    run_name = f"{config.api.provider}_{timestamp}"
    return runs_root / run_name, run_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run unified T2-Agent on ReMMDBench (5-class + 8-class)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file (one of configs/*.yaml).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of concurrent workers (default: 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N samples after filtering (0 = no limit).",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Comma-separated zero-based indices and ranges (e.g. 0,2,5-9).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single sample (index 0 by default) as a smoke test.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run name; defaults to <provider>_<timestamp> (or the "
        "latest matching directory when --resume is in effect).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable automatic resumption from a previous run directory.",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    dataset = load_realmmdbench(config.paths)

    selected_indices = _parse_indices(args.indices, len(dataset.samples))
    if args.smoke:
        selected_indices = selected_indices[:1] if selected_indices else [0]
    if args.limit > 0:
        selected_indices = selected_indices[: args.limit]
    if config.evaluation.max_samples > 0:
        selected_indices = selected_indices[: config.evaluation.max_samples]

    samples = [(i, dataset.samples[i]) for i in selected_indices]
    total = len(samples)
    if total == 0:
        raise SystemExit("No samples selected to run; check --indices / --limit / config.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir, run_name = _resolve_run_dir(
        config, args.run_name, timestamp, args.resume
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "details").mkdir(exist_ok=True)
    records_dir = config.paths.records_root / run_name
    records_dir.mkdir(parents=True, exist_ok=True)
    call_log_path = records_dir / "llm_calls.jsonl"
    call_log_path.touch(exist_ok=True)

    logger = CallLogger(call_log_path)
    set_default_logger(logger)

    _log(f"Provider: {config.api.provider} | Model: {config.api.model}")
    _log(f"ReMMDBench root: {config.paths.realmmdbench_root}")
    _log(f"Serper API file: {config.paths.serper_api_file} (index={config.serper.api_key_index})")
    _log(f"Selected samples: {total} | Workers: {args.max_workers} | Resume: {args.resume}")
    _log(f"Run output dir: {output_dir}")
    _log(f"Call log: {call_log_path}")

    _write_json(
        output_dir / "run_config.json",
        {
            "timestamp": timestamp,
            "run_name": run_name,
            "config_path": str(Path(args.config).resolve()),
            "provider": config.api.provider,
            "model": config.api.model,
            "primary_base_url": config.api.primary_base_url,
            "backup_base_urls": config.api.backup_base_urls,
            "remmdbench_root": str(config.paths.realmmdbench_root),
            "serper_api_file": str(config.paths.serper_api_file),
            "serper_api_key_index": config.serper.api_key_index,
            "selected_indices": selected_indices,
            "max_workers": args.max_workers,
            "resume": args.resume,
        },
    )

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pending: list[tuple[int, BenchmarkSample]] = []

    if args.resume:
        for idx, sample in samples:
            existing = _load_existing_detail(_detail_path(output_dir, idx, sample.sample_id))
            if existing is not None:
                records.append(_detail_to_record(existing))
                if existing.get("predicted_verdict") == "ERROR":
                    errors.append(existing)
            else:
                pending.append((idx, sample))
        resumed_count = len(records)
        _log(
            f"Resume scan: {resumed_count} samples already complete, "
            f"{len(pending)} pending."
        )
    else:
        pending = list(samples)

    progress = {"done": len(records), "total": total}
    progress_lock = Lock()

    wall_start = time.time()
    max_workers = max(1, args.max_workers)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _predict_one,
                    config,
                    sample,
                    output_dir,
                    idx,
                    total,
                    position,
                    progress,
                    progress_lock,
                ): (position, idx)
                for position, (idx, sample) in enumerate(pending, start=len(records) + 1)
            }
            for future in as_completed(futures):
                position, idx = futures[future]
                try:
                    record = future.result()
                    records.append(record)
                    if record["predicted_verdict"] == "ERROR":
                        errors.append(record)
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    _log(f"FATAL [{position}/{total}] index={idx}: {tb.splitlines()[-1]}")
                    errors.append({"index": idx, "error": tb})
    except KeyboardInterrupt:
        _log("Caught KeyboardInterrupt; writing partial summary before exiting...")

    wall_seconds = time.time() - wall_start
    summary = _run_summary(
        records,
        config,
        timestamp,
        output_dir,
        call_log_path,
        wall_seconds,
        indexed_total=total,
    )

    _log("=== FINISHED ===")
    _log(
        "Verdict accuracy: "
        f"{summary['verdict_metrics']['accuracy']:.4f} "
        f"({summary['verdict_metrics']['correct']}/{summary['verdict_metrics']['total']})"
    )
    _log(
        "Verdict macro P/R/F1: "
        f"{summary['verdict_metrics']['macro_precision']:.4f} / "
        f"{summary['verdict_metrics']['macro_recall']:.4f} / "
        f"{summary['verdict_metrics']['macro_f1']:.4f}"
    )
    _log(
        "Taxonomy exact-match: "
        f"{summary['taxonomy_metrics']['exact_match_accuracy']:.4f} "
        f"({summary['taxonomy_metrics']['exact_correct']}/{summary['taxonomy_metrics']['total']})"
    )
    _log(
        "Taxonomy macro P/R/F1: "
        f"{summary['taxonomy_metrics']['macro_precision']:.4f} / "
        f"{summary['taxonomy_metrics']['macro_recall']:.4f} / "
        f"{summary['taxonomy_metrics']['macro_f1']:.4f} | "
        f"Errors: {len(errors)} | Wall: {wall_seconds:.1f}s"
    )
    _log(f"Results saved to {output_dir}")
    _log(f"Eval summary: {output_dir / 'eval_summary.json'}")
    _log(f"LLM/tool call log: {call_log_path} ({logger.event_count} events)")


if __name__ == "__main__":
    main()
