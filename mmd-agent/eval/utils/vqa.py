"""Main evaluation loop: one agent run produces both 5cls verdict and 8-label distortions.

Behaviour summary:
    - For each sample, the agent runs the 3 stages (text, visual, cross-modal)
      ONCE. Each stage prompt is the unified prompt template that asks the
      model to output:
        (a) a Finish[...] signal in the 5-bucket scale, and
        (b) a Distortions: <codes> line listing which of the relevant
            taxonomy categories apply for this stage.
    - From those outputs we compute:
        - 5-class final verdict via combine_stage_signals.
        - 8-label distortion union via combine_distortion_predictions.
    - All raw LLM calls (prompt, model output, raw response payload) are
      logged to a per-sample ``llm_calls.jsonl`` so the operator can audit
      exactly what the model saw and produced.
    - Per-sample results are written to disk both as ``samples/<id>/result.json``
      and as a running ``<dataset_name>.jsonl`` checkpoint. The pipeline can
      resume from the JSONL after an interruption.

We intentionally avoid torch / DataLoader / sklearn so the same code can run
in the slim ``mmd`` conda env used for local smoke tests.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .metrics_and_plot import (
    compute_distortion_metrics_8label,
    compute_verdict_metrics_5cls,
    render_summary_report,
    save_eval_summary,
)
from .model_utils import call_chat_engine_multi_image
from .serper_search import search_key_entity
from .tools import (
    DISTORTION_CODE_TO_FULL_LABEL,
    FULL_DISTORTION_LABEL_ORDER,
    VERDICT_LABEL_ORDER,
    combine_distortion_predictions,
    combine_stage_signals,
    gt_distortion_to_binary_vector,
    parse_distortion_codes,
    parse_stage_signal,
)


def _clean_data(answer: str) -> str:
    if not isinstance(answer, str):
        return ""
    return answer.replace("\n", " ").replace("\t", " ").strip()


# -- IO helpers ---------------------------------------------------------------


def _load_checkpoint_predictions(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    predictions: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return predictions


def _append_checkpoint_prediction(jsonl_path: Path, prediction: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def _write_progress_file(progress_path: Path, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -- Metrics: compute + save + plot ------------------------------------------
#
# 5 分类指标 / 8 标签多标签指标 / 混淆矩阵热力图 / 柱状图 / 摘要文本都集中在
# utils.metrics_and_plot 模块。这里只剩一个轻量的 wrapper 负责把 predictions
# 喂进去，写盘，并 print 一份与落盘 .txt 同步的 summary。


def _save_metrics_and_print(
    run_root: Path,
    metrics_path: Path,
    predictions: list[dict],
    model_name: str,
    run_tag: str,
) -> tuple[dict, dict]:
    """Compute + save + plot + print 一次完整的 eval summary。

    返回 ``(verdict_metrics, distortion_metrics)`` 以便上层在需要时复用。
    """
    verdict_metrics = compute_verdict_metrics_5cls(predictions, VERDICT_LABEL_ORDER)
    distortion_metrics = compute_distortion_metrics_8label(predictions, FULL_DISTORTION_LABEL_ORDER)

    saved = save_eval_summary(
        out_dir=run_root,
        verdict_metrics=verdict_metrics,
        distortion_metrics=distortion_metrics,
        model_name=model_name,
        run_tag=run_tag,
        verdict_label_order=VERDICT_LABEL_ORDER,
        distortion_label_order=FULL_DISTORTION_LABEL_ORDER,
    )

    print("\n" + saved["text_report"], flush=True)
    print(
        f"Artifacts:\n"
        f"  metrics.json      : {saved['metrics_json']}\n"
        f"  eval_summary.txt  : {saved['metrics_txt']}\n"
        f"  confusion_matrix  : {saved['confusion_matrix_pdf']}\n"
        f"                      {saved['confusion_matrix_png']}\n"
        f"  per_label_bar     : {saved['per_label_bar_pdf']}\n"
        f"                      {saved['per_label_bar_png']}",
        flush=True,
    )
    return verdict_metrics, distortion_metrics


# -- Core sample-level pipeline ----------------------------------------------


def _process_sample(args, sample: dict[str, Any], serper_key: str, per_sample_dir: Path) -> tuple[dict, bool]:
    sample_name = sample["sample_name"]
    sample_dir = sample["sample_dir"]
    image_paths = sample["image_paths"]
    image_caption_context = sample.get("image_caption_context", "")
    provided_evidence = sample.get("provided_evidence", "")
    question_fix_text_check = sample["question_fix_text_check"]
    question_fix_image_check = sample["question_fix_image_check"]
    question_fix_consistency_reason = sample["question_fix_consistency_reason"]

    verdict_gt = sample.get("verdict_gt", "")
    distortion_taxonomy_gt = sample.get("distortion_taxonomy_gt", [])

    per_sample_dir.mkdir(parents=True, exist_ok=True)
    llm_calls_log = per_sample_dir / "llm_calls.jsonl"
    if llm_calls_log.exists():
        llm_calls_log.unlink()

    answer_dict: dict[str, Any] = {
        "sample_name": sample_name,
        "sample_dir": sample_dir,
        "model_name": args.model_name,
        "image_paths": image_paths,
        "verdict_gt": verdict_gt,
        "distortion_taxonomy_gt": distortion_taxonomy_gt,
        "language_code": sample.get("language_code", ""),
        "region_code": sample.get("region_code", ""),
        "theme_category": sample.get("theme_category", ""),
        "text_length_tier": sample.get("text_length_tier", ""),
    }

    question_all: list[str] = []
    answer_all: list[str] = []
    raw_calls: list[dict[str, Any]] = []

    def _log_call(stage: str, prompt: str, output: str, raw_response: Any) -> None:
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "sample_name": sample_name,
            "model_name": args.model_name,
            "base_url": args.base_url,
            "stage": stage,
            "prompt": prompt,
            "output": output,
            "raw_response": raw_response,
        }
        raw_calls.append({
            "stage": stage,
            "prompt": prompt,
            "output": output,
            "raw_response": raw_response,
        })
        _append_jsonl(llm_calls_log, record)

    try:
        # --- STAGE 1: TEXTUAL VERACITY (with optional Serper-grounded knowledge) ---
        text_check_action_1 = (
            question_fix_text_check.split("Action 1:")[0].strip()
            + " Please answer in the form: 'Finish: [key entity noun or key event phrase].'"
        )
        output, raw = call_chat_engine_multi_image(args, text_check_action_1, image_paths)
        _log_call("text_action_1_extract_entity", text_check_action_1, output, raw)

        key_entity, wiki_knowledge, search_raw = search_key_entity(output, serper_key)
        _log_call(
            "text_action_search",
            f"[Serper] query={key_entity}",
            wiki_knowledge,
            search_raw,
        )

        text_check_action_2 = question_fix_text_check.split("[Analysis]")[0].strip()
        text_check_action_2 = text_check_action_2.replace("[key entity noun]", key_entity or "")
        text_check_action_2 = text_check_action_2.replace("[External Knowledge]", wiki_knowledge or "")
        output, raw = call_chat_engine_multi_image(args, text_check_action_2, image_paths)
        _log_call("text_action_2_analysis", text_check_action_2, output, raw)

        analysis_chunk = output
        if "Analysis:" in analysis_chunk:
            analysis_chunk = analysis_chunk.split("Analysis:", 1)[1]
        analysis = _clean_data(analysis_chunk)

        text_check_action_3 = question_fix_text_check.replace("[key entity noun]", key_entity or "")
        text_check_action_3 = text_check_action_3.replace("[External Knowledge]", wiki_knowledge or "")
        text_check_action_3 = text_check_action_3.replace("[Analysis]", analysis)
        output, raw = call_chat_engine_multi_image(args, text_check_action_3, image_paths)
        _log_call("text_action_3_final_label", text_check_action_3, output, raw)
        question_all.append(text_check_action_3)
        answer_all.append(output)

        text_signal = parse_stage_signal(output, "text")
        text_codes = parse_distortion_codes(output, "text")

        # --- STAGE 2: VISUAL VERACITY ---
        visual_check_action_1 = question_fix_image_check.split("Observation:")[0].strip()
        output, raw = call_chat_engine_multi_image(args, visual_check_action_1, image_paths)
        _log_call("visual_action_1_describe", visual_check_action_1, output, raw)

        descr = output
        if "Action 1:" in descr:
            descr = descr.split("Action 1:", 1)[0]
        img_descrip = _clean_data(descr)
        if not img_descrip:
            img_descrip = image_caption_context or "The image set description is unavailable."

        visual_check_action_2 = question_fix_image_check.replace("[Fact-conflicting Description]", img_descrip)
        output, raw = call_chat_engine_multi_image(args, visual_check_action_2, image_paths)
        _log_call("visual_action_2_final_label", visual_check_action_2, output, raw)
        question_all.append(visual_check_action_2)
        answer_all.append(output)

        image_signal = parse_stage_signal(output, "image")
        image_codes = parse_distortion_codes(output, "image")

        # --- STAGE 3: CROSS-MODAL ---
        consistency_check_action_1 = question_fix_consistency_reason.replace(
            "[image content description]", img_descrip
        )
        output, raw = call_chat_engine_multi_image(args, consistency_check_action_1, image_paths)
        _log_call("cross_action_1_final_label", consistency_check_action_1, output, raw)
        question_all.append(consistency_check_action_1)
        answer_all.append(output)

        cross_signal = parse_stage_signal(output, "cross")
        cross_codes = parse_distortion_codes(output, "cross")

        # --- AGGREGATION ---
        final_verdict, rule_meta = combine_stage_signals(text_signal, image_signal, cross_signal)
        distortion_full_labels, distortion_pred_vector = combine_distortion_predictions(
            text_codes, image_codes, cross_codes
        )
        distortion_gt_vector = gt_distortion_to_binary_vector(distortion_taxonomy_gt)

        answer_dict.update({
            "key_entity": key_entity,
            "wiki_knowledge": wiki_knowledge,
            "image_description": img_descrip,
            "stage_signals": {
                "text_signal": text_signal,
                "image_signal": image_signal,
                "cross_signal": cross_signal,
            },
            "stage_distortion_codes": {
                "text_codes": text_codes,
                "image_codes": image_codes,
                "cross_codes": cross_codes,
            },
            "final_verdict": final_verdict,
            "rule_meta": rule_meta,
            "distortion_taxonomy_pred": distortion_full_labels,
            "distortion_pred_vs_gt": {
                "pred_vector": distortion_pred_vector,
                "gt_vector": distortion_gt_vector,
                "gt_labels": list(distortion_taxonomy_gt),
            },
            "questions_per_stage": question_all,
            "answers_per_stage": answer_all,
            "provided_evidence": provided_evidence,
            "raw_calls": raw_calls,
            "fallback": False,
        })

        _write_json(per_sample_dir / "result.json", answer_dict)
        return answer_dict, False
    except Exception as exc:
        error_msg = traceback.format_exc()
        import random as _rnd
        fallback_verdict = _rnd.choice(VERDICT_LABEL_ORDER)

        answer_dict.update({
            "fallback": True,
            "error": str(exc),
            "error_traceback": error_msg,
            "final_verdict": fallback_verdict,
            "rule_meta": {"reason": f"Random fallback due to API failure: {exc}"},
            "distortion_taxonomy_pred": [],
            "distortion_pred_vs_gt": {
                "pred_vector": gt_distortion_to_binary_vector([]),
                "gt_vector": gt_distortion_to_binary_vector(distortion_taxonomy_gt),
                "gt_labels": list(distortion_taxonomy_gt),
            },
            "questions_per_stage": question_all,
            "answers_per_stage": answer_all,
            "raw_calls": raw_calls,
            "stage_signals": {},
            "stage_distortion_codes": {},
            "provided_evidence": provided_evidence,
        })

        print(
            f"\n[WARN] Sample '{sample_name}' failed after retries. "
            f"Fallback verdict: {fallback_verdict}. Error: {exc}",
            flush=True,
        )
        _write_json(per_sample_dir / "result.json", answer_dict)
        return answer_dict, True


# -- Top-level entry called from run_mmd_agent.py ----------------------------


def evaluate_VQA_MMD_Agent_Unified(
    dataset,
    args,
    dataset_name: str,
    run_tag: str,
    serper_key: str,
):
    """Run the unified agent over ``dataset`` and persist all artifacts.

    Output layout (under ``args.answer_path / args.model_name / run_tag``):

        <run_root>/
          run_config.json
          progress.json
          <dataset_name>.jsonl          # running per-sample checkpoint
          <dataset_name>.json           # final consolidated predictions
          metrics.json                  # 5cls + 8label metrics
          samples/
            <sample_name>/
              llm_calls.jsonl
              result.json
    """
    run_root = Path(args.answer_path) / args.model_name / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    answer_path = run_root / f"{dataset_name}.json"
    checkpoint_path = run_root / f"{dataset_name}.jsonl"
    progress_path = run_root / "progress.json"
    metrics_path = run_root / "metrics.json"
    config_path = run_root / "run_config.json"
    samples_root = run_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    redacted_api_key = ""
    if getattr(args, "api_key", ""):
        redacted_api_key = args.api_key[:6] + "..." + args.api_key[-4:]
    _write_json(config_path, {
        "model_name": args.model_name,
        "base_url": args.base_url,
        "api_key_redacted": redacted_api_key,
        "dataset_name": dataset_name,
        "run_tag": run_tag,
        "sampled_root": str(args.sampled_root),
        "prompt_root": str(getattr(args, "prompt_root", "")),
        "answer_path": str(args.answer_path),
        "serper_key_file": str(getattr(args, "serper_key_file", "")),
        "serper_key_index": int(getattr(args, "serper_key_index", 0)),
        "max_samples": int(getattr(args, "max_samples", 0) or 0),
        "max_images": int(getattr(args, "max_images", 0) or 0),
        "temperature": float(getattr(args, "temperature", 0.0)),
        "max_new_tokens": "unlimited (LLM payload omits max_tokens)",
        "request_timeout": int(getattr(args, "request_timeout", 0) or 0),
        "retry_times": int(getattr(args, "retry_times", 0) or 0),
        "retry_interval": int(getattr(args, "retry_interval", 0) or 0),
        "num_workers": int(getattr(args, "num_workers", 1)),
        "image_detail": getattr(args, "image_detail", "low"),
        "started_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    })

    rerun_fallback = bool(getattr(args, "rerun_fallback", False))
    all_predictions = _load_checkpoint_predictions(checkpoint_path)

    if rerun_fallback:
        kept_predictions = [p for p in all_predictions if not p.get("fallback")]
        dropped_fallbacks = [p.get("sample_name") for p in all_predictions if p.get("fallback")]
        if dropped_fallbacks:
            print(
                f"[Resume] --rerun_fallback enabled; will retry "
                f"{len(dropped_fallbacks)} previously failed samples: {dropped_fallbacks[:10]}"
                + (" ..." if len(dropped_fallbacks) > 10 else ""),
                flush=True,
            )
        predictions = kept_predictions
        # rewrite the jsonl without the dropped fallback rows so we don't double-count later
        if dropped_fallbacks:
            tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".rewrite")
            with tmp_path.open("w", encoding="utf-8") as f:
                for row in kept_predictions:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp_path.replace(checkpoint_path)
    else:
        predictions = all_predictions

    completed_sample_names = {item.get("sample_name") for item in predictions}
    _write_progress_file(progress_path, {
        "dataset_name": dataset_name,
        "model_name": args.model_name,
        "completed_count": len(predictions),
        "completed_sample_names": sorted(completed_sample_names),
        "status": "running",
        "rerun_fallback": rerun_fallback,
    })

    pending_samples = [s for s in dataset if s["sample_name"] not in completed_sample_names]
    total_target = len(dataset)
    print(
        f"\nResume status: total_target={total_target}, "
        f"already_completed={len(predictions)}, pending={len(pending_samples)}, "
        f"rerun_fallback={rerun_fallback}",
        flush=True,
    )

    state_lock = threading.Lock()
    max_workers = max(1, int(getattr(args, "num_workers", 1)))

    def _run_one(sample: dict) -> tuple[dict, bool]:
        sample_name = sample["sample_name"]
        per_sample_dir = samples_root / sample_name
        return _process_sample(args, sample, serper_key, per_sample_dir)

    def _print_one_progress(answer_dict: dict, is_fallback: bool) -> None:
        """每完成一个样本，实时打印一行用于 terminal 监控。"""
        sample_name = answer_dict.get("sample_name", "?")
        final_verdict = answer_dict.get("final_verdict", "?")
        gt = answer_dict.get("verdict_gt", "?")
        hit = "✓" if (final_verdict == gt) else "✗"
        pred_codes = answer_dict.get("distortion_taxonomy_pred", []) or []
        gt_codes = answer_dict.get("distortion_pred_vs_gt", {}).get("gt_labels", []) or []
        tag = "[FB]" if is_fallback else "    "
        print(
            f"  {tag} {len(predictions):>4d}/{total_target}  {sample_name:<6s}  "
            f"verdict pred={final_verdict:<14s} gt={gt:<14s} {hit}  "
            f"distort pred={','.join([c[:2] for c in pred_codes]) or 'NONE':<12s} "
            f"gt={','.join([c[:2] for c in gt_codes]) or 'NONE'}",
            flush=True,
        )

    if max_workers == 1:
        for sample in tqdm(pending_samples, desc="Running agent (1 worker)"):
            answer_dict, is_fallback = _run_one(sample)
            sample_name = answer_dict["sample_name"]
            predictions.append(answer_dict)
            completed_sample_names.add(sample_name)
            _append_checkpoint_prediction(checkpoint_path, answer_dict)
            _print_one_progress(answer_dict, is_fallback)
            _write_progress_file(progress_path, {
                "dataset_name": dataset_name,
                "model_name": args.model_name,
                "completed_count": len(predictions),
                "completed_sample_names": sorted(completed_sample_names),
                "last_fallback_sample" if is_fallback else "last_completed_sample": sample_name,
                "status": "running",
            })
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one, sample): sample for sample in pending_samples}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Running agent ({max_workers} workers)"):
                answer_dict, is_fallback = future.result()
                sample_name = answer_dict["sample_name"]
                with state_lock:
                    predictions.append(answer_dict)
                    completed_sample_names.add(sample_name)
                    _append_checkpoint_prediction(checkpoint_path, answer_dict)
                    _print_one_progress(answer_dict, is_fallback)
                    _write_progress_file(progress_path, {
                        "dataset_name": dataset_name,
                        "model_name": args.model_name,
                        "completed_count": len(predictions),
                        "completed_sample_names": sorted(completed_sample_names),
                        "last_fallback_sample" if is_fallback else "last_completed_sample": sample_name,
                        "status": "running",
                    })

    predictions.sort(key=lambda item: item.get("sample_name", ""))
    _write_json(answer_path, predictions)

    verdict_metrics, distortion_metrics = _save_metrics_and_print(
        run_root=run_root,
        metrics_path=metrics_path,
        predictions=predictions,
        model_name=args.model_name,
        run_tag=run_tag,
    )

    _write_progress_file(progress_path, {
        "dataset_name": dataset_name,
        "model_name": args.model_name,
        "completed_count": len(predictions),
        "completed_sample_names": sorted(completed_sample_names),
        "status": "completed",
        "finished_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    })

    print(f"\nAll outputs saved under: {run_root}", flush=True)
    return {
        "run_root": str(run_root),
        "answer_path": str(answer_path),
        "metrics_path": str(metrics_path),
        "verdict_metrics": verdict_metrics,
        "distortion_metrics": distortion_metrics,
    }
