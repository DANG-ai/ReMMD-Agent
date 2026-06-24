"""Batch runner with thread-pool concurrency and resume-from-disk support."""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .data import BenchSample, load_sample, list_sample_ids
from .embedder import EmbeddingClient
from .llm import LLMClient
from .logging_utils import append_summary_row, sample_done, sample_result_path
from .pipeline import run_sample
from .rag import RagIndex


logger = logging.getLogger("remmd.runner")


def _load_existing_result(run_dir: Path, sid: str) -> dict[str, Any] | None:
    p = sample_result_path(run_dir, sid)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[sid=%s] could not load existing result: %s", sid, exc)
        return None


def run_batch(
    sample_ids: list[str],
    *,
    cfg: dict[str, Any],
    llm: LLMClient,
    embedder: EmbeddingClient,
    index: RagIndex,
    run_dir: Path,
    concurrency: int = 8,
    resume: bool = True,
) -> list[dict[str, Any]]:
    bench_root = cfg["paths"]["bench_root"]
    results: list[dict[str, Any]] = []
    # collect skip vs do
    todo: list[str] = []
    for sid in sample_ids:
        if resume and sample_done(run_dir, sid):
            existing = _load_existing_result(run_dir, sid)
            if existing is not None:
                results.append(existing)
                continue
        todo.append(sid)
    logger.info("resume: %d/%d already done; %d to run", len(results), len(sample_ids), len(todo))

    summary_lock = threading.Lock()
    pbar = tqdm(total=len(todo), desc="agent", unit="sample")

    def _worker(sid: str) -> dict[str, Any]:
        try:
            sample = load_sample(bench_root, sid)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[sid=%s] load_sample failed", sid)
            return {"sample_id": sid, "status": "load_failed", "errors": [{"stage": "load", "error": str(exc)}]}
        try:
            r = run_sample(
                sample,
                llm=llm,
                embedder=embedder,
                index=index,
                cfg=cfg,
                run_dir=run_dir,
                save_artifacts=True,
            )
            return r
        except Exception as exc:  # noqa: BLE001
            logger.exception("[sid=%s] unhandled error in run_sample", sid)
            return {
                "sample_id": sid,
                "status": "runtime_failed",
                "errors": [{"stage": "run_sample", "error": str(exc)}],
            }

    if concurrency <= 1:
        for sid in todo:
            r = _worker(sid)
            results.append(r)
            with summary_lock:
                append_summary_row(run_dir, _summary_row_from_result(r))
            pbar.update(1)
        pbar.close()
        return results

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_worker, sid): sid for sid in todo}
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                r = {"sample_id": sid, "status": "future_failed", "errors": [{"stage": "future", "error": str(exc)}]}
            results.append(r)
            with summary_lock:
                append_summary_row(run_dir, _summary_row_from_result(r))
            pbar.update(1)
    pbar.close()
    return results


def _summary_row_from_result(r: dict[str, Any]) -> dict[str, Any]:
    judge = r.get("judge") or {}
    gold = r.get("gold") or {}
    return {
        "sample_id": r.get("sample_id"),
        "status": r.get("status"),
        "n_errors": len(r.get("errors") or []),
        "pred_verdict": judge.get("level1_verdict"),
        "gold_verdict": gold.get("verdict"),
        "pred_taxonomy": judge.get("level2_taxonomy"),
        "gold_taxonomy": gold.get("distortion_taxonomy"),
        "language_code": r.get("language_code"),
        "region_code": r.get("region_code"),
        "timings": r.get("timings"),
    }


def reload_all_results(run_dir: Path, sample_ids: list[str]) -> list[dict[str, Any]]:
    """Re-read result.json for each sample from disk (useful for offline re-eval)."""
    out = []
    for sid in sample_ids:
        r = _load_existing_result(run_dir, sid)
        if r:
            out.append(r)
    return out


def collect_sample_ids(
    bench_root: str,
    *,
    limit: int | None = None,
    explicit: list[str] | None = None,
    seed: int = 42,
) -> list[str]:
    if explicit:
        return [s.strip() for s in explicit if s.strip()]
    all_ids = list_sample_ids(bench_root)
    if limit is not None and limit < len(all_ids):
        return all_ids[:limit]
    return all_ids
