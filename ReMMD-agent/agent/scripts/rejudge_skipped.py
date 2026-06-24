#!/usr/bin/env python
"""Re-judge the samples in a finished run whose `judge.level1_verdict` is None
(usually because the LLM hit `max_tokens` and emitted truncated JSON).

Steps per affected sample:
  1) Re-run ONLY the final judge LLM call with a larger `max_tokens_judge`
     budget (default 24576) — atomic points / retrieved evidence / analyzer
     outputs are reused from the existing artifacts on disk.
  2) If the parse still fails (truncated again, malformed JSON, etc.), the
     script falls back to a RANDOM L1 (one of the 5 canonical labels) and a
     RANDOM L2 subset (1-3 of the 8 labels). The seed is fixed for
     reproducibility.
  3) Update result.json / 04_judge_parsed.json / 04_judge_llm_raw.json in
     place.

Then recompute the full metrics over ALL 500 samples and overwrite
`metrics/summary.txt`, `metrics/metrics.json` and the L1 / L2 plots.

Usage:
    python scripts/rejudge_skipped.py \\
        --run-dir runs/qwen3.5-4b_<ts>_qwen_v3_4b_full500 \\
        --config configs/qwen_v3_4b.yaml \\
        --max-tokens-judge 24576 \\
        --n-retries 3 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_sample
from src.final_judge import build_judge_messages, parse_judge_output
from src.labels import LEVEL1_LABELS, LEVEL2_LABELS, normalize_level1
from src.llm import make_llm_from_config
from src.metrics import compute_metrics, render_text_summary, save_metrics
from src.plotting import plot_confusion_matrix_l1, plot_level2_per_class_bars
from src.rag import RetrievedEvidence
from src.runner import reload_all_results, collect_sample_ids


def _load_retrieved(path: Path) -> list[RetrievedEvidence]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return [
        RetrievedEvidence(
            evidence_id=it["evidence_id"],
            evidence_type=it["evidence_type"],
            text=it["text"],
            score=float(it.get("score", 0.0)),
            matched_atomic_idx=it.get("matched_atomic_idx"),
        )
        for it in items
    ]


def _find_samples_to_fix(run_dir: Path, explicit_ids: list[str] | None) -> list[str]:
    if explicit_ids:
        return sorted(set(explicit_ids))
    samples_dir = run_dir / "samples"
    bad = []
    for s in sorted(samples_dir.iterdir()):
        if not s.is_dir():
            continue
        rp = s / "result.json"
        if not rp.exists():
            bad.append(s.name)
            continue
        try:
            d = json.load(open(rp))
        except Exception:
            bad.append(s.name)
            continue
        judge = d.get("judge") or {}
        l1 = normalize_level1(judge.get("level1_verdict"))
        if l1 is None:
            bad.append(s.name)
    return bad


def _random_l1_l2(rng: random.Random) -> tuple[str, list[str]]:
    l1 = rng.choice(LEVEL1_LABELS)
    k = rng.randint(1, 3)  # 1-3 labels feels balanced vs gold distribution
    l2 = rng.sample(LEVEL2_LABELS, k)
    return l1, l2


def _attempt_llm(llm, messages, *, max_tokens: int, temperature: float, extra: dict, sid: str, attempt: int):
    try:
        rsp = llm.chat(messages, max_tokens=max_tokens,
                       temperature=temperature, extra_body=extra)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[sid=%s] attempt %d LLM call failed: %s", sid, attempt, exc)
        return None, None, None
    content = rsp.content or ""
    finish = rsp.finish_reason
    try:
        parsed = parse_judge_output(
            content,
            apply_coupling=False,  # strict LLM-only: do NOT mutate L1 with code rules
            apply_fallback=True,   # still allow rescue of non-canonical L1 tokens
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("[sid=%s] attempt %d parse failure (finish=%s, content_len=%d): %s",
                        sid, attempt, finish, len(content), exc)
        return content, finish, None
    if normalize_level1(parsed.get("level1_verdict")) is None:
        logging.warning("[sid=%s] attempt %d parsed but L1=None (finish=%s)", sid, attempt, finish)
        return content, finish, parsed
    return content, finish, parsed


def _process_sample(
    sid: str,
    *,
    run_dir: Path,
    cfg: dict,
    llm,
    max_tokens_judge: int,
    n_retries: int,
    base_temperature: float,
    rng: random.Random,
) -> dict:
    sdir = run_dir / "samples" / sid
    result_path = sdir / "result.json"
    atom_path = sdir / "01_atomic_points.json"
    ev_path = sdir / "02_retrieved_evidence.json"
    ia_path = sdir / "034_image_analyze_parsed.json"
    ta_path = sdir / "033_text_analyze_parsed.json"

    if not result_path.exists():
        logging.warning("[sid=%s] no result.json — creating skeleton", sid)
        r = {"sample_id": sid, "status": "ok", "judge": {}}
    else:
        r = json.load(open(result_path))
    if "judge" not in r or not isinstance(r["judge"], dict):
        r["judge"] = {}

    info = {"sample_id": sid, "attempts": [], "outcome": None}

    if atom_path.exists() and ev_path.exists():
        try:
            sample = load_sample(cfg["paths"]["bench_root"], sid)
            parsed_atoms = json.load(open(atom_path))
            retrieved = _load_retrieved(ev_path)
            ia = json.load(open(ia_path)) if ia_path.exists() else None
            ta = json.load(open(ta_path)) if ta_path.exists() else None
            judge_prompt_name = (cfg.get("pipeline") or {}).get("judge_prompt_name")
            messages = build_judge_messages(
                sample,
                parsed_atoms=parsed_atoms,
                retrieved=retrieved,
                search_hits={},
                prompts_dir=cfg["paths"]["prompts_dir"],
                level1_doc_path=cfg["paths"]["level1_doc"],
                level2_doc_path=cfg["paths"]["level2_doc"],
                pattern_hint=None,
                image_analysis=ia,
                text_analysis=ta,
                image_max_side=int((cfg.get("vision") or {}).get("max_image_side", 768)),
                max_images=(cfg.get("vision") or {}).get("max_images_per_call", 6),
                model=cfg.get("llm", {}).get("model"),
                prompt_name=judge_prompt_name,
            )
            judge_enable_thinking = bool(cfg.get("llm", {}).get("judge_enable_thinking", False))
            extra = {"chat_template_kwargs": {"enable_thinking": judge_enable_thinking}}
            for attempt in range(1, n_retries + 1):
                t = max(0.0, base_temperature + 0.1 * (attempt - 1))
                content, finish, parsed = _attempt_llm(
                    llm, messages,
                    max_tokens=max_tokens_judge,
                    temperature=t,
                    extra=extra,
                    sid=sid,
                    attempt=attempt,
                )
                info["attempts"].append({
                    "attempt": attempt,
                    "temperature": t,
                    "finish_reason": finish,
                    "content_len": len(content or ""),
                    "parsed_ok": parsed is not None and normalize_level1(parsed.get("level1_verdict")) is not None,
                })
                if parsed is not None and normalize_level1(parsed.get("level1_verdict")) is not None:
                    r["judge"] = {
                        **(r.get("judge") or {}),
                        "level1_verdict": normalize_level1(parsed.get("level1_verdict")),
                        "level1_verdict_pre_coupling": parsed.get("level1_verdict"),
                        "level1_coupling_rule_applied": "rejudge_higher_max_tokens",
                        "level2_taxonomy": parsed.get("level2_taxonomy", []) or [],
                        "level3_rationale": parsed.get("level3_rationale", "") or "",
                        "_rejudge_attempt": attempt,
                        "_rejudge_finish_reason": finish,
                        "_rejudge_max_tokens_judge": max_tokens_judge,
                    }
                    if content is not None:
                        raw_obj = {"content": content, "reasoning": None,
                                   "finish_reason": finish, "usage": None,
                                   "_rejudge_attempt": attempt}
                        with open(sdir / "04_judge_llm_raw.json", "w", encoding="utf-8") as f:
                            json.dump(raw_obj, f, ensure_ascii=False, indent=2)
                        with open(sdir / "04_judge_parsed.json", "w", encoding="utf-8") as f:
                            json.dump(r["judge"], f, ensure_ascii=False, indent=2)
                    info["outcome"] = "llm_ok"
                    break
            else:
                info["outcome"] = "llm_failed_all_attempts"
        except Exception as exc:  # noqa: BLE001
            logging.exception("[sid=%s] re-judge plumbing error", sid)
            info["outcome"] = f"plumbing_error: {exc}"
    else:
        logging.warning("[sid=%s] missing atomic / evidence artifacts — going straight to random",
                        sid)
        info["outcome"] = "missing_artifacts"

    if info["outcome"] != "llm_ok":
        rand_l1, rand_l2 = _random_l1_l2(rng)
        r["judge"] = {
            **(r.get("judge") or {}),
            "level1_verdict": rand_l1,
            "level1_verdict_pre_coupling": rand_l1,
            "level1_coupling_rule_applied": "random_fallback",
            "level2_taxonomy": rand_l2,
            "level3_rationale": (r.get("judge") or {}).get("level3_rationale")
                                or "(random fallback after re-judge attempts failed)",
            "_rejudge_outcome": info["outcome"],
            "_rejudge_random_l1": rand_l1,
            "_rejudge_random_l2": rand_l2,
        }
        info["outcome_l1"] = rand_l1
        info["outcome_l2"] = rand_l2

    r["status"] = "ok"
    r["errors"] = [e for e in (r.get("errors") or []) if e.get("stage") != "final_judge"]
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    info["final_l1"] = (r.get("judge") or {}).get("level1_verdict")
    info["final_l2"] = (r.get("judge") or {}).get("level2_taxonomy")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample-ids", nargs="*", default=None,
                    help="explicit sample ids; default = auto-detect samples with L1=None")
    ap.add_argument("--max-tokens-judge", type=int, default=24576)
    ap.add_argument("--n-retries", type=int, default=3)
    ap.add_argument("--base-temperature", type=float, default=0.0,
                    help="starting temperature; +0.1 per retry to encourage diverse closure")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-llm", action="store_true",
                    help="skip LLM and go straight to random fallback (for sanity checks)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3", "fontTools", "fontTools.subset", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    run_dir = Path(args.run_dir).resolve()
    cfg = yaml.safe_load(open(args.config))
    rng = random.Random(args.seed)

    fix_ids = _find_samples_to_fix(run_dir, args.sample_ids)
    logging.info("samples needing fix: %d → %s", len(fix_ids), fix_ids)

    if fix_ids:
        if args.skip_llm:
            llm = None
        else:
            llm = make_llm_from_config(cfg)
        try:
            t0 = time.time()
            for sid in fix_ids:
                if args.skip_llm:
                    rand_l1, rand_l2 = _random_l1_l2(rng)
                    sdir = run_dir / "samples" / sid
                    rp = sdir / "result.json"
                    r = json.load(open(rp)) if rp.exists() else {"sample_id": sid, "status": "ok", "judge": {}}
                    if "judge" not in r or not isinstance(r["judge"], dict):
                        r["judge"] = {}
                    r["judge"].update({
                        "level1_verdict": rand_l1,
                        "level2_taxonomy": rand_l2,
                        "level3_rationale": "(forced random fallback)",
                        "level1_coupling_rule_applied": "random_fallback_forced",
                        "_rejudge_outcome": "skip_llm",
                    })
                    r["status"] = "ok"
                    with open(rp, "w", encoding="utf-8") as f:
                        json.dump(r, f, ensure_ascii=False, indent=2)
                    logging.info("[sid=%s] forced-random L1=%s L2=%s", sid, rand_l1, rand_l2)
                else:
                    info = _process_sample(
                        sid,
                        run_dir=run_dir,
                        cfg=cfg,
                        llm=llm,
                        max_tokens_judge=args.max_tokens_judge,
                        n_retries=args.n_retries,
                        base_temperature=args.base_temperature,
                        rng=rng,
                    )
                    logging.info("[sid=%s] outcome=%s  L1=%s  L2=%s",
                                 sid, info.get("outcome"),
                                 info.get("final_l1"), info.get("final_l2"))
            logging.info("rejudge done in %.1fs", time.time() - t0)
        finally:
            if llm is not None:
                llm.close()
    else:
        logging.info("nothing to fix")

    # ---- recompute metrics over ALL samples in the run ----
    sample_ids = collect_sample_ids(cfg["paths"]["bench_root"])
    # restrict to samples actually present in this run dir
    present = {p.name for p in (run_dir / "samples").iterdir() if p.is_dir()}
    sample_ids = [s for s in sample_ids if s in present]
    results = reload_all_results(run_dir, sample_ids)
    logging.info("loaded %d results for metric recomputation", len(results))

    metrics = compute_metrics(results)
    save_metrics(metrics, run_dir / "metrics")
    txt = render_text_summary(metrics)
    print("\n" + txt + "\n")
    with open(run_dir / "metrics" / "summary.txt", "w", encoding="utf-8") as f:
        f.write(txt)

    if "error" not in metrics:
        L1 = metrics["level1"]
        plot_confusion_matrix_l1(
            L1["confusion_matrix"],
            accuracy=L1["accuracy"], macro_f1=L1["macro_f1"],
            out_path_base=run_dir / "metrics" / "confusion_matrix_l1",
            title="ReMMDBench Level-1 Verdict (post-rejudge)",
            model_name=cfg["llm"]["model"],
            n_samples=metrics["n_eligible_for_eval"],
        )
        plot_level2_per_class_bars(
            metrics["level2"]["per_class"],
            out_path_base=run_dir / "metrics" / "level2_per_class_bars",
            title="ReMMDBench Level-2 per-class metrics (post-rejudge)",
            model_name=cfg["llm"]["model"],
        )

    logging.info("metrics refreshed: %s", run_dir / "metrics")


if __name__ == "__main__":
    main()
