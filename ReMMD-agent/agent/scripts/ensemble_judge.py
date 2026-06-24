#!/usr/bin/env python
"""Re-judge an existing run with N independent judge calls per sample,
then majority-vote on L1 (and union L2). Cheap-ish: re-uses cached
atomic points + retrieved evidence; only the judge LLM call is repeated.

Usage:
    python scripts/ensemble_judge.py --run-dir runs/qwen3.5-9b_<ts>_<tag> --n 3 --temperature 0.5
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_sample
from src.final_judge import build_judge_messages, parse_judge_output
from src.labels import LEVEL2_LABELS, normalize_level1, normalize_level2_list
from src.llm import make_llm_from_config
from src.metrics import compute_metrics, render_text_summary, save_metrics
from src.plotting import plot_confusion_matrix_l1, plot_level2_per_class_bars
from src.rag import RetrievedEvidence


def _load_retrieved(path: Path) -> list[RetrievedEvidence]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out = []
    for it in items:
        out.append(RetrievedEvidence(
            evidence_id=it["evidence_id"],
            evidence_type=it["evidence_type"],
            text=it["text"],
            score=float(it.get("score", 0.0)),
            matched_atomic_idx=it.get("matched_atomic_idx"),
        ))
    return out


def _majority_l1(votes: list[str]) -> str:
    """Tie-break by LEVEL1 ordinal severity (prefer harder verdict on ties).

    Severity rank: True < Mostly True < Mixture < Mostly False < False.
    On ties the *median* of the votes wins (more informative than max).
    """
    from src.labels import LEVEL1_LABELS
    rank = {l: i for i, l in enumerate(LEVEL1_LABELS)}
    counts = Counter(votes)
    top_freq = max(counts.values())
    top = [v for v, c in counts.items() if c == top_freq]
    if len(top) == 1:
        return top[0]
    # tie: pick median rank among the votes
    sorted_votes = sorted(votes, key=lambda v: rank.get(v, 0))
    return sorted_votes[len(sorted_votes) // 2]


def _union_l2(votes: list[list[str]], threshold_frac: float = 0.34) -> list[str]:
    """Include label if predicted in at least `threshold_frac` of votes."""
    n = len(votes)
    if n == 0:
        return []
    threshold = max(1, int(n * threshold_frac + 0.5))
    counts = Counter()
    for v in votes:
        for l in set(v):
            counts[l] += 1
    return [l for l in LEVEL2_LABELS if counts.get(l, 0) >= threshold]


def _majority_l2(votes: list[list[str]]) -> list[str]:
    """Majority-vote each label: include L if >= ceil(N/2) votes have L."""
    n = len(votes)
    if n == 0:
        return []
    threshold = (n // 2) + 1
    counts = Counter()
    for v in votes:
        for l in set(v):
            counts[l] += 1
    return [l for l in LEVEL2_LABELS if counts.get(l, 0) >= threshold]


def _l3_from_winning_vote(l1_votes: list[str], l3_votes: list[str], winning_l1: str) -> str:
    """Pick the L3 rationale from the LLM call whose L1 matches the ensemble winner.
    If no call matches, fall back to the L3 from the first call (still an LLM
    output — never synthesized by code).
    """
    for lv, l3 in zip(l1_votes, l3_votes):
        if lv == winning_l1:
            return l3
    return l3_votes[0] if l3_votes else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"))
    ap.add_argument("--n", type=int, default=3, help="number of judge calls per sample")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-tokens-judge", type=int, default=12288)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out-tag", default="ensemble")
    ap.add_argument("--llm-only-mode", action="store_true",
                    help="Strict LLM-only mode: NO code-side rule overrides "
                         "L1 / L2 / L3. Per-call L1 = LLM raw verbatim. "
                         "Ensemble L1 = majority vote of N LLM L1 votes. "
                         "Ensemble L2 = union over the N votes (>= floor(N/2)+1). "
                         "Ensemble L3 = the L3 from the LLM call whose L1 wins. "
                         "Skips both _enforce_l1_l2_coupling and analyzer-L2 union.")
    ap.add_argument("--judge-prompt-name", default=None,
                    help="Override pipeline.judge_prompt_name from the config "
                         "(e.g. 'final_judge_v5' to use a v5 prompt with v3 results).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3", "fontTools", "fontTools.subset", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir.parent / (run_dir.name + f"_{args.out_tag}_n{args.n}_t{args.temperature}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    llm = make_llm_from_config(cfg)

    sample_ids = sorted(p.name for p in (run_dir / "samples").iterdir() if p.is_dir())
    logging.info("ensembling %d samples with n=%d temp=%.2f", len(sample_ids), args.n, args.temperature)

    def _process(sid: str):
        sdir = run_dir / "samples" / sid
        result_path = sdir / "result.json"
        atom_path = sdir / "01_atomic_points.json"
        ev_path = sdir / "02_retrieved_evidence.json"
        ia_path = sdir / "034_image_analyze_parsed.json"
        ta_path = sdir / "033_text_analyze_parsed.json"
        if not (result_path.exists() and atom_path.exists() and ev_path.exists()):
            return None
        try:
            r = json.load(open(result_path))
            sample = load_sample(cfg["paths"]["bench_root"], sid)
            parsed_atoms = json.load(open(atom_path))
            retrieved = _load_retrieved(ev_path)
            search_hits = {}
            ia = json.load(open(ia_path)) if ia_path.exists() else None
            ta = json.load(open(ta_path)) if ta_path.exists() else None
            judge_prompt_name = (
                args.judge_prompt_name
                or (cfg.get("pipeline") or {}).get("judge_prompt_name")
            )
            messages = build_judge_messages(
                sample,
                parsed_atoms=parsed_atoms,
                retrieved=retrieved,
                search_hits=search_hits,
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
            # Read enable_thinking from the config to honour the user's
            # Qwen-team-recommendation: "all qwen calls should enable thinking".
            judge_enable_thinking = bool(
                cfg.get("llm", {}).get("judge_enable_thinking", False)
            )
            extra = {"chat_template_kwargs": {"enable_thinking": judge_enable_thinking}}
            l1_votes = []
            l2_votes = []
            l3_votes = []
            # Run N independent LLM judge calls in parallel for THIS sample to
            # reduce wall-clock time. Each call uses a slightly different
            # temperature so the votes are diverse but each one is still a
            # "neutral" zero-bias LLM emission.
            from concurrent.futures import ThreadPoolExecutor as _TP
            def _one_call(temp_offset: int):
                t = max(0.0, args.temperature + 0.05 * temp_offset)
                # Use the LLM client's chat method which already honours the
                # qwen-team-recommended sampling defaults (top_p / top_k / min_p
                # / presence_penalty / repetition_penalty) loaded from the config.
                rsp = llm.chat(messages, max_tokens=args.max_tokens_judge,
                               temperature=t, extra_body=extra)
                try:
                    parsed = parse_judge_output(
                        rsp.content,
                        apply_coupling=not args.llm_only_mode,
                        apply_fallback=True,  # always allow non-canonical L1 token rescue
                    )
                    return parsed
                except Exception as exc:  # noqa: BLE001
                    logging.warning("[sid=%s] judge call temp=%.2f parse error: %s",
                                    sid, t, exc)
                    return None
            with _TP(max_workers=args.n) as inner:
                pcalls = list(inner.map(_one_call, range(args.n)))
            for parsed in pcalls:
                if parsed is None:
                    continue
                l1_votes.append(parsed.get("level1_verdict"))
                l2_votes.append(parsed.get("level2_taxonomy") or [])
                l3_votes.append(parsed.get("level3_rationale", ""))
            if not l1_votes:
                return r
            ens_l1 = _majority_l1([v for v in l1_votes if v])
            if args.llm_only_mode:
                # Strict LLM-only: ensemble L2 is a per-label majority vote
                # (>= ceil(N/2) of the N LLM calls flagged it), AND L3 comes
                # from the LLM call whose L1 won. NO _enforce_l1_l2_coupling.
                # NO analyzer-L2 union. Whatever the LLM ensemble emits stays.
                ens_l2 = _majority_l2(l2_votes)
                ens_l3 = _l3_from_winning_vote(l1_votes, l3_votes, ens_l1)
                r2 = copy.deepcopy(r)
                r2["judge"] = {
                    **(r2.get("judge") or {}),
                    "level1_verdict": ens_l1,
                    "level1_verdict_pre_coupling": ens_l1,
                    "level1_coupling_rule_applied": "ensemble_llm_only_mode",
                    "level2_taxonomy": ens_l2,
                    "level3_rationale": ens_l3,
                    "ensemble_votes_l1": l1_votes,
                    "ensemble_votes_l2": l2_votes,
                    "ensemble_n_successful": len(l1_votes),
                }
            else:
                # Legacy path: union L2 + apply coupling + add analyzer L2.
                ens_l2 = _union_l2(l2_votes)
                extras: list[str] = []
                if ia and ia.get("active_labels"):
                    extras.extend(normalize_level2_list(ia["active_labels"]))
                    if "V2 Visual Editing" in (ia.get("active_labels") or []):
                        imgs = ia.get("images") or []
                        if any(
                            isinstance(im, dict)
                            and im.get("matches_evidence_image_content") is False
                            and im.get("v2_edit_present")
                            for im in imgs
                        ):
                            if "C2 Contextual Inconsistency" not in extras:
                                extras.append("C2 Contextual Inconsistency")
                if ta and ta.get("active_labels"):
                    if (ta.get("alignment_level") or "").upper() in ("PARTIALLY_ALIGNED", "MISALIGNED"):
                        extras.extend(normalize_level2_list(ta["active_labels"]))
                for l in extras:
                    if l not in ens_l2:
                        ens_l2.append(l)
                from src.final_judge import _enforce_l1_l2_coupling
                ens_l1_final, rule = _enforce_l1_l2_coupling(ens_l1, ens_l2, None)
                r2 = copy.deepcopy(r)
                r2["judge"] = {
                    **(r2.get("judge") or {}),
                    "level1_verdict": ens_l1_final,
                    "level1_verdict_pre_coupling": ens_l1,
                    "level1_coupling_rule_applied": rule,
                    "level2_taxonomy": ens_l2,
                    "level3_rationale": l3_votes[0] if l3_votes else "",
                    "ensemble_votes_l1": l1_votes,
                    "ensemble_votes_l2": l2_votes,
                }
            r2["status"] = "ok"
            sout = out_dir / "samples" / sid
            sout.mkdir(exist_ok=True)
            with open(sout / "result.json", "w", encoding="utf-8") as f:
                json.dump(r2, f, ensure_ascii=False, indent=2)
            return r2
        except Exception as exc:  # noqa: BLE001
            logging.exception("[sid=%s] failed", sid)
            return None

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(_process, sid): sid for sid in sample_ids}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)
            done = len(results)
            if done % 10 == 0:
                logging.info("ensemble progress: %d / %d (elapsed %.1fs)", done, len(sample_ids), time.time() - t0)
    llm.close()

    metrics = compute_metrics(results)
    save_metrics(metrics, out_dir / "metrics")
    txt = render_text_summary(metrics)
    print(txt)
    with open(out_dir / "metrics" / "summary.txt", "w", encoding="utf-8") as f:
        f.write(txt)

    if "error" not in metrics:
        L1 = metrics["level1"]
        plot_confusion_matrix_l1(
            L1["confusion_matrix"],
            accuracy=L1["accuracy"], macro_f1=L1["macro_f1"],
            out_path_base=out_dir / "metrics" / "confusion_matrix_l1",
            title=f"ReMMDBench Level-1 Verdict (ensemble n={args.n})",
            model_name=cfg["llm"]["model"],
            n_samples=metrics["n_eligible_for_eval"],
        )
        plot_level2_per_class_bars(
            metrics["level2"]["per_class"],
            out_path_base=out_dir / "metrics" / "level2_per_class_bars",
            title=f"ReMMDBench L2 (ensemble n={args.n})",
            model_name=cfg["llm"]["model"],
        )
    logging.info("ensemble done in %.1fs → %s", time.time() - t0, out_dir)


if __name__ == "__main__":
    main()
