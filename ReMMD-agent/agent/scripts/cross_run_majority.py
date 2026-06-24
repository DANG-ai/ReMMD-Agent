#!/usr/bin/env python
"""Cross-run majority vote: build a new "ensemble" run directory whose
per-sample L1 verdict is the majority vote across N source runs (and L2 is
a per-label union). All votes come from LLM-emitted L1/L2 fields — there
are NO code-side L1↔L2 coupling rules and NO analyser-L2 unions; the
function is a pure aggregation of LLM outputs across the N source runs.

Usage:
  python scripts/cross_run_majority.py \
      --run-dirs runs/qwen3.5-9b_v3_full500 runs/qwen3.5-9b_v11_ens \
      --tag v3+v11ens \
      --tie-break median  # or 'severe', 'mild'
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.labels import LEVEL2_LABELS
from src.metrics import compute_metrics, render_text_summary, save_metrics
from src.plotting import plot_confusion_matrix_l1, plot_level2_per_class_bars

SEVERITY = {"True": 0, "Mostly True": 1, "Mixture": 2, "Mostly False": 3, "False": 4}


def majority_l1(votes: list[str], mode: str) -> str:
    if not votes:
        return ""
    c = Counter(votes)
    top = c.most_common(1)[0][1]
    cands = [v for v, n in c.items() if n == top]
    if len(cands) == 1:
        return cands[0]
    if mode == "severe":
        return sorted(cands, key=lambda v: SEVERITY.get(v, 0))[-1]
    if mode == "mild":
        return sorted(cands, key=lambda v: SEVERITY.get(v, 0))[0]
    sorted_votes = sorted(votes, key=lambda v: SEVERITY.get(v, 0))
    return sorted_votes[len(sorted_votes) // 2]


def majority_l2(votes: list[list[str]]) -> list[str]:
    """Per-label vote: include L if at least ceil(N/2) votes have it."""
    n = len(votes)
    if n == 0:
        return []
    threshold = max(1, (n // 2) + (n % 2))
    counts = Counter()
    for v in votes:
        for l in set(v):
            counts[l] += 1
    return [l for l in LEVEL2_LABELS if counts.get(l, 0) >= threshold]


def winning_l3(l1_votes, l3_votes, winning_l1):
    for lv, l3 in zip(l1_votes, l3_votes):
        if lv == winning_l1 and l3:
            return l3
    return l3_votes[0] if l3_votes else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--tag", default="cross_run_majority")
    ap.add_argument("--tie-break", choices=["median", "severe", "mild"], default="median")
    ap.add_argument("--out-base", default=None,
                    help="parent dir for output (default: runs)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("matplotlib",):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    run_dirs = [Path(d).resolve() for d in args.run_dirs]
    out_base = Path(args.out_base).resolve() if args.out_base else run_dirs[0].parent
    out_dir = out_base / f"cross_run_{args.tag}_tie-{args.tie_break}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)

    # Load each run's results
    runs = []
    for d in run_dirs:
        rmap = {}
        for s in (d / "samples").iterdir():
            rj = s / "result.json"
            if rj.exists():
                rmap[s.name] = json.loads(rj.read_text())
        runs.append((d.name, rmap))
        logging.info("loaded %d samples from %s", len(rmap), d.name)

    # Common samples
    sids = sorted(set.intersection(*[set(r.keys()) for _, r in runs]))
    logging.info("common samples across %d runs: %d", len(runs), len(sids))

    n_correct = 0
    all_results = []
    for sid in sids:
        # Collect L1/L2/L3 votes
        l1_votes = []
        l2_votes = []
        l3_votes = []
        sample_results = [r[sid] for _, r in runs]
        for sr in sample_results:
            j = sr.get("judge") or {}
            l1 = j.get("level1_verdict")
            if l1:
                l1_votes.append(l1)
            l2 = j.get("level2_taxonomy") or []
            l2_votes.append(list(l2))
            l3 = j.get("level3_rationale") or ""
            l3_votes.append(l3)

        ens_l1 = majority_l1(l1_votes, args.tie_break)
        ens_l2 = majority_l2(l2_votes)
        ens_l3 = winning_l3(l1_votes, l3_votes, ens_l1)

        # Build new result.json — start from the first run's result and overwrite judge
        base = copy.deepcopy(sample_results[0])
        base["judge"] = {
            **(base.get("judge") or {}),
            "level1_verdict": ens_l1,
            "level1_verdict_pre_coupling": ens_l1,
            "level1_coupling_rule_applied": "cross_run_majority_vote",
            "level2_taxonomy": ens_l2,
            "level2_taxonomy_source": "cross_run_majority_l2_threshold",
            "level3_rationale": ens_l3,
            "ensemble_votes_l1": l1_votes,
            "ensemble_votes_l2": l2_votes,
            "ensemble_votes_runs": [n for n, _ in runs],
            "ensemble_n_successful": len(l1_votes),
        }

        # Write per-sample
        sample_out = out_dir / "samples" / sid
        sample_out.mkdir(exist_ok=True)
        (sample_out / "result.json").write_text(json.dumps(base, ensure_ascii=False, indent=2))

        # Track for metrics
        gold = (base.get("gold") or {}).get("verdict")
        if ens_l1 == gold:
            n_correct += 1
        all_results.append(base)

    logging.info("naive accuracy: %.2f%%", 100.0 * n_correct / max(1, len(sids)))

    # Metrics
    metrics = compute_metrics(all_results)
    save_metrics(metrics, out_dir / "metrics")
    summary = render_text_summary(metrics)
    (out_dir / "metrics" / "summary.txt").write_text(summary)
    print(summary)

    l1m = metrics.get("level1") or {}
    plot_confusion_matrix_l1(
        l1m.get("confusion_matrix") or [],
        accuracy=l1m.get("accuracy") or 0.0,
        macro_f1=l1m.get("macro_f1") or 0.0,
        out_path_base=out_dir / "metrics" / "confusion_matrix_l1",
        n_samples=len(all_results),
    )
    l2m = metrics.get("level2") or {}
    plot_level2_per_class_bars(
        l2m.get("per_class") or {},
        out_path_base=out_dir / "metrics" / "level2_per_class_bars",
    )
    logging.info("done: %s", out_dir)


if __name__ == "__main__":
    main()
