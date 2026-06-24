#!/usr/bin/env python
"""Re-parse all `04_judge_llm_raw.json` files in a run directory using the
current minimal parser, regenerate `04_judge_parsed.json` and `result.json`,
and recompute metrics. Cheap (no LLM calls).

The current framework has the LLM judge produce L1 and L2 directly; we only
enforce two consistency rules in `parse_judge_output`:
  - L2 non-empty + L1=='True'  → bump L1 to 'Mostly True'
  - L2 empty + L1!='True' + clean findings → True

This script lets you re-evaluate a finished run after tweaking the prompt
of the judge — but note that prompt changes only affect FUTURE runs, not
existing raw outputs. Use `--metrics-only` instead if you only want fresh
plots.

Usage:
    python scripts/recompute_from_raw.py --run-dir runs/qwen3.5-9b_<ts>_<tag>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.final_judge import parse_judge_output, _enforce_l1_l2_coupling, _summarize_findings
from src.labels import normalize_level2_list
from src.metrics import compute_metrics, render_text_summary, save_metrics
from src.plotting import plot_confusion_matrix_l1, plot_level2_per_class_bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"))
    ap.add_argument("--policy", default=None,
                    choices=["off", "always", "selective"],
                    help="override pipeline.analyzer_union_policy from config")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    cfg = yaml.safe_load(open(args.config))

    policy = args.policy
    if policy is None:
        policy = cfg.get("pipeline", {}).get("analyzer_union_policy")
    if policy is None:
        legacy = cfg.get("pipeline", {}).get("use_analyzer_l2_union")
        if legacy is True:
            policy = "always"
        elif legacy is False:
            policy = "off"
        else:
            policy = "always"
    policy = str(policy).lower()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    logging.info("recomputing with analyzer_union_policy=%s", policy)

    samples_dir = run_dir / "samples"
    sample_ids = sorted(p.name for p in samples_dir.iterdir() if p.is_dir())
    logging.info("found %d samples in %s", len(sample_ids), run_dir)
    n_reparsed = 0
    n_unchanged = 0
    n_errors = 0
    results = []
    for sid in sample_ids:
        sdir = samples_dir / sid
        result_path = sdir / "result.json"
        raw_path = sdir / "04_judge_llm_raw.json"
        if not result_path.exists():
            continue
        with open(result_path, "r", encoding="utf-8") as f:
            r = json.load(f)
        if not raw_path.exists():
            results.append(r)
            n_unchanged += 1
            continue
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        content = raw.get("content") or ""
        try:
            new_judge = parse_judge_output(content)
            judge_l1_pre = new_judge.get("level1_verdict")
            judge_l2_pre = list(new_judge.get("level2_taxonomy") or [])
            # Multi-LLM L2 signal aggregation — selective union with analysers.
            extras: list[str] = []
            ia_active = []
            ta_active = []
            ia_path = sdir / "034_image_analyze_parsed.json"
            if ia_path.exists():
                with open(ia_path, "r", encoding="utf-8") as f:
                    ia = json.load(f)
                ia_active = normalize_level2_list(ia.get("active_labels") or [])
            ta_path = sdir / "033_text_analyze_parsed.json"
            if ta_path.exists():
                with open(ta_path, "r", encoding="utf-8") as f:
                    ta = json.load(f)
                align = (ta.get("alignment_level") or "").upper()
                if align in ("PARTIALLY_ALIGNED", "MISALIGNED"):
                    ta_active = normalize_level2_list(ta.get("active_labels") or [])

            do_union = False
            if policy == "always":
                do_union = True
            elif policy == "off":
                do_union = False
            elif policy == "selective":
                do_union = bool(judge_l2_pre) or (judge_l1_pre != "True")
            if do_union:
                extras.extend(ia_active)
                extras.extend(ta_active)

            if extras:
                judge_l2 = list(judge_l2_pre)
                for l in extras:
                    if l not in judge_l2:
                        judge_l2.append(l)
                new_judge["level2_taxonomy"] = judge_l2
                new_judge["level2_taxonomy_source"] = f"judge_union_analyzers_{policy}"
                counts = _summarize_findings(new_judge.get("subclaim_findings"))
                new_l1, rule = _enforce_l1_l2_coupling(new_judge["level1_verdict"], judge_l2, counts)
                if rule is not None:
                    new_judge["level1_verdict"] = new_l1
                    new_judge["level1_coupling_rule_applied"] = (
                        (new_judge.get("level1_coupling_rule_applied") or "") + ";" + rule
                    ).strip(";")
            else:
                new_judge["level2_taxonomy_source"] = f"judge_only_policy={policy}"
                counts = _summarize_findings(new_judge.get("subclaim_findings"))
                new_l1, rule = _enforce_l1_l2_coupling(
                    new_judge.get("level1_verdict"), judge_l2_pre, counts)
                if rule is not None:
                    new_judge["level1_verdict"] = new_l1
                    new_judge["level1_coupling_rule_applied"] = (
                        (new_judge.get("level1_coupling_rule_applied") or "") + ";" + rule
                    ).strip(";")
            r["judge"] = new_judge
            r["status"] = "ok"
            r["errors"] = [e for e in (r.get("errors") or []) if e.get("stage") != "final_judge"]
            with open(sdir / "04_judge_parsed.json", "w", encoding="utf-8") as f:
                json.dump(new_judge, f, ensure_ascii=False, indent=2)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
            n_reparsed += 1
        except Exception as exc:  # noqa: BLE001
            n_errors += 1
            logging.warning("[sid=%s] parse failure: %s", sid, exc)
        results.append(r)

    logging.info("reparsed=%d unchanged=%d errors=%d", n_reparsed, n_unchanged, n_errors)

    metrics = compute_metrics(results)
    save_metrics(metrics, run_dir / "metrics")
    txt = render_text_summary(metrics)
    print(txt)
    with open(run_dir / "metrics" / "summary.txt", "w", encoding="utf-8") as f:
        f.write(txt)

    if "error" not in metrics:
        L1 = metrics["level1"]
        plot_confusion_matrix_l1(
            L1["confusion_matrix"],
            accuracy=L1["accuracy"], macro_f1=L1["macro_f1"],
            out_path_base=run_dir / "metrics" / "confusion_matrix_l1",
            title="ReMMDBench Level-1 Verdict",
            model_name=cfg["llm"]["model"],
            n_samples=metrics["n_eligible_for_eval"],
        )
        plot_level2_per_class_bars(
            metrics["level2"]["per_class"],
            out_path_base=run_dir / "metrics" / "level2_per_class_bars",
            title="ReMMDBench Level-2 per-class metrics",
            model_name=cfg["llm"]["model"],
        )


if __name__ == "__main__":
    main()
