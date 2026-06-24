#!/usr/bin/env python
"""End-to-end evaluation entry point.

  python run_eval.py --tag debug100 --limit 100 --concurrency 8

Outputs land in agent/runs/<model>_<timestamp>_<tag>/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedder import make_embedder_from_config
from src.llm import make_llm_from_config
from src.logging_utils import dump_config, make_run_dir, setup_logging
from src.metrics import compute_metrics, render_text_summary, save_metrics
from src.plotting import plot_confusion_matrix_l1, plot_level2_per_class_bars
from src.rag import RagIndex
from src.runner import collect_sample_ids, reload_all_results, run_batch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"))
    ap.add_argument("--tag", default="", help="suffix to add to the run-dir name")
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N samples (ordered by id)")
    ap.add_argument("--sample-ids", nargs="*", default=None, help="explicit list of sample_ids")
    ap.add_argument("--concurrency", type=int, default=None, help="override config concurrency")
    ap.add_argument("--no-resume", action="store_true", help="do not skip already-completed samples")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="path to an existing run dir; reuse its samples/ artifacts and continue there instead of creating a new run dir")
    ap.add_argument("--metrics-only", action="store_true",
                    help="do not run the agent; only recompute metrics from --resume-from")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- run dir ----
    if args.resume_from:
        run_dir = Path(args.resume_from).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "samples").mkdir(exist_ok=True)
        (run_dir / "metrics").mkdir(exist_ok=True)
    else:
        run_dir = make_run_dir(cfg["paths"]["runs_dir"], cfg["llm"]["model"], tag=args.tag)
    setup_logging(run_dir)
    dump_config(run_dir, cfg)
    logging.info("run dir: %s", run_dir)

    sample_ids = collect_sample_ids(
        cfg["paths"]["bench_root"],
        limit=args.limit,
        explicit=args.sample_ids,
    )
    logging.info("will evaluate %d samples", len(sample_ids))

    if args.metrics_only:
        results = reload_all_results(run_dir, sample_ids)
        logging.info("loaded %d results for metrics-only run", len(results))
    else:
        emb = make_embedder_from_config(cfg)
        llm = make_llm_from_config(cfg)
        try:
            idx = RagIndex.build_or_load(
                corpus_path=cfg["paths"]["corpus_jsonl"],
                sample_to_evidence_path=cfg["paths"]["sample_to_evidence"],
                cache_dir=cfg["paths"]["rag_index_dir"],
                embedder=emb,
                model_tag=cfg["embedding"]["model"],
            )
            conc = args.concurrency if args.concurrency is not None else cfg["pipeline"].get("concurrency", 8)
            results = run_batch(
                sample_ids,
                cfg=cfg,
                llm=llm,
                embedder=emb,
                index=idx,
                run_dir=run_dir,
                concurrency=conc,
                resume=not args.no_resume,
            )
        finally:
            emb.close()
            llm.close()

    # ---- metrics ----
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

    logging.info("done. results in %s", run_dir)


if __name__ == "__main__":
    main()
