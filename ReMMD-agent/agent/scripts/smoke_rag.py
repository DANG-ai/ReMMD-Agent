#!/usr/bin/env python
"""Smoke-test: verify RAG retrieval works for a few samples without LLM calls.

Uses a small set of representative atomic-point-like queries hand-crafted
from each sample's text and prints the top-K hits per query.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_sample
from src.embedder import make_embedder_from_config
from src.rag import RagIndex, retrieve_for_atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"))
    ap.add_argument("--sample-ids", nargs="+", default=["001", "002", "023"])
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    emb = make_embedder_from_config(cfg)
    try:
        idx = RagIndex.build_or_load(
            corpus_path=cfg["paths"]["corpus_jsonl"],
            sample_to_evidence_path=cfg["paths"]["sample_to_evidence"],
            cache_dir=cfg["paths"]["rag_index_dir"],
            embedder=emb,
            model_tag=cfg["embedding"]["model"],
        )
        for sid in args.sample_ids:
            print("\n" + "=" * 80)
            print(f"SAMPLE {sid}")
            print("=" * 80)
            sample = load_sample(cfg["paths"]["bench_root"], sid)
            print(f"  language: {sample.language_code}  region: {sample.region_code}  theme: {sample.theme_category}")
            print(f"  text: {sample.text[:200]}...")
            print(f"  gold verdict: {sample.gold_verdict}")
            print(f"  gold taxonomy: {sample.gold_taxonomy}")

            # rows for sample
            rows = idx.rows_for_sample(sid)
            print(f"\n  memory-bank size for this sample: {len(rows)} evidence items")
            type_counts = {}
            for r in rows:
                t = idx.items[r].evidence_type
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"  by type: {type_counts}")

            # take the first sentence and a paragraph-level summary as queries
            queries = []
            if sample.text:
                sents = [s.strip() for s in sample.text.split(".") if s.strip()][:3]
                queries.extend(sents)
                queries.append(sample.text[:500])

            print(f"\n  queries: {len(queries)} (showing first 80 chars each):")
            for i, q in enumerate(queries):
                print(f"    [{i}] {q[:100]}")

            rag_cfg = cfg["rag"]
            hits = retrieve_for_atoms(
                index=idx,
                embedder=emb,
                sample_id=sid,
                atomic_points=queries,
                top_k_per_atom=rag_cfg["top_k_per_atom"],
                min_score=rag_cfg["min_score"],
                max_evidence_per_sample=rag_cfg["max_evidence_per_sample"],
                per_type_quota=rag_cfg.get("per_type_quota") or None,
            )
            print(f"\n  top-{len(hits)} retrieved:")
            for h in hits:
                print(f"    [{h.evidence_id}] type={h.evidence_type:>13}  score={h.score:.4f}  | {h.text[:140]}")
    finally:
        emb.close()


if __name__ == "__main__":
    main()
