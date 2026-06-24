#!/usr/bin/env python
"""Build (or reuse) the corpus-wide embedding cache for the memory bank."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.embedder import make_embedder_from_config
from src.rag import RagIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"))
    ap.add_argument("--force", action="store_true", help="ignore cache and rebuild")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.force:
        # invalidate cache by removing it
        cache_dir = Path(cfg["paths"]["rag_index_dir"])
        for p in cache_dir.glob("corpus_*"):
            p.unlink()

    emb = make_embedder_from_config(cfg)
    try:
        idx = RagIndex.build_or_load(
            corpus_path=cfg["paths"]["corpus_jsonl"],
            sample_to_evidence_path=cfg["paths"]["sample_to_evidence"],
            cache_dir=cfg["paths"]["rag_index_dir"],
            embedder=emb,
            model_tag=cfg["embedding"]["model"],
        )
        print(f"index ready: n_items={len(idx.items)} dim={idx.dim} n_samples_with_rows={sum(1 for v in idx._sample_rows.values() if v)}")
    finally:
        emb.close()


if __name__ == "__main__":
    main()
