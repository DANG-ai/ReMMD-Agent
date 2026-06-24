#!/usr/bin/env python
"""Idempotent corpus preparation script.

This was the very first preprocessing step we ran on
`/path/to/ReMMD-Agent/ReMMD-agent/rag_database/corpus.jsonl`: rename every
`evidence_id` from `img_ctx_<sample>_<idx>` to `ctx_<sample>_<idx>` so the
`evidence_id` no longer carries the `img_` prefix. Same rename is applied
to `sample_to_evidence.json`. Original files are backed up to `*.bak`.

Re-running this script after the rename is a no-op (still safe).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"),
    )
    ap.add_argument("--no-backup", action="store_true",
                    help="skip writing .bak side copies (default writes them)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(args.config))

    corpus_path = Path(cfg["paths"]["corpus_jsonl"])
    s2e_path = Path(cfg["paths"]["sample_to_evidence"])

    # 1) corpus.jsonl
    if not args.no_backup and not corpus_path.with_suffix(corpus_path.suffix + ".bak").exists():
        shutil.copy(corpus_path, corpus_path.with_suffix(corpus_path.suffix + ".bak"))
    fixed = 0
    total = 0
    out_lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            total += 1
            if obj.get("evidence_id", "").startswith("img_ctx_"):
                obj["evidence_id"] = obj["evidence_id"][len("img_"):]
                fixed += 1
            out_lines.append(json.dumps(obj, ensure_ascii=False))
    with open(corpus_path, "w", encoding="utf-8") as f:
        for ln in out_lines:
            f.write(ln + "\n")
    logging.info("corpus.jsonl: total=%d, fixed=%d", total, fixed)

    # 2) sample_to_evidence.json
    if not args.no_backup and not s2e_path.with_suffix(s2e_path.suffix + ".bak").exists():
        shutil.copy(s2e_path, s2e_path.with_suffix(s2e_path.suffix + ".bak"))
    with open(s2e_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    renamed = 0
    for sid, info in data.items():
        new_all = []
        for eid in info.get("all_evidence_ids", []):
            if eid.startswith("img_ctx_"):
                eid = eid[len("img_"):]
                renamed += 1
            new_all.append(eid)
        info["all_evidence_ids"] = new_all
        by_type = info.get("by_type", {})
        for k in list(by_type.keys()):
            by_type[k] = [
                (eid[len("img_"):] if eid.startswith("img_ctx_") else eid)
                for eid in by_type[k]
            ]
    with open(s2e_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info("sample_to_evidence.json: renamed entries: %d", renamed)


if __name__ == "__main__":
    main()
