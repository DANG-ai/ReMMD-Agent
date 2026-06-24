"""Per-sample RAG memory bank built on top of qwen3-embedding-8b.

We index the **whole corpus once** (cached on disk under rag_index_dir as a
single `(N, 4096)` numpy file plus a metadata file), then at query-time we
restrict the candidate set to the evidence_ids that belong to the current
sample (via `sample_to_evidence.json`). This avoids re-embedding for every
sample and keeps the system fast while preserving the per-sample memory-bank
semantics described in the workflow.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .embedder import EmbeddingClient

logger = logging.getLogger("remmd.rag")


@dataclass
class EvidenceItem:
    evidence_id: str
    sample_id: str
    evidence_type: str
    text: str


@dataclass
class RetrievedEvidence:
    evidence_id: str
    evidence_type: str
    text: str
    score: float
    matched_atomic_idx: int | None = None  # which atomic point hit it


class RagIndex:
    """In-memory store: L2-normalized embedding matrix + parallel metadata."""

    def __init__(self) -> None:
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.items: list[EvidenceItem] = []
        # evidence_id -> row index
        self._id_to_row: dict[str, int] = {}
        # sample_id -> list of row indices belonging to this sample
        self._sample_rows: dict[str, list[int]] = {}

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1] if self.embeddings.size else 0

    def rows_for_sample(self, sample_id: str) -> list[int]:
        return self._sample_rows.get(sample_id, [])

    def query(
        self,
        query_vec: np.ndarray,
        *,
        sample_id: str,
        top_k: int,
        min_score: float = 0.0,
    ) -> list[RetrievedEvidence]:
        """Cosine similarity (inner product on normalized vectors)."""
        rows = self.rows_for_sample(sample_id)
        if not rows:
            return []
        sub = self.embeddings[rows]  # (M, dim)
        # query_vec already L2-normalized; sub already L2-normalized
        sims = sub @ query_vec  # (M,)
        # take top-k indices
        k = min(top_k, sims.shape[0])
        if k <= 0:
            return []
        top_local = np.argpartition(-sims, k - 1)[:k]
        top_local = top_local[np.argsort(-sims[top_local])]
        results: list[RetrievedEvidence] = []
        for local_idx in top_local:
            score = float(sims[local_idx])
            if score < min_score:
                continue
            item = self.items[rows[local_idx]]
            results.append(
                RetrievedEvidence(
                    evidence_id=item.evidence_id,
                    evidence_type=item.evidence_type,
                    text=item.text,
                    score=score,
                )
            )
        return results

    # ------------- build / load -------------
    @classmethod
    def build_or_load(
        cls,
        *,
        corpus_path: str,
        sample_to_evidence_path: str,
        cache_dir: str,
        embedder: EmbeddingClient,
        model_tag: str,
    ) -> "RagIndex":
        cache_dir_p = Path(cache_dir)
        cache_dir_p.mkdir(parents=True, exist_ok=True)
        emb_path = cache_dir_p / f"corpus_embeddings.{model_tag}.npy"
        meta_path = cache_dir_p / f"corpus_meta.{model_tag}.json"

        idx = cls()
        # load all items (always; fast — 7.7k lines)
        items: list[EvidenceItem] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                items.append(
                    EvidenceItem(
                        evidence_id=obj["evidence_id"],
                        sample_id=str(obj["sample_id"]),
                        evidence_type=obj.get("evidence_type", "unknown"),
                        text=obj.get("text", ""),
                    )
                )
        idx.items = items
        idx._id_to_row = {it.evidence_id: i for i, it in enumerate(items)}

        # build per-sample rows lookup from sample_to_evidence.json (authoritative)
        with open(sample_to_evidence_path, "r", encoding="utf-8") as f:
            s2e = json.load(f)
        for sid, info in s2e.items():
            sid = str(sid)
            row_ids: list[int] = []
            for eid in info.get("all_evidence_ids", []):
                row = idx._id_to_row.get(eid)
                if row is None:
                    logger.warning("evidence_id %s referenced by sample %s not found in corpus", eid, sid)
                    continue
                row_ids.append(row)
            idx._sample_rows[sid] = row_ids
        # also include any evidence whose sample_id is set but absent from s2e
        for i, it in enumerate(items):
            idx._sample_rows.setdefault(it.sample_id, [])
            if i not in idx._sample_rows[it.sample_id]:
                # only append if not already represented (rare safety net)
                if it.evidence_id not in {idx.items[r].evidence_id for r in idx._sample_rows[it.sample_id]}:
                    idx._sample_rows[it.sample_id].append(i)

        # load or build embeddings
        if emb_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("n") == len(items) and meta.get("model_tag") == model_tag:
                    arr = np.load(emb_path)
                    if arr.shape[0] == len(items):
                        idx.embeddings = arr.astype(np.float32, copy=False)
                        logger.info(
                            "loaded cached embeddings: shape=%s from %s",
                            arr.shape, emb_path,
                        )
                        return idx
                logger.info("cached embeddings stale; rebuilding")
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to load cache (%s); rebuilding", exc)

        logger.info("embedding %d evidence items via %s...", len(items), model_tag)
        # embed in chunks
        all_embs: list[np.ndarray] = []
        chunk = 64  # number of items handed to embedder per call group (embedder itself sub-batches)
        for i in range(0, len(items), chunk):
            sub = [it.text for it in items[i : i + chunk]]
            arr = embedder.embed(sub)
            all_embs.append(arr)
            if (i // chunk) % 5 == 0:
                logger.info("  embedded %d / %d", i + len(sub), len(items))
        embeddings = np.concatenate(all_embs, axis=0).astype(np.float32, copy=False)
        assert embeddings.shape[0] == len(items)
        idx.embeddings = embeddings

        np.save(emb_path, embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"n": len(items), "dim": int(embeddings.shape[1]), "model_tag": model_tag}, f)
        logger.info("cached %d embeddings (dim=%d) to %s", len(items), embeddings.shape[1], emb_path)
        return idx


def retrieve_for_atoms(
    *,
    index: RagIndex,
    embedder: EmbeddingClient,
    sample_id: str,
    atomic_points: list[str],
    top_k_per_atom: int,
    min_score: float,
    max_evidence_per_sample: int,
    per_type_quota: dict[str, int] | None = None,
) -> list[RetrievedEvidence]:
    """For each atomic point, retrieve top-K; then dedup + optionally rebalance by type."""
    if not atomic_points:
        return []
    qvecs = embedder.embed(atomic_points)
    seen: dict[str, RetrievedEvidence] = {}
    for i, qv in enumerate(qvecs):
        hits = index.query(qv, sample_id=sample_id, top_k=top_k_per_atom, min_score=min_score)
        for h in hits:
            # keep highest-score occurrence, but track first matched atom for traceability
            cur = seen.get(h.evidence_id)
            if cur is None or h.score > cur.score:
                if cur is None:
                    h.matched_atomic_idx = i
                else:
                    h.matched_atomic_idx = cur.matched_atomic_idx
                seen[h.evidence_id] = h
    pooled = sorted(seen.values(), key=lambda r: -r.score)

    if per_type_quota:
        kept: list[RetrievedEvidence] = []
        type_counts: dict[str, int] = {}
        deferred: list[RetrievedEvidence] = []
        for r in pooled:
            q = per_type_quota.get(r.evidence_type)
            if q is None:
                # type not under quota; keep until cap
                if len(kept) < max_evidence_per_sample:
                    kept.append(r)
                continue
            if type_counts.get(r.evidence_type, 0) < q and len(kept) < max_evidence_per_sample:
                kept.append(r)
                type_counts[r.evidence_type] = type_counts.get(r.evidence_type, 0) + 1
            else:
                deferred.append(r)
        # fill remaining slots from deferred (highest score first)
        if len(kept) < max_evidence_per_sample:
            for r in deferred:
                if len(kept) >= max_evidence_per_sample:
                    break
                kept.append(r)
        return kept
    return pooled[:max_evidence_per_sample]
