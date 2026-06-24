"""Stand-alone L2 pattern detector (separate LLM call before final judge).

The motivation: when asked to do everything in one shot, qwen3.5-9b is too
lenient — it predicts True for ~60% of posts (vs 20% in gold). Splitting
"identify the 8 distortion patterns" into its own focused call forces
concrete pattern-level commitments. The final judge then derives the L1
verdict from these binary decisions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .data import BenchSample
from .labels import LEVEL2_LABELS, normalize_level2_list
from .llm import LLMClient, LLMResponse, extract_json_block
from .rag import RetrievedEvidence
from .search_tools import SearchResult


logger = logging.getLogger("remmd.pattern")


def _load_prompt(prompts_dir: str) -> str:
    return Path(prompts_dir, "pattern_detect.txt").read_text(encoding="utf-8")


def _safe_render(template: str, mapping: dict[str, Any]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _format_evidence_block(evidences: list[RetrievedEvidence]) -> str:
    if not evidences:
        return "(no evidence retrieved)"
    parts = []
    for i, ev in enumerate(evidences, 1):
        parts.append(
            f"[{i}] id={ev.evidence_id} | type={ev.evidence_type} | score={ev.score:.3f}\n"
            f"{ev.text.strip()}"
        )
    return "\n\n".join(parts)


def _format_search_block(search: dict[str, list[SearchResult]] | None) -> str:
    if not search or not any(v for v in (search or {}).values()):
        return "(no external search results)"
    out = []
    for tool, results in (search or {}).items():
        if not results:
            continue
        out.append(f"== {tool} ==")
        for i, r in enumerate(results, 1):
            out.append(f"[{tool} #{i}] {r.title}\n{r.snippet}\n{r.url}")
    return "\n\n".join(out) if out else "(no external search results)"


def _slim_atoms(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "post_summary": parsed.get("post_summary", ""),
        "image_level": parsed.get("image_level", []),
        "cross_modal": parsed.get("cross_modal", []),
        "sentence_level": parsed.get("sentence_level", []),
        "paragraph_level": parsed.get("paragraph_level", []),
    }


def build_pattern_messages(
    sample: BenchSample,
    *,
    parsed_atoms: dict[str, Any],
    retrieved: list[RetrievedEvidence],
    search_hits: dict[str, list[SearchResult]] | None,
    prompts_dir: str,
) -> list[dict[str, Any]]:
    tmpl = _load_prompt(prompts_dir)
    user = _safe_render(tmpl, {
        "sample_id": sample.sample_id,
        "language_code": sample.language_code or "unknown",
        "image_names_json": json.dumps(sample.image_names, ensure_ascii=False),
        "post_text": sample.text or "",
        "atomic_points_json": json.dumps(_slim_atoms(parsed_atoms), ensure_ascii=False, indent=2),
        "evidence_block": _format_evidence_block(retrieved),
        "search_block": _format_search_block(search_hits),
    })
    return [{"role": "user", "content": user}]


def parse_pattern_output(content: str) -> dict[str, Any]:
    obj = extract_json_block(content)
    if not isinstance(obj, dict):
        raise ValueError(f"pattern detector returned non-object: {type(obj)}")
    patterns = obj.get("patterns", {}) or {}
    active = []
    detail = {}
    for lbl in LEVEL2_LABELS:
        info = patterns.get(lbl) or {}
        present = bool(info.get("present", False))
        detail[lbl] = {
            "present": present,
            "evidence_id": info.get("evidence_id"),
            "explanation": (info.get("explanation") or "").strip(),
        }
        if present:
            active.append(lbl)
    # Prefer model-emitted active_labels if they normalize sensibly
    raw_active = obj.get("active_labels") or []
    norm_active = normalize_level2_list(raw_active if isinstance(raw_active, list) else [raw_active])
    if norm_active:
        active = norm_active
    return {
        "active_labels": active,
        "patterns": detail,
    }


def derive_l1_from_patterns(active_labels: list[str]) -> str:
    """Conservative mapping from L2 cardinality+severity to L1 verdict.

    Empirical gold averages over 500 samples:
       True       -> 0.00 L2 labels
       Mostly True-> 2.09 L2
       Mixture    -> 3.50 L2
       Mostly False-> 3.64 L2
       False      -> 4.72 L2
    """
    SEVERE = {"T1 Fabrication", "V1 Synthetic Visual Content"}
    n = len(active_labels)
    n_severe = sum(1 for x in active_labels if x in SEVERE)
    if n == 0:
        return "True"
    if n == 1:
        return "Mostly True"
    if n == 2:
        return "Mixture" if n_severe >= 1 else "Mostly True"
    if n == 3:
        return "Mostly False" if n_severe >= 1 else "Mixture"
    if n == 4:
        return "Mostly False"
    if n >= 5:
        return "False" if n_severe >= 1 else "Mostly False"
    return "Mostly True"
