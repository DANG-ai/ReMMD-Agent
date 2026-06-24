"""Dedicated text-distortion analysis: focused per-claim T1/T2/T3 + C1 detection.

Mirrors the image_analyzer's role but for textual distortions:
  T1 Fabrication — invented facts/quotes/people/numbers
  T2 Distortion  — real basis but overstated / dramatized / mis-attributed
  T3 Misleading Context — real content placed in wrong time/place/source/event-frame
  C1 Semantic Inconsistency — text says one entity/event, evidence says a different one

We give the model the post text + atomic points + retrieved evidence + image
descriptions (from atomic parser's visual_indicators) and ask it to decide each
T-label yes/no with concrete evidence_id citations. C1 is included here because
detecting it well requires careful entity comparison between text and the image
descriptions, which is text-comparison-heavy.

This stage runs BEFORE the final judge and feeds in as priors alongside the
image_analyzer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .data import BenchSample
from .llm import LLMClient, LLMResponse, extract_json_block, select_prompt_filename
from .rag import RetrievedEvidence


logger = logging.getLogger("remmd.text_analyzer")


def _load_prompt(
    prompts_dir: str,
    model: str | None = None,
    prompt_name: str | None = None,
) -> str:
    if prompt_name:
        return Path(prompts_dir, prompt_name + ".txt").read_text(encoding="utf-8")
    fname = select_prompt_filename(prompts_dir, "text_analyze", model)
    return Path(prompts_dir, fname).read_text(encoding="utf-8")


def _safe_render(template: str, mapping: dict[str, Any]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _format_evidence_block(retrieved: list[RetrievedEvidence]) -> str:
    """Pull all non-image_content evidence (news, fact_brief, fact_check, social).

    image_content is the image_analyzer's input — for the text analyzer we want
    the textual ground-truth about events.
    """
    keep = [e for e in retrieved if e.evidence_type != "image_content"]
    if not keep:
        return "(no textual evidence retrieved for this sample)"
    parts = []
    for i, ev in enumerate(keep, 1):
        parts.append(
            f"[{i}] id={ev.evidence_id} | type={ev.evidence_type} | score={ev.score:.3f}\n{ev.text.strip()}"
        )
    return "\n\n".join(parts)


def _format_image_descriptions(parsed_atoms: dict[str, Any]) -> str:
    """Compact visual descriptions from atomic parser's visual_indicators.

    Used so the text analyzer can detect C1 (entity mismatch) without seeing
    the actual images itself.
    """
    vi = parsed_atoms.get("visual_indicators") or []
    if not vi:
        return "(no visual descriptions — sample has no images)"
    parts = []
    for v in vi:
        if not isinstance(v, dict):
            continue
        idx = v.get("image_index", "?")
        name = v.get("image_name", "?")
        sd = (v.get("scene_description") or "").strip().replace("\n", " ")
        ocr = (v.get("ocr_text") or "").strip().replace("\n", " ")
        ents = v.get("key_entities") or []
        overlays = v.get("visible_overlays_annotations") or []
        align = (v.get("image_text_alignment") or "").strip().replace("\n", " ")
        parts.append(
            f"  • image {idx} ({name}): scene='{sd[:200]}' | ocr='{ocr[:120]}' | "
            f"entities={ents} | overlays={overlays} | alignment='{align[:140]}'"
        )
    return "\n".join(parts) if parts else "(no per-image descriptions available)"


def build_text_analyzer_messages(
    sample: BenchSample,
    *,
    parsed_atoms: dict[str, Any],
    retrieved: list[RetrievedEvidence],
    prompts_dir: str,
    model: str | None = None,
    prompt_name: str | None = None,
) -> list[dict[str, Any]]:
    tmpl = _load_prompt(prompts_dir, model=model, prompt_name=prompt_name)
    user = _safe_render(tmpl, {
        "sample_id": sample.sample_id,
        "language_code": sample.language_code or "unknown",
        "region_code": sample.region_code or "unknown",
        "theme_category": sample.theme_category or "unknown",
        "post_text": sample.text or "",
        "post_summary": (parsed_atoms.get("post_summary") or "").strip(),
        "atomic_points_json": json.dumps({
            "post_summary": parsed_atoms.get("post_summary", ""),
            "sentence_level": parsed_atoms.get("sentence_level", []),
            "paragraph_level": parsed_atoms.get("paragraph_level", []),
            "cross_modal": parsed_atoms.get("cross_modal", []),
        }, ensure_ascii=False, indent=2),
        "evidence_block": _format_evidence_block(retrieved),
        "image_descriptions_block": _format_image_descriptions(parsed_atoms),
    })
    return [{"role": "user", "content": user}]


def _heuristic_extract_text_analyzer(content: str) -> dict[str, Any] | None:
    """Best-effort recovery when the model truncates inside the JSON.

    Returns a partial dict with whatever boolean labels we could recover
    by string-matching the JSON keys, or None when nothing is recoverable.
    """
    import re as _re
    out: dict[str, Any] = {}
    m = _re.search(r'"alignment_level"\s*:\s*"(WELL_ALIGNED|PARTIALLY_ALIGNED|MISALIGNED)"', content)
    if m:
        out["alignment_level"] = m.group(1)
    for key in ("t1_fabrication_present", "t2_distortion_present",
                "t3_misleading_context_present", "c1_semantic_inconsistency_present"):
        m = _re.search(rf'"{key}"\s*:\s*(true|false)', content)
        if m:
            out[key] = (m.group(1) == "true")
    if not out:
        return None
    return out


def parse_text_analyzer_output(content: str) -> dict[str, Any]:
    try:
        obj = extract_json_block(content)
        # Tolerate models that wrap the object in an array `[{...}]`.
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        if not isinstance(obj, dict):
            raise ValueError(f"text_analyzer returned non-object: {type(obj)}")
    except Exception as exc:
        # Heuristic fallback: extract boolean labels from a truncated response.
        recovered = _heuristic_extract_text_analyzer(content)
        if recovered is None:
            raise
        logger.warning("text_analyzer JSON parse failed (%s); recovered %d labels heuristically", exc, len(recovered))
        obj = recovered

    label_votes = {
        "T1 Fabrication": bool(obj.get("t1_fabrication_present", False)),
        "T2 Distortion": bool(obj.get("t2_distortion_present", False)),
        "T3 Misleading Context": bool(obj.get("t3_misleading_context_present", False)),
        "C1 Semantic Inconsistency": bool(obj.get("c1_semantic_inconsistency_present", False)),
    }
    active = [k for k, v in label_votes.items() if v]
    return {
        "alignment_level": (obj.get("alignment_level") or "").strip(),
        "alignment_reasoning": (obj.get("alignment_reasoning") or "").strip(),
        "active_labels": active,
        "label_votes": label_votes,
        "t1_evidence": (obj.get("t1_evidence") or "").strip(),
        "t1_problematic_subclaims": obj.get("t1_problematic_subclaims") or [],
        "t2_evidence": (obj.get("t2_evidence") or "").strip(),
        "t2_problematic_subclaims": obj.get("t2_problematic_subclaims") or [],
        "t3_evidence": (obj.get("t3_evidence") or "").strip(),
        "t3_problematic_subclaims": obj.get("t3_problematic_subclaims") or [],
        "c1_evidence": (obj.get("c1_evidence") or "").strip(),
        "c1_problematic_subclaims": obj.get("c1_problematic_subclaims") or [],
        "global_assessment": (obj.get("global_assessment") or "").strip(),
    }


def run_text_analyzer(
    sample: BenchSample,
    llm: LLMClient,
    *,
    parsed_atoms: dict[str, Any],
    retrieved: list[RetrievedEvidence],
    prompts_dir: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    model: str | None = None,
) -> tuple[dict[str, Any], LLMResponse]:
    messages = build_text_analyzer_messages(
        sample,
        parsed_atoms=parsed_atoms,
        retrieved=retrieved,
        prompts_dir=prompts_dir,
        model=model,
    )
    extra = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
    resp = llm.chat(messages, max_tokens=max_tokens, temperature=temperature, extra_body=extra)
    parsed = parse_text_analyzer_output(resp.content)
    return parsed, resp
