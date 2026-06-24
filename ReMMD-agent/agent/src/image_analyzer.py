"""Dedicated vision pass: per-image V1/V2/C-label analysis.

Run AFTER atomic parsing + RAG retrieval, BEFORE the final judge. This call's
sole job is to look at each post image carefully (with the ground-truth
image_content evidence as a reference) and decide which visual / contextual
distortion patterns are present.

Why a separate stage?
  • The single-shot atomic parser is multi-tasking (text decomposition + visual
    indicators) and tends to produce conservative "edit_synth_likelihood=none"
    even when the gold annotation says V2.
  • Giving the model a focused, single-task prompt with clear yes/no questions
    plus the comparison image_content text yields markedly more critical
    judgements about visual distortions.

The stage outputs structured per-image JSON which the final judge consumes as a
strong prior. The final judge can still override based on text evidence.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .data import BenchSample, prepare_image_content_blocks, select_image_files_for_call
from .llm import LLMClient, LLMResponse, extract_json_block, select_prompt_filename
from .rag import RetrievedEvidence


logger = logging.getLogger("remmd.image_analyzer")


def _load_prompt(
    prompts_dir: str,
    model: str | None = None,
    prompt_name: str | None = None,
) -> str:
    if prompt_name:
        return Path(prompts_dir, prompt_name + ".txt").read_text(encoding="utf-8")
    fname = select_prompt_filename(prompts_dir, "image_analyze", model)
    return Path(prompts_dir, fname).read_text(encoding="utf-8")


def _safe_render(template: str, mapping: dict[str, Any]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _format_image_content_block(retrieved: list[RetrievedEvidence]) -> str:
    """Pull out only `image_content` evidence — that's what describes what the
    GROUND-TRUTH images of this topic look like, and the strongest signal for
    detecting V2 (image-from-different-event) and V1 (synthesised)."""
    img_evid = [e for e in retrieved if e.evidence_type == "image_content"]
    if not img_evid:
        return "(no image_content evidence retrieved for this sample)"
    parts = []
    for i, ev in enumerate(img_evid, 1):
        parts.append(
            f"[image_content {i}] id={ev.evidence_id} | score={ev.score:.3f}\n{ev.text.strip()}"
        )
    return "\n\n".join(parts)


def _format_post_summary(parsed_atoms: dict[str, Any]) -> str:
    summary = (parsed_atoms.get("post_summary") or "").strip()
    cross_modal = parsed_atoms.get("cross_modal") or []
    parts = []
    if summary:
        parts.append("Post summary: " + summary)
    if cross_modal:
        parts.append("How the post uses each image (per cross_modal atoms):")
        for c in cross_modal:
            parts.append(f"  • {c}")
    return "\n".join(parts) if parts else "(no summary available)"


def build_image_analyzer_messages(
    sample: BenchSample,
    *,
    parsed_atoms: dict[str, Any],
    retrieved: list[RetrievedEvidence],
    prompts_dir: str,
    image_max_side: int = 896,
    max_images: int | None = 6,
    model: str | None = None,
    prompt_name: str | None = None,
) -> list[dict[str, Any]] | None:
    """Returns None when there are no images to analyse."""
    chosen = select_image_files_for_call(sample, max_images=max_images)
    if not chosen:
        return None

    tmpl = _load_prompt(prompts_dir, model=model, prompt_name=prompt_name)
    text_part = _safe_render(tmpl, {
        "sample_id": sample.sample_id,
        "language_code": sample.language_code or "unknown",
        "region_code": sample.region_code or "unknown",
        "theme_category": sample.theme_category or "unknown",
        "image_names_json": json.dumps(sample.image_names, ensure_ascii=False),
        "n_images": str(len(chosen)),
        "post_text": sample.text or "",
        "post_summary_block": _format_post_summary(parsed_atoms),
        "image_content_evidence_block": _format_image_content_block(retrieved),
    })

    image_blocks = prepare_image_content_blocks(
        chosen, max_side=image_max_side, max_images=max_images,
    )
    roster_lines = []
    for i, p in enumerate(chosen, 1):
        roster_lines.append(f"  - image {i} → filename `{Path(p).name}`")
    tail = (
        "\nIMAGE ROSTER (in the order shown above):\n"
        + "\n".join(roster_lines)
        + "\n\nReturn the JSON object only — no markdown fences, no commentary."
    )
    content = [{"type": "text", "text": text_part}] + image_blocks + [{"type": "text", "text": tail}]
    return [{"role": "user", "content": content}]


def parse_image_analyzer_output(content: str) -> dict[str, Any]:
    obj = extract_json_block(content)
    # Some models wrap the entire output as a list of one object.
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError(f"image_analyzer returned non-object: {type(obj)}")
    images = obj.get("images") or []
    if not isinstance(images, list):
        raise ValueError("image_analyzer.images must be a list")

    # Aggregate L2 votes across images
    label_votes: dict[str, int] = {
        "V1 Synthetic Visual Content": 0,
        "V2 Visual Editing": 0,
        "C1 Semantic Inconsistency": 0,
        "C2 Contextual Inconsistency": 0,
        "C3 Pragmatic Inconsistency": 0,
    }
    image_findings = []
    for img in images:
        if not isinstance(img, dict):
            continue
        rec = {
            "image_index": img.get("image_index"),
            "image_name": img.get("image_name"),
            "scene_description": (img.get("scene_description") or "").strip(),
            "ocr_text": (img.get("ocr_text") or "").strip(),
            "matches_post_claim_about_image": img.get("matches_post_claim_about_image"),
            "matches_evidence_image_content": img.get("matches_evidence_image_content"),
            "v1_synthetic_present": bool(img.get("v1_synthetic_present", False)),
            "v2_edit_present": bool(img.get("v2_edit_present", False)),
            "c1_entity_mismatch": bool(img.get("c1_entity_mismatch", False)),
            "c2_context_mismatch": bool(img.get("c2_context_mismatch", False)),
            "c3_pragmatic_mismatch": bool(img.get("c3_pragmatic_mismatch", False)),
            "v1_evidence": (img.get("v1_evidence") or "").strip(),
            "v2_evidence": (img.get("v2_evidence") or "").strip(),
            "c_evidence": (img.get("c_evidence") or "").strip(),
            "explanation": (img.get("explanation") or "").strip(),
        }
        if rec["v1_synthetic_present"]:
            label_votes["V1 Synthetic Visual Content"] += 1
        if rec["v2_edit_present"]:
            label_votes["V2 Visual Editing"] += 1
        if rec["c1_entity_mismatch"]:
            label_votes["C1 Semantic Inconsistency"] += 1
        if rec["c2_context_mismatch"]:
            label_votes["C2 Contextual Inconsistency"] += 1
        if rec["c3_pragmatic_mismatch"]:
            label_votes["C3 Pragmatic Inconsistency"] += 1
        image_findings.append(rec)

    active_labels = [k for k, v in label_votes.items() if v > 0]
    return {
        "active_labels": active_labels,
        "label_votes": label_votes,
        "images": image_findings,
        "global_notes": (obj.get("global_notes") or "").strip(),
    }


def run_image_analyzer(
    sample: BenchSample,
    llm: LLMClient,
    *,
    parsed_atoms: dict[str, Any],
    retrieved: list[RetrievedEvidence],
    prompts_dir: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    image_max_side: int = 896,
    max_images: int | None = 6,
    model: str | None = None,
) -> tuple[dict[str, Any] | None, LLMResponse | None]:
    messages = build_image_analyzer_messages(
        sample,
        parsed_atoms=parsed_atoms,
        retrieved=retrieved,
        prompts_dir=prompts_dir,
        image_max_side=image_max_side,
        max_images=max_images,
        model=model,
    )
    if messages is None:
        return None, None
    extra = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
    resp = llm.chat(messages, max_tokens=max_tokens, temperature=temperature, extra_body=extra)
    parsed = parse_image_analyzer_output(resp.content)
    return parsed, resp
