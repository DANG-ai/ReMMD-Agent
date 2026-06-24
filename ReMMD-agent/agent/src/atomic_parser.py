"""Atomic-point parsing on a vision-language model.

Three modes (all preserved — pipeline switches via cfg.pipeline.atomic_mode):

  • "single":  ONE VLM call decomposes the post (text + images) into all four
               atomic levels plus a structured per-image visual analysis.
               (Original behaviour, kept for compatibility.)
  • "multi":   N concurrent passes of the same single-shot prompt with slight
               temperature variation, then UNION the produced atoms.
               (Original behaviour, kept for compatibility.)
  • "split":   THREE separate qwen calls run concurrently:
                   – image-only atomic parse  (prompt: image_atom_parse.txt)
                   – cross-modal atomic parse (prompt: cross_modal_atom_parse.txt)
                   – text-only atomic parse   (prompt: text_atom_parse.txt)
               The atoms / visual_indicators / cross_modal_findings are then
               MERGED into the same schema used by single/multi so the rest of
               the pipeline (RAG, analyzers, final judge) is unchanged.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
from pathlib import Path
from typing import Any

from .data import BenchSample, prepare_image_content_blocks, select_image_files_for_call
from .llm import LLMClient, LLMResponse, extract_json_block


logger = logging.getLogger("remmd.atomic")


def _load_prompt(prompts_dir: str) -> str:
    return Path(prompts_dir, "atomic_parse.txt").read_text(encoding="utf-8")


def _load_prompt_named(prompts_dir: str, name: str) -> str:
    """Load any prompt file by base-name (extension auto-added)."""
    return Path(prompts_dir, name + ".txt").read_text(encoding="utf-8")


def _safe_render(template: str, mapping: dict[str, Any]) -> str:
    """Plain `{key}` placeholder substitution that does NOT choke on stray `{` / `}`.

    We deliberately avoid `str.format` because the prompt body contains JSON
    examples with their own braces. Instead, replace each `{key}` explicitly.
    """
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def build_atomic_messages(
    sample: BenchSample,
    prompt_template: str,
    *,
    image_max_side: int = 768,
    max_images: int | None = 6,
) -> list[dict[str, Any]]:
    """Assemble the multimodal user message for the atomic parser.

    Layout:
      - [0]  text block: full instructions + post metadata + post text
      - [1..N] image_url blocks (one per attached image, resized + JPEG)
      - [N+1] tail text block: per-image filename roster (so the model can
              correlate the visual order with the post's `image_names`)
    """
    text_part = _safe_render(prompt_template, {
        "sample_id": sample.sample_id,
        "language_code": sample.language_code or "unknown",
        "region_code": sample.region_code or "unknown",
        "theme_category": sample.theme_category or "unknown",
        "image_names_json": json.dumps(sample.image_names, ensure_ascii=False),
        "n_images": str(len(sample.image_names)),
        "text": sample.text or "",
    })

    chosen = select_image_files_for_call(sample, max_images=max_images)
    image_blocks = prepare_image_content_blocks(
        chosen,
        max_side=image_max_side,
        max_images=max_images,
    )

    if image_blocks:
        # Tail text gives the model an explicit filename↔order mapping plus a
        # final reminder to reply in JSON only.
        roster_lines = []
        for i, p in enumerate(chosen, 1):
            roster_lines.append(f"  - image {i} → filename `{Path(p).name}`")
        tail = (
            "\nIMAGE ROSTER (in the order the images appear above):\n"
            + "\n".join(roster_lines)
            + "\n\nReturn the JSON object only — no markdown fences, no commentary."
        )
        content = [{"type": "text", "text": text_part}] + image_blocks + [{"type": "text", "text": tail}]
    else:
        content = text_part + "\n\nReturn the JSON object only — no markdown fences, no commentary."

    return [{"role": "user", "content": content}]


def parse_atomic_points(
    sample: BenchSample,
    llm: LLMClient,
    *,
    prompts_dir: str,
    max_tokens: int | None = None,
    image_max_side: int = 768,
    max_images: int | None = 6,
) -> tuple[dict[str, Any], LLMResponse]:
    """Returns (parsed_dict, raw_response).

    Raises ValueError if the LLM output cannot be coerced into the expected schema.
    """
    prompt = _load_prompt(prompts_dir)
    messages = build_atomic_messages(
        sample, prompt,
        image_max_side=image_max_side,
        max_images=max_images,
    )
    resp = llm.chat(messages, max_tokens=max_tokens)
    obj = extract_json_block(resp.content)
    if not isinstance(obj, dict):
        raise ValueError(f"atomic parser returned non-object: {type(obj)}")
    for k in ("post_summary", "image_level", "cross_modal", "sentence_level",
              "paragraph_level", "retrieval_queries", "visual_indicators"):
        obj.setdefault(k, [] if k != "post_summary" else "")
    return obj, resp


def _merge_atom_lists(*lists: list[Any]) -> list[Any]:
    """Union a set of atom-string lists; preserves order, dedups by lowercase."""
    out: list[Any] = []
    seen: set[str] = set()
    for lst in lists:
        for item in lst or []:
            if isinstance(item, str):
                key = item.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(item.strip())
            elif isinstance(item, dict):
                # For sentence_level / image_level / visual_indicators which
                # are dicts: dedup by sentence_index/image_index where present,
                # else fall back to the JSON serialised form.
                key = None
                if "sentence_index" in item:
                    key = f"sent::{item.get('sentence_index')}::{(item.get('sentence_text') or '')[:80].lower()}"
                elif "image_index" in item:
                    key = f"img::{item.get('image_index')}::{(item.get('image_name') or '').lower()}"
                else:
                    key = json.dumps(item, ensure_ascii=False, sort_keys=True)[:200]
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
    return out


def _merge_per_image_dicts(blocks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """For image_level / visual_indicators: merge atoms per image_index."""
    by_idx: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for lst in blocks or []:
        for b in lst or []:
            if not isinstance(b, dict):
                continue
            idx = b.get("image_index")
            if idx is None:
                continue
            if idx not in by_idx:
                by_idx[idx] = dict(b)
                order.append(idx)
                if "atoms" in b:
                    by_idx[idx]["atoms"] = list(b.get("atoms") or [])
            else:
                cur = by_idx[idx]
                # union atoms / lists
                if "atoms" in b:
                    cur["atoms"] = _merge_atom_lists(cur.get("atoms") or [], b.get("atoms") or [])
                # union list-typed fields
                for k in ("key_entities", "visible_overlays_annotations"):
                    if k in b and isinstance(b.get(k), list):
                        cur[k] = _merge_atom_lists(cur.get(k) or [], b.get(k) or [])
                # take longest text fields
                for k in ("scene_description", "ocr_text", "image_text_alignment"):
                    cur_v = cur.get(k) or ""
                    new_v = b.get(k) or ""
                    if len(new_v) > len(cur_v):
                        cur[k] = new_v
    return [by_idx[i] for i in order]


def _merge_sentence_blocks(blocks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """For sentence_level: merge atoms per sentence_index, keep all sentences."""
    by_idx: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for lst in blocks or []:
        for b in lst or []:
            if not isinstance(b, dict):
                continue
            idx = b.get("sentence_index")
            if idx is None:
                continue
            if idx not in by_idx:
                by_idx[idx] = dict(b)
                order.append(idx)
                if "atoms" in b:
                    by_idx[idx]["atoms"] = list(b.get("atoms") or [])
            else:
                cur = by_idx[idx]
                if "atoms" in b:
                    cur["atoms"] = _merge_atom_lists(cur.get("atoms") or [], b.get("atoms") or [])
    return [by_idx[i] for i in sorted(order)]


def merge_atomic_outputs(parsed_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Union atom strings from multiple atomic-parser passes.

    Atoms are deduped by lowercase. Per-sentence and per-image structures
    have their inner atom lists unioned.
    """
    merged: dict[str, Any] = {}
    # post_summary: keep the longest/most-detailed version
    summaries = [p.get("post_summary") for p in parsed_list if p.get("post_summary")]
    merged["post_summary"] = max(summaries, key=len) if summaries else ""

    # paragraph_level / cross_modal / retrieval_queries are flat string lists
    for k in ("paragraph_level", "cross_modal", "retrieval_queries"):
        merged[k] = _merge_atom_lists(*(p.get(k) or [] for p in parsed_list))

    # sentence_level: list of dict with sentence_index, atoms
    merged["sentence_level"] = _merge_sentence_blocks(
        [p.get("sentence_level") or [] for p in parsed_list]
    )
    # image_level: list of dict with image_index, atoms
    merged["image_level"] = _merge_per_image_dicts(
        [p.get("image_level") or [] for p in parsed_list]
    )
    # visual_indicators: list of dict with image_index + visual fields
    merged["visual_indicators"] = _merge_per_image_dicts(
        [p.get("visual_indicators") or [] for p in parsed_list]
    )
    return merged


def parse_atomic_points_multi(
    sample: BenchSample,
    llm: LLMClient,
    *,
    prompts_dir: str,
    n_passes: int = 3,
    temperatures: list[float] | None = None,
    max_tokens: int | None = None,
    image_max_side: int = 768,
    max_images: int | None = 6,
    enable_thinking: bool = False,
) -> tuple[dict[str, Any], list[LLMResponse]]:
    """Run `n_passes` atomic parses concurrently and merge the atoms.

    `temperatures` defaults to [0.0, 0.4, 0.8] for n_passes==3. The first
    pass is deterministic; subsequent passes use higher temperatures to
    encourage diverse atom exploration. We only use the model's chat call
    (no streaming).
    """
    if temperatures is None:
        # spaced over [0, 0.8]
        if n_passes <= 1:
            temperatures = [0.0]
        else:
            step = 0.8 / (n_passes - 1)
            temperatures = [round(step * i, 2) for i in range(n_passes)]
    assert len(temperatures) == n_passes, "temperatures must match n_passes"

    prompt = _load_prompt(prompts_dir)
    messages = build_atomic_messages(
        sample, prompt,
        image_max_side=image_max_side,
        max_images=max_images,
    )

    extra = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}

    def _one(temp: float) -> tuple[dict[str, Any] | None, LLMResponse | None, Exception | None]:
        try:
            r = llm.chat(messages, max_tokens=max_tokens, temperature=temp, extra_body=extra)
            obj = extract_json_block(r.content)
            if not isinstance(obj, dict):
                return None, r, ValueError(f"atomic returned non-object: {type(obj)}")
            for k in ("post_summary", "image_level", "cross_modal",
                      "sentence_level", "paragraph_level", "retrieval_queries",
                      "visual_indicators"):
                obj.setdefault(k, [] if k != "post_summary" else "")
            return obj, r, None
        except Exception as exc:  # noqa: BLE001
            return None, None, exc

    parsed_list: list[dict[str, Any]] = []
    raw_responses: list[LLMResponse] = []
    errors: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=n_passes) as pool:
        futures = [pool.submit(_one, t) for t in temperatures]
        for fut in cf.as_completed(futures):
            obj, resp, err = fut.result()
            if err is not None:
                errors.append(repr(err))
                if resp is not None:
                    raw_responses.append(resp)
                continue
            parsed_list.append(obj)
            if resp is not None:
                raw_responses.append(resp)

    if not parsed_list:
        # all passes failed — re-raise the first error
        raise RuntimeError(
            f"all {n_passes} atomic parser passes failed; first error: "
            + (errors[0] if errors else "unknown")
        )

    merged = merge_atomic_outputs(parsed_list)
    return merged, raw_responses


def flatten_atoms_for_retrieval(parsed: dict[str, Any]) -> list[str]:
    """Flatten the four-level atomic structure + visual descriptions into a deduped
    list of query strings.

    Order: retrieval_queries (broad probes), paragraph_level, cross_modal,
    sentence_level atoms, image_level atoms, visual_indicators descriptions,
    and (when present from split-mode) cross_modal_findings + quoted_text.
    Each atom is stripped; empty atoms removed; case-insensitive dedup.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(items):
        for it in items or []:
            if not it:
                continue
            s = str(it).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)

    _add(parsed.get("retrieval_queries"))
    _add(parsed.get("paragraph_level"))
    _add(parsed.get("cross_modal"))
    for block in parsed.get("sentence_level") or []:
        if isinstance(block, dict):
            _add(block.get("atoms"))
    for block in parsed.get("image_level") or []:
        if isinstance(block, dict):
            _add(block.get("atoms"))

    # Use vision descriptions as additional retrieval probes — these surface
    # entities and scene types that the post text might not mention but the
    # ground-truth image_content evidence will.
    for v in parsed.get("visual_indicators") or []:
        if not isinstance(v, dict):
            continue
        desc = (v.get("scene_description") or "").strip()
        if desc:
            _add([desc])
        ocr = (v.get("ocr_text") or "").strip()
        if ocr:
            _add([ocr])
        for obj_name in (v.get("key_entities") or []):
            _add([obj_name])

    # Split-mode extras: cross_modal_findings carries per-image text-binding
    # data that often surfaces the named entities / events that retrieval needs.
    for cmf in parsed.get("cross_modal_findings") or []:
        if not isinstance(cmf, dict):
            continue
        for k in ("post_claim_about_image", "what_image_shows",
                  "explicit_caption_or_overlay"):
            v = (cmf.get(k) or "").strip()
            if v:
                _add([v])

    # Split-mode extras: each verbatim quote is a strong fact_check probe.
    for q in parsed.get("quoted_text_and_attributions") or []:
        if not isinstance(q, dict):
            continue
        v = (q.get("quoted_text") or "").strip()
        if v:
            _add([v])

    return out


# =============================================================================
# SPLIT MODE — three independent qwen calls run concurrently, results merged.
# =============================================================================


def build_image_atom_messages(
    sample: BenchSample,
    prompt_template: str,
    *,
    image_max_side: int = 896,
    max_images: int | None = 6,
) -> list[dict[str, Any]]:
    """Image-only atomic-parse user message (images + minimal text context)."""
    text_part = _safe_render(prompt_template, {
        "sample_id": sample.sample_id,
        "language_code": sample.language_code or "unknown",
        "region_code": sample.region_code or "unknown",
        "theme_category": sample.theme_category or "unknown",
        "image_names_json": json.dumps(sample.image_names, ensure_ascii=False),
        "n_images": str(len(sample.image_names)),
        "text": sample.text or "",
    })

    chosen = select_image_files_for_call(sample, max_images=max_images)
    image_blocks = prepare_image_content_blocks(
        chosen, max_side=image_max_side, max_images=max_images,
    )
    if not image_blocks:
        # No images — return a text-only stub message; the caller will see an
        # empty image_level / visual_indicators and skip merging.
        return [{"role": "user",
                 "content": text_part + "\n\nReturn the JSON object only — no markdown fences, no commentary."}]
    roster_lines = []
    for i, p in enumerate(chosen, 1):
        roster_lines.append(f"  - image {i} → filename `{Path(p).name}`")
    tail = (
        "\nIMAGE ROSTER (in the order the images appear above):\n"
        + "\n".join(roster_lines)
        + "\n\nReturn the JSON object only — no markdown fences, no commentary."
    )
    content = [{"type": "text", "text": text_part}] + image_blocks + [{"type": "text", "text": tail}]
    return [{"role": "user", "content": content}]


def build_cross_modal_atom_messages(
    sample: BenchSample,
    prompt_template: str,
    *,
    image_max_side: int = 768,
    max_images: int | None = 6,
) -> list[dict[str, Any]]:
    """Cross-modal atomic-parse user message (images + full text)."""
    text_part = _safe_render(prompt_template, {
        "sample_id": sample.sample_id,
        "language_code": sample.language_code or "unknown",
        "region_code": sample.region_code or "unknown",
        "theme_category": sample.theme_category or "unknown",
        "image_names_json": json.dumps(sample.image_names, ensure_ascii=False),
        "n_images": str(len(sample.image_names)),
        "text": sample.text or "",
    })

    chosen = select_image_files_for_call(sample, max_images=max_images)
    image_blocks = prepare_image_content_blocks(
        chosen, max_side=image_max_side, max_images=max_images,
    )
    if not image_blocks:
        return [{"role": "user",
                 "content": text_part + "\n\nReturn the JSON object only — no markdown fences, no commentary."}]
    roster_lines = []
    for i, p in enumerate(chosen, 1):
        roster_lines.append(f"  - image {i} → filename `{Path(p).name}`")
    tail = (
        "\nIMAGE ROSTER (in the order the images appear above):\n"
        + "\n".join(roster_lines)
        + "\n\nReturn the JSON object only — no markdown fences, no commentary."
    )
    content = [{"type": "text", "text": text_part}] + image_blocks + [{"type": "text", "text": tail}]
    return [{"role": "user", "content": content}]


def build_text_atom_messages(
    sample: BenchSample,
    prompt_template: str,
) -> list[dict[str, Any]]:
    """Text-only atomic-parse user message (NO images)."""
    text_part = _safe_render(prompt_template, {
        "sample_id": sample.sample_id,
        "language_code": sample.language_code or "unknown",
        "region_code": sample.region_code or "unknown",
        "theme_category": sample.theme_category or "unknown",
        "image_names_json": json.dumps(sample.image_names, ensure_ascii=False),
        "n_images": str(len(sample.image_names)),
        "text": sample.text or "",
    })
    return [{"role": "user",
             "content": text_part + "\n\nReturn the JSON object only — no markdown fences, no commentary."}]


def parse_atomic_points_split(
    sample: BenchSample,
    llm: LLMClient,
    *,
    prompts_dir: str,
    max_tokens_image: int = 6144,
    max_tokens_cross_modal: int = 6144,
    max_tokens_text: int = 8192,
    image_max_side: int = 896,
    cross_modal_image_max_side: int = 768,
    max_images: int | None = 6,
    temperature_image: float = 0.0,
    temperature_cross_modal: float = 0.0,
    temperature_text: float = 0.0,
    enable_thinking: bool = False,
    image_prompt_name: str = "image_atom_parse",
    cross_modal_prompt_name: str = "cross_modal_atom_parse",
    text_prompt_name: str = "text_atom_parse",
) -> tuple[dict[str, Any], dict[str, LLMResponse | None], dict[str, Exception | None]]:
    """Run THREE independent qwen atomic-parse calls in PARALLEL and merge them.

    Returns:
      merged          — dict with the same schema downstream code expects:
                        post_summary, image_level, cross_modal, sentence_level,
                        paragraph_level, retrieval_queries, visual_indicators,
                        plus split-mode extras: cross_modal_findings,
                        quoted_text_and_attributions.
      raw_responses   — {"image": LLMResponse|None, "cross_modal": ..., "text": ...}
      errors          — {"image": Exception|None, ...}

    A failure in any single pass is non-fatal — the merge proceeds with whatever
    passes succeeded. If ALL three pass-passes fail, an exception is raised.
    """
    extra = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}

    image_tmpl = _load_prompt_named(prompts_dir, image_prompt_name)
    cm_tmpl = _load_prompt_named(prompts_dir, cross_modal_prompt_name)
    text_tmpl = _load_prompt_named(prompts_dir, text_prompt_name)

    image_msgs = build_image_atom_messages(
        sample, image_tmpl,
        image_max_side=image_max_side, max_images=max_images,
    )
    cm_msgs = build_cross_modal_atom_messages(
        sample, cm_tmpl,
        image_max_side=cross_modal_image_max_side, max_images=max_images,
    )
    text_msgs = build_text_atom_messages(sample, text_tmpl)

    def _call(messages, max_tokens, temperature):
        return llm.chat(messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_body=extra)

    raw: dict[str, LLMResponse | None] = {"image": None, "cross_modal": None, "text": None}
    errs: dict[str, Exception | None] = {"image": None, "cross_modal": None, "text": None}
    parsed: dict[str, dict[str, Any] | None] = {"image": None, "cross_modal": None, "text": None}

    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            "image": pool.submit(_call, image_msgs, max_tokens_image, temperature_image),
            "cross_modal": pool.submit(_call, cm_msgs, max_tokens_cross_modal, temperature_cross_modal),
            "text": pool.submit(_call, text_msgs, max_tokens_text, temperature_text),
        }
        for k, fut in futs.items():
            try:
                r = fut.result()
                raw[k] = r
                try:
                    obj = extract_json_block(r.content)
                    if isinstance(obj, dict):
                        parsed[k] = obj
                    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                        parsed[k] = obj[0]
                    else:
                        errs[k] = ValueError(
                            f"{k} parser returned non-object: {type(obj)}"
                        )
                except Exception as exc:  # noqa: BLE001
                    errs[k] = exc
            except Exception as exc:  # noqa: BLE001
                errs[k] = exc

    if not any(parsed.values()):
        first_err = next((e for e in errs.values() if e is not None), None)
        raise RuntimeError(
            f"all 3 split-atomic passes failed; first error: {first_err!r}"
        )

    merged = _merge_split_atomic(parsed)
    return merged, raw, errs


def _merge_split_atomic(parsed: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Combine the three independent parser outputs into the standard schema.

    Resulting schema (BACKWARD-COMPATIBLE with the rest of the codebase):
      post_summary               — from text parser (longest available wins)
      sentence_level             — from text parser
      paragraph_level            — from text parser
      retrieval_queries          — from text parser (plus the per-image OCR / scene)
      image_level                — from image parser
      visual_indicators          — from image parser
      cross_modal                — from cross_modal parser
      cross_modal_findings       — from cross_modal parser (NEW; flattened by retrieval helper)
      quoted_text_and_attributions — from text parser (NEW)
    """
    img = parsed.get("image") or {}
    cm = parsed.get("cross_modal") or {}
    txt = parsed.get("text") or {}

    out: dict[str, Any] = {}
    out["post_summary"] = (txt.get("post_summary") or "").strip()
    out["sentence_level"] = txt.get("sentence_level") or []
    out["paragraph_level"] = txt.get("paragraph_level") or []
    out["retrieval_queries"] = txt.get("retrieval_queries") or []
    out["quoted_text_and_attributions"] = txt.get("quoted_text_and_attributions") or []

    out["image_level"] = img.get("image_level") or []
    out["visual_indicators"] = img.get("visual_indicators") or []

    out["cross_modal"] = cm.get("cross_modal") or []
    out["cross_modal_findings"] = cm.get("cross_modal_findings") or []

    return out


def flatten_split_atom_passes_for_artifact(
    raw: dict[str, LLMResponse | None],
    errs: dict[str, Exception | None],
) -> dict[str, Any]:
    """Build a JSON-serialisable summary of the three split passes for disk."""
    out: dict[str, Any] = {}
    for k in ("image", "cross_modal", "text"):
        r = raw.get(k)
        e = errs.get(k)
        out[k] = {
            "content": (r.content if r else None),
            "reasoning": (r.reasoning if r else None),
            "finish_reason": (r.finish_reason if r else None),
            "usage": (r.usage if r else None),
            "error": (repr(e) if e else None),
        }
    return out
