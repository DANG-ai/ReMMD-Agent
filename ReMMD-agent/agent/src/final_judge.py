"""Final judge: vision-language model takes the post (text + images), atomic
points (incl. structured visual indicators), retrieved evidence, and emits a
3-level decision (verdict / taxonomy / rationale)."""
from __future__ import annotations

import json
import logging
import re as _re
from pathlib import Path
from typing import Any

from .data import BenchSample, prepare_image_content_blocks, select_image_files_for_call
from .labels import (
    LEVEL1_LABELS,
    LEVEL2_LABELS,
    load_level1_doc,
    load_level2_doc,
    normalize_level1,
    normalize_level2_list,
)
from .llm import LLMClient, LLMResponse, extract_json_block, select_prompt_filename
from .rag import RetrievedEvidence
from .search_tools import SearchResult


logger = logging.getLogger("remmd.judge")


def _heuristic_extract_judge_fields(content: str) -> dict[str, Any]:
    """Best-effort regex extraction of level1 / level2 / level3 from a malformed
    judge JSON. Used as a final fallback when `extract_json_block` cannot
    parse the LLM output (e.g. because the LLM wrote literal newlines or
    unescaped quotes inside a string field, OR the response was truncated by
    max_tokens). The recovered fields are STILL the LLM's own output — we
    only substitute the broken JSON parser, not the LLM's verdict.
    """
    out: dict[str, Any] = {}
    if not content:
        return out
    # level1_verdict — match the canonical labels directly so we tolerate
    # slight formatting glitches around the value.
    m = _re.search(r'"level1_verdict"\s*:\s*"(True|Mostly True|Mixture|Mostly False|False)"', content)
    if m:
        out["level1_verdict"] = m.group(1)
    else:
        m2 = _re.search(r'"level1_verdict"\s*:\s*"([^"\n]{1,40})"', content)
        if m2:
            out["level1_verdict"] = m2.group(1)
    # level2_taxonomy — find the bracket block and pull canonical labels out.
    m = _re.search(r'"level2_taxonomy"\s*:\s*\[(.*?)\]', content, _re.DOTALL)
    if m:
        block = m.group(1)
        l2 = _re.findall(r'"([^"\n]{1,80})"', block)
        out["level2_taxonomy"] = l2
    # level3_rationale — capture the value (allow escaped quotes inside).
    m = _re.search(r'"level3_rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', content, _re.DOTALL)
    if m:
        out["level3_rationale"] = m.group(1).replace("\\n", "\n").replace('\\"', '"')
    # Optional: pick up findings_counts via simple heuristic for downstream
    # JSON consumers (we only need the four keys).
    counts = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIED": 0}
    for label in counts:
        # match all "status": "<LABEL>" occurrences
        pattern = r'"status"\s*:\s*"' + label + r'"'
        counts[label] = len(_re.findall(pattern, content))
    out["_recovered_findings_counts"] = counts
    return out


def _load_prompt(
    prompts_dir: str,
    model: str | None = None,
    prompt_name: str | None = None,
) -> str:
    """Load the final-judge prompt.

    Priority:
      1. `prompt_name` (explicit config override, e.g. "final_judge_v2") wins.
      2. Otherwise we fall back to the model-aware selector (qwen -> final_judge.txt,
         gpt-5 family -> final_judge_gpt.txt when present).
    """
    if prompt_name:
        return Path(prompts_dir, prompt_name + ".txt").read_text(encoding="utf-8")
    fname = select_prompt_filename(prompts_dir, "final_judge", model)
    return Path(prompts_dir, fname).read_text(encoding="utf-8")


def _format_evidence_block(evidences: list[RetrievedEvidence]) -> str:
    if not evidences:
        return "(no evidence retrieved)"
    parts: list[str] = []
    for i, ev in enumerate(evidences, 1):
        parts.append(
            f"[{i}] id={ev.evidence_id} | type={ev.evidence_type} | score={ev.score:.3f}\n"
            f"{ev.text.strip()}"
        )
    return "\n\n".join(parts)


def _format_search_block(search: dict[str, list[SearchResult]]) -> str:
    if not search or not any(v for v in search.values()):
        return "(no external search results — search tools disabled or returned nothing)"
    parts: list[str] = []
    for tool, results in search.items():
        if not results:
            continue
        parts.append(f"== {tool} ==")
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{tool} #{i}] {r.title}\n{r.snippet}\n{r.url}"
            )
    return "\n\n".join(parts) if parts else "(no external search results)"


def _slim_atoms(parsed: dict[str, Any]) -> dict[str, Any]:
    """Compact view of atomic parser output for the final-judge prompt.

    Includes the new `visual_indicators` field so the judge sees what the
    upstream vision pass found per image.
    """
    return {
        "post_summary": parsed.get("post_summary", ""),
        "image_level": parsed.get("image_level", []),
        "cross_modal": parsed.get("cross_modal", []),
        "sentence_level": parsed.get("sentence_level", []),
        "paragraph_level": parsed.get("paragraph_level", []),
        "visual_indicators": parsed.get("visual_indicators", []),
    }


def _safe_render(template: str, mapping: dict[str, Any]) -> str:
    """Safer `{key}` substitution that does not break on JSON braces in the body."""
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _format_pattern_hint(pattern_hint: dict[str, Any] | None) -> str:
    if not pattern_hint:
        return "(no pattern hint available)"
    active = pattern_hint.get("active_labels") or []
    parts = []
    parts.append(
        "An upstream PATTERN DETECTOR (a focused L2 yes/no pass) flagged these as PRESENT: "
        + (", ".join(f"`{x}`" for x in active) if active else "(none)")
    )
    pat = pattern_hint.get("patterns") or {}
    parts.append("Per-pattern findings:")
    for lbl, info in pat.items():
        present = info.get("present", False)
        ev = info.get("evidence_id") or "—"
        expl = (info.get("explanation") or "").strip().replace("\n", " ")
        parts.append(f"  • {lbl}: present={'YES' if present else 'no'} (ev={ev}) — {expl[:160]}")
    parts.append("Treat these as a STRONG PRIOR. Only override a flagged label when the evidence clearly does not justify it.")
    return "\n".join(parts)


def _format_text_analysis_hint(text_analysis: dict[str, Any] | None) -> str:
    if not text_analysis:
        return "(no text analyzer output)"
    active = text_analysis.get("active_labels") or []
    align = text_analysis.get("alignment_level") or ""
    align_reason = text_analysis.get("alignment_reasoning") or ""
    parts = []
    parts.append("Upstream TEXT ANALYZER (focused T1/T2/T3/C1 yes-no pass) verdict:")
    if align:
        parts.append(f"  Overall post↔evidence alignment: {align} — {align_reason[:200]}")
    parts.append(
        "  Active L2 labels: "
        + (", ".join(f"`{x}`" for x in active) if active else "(none flagged)")
    )
    for prefix, label, ev_field, sub_field in [
        ("T1", "T1 Fabrication", "t1_evidence", "t1_problematic_subclaims"),
        ("T2", "T2 Distortion", "t2_evidence", "t2_problematic_subclaims"),
        ("T3", "T3 Misleading Context", "t3_evidence", "t3_problematic_subclaims"),
        ("C1", "C1 Semantic Inconsistency", "c1_evidence", "c1_problematic_subclaims"),
    ]:
        present = label in active
        ev = (text_analysis.get(ev_field) or "").strip().replace("\n", " ")
        subs = text_analysis.get(sub_field) or []
        parts.append(
            f"  • {label}: present={'YES' if present else 'no'} | evidence='{ev[:200]}'"
            + (f" | sub-claims={subs[:2]}" if subs else "")
        )
    parts.append(
        "  Use these as a STRONG textual prior. Re-confirm by reading the post + evidence yourself, "
        "but do NOT silently drop a flagged T1/T2/T3/C1 unless evidence directly contradicts it."
    )
    return "\n".join(parts)


def _format_image_analysis_hint(image_analysis: dict[str, Any] | None) -> str:
    if not image_analysis:
        return "(no image analyzer output — sample may have no images, or analyzer was disabled)"
    active = image_analysis.get("active_labels") or []
    images = image_analysis.get("images") or []
    parts = []
    parts.append(
        "Upstream IMAGE ANALYZER (focused per-image vision pass) verdict:"
    )
    parts.append(
        "  Active L2 labels (from union over images): "
        + (", ".join(f"`{x}`" for x in active) if active else "(none flagged)")
    )
    if images:
        parts.append("  Per-image findings:")
        for im in images:
            idx = im.get("image_index", "?")
            name = im.get("image_name", "?")
            flags = []
            if im.get("v1_synthetic_present"):
                flags.append(f"V1[{im.get('v1_evidence', '')[:80]}]")
            if im.get("v2_edit_present"):
                flags.append(f"V2[{im.get('v2_evidence', '')[:80]}]")
            if im.get("c1_entity_mismatch"):
                flags.append("C1")
            if im.get("c2_context_mismatch"):
                flags.append("C2")
            if im.get("c3_pragmatic_mismatch"):
                flags.append("C3")
            mc = im.get("matches_evidence_image_content")
            mp = im.get("matches_post_claim_about_image")
            scene = (im.get("scene_description") or "").replace("\n", " ")[:160]
            ocr = (im.get("ocr_text") or "").replace("\n", " ")[:80]
            parts.append(
                f"    • image {idx} ({name}): flags={flags or 'NONE'} | "
                f"matches_post={mp} | matches_evidence={mc} | scene='{scene}' | ocr='{ocr}'"
            )
    parts.append(
        "  Use these as a STRONG vision-grounded prior. Re-confirm by looking at the actual images, "
        "but do NOT silently drop V1/V2/C labels the analyzer flagged unless the evidence directly contradicts them."
    )
    return "\n".join(parts)


def build_judge_messages(
    sample: BenchSample,
    *,
    parsed_atoms: dict[str, Any],
    retrieved: list[RetrievedEvidence],
    search_hits: dict[str, list[SearchResult]] | None,
    prompts_dir: str,
    level1_doc_path: str,
    level2_doc_path: str,
    pattern_hint: dict[str, Any] | None = None,
    image_analysis: dict[str, Any] | None = None,
    text_analysis: dict[str, Any] | None = None,
    image_max_side: int = 768,
    max_images: int | None = 6,
    model: str | None = None,
    prompt_name: str | None = None,
) -> list[dict[str, Any]]:
    tmpl = _load_prompt(prompts_dir, model=model, prompt_name=prompt_name)
    text_part = _safe_render(tmpl, {
        "level1_choices": ", ".join(f"\"{x}\"" for x in LEVEL1_LABELS),
        "level2_choices": ", ".join(f"\"{x}\"" for x in LEVEL2_LABELS),
        "level1_doc": load_level1_doc(level1_doc_path),
        "level2_doc": load_level2_doc(level2_doc_path),
        "sample_id": sample.sample_id,
        "language_code": sample.language_code or "unknown",
        "region_code": sample.region_code or "unknown",
        "theme_category": sample.theme_category or "unknown",
        "image_names_json": json.dumps(sample.image_names, ensure_ascii=False),
        "n_images": str(len(sample.image_names)),
        "post_text": sample.text or "",
        "atomic_points_json": json.dumps(_slim_atoms(parsed_atoms), ensure_ascii=False, indent=2),
        "n_evidence": str(len(retrieved)),
        "evidence_block": _format_evidence_block(retrieved),
        "search_block": _format_search_block(search_hits or {}),
        "pattern_hint_block": _format_pattern_hint(pattern_hint),
        "image_analysis_block": _format_image_analysis_hint(image_analysis),
        "text_analysis_block": _format_text_analysis_hint(text_analysis),
    })

    chosen = select_image_files_for_call(sample, max_images=max_images)
    image_blocks = prepare_image_content_blocks(
        chosen, max_side=image_max_side, max_images=max_images,
    )
    if image_blocks:
        roster_lines = []
        for i, p in enumerate(chosen, 1):
            roster_lines.append(f"  - image {i} → filename `{Path(p).name}`")
        tail = (
            "\nIMAGE ROSTER (in the same order shown above):\n"
            + "\n".join(roster_lines)
            + "\n\nProduce the JSON object only. No markdown fences. No commentary."
        )
        content = [{"type": "text", "text": text_part}] + image_blocks + [{"type": "text", "text": tail}]
    else:
        content = text_part + "\n\nProduce the JSON object only. No markdown fences. No commentary."

    return [{"role": "user", "content": content}]


_SEVERE_L2 = {"T1 Fabrication", "V1 Synthetic Visual Content"}


_L1_ORDER = {"True": 0, "Mostly True": 1, "Mixture": 2, "Mostly False": 3, "False": 4}


def _summarize_findings(findings: Any) -> dict[str, int]:
    """Count subclaim_findings by status. Tolerates string-typo variants."""
    counts = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIED": 0}
    if not isinstance(findings, list):
        return counts
    for f in findings:
        if not isinstance(f, dict):
            continue
        s = str(f.get("status", "")).strip().upper().replace(" ", "_").replace("-", "_")
        if s in ("SUPPORTED", "AGREE", "AGREED"):
            counts["SUPPORTED"] += 1
        elif s in ("PARTIALLY_SUPPORTED", "PARTIAL", "PARTIALLY"):
            counts["PARTIALLY_SUPPORTED"] += 1
        elif s in ("CONTRADICTED", "CONTRADICT", "CONTRADICTS"):
            counts["CONTRADICTED"] += 1
        elif s in ("UNVERIFIED", "SILENT", "UNKNOWN"):
            counts["UNVERIFIED"] += 1
    return counts


def _enforce_l1_l2_coupling(
    l1: str | None,
    l2: list[str],
    findings_counts: dict[str, int] | None = None,
) -> tuple[str | None, str | None]:
    """Label-definition-based consistency post-processor.

    These are NOT count-based rules. They are derived directly from the
    OFFICIAL L1 / L2 label definitions in 一级标签.txt and 二级标签.docx.
    The LLM judge is still the primary decision-maker; this only enforces
    self-consistency between the LLM's L1 and L2 outputs:

      R_TAX:    L2 non-empty AND L1 == "True"  ⇒ bump L1 to "Mostly True".
                (Definition of "True" forbids any distortion.)

      R_NIL:    L2 empty AND L1 != "True" AND findings show no problems ⇒ True.
                (Empty L2 + clean findings is by definition True.)

      R_T1V1:   T1 AND V1 BOTH in L2 ⇒ floor L1 at "Mostly False".
                (Joint fabrication of facts AND visuals: per definitions of
                 T1 + V1, the post lacks both real-content and real-visual
                 basis ⇒ "Mostly False" or worse.)

      R_T1V1_SEVERE:
                T1 AND V1 BOTH in L2 AND ≥ 2 companion L2 labels present
                ⇒ floor L1 at "False".
                (Per the L1 definition of "False": "core claim has no real
                 basis; remaining factual fragments do not support the
                 central message." When BOTH facts (T1) AND visuals (V1)
                 are fabricated AND multiple cross-modal / editing
                 inconsistencies pile on, neither the textual evidence
                 nor the visual evidence supports the core claim — the
                 post matches the "False" definition.
                 This rule is anchored in label composition only, not in
                 label COUNT — the trigger is "T1 + V1 + ≥ 2 companions",
                 a specific severe-fabrication signature.)

    These rules are LABEL-COMPOSITION based (specific labels' definitions),
    NOT label-COUNT based. They generalise across datasets because they
    are anchored in the definitional semantics of the labels themselves.
    """
    if l1 is None:
        return l1, None
    f = findings_counts or {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "CONTRADICTED": 0, "UNVERIFIED": 0}
    rules: list[str] = []
    new_l1 = l1

    # R_TAX: L2 non-empty implies the post is not "True".
    if l2 and new_l1 == "True":
        new_l1 = "Mostly True"
        rules.append("R_TAX_l2_nonempty_implies_not_true")

    # R_NIL: L2 empty + clean findings → True. Only fires when there is no
    # hint of any problem (no contradictions, no partial support).
    if not l2 and new_l1 != "True":
        if f["CONTRADICTED"] == 0 and f["PARTIALLY_SUPPORTED"] == 0:
            new_l1 = "True"
            rules.append("R_NIL_l2_empty_clean_findings_implies_true")

    has_t1 = "T1 Fabrication" in l2
    has_v1 = "V1 Synthetic Visual Content" in l2
    companion_l2 = {x for x in l2
                    if x not in ("T1 Fabrication", "V1 Synthetic Visual Content")}

    # R_T1V1 (severe joint signal): if BOTH T1 AND V1 are present, both
    # content AND visuals are fabricated by definition. Per the L1
    # definitions, "True" / "Mostly True" / "Mixture" all require the
    # core claim to have substantial real basis — incompatible with
    # joint content+visual fabrication. Floor at "Mostly False".
    if has_t1 and has_v1 and _L1_ORDER[new_l1] < _L1_ORDER["Mostly False"]:
        new_l1 = "Mostly False"
        rules.append("R_T1_AND_V1_implies_at_least_mostly_false")

    # R_T1V1_SEVERE: T1 + V1 + ≥ 3 companion L2 labels matches the
    # definitional signature of "False" (core claim wholly fabricated AND
    # visuals fabricated AND multiple cross-modal inconsistencies). The
    # model frequently knows the post is severely problematic but stops
    # at "Mostly False" out of caution; this rule applies the L1 definition
    # mechanically when the L2 composition meets the bar.
    # We use ≥3 (not ≥2) to err on the side of letting Mostly False stand
    # for borderline cases; only the truly severe joint-fabrication patterns
    # are promoted to False.
    if has_t1 and has_v1 and len(companion_l2) >= 3 and _L1_ORDER[new_l1] < _L1_ORDER["False"]:
        new_l1 = "False"
        rules.append("R_T1V1_plus_3_companions_implies_false")

    rule = ";".join(rules) if rules else None
    return new_l1, rule


def parse_judge_output(
    content: str,
    *,
    apply_coupling: bool = True,
    apply_fallback: bool = True,
) -> dict[str, Any]:
    """Parse judge JSON.

    The LLM judge is the source of truth for L1 / L2 / L3. We only:
      1) Normalise the literal values to the canonical L1 / L2 label sets
         (this is data-layer normalisation — e.g. "true" → "True", or
         dropping unknown L2 labels — NOT a verdict override).

      2) Optionally apply the two label-composition consistency rules in
         `_enforce_l1_l2_coupling`. SET `apply_coupling=False` to honour
         the LLM verdict verbatim — required when the user wants verdicts
         to come ENTIRELY from the LLM with no code-side rules.

      3) Optionally apply a tiny `findings_counts`-based default when the
         LLM returns a non-canonical L1 token (e.g. a typo). SET
         `apply_fallback=False` to leave `level1_verdict` as None instead.

    Args:
        content: raw LLM response text (JSON expected).
        apply_coupling: when True (default), run the L1/L2 consistency
            post-processor in `_enforce_l1_l2_coupling`. When False, the
            LLM's L1 verdict is used verbatim and the only modification is
            the canonical-label normalisation in `normalize_level1` /
            `normalize_level2_list`.
        apply_fallback: when True (default), pick a sensible L1 default
            from the LLM's own findings if its L1 token was non-canonical.
            When False, `level1_verdict` will be None when the model
            emits a non-canonical token.
    """
    recovery_used: str | None = None
    try:
        obj = extract_json_block(content)
    except Exception as exc:  # noqa: BLE001
        # JSON unparseable — attempt regex fallback to surface the LLM's
        # own L1/L2/L3 output. This is NOT a code-side rule override; it is
        # a parsing rescue — the recovered values come from the LLM verbatim.
        rec = _heuristic_extract_judge_fields(content)
        if rec.get("level1_verdict") is None and not rec.get("level2_taxonomy"):
            raise ValueError(f"judge JSON unparseable and regex fallback empty: {exc!r}") from exc
        logger.warning("judge JSON parse failed (%s); regex-recovered %d keys",
                       exc, len(rec))
        obj = {
            "level1_verdict": rec.get("level1_verdict"),
            "level2_taxonomy": rec.get("level2_taxonomy", []),
            "level3_rationale": rec.get("level3_rationale", ""),
            "_recovered_findings_counts": rec.get("_recovered_findings_counts"),
            "_parse_recovery": "regex_fallback_after_json_failure",
        }
        recovery_used = "regex_fallback"

    if not isinstance(obj, dict):
        # Top-level JSON parsed but to the wrong shape (sub-object scrape, list, etc.).
        # Try regex recovery before giving up.
        rec = _heuristic_extract_judge_fields(content)
        if rec.get("level1_verdict") or rec.get("level2_taxonomy"):
            obj = {
                "level1_verdict": rec.get("level1_verdict"),
                "level2_taxonomy": rec.get("level2_taxonomy", []),
                "level3_rationale": rec.get("level3_rationale", ""),
                "_recovered_findings_counts": rec.get("_recovered_findings_counts"),
                "_parse_recovery": "regex_fallback_after_wrong_json_shape",
            }
            recovery_used = "regex_fallback_wrong_shape"
        else:
            raise ValueError(f"judge returned non-object: {type(obj)}")

    # Heuristic: if extract_json_block grabbed a sub-object (e.g. one of the
    # `subclaim_findings` entries) instead of the outer object, there will be
    # NO `level1_verdict` / `level2_taxonomy` keys. Detect this and fall back
    # to the regex extraction over the FULL content.
    has_judge_keys = (
        "level1_verdict" in obj or "level2_taxonomy" in obj or "verdict" in obj
    )
    if not has_judge_keys:
        rec = _heuristic_extract_judge_fields(content)
        if rec.get("level1_verdict") or rec.get("level2_taxonomy"):
            logger.warning("judge sub-object scrape detected; using regex fallback "
                           "(found keys: %s)", list(obj.keys())[:5])
            obj = {
                "level1_verdict": rec.get("level1_verdict"),
                "level2_taxonomy": rec.get("level2_taxonomy", []),
                "level3_rationale": rec.get("level3_rationale", ""),
                "_recovered_findings_counts": rec.get("_recovered_findings_counts"),
                "_parse_recovery": "regex_fallback_after_subobject_scrape",
            }
            recovery_used = "regex_fallback_subobject"

    raw_l1 = obj.get("level1_verdict") or obj.get("verdict")
    raw_l2 = obj.get("level2_taxonomy") or obj.get("distortion_taxonomy") or []
    rationale = obj.get("level3_rationale") or obj.get("rationale") or ""

    norm_l1 = normalize_level1(raw_l1)
    norm_l2 = normalize_level2_list(raw_l2 if isinstance(raw_l2, list) else [raw_l2])
    if "_recovered_findings_counts" in obj and obj["_recovered_findings_counts"]:
        findings_counts = obj["_recovered_findings_counts"]
    else:
        findings_counts = _summarize_findings(obj.get("subclaim_findings"))

    fallback_used = None
    if norm_l1 is None and apply_fallback:
        # Very rare: the model emitted a non-canonical L1 token.
        # Use the model's own findings to pick a reasonable default.
        # We deliberately do NOT consult L2 count here so we avoid the
        # over-fitting that count-based derivation tends to produce.
        if findings_counts.get("CONTRADICTED", 0) >= 2:
            norm_l1 = "Mostly False"
        elif findings_counts.get("CONTRADICTED", 0) == 1 or findings_counts.get("PARTIALLY_SUPPORTED", 0) >= 2:
            norm_l1 = "Mixture"
        elif findings_counts.get("PARTIALLY_SUPPORTED", 0) == 1:
            norm_l1 = "Mostly True"
        elif findings_counts.get("SUPPORTED", 0) > 0:
            norm_l1 = "True"
        else:
            norm_l1 = "Mostly True"
        fallback_used = "model_emitted_non_canonical_l1"

    if apply_coupling:
        coupled_l1, rule = _enforce_l1_l2_coupling(norm_l1, norm_l2, findings_counts)
    else:
        coupled_l1, rule = norm_l1, None

    if fallback_used is not None:
        rule = (rule + ";" if rule else "") + fallback_used
    if recovery_used is not None:
        rule = (rule + ";" if rule else "") + "parse_recovery=" + recovery_used
    out = {
        "level1_verdict": coupled_l1,
        "level1_verdict_raw": raw_l1,
        "level1_verdict_pre_coupling": norm_l1,
        "level1_coupling_rule_applied": rule,
        "subclaim_findings_counts": findings_counts,
        "level2_taxonomy": norm_l2,
        "level2_taxonomy_raw": raw_l2,
        "level3_rationale": rationale.strip() if isinstance(rationale, str) else "",
        "supporting_evidence_ids": obj.get("supporting_evidence_ids") or [],
        "reasoning_trace": obj.get("reasoning_trace") or "",
    }
    for k in ("core_claim", "key_subclaims", "subclaim_findings",
              "evidence_stances", "verdict_reasoning",
              "image_findings"):
        if k in obj:
            out[k] = obj[k]
    return out


def run_final_judge(
    sample: BenchSample,
    llm: LLMClient,
    *,
    parsed_atoms: dict[str, Any],
    retrieved: list[RetrievedEvidence],
    search_hits: dict[str, list[SearchResult]] | None,
    prompts_dir: str,
    level1_doc_path: str,
    level2_doc_path: str,
    max_tokens: int = 6144,
    image_max_side: int = 768,
    max_images: int | None = 6,
    model: str | None = None,
) -> tuple[dict[str, Any], LLMResponse]:
    messages = build_judge_messages(
        sample,
        parsed_atoms=parsed_atoms,
        retrieved=retrieved,
        search_hits=search_hits,
        prompts_dir=prompts_dir,
        level1_doc_path=level1_doc_path,
        level2_doc_path=level2_doc_path,
        image_max_side=image_max_side,
        max_images=max_images,
        model=model,
    )
    resp = llm.chat(messages, max_tokens=max_tokens)
    parsed = parse_judge_output(resp.content)
    return parsed, resp
