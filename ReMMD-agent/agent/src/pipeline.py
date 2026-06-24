"""End-to-end per-sample pipeline.

Stages:
  1. Atomic-point parsing (LLM #1)
  2. Per-atom RAG retrieval against the sample's memory bank
  3. External search (skipped when api keys are empty)
  4. Final judge (LLM #2)
  5. Persist all process artifacts to disk
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from .atomic_parser import flatten_atoms_for_retrieval, parse_atomic_points
from .data import BenchSample, load_sample
from .embedder import EmbeddingClient
from .final_judge import run_final_judge
from .image_analyzer import (
    build_image_analyzer_messages,
    parse_image_analyzer_output,
    run_image_analyzer,
)
from .text_analyzer import (
    build_text_analyzer_messages,
    parse_text_analyzer_output,
    run_text_analyzer,
)
from .llm import LLMClient
from .logging_utils import sample_done, write_sample_artifact
from .pattern_detector import build_pattern_messages, parse_pattern_output, derive_l1_from_patterns
from .rag import RagIndex, retrieve_for_atoms
from .search_tools import SearchResult, run_all_searches


logger = logging.getLogger("remmd.pipeline")


def _content_to_text_only(content: Any) -> str:
    """Strip image_url blocks from a multimodal `content` payload for on-disk
    artifact storage. We keep only the text segments and a `[image #N]`
    placeholder so the prompt is auditable without re-saving 5×100KB JPEGs.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    img_n = 0
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image_url":
            img_n += 1
            parts.append(f"[image #{img_n} omitted from artifact]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(parts)


def _serialize_evidence(items) -> list[dict[str, Any]]:
    out = []
    for r in items:
        out.append({
            "evidence_id": r.evidence_id,
            "evidence_type": r.evidence_type,
            "score": r.score,
            "matched_atomic_idx": r.matched_atomic_idx,
            "text": r.text,
        })
    return out


def _serialize_search(search: dict[str, list[SearchResult]]) -> dict[str, list[dict[str, Any]]]:
    return {k: [dataclasses.asdict(r) for r in v] for k, v in (search or {}).items()}


def run_sample(
    sample: BenchSample,
    *,
    llm: LLMClient,
    embedder: EmbeddingClient,
    index: RagIndex,
    cfg: dict[str, Any],
    run_dir: Path,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Run the full agent pipeline on a single sample.

    Returns a result dict; also writes process artifacts under run_dir/samples/<sid>/.
    """
    sid = sample.sample_id
    t0 = time.time()
    result: dict[str, Any] = {
        "sample_id": sid,
        "status": "pending",
        "language_code": sample.language_code,
        "region_code": sample.region_code,
        "theme_category": sample.theme_category,
        "image_names": sample.image_names,
        "gold": {
            "verdict": sample.gold_verdict,
            "distortion_taxonomy": sample.gold_taxonomy,
            "rationale": sample.gold_rationale,
        },
        "timings": {},
        "errors": [],
    }

    vision_cfg = cfg.get("vision", {})
    image_max_side = int(vision_cfg.get("max_image_side", 768))
    max_images = vision_cfg.get("max_images_per_call", 6)
    if max_images is not None:
        max_images = int(max_images)

    # Pipeline-level switches (all default to ORIGINAL behaviour for safety).
    pipe_cfg = cfg.get("pipeline", {})
    atomic_mode = str(pipe_cfg.get("atomic_mode", "multi")).lower()
    # Resolve optional explicit prompt names (so qwen v2 can opt in without
    # touching the model-aware default selector used by single/multi mode).
    judge_prompt_name = pipe_cfg.get("judge_prompt_name")
    image_analyzer_prompt_name = pipe_cfg.get("image_analyzer_prompt_name")
    text_analyzer_prompt_name = pipe_cfg.get("text_analyzer_prompt_name")
    # Split-mode prompt names (defaults match the new files; can be overridden).
    split_image_prompt_name = pipe_cfg.get(
        "split_image_prompt_name", "image_atom_parse")
    split_cross_modal_prompt_name = pipe_cfg.get(
        "split_cross_modal_prompt_name", "cross_modal_atom_parse")
    split_text_prompt_name = pipe_cfg.get(
        "split_text_prompt_name", "text_atom_parse")

    # ---------- 1) Atomic parsing ----------
    from .atomic_parser import (  # noqa: PLC0415
        _load_prompt,
        _load_prompt_named,
        build_atomic_messages,
        build_cross_modal_atom_messages,
        build_image_atom_messages,
        build_text_atom_messages,
        flatten_split_atom_passes_for_artifact,
        parse_atomic_points_multi,
        parse_atomic_points_split,
    )
    n_atomic_passes = int(pipe_cfg.get("atomic_n_passes", 1))
    try:
        t_a = time.time()

        # ===== SPLIT MODE — three independent qwen calls (image/cm/text) =====
        if atomic_mode == "split":
            # Persist the three prompts (text-only) for auditability.
            if save_artifacts:
                image_tmpl = _load_prompt_named(
                    cfg["paths"]["prompts_dir"], split_image_prompt_name)
                cm_tmpl = _load_prompt_named(
                    cfg["paths"]["prompts_dir"], split_cross_modal_prompt_name)
                text_tmpl = _load_prompt_named(
                    cfg["paths"]["prompts_dir"], split_text_prompt_name)
                img_msgs_art = build_image_atom_messages(
                    sample, image_tmpl,
                    image_max_side=int(vision_cfg.get(
                        "max_image_side_split_image", 896)),
                    max_images=max_images,
                )
                cm_msgs_art = build_cross_modal_atom_messages(
                    sample, cm_tmpl,
                    image_max_side=image_max_side,
                    max_images=max_images,
                )
                text_msgs_art = build_text_atom_messages(sample, text_tmpl)
                write_sample_artifact(
                    run_dir, sid, "00_atomic_prompt_split_image.txt",
                    _content_to_text_only(img_msgs_art[0]["content"]),
                )
                write_sample_artifact(
                    run_dir, sid, "00_atomic_prompt_split_cross_modal.txt",
                    _content_to_text_only(cm_msgs_art[0]["content"]),
                )
                write_sample_artifact(
                    run_dir, sid, "00_atomic_prompt_split_text.txt",
                    _content_to_text_only(text_msgs_art[0]["content"]),
                )

            parsed_atoms, raw_responses_split, split_errs = parse_atomic_points_split(
                sample,
                llm,
                prompts_dir=cfg["paths"]["prompts_dir"],
                max_tokens_image=cfg["llm"].get("max_tokens_atomic_image", 6144),
                max_tokens_cross_modal=cfg["llm"].get(
                    "max_tokens_atomic_cross_modal", 6144),
                max_tokens_text=cfg["llm"].get("max_tokens_atomic_text", 8192),
                image_max_side=int(vision_cfg.get(
                    "max_image_side_split_image", 896)),
                cross_modal_image_max_side=image_max_side,
                max_images=max_images,
                temperature_image=cfg["llm"].get(
                    "atomic_image_temperature",
                    cfg["llm"].get("temperature", 0.0)),
                temperature_cross_modal=cfg["llm"].get(
                    "atomic_cross_modal_temperature",
                    cfg["llm"].get("temperature", 0.0)),
                temperature_text=cfg["llm"].get(
                    "atomic_text_temperature",
                    cfg["llm"].get("temperature", 0.0)),
                enable_thinking=bool(cfg["llm"].get(
                    "atomic_enable_thinking", False)),
                image_prompt_name=split_image_prompt_name,
                cross_modal_prompt_name=split_cross_modal_prompt_name,
                text_prompt_name=split_text_prompt_name,
            )
            if save_artifacts:
                write_sample_artifact(
                    run_dir, sid, "01_atomic_llm_raw_split.json",
                    flatten_split_atom_passes_for_artifact(
                        raw_responses_split, split_errs),
                )
            result["atomic_mode"] = "split"
            result["atomic_split_errors"] = {
                k: (repr(e) if e else None) for k, e in split_errs.items()
            }
            # aggregate usage across the three split calls
            agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for r in raw_responses_split.values():
                if r is None:
                    continue
                u = r.usage or {}
                for k in agg:
                    agg[k] += int(u.get(k, 0) or 0)
            result["atomic_usage"] = agg

        # ===== MULTI MODE — original N-pass concurrent single-shot parser =====
        elif atomic_mode == "multi" and n_atomic_passes > 1:
            prompt = _load_prompt(cfg["paths"]["prompts_dir"])
            if save_artifacts:
                messages_for_artifact = build_atomic_messages(
                    sample, prompt,
                    image_max_side=image_max_side,
                    max_images=max_images,
                )
                text_only = _content_to_text_only(
                    messages_for_artifact[0]["content"])
                write_sample_artifact(
                    run_dir, sid, "00_atomic_prompt.txt", text_only)
            parsed_atoms, raw_responses = parse_atomic_points_multi(
                sample,
                llm,
                prompts_dir=cfg["paths"]["prompts_dir"],
                n_passes=n_atomic_passes,
                max_tokens=cfg["llm"].get("max_tokens", 8192),
                image_max_side=image_max_side,
                max_images=max_images,
                enable_thinking=bool(cfg["llm"].get(
                    "atomic_enable_thinking", False)),
            )
            if save_artifacts:
                for i, r in enumerate(raw_responses):
                    write_sample_artifact(
                        run_dir, sid, f"01_atomic_llm_raw_pass{i}.json",
                        {
                            "content": r.content,
                            "reasoning": r.reasoning,
                            "finish_reason": r.finish_reason,
                            "usage": r.usage,
                        },
                    )
            result["atomic_mode"] = "multi"
            result["atomic_passes"] = n_atomic_passes
            result["atomic_n_successful"] = len(raw_responses)
            agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for r in raw_responses:
                u = r.usage or {}
                for k in agg:
                    agg[k] += int(u.get(k, 0) or 0)
            result["atomic_usage"] = agg

        # ===== SINGLE MODE — original single-call atomic parser =====
        else:
            prompt = _load_prompt(cfg["paths"]["prompts_dir"])
            if save_artifacts:
                messages_for_artifact = build_atomic_messages(
                    sample, prompt,
                    image_max_side=image_max_side,
                    max_images=max_images,
                )
                text_only = _content_to_text_only(
                    messages_for_artifact[0]["content"])
                write_sample_artifact(
                    run_dir, sid, "00_atomic_prompt.txt", text_only)
            messages = build_atomic_messages(
                sample, prompt,
                image_max_side=image_max_side,
                max_images=max_images,
            )
            atomic_extra = {
                "chat_template_kwargs": {
                    "enable_thinking": bool(cfg["llm"].get(
                        "atomic_enable_thinking", False))
                }
            }
            atom_resp = llm.chat(
                messages,
                max_tokens=cfg["llm"].get("max_tokens", 8192),
                extra_body=atomic_extra,
            )
            if save_artifacts:
                write_sample_artifact(run_dir, sid, "01_atomic_llm_raw.json", {
                    "content": atom_resp.content,
                    "reasoning": atom_resp.reasoning,
                    "finish_reason": atom_resp.finish_reason,
                    "usage": atom_resp.usage,
                })
            from .llm import extract_json_block  # noqa: PLC0415
            parsed_atoms = extract_json_block(atom_resp.content)
            if not isinstance(parsed_atoms, dict):
                raise ValueError(
                    f"atomic parser returned non-object: {type(parsed_atoms)}")
            for k in ("post_summary", "image_level", "cross_modal",
                      "sentence_level", "paragraph_level", "retrieval_queries",
                      "visual_indicators"):
                parsed_atoms.setdefault(k, [] if k != "post_summary" else "")
            result["atomic_mode"] = "single"
            result["atomic_usage"] = atom_resp.usage

        result["timings"]["atomic_parse_s"] = time.time() - t_a
        result["atomic_points"] = parsed_atoms
        if save_artifacts and pipe_cfg.get("save_atomic_points", True):
            write_sample_artifact(
                run_dir, sid, "01_atomic_points.json", parsed_atoms)
    except Exception as exc:  # noqa: BLE001
        msg = f"atomic parse failed: {exc!r}"
        logger.error("[sid=%s] %s\n%s", sid, msg, traceback.format_exc())
        result["errors"].append({"stage": "atomic_parse", "error": str(exc)})
        result["status"] = "atomic_parse_failed"
        if save_artifacts:
            write_sample_artifact(run_dir, sid, "result.json", result)
        return result

    # ---------- 2) RAG retrieval ----------
    try:
        t_r = time.time()
        atom_queries = flatten_atoms_for_retrieval(parsed_atoms)
        rag_cfg = cfg.get("rag", {})
        # if very few atoms, fall back to using post text as one mega-query
        if len(atom_queries) < 3 and sample.text:
            atom_queries.append(sample.text[: cfg["embedding"]["max_input_chars"]])
        retrieved = retrieve_for_atoms(
            index=index,
            embedder=embedder,
            sample_id=sid,
            atomic_points=atom_queries,
            top_k_per_atom=rag_cfg.get("top_k_per_atom", 5),
            min_score=rag_cfg.get("min_score", 0.0),
            max_evidence_per_sample=rag_cfg.get("max_evidence_per_sample", 18),
            per_type_quota=rag_cfg.get("per_type_quota") or None,
        )
        result["timings"]["rag_retrieve_s"] = time.time() - t_r
        result["retrieved_evidence"] = _serialize_evidence(retrieved)
        result["atom_queries"] = atom_queries
        if save_artifacts and cfg["pipeline"].get("save_retrieved_evidence", True):
            write_sample_artifact(run_dir, sid, "02_atom_queries.json", atom_queries)
            write_sample_artifact(run_dir, sid, "02_retrieved_evidence.json", result["retrieved_evidence"])
    except Exception as exc:  # noqa: BLE001
        msg = f"rag retrieve failed: {exc!r}"
        logger.error("[sid=%s] %s", sid, msg)
        result["errors"].append({"stage": "rag_retrieve", "error": str(exc)})
        retrieved = []
        result["retrieved_evidence"] = []
        result["atom_queries"] = []

    # ---------- 3) Search tools (skipped if keys empty) ----------
    try:
        t_s = time.time()
        # use only a handful of broadest queries
        search_queries = atom_queries[:5] if 'atom_queries' in result else []
        search_hits = run_all_searches(search_queries, cfg=cfg)
        result["timings"]["search_s"] = time.time() - t_s
        ser = _serialize_search(search_hits)
        result["search_hits"] = ser
        if save_artifacts:
            write_sample_artifact(run_dir, sid, "03_search_hits.json", ser)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[sid=%s] search failed: %s", sid, exc)
        result["errors"].append({"stage": "search", "error": str(exc)})
        search_hits = {}
        result["search_hits"] = {}

    # ---------- 3.3) Dedicated text-distortion analyzer ----------
    text_analysis: dict[str, Any] | None = None
    if cfg["pipeline"].get("use_text_analyzer", True):
        try:
            t_ta = time.time()
            ta_messages = build_text_analyzer_messages(
                sample,
                parsed_atoms=parsed_atoms,
                retrieved=retrieved,
                prompts_dir=cfg["paths"]["prompts_dir"],
                model=cfg["llm"].get("model"),
                prompt_name=text_analyzer_prompt_name,
            )
            if save_artifacts:
                write_sample_artifact(
                    run_dir, sid, "00_text_analyze_prompt.txt",
                    _content_to_text_only(ta_messages[0]["content"]),
                )
            ta_extra = {
                "chat_template_kwargs": {
                    "enable_thinking": bool(cfg["llm"].get("text_analyzer_enable_thinking", False))
                }
            }
            ta_resp = llm.chat(
                ta_messages,
                max_tokens=cfg["llm"].get("max_tokens_text_analyzer", 4096),
                temperature=cfg["llm"].get("text_analyzer_temperature",
                                            cfg["llm"].get("temperature", 0.0)),
                extra_body=ta_extra,
            )
            if save_artifacts:
                write_sample_artifact(run_dir, sid, "033_text_analyze_llm_raw.json", {
                    "content": ta_resp.content,
                    "reasoning": ta_resp.reasoning,
                    "finish_reason": ta_resp.finish_reason,
                    "usage": ta_resp.usage,
                })
            text_analysis = parse_text_analyzer_output(ta_resp.content)
            result["text_analysis"] = text_analysis
            result["timings"]["text_analyzer_s"] = time.time() - t_ta
            if save_artifacts:
                write_sample_artifact(run_dir, sid, "033_text_analyze_parsed.json", text_analysis)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sid=%s] text analyzer failed: %s", sid, exc)
            result["errors"].append({"stage": "text_analyzer", "error": str(exc)})
            text_analysis = None

    # ---------- 3.4) Dedicated image analyzer (vision pass) ----------
    image_analysis: dict[str, Any] | None = None
    if cfg["pipeline"].get("use_image_analyzer", True) and sample.image_files:
        try:
            t_ia = time.time()
            ia_messages = build_image_analyzer_messages(
                sample,
                parsed_atoms=parsed_atoms,
                retrieved=retrieved,
                prompts_dir=cfg["paths"]["prompts_dir"],
                image_max_side=int(vision_cfg.get("max_image_side_analyzer", 896)),
                max_images=max_images,
                model=cfg["llm"].get("model"),
                prompt_name=image_analyzer_prompt_name,
            )
            if ia_messages is not None:
                if save_artifacts:
                    write_sample_artifact(
                        run_dir, sid, "00_image_analyze_prompt.txt",
                        _content_to_text_only(ia_messages[0]["content"]),
                    )
                ia_extra = {
                    "chat_template_kwargs": {
                        "enable_thinking": bool(cfg["llm"].get("image_analyzer_enable_thinking", False))
                    }
                }
                ia_resp = llm.chat(
                    ia_messages,
                    max_tokens=cfg["llm"].get("max_tokens_image_analyzer", 6144),
                    temperature=cfg["llm"].get("image_analyzer_temperature",
                                                cfg["llm"].get("temperature", 0.0)),
                    extra_body=ia_extra,
                )
                if save_artifacts:
                    write_sample_artifact(run_dir, sid, "034_image_analyze_llm_raw.json", {
                        "content": ia_resp.content,
                        "reasoning": ia_resp.reasoning,
                        "finish_reason": ia_resp.finish_reason,
                        "usage": ia_resp.usage,
                    })
                image_analysis = parse_image_analyzer_output(ia_resp.content)
                result["image_analysis"] = image_analysis
                result["timings"]["image_analyzer_s"] = time.time() - t_ia
                if save_artifacts:
                    write_sample_artifact(run_dir, sid, "034_image_analyze_parsed.json", image_analysis)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sid=%s] image analyzer failed: %s", sid, exc)
            result["errors"].append({"stage": "image_analyzer", "error": str(exc)})
            image_analysis = None

    # ---------- 3.5) Pattern detector (optional) ----------
    pattern_result = None
    if cfg["pipeline"].get("use_pattern_detector", False):
        try:
            t_pd = time.time()
            pat_messages = build_pattern_messages(
                sample,
                parsed_atoms=parsed_atoms,
                retrieved=retrieved,
                search_hits=search_hits,
                prompts_dir=cfg["paths"]["prompts_dir"],
            )
            if save_artifacts:
                write_sample_artifact(run_dir, sid, "00_pattern_prompt.txt", pat_messages[0]["content"])
            pat_extra = {
                "chat_template_kwargs": {
                    "enable_thinking": bool(cfg["llm"].get("pattern_enable_thinking", False))
                }
            }
            pat_resp = llm.chat(
                pat_messages,
                max_tokens=cfg["llm"].get("max_tokens_pattern", 4096),
                temperature=cfg["llm"].get("pattern_temperature",
                                          cfg["llm"].get("temperature", 0.0)),
                extra_body=pat_extra,
            )
            if save_artifacts:
                write_sample_artifact(run_dir, sid, "035_pattern_llm_raw.json", {
                    "content": pat_resp.content,
                    "reasoning": pat_resp.reasoning,
                    "finish_reason": pat_resp.finish_reason,
                    "usage": pat_resp.usage,
                })
            pattern_result = parse_pattern_output(pat_resp.content)
            result["pattern_detector"] = pattern_result
            result["timings"]["pattern_s"] = time.time() - t_pd
            if save_artifacts:
                write_sample_artifact(run_dir, sid, "035_pattern_parsed.json", pattern_result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[sid=%s] pattern detector failed: %s", sid, exc)
            result["errors"].append({"stage": "pattern_detector", "error": str(exc)})
            pattern_result = None

    # ---------- 4) Final judge (vision-aware) ----------
    from .final_judge import build_judge_messages, parse_judge_output  # noqa: PLC0415
    try:
        t_j = time.time()
        judge_messages = build_judge_messages(
            sample,
            parsed_atoms=parsed_atoms,
            retrieved=retrieved,
            search_hits=search_hits,
            prompts_dir=cfg["paths"]["prompts_dir"],
            level1_doc_path=cfg["paths"]["level1_doc"],
            level2_doc_path=cfg["paths"]["level2_doc"],
            pattern_hint=pattern_result,
            image_analysis=image_analysis,
            text_analysis=text_analysis,
            image_max_side=image_max_side,
            max_images=max_images,
            model=cfg["llm"].get("model"),
            prompt_name=judge_prompt_name,
        )
        if save_artifacts:
            text_only = _content_to_text_only(judge_messages[0]["content"])
            write_sample_artifact(run_dir, sid, "00_judge_prompt.txt", text_only)
        judge_extra = {
            "chat_template_kwargs": {
                "enable_thinking": bool(cfg["llm"].get("judge_enable_thinking", True))
            }
        }
        judge_resp = llm.chat(
            judge_messages,
            max_tokens=cfg["llm"].get("max_tokens_judge", 32768),
            temperature=cfg["llm"].get("judge_temperature", cfg["llm"].get("temperature", 0.0)),
            extra_body=judge_extra,
        )
        if save_artifacts and cfg["pipeline"].get("save_judge_io", True):
            write_sample_artifact(run_dir, sid, "04_judge_llm_raw.json", {
                "content": judge_resp.content,
                "reasoning": judge_resp.reasoning,
                "finish_reason": judge_resp.finish_reason,
                "usage": judge_resp.usage,
            })
        # ===== JUDGE OUTPUT POST-PROCESSING POLICY =====
        # `pipeline.disable_l1_l2_coupling` (default False) — when True, the
        # LLM judge's L1 / L2 / L3 outputs are taken VERBATIM. NO code-side
        # rule overrides L1 based on L2 composition. Only the data-layer
        # canonical-label normalisation runs. This is the strictest
        # "LLM-only" mode required when the pipeline must not encode any
        # decision logic in code.
        disable_coupling = bool(pipe_cfg.get("disable_l1_l2_coupling", False))
        judge_parsed = parse_judge_output(
            judge_resp.content,
            apply_coupling=not disable_coupling,
            apply_fallback=not disable_coupling,
        )
        # Multi-LLM L2 SIGNAL AGGREGATION — controlled by
        # `pipeline.analyzer_union_policy`:
        #
        #   "off"        : never union. The judge is fully authoritative on
        #                  L1 / L2 / L3. ALL THREE LEVELS COME FROM THE LLM
        #                  JUDGE — no code-side rules touch them. Required
        #                  for the strict LLM-only mode.
        #   "always"     : (legacy) union analyzer active L2 into judge L2
        #                  unconditionally; reapplies coupling rules.
        #   "selective"  : (legacy) union analyzer L2 only when the judge
        #                  is already in non-True mode; reapplies coupling.
        #
        # When `disable_l1_l2_coupling` is True, we FORCE policy = "off" so
        # the LLM verdict is honoured verbatim regardless of legacy hints.
        from .final_judge import _enforce_l1_l2_coupling, _summarize_findings  # noqa: PLC0415
        from .labels import normalize_level2_list as _norm_l2  # noqa: PLC0415
        policy = pipe_cfg.get("analyzer_union_policy")
        if policy is None:
            legacy = pipe_cfg.get("use_analyzer_l2_union")
            if legacy is True:
                policy = "always"
            elif legacy is False:
                policy = "off"
            else:
                policy = "selective"
        policy = str(policy).lower()
        if disable_coupling:
            # Strict LLM-only mode — no analyzer union, no L1/L2 coupling.
            policy = "off"
        judge_l1 = judge_parsed.get("level1_verdict")
        judge_l2_pre_union = list(judge_parsed.get("level2_taxonomy") or [])
        text_align = (text_analysis or {}).get("alignment_level", "").upper()
        active_image = _norm_l2((image_analysis or {}).get("active_labels") or [])
        active_text = _norm_l2((text_analysis or {}).get("active_labels") or []) if (
            text_align in ("PARTIALLY_ALIGNED", "MISALIGNED")) else []

        do_union = False
        if policy == "always":
            do_union = True
        elif policy == "off":
            do_union = False
        elif policy == "selective":
            do_union = bool(judge_l2_pre_union) or (judge_l1 != "True")

        all_extras: list[str] = []
        if do_union:
            all_extras.extend(active_image)
            all_extras.extend(active_text)
        if all_extras:
            judge_l2 = list(judge_l2_pre_union)
            for l in all_extras:
                if l not in judge_l2:
                    judge_l2.append(l)
            judge_parsed["level2_taxonomy"] = judge_l2
            judge_parsed["level2_taxonomy_source"] = f"judge_union_analyzers_{policy}"
            if not disable_coupling:
                counts = _summarize_findings(judge_parsed.get("subclaim_findings"))
                new_l1, rule = _enforce_l1_l2_coupling(
                    judge_parsed["level1_verdict"], judge_l2, counts)
                if rule is not None:
                    judge_parsed["level1_verdict"] = new_l1
                    judge_parsed["level1_coupling_rule_applied"] = (
                        (judge_parsed.get("level1_coupling_rule_applied") or "") + ";" + rule
                    ).strip(";")
        else:
            if disable_coupling:
                judge_parsed["level2_taxonomy_source"] = "judge_only_llm_verbatim"
            else:
                judge_parsed["level2_taxonomy_source"] = f"judge_only_policy={policy}"
                counts = _summarize_findings(judge_parsed.get("subclaim_findings"))
                new_l1, rule = _enforce_l1_l2_coupling(
                    judge_l1, judge_l2_pre_union, counts)
                if rule is not None:
                    judge_parsed["level1_verdict"] = new_l1
                    judge_parsed["level1_coupling_rule_applied"] = (
                        (judge_parsed.get("level1_coupling_rule_applied") or "") + ";" + rule
                    ).strip(";")
        result["timings"]["judge_s"] = time.time() - t_j
        result["judge"] = judge_parsed
        result["judge_usage"] = judge_resp.usage
        if save_artifacts and cfg["pipeline"].get("save_judge_io", True):
            write_sample_artifact(run_dir, sid, "04_judge_parsed.json", judge_parsed)
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        msg = f"final judge failed: {exc!r}"
        logger.error("[sid=%s] %s\n%s", sid, msg, traceback.format_exc())
        result["errors"].append({"stage": "final_judge", "error": str(exc)})
        result["status"] = "final_judge_failed"
        result["judge"] = {
            "level1_verdict": None,
            "level2_taxonomy": [],
            "level3_rationale": "",
        }

    result["timings"]["total_s"] = time.time() - t0
    if save_artifacts:
        write_sample_artifact(run_dir, sid, "result.json", result)
    return result


def maybe_skip_sample(run_dir: Path, sample_id: str, resume: bool) -> bool:
    if not resume:
        return False
    return sample_done(run_dir, sample_id)
