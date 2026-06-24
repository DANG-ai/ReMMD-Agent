"""Dataset loader for ReMMDBench.

ReMMDBench layout (one folder per sample, identified by zero-padded index):

    <BENCH_ROOT>/
        001/
            sample.json          # {"text": ..., "images": [...], ...}
            annotation.json      # {"verdict": "...", "distortion_taxonomy": [...]}
            images/
                01_img_1.jpg
                02_img_2.jpg
                ...
        002/
            ...

The dataset class below is intentionally framework-agnostic (no torch / no
DataLoader) so the same code runs on a stripped-down conda env and on the
server. It exposes a simple iterable + ``__getitem__`` so existing code that
calls ``len(dataset)`` and ``dataset[i]`` keeps working.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_truncate(text: Any, max_chars: int = 1200) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def ensure_ends_with_period(text: str) -> str:
    stripped = text.rstrip()
    if not stripped.endswith(("." , "!", "?", "。", "！", "？")):
        return stripped + "."
    return text


def _collect_sample_images(sample_dir: Path, sample: dict[str, Any], max_images: int) -> list[str]:
    """Return absolute paths of images for this sample, in document order.

    Priority:
      1. Files listed in ``sample.json``'s ``images`` array (joined with
         ``sample_dir / 'images'``).
      2. Files listed in ``materialized_images.json`` (legacy MMFakeBench format).
      3. Every image-like file found under ``sample_dir / 'images'``.
    """
    image_paths: list[str] = []
    image_dir = sample_dir / "images"

    listed = sample.get("images") or []
    if isinstance(listed, list):
        for name in listed:
            if not isinstance(name, str):
                continue
            candidate = image_dir / name
            if not candidate.exists():
                candidate = sample_dir / name
            if candidate.exists():
                image_paths.append(str(candidate.resolve()))

    if not image_paths:
        materialized = sample_dir / "materialized_images.json"
        if materialized.exists():
            try:
                for item in _read_json(materialized):
                    path_str = item.get("path") if isinstance(item, dict) else None
                    if not path_str:
                        continue
                    candidate = Path(path_str)
                    if not candidate.exists():
                        candidate = image_dir / candidate.name
                    if candidate.exists():
                        image_paths.append(str(candidate.resolve()))
            except Exception:
                pass

    if not image_paths and image_dir.exists():
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                image_paths.append(str(path.resolve()))

    deduped: list[str] = []
    seen: set[str] = set()
    for p in image_paths:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
        if len(deduped) >= max_images:
            break
    return deduped


def _build_external_evidence(sample: dict[str, Any]) -> str:
    """Re-use the MMFakeBench-style evidence formatting if present in sample.json.

    ReMMDBench samples do not currently carry ``evidence`` / ``rumor_entry``
    fields, but we keep this here for forward compatibility.
    """
    parts: list[str] = []

    source_records = sample.get("source_records") or []
    if source_records:
        lines = []
        for idx, record in enumerate(source_records[:3], start=1):
            screen_name = record.get("screen_name", "") if isinstance(record, dict) else ""
            text = _safe_truncate(record.get("text", "") if isinstance(record, dict) else "", 300)
            lines.append(f"{idx}. {screen_name}: {text}")
        parts.append("Source-side records:\n" + "\n".join(lines))

    evidence_items = sample.get("evidence") or []
    if evidence_items:
        lines = []
        for idx, item in enumerate(evidence_items[:5], start=1):
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "Untitled").strip()
            snippet = _safe_truncate(item.get("snippet", ""), 220)
            url = (item.get("url") or "").strip()
            lines.append(f"{idx}. {title}\nSnippet: {snippet}\nURL: {url}")
        parts.append("External evidence candidates:\n" + "\n".join(lines))

    rumor_entry = sample.get("rumor_entry")
    if isinstance(rumor_entry, dict):
        title = rumor_entry.get("article_title") or rumor_entry.get("section_title") or ""
        detail = rumor_entry.get("detail") or ""
        if title or detail:
            parts.append(
                "Fact-check style reference:\n"
                f"Title: {title}\n"
                f"Detail: {_safe_truncate(detail, 500)}"
            )

    if not parts:
        return "No structured external evidence candidates were provided."
    return "\n\n".join(parts)


class ReMMDBench_Dataset:
    """In-memory list of ReMMDBench samples, ready to be iterated.

    Parameters
    ----------
    root: str or Path
        Absolute path to the ReMMDBench root that contains numbered sample
        folders.
    prompt_root: str or Path
        Absolute or relative path to the prompt template directory that holds
        ``textual_veracity_check.txt`` etc.
    max_samples: optional int
        If provided, only ``max_samples`` samples are kept (with deterministic
        shuffling using ``seed``).
    seed: int
        Random seed used for sub-sampling.
    sample_filter: str
        If non-empty, only samples whose directory path contains this substring
        are kept.
    max_images: int
        Cap on the number of images sent per sample.
    """

    def __init__(
        self,
        root: str | Path,
        prompt_root: str | Path,
        max_samples: int | None = None,
        seed: int = 42,
        sample_filter: str = "",
        max_images: int = 10,
    ) -> None:
        self.root = Path(root).resolve()
        self.prompt_root = Path(prompt_root).resolve()
        self.max_images = max_images

        if not self.root.exists():
            raise FileNotFoundError(f"ReMMDBench root not found: {self.root}")

        self.template_text_judge = _read_text(self.prompt_root / "textual_veracity_check.txt")
        self.template_image_judge = _read_text(self.prompt_root / "visual_veracity_check.txt")
        self.template_consistency_judge = _read_text(self.prompt_root / "cross_modal_consistency_reason.txt")

        sample_paths = sorted(self.root.rglob("sample.json"))
        if sample_filter:
            sample_paths = [p for p in sample_paths if sample_filter in str(p.parent)]

        if max_samples is not None and len(sample_paths) > max_samples:
            rng = random.Random(seed)
            shuffled = sample_paths[:]
            rng.shuffle(shuffled)
            sample_paths = sorted(shuffled[:max_samples])

        self.dataset: list[dict[str, Any]] = []
        for sample_path in sample_paths:
            sample_dir = sample_path.parent
            try:
                sample = _read_json(sample_path)
            except Exception as exc:
                print(f"[WARN] Skipping unreadable sample {sample_path}: {exc}", flush=True)
                continue

            annotation_path = sample_dir / "annotation.json"
            annotation = _read_json(annotation_path) if annotation_path.exists() else {}

            text = ensure_ends_with_period(str(sample.get("text", "")).strip())
            image_paths = _collect_sample_images(sample_dir, sample, max_images)
            provided_evidence = _build_external_evidence(sample)

            item = {
                "sample_name": sample_dir.name,
                "sample_dir": str(sample_dir.resolve()),
                "text": text,
                "image_paths": image_paths,
                "image_caption_context": "",
                "provided_evidence": provided_evidence,
                "language_code": sample.get("language_code", ""),
                "region_code": sample.get("region_code", ""),
                "theme_category": sample.get("theme_category", ""),
                "text_length_tier": sample.get("text_length_tier", ""),
                "verdict_gt": annotation.get("verdict", ""),
                "distortion_taxonomy_gt": annotation.get("distortion_taxonomy", []) or [],
                "rationale_gt": annotation.get("rationale", ""),
            }

            item["question_fix_text_check"] = (
                self.template_text_judge
                .replace("[News caption]", text)
                .replace("[Provided Evidence]", provided_evidence)
            )
            item["question_fix_image_check"] = (
                self.template_image_judge
                .replace("[News caption]", text)
                .replace("[Provided Evidence]", provided_evidence)
            )
            item["question_fix_consistency_reason"] = (
                self.template_consistency_judge
                .replace("[News caption]", text)
                .replace("[Provided Evidence]", provided_evidence)
            )
            self.dataset.append(item)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)
