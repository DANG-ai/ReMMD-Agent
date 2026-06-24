"""Benchmark loader + image-encoding helpers for ReMMDBench."""
from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("remmd.data")


@dataclass
class BenchSample:
    sample_id: str
    text: str
    language_code: str
    region_code: str
    theme_category: str
    text_length_tier: str
    image_files: list[str] = field(default_factory=list)  # absolute paths
    image_names: list[str] = field(default_factory=list)  # filenames
    # gold (may be missing)
    gold_verdict: str | None = None
    gold_taxonomy: list[str] = field(default_factory=list)
    gold_rationale: str | None = None
    # for traceability
    sample_dir: str = ""


def _list_sample_ids(bench_root: str) -> list[str]:
    root = Path(bench_root)
    ids = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    return ids


def load_sample(bench_root: str, sample_id: str) -> BenchSample:
    sdir = Path(bench_root) / sample_id
    sample_path = sdir / "sample.json"
    ann_path = sdir / "annotation.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    image_names = s.get("images", []) or []
    image_files = []
    img_dir = sdir / "images"
    for name in image_names:
        p = img_dir / name
        if p.exists():
            image_files.append(str(p))
    bs = BenchSample(
        sample_id=sample_id,
        text=s.get("text", ""),
        language_code=s.get("language_code", ""),
        region_code=s.get("region_code", ""),
        theme_category=s.get("theme_category", ""),
        text_length_tier=s.get("text_length_tier", ""),
        image_files=image_files,
        image_names=image_names,
        sample_dir=str(sdir),
    )
    if ann_path.exists():
        with open(ann_path, "r", encoding="utf-8") as f:
            a = json.load(f)
        bs.gold_verdict = a.get("verdict")
        bs.gold_taxonomy = a.get("distortion_taxonomy", []) or []
        bs.gold_rationale = a.get("rationale")
    return bs


def iter_samples(bench_root: str, sample_ids: Iterable[str] | None = None) -> Iterable[BenchSample]:
    ids = list(sample_ids) if sample_ids is not None else _list_sample_ids(bench_root)
    for sid in ids:
        yield load_sample(bench_root, sid)


def list_sample_ids(bench_root: str) -> list[str]:
    return _list_sample_ids(bench_root)


def file_to_data_url(path: str) -> str:
    """Encode an image file as a data URL (raw bytes, no resize)."""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def encode_image_resized(
    path: str,
    *,
    max_side: int = 768,
    jpeg_quality: int = 88,
) -> str:
    """Resize an image to keep its longest side <= max_side, then encode as JPEG data URL.

    Vision token cost in qwen3.5-9b scales with image area; resizing keeps
    per-call cost predictable. Always re-encodes as JPEG (drops alpha).
    """
    from PIL import Image  # local import — keep module-import cheap
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest > max_side:
                f = max_side / float(longest)
                im = im.resize((max(1, int(round(w * f))),
                                max(1, int(round(h * f)))),
                               Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            data = buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.warning("encode_image_resized(%s) failed: %s — falling back to raw bytes", path, exc)
        return file_to_data_url(path)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def prepare_image_content_blocks(
    image_files: list[str],
    *,
    max_side: int = 768,
    max_images: int | None = 6,
    jpeg_quality: int = 88,
) -> list[dict[str, Any]]:
    """Convert local image paths into OpenAI-style multimodal `image_url` blocks.

    - Resizes each image so longest side <= `max_side` (keeps prompt cost bounded).
    - Caps the number of images at `max_images` (None = no cap).
    - Returns blocks ready to splice into a `messages[*]["content"]` array.

    The caller is responsible for prefixing/suffixing text blocks.
    """
    if not image_files:
        return []
    files = list(image_files)
    if max_images is not None and len(files) > max_images:
        files = files[:max_images]
    blocks: list[dict[str, Any]] = []
    for p in files:
        try:
            url = encode_image_resized(p, max_side=max_side, jpeg_quality=jpeg_quality)
        except Exception as exc:  # noqa: BLE001 — never block a sample on one bad image
            logger.warning("skipping image (encode failed): %s — %s", p, exc)
            continue
        blocks.append({
            "type": "image_url",
            "image_url": {"url": url},
        })
    return blocks


def select_image_files_for_call(
    sample: "BenchSample",
    *,
    max_images: int | None = 6,
) -> list[str]:
    """Pick up to `max_images` image paths from the sample (preserving order).

    The benchmark filenames follow `01_img_1.jpg`, `02_img_2.jpg`, ... so order
    on disk is meaningful (lead image first). We take the first N.
    """
    if not sample.image_files:
        return []
    if max_images is None:
        return list(sample.image_files)
    return list(sample.image_files[:max_images])
