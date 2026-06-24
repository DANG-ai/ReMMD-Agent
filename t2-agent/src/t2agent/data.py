"""ReMMDBench dataset loader.

Each sample lives in its own numbered directory under ``realmmdbench_root``:

    <root>/<id>/annotation.json   # verdict + distortion_taxonomy + rationale
    <root>/<id>/sample.json       # text + images + metadata
    <root>/<id>/images/<image_files>

The loader produces :class:`BenchmarkSample` records that always carry **both**
ground-truth labels (verdict and taxonomy) so a single unified agent can be
evaluated against both tasks in one run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PathConfig
from .labels import normalize_taxonomy_labels, normalize_verdict_label


@dataclass(slots=True)
class BenchmarkSample:
    """Single ReMMDBench sample with dual ground-truth labels."""

    benchmark_name: str
    sample_id: str
    sample_dir: Path
    text: str
    image_paths: list[Path]
    verdict: str
    """Canonical 5-class verdict ground truth, e.g. ``Mostly False``."""
    taxonomy_labels: list[str]
    """Canonical 8-class multi-label taxonomy ground truth."""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> str:
        """Backwards-compatible alias used by the legacy code that expects a single string label."""

        return self.verdict


@dataclass(slots=True)
class BenchmarkDataset:
    name: str
    samples: list[BenchmarkSample]

    def categories(self) -> list[str]:
        return sorted({sample.verdict for sample in self.samples})

    def select_min_per_category(self, minimum: int = 1) -> list[BenchmarkSample]:
        """Return the first ``minimum`` samples per verdict category, deduplicated."""

        if minimum <= 0:
            return list(self.samples)
        grouped: dict[str, list[BenchmarkSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.verdict, []).append(sample)
        selected: list[BenchmarkSample] = []
        seen: set[str] = set()
        for category in sorted(grouped):
            for sample in grouped[category][:minimum]:
                if sample.sample_id in seen:
                    continue
                selected.append(sample)
                seen.add(sample.sample_id)
        return selected


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_images(sample_dir: Path, image_names: list[Any]) -> list[Path]:
    """Resolve image file paths from the sample.json ``images`` list.

    The ReMMDBench layout puts image files in ``<sample_dir>/images/<filename>``;
    if an explicit path is present in the JSON we use it as-is, otherwise we
    join it with the ``images`` subdirectory.
    """

    image_dir = sample_dir / "images"
    resolved: list[Path] = []
    for raw in image_names:
        if not raw:
            continue
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            in_subdir = image_dir / candidate
            in_dir = sample_dir / candidate
            if in_subdir.exists():
                candidate = in_subdir
            elif in_dir.exists():
                candidate = in_dir
            else:
                candidate = in_subdir
        if candidate.exists():
            resolved.append(candidate.resolve())

    if not resolved and image_dir.exists():
        resolved = sorted(
            p.resolve()
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )

    return resolved


def _build_sample(sample_dir: Path) -> BenchmarkSample | None:
    annotation_path = sample_dir / "annotation.json"
    sample_path = sample_dir / "sample.json"
    if not annotation_path.exists() or not sample_path.exists():
        return None

    annotation = _read_json(annotation_path)
    sample_payload = _read_json(sample_path)

    text = str(sample_payload.get("text", "")).strip()
    raw_images = sample_payload.get("images") or sample_payload.get("image_paths") or []
    if not isinstance(raw_images, list):
        raw_images = [raw_images]
    image_paths = _resolve_images(sample_dir, raw_images)
    if not image_paths:
        return None

    verdict_raw = annotation.get("verdict")
    verdict = normalize_verdict_label(verdict_raw)
    if verdict is None:
        verdict = str(verdict_raw or "Mixture")
    taxonomy_labels = normalize_taxonomy_labels(annotation.get("distortion_taxonomy", []))

    metadata = {
        "verdict": verdict,
        "distortion_taxonomy": taxonomy_labels,
        "rationale": annotation.get("rationale"),
        "language_code": sample_payload.get("language_code"),
        "region_code": sample_payload.get("region_code"),
        "theme_category": sample_payload.get("theme_category"),
        "text_length_tier": sample_payload.get("text_length_tier"),
        "source_type": sample_payload.get("source_type"),
        "core_claim": sample_payload.get("core_claim"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    return BenchmarkSample(
        benchmark_name="RealMMDBench",
        sample_id=sample_dir.name,
        sample_dir=sample_dir.resolve(),
        text=text,
        image_paths=image_paths,
        verdict=verdict,
        taxonomy_labels=taxonomy_labels,
        metadata=metadata,
    )


def load_realmmdbench(paths: PathConfig) -> BenchmarkDataset:
    """Load every sample directory under ``paths.realmmdbench_root``."""

    root = paths.realmmdbench_root
    if not root.exists():
        raise FileNotFoundError(f"RealMMDBench root not found: {root}")

    samples: list[BenchmarkSample] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        sample = _build_sample(child)
        if sample is not None:
            samples.append(sample)

    if not samples:
        raise RuntimeError(
            f"No usable samples found under {root}; "
            "expected each sample directory to contain annotation.json + sample.json + images/."
        )

    return BenchmarkDataset(name="RealMMDBench", samples=samples)
