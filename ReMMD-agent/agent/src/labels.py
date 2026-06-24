"""Canonical L1 / L2 label sets and prior-document loaders."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# Single source of truth — must match annotation.json field "verdict"
LEVEL1_LABELS: list[str] = ["True", "Mostly True", "Mixture", "Mostly False", "False"]

# Single source of truth — must match annotation.json field "distortion_taxonomy"
LEVEL2_LABELS: list[str] = [
    "T1 Fabrication",
    "T2 Distortion",
    "T3 Misleading Context",
    "V1 Synthetic Visual Content",
    "V2 Visual Editing",
    "C1 Semantic Inconsistency",
    "C2 Contextual Inconsistency",
    "C3 Pragmatic Inconsistency",
]


@lru_cache(maxsize=4)
def load_level1_doc(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


@lru_cache(maxsize=4)
def load_level2_doc(path: str) -> str:
    """Load `二级标签.docx` (Word) and return as plain text."""
    import docx  # python-docx

    doc = docx.Document(path)
    parts: list[str] = []
    for p in doc.paragraphs:
        t = p.text.rstrip()
        if t:
            parts.append(t)
    return "\n".join(parts)


# ----------------- canonicalization -----------------
_L2_ALIASES: dict[str, str] = {
    # short codes
    "t1": "T1 Fabrication",
    "t2": "T2 Distortion",
    "t3": "T3 Misleading Context",
    "v1": "V1 Synthetic Visual Content",
    "v2": "V2 Visual Editing",
    "c1": "C1 Semantic Inconsistency",
    "c2": "C2 Contextual Inconsistency",
    "c3": "C3 Pragmatic Inconsistency",
}
_L2_FULL_LOOKUP: dict[str, str] = {label.lower(): label for label in LEVEL2_LABELS}
_L1_LOOKUP: dict[str, str] = {label.lower(): label for label in LEVEL1_LABELS}


def normalize_level1(value: str | None) -> str | None:
    """Return canonical L1 label, or None if not recognizable."""
    if not value:
        return None
    v = str(value).strip()
    v_low = v.lower()
    if v_low in _L1_LOOKUP:
        return _L1_LOOKUP[v_low]
    # tolerate punctuation/spacing variants
    v_clean = re.sub(r"[^a-z]", "", v_low)
    for k, full in _L1_LOOKUP.items():
        if re.sub(r"[^a-z]", "", k) == v_clean:
            return full
    return None


def normalize_level2_label(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip()
    v_low = v.lower()
    if v_low in _L2_FULL_LOOKUP:
        return _L2_FULL_LOOKUP[v_low]
    # match prefix code like "T1", "V2" anywhere at start
    m = re.match(r"^\s*([tvc][123])\b", v_low)
    if m:
        return _L2_ALIASES[m.group(1)]
    # try fuzzy: starts with code then space
    for code, full in _L2_ALIASES.items():
        if v_low.startswith(code):
            return full
    # match if substring appears (e.g. "Visual Editing")
    for full in LEVEL2_LABELS:
        if full.lower() in v_low or v_low in full.lower():
            return full
    return None


def normalize_level2_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        norm = normalize_level2_label(v)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out
