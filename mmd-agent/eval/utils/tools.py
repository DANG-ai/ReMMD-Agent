"""Stage-signal parsing, distortion-tag parsing, and aggregation.

A single agent run reads each stage output once and produces both the
5-class verdict and the 8-label distortion taxonomy in one pass:

* a 5-class final verdict (``True / Mostly True / Mixture / Mostly False / False``)
* a multi-label distortion prediction over the 8-label taxonomy:
    T1 Fabrication / T2 Distortion / T3 Misleading Context /
    V1 Synthetic Visual Content / V2 Visual Editing /
    C1 Semantic Inconsistency / C2 Contextual Inconsistency / C3 Pragmatic Inconsistency
"""

from __future__ import annotations

import re
from typing import Iterable


STAGE_SIGNAL_GROUPS: dict[str, list[str]] = {
    "text": [
        "TEXT_STRONG_SUPPORT",
        "TEXT_WEAK_SUPPORT",
        "TEXT_MIXED",
        "TEXT_WEAK_REFUTE",
        "TEXT_STRONG_REFUTE",
    ],
    "image": [
        "IMAGE_STRONG_SUPPORT",
        "IMAGE_WEAK_SUPPORT",
        "IMAGE_MIXED",
        "IMAGE_WEAK_REFUTE",
        "IMAGE_STRONG_REFUTE",
    ],
    "cross": [
        "CROSS_STRONG_SUPPORT",
        "CROSS_WEAK_SUPPORT",
        "CROSS_MIXED",
        "CROSS_WEAK_REFUTE",
        "CROSS_STRONG_REFUTE",
    ],
}

SIGNAL_SCORES: dict[str, int] = {sig: score
    for stage_signals in STAGE_SIGNAL_GROUPS.values()
    for sig, score in zip(stage_signals, [2, 1, 0, -1, -2])
}


VERDICT_LABEL_ORDER: list[str] = ["True", "Mostly True", "Mixture", "Mostly False", "False"]


STAGE_DISTORTION_CODES: dict[str, list[str]] = {
    "text": ["T1", "T2", "T3"],
    "image": ["V1", "V2"],
    "cross": ["C1", "C2", "C3"],
}

DISTORTION_CODE_TO_FULL_LABEL: dict[str, str] = {
    "T1": "T1 Fabrication",
    "T2": "T2 Distortion",
    "T3": "T3 Misleading Context",
    "V1": "V1 Synthetic Visual Content",
    "V2": "V2 Visual Editing",
    "C1": "C1 Semantic Inconsistency",
    "C2": "C2 Contextual Inconsistency",
    "C3": "C3 Pragmatic Inconsistency",
}

DISTORTION_LABEL_TO_CODE: dict[str, str] = {v: k for k, v in DISTORTION_CODE_TO_FULL_LABEL.items()}

FULL_DISTORTION_LABEL_ORDER: list[str] = [
    "T1 Fabrication",
    "T2 Distortion",
    "T3 Misleading Context",
    "V1 Synthetic Visual Content",
    "V2 Visual Editing",
    "C1 Semantic Inconsistency",
    "C2 Contextual Inconsistency",
    "C3 Pragmatic Inconsistency",
]


# -- Stage signal parsing -----------------------------------------------------


def parse_stage_signal(output: str, stage_name: str) -> str:
    """Find the model's chosen ``Finish[...]`` signal for the given stage."""
    if not output:
        return "UNKNOWN"
    valid_signals = STAGE_SIGNAL_GROUPS[stage_name]
    finish_pattern = re.compile(r"Finish\s*\[\s*([A-Z_]+)\s*\]", flags=re.IGNORECASE)

    candidates = [m.group(1).upper() for m in finish_pattern.finditer(output)]
    for sig in valid_signals:
        if sig in candidates:
            return sig

    # If model wrote `Finish[TEXT_STRONG_SUPPORT].` without brackets close to one
    # of our keywords, fall back to a substring scan.
    upper = output.upper()
    for sig in valid_signals:
        if sig in upper:
            return sig
    return "UNKNOWN"


# -- Distortion parsing -------------------------------------------------------


def parse_distortion_codes(output: str, stage_name: str) -> list[str]:
    """Extract the comma-separated distortion codes from the LAST ``Distortions: ...`` line.

    Only codes valid for the given stage are kept (e.g. only T1/T2/T3 for the
    text stage). If the line is missing or contains ``NONE`` / empty, returns
    an empty list.
    """
    if not output:
        return []
    valid = set(STAGE_DISTORTION_CODES[stage_name])

    distortion_lines = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[\*\-\>\s\`]*distortions?\s*[:：]", stripped, flags=re.IGNORECASE):
            distortion_lines.append(stripped)

    if not distortion_lines:
        match = re.search(
            r"distortions?\s*[:：]\s*([A-Za-z0-9,\s\[\]]+)",
            output,
            flags=re.IGNORECASE,
        )
        if not match:
            return []
        tail = match.group(1)
    else:
        line = distortion_lines[-1]
        tail = re.split(r"[:：]", line, maxsplit=1)[1] if ":" in line or "：" in line else line

    tail = tail.replace("[", " ").replace("]", " ").replace("`", " ").replace("*", " ")
    if tail.strip().upper() in {"NONE", "N/A", "NA", "EMPTY", "无", "无。"}:
        return []

    codes: list[str] = []
    for token in re.split(r"[,，;；\s]+", tail):
        token = token.strip().upper().rstrip(".")
        if not token:
            continue
        if token == "NONE":
            return []
        if token in valid and token not in codes:
            codes.append(token)
    return codes


# -- Aggregation: 5-class verdict ---------------------------------------------


def combine_stage_signals(text_signal: str, image_signal: str, cross_signal: str) -> tuple[str, dict]:
    """Reproduce the deterministic stage->verdict rule from the upstream code.

    Returns ``(verdict, rule_meta)`` where ``verdict`` is one of
    ``VERDICT_LABEL_ORDER`` and ``rule_meta`` records which branch fired.
    """
    signals = [text_signal, image_signal, cross_signal]
    if any(s == "UNKNOWN" for s in signals):
        known = [s for s in signals if s != "UNKNOWN"]
        if not known:
            return "Mixture", {
                "reason": "All stage outputs were unparsable.",
                "weighted_total": None,
            }
        known_scores = [SIGNAL_SCORES[s] for s in known]
        avg = sum(known_scores) / len(known_scores)
        if avg >= 1.0:
            return "Mostly True", {"reason": "Parsable stages lean supportive; unparsable stage(s) ignored.", "weighted_total": avg}
        if avg <= -1.0:
            return "Mostly False", {"reason": "Parsable stages lean refuting; unparsable stage(s) ignored.", "weighted_total": avg}
        return "Mixture", {"reason": "Parsable stages are inconclusive and unparsable stage(s) add uncertainty.", "weighted_total": avg}

    text_score = SIGNAL_SCORES[text_signal]
    image_score = SIGNAL_SCORES[image_signal]
    cross_score = SIGNAL_SCORES[cross_signal]
    weighted_total = 0.45 * text_score + 0.25 * image_score + 0.30 * cross_score

    if text_score == 2 and image_score >= 1 and cross_score >= 1:
        if image_score == 2 and cross_score == 2:
            return "True", {
                "reason": "All three stages strongly support the sample.",
                "weighted_total": weighted_total,
            }
        return "Mostly True", {
            "reason": "Text strongly supports; image and cross-modal also supportive.",
            "weighted_total": weighted_total,
        }

    if text_score >= 1 and image_score >= 0 and cross_score >= 0 and weighted_total >= 1.0:
        return "Mostly True", {
            "reason": "Text supports the claim and other stages do not refute it.",
            "weighted_total": weighted_total,
        }

    if text_score <= -2 and cross_score <= -1 and image_score <= -1:
        return "False", {
            "reason": "All stages refute the sample, with text strongly refuting.",
            "weighted_total": weighted_total,
        }

    if text_score <= -1 and image_score <= -1 and cross_score <= -1:
        return "Mostly False", {
            "reason": "All stages point toward refutation.",
            "weighted_total": weighted_total,
        }

    if text_score <= -1 and image_score <= 0 and cross_score <= 0 and weighted_total <= -0.7:
        return "Mostly False", {
            "reason": "Text refutes and other stages do not support the claim.",
            "weighted_total": weighted_total,
        }

    if weighted_total >= 1.5:
        return "True", {
            "reason": "Strong overall support across stages.",
            "weighted_total": weighted_total,
        }
    if weighted_total >= 0.6:
        return "Mostly True", {
            "reason": "Overall evidence leans supportive.",
            "weighted_total": weighted_total,
        }
    if weighted_total <= -1.5:
        return "False", {
            "reason": "Strong overall refutation across stages.",
            "weighted_total": weighted_total,
        }
    if weighted_total <= -0.6:
        return "Mostly False", {
            "reason": "Overall evidence leans toward refutation.",
            "weighted_total": weighted_total,
        }

    return "Mixture", {
        "reason": "Evidence is genuinely balanced between support and refutation.",
        "weighted_total": weighted_total,
    }


# -- Aggregation: 8-label multi-label distortion -----------------------------


def combine_distortion_predictions(
    text_codes: Iterable[str],
    image_codes: Iterable[str],
    cross_codes: Iterable[str],
) -> tuple[list[str], dict[str, int]]:
    """Union all per-stage distortion codes into the multi-label prediction.

    Returns
    -------
    full_label_list:
        Predicted distortion labels in canonical ``T1 Fabrication`` form,
        ordered as in ``FULL_DISTORTION_LABEL_ORDER`` for stable downstream
        comparison.
    binary_vector:
        Dict from each of the 8 canonical labels to 0/1, also in canonical
        order. Convenient for metric computation.
    """
    selected: set[str] = set()
    for codes, stage in (
        (text_codes, "text"),
        (image_codes, "image"),
        (cross_codes, "cross"),
    ):
        valid_for_stage = set(STAGE_DISTORTION_CODES[stage])
        for code in codes or []:
            code_up = str(code).strip().upper()
            if code_up in valid_for_stage:
                selected.add(code_up)

    full_labels = [DISTORTION_CODE_TO_FULL_LABEL[c]
                   for c in ["T1", "T2", "T3", "V1", "V2", "C1", "C2", "C3"] if c in selected]
    binary_vector = {label: int(label in full_labels) for label in FULL_DISTORTION_LABEL_ORDER}
    return full_labels, binary_vector


def gt_distortion_to_binary_vector(gt_labels: Iterable[str]) -> dict[str, int]:
    """Convert a ground-truth distortion_taxonomy list (canonical labels) to a binary dict."""
    gt_set = {str(label).strip() for label in gt_labels or []}
    return {label: int(label in gt_set) for label in FULL_DISTORTION_LABEL_ORDER}


# -- VQAEval (small literal-match utility, used by the original code) --------


class VQAEval:
    """Tiny utility class kept here for upstream-API compatibility.

    The original ``MMD-Agent`` codebase uses an ``evaluate(answer, gt_answers)``
    method that does a tokenized word-match. The new pipeline reads model
    outputs through structured parsers (``parse_stage_signal``,
    ``parse_distortion_codes``), so this class is rarely used at runtime, but
    we keep it so any custom user script that imports ``VQAEval`` still works.
    """

    PUNCT = list(";/\"[]{}()=+\\_-><@`,?!")

    def evaluate(self, answer: str, gt_answers) -> int:
        if not isinstance(answer, str):
            return 0
        answer = self._normalize(answer)
        if isinstance(gt_answers, list):
            for gt in gt_answers:
                if not gt:
                    continue
                if self._has_word(answer, self._normalize(gt)):
                    return 1
            return 0
        return int(self._has_word(answer, self._normalize(gt_answers)))

    def _normalize(self, s: str) -> str:
        s = s.replace("\n", " ").replace("\t", " ").strip()
        for p in self.PUNCT:
            s = s.replace(p, " " if p in s else "")
        return re.sub(r"\s+", " ", s).strip().lower()

    @staticmethod
    def _has_word(sentence: str, word: str) -> bool:
        if not word:
            return False
        pattern = r"\b" + re.escape(word.lower()) + r"\b"
        return re.search(pattern, sentence.lower()) is not None
