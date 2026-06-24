"""Serper-based key-entity search.

This module replaces the previous ``wiki_search.py``: instead of scraping
Wikipedia, we issue a single Serper (Google) search per key entity and
collapse the top-K organic snippets into a short knowledge string. The same
``(entity, knowledge)`` tuple shape is returned, so callers don't have to
change.

Each model invocation gets a single Serper key from ``serper_api.txt``. The
caller (``run_mmd_agent.py``) loads the list, selects the index for the
current model, and passes the resolved key into ``search_key_entity``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx


SERPER_URL = "https://google.serper.dev/search"
SERPER_MAX_RETRIES = 3
SERPER_RETRY_INTERVAL = 3
SERPER_TIMEOUT = 20.0


def load_serper_keys(serper_key_file: str | Path) -> list[str]:
    path = Path(serper_key_file)
    if not path.exists():
        raise FileNotFoundError(f"Serper API key file not found: {path}")
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys.append(line)
    if not keys:
        raise ValueError(f"No Serper API keys found in {path}")
    return keys


def select_serper_key(serper_key_file: str | Path, key_index: int) -> str:
    """Select the ``key_index``-th (1-based) Serper key from ``serper_key_file``."""
    keys = load_serper_keys(serper_key_file)
    if key_index < 1 or key_index > len(keys):
        raise IndexError(
            f"Serper key index {key_index} out of range (1..{len(keys)}) for file {serper_key_file}"
        )
    return keys[key_index - 1]


def _clean_entity(raw: str) -> str:
    """Trim the model's raw 'Finish: ...' payload into a usable query string."""
    if not raw:
        return ""
    raw = raw.strip()
    if raw.lower() == "none":
        return ""
    raw = raw.split("\n")[0]
    raw = raw.replace("[", " ").replace("]", " ")
    raw = re.sub(r"^[`*\"'\s]+|[`*\"'\s\.]+$", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return ""
    if len(raw.split()) > 10:
        raw = " ".join(raw.split()[:10])
    return raw


def _summarize_serper_response(payload: dict) -> str:
    """Collapse a Serper /search payload into a short knowledge paragraph."""
    parts: list[str] = []

    answer_box = payload.get("answerBox") or {}
    if isinstance(answer_box, dict):
        for key in ("answer", "snippet", "snippetHighlighted", "title"):
            value = answer_box.get(key)
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            if value:
                parts.append(str(value).strip())
                break

    kg = payload.get("knowledgeGraph") or {}
    if isinstance(kg, dict):
        kg_parts = []
        title = kg.get("title")
        descr = kg.get("description") or kg.get("descriptionSource")
        if title and descr:
            kg_parts.append(f"{title}: {descr}")
        elif descr:
            kg_parts.append(str(descr))
        attributes = kg.get("attributes") or {}
        if isinstance(attributes, dict):
            for k, v in list(attributes.items())[:3]:
                kg_parts.append(f"{k}: {v}")
        if kg_parts:
            parts.append(" ".join(kg_parts))

    organic = payload.get("organic") or []
    for item in organic[:3]:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if title and snippet:
            parts.append(f"{title} - {snippet}")
        elif snippet:
            parts.append(snippet)
        elif title:
            parts.append(title)

    knowledge = " ".join(parts).strip()
    if len(knowledge) > 1500:
        knowledge = knowledge[:1500] + "..."
    return knowledge


def serper_search(query: str, api_key: str, *, gl: str = "us", hl: str = "en") -> tuple[str, dict]:
    """Issue a single Serper /search call. Returns (knowledge_text, raw_payload).

    Serper 是公网服务 (google.serper.dev)，在内网服务器上必须走 HTTP(S) 代理。
    本函数始终以 ``trust_env=True`` 创建客户端，从而读取 ``HTTPS_PROXY`` /
    ``HTTP_PROXY`` 等环境变量。运行前请确保 terminal 已执行 ``proxy_on``。
    """
    if not query:
        return "", {}
    if not api_key:
        return "", {"error": "missing-serper-api-key"}

    body = {"q": query, "gl": gl, "hl": hl, "num": 5}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    last_error: Exception | None = None
    for attempt in range(SERPER_MAX_RETRIES):
        try:
            with httpx.Client(timeout=SERPER_TIMEOUT, trust_env=True) as client:
                resp = client.post(SERPER_URL, headers=headers, json=body)
            resp.raise_for_status()
            try:
                payload = resp.json()
            except json.JSONDecodeError:
                payload = {"raw_text": resp.text}
            return _summarize_serper_response(payload), payload
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            if attempt < SERPER_MAX_RETRIES - 1:
                time.sleep(SERPER_RETRY_INTERVAL)
                continue
    return "", {"error": str(last_error)}


def search_key_entity(model_output: str, api_key: str) -> tuple[str, str, dict]:
    """Extract a key entity from the model output and return (entity, knowledge, raw_payload).

    The original code split on ``Finish:`` followed by an entity name. We do
    the same here, but also accept a few alternate forms.
    """
    if not model_output:
        return "", "", {}

    text = model_output
    if "Thought 2" in text:
        text = text.split("Thought 2")[0]

    entity = ""
    for marker in ("Finish:", "FINISH:", "finish:", "Finish[", "FINISH["):
        if marker in text:
            tail = text.split(marker, 1)[1]
            if marker.endswith("["):
                if "]" in tail:
                    tail = tail.split("]", 1)[0]
            else:
                tail = tail.split("\n", 1)[0]
                tail = tail.split(".", 1)[0]
            entity = _clean_entity(tail)
            if entity:
                break

    if not entity:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[-3:]:
            entity = _clean_entity(line)
            if entity and entity.lower() not in {"none", "n/a", "na"}:
                break

    if not entity:
        return "", "", {}

    knowledge, raw = serper_search(entity, api_key)
    return entity, knowledge, raw
