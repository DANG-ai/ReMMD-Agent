"""External search tools.

Per spec, all api keys are intentionally empty: when a tool's key is empty
the tool is skipped and the pipeline falls back to memory-bank-only
retrieval. The HTTP code paths are implemented and tested for shape; when
keys are populated the tools become live.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


logger = logging.getLogger("remmd.search")


@dataclass
class SearchResult:
    source: str  # "google_serper" | "baidu" | "x"
    title: str = ""
    snippet: str = ""
    url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
    retry_max_attempts: int = 6,
    retry_backoff_seconds: float = 5.0,
    use_proxy: bool = True,
) -> httpx.Response:
    """Shared retry wrapper for outbound search calls.

    External search hits the public internet, so by default `use_proxy=True`
    (server needs `proxy_on`); this is the only outbound code path that
    requires the proxy.
    """
    last_exc: Exception | None = None
    with httpx.Client(timeout=timeout, trust_env=use_proxy) as client:
        for attempt in range(1, retry_max_attempts + 1):
            try:
                resp = client.request(
                    method, url,
                    headers=headers or {},
                    json=json_body,
                    params=params,
                )
                if resp.status_code >= 500 or resp.status_code in (408, 429):
                    raise httpx.HTTPStatusError(
                        f"transient http {resp.status_code}",
                        request=resp.request, response=resp,
                    )
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "search %s attempt %d/%d failed: %s",
                    url, attempt, retry_max_attempts, repr(exc)[:200],
                )
                if attempt < retry_max_attempts:
                    time.sleep(retry_backoff_seconds)
    assert last_exc is not None
    raise last_exc


# ----------------- Google via Serper API -----------------
def google_serper_search(
    query: str,
    *,
    api_key: str,
    endpoint: str = "https://google.serper.dev/search",
    n: int = 8,
    retry_max_attempts: int = 6,
    retry_backoff_seconds: float = 5.0,
    timeout: float = 30.0,
) -> list[SearchResult]:
    """Google Search via https://serper.dev/. Returns top-N organic + KG.

    When `api_key` is empty -> returns []. Code path is otherwise live.
    """
    if not api_key:
        logger.debug("google_serper: api_key empty -> skipped")
        return []
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    resp = _request_with_retry(
        "POST", endpoint,
        headers=headers,
        json_body={"q": query, "num": n},
        timeout=timeout,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    data = resp.json()
    out: list[SearchResult] = []
    for item in (data.get("organic") or [])[:n]:
        out.append(SearchResult(
            source="google_serper",
            title=item.get("title", ""),
            snippet=item.get("snippet", ""),
            url=item.get("link", ""),
            extra={"position": item.get("position"), "date": item.get("date")},
        ))
    # Optional knowledge graph entity
    kg = data.get("knowledgeGraph")
    if kg:
        out.append(SearchResult(
            source="google_serper",
            title=kg.get("title", ""),
            snippet=kg.get("description", ""),
            url=kg.get("website", ""),
            extra={"kind": "knowledge_graph"},
        ))
    return out


# ----------------- Baidu via 3rd-party API (placeholder) -----------------
def baidu_search(
    query: str,
    *,
    api_key: str,
    endpoint: str = "",
    n: int = 8,
    retry_max_attempts: int = 6,
    retry_backoff_seconds: float = 5.0,
    timeout: float = 30.0,
) -> list[SearchResult]:
    """Baidu web search via a 3rd-party API gateway.

    Many providers (RapidAPI, SerpAPI, custom proxies) expose Baidu search.
    Because there is no single canonical endpoint, the endpoint URL is also
    a config knob. We adopt a SerpAPI-style contract:

        GET {endpoint}?q=<query>&num=<n>&engine=baidu
        Header: Authorization: Bearer <api_key>
        Response.organic_results: [{title, snippet, link}]

    When either `api_key` or `endpoint` is empty -> returns [].
    """
    if not api_key or not endpoint:
        logger.debug("baidu: api_key or endpoint empty -> skipped")
        return []
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": query, "num": n, "engine": "baidu"}
    resp = _request_with_retry(
        "GET", endpoint,
        headers=headers, params=params,
        timeout=timeout,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    data = resp.json()
    out: list[SearchResult] = []
    for item in (data.get("organic_results") or data.get("results") or [])[:n]:
        out.append(SearchResult(
            source="baidu",
            title=item.get("title", ""),
            snippet=item.get("snippet", "") or item.get("abstract", ""),
            url=item.get("link", "") or item.get("url", ""),
            extra={k: item.get(k) for k in ("position", "date", "source") if k in item},
        ))
    return out


# ----------------- X (formerly Twitter) via API -----------------
def x_search(
    query: str,
    *,
    api_key: str,
    endpoint: str = "",
    n: int = 8,
    retry_max_attempts: int = 6,
    retry_backoff_seconds: float = 5.0,
    timeout: float = 30.0,
) -> list[SearchResult]:
    """X recent tweet search.

    Two compatible shapes are supported:

      1) Official X API v2 — if `endpoint` is empty we default to
         `https://api.twitter.com/2/tweets/search/recent`. Then `api_key`
         is treated as a Bearer token, and the response shape is
         {data: [{id, text, author_id, created_at}]}.

      2) A 3rd-party API (e.g. RapidAPI) — if `endpoint` is non-empty it is
         called directly with the same query/num contract, expecting
         {results: [...]} or {tweets: [...]}.

    When `api_key` is empty -> returns [].
    """
    if not api_key:
        logger.debug("x: api_key empty -> skipped")
        return []
    if not endpoint:
        endpoint = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "query": query,
            "max_results": min(max(n, 10), 100),  # X v2 requires 10..100
            "tweet.fields": "created_at,author_id,public_metrics,lang",
        }
        resp = _request_with_retry(
            "GET", endpoint,
            headers=headers, params=params,
            timeout=timeout,
            retry_max_attempts=retry_max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        data = resp.json()
        out: list[SearchResult] = []
        for item in (data.get("data") or [])[:n]:
            out.append(SearchResult(
                source="x",
                title="",
                snippet=item.get("text", ""),
                url=f"https://x.com/i/web/status/{item.get('id')}" if item.get("id") else "",
                extra={k: item.get(k) for k in ("created_at", "author_id", "public_metrics", "lang")},
            ))
        return out
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": query, "num": n}
    resp = _request_with_retry(
        "GET", endpoint,
        headers=headers, params=params,
        timeout=timeout,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    data = resp.json()
    out: list[SearchResult] = []
    for item in (data.get("results") or data.get("tweets") or [])[:n]:
        out.append(SearchResult(
            source="x",
            title=item.get("user", {}).get("screen_name", ""),
            snippet=item.get("text", "") or item.get("content", ""),
            url=item.get("url", ""),
            extra={k: item.get(k) for k in ("created_at", "favorite_count", "retweet_count") if k in item},
        ))
    return out


def run_all_searches(
    queries: list[str],
    *,
    cfg: dict[str, Any],
) -> dict[str, list[SearchResult]]:
    """Convenience wrapper; runs each enabled tool over each query and
    aggregates. Skips tools whose api_key is empty.

    Returns: {tool_name: [results...]}
    """
    out: dict[str, list[SearchResult]] = {"google_serper": [], "baidu": [], "x": []}
    if not queries:
        return out
    st = cfg.get("search_tools", {}) or {}

    g = st.get("google_serper", {}) or {}
    if g.get("api_key"):
        for q in queries:
            try:
                out["google_serper"].extend(
                    google_serper_search(
                        q, api_key=g["api_key"],
                        endpoint=g.get("endpoint", "https://google.serper.dev/search"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("google_serper failed on q=%r: %s", q, exc)

    b = st.get("baidu", {}) or {}
    if b.get("api_key") and b.get("endpoint"):
        for q in queries:
            try:
                out["baidu"].extend(
                    baidu_search(q, api_key=b["api_key"], endpoint=b["endpoint"])
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("baidu failed on q=%r: %s", q, exc)

    x = st.get("x", {}) or {}
    if x.get("api_key"):
        for q in queries:
            try:
                out["x"].extend(
                    x_search(q, api_key=x["api_key"], endpoint=x.get("endpoint", ""))
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("x failed on q=%r: %s", q, exc)
    return out
