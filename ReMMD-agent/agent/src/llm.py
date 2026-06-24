"""OpenAI-compatible chat client.

Supports both qwen-style models (max_tokens + chat_template_kwargs) and
OpenAI GPT-5 / o-series reasoning models (max_completion_tokens, fixed
default temperature).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx


logger = logging.getLogger("remmd.llm")


# GPT-5 / o-series reasoning models reject `max_tokens` and any
# non-default `temperature` value, and they ignore qwen-only
# `chat_template_kwargs`. We auto-translate the request shape for these
# model families.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    # Strip optional vendor prefix like "openai/gpt-5.2"
    if "/" in m:
        m = m.split("/", 1)[1]
    return any(m.startswith(p) for p in _REASONING_MODEL_PREFIXES)


# Public alias used by other modules to pick a model-specific prompt branch.
# We share the same notion of "GPT-5/o-series" used for request-shape adaptation:
# all of these benefit from the GPT-tuned calibration prompts.
def is_gpt_family(model: str) -> bool:
    """True if the model uses OpenAI GPT-5 / o-series request semantics
    AND should load the `*_gpt.txt` prompt variant when present."""
    return _is_reasoning_model(model)


def select_prompt_filename(prompts_dir: str, base_name: str, model: str | None) -> str:
    """Pick the model-specific prompt file when available.

    For GPT-5 / o-series models we look for `<base_name>_gpt.txt` first and
    fall back to `<base_name>.txt`. For everything else (qwen etc.) we use
    `<base_name>.txt` unchanged.

    `base_name` should be without extension, e.g. "final_judge".
    Returns a relative filename (e.g. "final_judge_gpt.txt"), not an absolute
    path; callers join with `prompts_dir` themselves.
    """
    from pathlib import Path as _Path
    if model and is_gpt_family(model):
        gpt_name = f"{base_name}_gpt.txt"
        if (_Path(prompts_dir) / gpt_name).exists():
            return gpt_name
    return f"{base_name}.txt"


@dataclass
class LLMResponse:
    content: str
    reasoning: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None
    raw: dict[str, Any]


class _NonRetryableHTTPError(Exception):
    """4xx HTTP error (e.g. Azure content filter). Should not be retried."""

    def __init__(self, message: str, *, status_code: int, body: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body

    @property
    def is_content_filter(self) -> bool:
        return "content management policy" in self.body or "content_filter" in self.body


# public alias so other modules can `except llm.NonRetryableHTTPError`
NonRetryableHTTPError = _NonRetryableHTTPError


class LLMClient:
    """Synchronous OpenAI-compatible chat client.

    qwen3.5-9b is a *thinking* model: the response may contain both a
    `reasoning` field (the chain-of-thought) and a `content` field (the
    user-visible answer). We surface both; downstream callers should use
    `content` for parsing.

    For GPT-5 / o-series reasoning models we automatically:
      - rename `max_tokens` -> `max_completion_tokens`
      - drop `temperature` (the only allowed value is the default 1.0)
      - drop qwen-only `chat_template_kwargs`

    For qwen models we surface all of the OFFICIAL recommended sampling
    parameters (temperature, top_p, top_k, min_p, presence_penalty,
    repetition_penalty) and use the full set on every call. Defaults match
    Qwen team recommendations for thinking-mode general tasks:
        temperature=1.0  top_p=0.95  top_k=20  min_p=0.0
        presence_penalty=1.5  repetition_penalty=1.0
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        default_max_tokens: int = 4096,
        timeout: float = 600.0,
        retry_max_attempts: int = 6,
        retry_backoff_seconds: float = 5.0,
        use_proxy: bool = False,
        proxy_url: str | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.default_max_tokens = default_max_tokens
        self.timeout = timeout
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        # Qwen official sampling defaults — set when the caller didn't pass
        # an explicit value. None means "do not include in payload".
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        # `limits` is bumped well above the default (10 connections) so we can
        # run with high concurrency (100+ threads each issuing requests).
        # When use_proxy=True with an explicit proxy_url we plumb it through
        # httpx directly so the client always uses the configured proxy
        # regardless of the env. Otherwise we fall back to env-driven proxies.
        client_kwargs: dict[str, Any] = dict(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=512,
                max_keepalive_connections=256,
                keepalive_expiry=120.0,
            ),
        )
        if use_proxy and proxy_url:
            client_kwargs["proxy"] = proxy_url
            client_kwargs["trust_env"] = False
        else:
            client_kwargs["trust_env"] = use_proxy
        self._client = httpx.Client(**client_kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request with built-in retry.

        For qwen models we attach the FULL Qwen-team-recommended sampling
        config (top_p / top_k / min_p / presence_penalty /
        repetition_penalty) so that thinking-mode behaves correctly.
        For GPT-5 / o-series we drop sampling params per OpenAI's API.

        Raises the last exception if all retries fail.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        eff_max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        eff_temperature = self.temperature if temperature is None else temperature
        # Sampling parameter resolution: per-call override > client default > omit.
        eff_top_p = top_p if top_p is not None else self.top_p
        eff_top_k = top_k if top_k is not None else self.top_k
        eff_min_p = min_p if min_p is not None else self.min_p
        eff_presence_penalty = (
            presence_penalty if presence_penalty is not None else self.presence_penalty
        )
        eff_repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if _is_reasoning_model(self.model):
            # GPT-5 / o-series: only max_completion_tokens, no temperature override,
            # no qwen-style chat_template_kwargs / sampling params.
            payload["max_completion_tokens"] = eff_max_tokens
            if extra_body:
                cleaned = {k: v for k, v in extra_body.items()
                           if k != "chat_template_kwargs"}
                if cleaned:
                    payload.update(cleaned)
        else:
            payload["temperature"] = eff_temperature
            payload["max_tokens"] = eff_max_tokens
            # Standard OpenAI fields go top-level.
            if eff_top_p is not None:
                payload["top_p"] = eff_top_p
            if eff_presence_penalty is not None:
                payload["presence_penalty"] = eff_presence_penalty
            # vLLM-extension fields (top_k, min_p, repetition_penalty) — vLLM
            # accepts them at the top level on its OpenAI-compatible server,
            # so we put them there. If the server is strict-OpenAI it will
            # ignore them. If the server is vLLM in OpenAI mode it honours them.
            if eff_top_k is not None:
                payload["top_k"] = eff_top_k
            if eff_min_p is not None:
                payload["min_p"] = eff_min_p
            if eff_repetition_penalty is not None:
                payload["repetition_penalty"] = eff_repetition_penalty
            if extra_body:
                payload.update(extra_body)

        last_exc: Exception | None = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                resp = self._client.post(url, headers=headers, json=payload)
                # Always-retryable: 5xx and well-known transient 4xx
                # (408 timeout, 429 throttle, 401/403 transient gateway auth)
                if (resp.status_code >= 500
                        or resp.status_code in (408, 429, 401, 403)):
                    raise httpx.HTTPStatusError(
                        f"transient http {resp.status_code}: {resp.text[:300]}",
                        request=resp.request,
                        response=resp,
                    )
                if resp.status_code >= 400:
                    body = resp.text[:600]
                    # Content filter / safety reject is the only true non-retryable
                    # 4xx in our usage: same prompt/image will always be rejected.
                    is_content_filter = (
                        "content management policy" in body
                        or "content_filter" in body
                        or "content safety" in body
                    )
                    if is_content_filter:
                        raise _NonRetryableHTTPError(
                            f"http {resp.status_code}: {body}",
                            status_code=resp.status_code,
                            body=body,
                        )
                    # Other 4xx (e.g. 400 with a transient cause): retry too.
                    raise httpx.HTTPStatusError(
                        f"transient http {resp.status_code}: {body}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                if "choices" not in data or not data["choices"]:
                    raise RuntimeError(f"empty choices in response: {str(data)[:300]}")
                choice = data["choices"][0]
                msg = choice.get("message", {}) or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning")
                return LLMResponse(
                    content=content,
                    reasoning=reasoning,
                    finish_reason=choice.get("finish_reason"),
                    usage=data.get("usage"),
                    raw=data,
                )
            except _NonRetryableHTTPError as exc:
                # 4xx errors are not retryable (content filter, bad request).
                logger.warning(
                    "LLM non-retryable %s: %s",
                    exc.status_code,
                    exc.body[:200],
                )
                raise
            except Exception as exc:  # noqa: BLE001 — retry on anything else
                last_exc = exc
                logger.warning(
                    "LLM attempt %d/%d failed: %s",
                    attempt,
                    self.retry_max_attempts,
                    repr(exc)[:300],
                )
                if attempt < self.retry_max_attempts:
                    time.sleep(self.retry_backoff_seconds)
        assert last_exc is not None
        raise last_exc


def make_llm_from_config(cfg: dict[str, Any]) -> LLMClient:
    llm_cfg = cfg["llm"]
    if not llm_cfg.get("use_proxy", False):
        # belt-and-suspenders: also strip env for this process so any underlying
        # library can't accidentally pick up a proxy.
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                  "all_proxy", "ALL_PROXY"):
            os.environ.pop(k, None)
    return LLMClient(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg.get("api_key", ""),
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0.0),
        default_max_tokens=llm_cfg.get("max_tokens", 4096),
        timeout=llm_cfg.get("request_timeout", 600),
        retry_max_attempts=llm_cfg.get("retry_max_attempts", 6),
        retry_backoff_seconds=llm_cfg.get("retry_backoff_seconds", 5),
        use_proxy=llm_cfg.get("use_proxy", False),
        proxy_url=llm_cfg.get("proxy_url"),
        # Qwen-recommended sampling parameters (loaded from config so the
        # caller can override per-deployment).
        top_p=llm_cfg.get("top_p"),
        top_k=llm_cfg.get("top_k"),
        min_p=llm_cfg.get("min_p"),
        presence_penalty=llm_cfg.get("presence_penalty"),
        repetition_penalty=llm_cfg.get("repetition_penalty"),
    )


_re = re


def _escape_control_chars_in_strings(s: str) -> str:
    """Escape literal CR/LF/TAB occurring INSIDE JSON string literals.

    Many LLMs emit a real newline inside a `"..."` string, which is invalid
    JSON. We rewrite those newlines as `\\n` (still inside the same string).
    """
    out = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
            elif ch == "\\":
                out.append(ch)
                escape = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def _try_loads(s: str) -> Any:
    """Try several recovery passes for slightly-broken JSON from the LLM."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Recovery 1: replace invalid `\'` escapes (JSON has no single-quote escape)
    s2 = s.replace("\\'", "'")
    try:
        return json.loads(s2)
    except json.JSONDecodeError:
        pass
    # Recovery 2: strip trailing commas inside arrays / objects
    s3 = _re.sub(r",(\s*[\]}])", r"\1", s2)
    try:
        return json.loads(s3)
    except json.JSONDecodeError:
        pass
    # Recovery 3: collapse curly/smart quotes
    table = {
        ord("\u201c"): '"', ord("\u201d"): '"',
        ord("\u2018"): "'", ord("\u2019"): "'",
        ord("\u00ab"): '"', ord("\u00bb"): '"',
    }
    s4 = s3.translate(table)
    try:
        return json.loads(s4)
    except json.JSONDecodeError:
        pass
    # Recovery 4: escape literal newlines/tabs inside strings
    s5 = _escape_control_chars_in_strings(s4)
    try:
        return json.loads(s5)
    except json.JSONDecodeError:
        pass
    raise json.JSONDecodeError("recovery failed", s, 0)


def extract_json_block(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response.

    Recovery order:
      1. raw / direct parse with several escape-tolerance passes
      2. strip ```json``` code fences
      3. find first balanced `{...}` and try; if that fails after recoveries,
         find first balanced `[...]` and try.
    """
    if not text:
        raise ValueError("empty text")
    text = text.strip()
    # Strip code fences
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            inner = "\n".join(lines[1:])
            if inner.rstrip().endswith("```"):
                inner = inner.rstrip()[:-3]
            text = inner.strip()

    # Try direct (with recovery)
    try:
        return _try_loads(text)
    except json.JSONDecodeError:
        pass

    # Bracket counting: PREFER object over array (top-level object is the contract for our prompts)
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            end_pos = -1
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == open_char:
                        depth += 1
                    elif ch == close_char:
                        depth -= 1
                        if depth == 0:
                            end_pos = i
                            break
            if end_pos != -1:
                candidate = text[start:end_pos + 1]
                try:
                    return _try_loads(candidate)
                except json.JSONDecodeError:
                    pass
            start = text.find(open_char, start + 1)
    raise ValueError(f"could not extract JSON from text: {text[:400]!r}")
