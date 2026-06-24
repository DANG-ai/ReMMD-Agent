"""OpenAI-compatible LLM client with caching, retry, and detailed logging.

The client targets the standard ``/v1/chat/completions`` interface so it works
for the GPT gateway as well as Qwen-style deployments. Every call:

1. Is dedup-cached on a sha256 of the request payload, so identical prompts
   produce identical observations (important for the MCTS expansion).
2. Is retried up to ``config.retry.attempts`` times with ``sleep_seconds``
   delay between attempts, switching between primary and backup base URLs.
3. Emits a JSONL log entry recording the prompt, the response, the elapsed
   time, and the cache hit / attempt count via :class:`CallLogger`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from .config import RuntimeConfig
from .logging_utils import CallLogger, TimedCall, get_default_logger, truncate_text


def _encode_image_as_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_type = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".webp":
        mime_type = "image/webp"
    elif suffix == ".gif":
        mime_type = "image/gif"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class LLMClient:
    """OpenAI-compatible client with caching, multi-URL failover and JSONL call logging."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        logger: CallLogger | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = config.paths.artifacts_root / "cache" / "llm" / config.api.provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._urls = [config.api.primary_base_url, *config.api.backup_base_urls]
        self._clients = [self._build_client(url) for url in self._urls]
        self._logger = logger if logger is not None else get_default_logger()

    def _build_client(self, base_url: str) -> OpenAI:
        http_client_kwargs: dict[str, Any] = {
            "timeout": float(self.config.api.timeout_seconds),
            "trust_env": False,
        }
        if self.config.api.proxy_url:
            http_client_kwargs["proxy"] = self.config.api.proxy_url
        http_client = httpx.Client(**http_client_kwargs)
        return OpenAI(
            base_url=base_url,
            api_key=self.config.api.api_key,
            http_client=http_client,
        )

    def _cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
        expect_json: bool,
    ) -> str:
        payload = {
            "provider": self.config.api.provider,
            "model": self.config.api.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_paths": [str(path) for path in image_paths or []],
            "expect_json": expect_json,
            "temperature": self.config.api.temperature,
            "max_output_tokens": self.config.api.max_output_tokens,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        cache_path = self._cache_path(cache_key)
        if not cache_path.exists():
            return None
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_cache(self, cache_key: str, payload: dict[str, Any]) -> None:
        with self._cache_path(cache_key).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_as_data_url(image_path)},
                }
            )
        messages.append({"role": "user", "content": content})
        return messages

    def _logger_or_none(self) -> CallLogger | None:
        return self._logger if self._logger is not None else get_default_logger()

    def _log_extra(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None,
        expect_json: bool,
        cache_key: str,
        purpose: str,
    ) -> dict[str, Any]:
        prompt_chars = self.config.logging.log_prompt_chars
        return {
            "provider": self.config.api.provider,
            "model": self.config.api.model,
            "purpose": purpose,
            "cache_key": cache_key,
            "expect_json": expect_json,
            "image_count": len(image_paths or []),
            "image_paths": [str(p) for p in image_paths or []],
            "system_prompt": truncate_text(system_prompt, prompt_chars),
            "user_prompt": truncate_text(user_prompt, prompt_chars),
        }

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path] | None = None,
        expect_json: bool = False,
        use_cache: bool = True,
        *,
        purpose: str = "agent",
    ) -> str:
        """Send a chat-completion request and return the text response.

        Parameters mirror the legacy code:
        * ``system_prompt`` / ``user_prompt`` : prompt strings.
        * ``image_paths`` : optional list of image files to attach.
        * ``expect_json`` : if True, asks the API for ``response_format`` JSON.
        * ``use_cache`` : if True, reads/writes the local cache.
        * ``purpose`` : free-form tag stored in the call log (e.g. "planner").
        """

        cache_key = self._cache_key(system_prompt, user_prompt, image_paths, expect_json)
        logger = self._logger_or_none()

        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                if logger is not None and self.config.logging.log_llm_calls:
                    with TimedCall(
                        "llm_call",
                        logger,
                        extra={
                            **self._log_extra(
                                system_prompt,
                                user_prompt,
                                image_paths,
                                expect_json,
                                cache_key,
                                purpose,
                            ),
                            "cached": True,
                            "attempt": 0,
                            "url": cached.get("url"),
                        },
                    ) as call:
                        call.set_result(
                            {
                                "response": truncate_text(
                                    str(cached.get("content", "")),
                                    self.config.logging.log_prompt_chars,
                                )
                            }
                        )
                return str(cached["content"])

        messages = self._build_messages(system_prompt, user_prompt, image_paths)

        last_error: Exception | None = None
        for attempt in range(self.config.retry.attempts):
            for client_index, (client, url) in enumerate(zip(self._clients, self._urls)):
                with TimedCall(
                    "llm_call",
                    logger if self.config.logging.log_llm_calls else None,
                    extra={
                        **self._log_extra(
                            system_prompt,
                            user_prompt,
                            image_paths,
                            expect_json,
                            cache_key,
                            purpose,
                        ),
                        "cached": False,
                        "attempt": attempt + 1,
                        "client_index": client_index,
                        "url": url,
                    },
                ) as call:
                    try:
                        kwargs: dict[str, Any] = {
                            "model": self.config.api.model,
                            "messages": messages,
                        }
                        if expect_json:
                            kwargs["response_format"] = {"type": "json_object"}
                        if self.config.api.temperature is not None:
                            kwargs["temperature"] = self.config.api.temperature
                        if self.config.api.max_output_tokens is not None:
                            kwargs["max_tokens"] = self.config.api.max_output_tokens
                        response = client.chat.completions.create(**kwargs)
                        content_text = response.choices[0].message.content or ""
                        if use_cache:
                            self._write_cache(
                                cache_key,
                                {
                                    "content": content_text,
                                    "attempt": attempt + 1,
                                    "client_index": client_index,
                                    "url": url,
                                    "model": self.config.api.model,
                                    "provider": self.config.api.provider,
                                },
                            )
                        call.set_result(
                            {
                                "response": truncate_text(
                                    content_text, self.config.logging.log_prompt_chars
                                )
                            }
                        )
                        return content_text
                    except Exception as error:  # noqa: BLE001
                        last_error = error
                        call.set_error(repr(error))
            if attempt + 1 < self.config.retry.attempts:
                time.sleep(self.config.retry.sleep_seconds)

        raise RuntimeError(
            f"LLM request failed after {self.config.retry.attempts} attempts"
        ) from last_error


def extract_json_object(text: str) -> dict[str, Any]:
    """Locate the first JSON object inside ``text`` and parse it."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response")
        return json.loads(text[start : end + 1])
