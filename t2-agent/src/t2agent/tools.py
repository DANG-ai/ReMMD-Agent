"""T2-Agent action tools.

The toolset mirrors the paper's action space, but the only external search
backend used here is the **Serper API**. The Wikipedia and Serper endpoints
share the same retry / logging path so every external call is auditable.

Tools available:
* ``Wikipedia[query]``        : Wikipedia REST API (en / zh).
* ``Google[query]``           : Serper Google search proxy.
* ``VQA[question]``           : LVLM-backed image question answering.
* ``Entity[image]``           : LVLM-backed entity extraction.
* ``Detect[image]``           : LVLM-backed image veracity detector.
* ``Counterfactual[image]``   : LVLM-backed counterfactual analysis.
* ``ImageUnderstanding[image]``: LVLM-backed image description.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Semaphore
from typing import Any
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from .config import RuntimeConfig, resolve_serper_key
from .llm import LLMClient
from .logging_utils import CallLogger, TimedCall, get_default_logger, truncate_text


_WEB_SEARCH_SEMAPHORE = Semaphore(3)


@dataclass(slots=True)
class ToolCall:
    tool_name: str
    tool_input: str


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    tool_input: str
    observation: str


class ToolBox:
    """All external tools the agent can invoke, plus a tool-result cache."""

    def __init__(
        self,
        config: RuntimeConfig,
        llm: LLMClient,
        *,
        logger: CallLogger | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self._logger = logger if logger is not None else get_default_logger()
        self.default_headers = {
            "User-Agent": "T2Agent-Unified/1.0 (academic; contact unavailable)",
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        }
        self.http_client = httpx.Client(
            timeout=float(config.api.timeout_seconds),
            headers=self.default_headers,
            follow_redirects=True,
            trust_env=True,
        )
        self.cache_dir = config.paths.artifacts_root / "cache" / "tools"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._serper_key = resolve_serper_key(config)

    def _logger_or_none(self) -> CallLogger | None:
        return self._logger if self._logger is not None else get_default_logger()

    def _cache_key(self, tool_call: ToolCall, text: str, image_paths: list[Path]) -> str:
        payload = {
            "tool": tool_call.tool_name.lower(),
            "input": tool_call.tool_input,
            "sample_text_sha": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "images": [str(p) for p in image_paths],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _request_text(self, method: str, url: str, **kwargs: Any) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.retry.attempts):
            try:
                headers = dict(self.default_headers)
                headers.update(kwargs.pop("headers", {}))
                response = self.http_client.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                return response.text
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt + 1 < self.config.retry.attempts:
                    time.sleep(self.config.retry.sleep_seconds)
        raise RuntimeError(f"HTTP request failed: {url}") from last_error

    def run(self, tool_call: ToolCall, text: str, image_paths: list[Path]) -> ToolResult:
        cache_key = self._cache_key(tool_call, text, image_paths)
        cache_path = self.cache_dir / f"{cache_key}.json"
        logger = self._logger_or_none()

        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if logger is not None and self.config.logging.log_tool_calls:
                with TimedCall(
                    "tool_call",
                    logger,
                    extra={
                        "tool": tool_call.tool_name,
                        "tool_input": tool_call.tool_input,
                        "cached": True,
                        "cache_key": cache_key,
                    },
                ) as call:
                    call.set_result(
                        {
                            "observation": truncate_text(
                                cached.get("observation", ""),
                                self.config.logging.log_prompt_chars,
                            )
                        }
                    )
            return ToolResult(
                tool_name=cached.get("tool", tool_call.tool_name),
                tool_input=cached.get("input", tool_call.tool_input),
                observation=cached["observation"],
            )

        tool_name = tool_call.tool_name.lower()
        with TimedCall(
            "tool_call",
            logger if self.config.logging.log_tool_calls else None,
            extra={
                "tool": tool_call.tool_name,
                "tool_input": tool_call.tool_input,
                "cached": False,
                "cache_key": cache_key,
            },
        ) as call:
            try:
                if tool_name == "wikipedia":
                    observation = self.wikipedia(tool_call.tool_input)
                elif tool_name in {"google", "web_search"}:
                    observation = self.web_search(tool_call.tool_input)
                elif tool_name == "vqa":
                    observation = self.vqa(image_paths[0], tool_call.tool_input)
                elif tool_name == "entity":
                    observation = self.entity(image_paths[0])
                elif tool_name == "detect":
                    observation = self.detect(image_paths[0], text)
                elif tool_name == "image_understanding":
                    observation = self.image_understanding(image_paths[0])
                elif tool_name == "counterfactual":
                    observation = self.counterfactual(image_paths[0], text)
                else:
                    observation = (
                        f"Tool '{tool_call.tool_name}' is not available. "
                        "Use one of the listed available actions."
                    )
            except Exception as error:  # noqa: BLE001
                call.set_error(repr(error))
                observation = f"Tool execution failed: {error}"

            call.set_result(
                {
                    "observation": truncate_text(
                        observation, self.config.logging.log_prompt_chars
                    )
                }
            )

        result = ToolResult(
            tool_name=tool_call.tool_name,
            tool_input=tool_call.tool_input,
            observation=observation,
        )
        cache_path.write_text(
            json.dumps(
                {
                    "tool": result.tool_name,
                    "input": result.tool_input,
                    "observation": result.observation,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def wikipedia(self, query: str) -> str:
        with _WEB_SEARCH_SEMAPHORE:
            api_url = (
                "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch="
            )
            if re.search(r"[\u4e00-\u9fff]", query):
                api_url = (
                    "https://zh.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch="
                )
            try:
                search_payload = json.loads(
                    self._request_text(
                        "GET",
                        api_url + quote_plus(query),
                        headers={"Accept": "application/json"},
                    )
                )
            except Exception:  # noqa: BLE001
                return self._serper_search(f"{query} site:wikipedia.org") or "No Wikipedia entry found."

            search_results = search_payload.get("query", {}).get("search", [])
            if not search_results:
                return "No Wikipedia entry found."
            title = search_results[0]["title"]
            lang_prefix = "en" if "en.wikipedia.org" in api_url else "zh"
            summary_url = (
                f"https://{lang_prefix}.wikipedia.org/api/rest_v1/page/summary/"
                f"{quote_plus(title)}"
            )
            try:
                summary_payload = json.loads(
                    self._request_text(
                        "GET", summary_url, headers={"Accept": "application/json"}
                    )
                )
            except Exception:  # noqa: BLE001
                suggestions = ", ".join(item["title"] for item in search_results[:5])
                return f"No summary available. Similar entities: {suggestions}"
            extract = summary_payload.get("extract")
            if extract:
                return extract
            suggestions = ", ".join(item["title"] for item in search_results[:5])
            return f"No exact entry. Similar entities: {suggestions}"

    def _serper_search(self, query: str) -> str | None:
        try:
            response = self.http_client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self._serper_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={"q": query, "num": self.config.serper.num_results},
                timeout=float(self.config.serper.timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:  # noqa: BLE001
            return None

        snippets: list[str] = []
        for item in payload.get("organic", [])[: self.config.serper.num_results]:
            title = str(item.get("title", "Untitled")).strip()
            link = str(item.get("link", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            line = f"{title}: {snippet}".strip()
            if link:
                line = f"{line} ({link})"
            snippets.append(line)

        answer_box = payload.get("answerBox") or {}
        answer_snippet = str(answer_box.get("answer") or answer_box.get("snippet") or "").strip()
        if answer_snippet:
            snippets.insert(0, f"[Answer Box] {answer_snippet}")

        if not snippets:
            return None
        return "\n".join(snippets)

    def web_search(self, query: str) -> str:
        with _WEB_SEARCH_SEMAPHORE:
            result = self._serper_search(query)
            if result:
                return result
            return "Serper search returned no results."

    def vqa(self, image_path: Path, question: str) -> str:
        system_prompt = (
            "You are an image question answering tool for misinformation verification. "
            "Answer only from visible image evidence. Be concise and factual."
        )
        return self.llm.complete(system_prompt, question, [image_path], purpose="tool.vqa")

    def entity(self, image_path: Path) -> str:
        system_prompt = (
            "You are an image entity recognition tool. Identify notable people, landmarks, "
            "logos, organizations, locations, and other salient entities visible in the image."
        )
        user_prompt = "List the visible entities in the image with brief certainty notes."
        return self.llm.complete(system_prompt, user_prompt, [image_path], purpose="tool.entity")

    def detect(self, image_path: Path, text: str) -> str:
        system_prompt = (
            "You are an image veracity detector for multimodal misinformation. "
            "Assess whether the image itself shows digital manipulation, synthetic content, "
            "counterfactual elements, or visual inconsistencies."
        )
        user_prompt = (
            "Inspect the image only. Mention any signs of visual editing, synthetic generation, "
            "counterfactual content, or reasons the image appears authentic.\n"
            f"Related text for context: {text}"
        )
        return self.llm.complete(system_prompt, user_prompt, [image_path], purpose="tool.detect")

    def image_understanding(self, image_path: Path) -> str:
        system_prompt = (
            "You are an image understanding tool for misinformation verification. "
            "Describe the key objects, people, actions, setting, and any visible text."
        )
        user_prompt = "Describe the image in a factual paragraph."
        return self.llm.complete(
            system_prompt, user_prompt, [image_path], purpose="tool.image_understanding"
        )

    def counterfactual(self, image_path: Path, text: str) -> str:
        system_prompt = (
            "You are a counterfactual detection tool. Determine whether the image depicts "
            "something physically implausible, logically impossible, or strongly indicative "
            "of synthetic generation."
        )
        user_prompt = (
            "Analyze whether the image contains counterfactual or physically implausible content.\n"
            f"Related text for context: {text}"
        )
        return self.llm.complete(
            system_prompt, user_prompt, [image_path], purpose="tool.counterfactual"
        )


def available_actions_for_subtask(
    subtask: str, selected_tools: list[str], image_paths: list[Path]
) -> list[str]:
    """Return the textual action menu the planner is allowed to choose from."""

    del image_paths  # signature parity with the legacy code
    selected = set(selected_tools)
    if subtask == "text":
        actions = []
        if "wikipedia" in selected:
            actions.append("Wikipedia[query]")
        if "web_search" in selected or "google" in selected:
            actions.append("Google[query]")
        actions.append("Finish[TEXT_SUPPORT or TEXT_REFUTE]")
        return actions
    if subtask == "image":
        actions = []
        if "counterfactual" in selected or "image_understanding" in selected:
            actions.append("Detect[image]")
        actions.append("Finish[IMAGE_SUPPORT or IMAGE_REFUTE]")
        return actions
    if subtask == "match":
        actions = []
        if "image_understanding" in selected:
            actions.append("VQA[question]")
        if "entity" in selected:
            actions.append("Entity[image]")
        actions.append("Finish[MATCH or MISMATCH]")
        return actions
    raise ValueError(f"Unsupported subtask: {subtask}")


# BeautifulSoup is kept importable for future fallbacks even though the
# Serper-only policy means we no longer scrape HTML directly.
_ = BeautifulSoup
