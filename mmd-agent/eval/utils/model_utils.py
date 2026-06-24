"""OpenAI-compatible chat-completions client.

We deliberately use the plain ``/chat/completions`` endpoint so the same code
works against:

* OpenAI / OpenAI-style proxies (e.g. gpt-5.2 at http://YOUR_GPT_ENDPOINT/v1)
* vLLM / SGLang servers (used to serve qwen3.6-27b, qwen3.5-9b, qwen3.5-4b)
* Any internal gateway that mimics OpenAI's REST schema

The function ``call_chat_engine_multi_image`` mirrors the call site used by
the original ``call_gpt_engine_df_multi_image`` so the loop in ``vqa.py``
stays close to the upstream MMD-Agent code.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx


def _image_path_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix == "jpg" else suffix
    if not mime:
        mime = "jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def _normalize_message_content(content: Any) -> str:
    """OpenAI-style responses can return content as either string or list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    out.append(part["text"])
                elif "text" in part and isinstance(part["text"], str):
                    out.append(part["text"])
                elif part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out).strip()
    if content is None:
        return ""
    return str(content)


def call_chat_engine_multi_image(args, prompt: str, image_paths: list[str]) -> tuple[str, dict]:
    """POST a single chat-completions request and return (text, raw_response_dict).

    ``args`` must expose the attributes:
      - args.api_key
      - args.base_url           (e.g. http://YOUR_GPT_ENDPOINT/v1)
      - args.model_name
      - args.system_prompt
      - args.temperature
      - args.max_new_tokens
      - args.request_timeout
      - args.retry_times
      - args.retry_interval
      - args.image_detail       ('low' | 'high' | 'auto')
      - args.trust_env          (bool, optional)
    """
    image_paths = image_paths or []
    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        try:
            url = _image_path_to_data_url(image_path)
        except FileNotFoundError:
            continue
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": getattr(args, "image_detail", "low") or "low",
                },
            }
        )

    messages: list[dict[str, Any]] = []
    system_prompt = (getattr(args, "system_prompt", "") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    payload: dict[str, Any] = {
        "model": args.model_name,
        "messages": messages,
        "temperature": float(getattr(args, "temperature", 0.0)),
    }
    extra_body = getattr(args, "extra_body", None)
    if isinstance(extra_body, dict) and extra_body:
        payload.update(extra_body)

    base_url_raw = args.base_url
    base_url = "".join(ch for ch in base_url_raw if ch not in ("\r", "\n", "\t")).strip()
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/chat/completions"):
        url = f"{base_url}/chat/completions"
    else:
        url = base_url

    api_key_raw = args.api_key or ""
    api_key_clean = "".join(ch for ch in api_key_raw if ch not in ("\r", "\n", "\t")).strip()
    model_name_clean = "".join(ch for ch in (args.model_name or "") if ch not in ("\r", "\n", "\t")).strip()
    payload["model"] = model_name_clean

    headers = {
        "Authorization": f"Bearer {api_key_clean}",
        "Content-Type": "application/json",
    }

    max_retries = max(1, int(getattr(args, "retry_times", 5)))
    retry_interval = float(getattr(args, "retry_interval", 6))
    timeout = float(getattr(args, "request_timeout", 180))
    trust_env = bool(getattr(args, "trust_env", False))

    payload_size_kb = len(json.dumps(payload)) / 1024.0

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout, trust_env=trust_env) as client:
                response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Non-JSON response from {url}: {response.text[:300]}"
                ) from exc

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Empty choices in response: {json.dumps(data)[:300]}")
            message = choices[0].get("message") or {}
            text = _normalize_message_content(message.get("content"))
            return text.strip(), data
        except (httpx.HTTPError, RuntimeError, OSError) as exc:
            last_error = exc
            # 退避策略：基础间隔 * (1 + attempt*0.5)，避免 502 等服务端临时拒绝时雪崩。
            wait = retry_interval * (1 + attempt * 0.5)
            if attempt < max_retries - 1:
                print(
                    f"  [Retry {attempt+1}/{max_retries}] {type(exc).__name__}: {str(exc)[:160]}. "
                    f"model={getattr(args,'model_name','?')} url={url[:80]} payload={payload_size_kb:.0f}KB "
                    f"Waiting {wait:.0f}s...",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise last_error
    raise last_error if last_error else RuntimeError("call_chat_engine_multi_image: unreachable")
