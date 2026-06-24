"""OpenAI-compatible embedding client for qwen3-embedding-8b (4096-d)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
import numpy as np


logger = logging.getLogger("remmd.embedder")


class EmbeddingClient:
    """Synchronous embedding client with batching + retry."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        max_input_chars: int = 32000,
        batch_size: int = 8,
        timeout: float = 600.0,
        retry_max_attempts: int = 6,
        retry_backoff_seconds: float = 5.0,
        use_proxy: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_input_chars = max_input_chars
        self.batch_size = max(1, int(batch_size))
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = httpx.Client(
            timeout=timeout,
            trust_env=use_proxy,
            limits=httpx.Limits(
                max_connections=512,
                max_keepalive_connections=256,
                keepalive_expiry=120.0,
            ),
        )
        self._dim: int | None = None

    @property
    def dim(self) -> int | None:
        return self._dim

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EmbeddingClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_input_chars:
            return text[: self.max_input_chars]
        return text

    def _post_batch(self, batch: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": batch}
        last_exc: Exception | None = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                resp = self._client.post(url, headers=headers, json=payload)
                if resp.status_code >= 500 or resp.status_code in (408, 429):
                    raise httpx.HTTPStatusError(
                        f"transient http {resp.status_code}: {resp.text[:200]}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                items = data.get("data") or []
                vecs = [it["embedding"] for it in items]
                if not vecs:
                    raise RuntimeError(f"empty embeddings batch: {str(data)[:300]}")
                if self._dim is None:
                    self._dim = len(vecs[0])
                return vecs
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "embed attempt %d/%d failed (batch_size=%d): %s",
                    attempt,
                    self.retry_max_attempts,
                    len(batch),
                    repr(exc)[:300],
                )
                if attempt < self.retry_max_attempts:
                    time.sleep(self.retry_backoff_seconds)
        assert last_exc is not None
        raise last_exc

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed many strings -> (N, dim) float32 L2-normalized array."""
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        clean = [self._truncate(t if t and t.strip() else " ") for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(clean), self.batch_size):
            batch = clean[i : i + self.batch_size]
            out.extend(self._post_batch(batch))
        arr = np.asarray(out, dtype=np.float32)
        # L2 normalize for cosine similarity via inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def make_embedder_from_config(cfg: dict[str, Any]) -> EmbeddingClient:
    emb_cfg = cfg["embedding"]
    if not emb_cfg.get("use_proxy", False):
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                  "all_proxy", "ALL_PROXY"):
            os.environ.pop(k, None)
    return EmbeddingClient(
        base_url=emb_cfg["base_url"],
        api_key=emb_cfg.get("api_key", ""),
        model=emb_cfg["model"],
        max_input_chars=emb_cfg.get("max_input_chars", 32000),
        batch_size=emb_cfg.get("batch_size", 8),
        timeout=emb_cfg.get("request_timeout", 600),
        retry_max_attempts=emb_cfg.get("retry_max_attempts", 6),
        retry_backoff_seconds=emb_cfg.get("retry_backoff_seconds", 5),
        use_proxy=emb_cfg.get("use_proxy", False),
    )
