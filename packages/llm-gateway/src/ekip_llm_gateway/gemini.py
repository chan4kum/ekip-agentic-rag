"""Google Gemini embeddings over REST (httpx; no vendor SDK).

Cost and latency design: identical texts in one call are embedded once, requests are batched
(<= 100 texts), batches run concurrently under a limit, transient failures are retried with
jittered exponential backoff (honouring Retry-After), and nothing is retried that would fail again.
Security: the key travels only in the `x-goog-api-key` header (never in a URL, log or error
message); error messages carry the HTTP status, never the provider's response body or the input.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx
from ekip_contracts.embedding import EmbeddingError, EmbeddingTask, EmbeddingUnavailableError

from ekip_llm_gateway.vectors import normalize, validate

_BASE_URL = "https://generativelanguage.googleapis.com"
_TASK = {
    EmbeddingTask.DOCUMENT: "RETRIEVAL_DOCUMENT",
    EmbeddingTask.QUERY: "RETRIEVAL_QUERY",
}
_RETRYABLE = {429, 500, 502, 503, 504}
MAX_BATCH = 100  # the API's documented per-request limit


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        client: httpx.AsyncClient | None = None,
        base_url: str = _BASE_URL,
        batch_size: int = MAX_BATCH,
        concurrency: int = 4,
        max_attempts: int = 4,
        timeout_s: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = lambda: random.uniform(0.5, 1.5),  # noqa: S311 - not security
    ) -> None:
        if not api_key:
            raise ValueError("a Gemini API key is required")
        if not 128 <= dimensions <= 3072:
            raise ValueError("dimensions must be between 128 and 3072")
        if not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH}")
        self._key = api_key
        self._model = model
        self._dims = dimensions
        self._client = client or httpx.AsyncClient(timeout=timeout_s, trust_env=True)
        self._owns_client = client is None
        self._url = f"{base_url.rstrip('/')}/v1beta/models/{model}:batchEmbedContents"
        self._batch = batch_size
        self._gate = asyncio.Semaphore(concurrency)
        self._attempts = max_attempts
        self._sleep = sleep
        self._jitter = jitter

    @property
    def model_version(self) -> str:
        return f"{self._model}@{self._dims}"

    @property
    def dimensions(self) -> int:
        return self._dims

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def embed(self, texts: Sequence[str], *, task: EmbeddingTask) -> list[tuple[float, ...]]:
        if any(not t.strip() for t in texts):
            raise EmbeddingError("cannot embed empty text")
        unique = list(dict.fromkeys(texts))  # identical texts are embedded once
        batches = [unique[i : i + self._batch] for i in range(0, len(unique), self._batch)]
        results = await asyncio.gather(*(self._embed_batch(b, task) for b in batches))
        by_text = {
            t: v
            for batch, vecs in zip(batches, results, strict=True)
            for t, v in zip(batch, vecs, strict=True)
        }
        return [by_text[t] for t in texts]

    async def _embed_batch(self, batch: list[str], task: EmbeddingTask) -> list[tuple[float, ...]]:
        body: dict[str, Any] = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": _TASK[task],
                    "outputDimensionality": self._dims,
                }
                for t in batch
            ]
        }
        async with self._gate:
            payload = await self._post_with_retries(body)
        try:
            vectors = [e["values"] for e in payload["embeddings"]]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("provider returned an unexpected response") from exc
        if len(vectors) != len(batch):
            raise EmbeddingError("provider returned the wrong number of vectors")
        out: list[tuple[float, ...]] = []
        for v in vectors:
            validate(v, self._dims)
            out.append(normalize(v))
        return out

    async def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        last = "unknown"
        for attempt in range(self._attempts):
            retry_after: float | None = None
            try:
                response = await self._client.post(
                    self._url, json=body, headers={"x-goog-api-key": self._key}
                )
            except (httpx.TimeoutException, httpx.TransportError):
                last = "network error"
            else:
                if response.status_code == 200:
                    try:
                        parsed: dict[str, Any] = response.json()
                    except ValueError as exc:
                        raise EmbeddingError("provider returned invalid JSON") from exc
                    return parsed
                if response.status_code not in _RETRYABLE:
                    raise EmbeddingError(
                        f"embedding request rejected (HTTP {response.status_code})"
                    )
                last = f"HTTP {response.status_code}"
                retry_after = _retry_after(response)
            if attempt + 1 < self._attempts:
                delay = (
                    retry_after if retry_after is not None else 0.5 * 2**attempt * self._jitter()
                )
                await self._sleep(min(delay, 30.0))
        raise EmbeddingUnavailableError(f"embedding provider unavailable ({last})")


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except ValueError:
        return None
