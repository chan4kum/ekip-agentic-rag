"""Hosted cross-encoder reranking over HTTP (Cohere / Jina / Voyage-style `POST /rerank`).

Request:  {"model", "query", "documents": [str], "top_n"}
Response: {"results": [{"index": int, "relevance_score": float}]}   (Voyage names the list "data")

A cross-encoder reads the query and each passage TOGETHER, so it judges relevance far better than
embedding distance, at the price of one model call per search and latency that grows with the
number of candidates. We cap candidates and passage length, and fall back to the fused order on
any failure (the retriever turns RerankError into a degraded-but-correct result).

Data handling: this sends passage text the caller is already allowed to see to a third party, so it
is OFF by default and enabled per deployment. The key travels only in the Authorization header;
errors carry the HTTP status, never the response body or any passage text.
"""

import asyncio
import math
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import httpx
from ekip_contracts.retrieval import RerankError, RetrievedChunk

_RETRYABLE = {429, 500, 502, 503, 504}


class HttpReranker:
    def __init__(
        self,
        api_key: str,
        *,
        url: str,
        model: str,
        client: httpx.AsyncClient | None = None,
        max_candidates: int = 100,
        max_chars: int = 2000,
        max_attempts: int = 2,
        timeout_s: float = 10.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = lambda: random.uniform(0.5, 1.5),  # noqa: S311 - not security
    ) -> None:
        if not api_key:
            raise ValueError("a rerank API key is required")
        if not url.startswith("https://"):
            raise ValueError("the rerank endpoint must use https")
        self._key = api_key
        self._url = url
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=timeout_s, trust_env=True)
        self._owns_client = client is None
        self._max_candidates = max_candidates
        self._max_chars = max_chars
        self._attempts = max_attempts
        self._sleep = sleep
        self._jitter = jitter

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, limit: int
    ) -> list[RetrievedChunk]:
        pool = list(candidates[: self._max_candidates])
        if not pool:
            return []
        body = {
            "model": self._model,
            "query": query,
            "documents": [c.text[: self._max_chars] for c in pool],
            "top_n": min(limit, len(pool)),
        }
        payload = await self._post(body)
        results = payload.get("results", payload.get("data"))
        if not isinstance(results, list):
            raise RerankError("reranker returned an unexpected response")
        scored: list[tuple[float, int]] = []
        seen: set[int] = set()
        for item in results:
            try:
                index, score = item["index"], float(item["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RerankError("reranker returned an unexpected response") from exc
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(pool)
                or index in seen
                or not math.isfinite(score)
            ):
                raise RerankError("reranker returned an invalid result")
            seen.add(index)
            scored.append((score, index))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [pool[i].model_copy(update={"score": max(s, 0.0)}) for s, i in scored[:limit]]

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last = "unknown"
        for attempt in range(self._attempts):
            try:
                response = await self._client.post(
                    self._url, json=body, headers={"Authorization": f"Bearer {self._key}"}
                )
            except (httpx.TimeoutException, httpx.TransportError):
                last = "network error"
            else:
                if response.status_code == 200:
                    try:
                        parsed: dict[str, Any] = response.json()
                    except ValueError as exc:
                        raise RerankError("reranker returned invalid JSON") from exc
                    if not isinstance(parsed, dict):
                        raise RerankError("reranker returned an unexpected response")
                    return parsed
                if response.status_code not in _RETRYABLE:
                    raise RerankError(f"rerank request rejected (HTTP {response.status_code})")
                last = f"HTTP {response.status_code}"
            if attempt + 1 < self._attempts:
                await self._sleep(0.5 * 2**attempt * self._jitter())
        raise RerankError(f"reranker unavailable ({last})")
