"""Chat model over any OpenAI-compatible endpoint: `POST {base_url}/chat/completions`.

OpenAI, Groq, DeepSeek, Azure OpenAI (v1 route) and a self-hosted vLLM all speak this shape, so one
small adapter covers them; the provider is just `base_url` + `model` + key from settings.

Rules this adapter keeps:
- Errors are `LLMError` carrying only the HTTP status or a fixed phrase. Never the response body
  (it can echo the prompt, which holds document text) and never the key.
- The key travels only in the Authorization header.
- Retries only for transient failures (429 / 5xx / network), with jittered exponential back-off.
- https is required, except for a local endpoint (localhost / 127.0.0.1) such as vLLM in dev.
- A reply without usable text is an error; callers never receive an empty "answer".
"""

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from ekip_contracts.llm import ChatMessage, LLMError

_RETRYABLE = {429, 500, 502, 503, 504}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
MAX_TOKENS_LIMIT = 8192
LimitField = Literal["max_tokens", "max_completion_tokens"]


class OpenAIChatModel:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        model: str,
        client: httpx.AsyncClient | None = None,
        limit_field: LimitField = "max_tokens",
        send_temperature: bool = True,
        max_attempts: int = 2,
        timeout_s: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = lambda: random.uniform(0.5, 1.5),  # noqa: S311 - not security
    ) -> None:
        if not api_key:
            raise ValueError("an LLM API key is required")
        if not model:
            raise ValueError("an LLM model name is required")
        parts = urlsplit(base_url)
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS
        ):
            raise ValueError("the LLM endpoint must use https (http only for localhost)")
        self._key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        # OpenAI's newer models want `max_completion_tokens` and reject a custom temperature;
        # Groq, DeepSeek and vLLM keep `max_tokens`. Chosen per deployment, never guessed.
        self._limit_field = limit_field
        self._send_temperature = send_temperature
        self._client = client or httpx.AsyncClient(timeout=timeout_s, trust_env=True)
        self._owns_client = client is None
        self._attempts = max(1, max_attempts)
        self._sleep = sleep
        self._jitter = jitter

    @property
    def model_name(self) -> str:
        return self._model

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int = 800, temperature: float = 0.0
    ) -> str:
        if not messages:
            raise LLMError("no messages to send")
        if not 1 <= max_tokens <= MAX_TOKENS_LIMIT:
            raise LLMError("max_tokens out of range")
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            self._limit_field: max_tokens,
        }
        if self._send_temperature:
            body["temperature"] = temperature
        return self._extract(await self._post(body))

    @staticmethod
    def _extract(payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("model returned an unexpected response") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("model returned no text")
        return content

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
                        parsed = response.json()
                    except ValueError as exc:
                        raise LLMError("model returned invalid JSON") from exc
                    if not isinstance(parsed, dict):
                        raise LLMError("model returned an unexpected response")
                    return parsed
                if response.status_code not in _RETRYABLE:
                    raise LLMError(f"model request rejected (HTTP {response.status_code})")
                last = f"HTTP {response.status_code}"
            if attempt + 1 < self._attempts:
                await self._sleep(0.5 * 2**attempt * self._jitter())
        raise LLMError(f"model unavailable ({last})")
