import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from ekip_contracts.embedding import EmbeddingError, EmbeddingTask, EmbeddingUnavailableError
from ekip_llm_gateway import (
    CachingEmbedder,
    GeminiEmbedder,
    HashingEmbedder,
    InMemoryEmbeddingCache,
)

DIMS = 128
KEY = "test-key-do-not-log"


def _vec(seed: float) -> list[float]:
    return [seed + i for i in range(DIMS)]


def _embedder(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> GeminiEmbedder:
    async def no_sleep(_: float) -> None:
        return None

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiEmbedder(
        KEY, dimensions=DIMS, client=client, sleep=no_sleep, jitter=lambda: 1.0, **kw
    )


def _ok(request: httpx.Request) -> httpx.Response:
    n = len(json.loads(request.content)["requests"])
    return httpx.Response(
        200, json={"embeddings": [{"values": _vec(float(i + 1))} for i in range(n)]}
    )


async def test_vectors_are_unit_length_and_model_version_names_dimensions() -> None:
    e = _embedder(_ok)
    [v] = await e.embed(["hello"], task=EmbeddingTask.DOCUMENT)
    assert len(v) == DIMS
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)
    assert e.model_version == f"gemini-embedding-001@{DIMS}"


async def test_request_shape_key_in_header_only_and_task_type() -> None:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return _ok(r)

    e = _embedder(handler)
    await e.embed(["a"], task=EmbeddingTask.QUERY)
    r = seen[0]
    assert r.headers["x-goog-api-key"] == KEY
    assert KEY not in str(r.url)
    body = json.loads(r.content)["requests"][0]
    assert body["taskType"] == "RETRIEVAL_QUERY"
    assert body["outputDimensionality"] == DIMS


async def test_identical_texts_are_sent_once_and_order_is_preserved() -> None:
    sent: list[list[str]] = []

    def handler(r: httpx.Request) -> httpx.Response:
        sent.append([q["content"]["parts"][0]["text"] for q in json.loads(r.content)["requests"]])
        return _ok(r)

    e = _embedder(handler)
    out = await e.embed(["a", "b", "a"], task=EmbeddingTask.DOCUMENT)
    assert sent == [["a", "b"]]
    assert out[0] == out[2] and out[0] != out[1]


async def test_large_input_is_split_into_batches() -> None:
    sizes: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(r.content)["requests"]))
        return _ok(r)

    e = _embedder(handler, batch_size=10)
    out = await e.embed([f"t{i}" for i in range(25)], task=EmbeddingTask.DOCUMENT)
    assert len(out) == 25 and sorted(sizes) == [5, 10, 10]


async def test_retries_transient_errors_then_succeeds() -> None:
    calls = 0

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"retry-after": "1"}) if calls < 3 else _ok(r)

    e = _embedder(handler)
    await e.embed(["x"], task=EmbeddingTask.DOCUMENT)
    assert calls == 3


async def test_gives_up_after_max_attempts_with_unavailable_error() -> None:
    calls = 0

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="secret provider body")

    e = _embedder(handler, max_attempts=3)
    with pytest.raises(EmbeddingUnavailableError) as exc:
        await e.embed(["x"], task=EmbeddingTask.DOCUMENT)
    assert calls == 3
    assert "secret provider body" not in str(exc.value) and KEY not in str(exc.value)


async def test_client_errors_are_not_retried_and_hide_the_body() -> None:
    calls = 0

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="echo of the input text")

    e = _embedder(handler)
    with pytest.raises(EmbeddingError) as exc:
        await e.embed(["x"], task=EmbeddingTask.DOCUMENT)
    assert calls == 1
    assert "echo" not in str(exc.value)
    assert not isinstance(exc.value, EmbeddingUnavailableError)


async def test_network_errors_are_retried() -> None:
    calls = 0

    def handler(r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("boom")
        return _ok(r)

    e = _embedder(handler)
    await e.embed(["x"], task=EmbeddingTask.DOCUMENT)
    assert calls == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"embeddings": []},  # wrong count
        {"embeddings": [{"values": [1.0, 2.0]}]},  # wrong dimension
        {"embeddings": [{"values": [float("nan")] * DIMS}]},  # not finite
        {"embeddings": [{"values": [0.0] * DIMS}]},  # zero vector
        {"nope": 1},  # wrong shape
    ],
)
async def test_malformed_responses_are_rejected(payload: dict[str, Any]) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload, allow_nan=True))

    e = _embedder(handler)
    with pytest.raises(EmbeddingError):
        await e.embed(["x"], task=EmbeddingTask.DOCUMENT)


async def test_empty_text_is_rejected_before_any_request() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    e = _embedder(handler)
    with pytest.raises(EmbeddingError):
        await e.embed(["ok", "   "], task=EmbeddingTask.DOCUMENT)


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        GeminiEmbedder("")
    with pytest.raises(ValueError):
        GeminiEmbedder("k", dimensions=64)
    with pytest.raises(ValueError):
        GeminiEmbedder("k", batch_size=101)


async def test_cache_embeds_each_text_once_and_keys_by_task() -> None:
    inner = HashingEmbedder(64)
    cached = CachingEmbedder(inner, InMemoryEmbeddingCache())
    a = await cached.embed(["alpha beta", "gamma", "alpha beta"], task=EmbeddingTask.DOCUMENT)
    assert inner.texts_embedded == 2 and a[0] == a[2]
    await cached.embed(["alpha beta"], task=EmbeddingTask.DOCUMENT)
    assert inner.calls == 1  # fully served from cache
    await cached.embed(["alpha beta"], task=EmbeddingTask.QUERY)
    assert inner.calls == 2  # different task, different key


async def test_cache_misses_when_model_version_changes() -> None:
    cache = InMemoryEmbeddingCache()
    one, two = HashingEmbedder(64), HashingEmbedder(32)
    await CachingEmbedder(one, cache).embed(["same text"], task=EmbeddingTask.DOCUMENT)
    await CachingEmbedder(two, cache).embed(["same text"], task=EmbeddingTask.DOCUMENT)
    assert two.texts_embedded == 1


async def test_cache_lru_is_bounded() -> None:
    cache = InMemoryEmbeddingCache(max_entries=2)
    c = CachingEmbedder(HashingEmbedder(64), cache)
    await c.embed(["one", "two", "three"], task=EmbeddingTask.DOCUMENT)
    assert len(cache) == 2


def dot(x: Sequence[float], y: Sequence[float]) -> float:
    return sum(p * q for p, q in zip(x, y, strict=True))


async def test_hashing_embedder_is_deterministic_and_similar_texts_are_closer() -> None:
    e = HashingEmbedder(256)
    [a, b, c] = await e.embed(
        [
            "maintenance procedure for landing gear",
            "landing gear maintenance procedure",
            "quarterly sales forecast",
        ],
        task=EmbeddingTask.DOCUMENT,
    )
    assert dot(a, b) > dot(a, c)
    assert (await e.embed(["landing gear"], task=EmbeddingTask.DOCUMENT)) == (
        await e.embed(["landing gear"], task=EmbeddingTask.DOCUMENT)
    )
