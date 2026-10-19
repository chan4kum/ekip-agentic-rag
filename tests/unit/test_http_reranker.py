"""HttpReranker against a mocked HTTP API: request shape, response validation, failures."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from ekip_contracts.retrieval import RerankError, RetrievedChunk
from ekip_llm_gateway import HttpReranker

URL = "https://rerank.example.test/v1/rerank"
SECRET_KEY = "rr-secret-key-123"  # noqa: S105 - test value


def chunk(cid: str, text: str = "passage") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id="d", text=text, score=0.5)


def make(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> HttpReranker:
    async def no_sleep(_: float) -> None:
        return None

    return HttpReranker(
        SECRET_KEY, url=URL, model="rerank-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=no_sleep, jitter=lambda: 1.0, **kw,
    )  # fmt: skip


def reply(results: list[dict[str, Any]], key: str = "results") -> httpx.Response:
    return httpx.Response(200, json={key: results})


async def test_request_shape_and_ordering() -> None:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return reply([{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}])

    out = await make(handler).rerank("q", [chunk("a"), chunk("b"), chunk("c")], limit=2)
    assert [h.chunk_id for h in out] == ["c", "a"] and out[0].score == 0.9
    body = json.loads(seen[0].content)
    assert body["query"] == "q" and body["model"] == "rerank-test" and body["top_n"] == 2
    assert body["documents"] == ["passage"] * 3
    assert seen[0].headers["authorization"] == f"Bearer {SECRET_KEY}"
    assert SECRET_KEY not in str(seen[0].url)


async def test_voyage_style_data_key_is_accepted() -> None:
    out = await make(lambda r: reply([{"index": 1, "relevance_score": 0.7}], "data")).rerank(
        "q", [chunk("a"), chunk("b")], limit=5
    )
    assert [h.chunk_id for h in out] == ["b"]


async def test_passages_are_truncated_and_candidates_capped() -> None:
    sent: list[dict[str, Any]] = []

    def handler(r: httpx.Request) -> httpx.Response:
        sent.append(json.loads(r.content))
        return reply([{"index": 0, "relevance_score": 1.0}])

    r = make(handler, max_candidates=2, max_chars=5)
    await r.rerank("q", [chunk("a", "abcdefghij"), chunk("b"), chunk("c")], limit=1)
    assert sent[0]["documents"] == ["abcde", "passa"]


async def test_nothing_to_rerank_makes_no_call() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called")

    assert await make(handler).rerank("q", [], limit=3) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"results": "nope"},
        {"results": [{"index": 5, "relevance_score": 1}]},  # out of range
        {"results": [{"index": -1, "relevance_score": 1}]},
        {"results": [{"index": 0, "relevance_score": 1}, {"index": 0, "relevance_score": 2}]},
        {"results": [{"index": True, "relevance_score": 1}]},
        {"results": [{"index": 0, "relevance_score": "x"}]},
        {"results": [{"index": 0}]},
        {"results": [{"index": 0, "relevance_score": float("nan")}]},
        {"other": []},
    ],
)
async def test_malformed_responses_are_rerank_errors(payload: dict[str, Any]) -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload).encode())

    with pytest.raises(RerankError):
        await make(handler).rerank("q", [chunk("a")], limit=1)


async def test_non_json_and_non_object_bodies_are_rerank_errors() -> None:
    for content in (b"<html>", b"[1,2]"):
        with pytest.raises(RerankError):
            await make(lambda r, c=content: httpx.Response(200, content=c)).rerank(  # type: ignore[misc]
                "q", [chunk("a")], limit=1
            )


async def test_transient_errors_are_retried_then_reported() -> None:
    calls: list[int] = []

    def flaky(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        return reply([{"index": 0, "relevance_score": 0.5}])

    out = await make(flaky).rerank("q", [chunk("a")], limit=1)
    assert len(calls) == 2 and out[0].chunk_id == "a"

    down = make(lambda r: httpx.Response(503, text="CONFIDENTIAL passage echo"))
    with pytest.raises(RerankError) as caught:
        await down.rerank("q", [chunk("a")], limit=1)
    assert "503" in str(caught.value) and "CONFIDENTIAL" not in str(caught.value)


async def test_permanent_errors_are_not_retried_and_hide_the_body() -> None:
    calls: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text=f"bad key {SECRET_KEY}")

    with pytest.raises(RerankError) as caught:
        await make(handler).rerank("q", [chunk("a")], limit=1)
    assert len(calls) == 1 and SECRET_KEY not in str(caught.value)


async def test_network_errors_are_rerank_errors() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(RerankError):
        await make(handler).rerank("q", [chunk("a")], limit=1)


def test_constructor_guards() -> None:
    with pytest.raises(ValueError):
        HttpReranker("", url=URL, model="m")
    with pytest.raises(ValueError):
        HttpReranker("k", url="http://insecure.test/rerank", model="m")
