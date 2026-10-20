"""The OpenAI-compatible chat adapter, driven by httpx's mock transport: no network, no key."""

import json

import httpx
import pytest
from ekip_contracts.llm import ChatMessage, LLMError
from ekip_llm_gateway import OpenAIChatModel

SECRET = "sk-test-secret"  # noqa: S105 - fake value for tests
MESSAGES = [ChatMessage("system", "be brief"), ChatMessage("user", "Torque? CONFIDENTIAL-TEXT")]


def ok(text: str = "Torque is 140 Nm [1].") -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": text}}]}
    )


async def no_sleep(_: float) -> None:
    return None


def model(handler, **kw) -> OpenAIChatModel:  # type: ignore[no-untyped-def]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIChatModel(
        SECRET,
        base_url=kw.pop("base_url", "https://api.example.com/v1/"),
        model="m-1",
        client=client,
        sleep=no_sleep,
        jitter=lambda: 1.0,
        **kw,
    )


async def test_sends_an_openai_shaped_request_and_returns_the_text() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok()

    m = model(handler)
    assert await m.complete(MESSAGES, max_tokens=50, temperature=0.2) == "Torque is 140 Nm [1]."
    request = seen[0]
    assert str(request.url) == "https://api.example.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    body = json.loads(request.content)
    assert body["model"] == "m-1" and body["max_tokens"] == 50 and body["temperature"] == 0.2
    assert body["messages"][1] == {"role": "user", "content": "Torque? CONFIDENTIAL-TEXT"}
    assert m.model_name == "m-1"


async def test_transient_failures_are_retried_then_succeed() -> None:
    codes = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503) if next(codes) == 503 else ok()

    assert await model(handler).complete(MESSAGES) == "Torque is 140 Nm [1]."


async def test_persistent_outage_raises_without_leaking_the_body_or_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"echo CONFIDENTIAL-TEXT {SECRET}")

    with pytest.raises(LLMError) as err:
        await model(handler).complete(MESSAGES)
    assert "CONFIDENTIAL" not in str(err.value) and SECRET not in str(err.value)
    assert "500" in str(err.value)


async def test_client_errors_are_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(LLMError, match="401"):
        await model(handler, max_attempts=3).complete(MESSAGES)
    assert calls == 1


async def test_network_errors_become_llm_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom CONFIDENTIAL-TEXT")

    with pytest.raises(LLMError, match="network") as err:
        await model(handler).complete(MESSAGES)
    assert "CONFIDENTIAL" not in str(err.value)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": "  "}}]},
        {"choices": [{"message": {"content": None}}]},
        ["not", "a", "dict"],
    ],
)
async def test_unusable_replies_are_errors(payload: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(LLMError):
        await model(handler).complete(MESSAGES)


async def test_invalid_json_is_an_error() -> None:
    with pytest.raises(LLMError, match="JSON"):
        await model(lambda r: httpx.Response(200, text="<html>")).complete(MESSAGES)


async def test_bad_arguments_are_rejected_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    m = model(handler)
    with pytest.raises(LLMError):
        await m.complete([])
    for bad in (0, 100_000):
        with pytest.raises(LLMError):
            await m.complete(MESSAGES, max_tokens=bad)


@pytest.mark.parametrize(
    ("key", "url", "name"),
    [
        ("", "https://a.com/v1", "m"),
        ("k", "https://a.com/v1", ""),
        ("k", "http://a.com/v1", "m"),
        ("k", "ftp://a.com", "m"),
    ],
)
def test_construction_is_fail_closed(key: str, url: str, name: str) -> None:
    with pytest.raises(ValueError):
        OpenAIChatModel(key, base_url=url, model=name)


def test_plain_http_is_allowed_for_a_local_endpoint_only() -> None:
    OpenAIChatModel("k", base_url="http://localhost:8000/v1", model="m")
    OpenAIChatModel("k", base_url="http://127.0.0.1:8000/v1", model="m")


async def test_openai_style_models_get_max_completion_tokens_and_no_temperature() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return ok()

    m = model(handler, limit_field="max_completion_tokens", send_temperature=False)
    await m.complete(MESSAGES, max_tokens=64, temperature=0.7)
    assert seen[0]["max_completion_tokens"] == 64
    assert "max_tokens" not in seen[0] and "temperature" not in seen[0]
