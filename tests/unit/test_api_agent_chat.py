"""/api/v1/chat served by the LangGraph agent: framing, citations, memory, wiring failures."""

import json
from typing import Any

import httpx
import pytest
from ekip_agents import RuleBasedRouter, build_agent_graph
from ekip_api.adapters.agent_chat import AgentChatStreamer
from ekip_api.main import create_app
from ekip_config.settings import Settings
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_persistence import InMemoryConversationStore
from fake_llm import FakeChatModel, outage
from fastapi import FastAPI
from idp_testkit import TestIdp, open_limits
from pydantic import SecretStr

IDP = TestIdp()
URL = "/api/v1/chat"
HIT = RetrievedChunk(
    chunk_id="c1", document_id="d1", text="Torque is 140 Nm.", score=1.0, title="Manual"
)


def make_app(streamer: object) -> FastAPI:
    return create_app(
        Settings(_env_file=None, app_env="test"),
        token_validator=IDP.validator(),
        rate_limits=open_limits(),
        chat_streamer=streamer,  # type: ignore[arg-type]
    )


def http(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def parse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in text.split("\n\n"):
        lines = [x for x in block.split("\n") if x and not x.startswith(":")]
        if lines:
            events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


class FixedRetriever:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        return self.hits


def agent_app(
    model: FakeChatModel, hits: list[RetrievedChunk] | None = None, store: Any = None
) -> FastAPI:
    graph = build_agent_graph(
        router=RuleBasedRouter(),
        retriever=FixedRetriever([HIT] if hits is None else hits),
        model=model,
        store=store,
    )
    return make_app(AgentChatStreamer(graph))


async def chat(
    app: FastAPI, message: str, conversation_id: str | None = None
) -> list[tuple[str, dict[str, Any]]]:
    body: dict[str, str] = {"message": message}
    if conversation_id:
        body["conversation_id"] = conversation_id
    async with http(app) as c:
        r = await c.post(URL, json=body, headers=IDP.auth())
    assert r.status_code == 200
    return parse(r.text)


def text_of(events: list[tuple[str, dict[str, Any]]]) -> str:
    return "".join(d["text"] for n, d in events if n == "token")


async def test_a_grounded_answer_streams_with_only_the_cited_sources() -> None:
    events = await chat(agent_app(FakeChatModel("Torque is 140 Nm [1].")), "torque value?")
    names = [n for n, _ in events]
    assert names[0] == "meta" and names[-1] == "done"
    assert names.index("token") < names.index("citations") < names.index("status")
    assert text_of(events) == "Torque is 140 Nm [1]."
    citations = dict(events)["citations"]["results"]
    assert [c["number"] for c in citations] == [1] and citations[0]["document_id"] == "d1"
    assert dict(events)["status"]["status"] == "answered"


async def test_no_evidence_streams_the_fixed_message_and_no_citations() -> None:
    events = await chat(agent_app(FakeChatModel(), hits=[]), "torque value?")
    assert "could not find" in text_of(events)
    assert dict(events)["citations"]["results"] == []
    assert dict(events)["status"]["status"] == "insufficient_evidence"


async def test_a_model_outage_is_a_safe_message_not_a_crash() -> None:
    events = await chat(agent_app(FakeChatModel(error=outage())), "torque value?")
    assert [n for n, _ in events][-1] == "done" and "error" not in dict(events)
    assert dict(events)["status"]["status"] == "failed"
    assert "provider unavailable" not in text_of(events)


async def test_the_conversation_id_reaches_the_agent_and_memory_carries_over() -> None:
    store, model = InMemoryConversationStore(), FakeChatModel("Torque is 140 Nm [1].")
    app = agent_app(model, store=store)
    first = await chat(app, "What is the torque value?", "conv-12345678")
    assert first[0][1]["conversation_id"] == "conv-12345678"
    await chat(app, "And for the other bolt?", "conv-12345678")
    assert "What is the torque value?" in model.calls[-1][1].content


async def test_a_generated_conversation_id_is_accepted_back() -> None:
    store, model = InMemoryConversationStore(), FakeChatModel("Torque is 140 Nm [1].")
    app = agent_app(model, store=store)
    conv = (await chat(app, "What is the torque value?"))[0][1]["conversation_id"]
    await chat(app, "And again?", conv)
    assert "What is the torque value?" in model.calls[-1][1].content


def test_agent_backend_without_a_model_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="LLM_MODEL"):
        create_app(
            Settings(_env_file=None, app_env="test", chat_backend="agent"),
            token_validator=IDP.validator(),
            rate_limits=open_limits(),
        )


def test_agent_backend_wires_up_when_configured() -> None:
    app = create_app(
        Settings(
            _env_file=None,
            app_env="test",
            chat_backend="agent",
            llm_model="m-1",
            llm_api_key=SecretStr("sk-fake"),  # fake: nothing calls the provider here
        ),
        token_validator=IDP.validator(),
        rate_limits=open_limits(),
    )
    assert type(app.state.chat_streamer).__name__ == "AgentChatStreamer"


async def test_the_stub_remains_the_default() -> None:
    async with http(
        create_app(
            Settings(_env_file=None, app_env="test"),
            token_validator=IDP.validator(),
            rate_limits=open_limits(),
        )
    ) as c:
        r = await c.post(URL, json={"message": "torque"}, headers=IDP.auth())
    assert isinstance(r, httpx.Response) and "[stub answer" in r.text
