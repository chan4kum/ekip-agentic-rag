"""Agent settings: safe defaults, validation, and what the wiring builds from them."""

from typing import Any

import pytest
from ekip_agents import RunStatus
from ekip_api.adapters.agent_wiring import build_agent_streamer
from ekip_config.settings import Settings
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from pydantic import ValidationError


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"chat_backend": "agent", "llm_model": "m", "llm_api_key": "k"}
    return Settings(_env_file=None, **{**base, **overrides})


class NoHits:
    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        return []


def test_defaults_keep_the_phase_6_behaviour() -> None:
    s = Settings(_env_file=None)
    assert s.chat_backend == "stub"
    assert s.agent_multistep is False and s.agent_groundedness is False
    assert s.agent_checkpointer == "none" and s.agent_approval is False
    assert s.agent_max_retrievals == 4 and s.agent_max_model_calls == 10


@pytest.mark.parametrize(
    "bad",
    [
        {"agent_max_subtasks": 0},
        {"agent_max_subtasks": 9},
        {"agent_max_steps": 3},
        {"agent_max_retrievals": 0},
        {"agent_max_model_calls": 1},
        {"agent_max_seconds": 0},
        {"agent_max_seconds": 301},
        {"agent_checkpointer": "redis"},
    ],
)
def test_out_of_range_values_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        settings(**bad)


def test_approval_needs_a_checkpointer() -> None:
    with pytest.raises(ValidationError, match="AGENT_CHECKPOINTER"):
        settings(agent_approval=True)
    assert settings(agent_approval=True, agent_checkpointer="memory").agent_approval


def test_the_chat_api_refuses_to_start_with_approval_on() -> None:
    with pytest.raises(ValueError, match="AGENT_APPROVAL"):
        build_agent_streamer(settings(agent_approval=True, agent_checkpointer="memory"), NoHits())


def test_wiring_adds_nothing_by_default() -> None:
    _, openers, closers = build_agent_streamer(settings(), NoHits())
    assert openers == [] and len(closers) == 1  # just the model client


def test_a_postgres_checkpointer_is_opened_and_closed_with_the_app() -> None:
    _, openers, closers = build_agent_streamer(settings(agent_checkpointer="postgres"), NoHits())
    assert len(openers) == 1 and len(closers) == 2


async def test_status_event_reports_truncation_and_groundedness_fields() -> None:
    from ekip_agents import RuleBasedRouter, build_agent_graph
    from ekip_api.adapters.agent_chat import AgentChatStreamer
    from ekip_contracts.identity import SecurityContext
    from fake_llm import FakeChatModel

    graph = build_agent_graph(router=RuleBasedRouter(), retriever=NoHits(), model=FakeChatModel())
    events = [
        e
        async for e in AgentChatStreamer(graph).stream(
            message="What is the torque value?",
            context=SecurityContext(subject="u-1", tenant_id="t", principal="alice"),
        )
    ]
    status = next(e.data for e in events if e.event == "status")
    assert status["status"] == RunStatus.INSUFFICIENT_EVIDENCE.value
    assert status["truncated"] == "" and status["grounded"] == "unchecked"
