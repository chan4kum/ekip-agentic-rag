"""Query rewriting: follow-ups become standalone searches, and every failure falls back safely."""

import pytest
from ekip_agents import (
    Graph,
    LLMQueryRewriter,
    RuleBasedRouter,
    RunStatus,
    build_agent_graph,
    clean_query,
    run_agent,
)
from ekip_contracts.conversation import Turn
from ekip_contracts.identity import SecurityContext
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_persistence import InMemoryConversationStore
from fake_llm import FakeChatModel, outage

ALICE = SecurityContext(subject="u-1", tenant_id="tenant-a", principal="alice", groups=frozenset())
CONV = "conv-12345678"
HIT = RetrievedChunk(
    chunk_id="c1", document_id="d1", text="Torque is 140 Nm.", score=1.0, title="Manual"
)
PRIOR = [Turn("What is the torque value for bolt A?", "Bolt A is 140 Nm [1].")]


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.scopes: list[SearchScope] = []

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        self.queries.append(query)
        self.scopes.append(scope)
        return [HIT]


async def test_no_history_means_no_model_call_and_the_question_is_unchanged() -> None:
    model = FakeChatModel("ignored")
    assert await LLMQueryRewriter(model).rewrite("torque?", []) == "torque?"
    assert model.calls == []


async def test_a_follow_up_is_rewritten_using_the_history() -> None:
    model = FakeChatModel("What is the torque value for the other bolt?")
    out = await LLMQueryRewriter(model).rewrite("And for the other bolt?", PRIOR)
    assert out == "What is the torque value for the other bolt?"
    prompt = model.calls[0][1].content
    assert "<history>" in prompt and "bolt A" in prompt and "[1]" not in prompt
    assert model.kwargs[0]["temperature"] == 0.0


async def test_a_model_outage_falls_back_to_the_original_question() -> None:
    rewriter = LLMQueryRewriter(FakeChatModel(error=outage()))
    assert await rewriter.rewrite("And the other?", PRIOR) == "And the other?"


@pytest.mark.parametrize(
    "reply", ["", "   ", "two\nlines", "x" * 501, "Sure!\nHere you go:\nquery"]
)
def test_unusable_replies_fall_back(reply: str) -> None:
    assert clean_query(reply, "original") == "original"


def test_quotes_and_whitespace_are_stripped_from_a_good_reply() -> None:
    assert clean_query('  "torque for bolt B"\n', "original") == "torque for bolt B"


def build(retriever: RecordingRetriever, model: FakeChatModel, store: object | None) -> Graph:
    return build_agent_graph(
        router=RuleBasedRouter(),
        retriever=retriever,
        model=model,
        store=store,  # type: ignore[arg-type]
        rewriter=LLMQueryRewriter(model),
    )


async def test_the_graph_searches_with_the_rewrite_but_answers_the_original_question() -> None:
    store, retriever = InMemoryConversationStore(), RecordingRetriever()
    model = FakeChatModel("Torque is 140 Nm [1].")
    g = build(retriever, model, store)
    await run_agent(g, "What is the torque value for bolt A?", ALICE, CONV)
    model.reply = "What is the torque value for the other bolt?"
    # the rewrite call returns the standalone query; the answer call is next
    replies = iter(["What is the torque value for the other bolt?", "Bolt B is 90 Nm [1]."])

    async def scripted(*_: object, **__: object) -> str:
        return next(replies)

    model.complete = scripted  # type: ignore[method-assign]
    final = await run_agent(g, "And for the other bolt?", ALICE, CONV)
    assert retriever.queries[-1] == "What is the torque value for the other bolt?"
    assert final["status"] is RunStatus.ANSWERED and final["question"] == "And for the other bolt?"


async def test_the_rewrite_cannot_change_who_is_searching() -> None:
    retriever = RecordingRetriever()
    model = FakeChatModel("Torque is 140 Nm [1].")
    g = build(retriever, model, None)
    await run_agent(g, "What is the torque value?", ALICE)
    assert retriever.scopes == [SearchScope("tenant-a", user="alice", groups=frozenset())]


async def test_a_first_question_costs_no_extra_model_call() -> None:
    retriever, model = RecordingRetriever(), FakeChatModel("Torque is 140 Nm [1].")
    await run_agent(build(retriever, model, InMemoryConversationStore()), "torque?", ALICE, CONV)
    assert len(model.calls) == 1 and retriever.queries == ["torque?"]
