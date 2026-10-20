import pytest
from ekip_agents import (
    AgentState,
    RunStatus,
    StateViolation,
    apply_update,
    make_verify_node,
    new_state,
)
from ekip_contracts.identity import SecurityContext
from ekip_contracts.retrieval import RetrievedChunk
from ekip_retrieval import ContextAssembler

ALICE = SecurityContext(subject="u-1", tenant_id="tenant-a", principal="alice")


def drafted(answer: str, sources: int = 2) -> AgentState:
    hits = [
        RetrievedChunk(chunk_id=f"c{i}", document_id=f"d{i}", text=f"fact {i}", score=1.0)
        for i in range(sources)
    ]
    state = new_state("q", ALICE)
    return apply_update(
        state, {"hits": hits, "context": ContextAssembler().assemble(hits), "answer": answer}
    )


async def test_an_answer_citing_real_sources_is_accepted() -> None:
    update = await make_verify_node()(drafted("Torque is 140 Nm [1]. Interval is 400 h [2]."))
    assert update["status"] is RunStatus.ANSWERED and "answer" not in update  # draft kept as is
    assert update["citations"].cited == (1, 2) and update["steps"] == 1


async def test_invented_source_numbers_fail_the_run_and_withhold_the_draft() -> None:
    update = await make_verify_node()(drafted("Torque is 140 Nm [1]. Also [7]."))
    assert update["status"] is RunStatus.FAILED and update["reason"] == "invalid_citations"
    assert update["answer"] == "" and update["citations"].unknown == (7,)


async def test_a_confident_answer_without_citations_is_not_trusted() -> None:
    update = await make_verify_node()(drafted("Torque is 140 Nm."))
    assert update["status"] is RunStatus.INSUFFICIENT_EVIDENCE
    assert update["reason"] == "no_citations" and update["answer"] == ""


async def test_a_citation_to_a_source_that_was_not_provided_is_invalid() -> None:
    update = await make_verify_node()(drafted("Fact [2].", sources=1))
    assert update["status"] is RunStatus.FAILED


@pytest.mark.parametrize("state", [{}, {"answer": "x [1]"}])
async def test_verify_needs_an_answer_and_a_context(state: AgentState) -> None:
    with pytest.raises(StateViolation):
        await make_verify_node()(state)


async def test_the_withheld_draft_cannot_survive_the_merge() -> None:
    state = drafted("Made-up claim [9].")
    merged = apply_update(state, await make_verify_node()(state))
    assert merged["answer"] == "" and merged["status"] is RunStatus.FAILED
