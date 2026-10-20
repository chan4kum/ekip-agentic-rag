import pytest
from ekip_agents import (
    Route,
    RuleBasedRouter,
    StateViolation,
    apply_update,
    make_classify_node,
    new_state,
)
from ekip_contracts.identity import SecurityContext

ALICE = SecurityContext(subject="u-1", tenant_id="tenant-a", principal="alice")
router = RuleBasedRouter()


@pytest.mark.parametrize(
    "text",
    ["hi", "Hello!", "  hey there  ", "thanks", "Thank you!", "good morning", "Good Evening team",
     "who are you?", "what can you do", "tell me a joke", "???", "...", "🙂", "ok"],
)  # fmt: skip
async def test_small_talk_and_empty_content_is_out_of_scope(text: str) -> None:
    assert await router.classify(text) is Route.OUT_OF_SCOPE


@pytest.mark.parametrize(
    "text",
    ["What is the torque value for AMM 32-11-01?", "hi, what is our leave policy?",
     "Compare the latest maintenance procedure with the previous version",
     "hello world example in the coding standards", "ok to carry forward leave?", "SB-2026-014"],
)  # fmt: skip
async def test_real_questions_go_to_the_documents_even_if_they_start_with_small_talk(
    text: str,
) -> None:
    assert await router.classify(text) is Route.KNOWLEDGE


async def test_unclear_input_defaults_to_the_safe_documents_path() -> None:
    assert await router.classify("blah blah") is Route.KNOWLEDGE


async def test_the_classify_node_returns_a_partial_update_and_counts_the_step() -> None:
    node = make_classify_node(router)
    state = new_state("What is the leave policy?", ALICE)
    update = await node(state)
    assert update == {"route": Route.KNOWLEDGE, "steps": 1}
    assert "route" not in state  # the node did not mutate its input
    assert apply_update(state, update)["route"] is Route.KNOWLEDGE


async def test_the_node_works_with_any_router_implementation() -> None:
    class AlwaysOut:
        async def classify(self, question: str) -> Route:
            return Route.OUT_OF_SCOPE

    update = await make_classify_node(AlwaysOut())(new_state("anything", ALICE))
    assert update["route"] is Route.OUT_OF_SCOPE


async def test_a_state_without_a_question_is_refused() -> None:
    with pytest.raises(StateViolation):
        await make_classify_node(router)({})
