import pytest
from ekip_agents import (
    PROTECTED_KEYS,
    Route,
    RunStatus,
    StateViolation,
    apply_update,
    check_update,
    new_state,
)
from ekip_contracts.identity import SecurityContext
from pydantic import ValidationError

ALICE = SecurityContext(
    subject="u-1", tenant_id="tenant-a", principal="alice", groups=frozenset({"eng"})
)


def test_a_run_starts_with_the_question_the_identity_and_a_running_status() -> None:
    state = new_state("  What is the torque value?  ", ALICE)
    assert state["question"] == "What is the torque value?"
    assert state["security"] is ALICE
    assert state["status"] is RunStatus.RUNNING and state["steps"] == 0 and state["hits"] == []


@pytest.mark.parametrize("bad", ["", "   ", "x" * 4001])
def test_empty_or_oversized_questions_never_start_a_run(bad: str) -> None:
    with pytest.raises(StateViolation):
        new_state(bad, ALICE)


def test_a_node_update_is_merged_without_touching_other_keys() -> None:
    state = new_state("q", ALICE)
    after = apply_update(state, {"route": Route.KNOWLEDGE, "steps": 1})
    assert after["route"] is Route.KNOWLEDGE and after["steps"] == 1
    assert after["question"] == "q" and after["security"] is ALICE
    assert "route" not in state  # the old state is not mutated


@pytest.mark.parametrize("key", sorted(PROTECTED_KEYS))
def test_no_node_may_replace_the_identity_or_the_question(key: str) -> None:
    mallory = SecurityContext(subject="x", tenant_id="tenant-b", principal="root")
    value = mallory if key == "security" else "something else"
    with pytest.raises(StateViolation, match="may not change"):
        apply_update(new_state("q", ALICE), {key: value})


def test_unknown_keys_are_rejected_so_typos_fail_loudly() -> None:
    with pytest.raises(StateViolation, match="unknown state keys"):
        check_update({"answr": "typo"})


def test_the_identity_object_itself_is_frozen() -> None:
    state = new_state("q", ALICE)
    with pytest.raises(ValidationError):  # SecurityContext is a frozen model
        state["security"].tenant_id = "tenant-b"  # type: ignore[misc]
