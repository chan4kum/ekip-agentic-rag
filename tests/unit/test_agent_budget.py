import pytest
from ekip_agents import Budget, BudgetExceeded, BudgetTracker, Resource


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def tracker(**limits: int | float) -> tuple[BudgetTracker, Clock]:
    clock = Clock()
    return BudgetTracker(Budget(**limits), clock=clock), clock  # type: ignore[arg-type]


def test_spending_within_the_limit_is_counted() -> None:
    t, _ = tracker(max_retrievals=3)
    t.spend(Resource.RETRIEVALS)
    t.spend(Resource.RETRIEVALS, 2)
    assert t.used(Resource.RETRIEVALS) == 3 and t.remaining(Resource.RETRIEVALS) == 0


def test_the_call_that_would_exceed_the_limit_raises_and_spends_nothing() -> None:
    t, _ = tracker(max_retrievals=3)
    t.spend(Resource.RETRIEVALS, 2)
    with pytest.raises(BudgetExceeded) as err:
        t.spend(Resource.RETRIEVALS, 2)
    assert err.value.resource == "retrievals" and t.used(Resource.RETRIEVALS) == 2


def test_each_resource_has_its_own_limit() -> None:
    t, _ = tracker(max_steps=1, max_tool_calls=5)
    t.spend(Resource.STEPS)
    with pytest.raises(BudgetExceeded, match="steps"):
        t.spend(Resource.STEPS)
    t.spend(Resource.TOOL_CALLS, 5)  # unaffected by the exhausted step budget


def test_the_deadline_stops_further_spending() -> None:
    t, clock = tracker(max_seconds=10.0)
    clock.now += 9.0
    t.spend(Resource.STEPS)
    clock.now += 2.0
    assert t.seconds_left == 0.0
    with pytest.raises(BudgetExceeded, match="time"):
        t.spend(Resource.STEPS)


def test_token_budget_is_enforced_in_bulk() -> None:
    t, _ = tracker(max_tokens=1000)
    t.spend(Resource.TOKENS, 900)
    with pytest.raises(BudgetExceeded, match="tokens"):
        t.spend(Resource.TOKENS, 200)


def test_trackers_do_not_share_state() -> None:
    a, _ = tracker(max_steps=1)
    b, _ = tracker(max_steps=1)
    a.spend(Resource.STEPS)
    b.spend(Resource.STEPS)  # b is independent of a


@pytest.mark.parametrize("limits", [{"max_steps": -1}, {"max_seconds": 0}, {"max_tokens": -5}])
def test_invalid_budgets_are_rejected(limits: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Budget(**limits)  # type: ignore[arg-type]


def test_negative_spending_is_rejected() -> None:
    t, _ = tracker()
    with pytest.raises(ValueError):
        t.spend(Resource.STEPS, -1)
