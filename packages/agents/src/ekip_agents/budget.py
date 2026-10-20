"""Hard limits on one agent run, enforced in code (not by asking the model to behave).

Spending is explicit: a node calls `spend(...)` BEFORE doing the work. If a limit would be
exceeded, `BudgetExceeded` is raised and the graph ends the run with the best supported answer or
an explicit failure. The clock is injected so deadlines are testable without sleeping.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class Resource(StrEnum):
    STEPS = "steps"
    RETRIEVALS = "retrievals"
    TOOL_CALLS = "tool_calls"
    MODEL_CALLS = "model_calls"
    TOKENS = "tokens"


class BudgetExceeded(Exception):
    """A limit was reached. `resource` says which one (a fixed code, safe to log)."""

    def __init__(self, resource: str) -> None:
        super().__init__(f"budget exceeded: {resource}")
        self.resource = resource


@dataclass(frozen=True)
class Budget:
    max_steps: int = 12
    max_retrievals: int = 4  # one per sub-question of the largest default plan
    max_tool_calls: int = 8
    max_model_calls: int = 10
    max_tokens: int = 20_000
    max_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "max_retrievals",
            "max_tool_calls",
            "max_model_calls",
            "max_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")

    def limit(self, resource: Resource) -> int:
        return {
            Resource.STEPS: self.max_steps,
            Resource.RETRIEVALS: self.max_retrievals,
            Resource.TOOL_CALLS: self.max_tool_calls,
            Resource.MODEL_CALLS: self.max_model_calls,
            Resource.TOKENS: self.max_tokens,
        }[resource]


class BudgetTracker:
    """Mutable counter for ONE run. Create a fresh one per question; never share across users."""

    def __init__(self, budget: Budget, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._started = clock()
        self._used: dict[Resource, int] = dict.fromkeys(Resource, 0)

    def used(self, resource: Resource) -> int:
        return self._used[resource]

    def remaining(self, resource: Resource) -> int:
        return max(self.budget.limit(resource) - self._used[resource], 0)

    @property
    def seconds_left(self) -> float:
        return max(self.budget.max_seconds - (self._clock() - self._started), 0.0)

    def check_deadline(self) -> None:
        if self.seconds_left <= 0:
            raise BudgetExceeded("time")

    def spend(self, resource: Resource, amount: int = 1) -> None:
        """Reserve `amount`. Raises BudgetExceeded, spending nothing, if it would not fit."""
        if amount < 0:
            raise ValueError("amount must not be negative")
        self.check_deadline()
        if self._used[resource] + amount > self.budget.limit(resource):
            raise BudgetExceeded(resource.value)
        self._used[resource] += amount
