"""The planner's output as validated data.

A plan is a list of sub-questions to search for, optionally ordered by dependencies. It is parsed
from model text, so it is treated as UNTRUSTED input: anything unexpected is rejected (fail closed).
A plan has deliberately no field for tenant, user, groups or tools; the model cannot grant itself
authority by emitting one (such a key is an error, not ignored).
"""

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

MAX_SUBTASKS_DEFAULT = 4
MAX_SUBTASKS_HARD = 8
MAX_QUESTION_CHARS = 500
TASK_ID = re.compile(r"^t[0-9]{1,2}$")
_ALLOWED_PLAN_KEYS = {"subtasks"}
_ALLOWED_TASK_KEYS = {"id", "question", "depends_on"}


class PlanError(Exception):
    """The model's plan is unusable. The message is a fixed phrase (never model text)."""


@dataclass(frozen=True)
class SubTask:
    id: str
    question: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    subtasks: tuple[SubTask, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.subtasks) <= MAX_SUBTASKS_HARD:
            raise PlanError("plan size out of range")
        ids = [t.id for t in self.subtasks]
        if len(set(ids)) != len(ids):
            raise PlanError("duplicate task id")
        known = set(ids)
        for task in self.subtasks:
            if not TASK_ID.fullmatch(task.id):
                raise PlanError("invalid task id")
            question = task.question.strip()
            if not question or len(question) > MAX_QUESTION_CHARS or "\n" in question:
                raise PlanError("invalid task question")
            if task.id in task.depends_on or not set(task.depends_on) <= known:
                raise PlanError("invalid dependency")
        self.waves()  # raises PlanError on a cycle

    def waves(self) -> list[list[SubTask]]:
        """Groups of tasks whose dependencies are all done: each wave can run in parallel."""
        done: set[str] = set()
        remaining = list(self.subtasks)
        waves: list[list[SubTask]] = []
        while remaining:
            ready = [t for t in remaining if set(t.depends_on) <= done]
            if not ready:
                raise PlanError("dependency cycle")
            waves.append(ready)
            done |= {t.id for t in ready}
            remaining = [t for t in remaining if t.id not in done]
        return waves


def single_task_plan(question: str) -> Plan:
    """The safe fallback when planning fails: just search for the user's own question."""
    return Plan((SubTask("t1", question.strip().replace("\n", " ")[:MAX_QUESTION_CHARS]),))


def _strip_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1) if match else text


def parse_plan(text: str, *, max_subtasks: int = MAX_SUBTASKS_DEFAULT) -> Plan:
    """Turn model text into a Plan, or raise PlanError. Never raises anything else."""
    try:
        data: Any = json.loads(_strip_fence(text))
    except ValueError as exc:
        raise PlanError("plan is not valid JSON") from exc
    if not isinstance(data, Mapping) or set(data) - _ALLOWED_PLAN_KEYS:
        raise PlanError("unexpected plan shape")
    raw = data.get("subtasks")
    if not isinstance(raw, list) or not 1 <= len(raw) <= min(max_subtasks, MAX_SUBTASKS_HARD):
        raise PlanError("plan size out of range")
    tasks = [_parse_task(item) for item in raw]
    return Plan(tuple(tasks))


def _parse_task(item: Any) -> SubTask:
    if not isinstance(item, Mapping) or set(item) - _ALLOWED_TASK_KEYS:
        raise PlanError("unexpected task shape")
    task_id, question = item.get("id"), item.get("question")
    depends: Any = item.get("depends_on", [])
    if not isinstance(task_id, str) or not isinstance(question, str):
        raise PlanError("unexpected task shape")
    if not isinstance(depends, list) or not all(isinstance(d, str) for d in depends):
        raise PlanError("unexpected task shape")
    return SubTask(task_id, question.strip(), _unique(depends))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
