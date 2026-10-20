"""Decide whether a question needs several searches (COMPLEX) or one (SIMPLE).

The default is a transparent rule check: no model call, no cost, deterministic. It is a heuristic
and is judged by the multi-document evaluation set, not assumed to be right. A wrong SIMPLE falls
back to the Phase 6 path (still correct, just one search); a wrong COMPLEX costs an extra planning
call. Neither can widen access.
"""

import re
from enum import StrEnum
from typing import Protocol


class Complexity(StrEnum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class ComplexityRouter(Protocol):
    async def assess(self, question: str) -> Complexity: ...


_COMPLEX_PATTERNS = (
    r"\bcompar(?:e|es|ed|ing|ison)\b",
    r"\bversus\b|\bvs\.?(?=\s|$)",
    r"\bdifferences?\b|\bdiffer(?:s|ed)?\b",
    r"\bwhat(?:'s| has| have| is)? changed\b|\bchanges?\s+(?:between|from|in)\b",
    r"\bhow (?:has|have|did|does) .{0,60}\bchange",
    r"\bbetween\b.{1,80}\band\b",
    r"\b(?:both|each of|all of)\b",
)
_COMPLEX = re.compile("|".join(_COMPLEX_PATTERNS), re.IGNORECASE)


class RuleBasedComplexityRouter:
    async def assess(self, question: str) -> Complexity:
        if _COMPLEX.search(question) or question.count("?") >= 2:
            return Complexity.COMPLEX
        return Complexity.SIMPLE
