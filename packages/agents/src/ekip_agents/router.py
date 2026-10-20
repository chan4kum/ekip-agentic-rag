"""Decide whether a question should be answered from the documents.

`QueryRouter` is the port. `RuleBasedRouter` is the first, deterministic adapter: no model, no
network, same answer every time. An LLM-based router can replace it later without touching the
graph, and this one stays useful as a fast pre-filter and a test double.

Bias: when unsure, answer from the documents. That path is safe (retrieval enforces access
control) and a wrongly refused real question costs the user more than one wasted search.
"""

import re
from typing import Protocol

from ekip_agents.state import Route


class QueryRouter(Protocol):
    async def classify(self, question: str) -> Route: ...


_SMALL_TALK = re.compile(
    r"""^\W*(
        hi|hello|hey|hiya|yo|thanks|thank\s+you|thx|ok|okay|cool|great|bye|goodbye|good\s+(morning|afternoon|evening|night)
        |how\s+are\s+you|who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|tell\s+me\s+a\s+joke
        |what('?s|\s+is)\s+your\s+name
    )(\s+(there|team|claude|assistant))?\W*$""",
    re.IGNORECASE | re.VERBOSE,
)
_HAS_CONTENT = re.compile(r"[^\W_]", re.UNICODE)  # at least one letter or digit


class RuleBasedRouter:
    async def classify(self, question: str) -> Route:
        text = question.strip()
        if not _HAS_CONTENT.search(text):  # "???", "...", emoji only
            return Route.OUT_OF_SCOPE
        if _SMALL_TALK.match(text):
            return Route.OUT_OF_SCOPE
        return Route.KNOWLEDGE
