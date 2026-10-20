"""Port for a chat language model. Agents depend on this, never on a provider SDK, so the model
(OpenAI, Groq, a local vLLM...) can change without touching agent code."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(Exception):
    """The model could not produce an answer (outage, quota, timeout, refusal to respond).
    The message must never contain prompt or document text."""


class ChatModel(Protocol):
    @property
    def model_name(self) -> str: ...

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int = 800, temperature: float = 0.0
    ) -> str:
        """Return the model's reply text. Raises LLMError on any failure."""
        ...
