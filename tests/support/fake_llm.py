"""A scripted chat model for tests: no network, records every prompt it receives."""

from collections.abc import Sequence

from ekip_contracts.llm import ChatMessage, LLMError


class FakeChatModel:
    def __init__(
        self, reply: str = "Torque is 140 Nm [1].", error: Exception | None = None
    ) -> None:
        self.reply, self.error = reply, error
        self.calls: list[list[ChatMessage]] = []
        self.kwargs: list[dict[str, float]] = []

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int = 800, temperature: float = 0.0
    ) -> str:
        self.calls.append(list(messages))
        self.kwargs.append({"max_tokens": max_tokens, "temperature": temperature})
        if self.error:
            raise self.error
        return self.reply


def outage() -> LLMError:
    return LLMError("provider unavailable")
