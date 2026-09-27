"""Conversation orchestration around a real model provider."""

from typing import Protocol, Sequence

from .messages import DEFAULT_INSTRUCTIONS, Message, format_messages


class ChatProvider(Protocol):
    def complete(self, messages: Sequence[Message]) -> str: ...


class FomoBrain:
    def __init__(
        self,
        provider: ChatProvider,
        *,
        instructions: str = DEFAULT_INSTRUCTIONS,
        max_turns: int = 8,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        Message("system", instructions)
        self.provider = provider
        self.instructions = instructions
        self.max_turns = max_turns
        self._history: list[Message] = []

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    def ask(self, text: str) -> str:
        request = format_messages(text, self._history, instructions=self.instructions)
        answer = self.provider.complete(request)
        assistant = Message("assistant", answer)
        # Only commit a complete turn after successful inference and validation.
        self._history = (self._history + [request[-1], assistant])[-2 * self.max_turns:]
        return assistant.content

    def reset(self) -> None:
        self._history.clear()