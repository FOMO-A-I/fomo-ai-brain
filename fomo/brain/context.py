"""Validated conversation messages and bounded prompt history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role!r}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("message content must be a non-empty string")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationContext:
    """Maintains a system prompt and at most ``max_turns`` complete turns."""

    def __init__(
        self,
        system_prompt: str,
        *,
        max_turns: int = 12,
        max_message_chars: int = 20_000,
    ) -> None:
        if max_turns < 1 or max_message_chars < 1:
            raise ValueError("context limits must be positive")
        self.system = Message("system", system_prompt)
        self.max_turns = max_turns
        self.max_message_chars = max_message_chars
        self._messages: list[Message] = []

    @property
    def messages(self) -> tuple[Message, ...]:
        return (self.system, *self._messages)

    def add(self, role: str, content: str) -> Message:
        if len(content) > self.max_message_chars:
            raise ValueError("message exceeds the configured character limit")
        message = Message(role, content)
        if role == "system":
            raise ValueError("the system message is fixed for this context")
        self._messages.append(message)
        self._trim()
        return message

    def extend(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.role == "system":
                raise ValueError("additional system messages are not allowed")
            self.add(message.role, message.content)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        # Never trim the system prompt and bound even a malformed one-sided
        # conversation; normal inference appends user/assistant pairs.
        self._messages = self._messages[-2 * self.max_turns :]