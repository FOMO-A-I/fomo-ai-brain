"""A bounded, in-process message context; this is not durable memory."""

from __future__ import annotations

from collections import deque
from threading import RLock
from typing import Iterable


class BoundedContext:
    """Keep the newest turns within explicit message and character limits."""

    def __init__(
        self,
        max_messages: int = 24,
        max_characters: int = 32_000,
        *,
        system_message: str | None = None,
    ) -> None:
        if max_messages < 1 or max_characters < 1:
            raise ValueError("context limits must be positive")
        if system_message is not None and len(system_message) > max_characters:
            raise ValueError("system message exceeds the context character limit")
        self.max_messages = max_messages
        self.max_characters = max_characters
        self._system_message = system_message
        self._messages: deque[dict[str, str]] = deque()
        self._characters = len(system_message or "")
        self._lock = RLock()

    @property
    def character_count(self) -> int:
        with self._lock:
            return self._characters

    def append(self, role: str, content: str) -> None:
        if role not in {"user", "assistant", "tool"}:
            raise ValueError("role must be user, assistant, or tool")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be non-empty text")
        if len(content) + len(self._system_message or "") > self.max_characters:
            raise ValueError("message and system prompt exceed the context character limit")
        with self._lock:
            self._messages.append({"role": role, "content": content})
            self._characters += len(content)
            while (
                len(self._messages) > self.max_messages
                or self._characters > self.max_characters
            ):
                removed = self._messages.popleft()
                self._characters -= len(removed["content"])

    def extend(self, messages: Iterable[dict[str, str]]) -> None:
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("each message must contain only role and content")
            self.append(message["role"], message["content"])

    def snapshot(self) -> list[dict[str, str]]:
        with self._lock:
            messages = [dict(message) for message in self._messages]
            if self._system_message is not None:
                return [{"role": "system", "content": self._system_message}, *messages]
            return messages

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self._characters = len(self._system_message or "")