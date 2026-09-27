"""Provider-neutral single-turn inference helpers."""

from __future__ import annotations

from collections.abc import Sequence

from .context import ConversationContext, Message
from .model import ModelBackend


def infer(
    backend: ModelBackend,
    prompt: str,
    *,
    context: ConversationContext | None = None,
    system_prompt: str = "You are FOMO AI. Be accurate and state uncertainty clearly.",
) -> str:
    """Run one turn and commit it to context only after successful inference."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must not be empty")
    active = context or ConversationContext(system_prompt)
    messages: Sequence[Message] = (*active.messages, Message("user", prompt))
    answer = backend.complete(messages)
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("model backend returned an empty or invalid completion")
    if len(prompt) > active.max_message_chars or len(answer.strip()) > active.max_message_chars:
        raise ValueError("turn exceeds the context's per-message character limit")
    active.add("user", prompt)
    active.add("assistant", answer.strip())
    return answer.strip()