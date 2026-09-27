"""Validated chat messages and prompt formatting."""

from dataclasses import dataclass
from typing import Literal, Sequence

Role = Literal["system", "user", "assistant"]

DEFAULT_INSTRUCTIONS = (
    "You are FOMO AI, a helpful engineering assistant. Give accurate, clear answers. "
    "Say when you are uncertain, and never claim you ran code or visited a link unless you did."
)


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError("role must be system, user, or assistant")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be non-empty text")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def format_messages(
    user_text: str,
    history: Sequence[Message] = (),
    *,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> list[Message]:
    """Put system instructions first, followed by completed turns and this request."""
    system = Message("system", instructions)
    user = Message("user", user_text)
    for message in history:
        if not isinstance(message, Message) or message.role not in ("user", "assistant"):
            raise ValueError("history may only contain Message(user/assistant) entries")
    return [system, *history, user]