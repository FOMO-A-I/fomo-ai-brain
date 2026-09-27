"""Validation for API messages and agent requests."""

from typing import Any

from fomo.brain.context import Message


def validate_messages(value: Any) -> list[Message]:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise ValueError("messages must be an array of 1–32 items")
    messages = []
    total = 0
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"role", "content"}:
            raise ValueError("each message must have only role and content")
        if raw["role"] not in ("system", "user", "assistant"):
            raise ValueError("only system, user and assistant messages are accepted")
        message = Message(raw["role"], raw["content"])
        total += len(message.content)
        if total > 32_000:
            raise ValueError("combined message text exceeds 32,000 characters")
        messages.append(message)
    if messages[-1].role != "user":
        raise ValueError("last message must be from the user")
    return messages


def validate_task(value: Any) -> tuple[str, list[dict[str, str]]]:
    if not isinstance(value, dict) or set(value) - {"prompt", "sources"} or "prompt" not in value:
        raise ValueError("expected prompt and optional sources")
    prompt = value["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10_000:
        raise ValueError("prompt must contain 1–10,000 characters")
    sources = value.get("sources", [])
    if not isinstance(sources, list) or len(sources) > 20:
        raise ValueError("sources must be an array of up to 20 items")
    for item in sources:
        if not isinstance(item, dict) or set(item) != {"title", "content"}:
            raise ValueError("each source needs title and content")
        if not isinstance(item["title"], str) or not isinstance(item["content"], str):
            raise ValueError("source title and content must be text")
        if not item["title"].strip() or len(item["title"]) > 200 or len(item["content"]) > 4000:
            raise ValueError("source exceeds length limit or has no title")
    return prompt.strip(), sources