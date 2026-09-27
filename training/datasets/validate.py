"""Strict validation for instruction-tuning and preference-pair JSONL."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from training.common import DatasetError, normalize_text, read_jsonl, validate_provenance


def _validate_messages(
    messages: Any, *, final_assistant: bool = True, require_user: bool = True
) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise DatasetError("messages must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    saw_user = False
    saw_assistant = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise DatasetError(f"messages[{index}] must be an object")
        role = normalize_text(message.get("role"), f"messages[{index}].role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise DatasetError(f"messages[{index}].role must be system, user, assistant, or tool")
        content = message.get("content")
        if content is None and role == "assistant" and message.get("tool_calls"):
            content = ""
        if not isinstance(content, str):
            raise DatasetError(f"messages[{index}].content must be text")
        content = content.strip()
        if content:
            content = normalize_text(content, f"messages[{index}].content")
        if role != "assistant" or not message.get("tool_calls"):
            if not content:
                raise DatasetError(f"messages[{index}].content must not be empty")
        item = dict(message)
        item["role"] = role
        item["content"] = content
        normalized.append(item)
        saw_user |= role == "user"
        saw_assistant |= role == "assistant"
    if (require_user and not saw_user) or (final_assistant and not saw_assistant):
        raise DatasetError("messages must contain at least one user and one assistant turn")
    if final_assistant and normalized[-1]["role"] != "assistant":
        raise DatasetError("messages must end with an assistant response")
    return normalized


def validate_sft_record(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise DatasetError("SFT record must be an object")
    validate_provenance(row)
    if "messages" in row:
        return {"messages": _validate_messages(row["messages"])}
    instruction = normalize_text(row.get("instruction"), "instruction")
    answer = row.get("response", row.get("output"))
    answer = normalize_text(answer, "response")
    messages = []
    if row.get("system") is not None:
        messages.append({"role": "system", "content": normalize_text(row["system"], "system")})
    messages.extend((
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": answer},
    ))
    return {"messages": messages}


def _validate_conversation_or_text(
    value: Any, field: str, *, is_prompt: bool = False
) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return normalize_text(value, field)
    messages = _validate_messages(
        value, final_assistant=not is_prompt, require_user=is_prompt
    )
    if is_prompt and messages[-1]["role"] != "user":
        raise DatasetError("a conversational DPO prompt must end with a user turn")
    return messages


def validate_dpo_record(
    row: Mapping[str, Any], *, human_preferences: bool = True
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise DatasetError("DPO record must be an object")
    validate_provenance(row, human_preferences=human_preferences)
    prompt = _validate_conversation_or_text(row.get("prompt"), "prompt", is_prompt=True)
    chosen = _validate_conversation_or_text(row.get("chosen"), "chosen")
    rejected = _validate_conversation_or_text(row.get("rejected"), "rejected")
    if chosen == rejected:
        raise DatasetError("chosen and rejected responses must differ")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def validate_file(path: str, kind: str, *, human_preferences: bool = True) -> dict[str, int]:
    if kind not in {"sft", "dpo"}:
        raise DatasetError("kind must be 'sft' or 'dpo'")
    counts = {"records": 0, "possible_secrets": 0}
    for row in read_jsonl(path):
        clean = validate_sft_record(row) if kind == "sft" else validate_dpo_record(
            row, human_preferences=human_preferences
        )
        validate_provenance(row, human_preferences=kind == "dpo" and human_preferences)
        from training.common import find_possible_secrets
        counts["possible_secrets"] += bool(find_possible_secrets(clean))
        counts["records"] += 1
    if counts["records"] == 0:
        raise DatasetError(f"{path}: dataset contains no records")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--kind", choices=("sft", "dpo"), required=True)
    parser.add_argument("--allow-nonhuman-preferences", action="store_true")
    args = parser.parse_args()
    result = validate_file(
        args.input, args.kind,
        human_preferences=not args.allow_nonhuman_preferences,
    )
    print(f"Valid {args.kind} dataset: {result['records']} records")
    if result["possible_secrets"]:
        print(
            f"WARNING: {result['possible_secrets']} records contain likely credential patterns. "
            "Review and remove secrets before training."
        )
    print("Automated checks are not a privacy, consent, license, or safety review.")


if __name__ == "__main__":
    main()