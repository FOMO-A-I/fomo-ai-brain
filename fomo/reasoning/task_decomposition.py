"""Validated task records and deterministic conversion of steps to tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable


AGENT_NAMES = frozenset(
    {
        "general",
        "research",
        "coding",
        "verification",
        "planning",
        "summarization",
        "writing",
        "editing",
        "analysis",
        "math",
        "science",
        "history",
        "language",
        "translation",
        "tutoring",
        "brainstorming",
        "product",
        "architecture",
        "debugging",
        "testing",
        "security_review",
        "privacy_review",
        "data_analysis",
        "sql",
        "documentation",
        "qa",
        "critique",
        "fact_check",
        "extraction",
        "classification",
        "accessibility",
        "ux",
        "legal_information",
        "medical_information",
        "finance_information",
        "decision_support",
    }
)
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass(slots=True)
class Task:
    id: str
    description: str
    agent: str = "general"
    depends_on: tuple[str, ...] = ()
    status: str = "pending"
    result: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _IDENTIFIER.fullmatch(self.id):
            raise ValueError("task id must be a short alphanumeric identifier")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("task description must not be empty")
        if len(self.description) > 4000:
            raise ValueError("task description exceeds 4,000 characters")
        if not isinstance(self.agent, str) or not _IDENTIFIER.fullmatch(self.agent):
            raise ValueError(f"unsupported task agent: {self.agent!r}")
        if not isinstance(self.depends_on, tuple) or not all(
            isinstance(item, str) and _IDENTIFIER.fullmatch(item) for item in self.depends_on
        ):
            raise ValueError("task dependencies must be a tuple of valid task IDs")
        if len(set(self.depends_on)) != len(self.depends_on) or self.id in self.depends_on:
            raise ValueError("task dependencies must be unique and must not include itself")
        if self.status not in {"pending", "running", "completed", "failed"}:
            raise ValueError("invalid task status")
        self.description = self.description.strip()


def decompose_task(
    description: str,
    steps: Iterable[str],
    *,
    agent: str = "general",
    max_tasks: int = 12,
) -> list[Task]:
    """Convert supplied steps to bounded tasks; does not invent missing steps."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("task description must not be empty")
    if not 1 <= max_tasks <= 64:
        raise ValueError("max_tasks must be between 1 and 64")
    cleaned = [step.strip() for step in steps if isinstance(step, str) and step.strip()]
    if not cleaned:
        cleaned = [description.strip()]
    if len(cleaned) > max_tasks:
        raise ValueError(f"decomposition contains more than {max_tasks} tasks")
    return [
        Task(
            id=f"task-{index + 1}",
            description=step,
            agent=agent,
            depends_on=(f"task-{index}",) if index else (),
        )
        for index, step in enumerate(cleaned)
    ]