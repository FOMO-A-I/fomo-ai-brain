"""Explicit task-based model routing with no implicit provider fallback."""

from __future__ import annotations

from collections.abc import Mapping

from .model import ModelBackend


class ModelRouter:
    """Select an injected backend by task name, otherwise use the explicit default."""

    def __init__(
        self,
        default: ModelBackend,
        *,
        task_models: Mapping[str, ModelBackend] | None = None,
    ) -> None:
        self._default = default
        self._task_models = dict(task_models or {})
        if any(not isinstance(name, str) or not name.strip() for name in self._task_models):
            raise ValueError("task route names must be non-empty strings")

    def for_task(self, task: str) -> ModelBackend:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task name must not be empty")
        return self._task_models.get(task, self._default)