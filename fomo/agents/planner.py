"""Strict JSON-based task planning over the shared model backend."""

from __future__ import annotations

import json
from typing import Any

from fomo.brain.context import Message
from fomo.brain.model import ModelBackend
from fomo.reasoning.execution_graph import ExecutionGraph, GraphError
from fomo.reasoning.task_decomposition import AGENT_NAMES, Task


class PlanningError(RuntimeError):
    """The model did not return a bounded, valid task plan."""


class Planner:
    def __init__(self, model: ModelBackend, *, max_tasks: int = 8) -> None:
        if not 1 <= max_tasks <= 32:
            raise ValueError("max_tasks must be between 1 and 32")
        self.model = model
        self.max_tasks = max_tasks

    def plan(self, request: str) -> list[Task]:
        if not isinstance(request, str) or not request.strip() or len(request) > 10_000:
            raise ValueError("request must contain 1–10,000 characters")
        messages = [
            Message(
                "system",
                "Create a short, safe execution plan for the user's request. "
                "Return only JSON with this schema: "
                '{"tasks":[{"id":"step-1","description":"...","agent":"general",'
                '"depends_on":[]}]} . Use 1 to '
                f"{self.max_tasks} tasks, unique short IDs, explicit dependencies, and "
                "only these agent names: general, research, coding, verification. "
                "Do not claim tools, files, or research are available unless supplied.",
            ),
            Message("user", request.strip()),
        ]
        raw = self.model.complete(messages)
        if not isinstance(raw, str) or not raw.strip() or len(raw) > 20_000:
            raise PlanningError("planner returned an empty or oversized response")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanningError("planner response is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"tasks"}:
            raise PlanningError("planner response must contain only a 'tasks' array")
        items = payload["tasks"]
        if not isinstance(items, list) or not 1 <= len(items) <= self.max_tasks:
            raise PlanningError(f"plan must contain 1 to {self.max_tasks} tasks")

        tasks: list[Task] = []
        for item in items:
            if not isinstance(item, dict) or set(item) != {
                "id",
                "description",
                "agent",
                "depends_on",
            }:
                raise PlanningError("each task must include exactly id, description, agent, depends_on")
            if not isinstance(item["agent"], str) or item["agent"] not in AGENT_NAMES:
                raise PlanningError(f"unsupported planned agent: {item['agent']!r}")
            dependencies = item["depends_on"]
            if not isinstance(dependencies, list) or not all(
                isinstance(value, str) for value in dependencies
            ):
                raise PlanningError("task dependencies must be an array of task IDs")
            try:
                tasks.append(
                    Task(
                        id=item["id"],
                        description=item["description"],
                        agent=item["agent"],
                        depends_on=tuple(dependencies),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise PlanningError(f"invalid task in plan: {exc}") from exc
        try:
            ExecutionGraph(tasks)
        except GraphError as exc:
            raise PlanningError(f"invalid task dependency graph: {exc}") from exc
        return tasks
