"""Strict JSON-based task planning over the shared model backend."""

from __future__ import annotations

import json

from fomo.brain.context import Message
from fomo.brain.model import ModelBackend
from fomo.reasoning.execution_graph import ExecutionGraph, GraphError
from fomo.reasoning.task_decomposition import Task

from .registry import AgentRegistry


class PlanningError(RuntimeError):
    """The model did not return a bounded, valid task plan."""


class Planner:
    def __init__(
        self,
        model: ModelBackend,
        agents: AgentRegistry | None = None,
        *,
        max_tasks: int = 8,
    ) -> None:
        if not 1 <= max_tasks <= 32:
            raise ValueError("max_tasks must be between 1 and 32")
        self.model = model
        self.max_tasks = max_tasks
        self.agents = agents

    def bind_registry(self, agents: AgentRegistry) -> None:
        """Bind planning to the exact registry that will execute the plan."""
        self.agents = agents

    def plan(self, request: str) -> list[Task]:
        if not isinstance(request, str) or not request.strip() or len(request) > 10_000:
            raise ValueError("request must contain 1–10,000 characters")
        if self.agents is None:
            raise PlanningError("planner requires an explicitly configured agent registry")
        if not self.agents.names:
            raise PlanningError("planner cannot plan with an empty agent registry")
        capabilities = self.agents.capabilities
        descriptions = [
            f"{name}: {capabilities[name].role} — {capabilities[name].capability}"
            for name in self.agents.names
            if name in capabilities
        ]
        names = ", ".join(self.agents.names)
        role_contracts = (
            " Registered role contracts:\n" + "\n".join(descriptions)
            if descriptions
            else ""
        )
        messages = [
            Message(
                "system",
                "Create a short, safe execution plan for the user's request. "
                "Return only JSON with this schema: "
                '{"tasks":[{"id":"step-1","description":"...","agent":"general",'
                '"depends_on":[]}]} . Use 1 to '
                f"{self.max_tasks} tasks, unique short IDs, explicit dependencies, and "
                f"only these configured agent names: {names}. "
                "Select agents only for work inside their declared contracts. "
                "All optional non-model tools are disabled unless explicitly listed as enabled; "
                "do not advertise browsing, code execution, filesystem, database, or other "
                "tools that are not enabled. Research may use only caller-supplied sources. "
                "Do not claim tools, files, or external research are available."
                + role_contracts,
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
            if not isinstance(item["agent"], str) or item["agent"] not in self.agents.names:
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
