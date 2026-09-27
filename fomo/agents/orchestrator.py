"""Bounded plan/execute orchestration using only explicitly registered agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from fomo.reasoning.execution_graph import ExecutionGraph, GraphError
from fomo.reasoning.retry_policy import RetryPolicy
from fomo.reasoning.task_decomposition import Task
from fomo.reasoning.verification import verify_result

from .planner import Planner
from .registry import AgentRegistry

MIN_VERIFICATION_CONFIDENCE = 0.5


class OrchestrationError(RuntimeError):
    """A planned task failed or could not be safely completed."""


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    answer: str
    tasks: tuple[Task, ...]
    outputs: Mapping[str, str]


class FomoOrchestrator:
    def __init__(
        self,
        planner: Planner,
        agents: AgentRegistry,
        *,
        max_tasks: int = 8,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if not 1 <= max_tasks <= 32:
            raise ValueError("max_tasks must be between 1 and 32")
        self.planner = planner
        self.agents = agents
        self.max_tasks = max_tasks
        # Inference has no side effects, but default to one attempt so transient
        # infrastructure errors are visible and are not silently masked.
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=1)

    def run(
        self,
        request: str,
        *,
        sources: Sequence[Mapping[str, str]] = (),
    ) -> OrchestrationResult:
        if not isinstance(request, str) or not request.strip() or len(request) > 10_000:
            raise ValueError("request must contain 1–10,000 characters")
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            raise ValueError("sources must be a sequence")
        tasks = self.planner.plan(request)
        if not tasks:
            raise OrchestrationError("planner returned an empty task plan")
        if len(tasks) > self.max_tasks:
            raise OrchestrationError(f"plan exceeds the {self.max_tasks}-task execution limit")
        try:
            graph = ExecutionGraph(tasks)
        except GraphError as exc:
            raise OrchestrationError(f"invalid task graph: {exc}") from exc

        outputs: dict[str, str] = {}
        base_context: dict[str, Any] = {"request": request, "sources": tuple(sources)}
        while not graph.finished:
            ready = graph.ready()
            if not ready:
                blocked = [task.id for task in tasks if task.status == "pending"]
                raise OrchestrationError(f"tasks are blocked by failed dependencies: {blocked}")
            for planned in ready:
                task = graph.start(planned.id)
                context = {
                    **base_context,
                    "task_results": {
                        dependency: outputs[dependency] for dependency in task.depends_on
                    },
                    "candidate": outputs.get(task.depends_on[-1]) if task.depends_on else None,
                }
                try:
                    agent = self.agents.get(task.agent)
                    output = self.retry_policy.run(lambda: agent.run(task, context))
                    checked = verify_result(output)
                    if not checked.passed:
                        raise OrchestrationError("; ".join(checked.issues))
                    if task.agent == "verification":
                        self._validate_verification_report(output)
                    graph.complete(task.id, output)
                    outputs[task.id] = output.strip()
                except Exception as exc:
                    graph.fail(task.id, str(exc))
                    raise OrchestrationError(f"task {task.id!r} ({task.agent}) failed: {exc}") from exc

        answer_task = next(
            (task for task in reversed(tasks) if task.agent != "verification"),
            tasks[-1],
        )
        answer = outputs.get(answer_task.id)
        if not answer:
            raise OrchestrationError("completed plan produced no answer")
        return OrchestrationResult(answer, tuple(tasks), dict(outputs))

    @staticmethod
    def _validate_verification_report(output: str) -> None:
        try:
            report = json.loads(output)
        except json.JSONDecodeError as exc:
            raise OrchestrationError("verification agent returned invalid JSON") from exc
        if not isinstance(report, dict) or set(report) != {
            "approved",
            "confidence",
            "issues",
        }:
            raise OrchestrationError("verification report has an invalid schema")
        approved = report["approved"]
        confidence = report["confidence"]
        issues = report["issues"]
        if type(approved) is not bool:
            raise OrchestrationError("verification approval must be a boolean")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise OrchestrationError("verification confidence must be numeric")
        if not 0 <= confidence <= 1:
            raise OrchestrationError("verification confidence must be between 0 and 1")
        if confidence < MIN_VERIFICATION_CONFIDENCE:
            raise OrchestrationError(
                f"verification confidence is below {MIN_VERIFICATION_CONFIDENCE}"
            )
        if (
            not isinstance(issues, list)
            or len(issues) > 20
            or not all(isinstance(issue, str) and len(issue) <= 1000 for issue in issues)
        ):
            raise OrchestrationError("verification issues must be a bounded list of strings")
        if not approved:
            raise OrchestrationError("verification agent rejected the result")