"""Small dependency graph with guarded state transitions."""

from __future__ import annotations

from collections.abc import Iterable

from .task_decomposition import Task


class GraphError(RuntimeError):
    """The execution graph is invalid or a state transition is unsafe."""


class ExecutionGraph:
    def __init__(self, tasks: Iterable[Task]) -> None:
        task_list = list(tasks)
        self.tasks = {task.id: task for task in task_list}
        if len(self.tasks) != len(task_list):
            raise GraphError("task identifiers must be unique")
        for task in task_list:
            missing = set(task.depends_on) - self.tasks.keys()
            if missing:
                raise GraphError(f"task {task.id!r} has unknown dependencies: {sorted(missing)}")
        self._assert_acyclic()

    def ready(self) -> tuple[Task, ...]:
        return tuple(
            task
            for task in self.tasks.values()
            if task.status == "pending"
            and all(self.tasks[dependency].status == "completed" for dependency in task.depends_on)
        )

    def start(self, task_id: str) -> Task:
        task = self._get(task_id)
        if task.status != "pending" or task not in self.ready():
            raise GraphError(f"task {task_id!r} is not ready to start")
        task.status = "running"
        return task

    def complete(self, task_id: str, result: str) -> Task:
        task = self._get(task_id)
        if task.status != "running":
            raise GraphError(f"task {task_id!r} is not running")
        if not isinstance(result, str) or not result.strip():
            raise GraphError("task completion must include a non-empty result")
        task.result = result.strip()
        task.error = None
        task.status = "completed"
        return task

    def fail(self, task_id: str, error: str) -> Task:
        task = self._get(task_id)
        if task.status != "running":
            raise GraphError(f"task {task_id!r} is not running")
        task.error = str(error)[:1000]
        task.status = "failed"
        return task

    @property
    def finished(self) -> bool:
        return all(task.status in {"completed", "failed"} for task in self.tasks.values())

    def _get(self, task_id: str) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise GraphError(f"unknown task: {task_id!r}") from exc

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise GraphError("task dependencies contain a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in self.tasks[task_id].depends_on:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in self.tasks:
            visit(task_id)