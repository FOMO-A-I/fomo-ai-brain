"""Explicit registry for agent capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from fomo.reasoning.task_decomposition import Task


class Agent(Protocol):
    def run(self, task: Task, context: Mapping[str, Any]) -> str: ...


class AgentNotFoundError(LookupError):
    """A requested agent capability has not been configured."""


class AgentRegistry:
    def __init__(self, agents: Mapping[str, Agent] | None = None) -> None:
        self._agents: dict[str, Agent] = {}
        for name, agent in (agents or {}).items():
            self.register(name, agent)

    def register(self, name: str, agent: Agent) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("agent name must not be empty")
        if name in self._agents:
            raise ValueError(f"agent {name!r} is already registered")
        if not callable(getattr(agent, "run", None)):
            raise TypeError("agent must implement run(task, context)")
        self._agents[name] = agent

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise AgentNotFoundError(
                f"agent {name!r} is not configured; register it explicitly before planning"
            ) from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._agents))