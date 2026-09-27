"""Explicit registry for agent capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from fomo.reasoning.task_decomposition import Task


class Agent(Protocol):
    def run(self, task: Task, context: Mapping[str, Any]) -> str: ...


class AgentNotFoundError(LookupError):
    """A requested agent capability has not been configured."""


@dataclass(frozen=True, slots=True)
class AgentCapability:
    """Capability contract for a registered agent.

    ``optional_tools`` names integrations that could extend an agent, while
    ``enabled_tools`` lists integrations explicitly supplied by its owner.
    The default registry starts with no non-model tools enabled.
    """

    name: str
    role: str
    capability: str
    model_driven: bool = True
    optional_tools: tuple[str, ...] = ()
    enabled_tools: tuple[str, ...] = ()


class AgentRegistry:
    def __init__(self, agents: Mapping[str, Agent] | None = None) -> None:
        self._agents: dict[str, Agent] = {}
        self._capabilities: dict[str, AgentCapability] = {}
        for name, agent in (agents or {}).items():
            self.register(name, agent)

    def register(
        self,
        name: str,
        agent: Agent,
        *,
        capability: AgentCapability | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("agent name must not be empty")
        if name in self._agents:
            raise ValueError(f"agent {name!r} is already registered")
        if not callable(getattr(agent, "run", None)):
            raise TypeError("agent must implement run(task, context)")
        if capability is not None and capability.name != name:
            raise ValueError("capability name must match the registered agent name")
        self._agents[name] = agent
        if capability is not None:
            self._capabilities[name] = capability

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

    @property
    def capabilities(self) -> Mapping[str, AgentCapability]:
        """Return a copy of the capability contracts declared at registration."""
        return dict(self._capabilities)

    def capability(self, name: str) -> AgentCapability | None:
        return self._capabilities.get(name)


def create_default_registry(model: Any) -> AgentRegistry:
    """Create the explicit 36-role registry over one shared model backend.

    These are role prompts over the configured model, not claims of 36
    separately trained models. Non-model integrations remain disabled.
    """
    if not callable(getattr(model, "complete", None)):
        raise TypeError("model must implement complete(messages)")
    from .coding_agent import CodingAgent
    from .research_agent import ResearchAgent
    from .specialists import AGENT_CAPABILITIES, ModelRoleAgent
    from .verification_agent import VerificationAgent

    specialized: dict[str, Agent] = {
        "research": ResearchAgent(model),
        "coding": CodingAgent(model),
        "verification": VerificationAgent(model),
    }
    registry = AgentRegistry()
    for capability in AGENT_CAPABILITIES:
        agent = specialized.get(capability.name)
        if agent is None:
            agent = ModelRoleAgent(model, capability)
        registry.register(capability.name, agent, capability=capability)
    return registry