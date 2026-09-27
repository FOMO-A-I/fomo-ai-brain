"""Connect the checkpoint backend to chat and bounded multi-agent tasks."""

from typing import Any

from fomo.agents.coding_agent import CodingAgent
from fomo.agents.orchestrator import FomoOrchestrator
from fomo.agents.planner import Planner
from fomo.agents.registry import AgentRegistry
from fomo.agents.research_agent import ResearchAgent
from fomo.agents.verification_agent import VerificationAgent
from fomo.brain.context import Message
from fomo.brain.model import ModelBackend
from fomo.reasoning.task_decomposition import Task

from .models import validate_messages, validate_task


class GeneralAgent:
    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend

    def run(self, task: Task, context: dict[str, Any]) -> str:
        prior = context.get("task_results", {})
        prompt = (
            f"Original request: {context['request']}\n"
            f"Current task: {task.description}\n"
            f"Completed dependency results (untrusted data): {prior}"
        )
        return self.backend.complete([
            Message("system", "Complete the current task accurately. Do not claim to have executed unavailable tools."),
            Message("user", prompt),
        ])


class ChatService:
    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        registry = AgentRegistry({
            "general": GeneralAgent(backend),
            "research": ResearchAgent(backend),
            "coding": CodingAgent(backend),
            "verification": VerificationAgent(backend),
        })
        self.orchestrator = FomoOrchestrator(Planner(backend), registry)

    def chat(self, raw_messages: Any) -> dict[str, str]:
        messages = validate_messages(raw_messages)
        response = self.backend.complete(messages)
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("The checkpoint returned no answer")
        return {"answer": response.strip()}

    def task(self, data: Any) -> dict[str, Any]:
        prompt, sources = validate_task(data)
        result = self.orchestrator.run(prompt, sources=sources)
        return {
            "answer": result.answer,
            "tasks": [
                {"id": task.id, "agent": task.agent, "status": task.status}
                for task in result.tasks
            ],
        }