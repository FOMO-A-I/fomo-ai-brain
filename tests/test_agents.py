import json
import unittest

from fomo.agents.coding_agent import CodingAgent
from fomo.agents.orchestrator import FomoOrchestrator, OrchestrationError
from fomo.agents.planner import Planner, PlanningError
from fomo.agents.registry import (
    AgentNotFoundError,
    AgentRegistry,
    create_default_registry,
)
from fomo.agents.research_agent import ResearchAgent
from fomo.agents.verification_agent import VerificationAgent
from fomo.reasoning.task_decomposition import AGENT_NAMES, Task


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages):
        self.requests.append(tuple(messages))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class EchoAgent:
    def run(self, task, context):
        previous = context.get("task_results", {})
        return task.description + (f" | prior={next(iter(previous.values()))}" if previous else "")


class FixedAgent:
    def __init__(self, response):
        self.response = response

    def run(self, task, context):
        return self.response


class PlannerAgentTests(unittest.TestCase):
    @staticmethod
    def _planner_registry():
        return AgentRegistry({name: EchoAgent() for name in AGENT_NAMES})

    def test_planner_parses_a_valid_bounded_plan(self):
        response = json.dumps(
            {
                "tasks": [
                    {
                        "id": "research",
                        "description": "Summarize the supplied source",
                        "agent": "research",
                        "depends_on": [],
                    },
                    {
                        "id": "verify",
                        "description": "Review the summary",
                        "agent": "verification",
                        "depends_on": ["research"],
                    },
                ]
            }
        )
        backend = ScriptedBackend([response])
        tasks = Planner(backend, self._planner_registry()).plan("Research and verify this")
        self.assertEqual([task.agent for task in tasks], ["research", "verification"])
        self.assertEqual(tasks[1].depends_on, ("research",))
        self.assertIn("Return only JSON", backend.requests[0][0].content)

    def test_planner_rejects_invalid_json_unknown_agents_cycles_and_empty_plan(self):
        for response in (
            "not json",
            '{"tasks":[]}',
            json.dumps(
                {
                    "tasks": [
                        {
                            "id": "one",
                            "description": "Do a thing",
                            "agent": "shell",
                            "depends_on": [],
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "tasks": [
                        {
                            "id": "one",
                            "description": "One",
                            "agent": "general",
                            "depends_on": ["two"],
                        },
                        {
                            "id": "two",
                            "description": "Two",
                            "agent": "general",
                            "depends_on": ["one"],
                        },
                    ]
                }
            ),
        ):
            with self.subTest(response=response), self.assertRaises(PlanningError):
                Planner(
                    ScriptedBackend([response]), self._planner_registry()
                ).plan("Do something")

    def test_default_registry_has_exactly_36_honest_role_contracts(self):
        backend = ScriptedBackend(["role response"])
        registry = create_default_registry(backend)
        self.assertEqual(len(registry.names), 36)
        self.assertEqual(set(registry.names), AGENT_NAMES)
        self.assertEqual(set(registry.capabilities), AGENT_NAMES)
        for name, contract in registry.capabilities.items():
            with self.subTest(agent=name):
                self.assertTrue(contract.role)
                self.assertTrue(contract.capability)
                self.assertTrue(contract.model_driven)
                self.assertEqual(contract.enabled_tools, ())
                self.assertIs(registry.get(name).model, backend)
        self.assertIn("web_browsing", registry.capability("research").optional_tools)
        self.assertNotIn("web_browsing", registry.capability("research").enabled_tools)

    def test_planner_requires_an_explicit_registry(self):
        with self.assertRaisesRegex(PlanningError, "explicitly configured agent registry"):
            Planner(ScriptedBackend(["unused"])).plan("Do a task")

    def test_specialist_runs_shared_model_under_its_declared_role_contract(self):
        backend = ScriptedBackend(["1. Inspect\n2. Implement"])
        agent = create_default_registry(backend).get("planning")
        result = agent.run(Task("steps", "Plan the feature", agent="planning"), {})
        self.assertIn("1. Inspect", result)
        self.assertIn("Task planner", backend.requests[0][0].content)
        self.assertIn(
            "optional integrations not explicitly configured are disabled",
            backend.requests[0][0].content,
        )

    def test_planner_enforces_the_configured_task_bound(self):
        response = json.dumps(
            {
                "tasks": [
                    {"id": "one", "description": "First", "agent": "general", "depends_on": []},
                    {"id": "two", "description": "Second", "agent": "general", "depends_on": []},
                ]
            }
        )
        with self.assertRaisesRegex(PlanningError, "1 to 1 tasks"):
            Planner(
                ScriptedBackend([response]),
                AgentRegistry({"general": EchoAgent()}),
                max_tasks=1,
            ).plan("Do two things")

    def test_planner_advertises_and_accepts_only_configured_registry_names(self):
        registry = AgentRegistry({"general": EchoAgent()})
        response = json.dumps(
            {
                "tasks": [
                    {
                        "id": "one",
                        "description": "Answer",
                        "agent": "general",
                        "depends_on": [],
                    }
                ]
            }
        )
        backend = ScriptedBackend([response])
        self.assertEqual(Planner(backend, registry).plan("Answer")[0].agent, "general")
        self.assertIn("only these configured agent names: general", backend.requests[0][0].content)
        unsupported = json.dumps(
            {
                "tasks": [
                    {
                        "id": "one",
                        "description": "Browse the web",
                        "agent": "research",
                        "depends_on": [],
                    }
                ]
            }
        )
        with self.assertRaisesRegex(PlanningError, "unsupported planned agent"):
            Planner(ScriptedBackend([unsupported]), registry).plan("Browse the web")

    def test_registry_requires_explicit_capabilities(self):
        registry = AgentRegistry({"general": EchoAgent()})
        self.assertEqual(registry.names, ("general",))
        with self.assertRaises(AgentNotFoundError):
            registry.get("coding")
        with self.assertRaises(ValueError):
            registry.register("general", EchoAgent())

    def test_research_agent_uses_only_supplied_sources(self):
        backend = ScriptedBackend(["Claim supported by [S1]."])
        agent = ResearchAgent(backend)
        task = Task("research", "Summarize the claim", agent="research")
        answer = agent.run(
            task,
            {"sources": [{"title": "Report", "content": "The report says the value is 4."}]},
        )
        self.assertIn("[S1]", answer)
        self.assertIn("Report", backend.requests[0][1].content)
        with self.assertRaisesRegex(ValueError, "caller-supplied sources"):
            agent.run(task, {"sources": []})

    def test_coding_agent_does_not_claim_execution(self):
        backend = ScriptedBackend(["def add(a, b):\n    return a + b"])
        answer = CodingAgent(backend).run(
            Task("code", "Write an add function", agent="coding"), {}
        )
        self.assertIn("Never claim to have executed code", backend.requests[0][0].content)
        self.assertIn("def add", answer)

    def test_verification_agent_parses_review_and_rejects_bad_model_response(self):
        agent = VerificationAgent(ScriptedBackend(['{"approved":true,"confidence":0.9,"issues":[]}']))
        report = agent.review("The code and tests satisfy the stated criteria.", criteria="Check it")
        self.assertTrue(report.approved)
        self.assertEqual(report.confidence, 0.9)
        broken = VerificationAgent(ScriptedBackend(["yes"]))
        with self.assertRaisesRegex(RuntimeError, "valid JSON"):
            broken.review("Candidate")
        self.assertFalse(agent.review("").approved)

    def test_orchestrator_executes_plan_with_dependencies_and_preserves_final_answer(self):
        plan = json.dumps(
            {
                "tasks": [
                    {
                        "id": "draft",
                        "description": "Draft response",
                        "agent": "general",
                        "depends_on": [],
                    },
                    {
                        "id": "review",
                        "description": "Check response",
                        "agent": "verification",
                        "depends_on": ["draft"],
                    },
                ]
            }
        )
        review = json.dumps({"approved": True, "confidence": 0.8, "issues": []})
        general = EchoAgent()
        verifier = VerificationAgent(ScriptedBackend([review]))
        orchestrator = FomoOrchestrator(
            Planner(ScriptedBackend([plan])),
            AgentRegistry({"general": general, "verification": verifier}),
        )
        result = orchestrator.run("Answer the question")
        self.assertEqual(result.answer, "Draft response")
        self.assertEqual([task.status for task in result.tasks], ["completed", "completed"])
        self.assertTrue(json.loads(result.outputs["review"])["approved"])

    def test_orchestrator_fails_explicitly_when_agent_is_not_registered(self):
        plan = json.dumps(
            {
                "tasks": [
                    {
                        "id": "code",
                        "description": "Write code",
                        "agent": "coding",
                        "depends_on": [],
                    }
                ]
            }
        )
        orchestrator = FomoOrchestrator(
            Planner(ScriptedBackend([plan])), AgentRegistry({"general": EchoAgent()})
        )
        with self.assertRaisesRegex(PlanningError, "unsupported planned agent"):
            orchestrator.run("Write code")

    def _orchestrator_with_report(self, report):
        plan = json.dumps(
            {
                "tasks": [
                    {
                        "id": "draft",
                        "description": "Draft response",
                        "agent": "general",
                        "depends_on": [],
                    },
                    {
                        "id": "review",
                        "description": "Review draft",
                        "agent": "verification",
                        "depends_on": ["draft"],
                    },
                ]
            }
        )
        return FomoOrchestrator(
            Planner(ScriptedBackend([plan])),
            AgentRegistry({"general": EchoAgent(), "verification": FixedAgent(report)}),
        )

    def test_orchestrator_rejects_unapproved_verification_instead_of_returning_answer(self):
        report = json.dumps({"approved": False, "confidence": 0.95, "issues": ["unsafe"]})
        with self.assertRaisesRegex(OrchestrationError, "rejected"):
            self._orchestrator_with_report(report).run("Answer the question")

    def test_orchestrator_requires_valid_verification_schema_and_confidence(self):
        reports = (
            "not JSON",
            json.dumps({"approved": "true", "confidence": 0.9, "issues": []}),
            json.dumps({"approved": True, "confidence": 0.2, "issues": []}),
            json.dumps({"approved": True, "confidence": 0.9, "issues": ["x" * 1001]}),
        )
        for report in reports:
            with self.subTest(report=report[:80]), self.assertRaises(OrchestrationError):
                self._orchestrator_with_report(report).run("Answer the question")


if __name__ == "__main__":
    unittest.main()