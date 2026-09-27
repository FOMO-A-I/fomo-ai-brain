import json
import unittest

from fomo.agents.coding_agent import CodingAgent
from fomo.agents.orchestrator import FomoOrchestrator, OrchestrationError
from fomo.agents.planner import Planner, PlanningError
from fomo.agents.registry import AgentNotFoundError, AgentRegistry
from fomo.agents.research_agent import ResearchAgent
from fomo.agents.verification_agent import VerificationAgent
from fomo.reasoning.task_decomposition import Task


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
        tasks = Planner(backend).plan("Research and verify this")
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
                Planner(ScriptedBackend([response])).plan("Do something")

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
        with self.assertRaisesRegex(OrchestrationError, "not configured"):
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