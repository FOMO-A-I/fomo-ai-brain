import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fomo.api.chat import ChatService
from fomo.api.server import make_server
from fomo.memory.retrieval import MemoryRetriever
from fomo.memory.vector_store import SQLiteVectorStore
from fomo.tools.python_tool import SandboxedPythonTool
from fomo.tools.research_tool import SafeWebResearchProvider
from fomo.tools.sandbox import SandboxClient


class RuntimeBackend:
    def __init__(self, agent="general"):
        self.agent = agent
        self.completions = []

    def complete(self, messages):
        self.completions.append(messages)
        if "Create a short, safe execution plan" in messages[0].content:
            return json.dumps(
                {
                    "tasks": [
                        {
                            "id": "task",
                            "description": "Perform the requested work",
                            "agent": self.agent,
                            "depends_on": [],
                        }
                    ]
                }
            )
        if self.agent == "research":
            return "Evidence-based answer [S1]"
        return "A safe model response"


class FakeExternalSandbox(SandboxClient):
    def __init__(self):
        self.calls = []

    def execute(self, code, *, timeout_seconds=5):
        self.calls.append((code, timeout_seconds))
        return {"exit_code": 0, "stdout": "sandbox result", "stderr": ""}


class FakeResearchProvider(SafeWebResearchProvider):
    def __init__(self):
        super().__init__()
        self.calls = []

    def fetch(self, urls, *, approved_urls=(), approved_domains=()):
        self.calls.append((tuple(urls), tuple(approved_urls), tuple(approved_domains)))
        return [{"title": "Approved source", "content": "Verified public evidence", "url": urls[0]}]


class RuntimeIntegrationTests(unittest.TestCase):
    def test_planner_prompt_matches_live_tool_registry(self):
        disabled_backend = RuntimeBackend(agent="research")
        disabled = ChatService(disabled_backend)
        disabled.orchestrator.planner.plan("Summarize supplied sources")
        disabled_prompt = disabled_backend.completions[0][0].content
        self.assertIn("Research may use only caller-supplied sources", disabled_prompt)
        self.assertIn("must not plan code execution", disabled_prompt)

        enabled_backend = RuntimeBackend(agent="research")
        enabled = ChatService(
            enabled_backend,
            web_provider=FakeResearchProvider(),
            python_tool=SandboxedPythonTool(FakeExternalSandbox()),
        )
        enabled.orchestrator.planner.plan("Plan approved research and authorized execution")
        enabled_prompt = enabled_backend.completions[0][0].content
        self.assertIn("fetch explicitly approved web URLs", enabled_prompt)
        self.assertIn("execute caller-supplied Python through the configured", enabled_prompt)
        self.assertIn("Enabled tools: web_browsing", enabled_prompt)
        self.assertIn("Enabled tools: code_execution", enabled_prompt)
        self.assertNotIn("Research may use only caller-supplied sources", enabled_prompt)
        research_contract = next(
            line for line in enabled_prompt.splitlines() if line.startswith("research:")
        )
        coding_contract = next(
            line for line in enabled_prompt.splitlines() if line.startswith("coding:")
        )
        self.assertNotIn("do not browse", research_contract.lower())
        self.assertNotIn("do not execute code", coding_contract.lower())

    def test_execution_is_denied_by_default_and_prompt_cannot_authorize_it(self):
        backend = RuntimeBackend(agent="coding")
        service = ChatService(backend)

        with self.assertRaisesRegex(ValueError, "disabled"):
            service.task(
                {
                    "prompt": "Run this code",
                    "execution": {"code": "print(1)", "timeout_seconds": 5},
                }
            )
        answer = service.task({"prompt": "Run code using a tool if available"})
        self.assertIn("safe model response", answer["answer"])
        self.assertIsNone(service.orchestrator.agents.get("coding").python_tool)

    def test_explicitly_enabled_external_sandbox_executes_authorized_request(self):
        backend = RuntimeBackend(agent="coding")
        sandbox = FakeExternalSandbox()
        service = ChatService(
            backend, python_tool=SandboxedPythonTool(sandbox)
        )
        result = service.task(
            {
                "prompt": "Evaluate the supplied snippet",
                "execution": {"code": "print(1 + 1)", "timeout_seconds": 7},
            }
        )
        self.assertEqual(sandbox.calls, [("print(1 + 1)", 7)])
        self.assertIn("sandbox result", result["answer"])

    def test_server_requires_explicit_optional_tool_enablement(self):
        token = "integration-test-token-long-enough"
        sandbox = FakeExternalSandbox()
        server = make_server(
            "127.0.0.1",
            0,
            RuntimeBackend(agent="coding"),
            token,
            enable_sandbox_execution=True,
            python_tool=SandboxedPythonTool(sandbox),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        payload = {
            "prompt": "Evaluate this code",
            "execution": {"code": "print(2)", "timeout_seconds": 3},
        }
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/tasks",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        self.assertEqual(result["tasks"][0]["agent"], "coding")
        self.assertEqual(sandbox.calls, [("print(2)", 3)])

        disabled_server = make_server(
            "127.0.0.1", 0, RuntimeBackend(agent="coding"), token
        )
        disabled_thread = threading.Thread(
            target=disabled_server.serve_forever, daemon=True
        )
        disabled_thread.start()
        self.addCleanup(disabled_server.server_close)
        self.addCleanup(disabled_server.shutdown)
        disabled_request = Request(
            f"http://127.0.0.1:{disabled_server.server_port}/v1/tasks",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
        )
        with self.assertRaises(HTTPError) as rejected:
            urlopen(disabled_request, timeout=5)
        self.assertEqual(rejected.exception.code, 400)
        self.assertIn("disabled", rejected.exception.read().decode())

    def test_browsing_uses_only_explicitly_approved_urls(self):
        backend = RuntimeBackend(agent="research")
        provider = FakeResearchProvider()
        service = ChatService(backend, web_provider=provider)
        result = service.task(
            {
                "prompt": "Research this source",
                "browse": {
                    "urls": ["https://example.org/article"],
                    "approved_urls": [],
                    "approved_domains": ["example.org"],
                },
            }
        )
        self.assertIn("Evidence-based answer", result["answer"])
        self.assertEqual(
            provider.calls,
            [
                (
                    ("https://example.org/article",),
                    (),
                    ("example.org",),
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "not explicitly approved"):
            service.task(
                {
                    "prompt": "Research an unapproved source",
                    "browse": {
                        "urls": ["https://evil.example/article"],
                        "approved_urls": [],
                        "approved_domains": ["example.org"],
                    },
                }
            )

    def test_memory_and_full_default_registry_are_wired_together(self):
        class Embedder:
            def embed(self, text):
                return [1.0, 0.0]

        backend = RuntimeBackend()
        with tempfile.TemporaryDirectory() as directory:
            retriever = MemoryRetriever(
                Embedder(), SQLiteVectorStore(Path(directory) / "memory.sqlite3")
            )
            retriever.remember("operator", "A retained operator fact")
            service = ChatService(backend, retriever, "operator")
            self.assertEqual(len(service.orchestrator.agents.names), 36)
            self.assertIs(service.memory_retriever, retriever)
            service.task({"prompt": "Use stored context"})
            planner_input = next(
                call
                for call in backend.completions
                if "Create a short, safe execution plan" in call[0].content
            )
            self.assertTrue(
                any("A retained operator fact" in message.content for message in planner_input)
            )

    def test_legacy_prompt_and_sources_task_payload_remains_valid(self):
        backend = RuntimeBackend()
        service = ChatService(backend)
        result = service.task(
            {
                "prompt": "Summarize supplied information",
                "sources": [{"title": "Source", "content": "A caller-provided fact."}],
            }
        )
        self.assertEqual(result["answer"], "A safe model response")
        self.assertEqual(result["tasks"][0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()