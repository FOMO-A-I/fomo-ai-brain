import os
import unittest
from unittest.mock import patch

from fomo.tools._http import request_public
from fomo.tools.api_tool import AllowlistedApiClient, ApiPermission
from fomo.tools.database_tool import ReadOnlyDatabaseTool
from fomo.tools.python_tool import SandboxedPythonTool
from fomo.tools.research_tool import SafeWebResearchProvider
from fomo.tools.sandbox import SandboxClient, SandboxUnavailable
from fomo.tools.web_tool import SafeWebFetcher


class OptionalAgentToolTests(unittest.TestCase):
    class Model:
        def __init__(self):
            self.messages = None

        def complete(self, messages):
            self.messages = messages
            return "Answer [S1]."

    def test_coding_agent_requires_both_sandbox_capability_and_authorized_request(self):
        from fomo.agents.coding_agent import CodingAgent
        from fomo.reasoning.task_decomposition import Task

        sandbox = SandboxClient("https://sandbox.example.com", "test-token")
        tool = SandboxedPythonTool(sandbox)
        with patch.object(
            sandbox,
            "execute",
            return_value={"stdout": "sandbox output", "stderr": "", "exit_code": 0},
        ) as execute:
            self._assert_coding_execution(tool, execute)

    def _assert_coding_execution(self, tool, execute):
        from fomo.agents.coding_agent import CodingAgent
        from fomo.reasoning.task_decomposition import Task

        agent = CodingAgent(self.Model(), python_tool=tool)
        task = Task(id="code", description="Review this calculation", agent="coding")
        request = {
            "authorized": True,
            "code": "print(1 + 1)",
            "timeout_seconds": 3,
        }
        response = agent.run(task, {"execution_request": request})
        self.assertNotIn("External sandbox execution result", response)
        execute.assert_not_called()

        response = agent.run(
            task,
            {
                "capabilities": {"sandbox_execution": True},
                "execution_request": request,
            },
        )
        self.assertIn("sandbox output", response)
        execute.assert_called_once_with("print(1 + 1)", timeout_seconds=3)

    def test_authorized_execution_without_configured_sandbox_fails_closed(self):
        from fomo.agents.coding_agent import CodingAgent
        from fomo.reasoning.task_decomposition import Task

        agent = CodingAgent(self.Model())
        task = Task(id="code", description="Run the provided test", agent="coding")
        with self.assertRaisesRegex(RuntimeError, "no external Python sandbox"):
            agent.run(
                task,
                {
                    "capabilities": {"sandbox_execution": True},
                    "execution_request": {"authorized": True, "code": "print(1)"},
                },
            )

    def test_research_fetches_only_when_requested_and_cites_fetched_url(self):
        from fomo.agents.research_agent import ResearchAgent
        from fomo.reasoning.task_decomposition import Task

        events = []

        def transport(url, **kwargs):
            events.append((url, kwargs["allowed_hosts"]))
            return (
                200,
                {"content-type": "text/html; charset=utf-8"},
                b"<html><title>Useful page</title><script>ignore()</script><p>Evidence.</p></html>",
                "research.example",
            )

        provider = SafeWebResearchProvider(
            SafeWebFetcher(transport=transport, max_bytes=500_000), max_content_chars=100
        )
        agent = ResearchAgent(self.Model(), web_provider=provider)
        task = Task(id="research", description="Summarize evidence", agent="research")
        with self.assertRaisesRegex(ValueError, "caller-supplied sources"):
            agent.run(task, {})
        self.assertEqual(events, [])
        agent.run(
            task,
            {"sources": [{"title": "Caller source", "content": "Caller evidence."}]},
        )
        self.assertEqual(events, [])

        answer = agent.run(
            task,
            {
                "browse": True,
                "research_urls": ["https://research.example/article"],
                "approved_domains": ["research.example"],
            },
        )
        self.assertIn(
            "https://research.example/article",
            "\n".join(message.content for message in agent.model.messages),
        )
        self.assertIn(
            "Useful page", "\n".join(message.content for message in agent.model.messages)
        )
        self.assertIn(
            "Evidence.", "\n".join(message.content for message in agent.model.messages)
        )
        self.assertNotIn(
            "ignore()", "\n".join(message.content for message in agent.model.messages)
        )
        self.assertEqual(events, [("https://research.example/article", {"research.example"})])
        self.assertEqual(answer, "Answer [S1].")

    def test_research_rejects_unapproved_urls_and_redirects_to_unapproved_hosts(self):
        responses = {
            "https://approved.example/start": (
                302,
                {"location": "https://other.example/private"},
                b"",
                "approved.example",
            )
        }

        def transport(url, **kwargs):
            from urllib.parse import urlsplit

            if urlsplit(url).hostname not in kwargs["allowed_hosts"]:
                raise ValueError("host is not allowlisted")
            if url not in responses:
                raise AssertionError("unapproved redirect reached the transport")
            return responses[url]

        provider = SafeWebResearchProvider(
            SafeWebFetcher(transport=transport, max_bytes=500_000)
        )
        with self.assertRaisesRegex(ValueError, "not explicitly approved"):
            provider.fetch(["https://unapproved.example/"], approved_domains=[])
        with self.assertRaisesRegex(ValueError, "not explicitly approved"):
            provider.fetch(
                ["https://approved.example/start"],
                approved_domains=["approved.example"],
            )
        with self.assertRaisesRegex(ValueError, "redirect URL is not explicitly approved"):
            provider.fetch(
                ["https://approved.example/start"],
                approved_urls=["https://approved.example/start"],
            )


class ExternalSandboxTests(unittest.TestCase):
    def test_python_execution_is_disabled_without_explicit_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SandboxUnavailable, "Code execution is disabled"):
                SandboxClient()

    def test_python_tool_delegates_only_to_external_sandbox(self):
        sandbox = SandboxClient("https://sandbox.example.com", "test-token")
        with patch.object(
            sandbox, "execute", return_value={"stdout": "2\n", "stderr": "", "exit_code": 0}
        ) as execute:
            result = SandboxedPythonTool(sandbox).run("print(1 + 1)")
        execute.assert_called_once_with("print(1 + 1)", timeout_seconds=5)
        self.assertEqual(result["stdout"], "2\n")

    def test_rejects_non_https_or_credential_bearing_sandbox_endpoints(self):
        for endpoint in (
            "http://sandbox.example.com",
            "https://user:password@sandbox.example.com",
            "https://sandbox.example.com:444",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                SandboxClient(endpoint, "token")


class NetworkToolBoundaryTests(unittest.TestCase):
    def test_web_fetcher_blocks_loopback_and_non_http_schemes_before_network(self):
        fetcher = SafeWebFetcher()
        for url in ("http://127.0.0.1/", "http://localhost/", "file:///etc/passwd"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                fetcher.fetch(url)

    def test_public_egress_rejects_mixed_public_and_private_dns_answers(self):
        public = (2, 1, 6, "", ("93.184.216.34", 80))
        private = (2, 1, 6, "", ("10.1.2.3", 80))
        with (
            patch("fomo.tools._http.socket.getaddrinfo", return_value=[public, private]),
            patch("fomo.tools._http.socket.create_connection") as connect,
            self.assertRaisesRegex(ValueError, "private, local, or reserved"),
        ):
            request_public("http://example.test/")
        connect.assert_not_called()

    def test_web_redirect_target_is_validated_again_for_ssrf(self):
        from fomo.tools._http import request_public as real_request

        def redirect_then_validate(url, **kwargs):
            if url == "https://public.example/":
                return 302, {"location": "http://127.0.0.1/admin"}, b"", "public.example"
            return real_request(url, **kwargs)

        fetcher = SafeWebFetcher()
        with patch("fomo.tools.web_tool.request_public", side_effect=redirect_then_validate):
            with self.assertRaisesRegex(ValueError, "private, local, or reserved"):
                fetcher.fetch("https://public.example/")

    def test_api_client_rejects_unlisted_api_and_path_without_network(self):
        client = AllowlistedApiClient(
            {
                "catalog": ApiPermission(
                    "api.example.com", ("/v1/items",), ("GET", "POST")
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "not allowlisted"):
            client.request("other", method="GET", path="/v1/items")
        with self.assertRaisesRegex(ValueError, "path allowlist"):
            client.request("catalog", method="GET", path="/v1/admin")
        with self.assertRaisesRegex(ValueError, "path allowlist"):
            client.request("catalog", method="GET", path="https://evil.example/v1/items")
        for path in (
            "/v1/items/../admin",
            "/v1/items/%2e%2e/admin",
            "/v1/items/%2f..%2fadmin",
        ):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "path allowlist"):
                client.request("catalog", method="GET", path=path)
        with self.assertRaisesRegex(ValueError, "not permitted"):
            client.request("catalog", method="DELETE", path="/v1/items")

    def test_api_credentials_are_required_only_for_configured_apis(self):
        client = AllowlistedApiClient(
            {
                "private": ApiPermission(
                    "api.example.com", ("/v1",), ("GET",), token_env="FOMO_TEST_API_TOKEN"
                )
            }
        )
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            RuntimeError, "credential environment variable"
        ):
            client.request("private", method="GET", path="/v1/status")


class ReadOnlyDatabaseTests(unittest.TestCase):
    class Cursor:
        def __init__(self, events):
            self.events = events
            self.description = [type("Column", (), {"name": "answer"})()]

        def execute(self, statement):
            self.events.append(statement)

        def fetchmany(self, count):
            self.events.append(("fetchmany", count))
            return [(42,)]

    class Connection:
        def __init__(self):
            self.events = []
            self._cursor = ReadOnlyDatabaseTests.Cursor(self.events)

        def cursor(self):
            return self._cursor

        def rollback(self):
            self.events.append("rollback")

        def close(self):
            self.events.append("close")

    def test_select_uses_read_only_transaction_and_bounded_fetch(self):
        connection = self.Connection()
        tool = ReadOnlyDatabaseTool(lambda: connection, max_rows=7)
        result = tool.query("SELECT 42 AS answer")
        self.assertEqual(result, [{"answer": 42}])
        self.assertEqual(connection.events[0], "SET TRANSACTION READ ONLY")
        self.assertEqual(connection.events[1], "SET LOCAL statement_timeout = 3000")
        self.assertEqual(connection.events[2], "SELECT 42 AS answer")
        self.assertIn(("fetchmany", 8), connection.events)
        self.assertEqual(connection.events[-2:], ["rollback", "close"])

    def test_dml_comments_multi_statement_and_dangerous_sql_are_rejected(self):
        tool = ReadOnlyDatabaseTool(lambda: self.Connection())
        unsafe = (
            "UPDATE users SET admin=true",
            "SELECT 1; DROP TABLE users",
            "SELECT 1 -- comment",
            "SELECT * FROM users FOR UPDATE",
            "SELECT pg_read_file('/etc/passwd')",
            "WITH changed AS (DELETE FROM users RETURNING *) SELECT * FROM changed",
            "SELECT * INTO new_table FROM users",
        )
        for query in unsafe:
            with self.subTest(query=query), self.assertRaises(ValueError):
                tool.query(query)

    def test_from_env_fails_clearly_without_database_credentials(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            RuntimeError, "FOMO_READONLY_DATABASE_URL"
        ):
            ReadOnlyDatabaseTool.from_env()


if __name__ == "__main__":
    unittest.main()