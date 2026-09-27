import os
import unittest
from unittest.mock import patch

from fomo.tools._http import request_public
from fomo.tools.api_tool import AllowlistedApiClient, ApiPermission
from fomo.tools.database_tool import ReadOnlyDatabaseTool
from fomo.tools.python_tool import SandboxedPythonTool
from fomo.tools.sandbox import SandboxClient, SandboxUnavailable
from fomo.tools.web_tool import SafeWebFetcher


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