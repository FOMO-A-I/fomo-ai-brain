import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fomo.api.server import RateLimit, make_server
from fomo.brain.model import CheckpointNotConfigured, LocalTransformersBackend


class TestBackend:
    """Test double only; production startup uses a local checkpoint."""

    def complete(self, messages):
        if "Create a short, safe execution plan" in messages[0].content:
            return json.dumps({"tasks": [
                {"id": "answer", "description": "Reply accurately", "agent": "general", "depends_on": []}
            ]})
        return "A response from the injected test backend"

    def stream(self, messages):
        yield "First "
        yield "second"


class APITests(unittest.TestCase):
    def setUp(self):
        self.token = "a-long-enough-test-token"
        self.server = make_server("127.0.0.1", 0, TestBackend(), self.token)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def request(self, path, body=None, *, authorized=True):
        headers = {}
        data = None
        if authorized:
            headers["Authorization"] = "Bearer " + self.token
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        with urlopen(Request(self.url + path, data=data, headers=headers), timeout=5) as response:
            return response.status, response.read(), response.headers["Content-Type"]

    def test_auth_chat_tasks_and_true_stream_interface(self):
        self.assertEqual(json.loads(self.request("/health", authorized=False)[1])["ready"], False)
        with self.assertRaises(HTTPError) as unauthorized:
            self.request("/v1/chat", {"messages": [{"role": "user", "content": "Hi"}]}, authorized=False)
        self.assertEqual(unauthorized.exception.code, 401)
        status, body, _ = self.request("/v1/chat", {"messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(status, 200)
        self.assertIn("answer", json.loads(body))
        status, body, _ = self.request("/v1/tasks", {"prompt": "Describe caching"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["tasks"][0]["status"], "completed")
        status, body, content_type = self.request("/v1/chat/stream", {
            "messages": [{"role": "user", "content": "Hi"}]
        })
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", content_type)
        self.assertIn(b'{"text": "First "}', body)
        self.assertIn(b"event: done", body)

    def test_bad_requests_and_missing_checkpoint_are_explicit(self):
        with self.assertRaises(HTTPError) as invalid:
            self.request("/v1/chat", {"messages": []})
        self.assertEqual(invalid.exception.code, 400)
        with self.assertRaises(HTTPError) as missing_token:
            self.request("/metrics", authorized=False)
        self.assertEqual(missing_token.exception.code, 401)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(CheckpointNotConfigured):
                LocalTransformersBackend(Path(temp) / "no-checkpoint")
        with self.assertRaises(ValueError):
            make_server("0.0.0.0", 0, TestBackend(), self.token)
        limiter = RateLimit(rate_per_minute=1, burst=1)
        self.assertTrue(limiter.allow())
        self.assertFalse(limiter.allow())


if __name__ == "__main__":
    unittest.main()