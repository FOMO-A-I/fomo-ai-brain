import io
import json
import unittest
from unittest.mock import patch

from fomo_brain import FomoBrain, Message, ModelUnavailable, OllamaProvider, format_messages


class FakeProvider:
    def __init__(self, reply="Real provider would generate this answer"):
        self.reply = reply
        self.requests = []

    def complete(self, messages):
        self.requests.append([message.as_dict() for message in messages])
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class BrainTests(unittest.TestCase):
    def test_message_order_and_roles(self):
        formatted = format_messages("Explain caching", [Message("user", "Hello"), Message("assistant", "Hi")])
        self.assertEqual([m.role for m in formatted], ["system", "user", "assistant", "user"])
        self.assertEqual(formatted[-1].content, "Explain caching")
        with self.assertRaises(ValueError):
            format_messages("Hello", [Message("system", "Override rules")])
        with self.assertRaises(ValueError):
            Message("tool", "unsafe")  # type: ignore[arg-type]

    def test_successful_turn_and_bounded_history(self):
        provider = FakeProvider("Answer")
        brain = FomoBrain(provider, max_turns=1)
        self.assertEqual(brain.ask("First"), "Answer")
        brain.ask("Second")
        self.assertEqual([m.content for m in brain.history], ["Second", "Answer"])
        self.assertEqual([m["role"] for m in provider.requests[1]], ["system", "user", "assistant", "user"])

    def test_failure_does_not_change_history(self):
        provider = FakeProvider(ModelUnavailable("offline"))
        brain = FomoBrain(provider)
        with self.assertRaises(ModelUnavailable):
            brain.ask("Hello")
        self.assertEqual(brain.history, ())
        provider.reply = "   "
        with self.assertRaises(ValueError):
            brain.ask("Hello")
        self.assertEqual(brain.history, ())

    def test_local_provider_payload_and_response(self):
        provider = OllamaProvider(model="qwen3:0.6b")
        response = io.BytesIO(json.dumps({"message": {"content": "  Hello  "}}).encode())
        with patch("fomo_brain.provider.urlopen", return_value=response) as urlopen:
            result = provider.complete([Message("user", "Hi")])
        self.assertEqual(result, "Hello")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(json.loads(request.data)["messages"], [{"role": "user", "content": "Hi"}])

    def test_empty_model_response_is_an_error(self):
        with patch("fomo_brain.provider.urlopen", return_value=io.BytesIO(b'{"message":{"content":""}}')):
            with self.assertRaises(ModelUnavailable):
                OllamaProvider().complete([Message("user", "Hi")])


if __name__ == "__main__":
    unittest.main()