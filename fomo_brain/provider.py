"""Local Ollama adapter. No remote API keys or simulated answers."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Sequence

from .messages import Message


class ModelUnavailable(RuntimeError):
    """The configured model cannot return a valid response."""


class OllamaProvider:
    def __init__(
        self,
        model: str = "qwen3:0.6b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, messages: Sequence[Message]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "stream": False,
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise ModelUnavailable(
                "Could not reach the local model. Start Ollama and pull the configured model."
            ) from exc
        answer = result.get("message", {}).get("content") if isinstance(result, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise ModelUnavailable("The model returned no answer; check the model and Ollama logs.")
        return answer.strip()