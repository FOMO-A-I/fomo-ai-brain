"""Bounded failover across explicitly configured, trusted inference workers."""

import json
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class WorkerPool:
    def __init__(self, base_urls: list[str], *, timeout: float = 45) -> None:
        if not base_urls or any(not url.startswith(("http://", "https://")) for url in base_urls):
            raise ValueError("Configure one or more trusted HTTP worker URLs")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_urls = [url.rstrip("/") for url in base_urls]
        self.timeout = timeout
        self._next = 0
        self._lock = threading.Lock()

    def complete(self, messages: list[dict], token: str) -> dict:
        if not token or len(token) < 16:
            raise ValueError("A worker authentication token is required")
        body = json.dumps({"messages": messages}).encode("utf-8")
        with self._lock:
            first = self._next
            self._next = (self._next + 1) % len(self.base_urls)
        failures = []
        for offset in range(len(self.base_urls)):
            base = self.base_urls[(first + offset) % len(self.base_urls)]
            req = Request(
                base + "/v1/infer",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
                method="POST",
            )
            try:
                with urlopen(req, timeout=self.timeout) as response:
                    result = json.load(response)
                if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
                    raise ValueError("Malformed worker response")
                return result
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                failures.append(f"{base}: {type(exc).__name__}")
        raise RuntimeError("No inference worker completed the request: " + "; ".join(failures))