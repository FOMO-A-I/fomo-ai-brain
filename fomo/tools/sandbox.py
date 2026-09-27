"""Client for a separately isolated, explicitly configured code sandbox."""

from __future__ import annotations

import json
import os
from urllib.parse import urlsplit

from ._http import request_public


class SandboxUnavailable(RuntimeError):
    """The external sandbox is absent or returned an invalid response."""


class SandboxClient:
    """Submit code only to a trusted HTTPS sandbox service; never execute locally."""

    def __init__(
        self,
        endpoint: str | None = None,
        token: str | None = None,
        *,
        timeout_seconds: float = 10,
        max_output_bytes: int = 64_000,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("FOMO_SANDBOX_URL", "")
        self.token = token or os.environ.get("FOMO_SANDBOX_TOKEN", "")
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        if not self.endpoint or not self.token:
            raise SandboxUnavailable(
                "Code execution is disabled: configure FOMO_SANDBOX_URL and "
                "FOMO_SANDBOX_TOKEN for an isolated external sandbox"
            )
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or (parsed.port not in (None, 443))
        ):
            raise ValueError("sandbox endpoint must be a trusted HTTPS URL on port 443")
        if not 1 <= timeout_seconds <= 60 or not 1 <= max_output_bytes <= 1_000_000:
            raise ValueError("sandbox response and timeout limits are out of range")
        self._parsed = parsed

    def execute(self, code: str, *, timeout_seconds: int = 5) -> dict:
        if not isinstance(code, str) or not code.strip() or len(code) > 20_000:
            raise ValueError("code must contain 1–20,000 characters")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 30:
            raise ValueError("execution timeout must be between 1 and 30 seconds")
        payload = json.dumps(
            {"language": "python", "code": code, "timeout_seconds": timeout_seconds}
        ).encode("utf-8")
        path = self._parsed.path.rstrip("/") + "/v1/execute"
        endpoint = f"https://{self._parsed.hostname}{path}"
        try:
            status, _, raw, _ = request_public(
                endpoint,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                body=payload,
                timeout=self.timeout_seconds,
                max_bytes=self.max_output_bytes,
                allowed_hosts={self._parsed.hostname.lower().rstrip(".")},
            )
            if status != 200:
                raise SandboxUnavailable(
                    f"Sandbox returned HTTP {status}; code was not run locally"
                )
            try:
                result = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SandboxUnavailable("Sandbox returned invalid JSON") from exc
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("stdout", ""), str)
                or not isinstance(result.get("stderr", ""), str)
                or type(result.get("exit_code")) is not int
            ):
                raise SandboxUnavailable("Sandbox returned an invalid execution result")
            return {
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "exit_code": result["exit_code"],
            }
        except OSError as exc:
            raise SandboxUnavailable(f"External sandbox request failed: {exc}") from exc
        except ValueError as exc:
            raise SandboxUnavailable(f"External sandbox request failed: {exc}") from exc