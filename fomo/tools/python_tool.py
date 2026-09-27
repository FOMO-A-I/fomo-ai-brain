"""Python code tool that has no path to local process execution."""

from __future__ import annotations

from .sandbox import SandboxClient


class SandboxedPythonTool:
    def __init__(self, sandbox: SandboxClient) -> None:
        if not isinstance(sandbox, SandboxClient):
            raise TypeError("an explicitly configured SandboxClient is required")
        self.sandbox = sandbox

    def run(self, code: str, *, timeout_seconds: int = 5) -> dict:
        return self.sandbox.execute(code, timeout_seconds=timeout_seconds)