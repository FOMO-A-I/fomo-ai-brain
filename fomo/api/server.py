"""Small local-only JSON/SSE API. A website must call it from its own backend."""

import hmac
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fomo.agents.orchestrator import OrchestrationError
from fomo.agents.planner import PlanningError
from fomo.brain.model import CheckpointNotConfigured, LocalTransformersBackend, ModelLoadError
from fomo.memory.embeddings import EmbeddingUnavailable, LocalSentenceTransformerEmbedder
from fomo.memory.retrieval import MemoryRetriever
from fomo.memory.vector_store import SQLiteVectorStore
from fomo.tools.python_tool import SandboxedPythonTool
from fomo.tools.research_tool import SafeWebResearchProvider
from fomo.tools.sandbox import SandboxClient

from .chat import ChatService
MAX_REQUEST_BYTES = 64 * 1024


class RateLimit:
    """In-process capacity guard; not a distributed or per-user quota."""

    def __init__(self, rate_per_minute: int = 30, burst: int = 20) -> None:
        self.rate = rate_per_minute / 60
        self.burst = burst
        self.tokens = float(burst)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens < 1:
                return False
            self.tokens -= 1
            return True


def make_server(
    host: str,
    port: int,
    backend: LocalTransformersBackend,
    token: str,
    *,
    memory_retriever: MemoryRetriever | None = None,
    memory_scope_id: str | None = None,
    enable_web_research: bool = False,
    web_provider: SafeWebResearchProvider | None = None,
    enable_sandbox_execution: bool = False,
    python_tool: SandboxedPythonTool | None = None,
) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError("Bind only to loopback; use a secured reverse proxy for remote access")
    if not token or len(token) < 16:
        raise ValueError("Set FOMO_API_TOKEN to at least 16 characters")
    if type(enable_web_research) is not bool or type(enable_sandbox_execution) is not bool:
        raise ValueError("optional tool enable settings must be booleans")
    if web_provider is not None and not enable_web_research:
        raise ValueError("web provider injection requires explicit web research enablement")
    if python_tool is not None and not enable_sandbox_execution:
        raise ValueError("sandbox injection requires explicit sandbox execution enablement")
    if enable_web_research and web_provider is None:
        web_provider = SafeWebResearchProvider()
    if enable_sandbox_execution and python_tool is None:
        # The client requires externally configured HTTPS endpoint and credentials.
        # It never provides local execution as a fallback.
        python_tool = SandboxedPythonTool(SandboxClient())
    if memory_retriever is None and memory_scope_id is None:
        memory_database = os.getenv("FOMO_MEMORY_DB_PATH")
        embedding_model = os.getenv("FOMO_EMBEDDING_MODEL_PATH")
        configured_scope = os.getenv("FOMO_MEMORY_SCOPE_ID")
        configured = (memory_database, embedding_model, configured_scope)
        if any(configured):
            if not all(configured):
                raise ValueError(
                    "Configure FOMO_MEMORY_DB_PATH, FOMO_EMBEDDING_MODEL_PATH, "
                    "and FOMO_MEMORY_SCOPE_ID together"
                )
            memory_retriever = MemoryRetriever(
                LocalSentenceTransformerEmbedder(embedding_model),
                SQLiteVectorStore(memory_database),
            )
            memory_scope_id = configured_scope
    service = ChatService(
        backend,
        memory_retriever,
        memory_scope_id,
        web_provider=web_provider,
        python_tool=python_tool,
    )
    limiter = RateLimit()
    counters: dict[str, int] = {"ok": 0, "error": 0}
    duration_sum: dict[str, float] = {}
    metrics_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "FomoAI/0.1"

        def log_message(self, fmt: str, *args: object) -> None:
            # The HTTP request line may contain private prompt text.
            pass

        def reply(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            operation = self.path.split("?", 1)[0]
            if operation not in ("/health", "/v1/chat", "/v1/infer", "/v1/tasks", "/v1/chat/stream", "/v1/memory", "/metrics"):
                operation = "other"
            with metrics_lock:
                counters["ok" if status < 400 else "error"] += 1
                duration_sum[operation] = duration_sum.get(operation, 0.0) + max(
                    0.0, time.monotonic() - getattr(self, "_started_at", time.monotonic())
                )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authenticate(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, "Bearer " + token):
                self.reply(401, {"error": "Authentication required"})
                return False
            if not limiter.allow():
                self.reply(429, {"error": "Local request rate exceeded"})
                return False
            return True

        def read_body(self) -> dict:
            if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
                raise ValueError("Expected application/json")
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length") from exc
            if not 0 <= length <= MAX_REQUEST_BYTES:
                raise ValueError("Request body must be under 64 KB")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            return payload

        def do_GET(self) -> None:
            self._started_at = time.monotonic()
            if self.path == "/health":
                self.reply(200, {"status": "configured", "ready": getattr(backend, "_model", None) is not None})
                return
            if not self.authenticate():
                return
            if self.path == "/metrics":
                with metrics_lock:
                    text = (
                        "# TYPE fomo_api_responses_total counter\n"
                        f'fomo_api_responses_total{{status="ok"}} {counters["ok"]}\n'
                        f'fomo_api_responses_total{{status="error"}} {counters["error"]}\n'
                        "# TYPE fomo_api_request_duration_seconds_total counter\n"
                        + "".join(
                            f'fomo_api_request_duration_seconds_total{{route="{route}"}} {seconds:.6f}\n'
                            for route, seconds in sorted(duration_sum.items())
                        )
                    ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                self.wfile.write(text)
            elif self.path == "/v1/memory":
                if service.memory_retriever is None:
                    self.reply(503, {"error": "Memory is not configured"})
                    return
                try:
                    memories = service.memory_retriever.store.list(service.memory_scope_id, 50)
                except Exception:
                    self.reply(500, {"error": "Memory read failed"})
                    return
                self.reply(200, {"memories": memories})
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self) -> None:
            self._started_at = time.monotonic()
            if not self.authenticate():
                return
            try:
                data = self.read_body()
                if self.path in ("/v1/chat", "/v1/infer"):
                    if set(data) != {"messages"}:
                        raise ValueError("Expected only messages")
                    self.reply(200, service.chat(data["messages"]))
                elif self.path == "/v1/tasks":
                    self.reply(200, service.task(data))
                elif self.path == "/v1/memory":
                    if service.memory_retriever is None:
                        self.reply(503, {"error": "Memory is not configured"})
                        return
                    if set(data) - {"content", "metadata"} or "content" not in data:
                        raise ValueError("Expected content and optional metadata")
                    memory_id = service.memory_retriever.remember(
                        service.memory_scope_id,
                        data["content"],
                        data.get("metadata"),
                    )
                    self.reply(201, {"id": memory_id})
                elif self.path == "/v1/chat/stream":
                    if set(data) != {"messages"}:
                        raise ValueError("Expected only messages")
                    messages = service.prepare_chat_messages(data["messages"])
                    stream = getattr(backend, "stream", None)
                    if not callable(stream):
                        self.reply(501, {"error": "True checkpoint streaming is unavailable"})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    outcome = "error"
                    try:
                        for token_chunk in stream(messages):
                            line = "data: " + json.dumps({"text": token_chunk}, ensure_ascii=False) + "\n\n"
                            self.wfile.write(line.encode("utf-8"))
                            self.wfile.flush()
                        self.wfile.write(b"event: done\ndata: {}\n\n")
                        outcome = "ok"
                    except Exception:
                        try:
                            self.wfile.write(b'event: error\ndata: {"error":"inference interrupted"}\n\n')
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                    finally:
                        with metrics_lock:
                            counters[outcome] += 1
                            duration_sum["/v1/chat/stream"] = duration_sum.get(
                                "/v1/chat/stream", 0.0
                            ) + max(0.0, time.monotonic() - self._started_at)
                        self.close_connection = True
                else:
                    self.reply(404, {"error": "Not found"})
            except (ValueError, json.JSONDecodeError) as exc:
                self.reply(400, {"error": str(exc)})
            except (CheckpointNotConfigured, ModelLoadError) as exc:
                self.reply(503, {"error": str(exc)})
            except EmbeddingUnavailable as exc:
                self.reply(503, {"error": str(exc)})
            except (PlanningError, OrchestrationError) as exc:
                self.reply(422, {"error": str(exc)})
            except Exception:
                # Never send exception details, local file paths, or model input to a client.
                print("FOMO API request failed unexpectedly", file=sys.stderr)
                self.reply(500, {"error": "Internal request failure"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    try:
        backend = LocalTransformersBackend()
    except CheckpointNotConfigured as exc:
        raise SystemExit(str(exc)) from exc
    token = os.getenv("FOMO_API_TOKEN", "")
    host = os.getenv("FOMO_HOST", "127.0.0.1")
    port = int(os.getenv("FOMO_PORT", "8765"))
    def enabled(name: str) -> bool:
        value = os.getenv(name, "0")
        if value not in {"0", "1"}:
            raise ValueError(f"{name} must be set to 0 or 1")
        return value == "1"

    server = make_server(
        host,
        port,
        backend,
        token,
        enable_web_research=enabled("FOMO_ENABLE_WEB_RESEARCH"),
        enable_sandbox_execution=enabled("FOMO_ENABLE_SANDBOX_EXECUTION"),
    )
    print(f"FOMO API configured at http://{host}:{server.server_port}; checkpoint loads on first request")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()