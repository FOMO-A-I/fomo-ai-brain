"""Low-cardinality in-process Prometheus metrics for one API process."""

import threading
import time


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str], int] = {}
        self._total_seconds: dict[str, float] = {}
        self.started_at = time.monotonic()

    def record(self, operation: str, status: str, elapsed: float) -> None:
        if operation not in ("chat", "health", "knowledge") or status not in ("ok", "error"):
            raise ValueError("Unbounded metric labels are not allowed")
        with self._lock:
            key = (operation, status)
            self._counts[key] = self._counts.get(key, 0) + 1
            self._total_seconds[operation] = self._total_seconds.get(operation, 0.0) + max(0.0, elapsed)

    def render(self) -> str:
        with self._lock:
            lines = ["# TYPE fomo_requests_total counter"]
            for (operation, status), count in sorted(self._counts.items()):
                lines.append(f'fomo_requests_total{{operation="{operation}",status="{status}"}} {count}')
            lines.append("# TYPE fomo_request_duration_seconds_total counter")
            for operation, seconds in sorted(self._total_seconds.items()):
                lines.append(f'fomo_request_duration_seconds_total{{operation="{operation}"}} {seconds:.6f}')
            lines.append("# TYPE fomo_process_uptime_seconds gauge")
            lines.append(f"fomo_process_uptime_seconds {time.monotonic() - self.started_at:.3f}")
            return "\n".join(lines) + "\n"