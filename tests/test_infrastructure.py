import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from infrastructure.monitoring.metrics import Metrics
from infrastructure.workers.router import WorkerPool


class InfrastructureTests(unittest.TestCase):
    def test_metrics_are_low_cardinality(self):
        metrics = Metrics()
        metrics.record("chat", "ok", 0.125)
        self.assertIn('fomo_requests_total{operation="chat",status="ok"} 1', metrics.render())
        with self.assertRaises(ValueError):
            metrics.record("user-input-as-metric-label", "ok", 0.1)

    def test_worker_failover_and_round_robin(self):
        pool = WorkerPool(["https://first.example", "https://second.example"])
        response = io.BytesIO(json.dumps({"answer": "real backend response"}).encode())
        with patch("infrastructure.workers.router.urlopen", side_effect=[URLError("offline"), response]) as call:
            self.assertEqual(pool.complete([{"role": "user", "content": "hello"}], "a-long-enough-test-token")["answer"], "real backend response")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(call.call_args.args[0].full_url, "https://second.example/v1/infer")


if __name__ == "__main__":
    unittest.main()