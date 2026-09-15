from __future__ import annotations

import time
import unittest

from app.engines.telemetry import ProcessTelemetry


class ProcessTelemetryTests(unittest.TestCase):
    def test_stage_probe_returns_timing_and_resource_fields(self):
        probe = ProcessTelemetry(sample_interval=0.001)
        with probe:
            time.sleep(0.005)
        self.assertIsNotNone(probe.result)
        assert probe.result is not None
        self.assertGreaterEqual(probe.result.wall_seconds, 0.005)
        self.assertGreaterEqual(probe.result.peak_rss_bytes, 0)
        self.assertGreaterEqual(probe.result.peak_thread_count, 0)
        self.assertIn("cpu_to_wall", probe.result.as_dict())


if __name__ == "__main__":
    unittest.main()

