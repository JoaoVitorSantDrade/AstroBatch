from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from temporal_analysis import (
    build_session_temporal_report,
    build_temporal_report,
    enrich_flow_frames,
    parse_date_obs,
    read_fits_timestamp,
    write_temporal_report,
)


class TemporalAnalysisTests(unittest.TestCase):
    def test_parse_iso_fraction_timezone_and_timezone_less_fits_value(self):
        with_zone = parse_date_obs("2026-01-02T03:04:05.125-03:00")
        self.assertEqual(with_zone["timestamp_state"], "valid")
        self.assertTrue(with_zone["timezone_present"])
        self.assertEqual(with_zone["timezone"], "-03:00")
        self.assertEqual(with_zone["timestamp_utc"], "2026-01-02T06:04:05.125000Z")
        self.assertEqual(with_zone["timestamp_normalized"], with_zone["timestamp_utc"])

        without_zone = parse_date_obs("2026-01-02 03:04:05.125")
        self.assertEqual(without_zone["timestamp_state"], "valid")
        self.assertFalse(without_zone["timezone_present"])
        self.assertIsNone(without_zone["timezone"])
        self.assertEqual(without_zone["timestamp_utc"], "2026-01-02T03:04:05.125000Z")

    def test_invalid_and_missing_values_are_explicitly_unknown(self):
        for value in (None, "", "not a timestamp"):
            parsed = parse_date_obs(value)
            self.assertEqual(parsed["timestamp_state"], "unknown")
            self.assertIsNone(parsed["timestamp_utc"])
            self.assertIsNone(parsed["timestamp_normalized"])

    def test_grouping_sorts_out_of_order_frames_and_uses_strictly_greater_gap(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            values = {
                "frame_003.fits": "2026-01-01T00:30:00Z",
                "frame_001.fits": "2026-01-01T00:00:00Z",
                # Exactly fifteen minutes stays in the same logical group.
                "frame_002.fits": "2026-01-01T00:15:00Z",
                "frame_004.fits": "2026-01-01T00:46:00Z",
                "frame_007.fits": "2026-01-01T00:46:00Z",
                "frame_005.fits": None,
                "frame_006.fits": "not valid",
            }
            flow_frames = {}
            before = {}
            for name, stamp in values.items():
                header = fits.Header()
                if stamp is not None:
                    header["DATE-OBS"] = stamp
                path = root / name
                fits.PrimaryHDU(np.ones((4, 4), dtype=np.uint16), header=header).writeto(path)
                before[name] = path.read_bytes()
                flow_frames[name] = {
                    "status": "accepted",
                    "fwhm": 2.0,
                    "star_count": 20,
                    "roundness": 0.9,
                }
            report = build_temporal_report(
                root,
                {"frames": flow_frames},
                gap_minutes=15,
            )
            self.assertEqual(
                [frame["frame"] for frame in report["frames"]],
                ["frame_001.fits", "frame_002.fits", "frame_003.fits", "frame_004.fits", "frame_007.fits", "frame_005.fits", "frame_006.fits"],
            )
            timed_groups = [group for group in report["groups"] if group["timing_state"] == "valid"]
            self.assertEqual(len(timed_groups), 2)
            self.assertEqual(timed_groups[0]["frames"], ["frame_001.fits", "frame_002.fits", "frame_003.fits"])
            self.assertEqual(timed_groups[1]["frames"], ["frame_004.fits", "frame_007.fits"])
            self.assertEqual(report["unknown_timestamp_frames"], ["frame_005.fits", "frame_006.fits"])
            self.assertTrue(report["files_unchanged"])
            for name, payload in before.items():
                self.assertEqual((root / name).read_bytes(), payload)

    def test_enrichment_and_atomic_sidecar_preserve_flow_compatibility(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            path = root / "one.fits"
            fits.PrimaryHDU(np.zeros((2, 2), dtype=np.uint16), fits.Header({"DATE-OBS": "2026-01-01T00:00:00Z"})).writeto(path)
            flow = {"schema_version": 2, "frames": {"one.fits": {"status": "accepted", "matrix": np.eye(3).tolist()}}}
            enriched = enrich_flow_frames(root, flow)
            self.assertEqual(enriched["schema_version"], 2)
            self.assertEqual(enriched["frames"]["one.fits"]["timestamp_state"], "valid")
            write_temporal_report(root / "temporal_analysis.json", {"schema_version": 1, "ok": True})
            self.assertEqual(json.loads((root / "temporal_analysis.json").read_text(encoding="utf-8"))["ok"], True)
            self.assertEqual(read_fits_timestamp(path)["timestamp_state"], "valid")

    def test_session_report_reuses_normalized_flow_timestamps(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            batch = root / "batch_01"
            batch.mkdir()
            path = batch / "frame_001.fits"
            fits.PrimaryHDU(np.ones((2, 2), dtype=np.uint16)).writeto(path)
            flow = {
                "schema_version": 2,
                "frames": {
                    "frame_001.fits": {
                        "status": "accepted",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "timestamp_utc": "2026-01-01T00:00:00.000000Z",
                        "timestamp_normalized": "2026-01-01T00:00:00.000000Z",
                        "timestamp_state": "valid",
                        "timezone": "Z",
                        "timezone_present": True,
                        "epoch_s": 1767225600.0,
                    }
                },
            }
            (batch / "flow_local.json").write_text(json.dumps(flow), encoding="utf-8")
            with patch("temporal_analysis.read_fits_timestamp", side_effect=AssertionError("header reread")):
                report = build_session_temporal_report(root)
            self.assertEqual(report["groups"][0]["frames"], ["batch_01/frame_001.fits"])

    def test_unknown_flow_timestamp_is_cached_without_reopening_header(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            batch = root / "batch_01"
            batch.mkdir()
            path = batch / "frame_001.fits"
            fits.PrimaryHDU(np.ones((2, 2), dtype=np.uint16)).writeto(path)
            flow = {
                "frames": {
                    "frame_001.fits": {
                        "status": "accepted",
                        "timestamp_state": "unknown",
                        "timestamp": None,
                        "timestamp_utc": None,
                        "timestamp_normalized": None,
                    }
                }
            }
            (batch / "flow_local.json").write_text(json.dumps(flow), encoding="utf-8")
            with patch("temporal_analysis.read_fits_timestamp", side_effect=AssertionError("header reread")):
                report = build_session_temporal_report(root)
            self.assertEqual(report["unknown_timestamp_frames"], ["batch_01/frame_001.fits"])

    def test_session_view_groups_across_batches_without_touching_fits(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for batch_name, stamp in (
                ("batch_01", "2026-01-01T23:59:59Z"),
                ("batch_02", "2026-01-02T00:20:00Z"),
            ):
                batch = root / batch_name
                batch.mkdir()
                path = batch / "frame_001.fits"
                fits.PrimaryHDU(
                    np.ones((2, 2), dtype=np.uint16),
                    fits.Header({"DATE-OBS": stamp}),
                ).writeto(path)
            report = build_session_temporal_report(root, gap_minutes=15)
            self.assertEqual(len(report["groups"]), 2)
            self.assertEqual(report["groups"][1]["gap_before_seconds"], 1201.0)
            self.assertEqual(report["groups"][0]["frames"], ["batch_01/frame_001.fits"])
            self.assertTrue(report["files_unchanged"])


if __name__ == "__main__":
    unittest.main()
