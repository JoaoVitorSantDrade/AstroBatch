from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

import astroflow_logic as flow


class FlowWorkflowTests(unittest.TestCase):
    @staticmethod
    def _translation(dx, dy=0.0):
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 2] = dx
        matrix[1, 2] = dy
        return matrix

    def test_reference_rebase_reuses_graph_and_preserves_final_geometry(self):
        local_c = self._translation(5.0)
        flow_data = {
            "schema_version": 3,
            "batch_anchor": "01.fits",
            "transform_revision": "oldrev",
            "frames": {
                "01.fits": {"status": "accepted", "matrix": np.eye(3).tolist(), "hop_count": 0},
                "02.fits": {"status": "accepted", "matrix": self._translation(2.0).tolist(), "hop_count": 1},
                "03.fits": {"status": "accepted", "matrix": local_c.tolist(), "hop_count": 2},
            },
            "registration_graph": {
                "root": "01.fits",
                "edges": [
                    {"source": "01.fits", "target": "02.fits", "relative_matrix": self._translation(2.0).tolist()},
                    {"source": "02.fits", "target": "03.fits", "relative_matrix": self._translation(3.0).tolist()},
                ],
            },
        }
        with patch.object(flow, "load_fits_data", side_effect=AssertionError("rebase decoded FITS")):
            rebased = flow.rebase_flow_reference(flow_data, "02.fits")
        np.testing.assert_allclose(
            np.asarray(rebased["frames"]["03.fits"]["matrix"]), self._translation(3.0)
        )
        self.assertEqual(rebased["batch_anchor"], "02.fits")
        self.assertTrue(rebased["reference_provenance"]["analysis_reused"])
        global_matrix = self._translation(10.0)
        np.testing.assert_allclose(
            flow.rebase_global_transform(global_matrix, self._translation(2.0))
            @ np.asarray(rebased["frames"]["03.fits"]["matrix"]),
            global_matrix @ local_c,
        )

    def test_global_anchor_reuses_only_equivalent_local_detection(self):
        stars = np.arange(70, dtype=np.float32).reshape(35, 2)
        info = {
            "anchor_stars": stars.tolist(),
            "anchor_shape": [512, 512],
            "fwhm": 3.0,
            "anchor_detection": {
                "fwhm": 3.0,
                "sigma": 4.0,
                "sigma_used": 4.0,
                "max_stars": 250,
                "engine": "DAO",
                "engine_profile": "Stable",
            },
        }
        cached = flow._cached_anchor_for_global(info, 3.0, 4.0, "DAO", "Stable")
        self.assertIsNotNone(cached)
        np.testing.assert_array_equal(cached[0], stars)
        self.assertEqual(cached[1], (512, 512))

        # A lower local sigma means Global's adaptive loop would have produced
        # a different catalogue, so it must reread the FITS anchor.
        info["anchor_detection"]["sigma_used"] = 3.5
        self.assertIsNone(flow._cached_anchor_for_global(info, 3.0, 4.0, "DAO", "Stable"))

    def test_global_anchor_cache_avoids_fits_decode(self):
        stars = np.arange(70, dtype=np.float32).reshape(35, 2)
        info = {
            "anchor_path": Path("anchor.fits"),
            "anchor_stars": stars.tolist(),
            "anchor_shape": [128, 256],
            "fwhm": 3.0,
            "anchor_detection": {
                "fwhm": 3.0, "sigma": 4.0, "sigma_used": 4.0,
                "max_stars": 250, "engine": "DAO", "engine_profile": "Stable",
            },
        }
        with patch.object(flow, "load_fits_data", side_effect=AssertionError("FITS reread")):
            result = flow._detect_anchor_stars_task(info, 3.0, 4.0, "DAO", "Stable")
        np.testing.assert_array_equal(result[2], stars)
        self.assertEqual(result[1], (128, 256))
        self.assertIsNone(result[-1])

    def test_local_flow_batch_plan_caps_outer_memory_and_honours_override(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            batches = []
            for index in range(3):
                batch = root / f"batch_{index:02d}"
                batch.mkdir()
                fits.PrimaryHDU(np.zeros((8, 8), dtype=np.uint16)).writeto(batch / "01.fits")
                batches.append(batch)
            self.assertEqual(
                flow._local_flow_batch_plan(batches, {"memory_budget_mb": 512}, 4),
                (1, 4, 512),
            )
            self.assertEqual(
                flow._local_flow_batch_plan(
                    batches, {"memory_budget_mb": 512, "flow_batch_workers": 2}, 4
                ),
                (2, 2, 256),
            )

    def test_persisted_flow_fingerprint_rejects_changed_input(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            frame = root / "01.fits"
            frame.write_bytes(b"first")
            parameters = {"neighbor_window": 4, "min_stars": 4}
            persisted = {
                "batch_anchor": "01.fits",
                "parameters": parameters,
                "engine": "DAO",
                "engine_profile": "Stable",
                "transform_fallback": "Disabled",
            }
            persisted["input_fingerprint"] = flow._registration_input_fingerprint(
                [frame],
                {**parameters, "engine": "DAO",
                 "registration_strategy": "neighbor_bfs", "engine_profile": "Stable",
                 "fallback": "Disabled"},
            )
            self.assertTrue(flow._persisted_local_flow_is_current(root, persisted))
            frame.write_bytes(b"changed")
            self.assertFalse(flow._persisted_local_flow_is_current(root, persisted))

    def test_reference_change_cancellation_keeps_previous_revision(self):
        with TemporaryDirectory() as td:
            root = Path(td); batch = root / "batch_001"; batch.mkdir()
            payload = {
                "batch_anchor": "01.fits",
                "transform_revision": "old",
                "frames": {
                    "01.fits": {"status": "accepted", "matrix": np.eye(3).tolist()},
                    "02.fits": {"status": "accepted", "matrix": self._translation(2.0).tolist()},
                },
            }
            path = batch / "flow_local.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            before = path.read_bytes()
            event = Event(); event.set()
            result = flow.apply_reference_change(batch, "02.fits", root, event)
            self.assertEqual(result["status"], "cancelled")
            self.assertEqual(path.read_bytes(), before)

    def test_reference_change_write_failure_rolls_back_local_revision(self):
        with TemporaryDirectory() as td:
            root = Path(td); batch = root / "batch_001"; batch.mkdir()
            payload = {
                "batch_anchor": "01.fits",
                "transform_revision": "old",
                "frames": {
                    "01.fits": {"status": "accepted", "matrix": np.eye(3).tolist()},
                    "02.fits": {"status": "accepted", "matrix": self._translation(2.0).tolist()},
                },
            }
            path = batch / "flow_local.json"
            path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            before = path.read_bytes()
            with patch(
                "app.infrastructure.json_store.atomic_json_write",
                side_effect=[None, OSError("disk full")],
            ):
                result = flow.apply_reference_change(batch, "02.fits", root)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(path.read_bytes(), before)

    def test_reference_change_switches_one_manifest_to_immutable_local_global_pair(self):
        with TemporaryDirectory() as td:
            root = Path(td); batch = root / "batch_001"; batch.mkdir()
            first = batch / "01.fits"; second = batch / "02.fits"
            first.write_bytes(b"first"); second.write_bytes(b"second")
            parameters = {"neighbor_window": 4, "min_stars": 4}
            local = {
                "schema_version": 3,
                "batch_anchor": "01.fits",
                "transform_revision": "old",
                "parameters": parameters,
                "engine": "DAO",
                "engine_profile": "Stable",
                "transform_fallback": "Disabled",
                "registration_strategy": "neighbor_bfs",
                "frames": {
                    "01.fits": {"status": "accepted", "matrix": np.eye(3).tolist()},
                    "02.fits": {"status": "accepted", "matrix": self._translation(2).tolist()},
                },
            }
            local["input_fingerprint"] = flow._registration_input_fingerprint(
                [first, second], {**parameters, "registration_strategy": "neighbor_bfs",
                                  "engine": "DAO", "engine_profile": "Stable", "fallback": "Disabled"}
            )
            (batch / "flow_local.json").write_text(__import__("json").dumps(local), encoding="utf-8")
            global_data = {
                "schema_version": 3,
                "batches": {"batch_001": {"status": "accepted", "matrix": np.eye(3).tolist()}},
                "source_revisions": {"batch_001": "old"},
            }
            (root / "global_flow.json").write_text(__import__("json").dumps(global_data), encoding="utf-8")
            result = flow.apply_reference_change(batch, "02.fits", root, Event())
            self.assertEqual(result["status"], "success")
            manifest = __import__("json").loads((root / "flow_revision.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertTrue((root / manifest["local_flows"]["batch_001"]).exists())
            self.assertTrue((root / manifest["global_flow"]).exists())
            from astroalign_logic import load_global_flow, load_local_flow
            self.assertEqual(load_local_flow(batch)["batch_anchor"], "02.fits")
            np.testing.assert_allclose(
                np.asarray(load_global_flow(root)["batches"]["batch_001"]["matrix"]),
                self._translation(2),
            )

    def test_global_flow_seeds_all_direct_matches_before_neighbor_retry(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            batches = []
            for index in range(1, 5):
                batch = root / f"batch_{index:02d}"; batch.mkdir()
                fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(batch / "anchor.fits")
                local = {
                    "batch_anchor": "anchor.fits",
                    "anchor_metrics": {"star_count": 5, "fwhm": 2.0, "quality": 2.5},
                    "statistics": {"accepted_frames": 1, "total_frames": 1, "coverage": 1.0},
                    "frames": {"anchor.fits": {"status": "accepted", "matrix": np.eye(3).tolist()}},
                }
                (batch / "flow_local.json").write_text(__import__("json").dumps(local), encoding="utf-8")
                batches.append(batch)
            stars = np.asarray([[1, 1], [3, 1], [5, 1], [1, 5], [5, 5]], dtype=np.float32)
            metrics = {"status": "accepted", "matches": 5, "inliers": 5, "inlier_ratio": 1.0, "rms": 0.1}
            calls = []

            def detect(info, *args):
                return info, (8, 8), stars, 2.0, 5.0, np.zeros((8, 8), np.float32)

            def pair(ref_info, target_info, *args):
                key = (ref_info["batch_name"], target_info["batch_name"])
                calls.append(key)
                if key == ("batch_02", "batch_04"):
                    return None, {"status": "rejected", "reason": "no_overlap", "matches": 0, "inliers": 0, "inlier_ratio": 0.0, "rms": 999.0}
                if key == ("batch_03", "batch_04"):
                    return np.eye(3), dict(metrics)
                return np.eye(3), dict(metrics)

            with patch.object(flow, "_persisted_local_flow_is_current", return_value=True), \
                 patch.object(flow, "_detect_anchor_stars_task", side_effect=detect), \
                 patch.object(flow, "_estimate_global_pair", side_effect=pair):
                result = flow.process_all_flows(
                    root,
                    {"skip_local_flow": True, "global_master": "batch_02"},
                    lambda *_: None,
                    lambda *_: None,
                    Event(),
                )
            self.assertIn(result["status"], {"success", "partial"})
            persisted = __import__("json").loads((root / "global_flow.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["batches"]["batch_03"]["strategy"], "master_direct")
            self.assertEqual(persisted["batches"]["batch_04"]["strategy"], "neighbor_chain")
            self.assertEqual(calls.count(("batch_02", "batch_03")), 1)

    def test_neighbor_graph_reaches_both_ends_and_bypasses_bad_frame(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            files = []
            for index in range(1, 6):
                path = root / f"{index:02d}.fits"
                path.write_bytes(b"placeholder")
                files.append(path)
            stars = np.asarray([[2, 2], [4, 2], [6, 2], [2, 6], [6, 6]], dtype=np.float32)

            def prepared(path, *args):
                return path.name, {
                    "path": path,
                    "phase_data": np.zeros((8, 8), dtype=np.float32),
                    "stars": stars,
                    "fwhm": 2.0,
                    "metrics": {},
                    "status": "prepared",
                }

            def edge(ref_name, target_name, *_args, **_kwargs):
                # 02 is deliberately unusable; 01 is recovered from the
                # reference by the bounded four-frame fallback.
                if target_name == "02.fits":
                    return None, {"reason": "injected_bad_frame", "matches": 0, "inliers": 0, "inlier_ratio": 0.0, "rms": 999.0}, "bad"
                return np.eye(3), {"matches": 5, "inliers": 5, "inlier_ratio": 1.0, "rms": 0.1, "phase_response": 1.0, "spatial_inlier_coverage": 0.1}, "neighbor"

            limits = {"min_stars": 4, "min_inliers": 4, "min_ratio": 0.15, "max_rms": 4.0,
                      "max_translation": 1500.0, "max_rotation": 10.0, "min_scale": 0.95, "max_scale": 1.05}
            with patch.object(flow, "_process_single_frame", side_effect=prepared), patch.object(flow, "_attempt_local_edge", side_effect=edge):
                result = flow._build_local_registration_graph(
                    files,
                    root / "03.fits",
                    {"min_stars": 4, "neighbor_window": 4},
                    limits,
                    25.0,
                    4.0,
                    "DAO",
                    "Stable",
                    "Disabled",
                    1,
                    SimpleNamespace(max_in_flight=2),
                    lambda: False,
                    lambda *_: None,
                    root.name,
                )
            persisted, _anchor, _quality, accepted = result
            self.assertEqual(accepted, 4)
            self.assertEqual(persisted["frames"]["02.fits"]["status"], "rejected")
            self.assertEqual(persisted["frames"]["01.fits"]["status"], "accepted")
            self.assertEqual(persisted["frames"]["05.fits"]["status"], "accepted")
            self.assertEqual(persisted["registration_graph"]["strategy"], "neighbor_bfs")

    def test_confidence_uses_spatial_and_phase_evidence(self):
        limits = {"min_ratio": 0.15, "max_rms": 4.0}
        self.assertEqual(flow._classify_flow_confidence({"phase_response": .3, "inlier_ratio": .8, "rms": 1., "spatial_inlier_coverage": .04}, limits), "accepted")
        self.assertEqual(flow._classify_flow_confidence({"phase_response": .01, "inlier_ratio": .8, "rms": 1., "spatial_inlier_coverage": .04}, limits), "low_confidence")

    def test_header_budget_clamps_workers_without_loading_pixels(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "frame.fits"
            fits.PrimaryHDU(np.zeros((20, 20), dtype=np.uint16)).writeto(path)
            stars = np.array([[2, 2], [4, 2], [6, 2], [2, 6], [6, 6]], dtype=np.float32)
            result_frame = lambda p, *args: (p.name, {"path": p, "data": None, "phase_data": np.zeros((8, 8), np.float32), "stars": stars, "fwhm": 2., "metrics": {}, "status": "prepared"})
            with patch.object(flow, "_process_single_frame", side_effect=result_frame):
                result = flow.process_local_flow(Path(td), {"flow_workers": 8, "memory_budget_mb": 64, "min_stars": 4}, lambda _: None, Event())
            self.assertEqual(result["total_frames"], 1)
            self.assertLessEqual(result["valid_frames"], 1)

    def test_phase_correlation_is_reused_for_relaxed_attempt(self):
        with patch("astroflow_logic.cv2.phaseCorrelate", return_value=((0., 0.), 1.0)) as phase:
            flow._match_incremental_stars(np.zeros((0, 2)), np.zeros((0, 2)), (0., 0.), 1.)
        self.assertEqual(phase.call_count, 0)

    def test_manual_anchor_contract_preserves_all_names(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for name in ("01.fits", "02.fits", "03.fits"):
                fits.PrimaryHDU(np.zeros((8, 8), dtype=np.uint16)).writeto(root / name)
            stars = np.array([[2, 2], [4, 2], [6, 2], [2, 6], [6, 6]], dtype=np.float32)
            def prepared(p, *args):
                return p.name, {"path": p, "data": None, "phase_data": np.zeros((8, 8), np.float32), "stars": stars, "fwhm": 2., "metrics": {}, "status": "prepared"}
            with patch.object(flow, "_process_single_frame", side_effect=prepared):
                result = flow.process_local_flow(root, {"custom_anchors": {root.name: "02.fits"}, "min_stars": 4}, lambda _: None, Event())
            with (root / "flow_local.json").open(encoding="utf-8") as handle:
                persisted = __import__("json").load(handle)
            self.assertEqual(persisted["batch_anchor"], "02.fits")
            self.assertEqual(set(persisted["frames"]), {"01.fits", "02.fits", "03.fits"})


if __name__ == "__main__":
    unittest.main()
