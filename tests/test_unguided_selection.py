import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import stacking_logic as stack
import astroflow_logic as flow


def frame(name, roundness=None, count=12):
    return stack.FrameInfo(Path(name), name, "batch_001",
                           {"roundness": roundness, "shape_star_count": count},
                           10.0, 20.0, 2.0, 5.0, 0.0, False, (16, 16), "Mono", 1)


class UnguidedSelectionTests(unittest.TestCase):
    def test_filter_keeps_all_good_subs_without_percentage_loss(self):
        frames = [frame("round.fits", .95), frame("trail.fits", .4),
                  frame("old.fits"), frame("sparse.fits", .99, 1)]
        config = stack.StackingConfig(selection_mode="All", trail_filter_enabled=True)
        self.assertEqual([f.name for f in stack.select_frames(frames, config)], ["round.fits"])

    def test_empty_filter_never_restores_rejected_frames(self):
        frames = [frame("bad.fits", .2), frame("unknown.fits")]
        for mode in ("All", "BestPercentage"):
            config = stack.StackingConfig(selection_mode=mode, trail_filter_enabled=True)
            self.assertEqual(stack.select_frames(frames, config), [])

    def test_legacy_metrics_work_with_filter_disabled(self):
        frames = [frame("old.fits")]
        self.assertEqual(stack.select_frames(frames, stack.StackingConfig()), frames)

    def test_roundness_selection_does_not_fallback_to_unknown_shapes(self):
        config = stack.StackingConfig(selection_metric="roundness")
        self.assertEqual(stack.select_frames([frame("old.fits")], config), [])
        self.assertEqual(stack.select_frames([frame("bad.fits", 1.5)], config), [])

    def test_ties_do_not_depend_on_inspection_completion_order(self):
        frames = [frame(f"{i}.fits", .9) for i in range(10)]
        config = stack.StackingConfig(selection_percentage=40)
        self.assertEqual(stack.select_frames(frames, config), stack.select_frames(frames[::-1], config))

    def test_flow_metadata_to_fits_inspection_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for name, roundness in (("good", .94), ("trailed", .3)):
                path = root / f"{name}.fits"
                fits.PrimaryHDU(np.ones((16, 16), dtype=np.uint16)).writeto(path)
                metadata = {"frames": {path.name: {"roundness": roundness,
                            "shape_star_count": 12, "fwhm": 2, "star_count": 20}}}
                frames.append(stack._inspect_single_frame(path, "batch_001", {"batch_001": metadata}))
            config = stack.StackingConfig(output_dir=root, selection_mode="All", trail_filter_enabled=True)
            selected = stack.select_frames(frames, config)
            report = json.loads(stack.write_selection_report(frames, selected, config).read_text(encoding="utf-8"))
            self.assertEqual(report["selected_frames"], 1)
            self.assertEqual([row["reason"] for row in report["frames"]], ["selected", "elongated_stars"])
            self.assertEqual(report["stage"], "selection_before_stacking")

    def test_invalid_filter_parameters_are_rejected(self):
        for threshold in (float("nan"), float("inf"), -1, 2):
            with self.assertRaises(ValueError):
                stack._validate_config(stack.StackingConfig(min_roundness=threshold))
        with self.assertRaises(ValueError):
            stack._validate_config(stack.StackingConfig(min_shape_stars=0))

    def test_synthetic_flow_to_stack_rejects_trailing_and_saves_uint16(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch_001"
            batch.mkdir()
            yy, xx = np.indices((192, 192))
            for index in range(5):
                data = np.full(xx.shape, 100.0)
                sx, sy = (2.0, 2.0) if index < 4 else (4.0, 1.2)
                for y in (28, 68, 108, 148):
                    for x in (28, 68, 108, 148):
                        data += 2000 * np.exp(-((xx-x)**2/(2*sx*sx) + (yy-y)**2/(2*sy*sy)))
                fits.PrimaryHDU(data.astype(np.uint16)).writeto(batch / f"frame_{index}.fits")
            flow.process_local_flow(batch, {"engine": "opencv-components", "engine_profile": "Fast", "sigma": 3.0}, lambda _: None)
            metadata = json.loads((batch / "flow_local.json").read_text(encoding="utf-8"))
            self.assertGreater(metadata["frames"]["frame_0.fits"]["roundness"], .9)
            self.assertLess(metadata["frames"]["frame_4.fits"]["roundness"], .65)
            config = stack.StackingConfig(base_dir=root, input_dir=root, output_dir=root / "out",
                                          trail_filter_enabled=True, selection_mode="All",
                                          method="Mean", rejection_method="SigmaClip", normalize=False,
                                          workers=1, chunk_size=64)
            result = stack.process_stacking(config)
            self.assertEqual(result["status"], "success", result)
            self.assertEqual(result["n_frames"], 4)
            report = json.loads(Path(result["selection_report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["frames"][-1]["reason"], "elongated_stars")
            with fits.open(result["output_path"], memmap=False) as hdus:
                self.assertEqual(hdus[0].data.dtype.kind, "u")
                self.assertEqual(hdus[0].data.dtype.itemsize, 2)
                self.assertTrue(hdus[0].header["TRAILFLT"])


if __name__ == "__main__":
    unittest.main()
