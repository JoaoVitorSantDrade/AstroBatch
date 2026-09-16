import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.application.runner import PipelineRunner, OperationResult
from app.application.pipelines import execute_pipeline
from app.application.commands import AlignCommand, BatchCommand, CalibrationCommand, FlowCommand, HDRCommand, ReferenceChangeCommand, StackCommand
from app.application.log_buffer import ActivityBuffer
from app.infrastructure.json_store import SettingsRepository
from views.flow_model import FlowViewModel


class RunnerTests(unittest.TestCase):
    def test_calibration_command_normalizes_paths_and_preserves_legacy_shape(self):
        command = CalibrationCommand.from_values(
            "~/lights", "~/calibrated", False, None, True, "flat.fits", True, False
        )
        config = command.to_legacy_config()
        self.assertTrue(Path(config["input_dir"]).is_absolute())
        self.assertTrue(Path(config["output_dir"]).is_absolute())
        self.assertFalse(config["apply_dark"])
        self.assertEqual(config["flat_path"], "flat.fits")

    def test_calibration_command_rejects_non_boolean_options(self):
        with self.assertRaises(ValueError):
            CalibrationCommand.from_values("lights", "out", apply_dark="yes")

    def test_pipeline_accepts_typed_calibration_command(self):
        command = CalibrationCommand.from_values("lights", "out")
        with patch("calibration_logic.run_calibration_pipeline", return_value={"status": "success"}) as run:
            result = execute_pipeline("Calibration", (command,), Mock(), Mock(), threading.Event())
        self.assertEqual(result.outcome, "success")
        self.assertEqual(run.call_args.args[0], command.to_legacy_config())

    def test_batch_command_validates_values_and_adapts_legacy_config(self):
        command = BatchCommand.from_values(
            "input", "output", 3, 1000, True, False, False,
            "Crop", "Nearest", .25,
        )
        self.assertEqual(command.to_legacy_config().crop_size, 1000)
        with self.assertRaises(ValueError):
            BatchCommand.from_values(
                "input", "output", 0, 1000, True, False, False,
                "Crop", "Nearest", .25,
            )

    def test_flow_command_adapter_is_accepted_by_pipeline(self):
        command = FlowCommand.from_values(
            "batch", custom_anchors={}, global_master="Auto", fwhm=4,
            sigma=5, matching_radius=15, ransac=3, debug_images=False,
            min_stars=4, min_inliers=4, min_ratio=.15, max_stars=150,
            engine="DAO", engine_profile="Stable", detector_engine="",
            transform_fallback="Disabled", memory_budget_mb=512, flow_workers=2,
        )
        with patch("astroflow_logic.process_all_flows", return_value={"status": "success"}) as run:
            result = execute_pipeline("Flow", (command,), Mock(), Mock(), threading.Event())
        self.assertEqual(result.outcome, "success")
        self.assertEqual(run.call_args.args[0], command.batch_dir)
        self.assertEqual(run.call_args.args[1], command.to_legacy_config())

    def test_flow_command_validates_temporal_controls_and_preserves_defaults(self):
        values = dict(
            custom_anchors={}, global_master="Auto", fwhm=4, sigma=5,
            matching_radius=15, ransac=3, debug_images=False,
            min_stars=4, min_inliers=4, min_ratio=.15, max_stars=150,
            engine="DAO", engine_profile="Stable", detector_engine="",
            transform_fallback="Disabled", memory_budget_mb=512, flow_workers=2,
        )
        command = FlowCommand.from_values("batch", **values)
        config = command.to_legacy_config()
        self.assertTrue(config["temporal_analysis_enabled"])
        self.assertEqual(config["temporal_gap_minutes"], 15.0)
        self.assertEqual(config["temporal_seeing_sigma"], 3.0)
        with self.assertRaises(ValueError):
            FlowCommand.from_values("batch", **values, temporal_gap_minutes=0)

    def test_flow_view_model_keeps_legacy_positional_constructor_order(self):
        values = [object() for _ in range(16)]
        model = FlowViewModel(*values)
        self.assertIs(model.resource_workers_var, values[15])
        self.assertEqual(model.custom_anchors, {})
        self.assertIsNone(model.flow_temporal_enabled_var)

    def test_align_command_adapter_preserves_legacy_arguments(self):
        command = AlignCommand.from_values(
            "base", "output", debayer_pattern="Auto", debayer_method="Bilinear",
            interpolation="Lanczos", overwrite=False, dry_run=False,
            keep_header=True, delete_intermediates=False, compress_output=True,
            engine_profile="Stable", warp_engine="", memory_budget_mb=512,
            workers=2, quality_gate=False, quality_max_shift=1.5,
        )
        with patch("astroalign_logic.process_all_alignments", return_value=(1, 0)) as run:
            result = execute_pipeline("Align", (command,), Mock(), Mock(), threading.Event())
        self.assertEqual(result.outcome, "success")
        self.assertEqual(run.call_args.args[0], command.base_dir)
        self.assertEqual(run.call_args.args[1], command.output_dir)
        self.assertEqual(run.call_args.args[2], command.to_legacy_config())

    def test_align_command_validates_chromatic_registration_mode(self):
        command = AlignCommand.from_values(
            "base", "output", rgb_registration=True,
            rgb_registration_mode="similarity", overwrite=False,
            dry_run=False, keep_header=True, delete_intermediates=False,
            compress_output=True, engine_profile="Stable", warp_engine="",
            memory_budget_mb=512, workers=2, quality_gate=False,
            quality_max_shift=1.5,
        )
        self.assertEqual(command.config["rgb_registration_mode"], "similarity")
        hybrid = AlignCommand.from_values(
            "base", "output", rgb_registration=True,
            rgb_registration_mode="hybrid", overwrite=False,
            dry_run=False, keep_header=True, delete_intermediates=False,
            compress_output=True, engine_profile="Stable", warp_engine="",
            memory_budget_mb=512, workers=2, quality_gate=False,
            quality_max_shift=1.5,
        )
        self.assertEqual(hybrid.config["rgb_registration_mode"], "hybrid")
        with self.assertRaises(ValueError):
            AlignCommand.from_values(
                "base", "output", rgb_registration=True,
                rgb_registration_mode="radial", overwrite=False,
                dry_run=False, keep_header=True, delete_intermediates=False,
                compress_output=True, engine_profile="Stable", warp_engine="",
                memory_budget_mb=512, workers=2, quality_gate=False,
                quality_max_shift=1.5,
            )

    def test_reference_change_command_uses_shared_runner_adapter(self):
        command = ReferenceChangeCommand.from_values("batch_001", "03.fits", "session")
        with patch("astroflow_logic.apply_reference_change", return_value={"status": "success"}) as apply:
            result = execute_pipeline("ReferenceChange", (command,), Mock(), Mock(), threading.Event())
        self.assertEqual(result.outcome, "success")
        self.assertEqual(apply.call_args.args[:3], (command.batch_dir, "03.fits", command.base_dir))

    def test_stack_command_owns_validation_and_adapts_settings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); input_dir = root / "aligned"; input_dir.mkdir()
            command = StackCommand.from_values(
                input_dir, root / "stacked", min_roundness=.7,
                min_shape_stars=5, selection_percentage=80,
                rejection_low=3, rejection_high=3,
                trail_filter_enabled=False, normalize=True,
                compress_output=True, apply_dither_correction=False,
                engine_profile="Stable",
            )
            self.assertEqual(command.to_legacy_config()["output_bit_depth"], "16-bit")
            custom = StackCommand.from_values(
                input_dir,
                root / "stacked_custom",
                feature_profile="Intelligent",
                selection_profile="Custom",
                selection_weights="fwhm=0.4,snr=0.6",
                min_roundness=.7,
                min_shape_stars=5,
                selection_percentage=80,
                rejection_low=3,
                rejection_high=3,
                trail_filter_enabled=False,
                normalize=True,
                compress_output=True,
                apply_dither_correction=False,
                engine_profile="Stable",
            )
            self.assertEqual(custom.config["selection_weights"], {"fwhm": .4, "snr": .6})
            with self.assertRaises(ValueError):
                StackCommand.from_values(
                    input_dir,
                    root / "stacked_invalid_custom",
                    selection_profile="Custom",
                    selection_weights="fwhm=-1,snr=nan",
                )
            with self.assertRaises(ValueError):
                StackCommand.from_values(input_dir, root / "stacked", min_roundness=1.5)

    def test_hdr_command_owns_hdr_config_validation_and_adapts_pipeline(self):
        command = HDRCommand.from_values({
            "input_paths": ["one.fits", "two.fits"],
            "output_path": "hdr.fits",
            "row_band": 64,
            "noise_floor": 1.0,
        })
        self.assertEqual(command.to_legacy_config().row_band, 64)
        with self.assertRaises(ValueError):
            HDRCommand.from_values({
                "input_paths": ["one.fits"],
                "output_path": "hdr.fits",
                "row_band": 0,
            })

    def test_pipeline_accepts_typed_hdr_command(self):
        command = HDRCommand.from_values({
            "input_paths": ["one.fits", "two.fits"],
            "output_path": "hdr.fits",
        })
        with patch("hdr_logic.run_hdr_pipeline", return_value={"status": "success"}) as run:
            result = execute_pipeline("HDR", (command,), Mock(), Mock(), threading.Event())
        self.assertEqual(result.outcome, "success")
        self.assertIs(run.call_args.args[0], command.to_legacy_config())

    def test_progress_coalesced_completion_once_on_consumer_thread(self):
        runner=PipelineRunner(Mock())
        def operation(log, progress, cancel):
            for i in range(10000): progress(i,10000,'work')
            return OperationResult('success','done')
        thread=runner.start('Test',operation); thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(runner.busy)
        with self.assertRaises(RuntimeError): runner.start('Second',operation)
        progress, result=runner.drain()
        self.assertEqual(progress.current,9999); self.assertEqual(result.outcome,'success')
        self.assertEqual(runner.drain(),(None,None)); self.assertFalse(runner.busy)

    def test_failure_and_start_failure_unlock(self):
        log=Mock(); runner=PipelineRunner(log)
        def broken(*args): raise RuntimeError('disk full')
        runner.start('Test',broken).join(3)
        self.assertEqual(runner.drain()[1].outcome,'failed'); log.assert_called_once()
        with patch('threading.Thread.start',side_effect=RuntimeError('start failed')):
            with self.assertRaises(RuntimeError): runner.start('Test',broken)
        self.assertFalse(runner.busy)

    def test_cancel_cannot_be_reported_as_success(self):
        runner=PipelineRunner(Mock()); entered=threading.Event()
        def operation(log,progress,cancel):
            entered.set(); cancel.wait(3)
            return OperationResult('success','done')
        thread=runner.start('Test',operation); self.assertTrue(entered.wait(2))
        runner.cancel(); thread.join(3)
        self.assertEqual(runner.drain()[1].outcome,'cancelled')

    def test_partial_align_and_empty_calibration_are_not_success(self):
        with patch('astroalign_logic.process_all_alignments',return_value=(2,1)):
            result=execute_pipeline('Align',(Path('.'),Path('.'),{}),Mock(),Mock(),threading.Event())
        self.assertEqual(result.outcome,'partial')
        with patch('calibration_logic.run_calibration_pipeline',return_value=None):
            result=execute_pipeline('Calibration',({},),Mock(),Mock(),threading.Event())
        self.assertEqual(result.outcome,'failed')

    def test_activity_flood_retains_errors_and_reports_loss(self):
        buffer=ActivityBuffer(capacity=5)
        buffer.put('ERROR: disk failure')
        for i in range(50): buffer.put(f'progress {i}')
        self.assertEqual(buffer.qsize(),5)
        self.assertIn('46 mensagens',buffer.get_nowait())
        self.assertIn('ERROR',buffer.get_nowait())
        for _ in range(4): buffer.get_nowait()
        with self.assertRaises(queue.Empty): buffer.get_nowait()

    def test_settings_migrate_and_failed_replace_preserves_file(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'settings.json'; repo=SettingsRepository(path)
            repo.save({'AstroFlow':{'sigma':3}})
            old=path.read_bytes(); self.assertEqual(repo.load()['_schema_version'],2)
            with patch('app.infrastructure.json_store.os.replace',side_effect=OSError('disk')):
                with self.assertRaises(OSError): repo.save({'changed':True})
            self.assertEqual(path.read_bytes(),old)
            self.assertEqual(list(path.parent.glob('*.tmp')),[])


if __name__=='__main__': unittest.main()
