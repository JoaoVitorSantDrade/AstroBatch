from __future__ import annotations

from pathlib import Path
from typing import Any

from astrobatch.core.fits import discover_fits, inspect_fits
from astrobatch.core.jobs import JobContext
from astrobatch.core.models import Artifact, Stage, StageResult
from astrobatch.core.resources import estimate_resources
from astrobatch.project.workspace import Project

STAGE_LABELS = {
    Stage.IMPORT: "Import", Stage.CALIBRATE: "Calibrate", Stage.BATCH: "Analyze", Stage.FLOW: "Flow", Stage.ALIGN: "Align", Stage.STACK: "Stack", Stage.REVIEW: "Review",
}


class PipelineService:
    """Typed V2 facade over the established scientific processing modules."""

    def __init__(self, project: Project):
        self.project = project

    def run(self, stage: Stage, context: JobContext) -> StageResult:
        return getattr(self, f"run_{stage.value}")(context)

    def _settings(self, stage: Stage) -> dict[str, Any]:
        return self.project.settings.setdefault(stage.value, {})

    def _source(self, stage: Stage, fallback: Stage | None = None) -> Path:
        settings = self._settings(stage)
        explicit = settings.get("input_dir")
        if explicit:
            source = Path(explicit).expanduser()
        elif fallback and (artifact := self.project.artifact_for(fallback)):
            source = artifact.filesystem_path
        else:
            source = Path(self.project.source_dir).expanduser()
        if not source.is_dir():
            raise ValueError(f"Select a valid input directory for {STAGE_LABELS[stage]}.")
        return source.resolve()

    def _output(self, stage: Stage) -> Path:
        configured = self._settings(stage).get("output_dir")
        output = Path(configured).expanduser() if configured else self.project.workspace.stage_output(stage)
        output.mkdir(parents=True, exist_ok=True)
        return output.resolve()

    @staticmethod
    def _callbacks(context: JobContext):
        def log(message: str) -> None:
            context.log(message)

        def progress(current: int, total: int, message: str = "") -> None:
            context.progress(current, total, message)

        return log, progress

    def run_import(self, context: JobContext) -> StageResult:
        source = self._source(Stage.IMPORT)
        frames = discover_fits(source)
        if not frames:
            raise ValueError("The selected source directory contains no FITS frames.")
        shape, _ = inspect_fits(frames[0])
        context.log(f"Found {len(frames)} FITS frames ({' × '.join(map(str, shape))}).")
        return StageResult(Stage.IMPORT, f"Imported {len(frames)} source frames", [Artifact("source_frames", str(source), "directory", {"frame_count": len(frames), "shape": shape})], {"frame_count": len(frames), "shape": shape})

    def run_calibrate(self, context: JobContext) -> StageResult:
        from astrobatch.processing.calibration import run_calibration_pipeline

        source = self._source(Stage.CALIBRATE, Stage.IMPORT)
        output = self._output(Stage.CALIBRATE)
        settings = self._settings(Stage.CALIBRATE)
        log, progress = self._callbacks(context)
        config = {"input_dir": str(source), "output_dir": str(output), "apply_dark": bool(settings.get("apply_dark", False)), "dark_path": settings.get("dark_path", ""), "apply_flat": bool(settings.get("apply_flat", False)), "flat_path": settings.get("flat_path", ""), "create_master": bool(settings.get("create_master", True)), "overwrite": bool(settings.get("overwrite", False))}
        run_calibration_pipeline(config, log, progress, context.cancel_event)
        return StageResult(Stage.CALIBRATE, "Calibration completed", [Artifact("calibrated_frames", str(output), "directory")])

    def run_batch(self, context: JobContext) -> StageResult:
        from astrobatch.processing.batch import ProcessingConfig, process_fits_logic

        source = self._source(Stage.BATCH, Stage.CALIBRATE)
        output = self._output(Stage.BATCH)
        settings = self._settings(Stage.BATCH)
        config = ProcessingConfig(input_dir=source, output_dir=output, threshold_factor=float(settings.get("threshold_factor", 3.0)), crop_size=int(settings.get("crop_size", 1000)), dry_run=bool(settings.get("dry_run", False)), copy_files=bool(settings.get("copy_files", True)), overwrite=bool(settings.get("overwrite", False)), opt_method=str(settings.get("opt_method", "Crop")), downsample_method=str(settings.get("downsample_method", "Nearest")), downsample_scale=float(settings.get("downsample_scale", 0.25)))
        log, progress = self._callbacks(context)
        processed, batches = process_fits_logic(config, log, progress, context.cancel_event)
        return StageResult(Stage.BATCH, f"Analyzed {processed} frames into {batches} batches", [Artifact("batches", str(output), "directory", {"processed": processed, "batches": batches})], {"processed": processed, "batches": batches})

    def run_flow(self, context: JobContext) -> StageResult:
        from astrobatch.processing.flow import process_all_flows

        source = self._source(Stage.FLOW, Stage.BATCH)
        log, progress = self._callbacks(context)
        process_all_flows(source, dict(self._settings(Stage.FLOW)), log, progress, context.cancel_event)
        return StageResult(Stage.FLOW, "Flow analysis completed", [Artifact("flow_data", str(source), "directory")])

    def run_align(self, context: JobContext) -> StageResult:
        from astrobatch.processing.align import process_all_alignments

        source = self._source(Stage.ALIGN, Stage.FLOW)
        output = self._output(Stage.ALIGN)
        log, progress = self._callbacks(context)
        processed, failed = process_all_alignments(source, output, dict(self._settings(Stage.ALIGN)), log, progress, context.cancel_event)
        return StageResult(Stage.ALIGN, f"Aligned {processed} frames ({failed} failed)", [Artifact("aligned_frames", str(output), "directory", {"processed": processed, "failed": failed})], {"processed": processed, "failed": failed})

    def run_stack(self, context: JobContext) -> StageResult:
        from astrobatch.processing.stacking import process_all_stacking

        source = self._source(Stage.STACK, Stage.ALIGN)
        output = self._output(Stage.STACK)
        settings = dict(self._settings(Stage.STACK))
        frames = discover_fits(source)
        if frames:
            shape, _ = inspect_fits(frames[0])
            height, width = shape[-2:]
            estimate = estimate_resources(len(frames), width, height, 1, settings.get("memory_budget_mb"))
            settings.setdefault("chunk_size", estimate.safe_chunk_rows)
            settings.setdefault("memory_budget_mb", estimate.budget_mb)
            context.log(f"Memory preflight: {estimate.available_mb} MiB free; using {estimate.safe_chunk_rows} row chunks.")
        settings.update({"input_dir": str(source), "output_dir": str(output), "output_bit_depth": "16-bit"})
        log, progress = self._callbacks(context)
        result = process_all_stacking(source, settings, progress, log, context.cancel_event)
        if not result or result.get("status") != "success":
            raise RuntimeError((result or {}).get("reason", "Stacking did not complete"))
        artifact = Artifact("stacked_image", str(result["output_path"]), "fits", {"frames": result.get("n_frames", 0)})
        return StageResult(Stage.STACK, f"Stacked {result.get('n_frames', 0)} frames", [artifact], result)

    def run_review(self, context: JobContext) -> StageResult:
        artifact = self.project.artifact_for(Stage.STACK)
        if artifact is None:
            raise ValueError("Run Stack before opening the final review.")
        return StageResult(Stage.REVIEW, "Stack is ready for review", [artifact])
