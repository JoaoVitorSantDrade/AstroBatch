"""Stage registry and presentation composition helpers.

The registry deliberately knows how to assemble passive models, but it does
not know about the root Tk controller. The composition root supplies a small
set of variables and callbacks through :class:`StageContext`; views receive
only the resulting model and their parent widget.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

from .commands import (
    AlignCommand,
    BatchCommand,
    CalibrationCommand,
    FlowCommand,
    HDRCommand,
    StackCommand,
)
from .pipelines import execute_pipeline
from views.align_model import AlignViewModel
from views.align_view import AlignView
from views.batch_model import BatchViewModel
from views.batch_view import BatchView
from views.calibration_model import CalibrationViewModel
from views.calibration_view import CalibrationView
from views.flow_model import FlowViewModel
from views.flow_view import FlowView
from views.hdr_model import HDRViewModel
from views.hdr_view import HDRView
from views.stack_model import StackViewModel
from views.stacking_view import StackingView


@dataclass(frozen=True)
class StageContext:
    """Explicit bindings made available while composing one stage."""

    variables: Mapping[str, Any]
    callbacks: Mapping[str, Callable]
    custom_anchors: MutableMapping[str, str]
    background: str = "#ffffff"

    def var(self, name: str) -> Any:
        return self.variables[name]

    def callback(self, name: str) -> Callable:
        return self.callbacks[name]


@dataclass(frozen=True)
class StageDefinition:
    identifier: str
    label: str
    run_suffix: str
    view_factory: Callable
    model_factory: Callable[[StageContext], Any]
    command_builder: Callable | None
    execution_adapter: Callable
    operation_controls: tuple[str, str]
    view_controls: tuple[str, str] = ("run_button", "cancel_button")
    use_scrollable_host: bool = True

    def build_model(self, context: StageContext) -> Any:
        return self.model_factory(context)

    def build_view(self, parent: Any, context: StageContext) -> tuple[Any, Any]:
        model = self.build_model(context)
        return model, self.view_factory(parent, model)


def _build_calibration_model(context: StageContext) -> CalibrationViewModel:
    v, c = context.var, context.callback
    return CalibrationViewModel(
        v("calib_input"), v("calib_output"), v("apply_dark"), v("dark_path"),
        v("apply_flat"), v("flat_path"), v("calib_create_master"),
        v("calib_overwrite"),
        lambda: c("browse_dir")(v("calib_input")),
        lambda: c("browse_dir")(v("calib_output")),
        lambda: c("browse_file_or_dir")(v("dark_path")),
        lambda: c("browse_file_or_dir")(v("flat_path")),
        c("start_calibration"), c("cancel_processing"),
    )


def _build_batch_model(context: StageContext) -> BatchViewModel:
    v, c = context.var, context.callback
    return BatchViewModel(
        v("batch_input_dir"), v("batch_dir"), v("opt_method"), v("crop_size"),
        v("downsample_method"), v("downsample_scale"), v("threshold"),
        v("copy_files"), v("batch_overwrite"), v("dry_run"),
        lambda: c("browse_dir")(v("batch_input_dir")),
        lambda: c("browse_dir")(v("batch_dir")),
        c("toggle_opt_options"), c("start_batch"), c("cancel_processing"),
    )


def _build_flow_model(context: StageContext) -> FlowViewModel:
    v, c = context.var, context.callback
    start = c("start_flow_processing")
    cancel = c("cancel_processing")
    return FlowViewModel(
        v("batch_dir"), v("flow_global_master"), v("flow_fwhm"),
        v("flow_sigma"), v("flow_matching_radius"), v("flow_ransac"),
        v("flow_debug"), v("flow_min_stars"), v("flow_min_inliers"),
        v("flow_min_ratio"), v("flow_engine"), v("flow_profile"),
        v("flow_detector_engine"), v("flow_transform_fallback"),
        v("resource_memory"), v("resource_workers"),
        flow_temporal_enabled_var=v("flow_temporal_enabled"),
        flow_temporal_gap_var=v("flow_temporal_gap"),
        flow_temporal_seeing_sigma_var=v("flow_temporal_seeing_sigma"),
        custom_anchors=context.custom_anchors, bg=context.background,
        start=start, start_global=start, cancel=cancel,
        open_anchor_selector=c("open_anchor_selector"),
        show_preview=c("show_astroflow_preview"),
        show_visualization=c("show_flow_visualization"),
        show_flow_visualization=c("show_flow_visualization"),
        show_temporal_analysis=c("show_temporal_analysis"),
        save_settings=c("save_settings"),
        print_to_console=c("print_to_console"),
        start_operation=c("start_operation"), is_busy=c("is_busy"),
        BG=context.background,
        show_astroflow_preview=c("show_astroflow_preview"),
        start_flow_processing=start, cancel_processing=cancel,
        _start_operation=c("start_operation"),
    )


def _build_align_model(context: StageContext) -> AlignViewModel:
    v, c = context.var, context.callback
    start, cancel = c("start_align_processing"), c("cancel_processing")
    browse = c("browse_dir")
    return AlignViewModel(
        v("batch_dir"), v("align_output_dir"), v("align_debayer_pattern"),
        v("align_debayer_method"), v("align_interpolation"),
        v("align_rgb_registration"), v("align_rgb_registration_mode"), v("align_overwrite"),
        v("align_delete_intermediates"), v("align_dry_run"),
        v("align_keep_header"), v("align_compress_output"),
        v("align_profile"), v("align_warp_engine"), v("align_quality_gate"),
        v("align_quality_shift"), start, cancel,
        lambda: browse(v("batch_dir")), lambda: browse(v("align_output_dir")),
        browse_dir=browse, start_align_processing=start,
        cancel_processing=cancel,
    )


def _build_stack_model(context: StageContext) -> StackViewModel:
    v, c = context.var, context.callback
    browse = c("browse_dir")
    start, cancel = c("start_stacking"), c("cancel_processing")
    return StackViewModel(
        v("stack_input_dir"), v("stack_output_dir"),
        v("stack_selection_mode"), v("stack_selection_percentage"),
        v("stack_selection_percentage_text"), v("stack_selection_metric"),
        v("stack_trail_filter"), v("stack_min_roundness"),
        v("stack_min_shape_stars"), v("stack_method"),
        v("stack_rejection_method"), v("stack_rejection_low"),
        v("stack_rejection_high"), v("stack_normalize"),
        v("stack_normalize_method"), v("stack_dither_correction"),
        v("stack_output_name"), v("stack_output_bit_depth"),
        v("stack_compress"), v("stack_profile"), v("stack_reducer_engine"),
        context.background, lambda variable: browse(variable),
        lambda: browse(v("stack_output_dir")),
        c("use_align_output_for_stack"), c("apply_unguided_preset"),
        start, cancel, BG=context.background, browse_dir=browse,
        use_align_output_for_stack=c("use_align_output_for_stack"),
        start_stacking=start, cancel_processing=cancel,
    )


def _build_hdr_model(context: StageContext) -> HDRViewModel:
    v, c = context.var, context.callback
    return HDRViewModel(
        v("hdr_input"), v("hdr_output"), v("hdr_saturation"), v("hdr_noise"),
        v("hdr_rowband"), v("hdr_exptime"),
        lambda: c("browse_dir")(v("hdr_input")),
        lambda: c("browse_save_file")(v("hdr_output")),
        c("use_align_output_for_hdr"), c("start_hdr"), c("cancel_processing"),
    )


STAGE_DEFINITIONS = (
    StageDefinition(
        "Calibration", "1  Calibration", "calib", CalibrationView,
        _build_calibration_model, CalibrationCommand.from_values, execute_pipeline,
        ("btn_run_calib", "btn_cancel_calib"),
    ),
    StageDefinition(
        "Batch", "2  Batch", "batch", BatchView, _build_batch_model,
        BatchCommand.from_values, execute_pipeline,
        ("btn_run_batch", "btn_cancel_batch"),
    ),
    StageDefinition(
        "Flow", "3  Flow", "flow", FlowView, _build_flow_model,
        FlowCommand.from_values, execute_pipeline,
        ("btn_run_flow", "btn_cancel_flow"),
        view_controls=("btn_run_flow", "btn_cancel_flow"),
    ),
    StageDefinition(
        "Align", "4  Align", "align", AlignView, _build_align_model,
        AlignCommand.from_values, execute_pipeline,
        ("btn_run_align", "btn_cancel_align"),
        view_controls=("btn_run_align", "btn_cancel_align"),
    ),
    StageDefinition(
        "Stack", "5  Stack", "stack", StackingView, _build_stack_model,
        StackCommand.from_values, execute_pipeline,
        ("btn_run_stack", "btn_cancel_stack"),
        view_controls=("btn_run_stack", "btn_cancel_stack"),
        use_scrollable_host=False,
    ),
    StageDefinition(
        "HDR", "6  HDR", "hdr", HDRView, _build_hdr_model,
        HDRCommand.from_values, execute_pipeline,
        ("btn_run_hdr", "btn_cancel_hdr"),
    ),
)


def stage_definition(identifier: str) -> StageDefinition:
    for definition in STAGE_DEFINITIONS:
        if definition.identifier == identifier:
            return definition
    raise KeyError(identifier)
