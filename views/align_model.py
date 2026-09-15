"""Bindings consumed by the passive Align form."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AlignViewModel:
    batch_dir_var: Any
    align_output_dir_var: Any
    align_debayer_pattern_var: Any
    align_debayer_method_var: Any
    align_interpolation_var: Any
    align_rgb_registration_var: Any
    align_overwrite_var: Any
    align_delete_intermediates_var: Any
    align_dry_run_var: Any
    align_keep_header_var: Any
    align_compress_output_var: Any
    align_profile_var: Any
    align_warp_engine_var: Any
    align_quality_gate_var: Any
    align_quality_shift_var: Any
    start: Callable
    cancel: Callable
    browse_batch: Callable
    browse_output: Callable
    browse_dir: Callable | None = None
    start_align_processing: Callable | None = None
    cancel_processing: Callable | None = None
