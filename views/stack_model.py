"""Bindings consumed by the passive Stack form."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class StackViewModel:
    stack_input_dir_var: Any
    stack_output_dir_var: Any
    stack_selection_mode_var: Any
    stack_selection_percentage_var: Any
    stack_selection_percentage_text_var: Any
    stack_selection_metric_var: Any
    stack_trail_filter_var: Any
    stack_min_roundness_var: Any
    stack_min_shape_stars_var: Any
    stack_method_var: Any
    stack_rejection_method_var: Any
    stack_rejection_low_var: Any
    stack_rejection_high_var: Any
    stack_normalize_var: Any
    stack_normalize_method_var: Any
    stack_dither_correction_var: Any
    stack_output_name_var: Any
    stack_output_bit_depth_var: Any
    stack_compress_var: Any
    stack_profile_var: Any
    stack_reducer_engine_var: Any
    bg: str
    browse_input: Callable
    browse_output: Callable
    use_align_output: Callable
    apply_unguided_preset: Callable
    start: Callable
    cancel: Callable
    BG: str = "#ffffff"
    browse_dir: Callable | None = None
    use_align_output_for_stack: Callable | None = None
    start_stacking: Callable | None = None
    cancel_processing: Callable | None = None
    # New feature controls are optional at the model boundary so third-party
    # callers using the original positional constructor remain compatible.
    stack_feature_profile_var: Any = None
    stack_selection_profile_var: Any = None
    stack_selection_weights_var: Any = None
    stack_trail_policy_var: Any = None
    stack_reduction_storage_var: Any = None
    stack_spill_directory_var: Any = None
    stack_spill_limit_var: Any = None
