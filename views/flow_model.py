"""Bindings consumed by the passive Flow form and its preview controller."""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class FlowViewModel:
    batch_dir_var: Any
    flow_global_master_var: Any
    flow_fwhm_var: Any
    flow_sigma_var: Any
    flow_matching_radius_var: Any
    flow_ransac_var: Any
    flow_debug_var: Any
    flow_min_stars_var: Any
    flow_min_inliers_var: Any
    flow_min_ratio_var: Any
    flow_engine_var: Any
    flow_profile_var: Any
    flow_detector_engine_var: Any
    flow_transform_fallback_var: Any
    resource_memory_var: Any
    resource_workers_var: Any
    custom_anchors: dict[str, str] = field(default_factory=dict)
    bg: str = "#ffffff"
    start: Callable | None = None
    start_global: Callable | None = None
    cancel: Callable | None = None
    open_anchor_selector: Callable | None = None
    show_preview: Callable | None = None
    show_visualization: Callable | None = None
    show_flow_visualization: Callable | None = None
    save_settings: Callable | None = None
    print_to_console: Callable | None = None
    start_operation: Callable | None = None
    is_busy: Callable | None = None
    worker: Any = None
    BG: str = "#ffffff"
    show_astroflow_preview: Callable | None = None
    start_flow_processing: Callable | None = None
    cancel_processing: Callable | None = None
    _start_operation: Callable | None = None
    # Append optional fields so every pre-temporal positional constructor
    # keeps exactly the same argument order.
    flow_temporal_enabled_var: Any = None
    flow_temporal_gap_var: Any = None
    flow_temporal_seeing_sigma_var: Any = None
    show_temporal_analysis: Callable | None = None
