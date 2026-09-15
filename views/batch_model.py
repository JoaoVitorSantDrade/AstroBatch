"""Bindings consumed by the passive Batch form."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BatchViewModel:
    input_dir: Any
    output_dir: Any
    opt_method: Any
    crop_size: Any
    downsample_method: Any
    downsample_scale: Any
    threshold: Any
    copy_files: Any
    overwrite: Any
    dry_run: Any
    browse_input: Callable
    browse_output: Callable
    toggle_options: Callable
    start: Callable
    cancel: Callable
