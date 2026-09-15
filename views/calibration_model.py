"""Bindings consumed by the passive Calibration form."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CalibrationViewModel:
    input_dir: Any
    output_dir: Any
    apply_dark: Any
    dark_path: Any
    apply_flat: Any
    flat_path: Any
    create_master: Any
    overwrite: Any
    browse_input: Callable
    browse_output: Callable
    browse_dark: Callable
    browse_flat: Callable
    start: Callable
    cancel: Callable
