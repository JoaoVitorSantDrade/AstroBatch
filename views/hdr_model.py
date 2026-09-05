"""Bindings and commands consumed by HDRView; independent of the root UI."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class HDRViewModel:
    input_folder: Any
    output_file: Any
    saturation: Any
    noise: Any
    row_band: Any
    exposure: Any
    browse_input: Callable
    browse_output: Callable
    use_align_output: Callable
    start: Callable
    cancel: Callable
