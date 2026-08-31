from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProcessingConfig:
    input_dir: Path
    output_dir: Path
    threshold_factor: float
    crop_size: int
    dry_run: bool
    copy_files: bool
    overwrite: bool
    opt_method: str
    downsample_method: str
    downsample_scale: float