from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    available_mb: int
    budget_mb: int
    estimated_working_mb: int
    safe_chunk_rows: int


def estimate_resources(frame_count: int, width: int, height: int, channels: int = 1, budget_mb: int | None = None) -> ResourceEstimate:
    available_mb = max(1, int(psutil.virtual_memory().available // (1024 * 1024)))
    effective_budget = min(available_mb * 3 // 4, budget_mb or max(512, available_mb // 2))
    bytes_per_row = max(1, frame_count * width * channels * 4 * 2)
    safe_rows = max(32, min(height, int(effective_budget * 1024 * 1024 // bytes_per_row)))
    estimated = int(min(effective_budget, bytes_per_row * min(height, 2048) // (1024 * 1024)))
    return ResourceEstimate(available_mb, effective_budget, estimated, safe_rows)
