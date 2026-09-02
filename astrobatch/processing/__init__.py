"""Scientific processing engines used by AstroBatch V2 services.

These modules are deliberately UI-agnostic.  Public orchestration belongs in
``astrobatch.services`` and communicates with the desktop application through
``JobContext`` events.
"""

from . import align, batch, calibration, cpu_kernels, flow, stacking

__all__ = ["align", "batch", "calibration", "cpu_kernels", "flow", "stacking"]
