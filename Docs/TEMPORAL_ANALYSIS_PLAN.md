# Temporal session analysis

## Delivered

- `temporal_analysis.py` parses the supported ISO/FITS `DATE-OBS` forms,
  preserves an original timezone when present, and marks missing or malformed
  timestamps as `unknown`.
- Frames are ordered by normalized UTC time (stable natural-name tie-breaker).
  A gap strictly greater than the configured default of 15 minutes begins a
  new logical group. Untimed frames are retained in an explicit `unknown`
  group.
- Flow frame JSON receives timestamp fields without removing any legacy keys.
  Batch and session `temporal_analysis.json` sidecars are written atomically.
- Seeing summaries use existing Flow FWHM, roundness, star-count and quality
  metrics. Robust outlier suggestions are advisory only and are surfaced by a
  native Flow chart; no frame is deleted, reweighted or automatically excluded.

## Deliberate non-scope

Thermal dark/flat interpolation and field-rotation prediction are not inferred
from time alone and remain outside this delivery. They require validated
temperature/exposure metadata and mount/sky geometry, respectively.
