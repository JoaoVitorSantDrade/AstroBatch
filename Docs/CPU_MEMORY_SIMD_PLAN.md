# CPU Memory, Cache, and SIMD Upgrade Tracker

## Stage 1 — CPU runtime baseline — Complete

- Added Numba and Bottleneck; removed CuPy requirements.
- Added cacheable, non-fastmath CPU kernels and background warm-up.
- Added bitwise tests for calibration, extrema, and master creation.

## Stage 2 — GPU removal — Complete

- Removed GPU configuration, UI, metadata, execution paths, and dependencies.
- Legacy `use_gpu` fields are ignored by the stack config parser.

## Stage 3 — Bounded-RAM calibration — Complete

- Master creation uses a reusable full-width row-band buffer and temporary
  memory-mapped output instead of retaining all source frames.
- Band height is derived from free RAM and frame geometry.

## Stage 4 — Cache-local stacking — Complete

- Numba handles bitwise-safe masked extrema; median and rejection keep NumPy's
  established exact reduction path.
- GPU transfers and GPU buffer allocations are removed.

## Stage 5 — Flow and alignment locality — Complete

- Alignment now writes directly into a contiguous HWC output buffer, avoiding
  RGB channel-list and final-stack allocation.
- Flow retains its measured scalar quad hash implementation.

## Validation

- Full suite: 29 tests passed in the workspace virtual environment.
- Warm-cache run: 0.149s. The 15 x 256 x 256 stacking band allocated 3.75 MiB
  of values plus 0.94 MiB of validity data.
- Warm benchmarks: flow matching 0.000158s, asterism hashing 0.037834s,
  masked mean 0.002004s, sigma rejection 0.067229s, and RGB bilinear warp
  0.000661s. Peak RSS deltas are emitted by the benchmark runner.
