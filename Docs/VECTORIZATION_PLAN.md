# CPU Vectorization Tracker

## Scope

CPU-only optimizations in `astroflow_logic.py`, `astroalign_logic.py`, and
`stacking_logic.py`. Public configuration and optional CuPy behavior remain
unchanged. Performance is measured with
`python benchmarks/vectorization_benchmark.py`; timing comparisons are valid
only for the same hardware and dependency environment.

## Stage 1 — Baseline and tracker — Complete

- Added deterministic regression coverage in `tests/test_vectorization_paths.py`.
- Added the repeatable synthetic workload benchmark in
  `benchmarks/vectorization_benchmark.py`.
- Pre-change baseline captured on 2026-09-02 for 180 stars / 15 x 128 x 128
  stack blocks: incremental matching 0.000351s, asterism hashing 0.040261s,
  sigma rejection 0.019017s, masked mean 0.001611s (best of three).
- Alignment baseline could not run in this workspace because
  `colour_demosaicing` and `scikit-image` are not installed.

## Stage 2 — AstroFlow — Complete

- Vectorized KD-tree candidate filtering, stable distance ordering, and unique
  previous-star selection in incremental matching.
- Kept the fixed six-pair quad calculation scalar after benchmarking confirmed
  that temporary-array allocation was slower than its tiny loop.

## Stage 3 — AstroAlign — Complete

- Replaced repeated per-channel affine warps with one OpenCV interleaved-image
  affine warp; valid-mask generation uses the same native path.
- Identity, translation, and legacy cubic-mode regression tests pass in the
  workspace virtual environment. Bicubic/Lanczos retain the legacy scikit-image
  kernel because OpenCV cubic interpolation is not pixel compatible.

## Stage 4 — Stacking — Complete

- Added mask-aware vectorized mean, sum, min, max, and median combining to
  avoid the old writable NaN-filled block copy in the no-rejection CPU path.
- Sigma/MAD/Winsorized behavior and GPU routing are unchanged.

## Final validation — Complete

- Full suite: 24 tests passed in the workspace virtual environment.
- Post-change (same synthetic 180-star / 15 x 128 x 128 workload, best of
  five): incremental matching 0.000187s; sigma rejection 0.021162s
  (unchanged code path); masked mean 0.000433s. Asterism hashing remains on the
  original scalar implementation after its vectorized variant regressed.
- RGB bilinear warp: 0.002011s native versus 0.015398s legacy per-channel warp
  on a 512 x 512 image. Nearest/bilinear have regression coverage; cubic modes
  preserve the prior exact implementation.
