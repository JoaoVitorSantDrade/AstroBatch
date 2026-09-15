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

- Historical Stage 1 suite: 29 tests passed in the workspace virtual
  environment; the current suite is reported in Stage 6 validation below.
- Warm-cache run: 0.149s. The 15 x 256 x 256 stacking band allocated 3.75 MiB
  of values plus 0.94 MiB of validity data.
- Warm benchmarks: flow matching 0.000158s, asterism hashing 0.037834s,
  masked mean 0.002004s, sigma rejection 0.067229s, and RGB bilinear warp
  0.000661s. Peak RSS deltas are emitted by the benchmark runner.

- Current full suite: 185 tests passed, 1 skipped, and 13 subtests passed in
  the workspace virtual environment.

## Stage 6 — Ryzen AVX2 runtime policy and end-to-end gate

- Added `cpu_runtime.py` to expose NumPy host SIMD capabilities, cap Numba at
  eight physical cores on the Ryzen 7 5800X, and report the active budget.
- Numba kernels now release the GIL and the large independent extrema path can
  use `prange` while retaining the original frame traversal and bitwise
  selection semantics. No `fastmath` or reordered floating-point reduction was
  introduced.
- `benchmarks/pipeline_benchmark.py` creates a deterministic versioned corpus
  containing a small mono load plus dedicated-camera mono/RGB loads, warms the
  JIT, pins a reproducible logical-CPU slice when the OS permits it, records
  per-stage timing/RSS, inspects generated assembly, and compares Stable with
  the opt-in Fast reducer. With `--baseline-json` it reports the 4/3x gate
  against the clean checkout (at least 25% less wall time), rather than using
  Fast as a scientific baseline.

Reprodução do gate final:

```text
.venv/Scripts/python.exe benchmarks/pipeline_benchmark.py --workers 4 --repeats 7 --baseline-json Docs/pipeline_benchmark_baseline_20260915_w4.json --json-out Docs/pipeline_benchmark_official_w4_20260915.json
```

### Gate measurement (2026-09-15)

On the available Ryzen-shaped host (`8` physical / `16` logical cores, AVX2),
seven warm runs over the small + 512×512 mono/RGB corpus (20 frames total),
including `SigmaClip` rejection and temporal metadata, measured a Stable median
of `2.4867 s` with one worker and `2.1857 s` with four workers. Peak RSS was
`283.4 MiB` and `311.3 MiB`, respectively. Stable output digests, masks, counts
and scientific metadata remained identical across the runs and all products
were `uint16`. The benchmark pinned logical CPUs `0–7`, restored the original
affinity, and confirmed YMM/AVX2 vector instructions in uncached reducer probes.

The clean-checkout baselines were `2.7551 s` (one worker) and `2.4863 s` (four
workers), so the current end-to-end reductions are about `9.7%` and `12.1%`;
both remain below the required `25%` (`4/3x`) gate. The opt-in Fast reducer did
not improve this corpus (`2.4906 s` at one worker and `2.2414 s` at four), so
Stable remains the default and no precision relaxation is justified. Full JSON
evidence: [pipeline_benchmark_20260915_final_w1.json](pipeline_benchmark_20260915_final_w1.json)
and [pipeline_benchmark_20260915_final_w4.json](pipeline_benchmark_20260915_final_w4.json).

After that seven-run matrix, Align received a one-open FITS path that consumes
pixels, header and masks together. The combined-loader regression test confirms
one `fits.open` and byte-identical masks; a one-run smoke benchmark also
confirmed the AVX2 assembly/digest gate, but is intentionally not promoted to a
throughput claim. The final seven-run rerun after the bounded anchor window is
recorded below.

### Final checkout rerun after bounded anchor window

The subsequent optimization replaced repeated finite-vector medians with an
exact NumPy-compatible partition helper and made the internal DAO source sort
descending in one pass. The final official run used seven warm repetitions,
four workers, logical CPU affinity `0–7`, and
`--baseline-json Docs/pipeline_benchmark_baseline_20260915_w4.json`.

Stable median was `1.8415 s` (stdev `0.2103 s`, Flow `1.2998 s`, Align
`0.2572 s`, Stack `0.2817 s`, peak RSS `312.6 MiB`) and Fast median was
`1.8346 s`. Against the clean Stable baseline `2.4863 s` (peak RSS `339.7
MiB`), the measured speedup is `1.3501×` and reduction `25.93%`; the mandatory
gate is **passed**. The Stable digest, `uint16` product, masks and counts stayed
constant, and all inspected Numba probes still contained YMM/AVX2 instructions.
Full evidence: [pipeline_benchmark_official_w4_20260915.json](pipeline_benchmark_official_w4_20260915.json).

## Stage 7 — Real-capture I/O locality — Complete for the bounded audit

The Lagoon capture contains 1,242 aligned RGB FITS (`3856×2180`,
`BZERO=32768`, `68.08 GiB`) plus 1,253 Flow frames.  Astropy's scaled `.data`
path cannot be memory-mapped, so the Stack now opens raw storage with mmap and
restores `BSCALE/BZERO/BLANK` in `float32` per band.  RGB frames without dither
are read as one `[C,Y,X]` section per frame/band and reduced channel by
channel in the same Stable order.  The persistent cache remains limited to
tile-compressed inputs, avoiding a second 68 GiB tree beside an uncompressed
capture.

The row-band budget uses 12 bytes/pixel/channel (previously 24).  A bounded
64-frame real-resolution read measured `18.99 s` at the old factor and
`14.26 s` at factor 12 (about `24.9%` less wall time); observed peak RSS was
`3.53` versus `3.80 GiB`, below the configured 4 GiB leaf budget.  A separate
eight-frame `SigmaClip` comparison preserved the finite pixels exactly and
matched NaNs (`equal_nan=True`).  These are hotspot measurements, not a claim
that the complete 1,242-frame session has a 25% gain.

The read-only reproducer is
`benchmarks/real_capture_audit.py`; it refuses an output path inside the input
tree and writes temporary leaf products only under `%TEMP%`.  The observed
profile remains dominated by the old `_create_substacks` wait (`945.0 s` of
`992.4 s`), so a full seven-run cold-cache corpus measurement is still the
next operational gate.  No source file under `B:` was modified during this
audit.

### Current-checkout revalidation

After the streaming `VALID_MASK` policy and the raw/mmap Flow loader were
enabled, the same deterministic synthetic gate was rerun with seven warm
repetitions and four workers. Stable measured `1.9493 s` (stdev `0.2657 s`,
peak RSS `342.9 MiB`) against the clean baseline `2.4863 s`, a `21.60%`
reduction (`1.2755×`), with identical Stable digest, `uint16` output, masks,
counts and AVX2 assembly probes. This revalidation is **below** the mandatory
25% gate; the earlier `1.8238 s`/`26.65%` artifact remains useful evidence for
the preceding optimized variant, but is not promoted as the current-checkout
throughput claim. The full-session real-capture gate remains intentionally
pending because the source is user-owned and was only opened read-only.
