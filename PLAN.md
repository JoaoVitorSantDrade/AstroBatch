# V2 Engines and Performance Architecture

## Summary

Refactor Flow, Align and Stack around an internal CPU-first engine registry with two profiles:

- **Stable** preserves V1 behavior and numerical compatibility.
- **Fast** selects validated accelerated engines with defined scientific tolerances.
- The UI shows the profile by default; technical engine choices live under **Advanced**.

Astroalign will be added as an optional **asterism-transform fallback**, not as the default Fast engine: its triangle matching is appropriate for stellar images without WCS and is robust to seeing/PSF differences, but is a reliability path rather than a throughput optimization. [Astroalign documentation](https://astroalign.quatrope.org/en/latest/)

## Architecture and interfaces

- Add `app/engines/` with:
  - `EngineDescriptor`: id, stage, profile support, capabilities, availability reason.
  - Protocols: `StarDetector`, `TransformEstimator`, `WarpEngine`, `ChannelRefiner`, `StackReducer`.
  - `EngineRegistry`: explicit built-in registration, validation and fallback. External package discovery is deferred.
  - `EngineSelection`: typed persisted settings; legacy `flow_engine` maps to its detector choice.
- Engines receive typed inputs, cancellation, worker budget and event sink; they return typed results/metrics. Pipelines depend on protocols rather than OpenCV, Numba, Photutils, SEP or Astroalign imports.
- Add an execution-budget policy so frame/batch thread pools and parallel Numba kernels never oversubscribe CPU.
- Keep existing public processing functions as V1 adapters until each pipeline has migrated.

## Engines, methods and kernels

- **Flow**
  - Stable retains DAO, current OpenCV contour detection, KD-tree matching and affine RANSAC.
  - Fast adds `opencv-components`: native connected-components statistics plus NumPy filtering, removing Python contour/moment work.
  - Add optional `sep` detection with lazy import and explicit unavailable status. SEP provides source extraction plus spatial background estimation. [SEP API](https://sep.readthedocs.io/en/stable/reference.html)
  - Add optional `astroalign-asterism` transform fallback. It receives already-detected `(x, y)` stars, calls Astroalign’s coordinate-capable `find_transform`, converts its `SimilarityTransform` to the pipeline’s 3×3 matrix, and reuses AstroBatch’s own OpenCV warp/mask path. This prevents duplicate image detection and preserves FITS/mask behavior. Astroalign exposes both coordinate input and a `max_control_points` control. [Astroalign API](https://astroalign.quatrope.org/en/latest/api.html)
  - Expose the fallback as `Disabled | Astroalign asterisms` in Advanced; it is attempted only after the primary matcher rejects a frame.
  - Fix Flow worker count so `ThreadPoolExecutor` always receives an integer.

- **Align**
  - Extract current OpenCV and scikit-image paths into `WarpEngine`s.
  - Stable keeps OpenCV nearest/bilinear and compatibility implementations for bicubic/Lanczos.
  - Fast adds native OpenCV cubic/Lanczos4 and native RGB micro-registration, enabled only after tolerance validation.
  - Keep Astroalign out of image warping: its transform-estimation engine augments Flow; AstroBatch retains one output/mask implementation.
  - Flatten eligible frame work across batches under one bounded executor.

- **Stack**
  - Introduce `StackReducer`; Stable retains NumPy/Astropy behavior.
  - Add cacheable Numba Fast kernels for masked sum+count, mean, min/max, normalization/mask application and weighted leaf merge.
  - Use `njit(parallel=True)`/`prange` only for independent per-pixel work and only when the execution budget reserves kernel threads. [Numba parallel-loop documentation](https://numba.readthedocs.io/en/stable/user/parallel.html)
  - Fast initially covers Mean, Sum, Minimum and Maximum without rejection. Median, SigmaClip, MAD and Winsorized remain Stable until benchmarked candidates meet tolerance and memory gates.
  - Preserve Astropy’s optimized string-configured statistics and Bottleneck support for sigma clipping. [Astropy performance guidance](https://docs.astropy.org/en/stable/stats/index.html)

## Migration and UI

1. Add engine contracts, registry, profile/selection persistence and legacy adapters without algorithm changes.
2. Migrate Flow detectors, primary transform, and Astroalign fallback.
3. Migrate Align warp/refinement engines and global scheduling.
4. Migrate Stack reducers and kernels behind capability/profile gates.
5. Replace direct Tk-variable algorithm selection with typed commands and registry-backed Advanced controls.

Missing optional dependencies never silently alter a result: the UI explains the unavailable engine, while profile fallback occurs only under explicit policy.

## Test plan

- Registry and adapter tests: legacy migration, engine resolution, unavailable SEP/Astroalign, cancellation, fallback and event propagation.
- Stable golden tests preserve current output, flow metadata and FITS behavior.
- Astroalign fallback fixtures cover large shift/rotation/scale, variable PSF and failure (`MaxIterError`); verify matrix validity, mask preservation and no duplicate detector invocation.
- Fast tests use operation-specific tolerances, valid-mask identity, transform acceptance metrics and FITS integrity.
- Concurrency tests verify integer worker counts and no nested oversubscription.
- Repeated-median benchmarks cover Flow detection/matching, Align warps, and Stack reducers across mono/RGB and compressed/uncompressed FITS. Engines not faster than Stable on the reference workload remain experimental and are excluded from automatic Fast selection.

## Assumptions

- Engines remain internal built-ins in this delivery; third-party plugin discovery is deferred.
- CPU is the immediate target; interfaces leave room for GPU later.
- Stable remains default. Fast may have small documented floating-point differences within tested scientific tolerances.
- Astroalign is optional and fallback-only; it improves registration resilience, not the default throughput path.
