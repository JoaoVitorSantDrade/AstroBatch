# Workflow, usability and performance review

> Current status: see **Resumed implementation and UI architecture foundation (2026-09-05)** at the end of this document. Final verification of this delivery: **126 tests passed**, compilation and whitespace checks passed. Earlier stop/audit sections are historical. Real-capture validation and the complete migration of legacy views remain pending.

## Stage 1: inspect and reproduce

- [x] Verify the active checkout and preserve existing edits.
- [x] Baseline: 50 tests pass in the project virtual environment.
- [x] Identify false Batch completion, unprotected worker imports, unbounded console refresh and global mouse-wheel interception.

Exit gate: changes address observable behavior without changing scientific algorithms.

## Stage 2: implement

- [x] Report Batch disk failures and cancellation accurately; always stop its disk worker.
- [x] Handle optional processing import errors within worker cleanup.
- [x] Bound and batch console rendering; preserve severity and ordering.
- [x] Scope scrolling to the hovered view and show Stack in the workflow header.

Exit gate: regression tests cover failure, cancellation and event-loop fairness.

## Stage 3: validate

- [x] Compile modified Python and run the complete suite.
- [x] Measure console rendering before/after with a repeatable Tk workload.
- [x] Record limitations and follow-up opportunities.

## Applicable techniques and next improvements

Tk callbacks must yield frequently: batch console inserts and bound each timer callback. This follows the [Python Tk threading model](https://docs.python.org/3/library/tkinter.html#threading-model).

Cancel pending preparation futures during cleanup; already running reads must finish. See [Executor shutdown](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.shutdown).

Further work: typed operation results across Calibration/Flow/Align to distinguish partial success; asynchronous previews; explicit stage handoff buttons that reuse output paths; phase-local ETA; RAM-derived Batch prefetch limits. These need dedicated behavioral and representative-data validation. Retain native controls, CPU-first processing, Stable defaults and uint16 FITS persistence.

## Validation results (2026-09-04)

- Project venv: all 55 tests passed (50 existing + 5 regression tests).
- Python compilation and `git diff --check` passed.
- Real withdrawn Tk Text widget, 4,000 messages, one warm-up and five measured runs:

| Measurement (median) | Previous implementation | Updated implementation |
| --- | ---: | ---: |
| Total rendering time | 5,965.37 ms | 41.74 ms |
| Longest refresh callback | 5,965.37 ms | 3.01 ms |

Reproduce with `.venv/Scripts/python.exe benchmarks/console_benchmark.py <baseline-revision>`; default baseline is HEAD. Scheduling delays are excluded: the new renderer yields for 50 ms between batches. This synthetic benchmark validates console overhead, not scientific pipeline throughput or full visible-window performance. Tk initialization required execution outside the sandbox. No full visual/interactive usability review was performed.

Regression coverage includes disk-copy failure propagation, cancellation without a completion event, cleanup after an unexpected processing exception, bounded ordered severity-tagged logging, and import failure cleanup for all five processing workers.

## Remaining limitations

Batch still counts analyzed frames separately from successfully persisted files; skipped/invalid inputs and partial results need a shared result model. Overwrite replacement safety was implemented in the 2026-09-05 follow-up below. The log producer queue remains unbounded even though rendering is bounded; sustained extreme logging needs an explicit retention/backpressure policy. Full-size FITS profiling and measured RAM limits remain necessary before further numerical or worker-count optimization. Existing FITS header-length warnings persist in tests.


# Unguided short-exposure workflow (2026-09-04)

Objective: preserve useful integration time from many short subs while screening mount-induced trailing, and reduce per-frame processing overhead.

## Stage 1: design and compatibility

- [x] Inspect Flow metric propagation, Stack selection and native settings.
- [x] Delegate bounded measurement, UI and performance work to three gpt-5.6-luna agents; parent owns integration/review.
- [x] Preserve existing edits, Stable defaults, CPU-first processing and uint16 outputs.

Reference: [Siril stacking criteria](https://siril.readthedocs.io/en/stable/preprocessing/stacking.html) include star roundness and FWHM. AstroBatch will use an explicitly approximate second-moment roundness, not claim equivalence with a fitted PSF.

## Stage 2: implementation gates

- [x] Measure star shape once from bounded cutouts around existing Flow detections.
- [x] Add opt-in trail screening with explicit handling of missing metrics and deterministic selection.
- [x] Save a per-frame JSON selection report before expensive stacking work.
- [x] Expose native controls, an unguided preset and explicit Align-to-Stack handoff.
- [x] Benchmark and implement a numerically validated Batch comparison optimization.

Exit gate: synthetic round/trailing fixtures, compatibility with old metadata when screening is off, filter exclusions never silently restored, and measured performance benefit.

## Stage 3: integrated validation

- [x] Compile all changed modules, run focused tests and full suite.
- [x] Verify a synthetic Flow-to-Stack metadata/selection path and report output.
- [x] Record benchmark evidence and scientific/validation limits.


## Audit and follow-up delivery (2026-09-05)

Status: the original three delivery stages and the unguided workflow stages above are complete. The broader opportunities listed below are backlog, not completed features. Initial audit found the unguided code present but integrated validation unfinished; that gap is now closed.

### Implemented follow-up plan

1. [x] Reduce retained Flow RAM: release unused raw science arrays after star/phase preparation. Keep phase-correlation arrays and stars unchanged. A 24,000,000-pixel float32 array is 96 MB (91.6 MiB) per frame; this is an allocation calculation, not a measured process-RSS reduction. Flow still retains phase arrays per batch.
2. [x] Gate the Batch comparison optimization by workload: native OpenCV statistics for large float32 2D differences; retain NumPy for smaller cases, including mixed invalid pixels. Fall back to centered float64 variance when a large DC offset risks cancellation. No new dependencies.
3. [x] Protect existing Batch outputs: copy to a unique temporary sibling, replace only after successful copy, then remove the source for a move. Tests verify partial copy failure preserves both original files. Replacement moves can require extra copying; this is a reliability tradeoff, not a speed optimization. New-file transfers retain their existing behavior.
4. [x] Validate settings, actual Tk bindings, synthetic Flow metadata, selection report and uint16 output together.

### Evidence

- Final full suite: **85 tests passed**, no skips, using the project venv outside the sandbox so real Tk tests execute. Compile and diff whitespace checks passed.
- End-to-end synthetic test: five 192x192 FITS star fields, four circular and one elongated. Flow writes measured shape metadata; trail filtering keeps four; Mean + SigmaClip stacking succeeds; output is uint16 and records `TRAILFLT`, `MINROUND`, `MINSHAPE`.
- Live Tk tests cover persistence registry/control binding, preset behavior, invalid numeric settings and explicit Align handoff. Full visual/interactive layout review across screen sizes is still outstanding.
- Batch benchmark, median of nine measured runs after warm-up:

| Workload | Previous | Updated | Speedup |
| --- | ---: | ---: | ---: |
| 512x512 finite | 0.410 ms | 0.364 ms | 1.12x |
| 512x512 mixed NaN/Inf | 0.775 ms | 0.684 ms | 1.13x |
| 2048x2048 finite | 15.539 ms | 10.932 ms | 1.42x |
| 2048x2048 mixed NaN/Inf | 21.741 ms | 13.133 ms | 1.66x |

Reproduce: `.venv/Scripts/python.exe benchmarks/batch_score_benchmark.py`. Results are synthetic per-comparison timings, not whole-pipeline speedups. Large-image scores differ from the former float32 norm by about 0.00033 on a score of 28.3; the new reduction is checked against centered float64 variance. Frames exactly on a drift threshold can change classification due to rounding. Small-image scoring retains the previous arithmetic.

### Using the unguided workflow

1. Run AstroFlow again to populate shape metrics for older datasets, then Align as usual.
2. In Stack, click **Usar saída do Align** to reuse that path explicitly.
3. Click **Preset: Subs sem guiagem**. This enables a starting minimum roundness of 0.65 with at least five measured stars and selects All + Mean + SigmaClip + Stable. All means all frames passing the filter.
4. Adjust minimum roundness for the data. A value near 1 prefers circular stars; this is an approximate second-moment axis ratio, not a fitted PSF or a guaranteed mount-error diagnosis.
5. Review `<output-name>_selection.json` in the output directory. It includes selected/excluded paths and reasons, and is written before expensive stacking, including when too few frames pass. FITS headers record the filter settings.

Unknown shape metrics are excluded only when the trail filter is active (or cannot rank under the explicitly chosen roundness metric). Existing configurations keep the filter off. Measurements use at most 64 cutouts of radius 8 around existing detections, once per frame. Saturation, overlapping stars, strong optical aberrations, faint stars and long trails can bias these approximate measurements; validate thresholds on real subs. The software screens bad frames; it does not restore detail lost to motion within an exposure.

### Delegation audit

Three gpt-5.6-luna agents implemented the measurement, UI and initial performance work under the earlier lower-cost request. They used high effort in that phase. One stopped at a usage limit; the parent finished its validation and fixes. The subsequent explicit Luna Low request was attempted, but the service rejected a new agent with `agent thread limit reached`. Remaining integration, tests and overwrite safety were completed locally; they were not delegated at Low effort.

### Prioritized next plan (not implemented)

1. **Bound Flow phase storage and Batch prefetch by RAM.** Gate: stress tests with hundreds of large mono/RGB frames, retained transform equivalence, cancellation latency and measured peak RSS. Avoid claiming that dropping raw arrays alone bounds whole-batch memory.
2. **Asynchronous previews and richer operation results.** Gate: UI input stays responsive while previews load; failures/partial outputs and saved-file counts are accurate across Calibration/Flow/Align.
3. **Bound console producer retention.** Gate: sustained log flood has bounded memory, preserves critical errors and explicitly reports dropped informational messages.
4. **Real unguided-data validation.** Gate: compare retained integration time, stellar roundness and background noise on the same session with filter off/on; tune thresholds without excessive rejection. A weighted-mean option needs consistent per-pixel coverage/rejection weights across the hierarchical reducer before delivery.

## HDR delivery plan (2026-09-05; original gates, partially implemented)

The HDR backend is CPU-first and consumes linear, calibrated, already aligned FITS images. Classical calibrated fusion is sufficient; camera-response estimation from Debevec and Malik (1997) is not used for linear FITS. Neural reconstruction is explicitly out of scope.

- [ ] 1. Define typed HDR configuration and reject missing/invalid EXPTIME, incompatible geometry, filter, binning or Bayer metadata. Gate: invalid-metadata tests and a schema usable by native UI controls.
- [ ] 2. Fuse equal-exposure frames with inverse-variance noise weighting; support mixed exposures when metadata is valid. Gate: synthetic SNR improvement and exposure-normalized radiance recovery.
- [ ] 3. Apply raw sensor saturation thresholds and invalid-pixel rejection; preserve conservative disagreement/ghost confidence masks. Gate: clipped pixels become invalid and are never hallucinated, while faint low-SNR structure is retained.
- [ ] 4. Stream bounded row bands and honor cancellation. Gate: peak memory scales with a band, progress reports phase/counts, and cancellation produces no partial success.
- [ ] 5. Persist only uint16 FITS with CALNORM/CALMIN/CALMAX, VALID_MASK and DISAGREE extensions, atomic replacement, provenance and group summary JSON. Gate: reopen/restore round trip and no floating FITS output.
- [ ] 6. Quantify uint16 quantization error and warn when its step exceeds the estimated noise precision; never claim arbitrary lossless HDR. Gate: header and result report expose the bound.
- [ ] 7. Benchmark row-band HDR throughput and validate synthetic fixtures. Gate: repeatable benchmark plus tests for alignment mismatch, NaN, saturation and cancellation. Real capture validation remains pending until representative data exists.

Source: [Debevec and Malik (1997)](https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf), foundational HDR radiance recovery. The equal-exposure mode is the primary supported mode and does not require two exposure groups.

## Implementation handoff and stop audit (2026-09-05)

**Stopped on explicit user instruction: document changes, residual bugs and remaining work, then stop.** No additional implementation or test run was performed after that instruction. No commit, deployment or release was made. Existing checkout changes from earlier work were preserved.

The latest implementation started with three **gpt-5.6-luna / low effort** agents for Flow, Align and HDR. The parent reviewed their work and requested corrections. After the user explicitly authorized finishing incomplete work locally, the parent implemented additional correctness fixes, native controls, regression tests and a quality benchmark. Agent completion messages alone were not treated as proof that a delivery gate passed.

The user has no representative capture folder yet and plans **equal exposure lengths**. Equal-exposure fusion improves noise under suitable assumptions; it cannot recover detail clipped in every input. Mixed-exposure metadata/group summaries exist, but mixed-exposure scientific validation remains incomplete. No neural models, neural dependencies or GPU processing were added. Stable remains the default; science outputs remain uint16 FITS.

### Status against the requested delivery order

| Order | Delivery | Current status |
| --- | --- | --- |
| 1 | Flow benchmark corpus and unified resource budget | Shared CPU/RAM estimator and bounded local preparation implemented. Initial FITS benchmark exists, but corpus correctness and peak-RSS validation remain incomplete. |
| 2 | Flow confidence and recovery ladder | Evidence labels, phase reuse, normal/relaxed/anchor recovery, rolling temporal history and manual-anchor fix implemented. Confidence calibration and several stress/negative cases remain open. |
| 3 | Align staged scheduler and post-warp verification | Bounded compute submission, single writer, atomic output and shared-reference quality gate implemented. Full integrated scheduler/cancellation/resource validation remains open. |
| 4 | RGB quality gate and interpolation benchmark | Stable/Fast correction signs, confidence/displacement limits, geometric mask propagation and interpolation benchmark implemented. Real chromatic data and mask-edge tests remain open. |
| 5 | HDR configuration, grouping and exposure validation | Backend configuration, native HDR tab, metadata checks and exposure-group summaries implemented. UI validation coverage and complete metadata semantics remain open. |
| 6 | Calibrated HDR fusion, masks and uint16 metadata | Row-band fusion, calibration restoration, masks, atomic persistence, provenance and quantization reporting implemented. Scientific and persistence edge cases below remain open. |
| 7 | Real-data validation and UI polish | Some native controls/lifecycle fixes implemented. Real-data validation is pending because captures do not yet exist; final UI review is unfinished. |

None of these rows should be interpreted as a claim that the entire seven-stage plan is complete.

### What changed in the code

**Resources and Flow — `app/engines/execution.py`, `astroflow_logic.py`:**

- Added a backward-compatible `ExecutionBudget.for_frame_pipeline()` that accounts for reserved images and two in-flight preparation results per worker. Added header-only science-image sizing, including compressed image extensions with empty primary HDUs.
- Local Flow prepares its anchor once and consumes other images through a bounded FIFO of futures instead of collecting all prepared phase images before matching. Local batches execute sequentially to avoid nested pools. Cancellation is passed into local work and prevents writing its final JSON after cancellation.
- Released returned/unused anchor phase arrays. Global anchor detection now decodes one anchor at a time and retains star catalogues instead of full phase arrays. This trades some global detection parallelism for lower retained RAM; no whole-pipeline speedup is claimed.
- Preserved every filename when choosing a manual middle anchor. Changed temporal history from a frozen initial eight samples to a rolling window.
- Reused the previous/current phase correlation between normal and relaxed attempts. Anchor correlation is computed only if recovery reaches that attempt; anchor jumps bypass the consecutive-motion check and are not added to its history.
- Added confidence labels/reasons using phase response, inlier ratio, residual and spatial coverage. Spatial coverage now excludes correspondences with residual above a fixed four-pixel threshold. These labels are heuristics, not calibrated probabilities or a fitted statistical uncertainty model.

**Align — `astroalign_logic.py`, `views/align_view.py`, `main.py`:**

- Added bounded task submission and a dedicated single-thread writer. A compute worker waits for its write to complete before reporting success, which bounds retained write payloads by active compute workers.
- Applied the shared RAM estimate to actual FITS header dimensions and clamped explicit in-flight requests to the computed capacity. Dry-run scheduling does not require reading source images for sizing.
- FITS writes use unique sibling temporary files and atomic replacement. Cancellation is checked inside the writer, including immediately before replacement; interrupted writes preserve an existing destination.
- Added a shared reference preview in output coordinates and post-warp metrics: normalized residual, phase confidence, coverage and residual displacement expressed in full-resolution pixels. The opt-in gate now checks displacement; checking normalized RMS alone had incorrectly accepted a five-pixel shift.
- Added FITS quality cards and `.fits.align.json` sidecars. Retained a subset of scientific calibration/exposure headers even when cosmetic header retention is disabled.
- Corrected RGB translation signs in both Fast and Stable paths. Weak/non-finite shifts and corrections exceeding two pixels are skipped before modifying channels. Applied-channel geometry contributes to the output validity mask. Zero and negative source intensities are not themselves interpreted as invalid geometry.
- Added native controls for the opt-in residual gate, maximum displacement and shared Flow/Align worker and estimated-RAM settings, with persistence wiring.

**HDR — `hdr_logic.py`, `views/hdr_view.py`, `main.py`:**

- Added a native HDR tab with aligned FITS folder, output file, optional saturation, optional exposure override, noise floor and row-band controls. Added settings persistence, save dialog, worker guard, run/cancel lifecycle integration, progress/error reporting and safer exception callback capture.
- Inputs must have matching geometry and compatible available metadata. Missing/invalid exposure is rejected unless overridden. Output cannot overwrite an input through the backend; the UI excludes the chosen output from its input listing.
- Fusion reads row bands with `memmap=False`, applies input validity masks and explicit/header saturation thresholds to encoded source values before CALNORM restoration, then normalizes by exposure. It uses a constant user-supplied noise-floor model with exposure-squared weights; it is not an estimated per-frame Poisson/PSF noise model.
- Replaced the biased observed-pixel-dependent weighting and the retained list of all per-band images. Scatter diagnostics now use streaming sums/counts. `DISAGREE` is an advisory scatter mask, not motion reconstruction or automatic outlier rejection.
- RGB output uses conservative 2D `VALID_MASK` and `DISAGREE` extensions. No output is written when all pixels are invalid. Cancellation is checked between inputs/bands and immediately before atomic replacement.
- Persisted output is uint16 with `CALNORM`, `CALMIN`, `CALMAX`, `CALHDR`, `HDRNFRM`, `HDRBITS`, `HDRQERR` and a precision note. Exposure is reset to one second for normalized radiance, and the output unit is derived from the input unit. Source saturation cards are removed from the newly encoded output.
- JSON provenance records input paths/exposures, exposure groups, weighting description, valid/disagreement counts, quantization step and a noise-relative precision warning. Failure to write JSON after successful science output is reported as a provenance warning rather than pretending no science file was saved.

### Validation actually performed

- An intermediate full suite passed **103 tests** before the last parent fixes. This is not a final-checkout pass claim.
- A subsequent full-suite run found **three failures and one error**, all in successful HDR paths: a NumPy boolean in the new precision report was not JSON serializable. The value was changed to a Python `bool`.
- After that fix, the focused command below passed **15 tests** (delivery validation, HDR and scheduler). It covers compressed-header sizing/budget arithmetic, RGB correction direction and oversized-shift skipping in both profiles, cancellation during an Align write preserving existing bytes, raw-domain CALNORM saturation exclusion, unbiased equal-exposure averaging, final-band HDR cancellation, invalid numeric HDR settings and actual Flow anchor recovery with phase-call counting.

  `.venv/Scripts/python.exe -m unittest tests.test_delivery_validation tests.test_hdr_pipeline tests.test_align_scheduler -q`

- The existing actual-worker Align regression accepts a correctly transformed 700×900 shifted FITS image and rejects the identity transform against a shared reference. It was exercised earlier; a final complete-suite rerun remains necessary.
- Compilation was checked during implementation; not every final file was recompiled separately after every subsequent edit. Existing FITS long-keyword warnings remain.
- `benchmarks/quality_delivery_benchmark.py` executed successfully. For eight independent synthetic 256×256 equal-exposure frames with injected noise sigma 10, fusion took **0.119 s**, and measured noise improvement was **2.830×**, versus the independent-noise expectation of **2.828×**. This is a synthetic correctness/throughput example, not a real-capture performance promise.

Interpolation benchmark: a 384×384 Gaussian source with a known fractional shift, one warm-up and five timed runs. RMS is against the analytical shifted Gaussian in synthetic intensity units; times are environment-specific.

| Mode | Stable median | Fast median | Stable RMS | Fast RMS |
| --- | ---: | ---: | ---: | ---: |
| Nearest | 0.095 ms | 0.065 ms | 1.7314 | 1.7314 |
| Bilinear | 0.122 ms | 0.084 ms | 0.2517 | 0.2517 |
| Bicubic | 6.261 ms | 0.666 ms | 0.0162 | 0.1640 |
| Lanczos label | 6.285 ms | 2.597 ms | 0.0162 | 0.0385 |

These largely compare pre-existing engine kernels, not speedups attributable to the new scheduler. Stable's existing Lanczos label still selects the legacy cubic implementation; the benchmark exposes this distinction. No default was changed based on timing alone.

### Residual bugs and review findings

1. **Flow benchmark corpus is not an acceptance gate yet.** `benchmarks/flow_workflow_benchmark.py` currently calls a translated image a `trail` case without actually simulating trailing, advertises seed 42 while generating repetition seeds 0–2, and reports counts/timing without known-transform residuals. Its reported rejection is not proof of trail sensitivity. A replacement patch was attempted but rejected before applying, so the old benchmark remains.
2. **RAM limits are estimates, not strict process-memory caps.** At least one worker and two in-flight slots are retained even if the configured budget cannot fit that minimum. Library temporaries, RGB debayer expansion, allocator retention and full output buffers are not measured. HDR retains full output/mask arrays plus bands; its memory is not only one band. No stress-scale peak-RSS result exists.
3. **Confidence is incomplete.** Anchor and optional asterism-fallback records are not yet uniformly covered by confidence/provenance fields. Spatial coverage uses a fixed residual cutoff rather than the configured RANSAC threshold. Some agent-written tests have weak or unrelated assertions; the new behavioral tests improve coverage but do not replace a full audit.
4. **Scientific mask propagation is incomplete.** Align does not yet comprehensively carry input validity/sensor-saturation masks through debayer, global warp and RGB corrections. Interpolation can dilute clipped pixels before HDR sees them. Unknown saturation is reported and cannot safely be inferred from a CALNORM maximum. Do not claim that all saturated stellar cores are detected or recoverable.
5. **Align validation/storage gaps remain.** RGB mask-edge coverage needs direct output assertions; sequential conservative mask intersections can over-exclude edges. Advisory metrics with no usable reference are not explicitly represented as a distinct `unverified` state. Quality JSON is not atomic and write failures are silently ignored. Stack does not yet consume Align's new sidecars for selection.
6. **HDR metadata and input compatibility need stricter semantics.** Available metadata is compared but missing-vs-present gain/filter/binning information may pass. Full calibration provenance, exposure aliases, camera metadata and already-rate-valued input conventions are not comprehensively validated. Saturation thresholds are interpreted in the supplied encoded-image domain; original sensor thresholds need correct mask propagation after calibration. Arbitrary unit text is not a dimensional-analysis system.
7. **HDR configuration/persistence edge cases remain.** Dictionary parsing clamps or converts some row-band/disagreement values while direct dataclass validation rejects them, so validation is inconsistent. The unused full-image `_read` helper remains. Science FITS and JSON are not a transactional pair; output provenance should also be made durable in FITS or an atomic manifest. Quantization warning text in FITS is generic, while the JSON warning compares with an idealized noise model.
8. **HDR scientific validation is partial.** Mixed exposures, repeated fusion, differing backgrounds, realistic Poisson/read noise, mask discontinuities, all-channel clipping and misregistered inputs need numerical tests. The backend assumes images are already calibrated/aligned; geometry equality alone does not verify alignment. Scatter flags do not remove ghosts/outliers. This is not PSF-aware proper coaddition or arbitrary lossless HDR.
9. **UI review remains unfinished.** The new HDR and Align controls need real Tk interaction/layout tests, typing/focus/error cases, small-screen review and explicit Align-to-HDR handoff. Shared Flow/Align resource settings are currently displayed in the Align panel. HDR's input scan is one folder level. Newly added UI lifecycle behavior has not received complete dedicated regression coverage.
10. **Final integration checks remain open.** No final full suite, full compile sweep or final whitespace check was run after the last changes. No representative real-data validation exists. Historical unbounded console-producer retention and broader shared operation-result work also remain backlog.

### Next work, in order, when explicitly resumed

1. Run compilation, the full suite and diff checks on this exact checkout; fix any regressions before marking any delivery gate complete. Add actual HDR UI and RGB mask-output tests.
2. Repair the Flow benchmark with real blurred trails, unrelated/no-overlap fields, truthful seeds, repeated identical fixtures and inverse-transform errors. Add measured memory/cancellation stress workloads and an explicit oversized-frame policy.
3. Complete science-mask/calibration provenance through Align, uniform Flow confidence records and durable Align quality reporting/Stack consumption.
4. Tighten HDR metadata/unit/config validation, exercise mixed-exposure and repeated-fusion radiometry, and make provenance persistence atomic. Validate scatter thresholds under realistic noise.
5. Finish UI handoffs and layout review. When captures exist, validate equal-exposure unguided subs first, then mixed exposures only if they become relevant. Record rejection rates, retained integration time, stellar residuals/roundness and background noise.

### Research basis and limits

- [Guizar-Sicairos et al., efficient subpixel registration (2008)](https://opg.optica.org/ol/abstract.cfm?URI=ol-33-2-156): relevant to Fourier/subpixel registration and bounded refinement. Existing library routines are used; this work does not claim a new registration algorithm.
- [Zackay and Ofek, proper coaddition](https://arxiv.org/abs/1512.06879): motivates careful noise/PSF assumptions and scientific validation. The current constant-noise weighted fusion does **not** implement their PSF-aware proper coaddition method or inherit its optimality guarantees.
- [MAGSAC, Barath et al. (2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Barath_MAGSAC_Marginalizing_Sample_Consensus_CVPR_2019_paper.html): a research candidate for robust model estimation. It was not integrated into the current partial-affine estimator; standard RANSAC remains. Do not describe this checkout as implementing MAGSAC or a fully state-of-the-art registration pipeline.

Work is stopped. All unchecked items and residual findings above are recorded for a future explicitly authorized continuation.

## Resumed implementation and UI architecture foundation (2026-09-05)

The user resumed bug fixing and implementation, with maintainability and easier feature additions as the next UI phase's goals. Three existing agents briefly resumed bounded tasks under the earlier delegation authorization; they were stopped when the user requested **no subagents**. All subsequent fixes, architecture integration and final verification were performed by the parent. Existing unrelated changes, including the already-modified `requirements.txt`, were preserved. No commit or deployment was made.

### Delivered architecture changes

- **One runner for all six stages:** `app/application/runner.py` owns thread creation, cooperative cancellation, coalesced progress and a single completion result. Main processing workers no longer call Tk. The UI polls events, renders progress and unlocks controls. A pending completion prevents another operation from starting prematurely; thread-start failures release the runner. Legacy worker entry points remain thin compatibility bridges.
- **Explicit pipeline adapters:** `app/application/pipelines.py` adapts Calibration, Batch, Flow, Align, Stack and HDR to `OperationResult`. Partial Align results and unconfirmed/empty results are no longer reported as successful completion. Calibration and Flow now return outcome summaries. The UI reserves 100% for confirmed success.
- **Shared configuration/persistence boundaries:** `ResourceSettings` validates worker/RAM controls independently of Tk. `SettingsRepository` reads the legacy section format and saves schema version 2 through atomic replacement. The same JSON writer is available for scientific sidecars.
- **Bounded activity retention:** the activity buffer limits retained messages and per-message size, preferentially keeps error diagnostics, and explicitly reports dropped messages. It never blocks a processing worker waiting for the GUI. This is bounded recent activity, not an unlimited audit log.
- **First passive view:** HDR now receives `HDRViewModel` variables and commands rather than the root controller. This is the pattern for migrating the other tabs. HDR backend parsing is exposed as `build_hdr_config` and validated through a shared validator.
- **Reusable form scrolling:** Calibration, Batch, Flow, Align and HDR use `ScrollableHost`; Stack keeps its existing canvas. Mouse-wheel bindings are scoped to each form. Pending host callbacks are cancelled on destruction. The application's minimum window height was reduced from 1020 to 700, with scrolling for longer forms.
- **Less duplicated UI wiring:** one operation-button mapping controls lock/unlock for all stages. Shared Flow/Align resource controls now live in the project panel. HDR has a working save-file dialog, a native Align-output handoff, clearer field labels and six editable native fields.

This is an implemented migration foundation, not a claim that `main.py` has become a composition-only root. The five legacy views and several preview/selection dialogs still depend on it.

### Bugs fixed since the stop audit

| Earlier finding | Current implementation |
| --- | --- |
| Invalid Flow corpus and missing transform evidence | Benchmark now uses Gaussian star fields, independent noise, a real nine-pixel trail, deterministic seeds/repetitions, unrelated fields and current-to-anchor inverse-transform errors on five probe points. |
| Budget could exceed its own minimum silently | Frame budgets now reject a working set too large for the configured target with the approximate required MiB; one available slot produces one in-flight frame. Local Flow no longer falls back to unrestricted workers after a budget error. The budget remains an allocation estimate, not a measured process-RSS cap. |
| Inconsistent Flow confidence records | Reference records explicitly say `reference`; optional asterism recovery records its method and conservative low-confidence explanation. Spatial inlier coverage uses the configured RANSAC residual threshold. |
| Lost science/saturation masks | Calibration persists `VALID_MASK` and `SAT_MASK` before losing sensor clipping information to normalization. Align propagates masks through the spatial transform, conservative interpolation/debayer support and RGB corrections. HDR consumes both masks. Original RGB masks are reused for each channel rather than repeatedly shifting the already-intersected mask. |
| RGB science could be mistaken for a mask | Align previously only accepted a 2D science HDU, so an RGB FITS with a 2D validity extension could load the mask as science. It now accepts supported mono/RGB science layouts and excludes metadata/mask HDUs. Calibration also excludes mask extensions from science selection. |
| Ambiguous alignment quality and unsafe sidecars | Missing reference is explicitly `unverified`; quality JSON uses atomic replacement and write failures are reported. Stack loads these diagnostics under separate alignment keys without silently changing existing selection criteria. |
| HDR validation inconsistent or ambiguous | Dictionary/dataclass validation shares one validator; fractional/invalid row bands and boolean numeric values are rejected. EXPTIME/EXPOSURE aliases are checked, conflicting aliases fail, and missing-vs-present compatibility metadata fails. Invalid CALNORM ranges are rejected. |
| HDR divided rate-valued input twice | Inputs whose BUNIT ends in `/s` are treated as rates. Output exposure remains one second; obsolete exposure aliases are removed. Mixed-exposure normalization and repeated fusion have numerical regression tests. |
| HDR science/provenance could diverge | Full UTF-8 JSON provenance is embedded in the same atomic FITS product as a uint8 `HDR_META` extension. Science remains uint16. The external JSON is an atomic convenience copy; its failure cannot destroy the embedded provenance. FITS also records the noise-relative quantization warning. |
| All-channel RGB validity and saturation were incomplete | HDR rejects output with no valid all-channel coverage, validates mask dimensions, and excludes saturated source masks. Non-finite restored/rate values are excluded before accumulation. |
| UI controls and background completion fragile | Dedicated native tests now cover HDR typing, save dialog, handoff, all-stage button lifecycle, partial completion, form scrolling and invalid resource settings. An obsolete source-count assertion was replaced with a behavioral lock/unlock test. |

### Final verification for this delivery

- Baseline at resumption: **109 tests passed**.
- Final: **126 tests passed, no skips**, using `.venv/Scripts/python.exe -m unittest discover -s tests -q` (8.324 seconds on this run).
- `.venv/Scripts/python.exe -m compileall -q main.py astroflow_logic.py astroalign_logic.py calibration_logic.py hdr_logic.py stacking_logic.py app views tests benchmarks` passed. Imports and execution in the final full suite also covered the last fixes.
- `git diff --check` passed. Git's LF/CRLF notices and existing FITS HIERARCH warnings are informational and remain present.
- New tests include runner failure/cancellation/re-entry, progress coalescing, bounded log overflow, atomic settings preservation, HDR exposure/rate/config semantics, embedded provenance, repeated fusion, RGB output masks and the actual **Calibration → Align → HDR** clipping-mask path.
- Native Tk tests passed with real widgets, including a 1000×760 form-layout exercise. This is functional/layout coverage; no screenshot-based visual review was performed.

Latest Flow benchmark: fixed seed 42, 256×256 FITS, three repetitions of each fixture. Times are per two-frame local Flow workload and are not before/after speedup claims.

| Fixture | Median | Decision | Maximum probe error |
| --- | ---: | --- | ---: |
| Translation | 67.59 ms | Accepted | 0.416 px |
| Rotation | 67.98 ms | Accepted | 0.208 px |
| Low SNR | 65.77 ms | Accepted | 0.029 px |
| Nine-pixel trailing | 60.74 ms | Rejected | N/A |
| Unrelated field | 69.45 ms | Rejected | N/A |

Reproduce with `.venv/Scripts/python.exe benchmarks/flow_workflow_benchmark.py`. Five synthetic cases do not establish a general false-positive rate. The earlier interpolation/HDR benchmark remains a separate historical measurement and was not relabeled as a new scheduler speedup.

### Remaining limitations and next UI phase

1. **Finish the presentation migration.** Move the five remaining tabs to passive view models/commands, extract preview/anchor dialogs and their asynchronous lifecycle from `main.py`, and centralize stage registration so adding a stage does not require repeated UI wiring. Preserve all native settings and existing behavior.
2. **Complete typed commands.** Resources and HDR have shared validation; the older stages still build legacy dictionaries/configs. Move remaining validation into testable application commands, then reduce `main.py` to composition and rendering. Versioned settings storage exists, but the Tk-variable registry is still the form binding layer.
3. **Expand scientific validation.** No real unguided captures are available. Validate retained integration time, star shapes, registration residuals and background noise on equal-exposure data first. Optional asterism confidence is deliberately conservative, not probabilistic. HDR uses a constant-noise model and advisory scatter flags, not PSF-aware proper coaddition or ghost reconstruction.
4. **Measure realistic resource use.** Full sensor/RGB peak-RSS, compression-heavy throughput and cancellation latency under large workloads are not yet measured. The conservative frame estimate can reject a low configured RAM target; increasing the target is explicit. HDR still retains complete output/mask planes plus row-band accumulators.
5. **Remaining metadata/coverage scope.** Unit handling recognizes `/s` rates and checks compatible available metadata; it is not a full FITS units/calibration-provenance system. Unknown sensor clipping cannot be reconstructed without metadata/masks. Calibration remains a mono/CFA stage and does not add a general RGB calibration workflow. Conservative interpolation/debayer mask footprints may exclude extra edge pixels and need real-data tuning.
6. **Compatibility and visual review.** Stable's historical Lanczos label still routes through its legacy cubic interpolation path; correcting that naming/behavior needs an explicit compatibility migration. Full visual review across themes/DPI remains pending. Existing preview code has not all migrated to the new runner, and the bounded activity buffer intentionally discards older entries under sustained floods.

The current bug-fix and architecture-foundation delivery is verified and ready for review. The next phase is the remaining UI architecture migration above; neither that full migration nor real-data scientific validation is marked complete.
