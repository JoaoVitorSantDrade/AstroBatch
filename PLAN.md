# AstroBatch roadmap

## Current baseline

AstroBatch is CPU-first and Stable remains the default. Existing processing APIs,
uint16 FITS output, FITS masks, compression/cache behavior and settings format
remain compatibility boundaries.

Delivered in the current checkout:

- One PipelineRunner and typed adapters for Calibration, Batch, Flow, Align,
  Stack, HDR and reference changes, with explicit success, partial, failure and
  cancellation outcomes.
- Local/global registration graphs with natural ordering, neighbor recovery,
  direct global seeds, fingerprints, parent/hop provenance and legacy strategy
  compatibility.
- Align quality checks that exclude self previews, use valid/unsaturated masks,
  and load graph references through a bounded lazy cache.
- Reference rebasing that validates current inputs, preserves final geometry,
  updates both directions of global graph edges, and switches immutable local /
  global snapshots through one active manifest. Legacy JSON files remain a
  fallback for older projects.
- Passive view models for all six stages, StackCommand validation, scoped
  scrolling, and a shared two-worker preview service. Workers return data or
  errors through queues; Tk polling owns widget updates.
- A stage-definition registry that builds all six passive model/view pairs from
  an explicit `StageContext`, including operation controls and scroll-host
  policy. main.py now supplies bindings and shared feedback only.
- A dedicated anchor-selection controller owning pending/applied reference
  state, preview generations, dialog shutdown and reference-change handoff.
- Flow reference cards show accepted/rejected status, recovery method, parent,
  hop count and confidence when persisted graph data is available.

Synthetic benchmarks remain tied to their original workloads. Historical GPU
material is complete documentation, not a new feature backlog.

## Next implementation slices

1. **Finish manual presentation review.** Exercise the six registry-built tabs
   and anchor dialog at small window sizes, with keyboard focus, scrolling,
   themes and DPI changes. Keep settings persistence and shared feedback in
   the composition root.

2. **Complete revision consumers and provenance.** Route every Flow reader,
   visualization and Stack handoff through the active manifest. Add explicit
   stale/disconnected status to the UI and expose revision, parent, hop count,
   recovery method and rejection reason. Keep final-geometry revisions separate
   from reference metadata revisions so an equivalent rebase does not force
   unnecessary Align regeneration.

3. **Maintain asynchronous preview lifecycle.** The service now bounds
   submitted work and result retention, reports worker errors, discards stale
   generations and closes idempotently. Keep these tests alongside future
   preview changes and ensure workers never call Tk.

4. **Measure resources and real captures.** Add repeatable full-sensor mono/RGB
   workloads for peak RSS, compressed/uncompressed throughput, cancellation
   latency and recovered-frame error. Validate equal-exposure unguided captures
   for retained integration, residuals, stellar roundness, background noise,
   clipping masks and uint16 quantization. Leave this gate open until captures
   exist.

5. **Usability and compatibility review.** Test small windows, themes, DPI,
   keyboard focus, scrolling, preview failures, cancellation and settings
   round trips. Keep current output dimensions; expanded union canvases, new
   engines, noise models and the Stable Lanczos compatibility decision require
   separate proposals.

## Acceptance rules

- Every stage preserves native controls, defaults, presets, saved settings and
  success/partial/failure/cancellation behavior.
- A failed write or cancellation cannot activate a mixed Flow revision.
- Existing numerical, FITS, mask, uint16 and rejection regressions remain green.
- Architecture changes stay separate from numerical changes, and benchmark
  claims remain workload-specific.
