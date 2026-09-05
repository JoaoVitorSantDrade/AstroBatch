# AstroBatch V2 Architecture Path

## Implemented migration foundation (2026-09-05)

The following are now active code paths, superseding the corresponding
"next" steps in the historical sequence below:

- `app/application/runner.py`: one worker/cancellation/completion policy for
  all six stages, immutable progress/results, coalesced progress and UI polling.
- `app/application/pipelines.py`: lazy pipeline adapters with explicit success,
  partial, failed and cancelled outcomes. Workers do not call Tk.
- `app/application/commands.py`: validated resource settings independent of Tk.
- `app/application/log_buffer.py`: bounded, nonblocking recent activity.
- `app/infrastructure/json_store.py`: versioned settings and atomic JSON writes.
- `app/infrastructure/fits_masks.py`: shared scientific mask loading.
- `views/hdr_model.py` and `views/hdr_view.py`: first passive form, supplied
  variables and commands without a reference to the root controller.
- `views/scrollable_host.py`: reusable scoped scrolling for legacy and new forms.

`main.py` composes these pieces and retains compatibility worker entry points.
It still owns the other form variables and preview/anchor dialogs. This is an
incremental migration, not a completed separation of every view and command.

For a new feature, add its domain callable and validated command, adapt its
result in `execute_pipeline`, provide a passive view model, and register its
view/buttons in the composition root. Do not create another per-stage thread,
Tk callback from a worker, or independent settings writer. Test its adapter and
command without Tk, then test the native form binding and shared lifecycle.

Next: migrate the remaining views and preview lifecycle, extract the remaining
typed commands, then consolidate stage registration. Preserve field parity
before changing the visual design. Current verification is recorded in
`Docs/WORKFLOW_REVIEW.md` (126 tests passed for this delivery).

## Why change now

The application currently works as a single Tk controller: `main.py` owns Tk
state, persistent settings, button state, worker threads, pipeline invocation,
progress rendering, and error presentation. Each processing module has a
slightly different callback contract. This makes a small UI change risky and
forces bug fixes to be repeated across stages.

The V1 UI remains the production interface. The work completed now establishes
the first V2 boundary: every pipeline reports progress through one visible
operation lifecycle (`start -> progress -> success/cancelled/failed`) and the
footer renders phase, item counts, percentage, elapsed time, ETA, and a
severity-coloured activity log.

## Target layout

```text
app/
  application/
    commands.py          # typed command/config objects per stage
    runner.py            # one cancellation, worker, outcome and event policy
    events.py            # ProgressEvent, LogEvent, OperationFinished
  domain/
    calibration.py       # pure orchestration, no Tk imports
    batch.py
    flow.py
    align.py
    stack.py
  infrastructure/
    settings.py          # JSON persistence and versioned migrations
    filesystem.py        # paths, discovery, atomic output helpers
  presentation/
    tkinter/
      app.py             # composition root only
      views/             # passive widgets bound to view models
      feedback.py        # status/progress/activity renderer
tests/
  unit/                  # domain and application behavior
  integration/           # filesystem/pipeline contracts
  ui/                    # no-display view-model and event rendering tests
```

Dependencies point inward: presentation depends on application; application
depends on domain; infrastructure implements ports defined by application or
domain. Processing code must never import Tkinter.

## Migration sequence

1. **Completed in V1:** make execution feedback truthful and consistent. A
   cancelled or failed operation keeps its last real progress instead of being
   shown as complete. The activity log is user-readable and copyable.
2. **Next small PR:** introduce `OperationEvent` and adapt the five existing
   callback signatures at their boundaries. Keep the legacy callbacks as thin
   adapters, so processing algorithms do not need a big-bang rewrite.
3. **Extract runner:** move thread startup, cancellation, button locking, and
   final outcome handling from `main.py` into one `PipelineRunner`. Each stage
   supplies a command and a callable only.
4. **Extract settings and commands:** replace the Tk-variable registry with
   typed dataclasses plus a versioned settings repository. Validate before an
   operation enters the runner.
5. **Make views passive:** each tab renders a view model and emits commands.
   Views do not reach into the root application or invoke logic modules.
6. **V2 shell:** compose the extracted pieces under `app/` and retain a
   compatibility launcher until all stages use the new runner.

## Guardrails and acceptance criteria

- One source of truth for an operation outcome; only successful work is 100%.
- `ProgressEvent` always includes stage, current/total when known, and a human
  phase. Indeterminate work explicitly reports that it is preparing.
- Every error is both visible to the user and preserved with traceback/details
  in the activity log.
- Cancellation is cooperative, idempotent, and testable without Tk.
- Unit tests run without a display or installed optional UI dependencies.
- Existing pipeline APIs stay compatible until their adapter is removed in a
  deliberate major-version change.

## First backlog slices

| Slice | Benefit | Risk |
| --- | --- | --- |
| `OperationEvent` + adapters | Uniform feedback and testable event stream | Low |
| `PipelineRunner` extraction | One place for worker/cancel bugs | Medium |
| Typed commands/settings migration | Better validation and migrations | Medium |
| Passive tab view models | Faster UI changes and focused UI tests | Medium |

Do not combine algorithm optimisation with these boundary changes. Keep each
slice independently runnable and covered by its focused tests.

## Engine boundary now available

`app/engines/` now provides the V2 engine contract and explicit built-in
registry. The UI persists an `engine_profile` (`Stable` or `Fast`) per Flow,
Align and Stack stage. Advanced selections are optional: an empty selection
uses the profile default.

- Flow: `dao`, `opencv-contours`, `opencv-components`, optional `sep`, and
  optional `astroalign-asterism` fallback.
- Align: `opencv-stable` and `opencv-fast`. Stable retains the prior
  scikit-image compatibility path for cubic/Lanczos; Fast enables native
  OpenCV alternatives.
- Stack: `stable-numpy` and `fast-numba`. Fast uses cacheable kernels for
  no-rejection reductions and retains the stable reducer for median/rejection.

Optional engines are intentionally lazy dependencies. Install `sep` to enable
the SEP detector and `astroalign` to enable the asterism fallback; choosing an
unavailable optional engine produces a visible Flow error instead of silently
changing the selected algorithm.
