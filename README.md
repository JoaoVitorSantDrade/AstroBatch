# AstroBatch V2

AstroBatch is a Windows desktop workflow for processing astrophotography FITS
sessions. It guides a project from raw frames to a reviewable 16-bit FITS
stack, while keeping processing jobs responsive, cancellable, and traceable.

## What it does

1. **Import** — validates a source folder of FITS frames.
2. **Calibrate** — applies optional Dark and Flat masters and writes normalized
   16-bit FITS products.
3. **Analyze** — groups compatible frames into batches.
4. **Flow** — detects stars and calculates local/global transforms.
5. **Align** — applies transforms, debayering, registration, and valid masks.
6. **Stack** — selects frames, handles rejection/normalization, estimates safe
   memory usage, and writes the final 16-bit FITS image.
7. **Review** — previews the result with auto-stretch and FITS metadata.

## Install and run

AstroBatch requires Python 3.10+ on Windows.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
astrobatch
```

If PowerShell blocks activation, run the commands with
`.\.venv\Scripts\python.exe` instead. The application can also be started with
`python main.py`.

## Projects and outputs

Create a project workspace from the app and select its source FITS directory.
AstroBatch stores `astrobatch.project.json` in that workspace, alongside an
`outputs/` directory with one folder per pipeline stage. The manifest records
settings, published artifacts, stage state, and run history. It is intentionally
separate from V1's `astro_config.json`; V2 does not migrate that file.

The scientific contract remains compatible with V1: calibration and final stack
products use 16-bit FITS, calibration metadata records the normalization range,
and aligned outputs retain their valid masks.

## Architecture

```text
astrobatch/
  core/        Job lifecycle, events, models, FITS and resource utilities
  project/     Versioned workspace manifest and atomic persistence
  processing/  UI-independent calibration, batch, flow, alignment, stacking, CPU kernels
  services/    Typed stage orchestration over the processing engines
  ui/          PySide6 shell, stage pages, structured log, FITS viewer
```

`JobManager` permits one heavy job at a time. It runs processing in a worker
thread and publishes typed progress, log, artifact, completion, failure, and
cancellation events to the PySide6 UI. Processing modules do not import or
manipulate UI frameworks.

The root-level `*_logic.py` and `cpu_kernels.py` files are deprecated
compatibility shims. New code must import from `astrobatch.processing`.

## Development

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -W ignore -m unittest discover -s tests -q
```

Performance probes live in `benchmarks/`. Architecture and historical planning
notes live in `Docs/`; see [the documentation index](Docs/README.md).

## Packaging

The PyInstaller specification is at `packaging/astrobatch.spec`. After installing
the development extras, build a Windows executable with:

```powershell
.\.venv\Scripts\pyinstaller.exe packaging\astrobatch.spec
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `astrobatch/` | V2 application package |
| `tests/` | Unit, integration, architecture, and UI smoke tests |
| `benchmarks/` | Repeatable processing performance checks |
| `Docs/` | Product, architecture, performance, and historical roadmap docs |
| `packaging/` | Windows distribution assets |
| `main.py` | Small compatibility launcher for the V2 application |
