# AstroBatch V2 Roadmap

This document outlines the planned improvements and execution strategy to upgrade AstroBatch from its current V1 state to a more robust, performant, and user-friendly V2.

## Phase 1: Core Architecture & UI Responsiveness (High Priority)
The primary goal of Phase 1 is to ensure the application never freezes during heavy astronomical calculations.

*   **Task 1.1: Threading Infrastructure**
    *   Create a generic `WorkerThread` class or utilize `concurrent.futures.ThreadPoolExecutor`.
    *   Implement a thread-safe signaling mechanism (e.g., `queue.Queue` or Tkinter's `event_generate`) to send progress updates (percentages, log messages) from the worker thread back to the main UI thread.
*   **Task 1.2: Decouple Logic and UI**
    *   Refactor `calibration_logic.py`, `astroalign_logic.py`, and `stacking_logic.py` so they accept progress callback functions as arguments, completely removing any direct UI manipulation from the logic files.
*   **Task 1.3: Apply Threading to Views**
    *   Update `calibration_view.py`, `align_view.py`, and `stacking_view.py` to launch their respective processes using the new threading infrastructure.

## Phase 2: Performance & Memory Optimization (Medium Priority)
Implement the strategies outlined in the existing planning documents to handle large datasets (hundreds of high-res images).

*   **Task 2.1: Dynamic Memory Management**
    *   Integrate the `psutil` library.
    *   Before stacking or aligning, calculate the total required memory based on image dimensions and count.
    *   Automatically adjust `chunk_size` in `stacking_logic.py` to prevent `MemoryError` and OS swapping.
*   **Task 2.2: Math Optimization (`numexpr`)**
    *   Audit `cpu_kernels.py` and logic files. Ensure `numexpr` is used as the default fallback for large array arithmetic (addition, division, mean calculations) where applicable, replacing standard `numpy` operations for speed.
*   **Task 2.3: Bit Depth Enforcement**
    *   Execute `BIT_DEPTH_ACTION_PLAN.md`. Ensure all intermediate temporary files are saved efficiently (e.g., float32) and the final output format is user-selectable but defaults to standard 32-bit float FITS.

## Phase 3: User Experience & Features (Medium/Low Priority)
Enhance the usability and feedback mechanisms of the application.

*   **Task 3.1: Integrated Logging Console**
    *   Add a scrollable text widget to `main.py` (perhaps a drawer or a bottom panel).
    *   Implement a custom Python `logging` handler that outputs to this widget.
    *   Replace `print()` statements throughout the codebase with proper `logger.info()`, `logger.warning()`, and `logger.error()`.
*   **Task 3.2: Configuration Persistence**
    *   Create a `config.py` module.
    *   Save UI states (checkboxes, selected directories, dropdown values) to a `config.json` file when the app closes and load them on startup.
*   **Task 3.3: Basic FITS Preview (Stretch Goal)**
    *   Research integrating `matplotlib.backends.backend_tkagg` within a `customtkinter` frame.
    *   Implement a simple "Preview" button next to file lists that opens a modal window showing the image with a basic Auto-Stretch applied (to make linear data visible).

## Phase 4: Testing & Release
*   **Task 4.1: Expand Test Suite**
    *   Update existing tests in `tests/` to accommodate async logic.
    *   Add tests for memory chunking edge cases.
*   **Task 4.2: Packaging**
    *   Create a `pyinstaller.spec` file to compile AstroBatch into a standalone executable (.exe) for easier distribution to Windows users without requiring a Python environment setup.


This plan sets a clear path for evolving AstroBatch into a robust tool. The immediate focus for V2 should absolutely be Phase 1 (Threading). Given the nature of astrophotography processing (stacking 200+ images), a freezing UI will be the most significant pain point for your users.