r"""Compare Tk console render CPU time with a git revision (default: HEAD).

Run: .venv\Scripts\python.exe benchmarks/console_benchmark.py [revision]
Uses a withdrawn real Tk Text widget; excludes the 50 ms scheduling intervals.
"""
import ast
import json
import queue
import statistics
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import AstroProcessManager


def main():
    revision = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:main.py"], cwd=ROOT, encoding="utf-8"
    )
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "AstroProcessManager")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_drain_log_queue")
    scope = {"queue": queue, "time": time, "tk": tk}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "baseline", "exec"), scope)
    root = tk.Tk()
    root.withdraw()
    try:
        results = {}
        for name, callback in (("baseline", scope["_drain_log_queue"]),
                               ("current", AstroProcessManager._drain_log_queue)):
            totals, longest = [], []
            for repeat in range(6):
                widget = tk.Text(root)
                app = SimpleNamespace(
                    log_queue=queue.Queue(), console_text=widget,
                    console_autoscroll_var=tk.BooleanVar(root, True),
                    _console_tag=AstroProcessManager._console_tag,
                    after=lambda *args: None, _drain_log_queue=lambda: None,
                )
                for index in range(4000):
                    app.log_queue.put(f"Frame {index}: processing complete\n")
                durations = []
                while not app.log_queue.empty():
                    start = time.perf_counter()
                    callback(app)
                    durations.append((time.perf_counter() - start) * 1000)
                assert "Frame 3999:" in widget.get("1.0", "end")
                if repeat:
                    totals.append(sum(durations))
                    longest.append(max(durations))
                widget.destroy()
            results[name] = {"total_render_ms_median": statistics.median(totals),
                             "longest_callback_ms_median": statistics.median(longest)}
        print(json.dumps(results, indent=2))
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
