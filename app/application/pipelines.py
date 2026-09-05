"""Adapt legacy pipeline APIs to a single application outcome contract."""
from .runner import OperationResult


def execute_pipeline(stage, args, log, progress, cancel):
    if stage == "Calibration":
        from calibration_logic import run_calibration_pipeline
        result = run_calibration_pipeline(args[0], log, progress, cancel)
    elif stage == "Batch":
        from batch_logic import process_fits_logic
        count, batches = process_fits_logic(args[0], log, progress, cancel)
        result = {"status": "success" if count else "failed", "message": f"Batch: {count} frames, {batches} batches."}
    elif stage == "Flow":
        from astroflow_logic import process_all_flows
        result = process_all_flows(args[0], args[1], log, progress, cancel)
    elif stage == "Align":
        from astroalign_logic import process_all_alignments
        saved, failed = process_all_alignments(args[0], args[1], args[2], log, progress, cancel)
        result = {"status": "partial" if saved and failed else "failed" if failed or not saved else "success",
                  "message": f"Align: {saved} processados, {failed} falhas."}
    elif stage == "Stack":
        from stacking_logic import process_all_stacking
        result = process_all_stacking(args[0], args[1], progress, log, cancel)
    elif stage == "HDR":
        from hdr_logic import run_hdr_pipeline
        result = run_hdr_pipeline(args[0], log, progress, cancel)
    else:
        raise ValueError(f"Unknown pipeline: {stage}")
    if cancel.is_set():
        return OperationResult("cancelled", f"{stage} cancelado.")
    if not isinstance(result, dict):
        return OperationResult("failed", f"{stage}: resultado não confirmado; consulte Atividade.")
    outcome = str(result.get("status", "failed"))
    if outcome == "error":
        outcome = "failed"
    message = result.get("message") or result.get("error") or result.get("reason")
    if not message:
        message = f"{stage} concluído." if outcome == "success" else f"{stage}: {outcome}."
    if result.get("output_path"):
        log(f"[{stage}] Saída: {result['output_path']}\n")
    return OperationResult(outcome, str(message))
