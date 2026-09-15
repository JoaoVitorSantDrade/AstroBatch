"""Adapt legacy pipeline APIs to a single application outcome contract."""
from .runner import OperationResult


def execute_pipeline(stage, args, log, progress, cancel):
    if stage == "Calibration":
        from calibration_logic import run_calibration_pipeline
        config = args[0]
        if hasattr(config, "to_legacy_config"):
            config = config.to_legacy_config()
        result = run_calibration_pipeline(config, log, progress, cancel)
    elif stage == "Batch":
        from batch_logic import process_fits_logic
        config = args[0]
        if hasattr(config, "to_legacy_config"):
            config = config.to_legacy_config()
        count, batches = process_fits_logic(config, log, progress, cancel)
        result = {"status": "success" if count else "failed", "message": f"Batch: {count} frames, {batches} batches."}
    elif stage == "Flow":
        from astroflow_logic import process_all_flows
        if len(args) == 1 and hasattr(args[0], "batch_dir"):
            command = args[0]
            batch_dir, config = command.batch_dir, command.to_legacy_config()
        else:
            batch_dir, config = args[0], args[1]
            if hasattr(config, "to_legacy_config"):
                config = config.to_legacy_config()
        result = process_all_flows(batch_dir, config, log, progress, cancel)
    elif stage == "ReferenceChange":
        from astroflow_logic import apply_reference_change
        command = args[0]
        if hasattr(command, "batch_dir"):
            result = apply_reference_change(
                command.batch_dir,
                command.new_reference,
                command.base_dir,
                cancel,
            )
        else:
            result = apply_reference_change(args[0], args[1], args[2] if len(args) > 2 else None, cancel)
    elif stage == "Align":
        from astroalign_logic import process_all_alignments
        if len(args) == 1 and hasattr(args[0], "base_dir"):
            command = args[0]
            base_dir, output_dir, config = command.base_dir, command.output_dir, command.to_legacy_config()
        else:
            base_dir, output_dir, config = args[0], args[1], args[2]
            if hasattr(config, "to_legacy_config"):
                config = config.to_legacy_config()
        saved, failed = process_all_alignments(base_dir, output_dir, config, log, progress, cancel)
        result = {"status": "partial" if saved and failed else "failed" if failed or not saved else "success",
                  "message": f"Align: {saved} processados, {failed} falhas."}
    elif stage == "Stack":
        from stacking_logic import process_all_stacking
        if len(args) == 1 and hasattr(args[0], "input_dir"):
            command = args[0]
            result = process_all_stacking(command.input_dir, command.to_legacy_config(), progress, log, cancel)
        else:
            config = args[1]
            if hasattr(config, "to_legacy_config"):
                config = config.to_legacy_config()
            result = process_all_stacking(args[0], config, progress, log, cancel)
    elif stage == "HDR":
        from hdr_logic import run_hdr_pipeline
        config = args[0]
        if hasattr(config, "to_legacy_config"):
            config = config.to_legacy_config()
        result = run_hdr_pipeline(config, log, progress, cancel)
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
