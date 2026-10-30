"""Failure-only diagnostics that preserve the original exception and ownership."""
import sys


def report_memory_failure(error, metrics):
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return
    try:
        from diffusers_dynamic_offloader import diagnose_memory_failure

        report = diagnose_memory_failure(error)
        metrics["memory_failure"] = report
        if report["message"]:
            error.add_note(report["message"])
            print("Memory diagnostic: " + report["message"], file=sys.stderr, flush=True)
    except Exception as diagnostic_error:
        # Older installed DDO packages may not expose this optional diagnostic.
        metrics["memory_failure"] = {"schema": 1, "classification": "unavailable",
                                     "diagnostic_error": str(diagnostic_error)}


def stop_monitor_preserving_error(monitor, metrics):
    """Do not replace an active model error with a secondary monitor error."""
    original = sys.exc_info()[1]
    try:
        return monitor.stop()
    except Exception as error:
        if original is None:
            raise
        metrics.setdefault("cleanup_errors", []).append(str(error))
        original.add_note(f"Memory monitoring also failed; peaks are incomplete: {error}")
