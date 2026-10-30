"""CPU-only original-error preservation and friendly reporting contracts."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.helpers.failure_reporting import report_memory_failure, stop_monitor_preserving_error


def test_report_preserves_original_and_prints_message(capsys):
    error = RuntimeError("original CUDA fault")
    metrics = {}
    with patch("diffusers_dynamic_offloader.diagnose_memory_failure", return_value={
            "message": "Low system commit; cause unproven", "classification": "low"}):
        report_memory_failure(error, metrics)
    assert str(error) == "original CUDA fault"
    assert "cause unproven" in error.__notes__[0]
    assert "Low system commit" in capsys.readouterr().err
    assert metrics["memory_failure"]["classification"] == "low"


def test_diagnostic_failure_never_replaces_original():
    metrics = {}
    with patch("diffusers_dynamic_offloader.diagnose_memory_failure", side_effect=OSError("query")):
        report_memory_failure(RuntimeError("model"), metrics)
    assert metrics["memory_failure"]["classification"] == "unavailable"


def test_secondary_monitor_error_does_not_mask_model_error():
    def fail():
        raise RuntimeError("monitor")
    monitor = SimpleNamespace(stop=fail)
    metrics = {}
    original = ValueError("model")
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        finally:
            stop_monitor_preserving_error(monitor, metrics)
    assert caught.value is original
    assert metrics["cleanup_errors"] == ["monitor"]
    assert "peaks are incomplete" in original.__notes__[0]
    with pytest.raises(RuntimeError, match="monitor"):
        stop_monitor_preserving_error(monitor, {})


def test_interrupt_is_not_diagnosed():
    metrics = {}
    report_memory_failure(KeyboardInterrupt(), metrics)
    assert metrics == {}
