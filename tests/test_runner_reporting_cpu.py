"""CPU-only reporting/lifecycle contracts; no model download or CUDA execution."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.helpers.runner_reporting import RunnerReporting, add_reporting_arguments


class Tracker:
    def __init__(self, device, metrics, **kwargs):
        self.metrics = metrics
        self.global_peak_vram = 1
        self.global_peak_vram_absolute = 2
        self.global_peak_ram = 3
        self.monitor = SimpleNamespace(ram_metric=kwargs["memory_monitor"])
        self.fail_stop = False
        self.ended = []
        self.reports = []

    def step_start(self, name):
        return 0

    def step_end(self, name, start):
        self.ended.append(name)
        if self.fail_stop:
            raise RuntimeError("monitor failure")

    def record_ddo_metrics(self, report, component):
        self.reports.append(component)


def arguments(tmp_path):
    parser = add_reporting_arguments(argparse.ArgumentParser())
    args = parser.parse_args([])
    args.output_dir = tmp_path
    args.show_metrics = False
    return args


def test_reporting_defaults_are_opt_in_for_cuda_and_purge(tmp_path):
    args = arguments(tmp_path)
    assert args.memory_monitor == "rss"
    assert args.metrics_level == 2
    assert not args.ddo_gpu_timing
    assert not args.purge_before_transformer


def test_success_and_partial_error_reports_use_same_schema(tmp_path):
    args = arguments(tmp_path)
    metrics = {"events": [], "steps": []}
    with RunnerReporting(args, metrics, "cpu", tracker_factory=Tracker) as session, session.phase("prepare"):
        metrics["prepared"] = True
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["status"] == "completed"
    assert saved["reporting_schema_version"] == 2
    assert saved["requested_controls"]["memory_monitor"] == "rss"
    assert saved["global_peak_ram_gb"] == 3
    assert not (tmp_path / "metrics.json.tmp").exists()

    error = RuntimeError("original forward failure")
    with (
        pytest.raises(RuntimeError) as caught,
        RunnerReporting(args, metrics, "cpu", tracker_factory=Tracker) as session,
        session.phase("denoise"),
    ):
        raise error
    assert caught.value is error
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["status"] == "failed"
    assert saved["failed_phase"] == "denoise"
    assert saved["prepared"] is True


def test_monitor_failure_does_not_mask_original_failure(tmp_path):
    error = ValueError("model failure")
    with (
        pytest.raises(ValueError) as caught,
        RunnerReporting(arguments(tmp_path), {"events": [], "steps": []}, "cpu", tracker_factory=Tracker) as session,
    ):
        session.tracker.fail_stop = True
        with session.phase("denoise"):
            raise error
    assert caught.value is error
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["cleanup_errors"] == ["monitor failure"]
    assert saved["error_notes"]


def test_interruption_stops_phase_and_saves_report(tmp_path):
    with (
        pytest.raises(KeyboardInterrupt),
        RunnerReporting(arguments(tmp_path), {"events": [], "steps": []}, "cpu", tracker_factory=Tracker) as session,
        session.phase("denoise"),
    ):
        raise KeyboardInterrupt()
    assert session.tracker.ended == ["denoise"]
    assert json.loads((tmp_path / "metrics.json").read_text())["status"] == "interrupted"


def test_disabled_metrics_do_not_write_files(tmp_path):
    args = arguments(tmp_path)
    args.save_metrics = False
    with RunnerReporting(args, {"events": [], "steps": []}, "cpu", tracker_factory=Tracker):
        pass
    assert not (tmp_path / "metrics.json").exists()


def test_public_report_consumption_respects_detail_level(tmp_path):
    args = arguments(tmp_path)
    with RunnerReporting(args, {"events": [], "steps": []}, "cpu", tracker_factory=Tracker) as session:
        session.report(object(), "transformer")
        args.metrics_level = 1
        session.report(object(), "text_encoder")
    assert session.tracker.reports == ["transformer"]
