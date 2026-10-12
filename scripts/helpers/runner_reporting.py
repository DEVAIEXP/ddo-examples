"""Shared reporting lifecycle; model loading and phase orchestration stay in runners."""

import argparse
import importlib.metadata
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .inference_utils import RunTracker


def add_reporting_arguments(parser):
    parser.add_argument(
        "--memory-monitor",
        choices=("rss", "full"),
        default="rss",
        help="RSS is cheap; full USS monitoring may affect performance.",
    )
    parser.add_argument(
        "--metrics-level",
        choices=(1, 2),
        type=int,
        default=2,
        help="1: phase/workload observations; 2: also include public DDO component reports.",
    )
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-profile", action="store_true")
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-denoise-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ddo-gpu-timing", action="store_true", help="Opt-in CUDA boundary events; no per-step synchronization."
    )
    parser.add_argument(
        "--purge-before-transformer",
        action="store_true",
        help="Optional Windows-only standby helper; success is reported, not assumed.",
    )
    return parser


def environment_metadata():
    versions = {}
    for package in ("torch", "diffusers", "transformers", "diffusers-dynamic-offloader", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "executable": sys.executable, "package_versions": versions}


class RunnerReporting:
    """Compose RunTracker with phase/error persistence, without changing its schema."""

    def __init__(self, args, metrics, device, *, tracker_factory=RunTracker):
        self.args, self.metrics = args, metrics
        metrics.update(
            reporting_schema_version=2,
            status="running",
            requested_controls={
                key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            },
            environment=environment_metadata(),
        )
        self.tracker = tracker_factory(
            device, metrics, show_metrics=args.show_metrics, memory_monitor=args.memory_monitor
        )
        self.started = time.perf_counter()

    def __enter__(self):
        return self

    @contextmanager
    def phase(self, name):
        original = None
        start = self.tracker.step_start(name)
        try:
            yield
        except BaseException as error:
            original = error
            self.metrics["failed_phase"] = name
            raise
        finally:
            try:
                self.tracker.step_end(name, start)
            except BaseException as monitor_error:
                if original is None:
                    raise
                self.metrics.setdefault("cleanup_errors", []).append(str(monitor_error))
                original.add_note(f"Phase monitoring also failed: {monitor_error}")

    def report(self, value, component):
        if self.args.metrics_level >= 2:
            self.tracker.record_ddo_metrics(value, component)
        if self.args.show_profile and (not self.args.show_metrics or self.args.metrics_level < 2):
            print(value.format_profile(), flush=True)

    def purge_before_transformer(self):
        from diffusers_dynamic_offloader import maybe_purge_windows_standby_cache

        result = maybe_purge_windows_standby_cache(
            "before_transformer",
            preset="one_shot_fast",
            environ={
                "DDO_PURGE_WINDOWS_STANDBY_BEFORE_TRANSFORMER": "1" if self.args.purge_before_transformer else "0"
            },
            print_message=self.args.show_metrics,
            record_event=self.tracker.record_event,
        )
        self.metrics["standby_purge"] = {
            "requested": self.args.purge_before_transformer,
            "result": result,
        }

    def __exit__(self, exc_type, error, traceback):
        if error is not None:
            self.metrics.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error_type=type(error).__name__,
                error=str(error),
                error_notes=list(getattr(error, "__notes__", ())),
            )
        else:
            self.metrics["status"] = "completed"
        self.metrics.update(
            total_elapsed_sec=time.perf_counter() - self.started,
            global_peak_vram_gb=self.tracker.global_peak_vram,
            global_peak_ram_gb=self.tracker.global_peak_ram,
        )
        try:
            if self.args.save_metrics:
                self.args.output_dir.mkdir(parents=True, exist_ok=True)
                target = self.args.output_dir / "metrics.json"
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")
                temporary.replace(target)
            if self.args.show_metrics:
                print(
                    f"Status: {self.metrics['status']} | Total: {self.metrics['total_elapsed_sec']:.2f} s"
                    f" | Absolute GPU peak: {self.tracker.global_peak_vram_absolute:.2f} GiB"
                    f" | RAM ({self.tracker.monitor.ram_metric}): {self.tracker.global_peak_ram:.2f} GiB",
                    flush=True,
                )
                print(f"Output directory: {self.args.output_dir}", flush=True)
        except BaseException as save_error:
            if error is None:
                raise
            error.add_note(f"Reporting also failed: {save_error}")
        return False
