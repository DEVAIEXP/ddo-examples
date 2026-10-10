import contextlib
import ctypes
import gc
import importlib.metadata
import math
import os
import threading
import time
from pathlib import Path

import psutil
import torch


def is_wsl() -> bool:
    if os.name == "nt":
        return False
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def should_malloc_trim() -> bool:
    value = os.environ.get("DIFFUSERS_RUNNER_MALLOC_TRIM")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return not is_wsl()


def get_sdnq_version():
    try:
        return importlib.metadata.version("sdnq")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
    except Exception:
        try:
            import sdnq
        except Exception:
            return "unavailable"
        return getattr(sdnq, "__version__", "unknown")


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if should_malloc_trim():
        with contextlib.suppress(Exception):
            ctypes.CDLL("libc.so.6").malloc_trim(0)


def get_ram_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) / 1024**2
    except FileNotFoundError:
        pass
    return psutil.Process().memory_full_info().uss / 1024**3


def get_gpu_used_gb(device="cuda:0"):
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1024**3


def get_ram_rss_gb():
    """Cheap resident working-set observation, explicitly NOT USS."""
    return psutil.Process().memory_info().rss / 1024**3


class PeakMemoryMonitor:
    def __init__(self, device="cuda:0", interval=0.1, *, mode="full"):
        if mode not in ("full", "rss"):
            raise ValueError("memory monitor mode must be full or rss")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("memory monitor interval must be finite and positive")
        self.device = device
        self.mode = mode
        self.ram_metric = "rss" if mode == "rss" else "uss" if os.name == "nt" else "rss_anon_with_uss_fallback"
        self.peak_ram = 0.0
        self.peak_vram = 0.0
        self._running = False
        self._thread = None
        self._interval = interval
        self._stop_event = threading.Event()
        self._error = None
        self._sampling_stats = {}

    def metadata(self):
        return {"mode": self.mode, "ram_metric": self.ram_metric,
                "gpu_metric": "cuda_mem_get_info_used_minus_baseline",
                "gpu_absolute_metric": "cuda_mem_get_info_used",
                "memory_unit": "GiB",
                "interval_sec": self._interval}

    def sample_once(self):
        start = time.perf_counter()
        try:
            ram = get_ram_rss_gb() if self.mode == "rss" else get_ram_gb()
        finally:
            key = "ram_query_seconds"
            self._sampling_stats[key] = self._sampling_stats.get(key, 0.0) + time.perf_counter() - start
        start = time.perf_counter()
        try:
            vram = get_gpu_used_gb(self.device)
        finally:
            key = "gpu_query_seconds"
            self._sampling_stats[key] = self._sampling_stats.get(key, 0.0) + time.perf_counter() - start
        self._sampling_stats["samples"] = self._sampling_stats.get("samples", 0) + 1
        self.peak_ram = max(self.peak_ram, ram)
        self.peak_vram = max(self.peak_vram, vram)
        return vram, ram

    def sampling_report(self):
        return {**self.metadata(), **self._sampling_stats,
                "error_type": None if self._error is None else type(self._error).__name__}

    def start(self):
        if self._running:
            raise RuntimeError("memory monitor is already running")
        self.peak_ram = self.peak_vram = 0.0
        self._sampling_stats = {}
        self._error = None
        self._stop_event.clear()
        self.sample_once()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        try:
            while not self._stop_event.is_set():
                self.sample_once()
                self._stop_event.wait(self._interval)
        except Exception as error:
            self._error = error
            self._stop_event.set()

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            raise RuntimeError("memory monitoring failed; peaks are incomplete") from self._error
        return self.peak_vram, self.peak_ram


class RunTracker:
    def __init__(self, device, run_metrics, interval=0.1, show_metrics=True, *, memory_monitor="full"):
        self.device = device
        self.run_metrics = run_metrics
        self.show_metrics = show_metrics
        self.monitor = PeakMemoryMonitor(device, interval=interval, mode=memory_monitor)
        self.run_metrics["memory_monitor"] = self.monitor.metadata()
        torch.cuda.init()
        self.vram_baseline = get_gpu_used_gb(device)
        self.run_metrics["vram_baseline_gb"] = self.vram_baseline
        self.global_peak_vram_absolute = self.vram_baseline
        self.run_metrics["global_peak_vram_absolute_gb"] = self.global_peak_vram_absolute
        self.script_start = time.time()
        self.global_peak_vram = 0.0
        self.global_peak_ram = 0.0
        if self.show_metrics:
            print(f"VRAM baseline: {self.vram_baseline:.2f} GB")

    def record_event(self, name, elapsed, **metadata):
        event = {"name": name, "elapsed_sec": round(elapsed, 4)}
        event.update(metadata)
        self.run_metrics["events"].append(event)
        if self.show_metrics:
            print(f"  [event] {name}: {elapsed:.4f}s")
        return event

    def record_ddo_metrics(self, report, component="transformer"):
        """Consume DDO's public report without depending on runtime internals.

        ``report`` intentionally follows the small ``as_dict`` /
        ``format_profile`` protocol exposed by DDO.  Keeping this boundary here
        means model runners do not need to know hook-state or VMM attribute
        names when DDO adds telemetry.
        """
        payload = report.as_dict()
        self.run_metrics.setdefault("dynamic_offload_reports", {})[component] = payload
        if self.show_metrics:
            print(report.format_profile())
        return payload

    def step_start(self, step_name):
        self.monitor.start()
        if self.show_metrics:
            print(f"\n{'-' * 70}")
            print(f"  {step_name}")
            print(f"{'-' * 70}")
        return time.time()

    def sample_memory(self):
        absolute_vram, ram = self.monitor.sample_once()
        vram = absolute_vram - self.vram_baseline
        self.global_peak_ram = max(self.global_peak_ram, ram)
        self.global_peak_vram = max(self.global_peak_vram, vram)
        self.global_peak_vram_absolute = max(self.global_peak_vram_absolute, absolute_vram)
        self.run_metrics["global_peak_vram_absolute_gb"] = self.global_peak_vram_absolute
        return vram, ram

    def step_end(self, step_name, t0):
        elapsed = time.time() - t0
        peak_vram_abs, peak_ram = self.monitor.stop()
        peak_vram = peak_vram_abs - self.vram_baseline
        if self.show_metrics:
            print(f"  [{step_name}] {elapsed:.1f}s | Peak VRAM: {peak_vram:.2f} GB | Peak RAM ({self.monitor.ram_metric}): {peak_ram:.2f} GB")
        self.global_peak_vram = max(self.global_peak_vram, peak_vram)
        self.global_peak_vram_absolute = max(self.global_peak_vram_absolute, peak_vram_abs)
        self.run_metrics["global_peak_vram_absolute_gb"] = self.global_peak_vram_absolute
        self.global_peak_ram = max(self.global_peak_ram, peak_ram)
        metric = {
            "name": step_name,
            "elapsed_sec": round(elapsed, 4),
            "peak_vram_gb": round(peak_vram, 4),
            "peak_vram_absolute_gb": peak_vram_abs,
            "peak_ram_gb": round(peak_ram, 4),
            "ram_metric": self.monitor.ram_metric,
            "memory_monitor": self.monitor.sampling_report(),
        }
        self.run_metrics["steps"].append(metric)
        return metric

    def total_elapsed(self):
        return time.time() - self.script_start
