import contextlib
import ctypes
import gc
import importlib.metadata
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


def get_pinned_host_memory_stats():
    """Return the CUDA host-allocator counters when the local PyTorch build exposes them."""
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "host_memory_stats"):
        return {}
    with contextlib.suppress(Exception):
        stats = torch.cuda.host_memory_stats()
        return {
            "allocated_gb": round(stats.get("allocated_bytes.current", 0) / 1024**3, 4),
            "active_gb": round(stats.get("active_bytes.current", 0) / 1024**3, 4),
        }
    return {}


def release_pinned_host_cache():
    """Release unused PyTorch CUDA host-cache blocks at an explicit stage boundary.

    ``pin_memory`` allocations can otherwise remain cached for the rest of a long
    process. This is intentionally opt-in because that cache improves repeated
    inference throughput.
    """
    before = get_pinned_host_memory_stats()
    released = False
    if torch.cuda.is_available():
        empty_host_cache = getattr(torch._C, "_accelerator_emptyHostCache", None)
        if callable(empty_host_cache):
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
                empty_host_cache()
                released = True
    return {"released": released, "before": before, "after": get_pinned_host_memory_stats()}


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


class PeakMemoryMonitor:
    def __init__(self, device="cuda:0", interval=0.1, *, mode="full"):
        if mode not in {"full", "rss"}:
            raise ValueError("memory monitor mode must be full or rss")
        self.ram_metric = "rss" if mode == "rss" else "uss" if os.name == "nt" else "rss_anon_with_uss_fallback"
        self._get_ram_gb = get_ram_gb if mode == "full" else lambda: psutil.Process().memory_info().rss / 1024**3
        self.device = device
        self.peak_ram = 0.0
        self.peak_vram = 0.0
        self._running = False
        self._thread = None
        self._interval = interval

    def start(self):
        self.peak_ram = self._get_ram_gb()
        self.peak_vram = get_gpu_used_gb(self.device)
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            self.peak_ram = max(self.peak_ram, self._get_ram_gb())
            self.peak_vram = max(self.peak_vram, get_gpu_used_gb(self.device))
            time.sleep(self._interval)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        return self.peak_vram, self.peak_ram


class RunTracker:
    def __init__(self, device, run_metrics, interval=0.1, show_metrics=True, *, memory_monitor="full"):
        self.device = device
        self.run_metrics = run_metrics
        self.show_metrics = show_metrics
        self.monitor = PeakMemoryMonitor(device, interval=interval, mode=memory_monitor)
        self.run_metrics["memory_monitor"] = {
            "mode": memory_monitor, "ram_metric": self.monitor.ram_metric,
            "gpu_metric": "cuda_mem_get_info_used_minus_baseline",
            "gpu_absolute_metric": "cuda_mem_get_info_used", "unit": "GiB",
        }
        torch.cuda.init()
        self.vram_baseline = get_gpu_used_gb(device)
        self.run_metrics["vram_baseline_gb"] = self.vram_baseline
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
        """Store DDO's public serializable report without reading runtime internals."""
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
        ram = self.monitor._get_ram_gb()
        vram = get_gpu_used_gb(self.device) - self.vram_baseline
        self.global_peak_ram = max(self.global_peak_ram, ram)
        self.global_peak_vram = max(self.global_peak_vram, vram)
        return vram, ram

    def step_end(self, step_name, t0):
        elapsed = time.time() - t0
        peak_vram_abs, peak_ram = self.monitor.stop()
        peak_vram = peak_vram_abs - self.vram_baseline
        if self.show_metrics:
            print(f"  [{step_name}] {elapsed:.1f}s | Peak VRAM: {peak_vram:.2f} GB | Peak RAM: {peak_ram:.2f} GB")
        self.global_peak_vram = max(self.global_peak_vram, peak_vram)
        self.global_peak_ram = max(self.global_peak_ram, peak_ram)
        metric = {
            "name": step_name,
            "elapsed_sec": round(elapsed, 4),
            "peak_vram_gb": round(peak_vram, 4),
            "peak_vram_absolute_gb": round(peak_vram_abs, 4),
            "peak_ram_gb": round(peak_ram, 4),
        }
        self.run_metrics["steps"].append(metric)
        return metric

    def total_elapsed(self):
        return time.time() - self.script_start
