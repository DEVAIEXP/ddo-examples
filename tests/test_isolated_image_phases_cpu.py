"""CPU contracts for independent image-phase peaks and official pipeline calls."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.image_suite import verify_phase_memory
from scripts.helpers import inference_utils as utils
from scripts.helpers.qwen_pipeline_phases import measure_qwen_pipeline_phases

ROOT = Path(__file__).resolve().parents[1]


def make_tracker(monkeypatch):
    clock = [0.0]
    usage = [1.0, 2.0]
    monkeypatch.setattr(utils, "get_gpu_used_gb", lambda device: usage[0])
    monkeypatch.setattr(utils, "get_ram_rss_gb", lambda: usage[1])
    monkeypatch.setattr(utils.time, "time", lambda: clock[0])
    # Drive the real sampler deterministically instead of creating a poll thread.
    monkeypatch.setattr(utils.threading, "Thread", lambda **kw: SimpleNamespace(start=lambda: None, join=lambda: None))
    tracker = utils.RunTracker.__new__(utils.RunTracker)
    tracker.monitor = utils.PeakMemoryMonitor(mode="rss")
    tracker.run_metrics = {"steps": []}
    tracker.vram_baseline = 1.0
    tracker.global_peak_vram_absolute = 1.0
    tracker.global_peak_vram = tracker.global_peak_ram = 0.0
    tracker.show_metrics = False

    def sample(gpu, ram, seconds):
        usage[:] = [gpu, ram]
        clock[0] += seconds
        tracker.monitor.sample_once()

    return tracker, sample


def test_child_peaks_exclude_setup_decode_and_cleanup(monkeypatch):
    tracker, sample = make_tracker(monkeypatch)
    start = tracker.step_start("generate")
    sample(9, 30, 10)  # Transformer setup, outside denoising.
    sample(2, 4, 1)
    with tracker.subphase("denoise"):
        sample(5, 8, 7)
    sample(10, 40, 20)  # Cleanup, outside denoising.
    sample(2, 4, 1)
    with tracker.subphase("decode"):
        sample(3, 6, 2)
    tracker.step_end("generate", start)
    denoise, decode, parent = tracker.run_metrics["steps"]
    assert (denoise["elapsed_sec"], denoise["peak_vram_absolute_gb"], denoise["peak_ram_gb"]) == (7, 5, 8)
    assert (decode["elapsed_sec"], decode["peak_vram_absolute_gb"], decode["peak_ram_gb"]) == (2, 3, 6)
    assert (parent["peak_vram_absolute_gb"], parent["peak_ram_gb"]) == (10, 40)
    assert denoise["parent_phase"] == "generate"
    assert denoise["memory_monitor"]["samples"] == 2
    assert tracker.global_peak_vram_absolute == 10
    assert not tracker.monitor._windows
    verify_phase_memory(tracker.run_metrics, ["denoise", "decode"])


def test_failed_sampling_does_not_replace_model_exception(monkeypatch):
    tracker, sample = make_tracker(monkeypatch)
    tracker.step_start("generate")
    original = ValueError("model failed")
    with pytest.raises(ValueError) as caught:
        with tracker.subphase("denoise"):
            tracker.monitor._error = RuntimeError("query failed")
            raise original
    assert caught.value is original
    assert tracker.run_metrics["cleanup_errors"]
    assert tracker.run_metrics["steps"] == []
    assert not tracker.monitor._windows


class Processor:
    def postprocess(self, image, **kwargs):
        return image


class Pipeline:
    def __init__(self):
        self.image_processor = Processor()

    @contextmanager
    def progress_bar(self, total):
        yield total


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_qwen_pre_decode_cleanup_is_separate_and_restores_on_failure(monkeypatch, cleanup_fails):
    tracker, sample = make_tracker(monkeypatch)
    tracker.step_start("generate")
    pipe = Pipeline()
    metrics = {}
    events = []

    def cleanup():
        events.append("cleanup")
        sample(8, 12, 3)
        if cleanup_fails:
            raise RuntimeError("cleanup failed")
        sample(2, 3, 0)

    def execute():
        with measure_qwen_pipeline_phases(pipe, tracker, metrics, before_decode=cleanup):
            with pipe.progress_bar(total=1):
                sample(4, 6, 2)
            events.append("decode")
            sample(5, 7, 4)
            pipe.image_processor.postprocess(object())

    if cleanup_fails:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            execute()
        assert events == ["cleanup"]
        assert "pre_decode_cleanup" not in metrics
        assert [x["name"] for x in tracker.run_metrics["steps"]] == ["denoise", "cleanup_before_decode"]
    else:
        execute()
        assert events == ["cleanup", "decode"]
        phases = {x["name"]: x for x in tracker.run_metrics["steps"]}
        assert phases["cleanup_before_decode"]["elapsed_sec"] == 3
        assert phases["denoise"]["elapsed_sec"] == 2
        assert phases["decode"]["elapsed_sec"] == 4
        assert phases["decode"]["peak_vram_absolute_gb"] == 5
        assert metrics["pre_decode_cleanup"]["status"] == "completed"
        assert metrics["isolated_phase_measurements"]["cuda_synchronization_added"] is True
    assert "progress_bar" not in vars(pipe)
    assert "postprocess" not in vars(pipe.image_processor)
    assert not tracker.monitor._windows


@pytest.mark.parametrize("failure", [None, "denoise", "decode", "postprocess", "missing"])
def test_qwen_boundaries_restore_methods_and_preserve_failure(monkeypatch, failure):
    tracker, sample = make_tracker(monkeypatch)
    start = tracker.step_start("generate")
    pipe = Pipeline()
    output = object()
    original_error = ValueError("original pipeline failure")
    if failure == "postprocess":
        def broken(*args, **kwargs):
            raise original_error
        pipe.image_processor.postprocess = broken
    before = dict(vars(pipe.image_processor))
    metrics = {}

    def execute():
        with measure_qwen_pipeline_phases(pipe, tracker, metrics):
            sample(9, 30, 10)  # Encoding belongs only to the enclosing pipeline.
            sample(2, 4, 1)
            with pipe.progress_bar(total=2) as bar:
                assert bar == 2
                sample(4, 8, 6)
                if failure == "denoise":
                    raise original_error
            sample(6, 10, 2)  # Latent preparation and VAE decode.
            if failure == "decode":
                raise original_error
            if failure != "missing":
                assert pipe.image_processor.postprocess(output, output_type="pil") is output

    if failure:
        expected = RuntimeError if failure == "missing" else ValueError
        with pytest.raises(expected) as caught:
            execute()
        if failure != "missing":
            assert caught.value is original_error
        assert "isolated_phase_measurements" not in metrics
    else:
        execute()
        assert metrics["isolated_phase_measurements"]["cuda_synchronization_added"] is False
        denoise, decode = tracker.run_metrics["steps"]
        assert (denoise["elapsed_sec"], denoise["peak_vram_absolute_gb"], denoise["peak_ram_gb"]) == (6, 4, 8)
        assert (decode["elapsed_sec"], decode["peak_vram_absolute_gb"], decode["peak_ram_gb"]) == (2, 6, 10)
    assert "progress_bar" not in vars(pipe)
    assert vars(pipe.image_processor) == before
    assert not tracker.monitor._windows
    sample(11, 50, 20)  # Hook cleanup and saving are outside both intervals.
    tracker.step_end("generate", start)


@pytest.mark.parametrize("runner", [
    "run_dynamic_modular_distilled.py", "run_dynamic_modular_vmm_distilled.py",
    "run_dynamic_modular_vmm_diagnostics_distilled.py",
    "run_dynamic_modular_sdnq_distilled.py", "run_dynamic_minimal.py",
])
def test_ltx_monitors_the_actual_call_before_readback_and_retirement(runner):
    source = (ROOT / "scripts/ltx2_image" / runner).read_text(encoding="utf-8")
    tree = ast.parse(source)
    measured = [n for n in ast.walk(tree) if isinstance(n, ast.With)
                and any("tracker.subphase(" in ast.unparse(item.context_expr) for item in n.items)]
    assert len(measured) == 1
    calls = [n for n in ast.walk(measured[0]) if isinstance(n, ast.Call)]
    assert any(ast.unparse(n.func) == "denoise_pipe" for n in calls)
    assert not any(ast.unparse(n.func) == "denoise_state.to" for n in calls)
    assert not any("cleanup" in ast.unparse(n.func) for n in calls)


@pytest.mark.parametrize("key,value", [
    ("peak_vram_absolute_gb", None), ("peak_ram_gb", float("nan")),
    ("elapsed_sec", True), ("status", "failed"),
])
def test_suite_rejects_incomplete_phase_memory(key, value):
    step = {"name": "denoise", "elapsed_sec": 2, "peak_vram_absolute_gb": 3,
            "peak_ram_gb": 4, "ram_metric": "rss",
            "memory_monitor": {"ram_metric": "rss", "samples": 3, "error_type": None}}
    step[key] = value
    with pytest.raises(ValueError):
        verify_phase_memory({"steps": [step]}, ["denoise"])


def test_suite_does_not_accept_aggregate_as_phase_memory():
    with pytest.raises(ValueError, match="separately monitored"):
        verify_phase_memory({"steps": [{"name": "generate", "peak_ram_gb": 20}]}, ["denoise"])
