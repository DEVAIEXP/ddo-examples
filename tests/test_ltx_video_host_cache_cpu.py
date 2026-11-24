"""Native Linux linear video cleanup defaults and reporting without CUDA."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts.helpers.ltx2 import host_cache
from scripts.helpers.pytorch_allocator import configure_pytorch_allocator


@pytest.mark.parametrize(("system", "release", "preset", "expected"), [
    ("linux", "6.17.0-generic", "one_shot_fast", True),
    ("linux", "6.17.0-generic", "low_ram_safe", True),
    ("linux", "6.17.0-generic", "auto", True),
    ("linux", "6.18.33.2-microsoft-standard-WSL2", "one_shot_fast", False),
    ("win32", "10", "one_shot_fast", False),
    ("linux", "6.17.0-generic", "off", False),
    ("linux", "6.17.0-generic", "diffusers_leaf_offload_compat", False),
    ("linux", "6.17.0-generic", "diffusers_offload_compat", False),
])
def test_default_preserves_platform_and_route_boundaries(monkeypatch, system, release, preset, expected):
    monkeypatch.setattr(host_cache.sys, "platform", system)
    monkeypatch.setattr(host_cache.platform, "release", lambda: release)
    assert host_cache.default_host_cache_release(preset) is expected


def test_public_helper_is_lazy_and_reports_cleanup(monkeypatch):
    calls = []
    observation = {"status": "released", "elapsed_seconds": 1.25, "allocator_bytes_delta": None}
    report = SimpleNamespace(as_dict=lambda: {"setup": {"host_cache_release": observation}})

    def release(**kwargs):
        assert kwargs == {"enabled": True, "phase": "before_vae", "device": "cuda:0"}
        calls.append("release")
        return report

    monkeypatch.setitem(sys.modules, "diffusers_dynamic_offloader",
                        SimpleNamespace(release_pytorch_host_cache=release))
    tracker = SimpleNamespace(
        record_ddo_metrics=lambda value, **kwargs: calls.append((value, kwargs)),
        record_event=lambda *args, **kwargs: calls.append((args, kwargs)))
    host_cache.release_host_cache_before_vae(tracker, "cuda:0")
    assert calls == ["release", (report, {"component": "host_cache_before_vae"}),
                     (("release_pytorch_host_cache_before_vae", 1.25),
                      {"status": "released", "allocator_bytes_delta": None})]


def test_release_errors_propagate(monkeypatch):
    def release(**kwargs):
        raise RuntimeError("completion failed")

    monkeypatch.setitem(sys.modules, "diffusers_dynamic_offloader",
                        SimpleNamespace(release_pytorch_host_cache=release))
    with pytest.raises(RuntimeError, match="completion failed"):
        host_cache.release_host_cache_before_vae(SimpleNamespace(), "cuda:0")


@pytest.mark.parametrize("explicit", [None, True, False])
def test_linear_parser_allows_default_and_override(monkeypatch, explicit):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU parser initialized CUDA"))
    from scripts.ltx2_video import run_dynamic_modular_distilled as runner

    monkeypatch.setattr(runner, "default_host_cache_release", lambda preset: True)
    flags = [] if explicit is None else ["--release-pytorch-host-cache-before-vae" if explicit
                                        else "--no-release-pytorch-host-cache-before-vae"]
    assert runner.parse_args(flags).release_pytorch_host_cache_before_vae is (explicit is not False)
    assert not hasattr(runner.parse_args([], official_leaf=True), "release_pytorch_host_cache_before_vae")


def test_cleanup_runs_after_retirement_before_decode_and_skips_calibration():
    source = (Path(__file__).parents[1] / "scripts/ltx2_video/run_dynamic_modular_distilled.py").read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    release = next(node for node in calls if isinstance(node.func, ast.Name)
                   and node.func.id == "release_host_cache_before_vae")
    retire = next(node for node in ast.walk(tree) if isinstance(node, ast.Delete)
                  and {x.id for x in node.targets if isinstance(x, ast.Name)}
                  == {"cleanup", "profile", "result", "transformer", "scheduler"})
    calibration = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                       and isinstance(node.test, ast.Attribute) and node.test.attr == "build_ddo_profile"
                       and any(isinstance(x, ast.Return) for x in node.body))
    # Decode runs as per-component phases: load_decode_<name>, decode_<name>, then save_video.
    decode = min((node for node in calls if isinstance(node.func, ast.Attribute)
                  and node.func.attr == "step_start" and node.args
                  and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].left, ast.Constant)
                  and node.args[0].left.value == "load_decode_"), key=lambda node: node.lineno)
    assert retire.lineno < calibration.lineno < release.lineno < decode.lineno


def test_allocator_startup_environment_is_captured_before_runner_helpers(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/ltx2_video/run_dynamic_modular_distilled.py"
    tree = ast.parse(path.read_text())
    capture = next(node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "PYTORCH_ALLOCATOR_STARTUP_ENVIRONMENT"
                           for target in node.targets))
    first_helper = next(node for node in tree.body if isinstance(node, ast.ImportFrom)
                        and node.module.startswith("scripts.helpers")
                        and node.module != "scripts.helpers.pytorch_allocator")
    assert capture.lineno < first_helper.lineno
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    namespace = {"configure_pytorch_allocator": configure_pytorch_allocator}
    exec(compile(ast.Module(body=[capture], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "later-change")
    assert namespace["PYTORCH_ALLOCATOR_STARTUP_ENVIRONMENT"] == {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True", "PYTORCH_CUDA_ALLOC_CONF": None}
