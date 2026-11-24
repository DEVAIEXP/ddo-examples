"""Benchmark defaults: standard Hub cache, pinned snapshot download and case listing."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import image_suite as suite
from benchmarks import run_benchmark as video

LAUNCHER = ("import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); "
            "runpy.run_module('benchmarks.run_benchmark',run_name='__main__')")


def run_controller(*args):
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", LAUNCHER, str(suite.ROOT), *args], check=False,
        capture_output=True, text=True, timeout=60, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )


def test_hub_snapshot_uses_the_standard_hub_cache(monkeypatch, tmp_path):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    assert suite.hub_snapshot("Org/Model-1.1", "abc") == tmp_path / "models--Org--Model-1.1" / "snapshots" / "abc"


def test_ensure_hub_snapshot_downloads_only_a_missing_pinned_cache_snapshot(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    snapshot = tmp_path / "models--Org--Model" / "snapshots" / "rev1"
    suite.ensure_hub_snapshot(snapshot, "model_index.json", allow_patterns=["transformer/*"])
    assert calls == [{"repo_id": "Org/Model", "revision": "rev1", "cache_dir": tmp_path,
                      "allow_patterns": ["transformer/*"]}]

    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}")
    suite.ensure_hub_snapshot(snapshot, "model_index.json")
    suite.ensure_hub_snapshot(tmp_path / "explicit-local-folder", "model_index.json")
    assert len(calls) == 1


@pytest.mark.parametrize(("platform", "recipe", "family", "expected"), [
    ("windows", "ltx23-image", "runtime",
     ["official-leaf", "one-shot-fast", "low-ram-safe", "low-ram-safe-pre-vae-purge", "nvme-baseline", "vmm-fast"]),
    ("linux", "qwen-image21", "runtime", ["official-leaf", "one-shot-fast", "low-ram-safe", "vmm-fast"]),
    ("windows", "qwen-image21", "components", ["vmm-fast", "nvme-baseline"]),
    ("windows", "ltx23-image", "precision", ["sdnq-int4-leaf", "sdnq-int4-ddo"]),
    ("windows", "ltx23-video-8s", "precision", ["sdnq-int4-leaf", "sdnq-int4-ddo", "fp8-weight-only"]),
    ("wsl", "ltx23-video-8s", "runtime", ["official-leaf", "wsl-compat", "vmm-fast"]),
    ("linux", "ltx23-video-8s", "attention", ["cudnn-sdpa", "flash-attn2", "sol-attn"]),
])
def test_list_cases_prints_names_and_summaries(platform, recipe, family, expected):
    result = run_controller("--platform", platform, "--recipe", recipe, "--family", family, "--list-cases")
    assert result.returncode == 0, result.stderr
    rows = [line.split(None, 1) for line in result.stdout.splitlines()]
    assert [row[0] for row in rows] == expected
    assert all(len(row) == 2 for row in rows)


def test_case_summaries_identify_leaf_reference_and_presets(monkeypatch, tmp_path):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    args = argparse.Namespace(recipe="ltx23-image", family="runtime", case=None, platform="windows",
                              checkpoint=None, custom_blocks=None, nvme_resident_gib=1.5)
    cases = suite.build_plan(args)["cases"]
    assert suite.describe_case(cases["official-leaf"]).startswith("Diffusers Leaf reference")
    assert suite.describe_case(cases["vmm-fast"]).startswith("vmm / auto, transformer vmm_fast_runtime")
    assert "purge before VAE" in suite.describe_case(cases["low-ram-safe-pre-vae-purge"])
    assert video.video_case_description("one-shot-fast", "runtime") == "linear / one_shot_fast, transformer linear_runtime"
    assert Path(suite.build_plan(args)["checkpoint"]).is_relative_to(tmp_path)
