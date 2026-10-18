"""Exercise real parser bodies without importing model modules or bootstrapping CUDA."""

import argparse
import ast
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.helpers.runner_controls import parse_common_arguments, purge_for_runner, runner_settings_environ, record_runner_report
from scripts.helpers.ltx2.lora import add_lora_arguments, validate_lora_arguments

ROOT = Path(__file__).parents[1]
RUNNERS = sorted(path for model in ("ltx2_image", "ltx2_video", "qwen_image_21")
                 for path in (ROOT / "scripts" / model).glob("run*.py"))


def parser_body(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = dict(argparse=argparse, math=math, os=os, sys=sys, Path=Path, ROOT=ROOT,
                     parse_common_arguments=parse_common_arguments, __doc__="CPU parser contract",
                     add_lora_arguments=add_lora_arguments, validate_lora_arguments=validate_lora_arguments,
                     parse_optional_float=lambda value: None if value in (None, "") else float(value))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id.isupper():
            namespace.setdefault(node.id, 0)
    namespace["ROOT"] = ROOT
    namespace["DDO_VMM_ALLOCATOR_ROOT"] = ROOT.parent / "diffusers-dynamic-offloader/ddo-vmm-allocator"
    namespace.setdefault("MODEL_PATH", "model")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parse_args"]


@pytest.mark.parametrize("path", RUNNERS, ids=lambda path: str(path.relative_to(ROOT)))
def test_every_runner_accepts_the_same_explicit_lifecycle_switches(path, monkeypatch):
    function = parser_body(path)
    monkeypatch.setattr("sys.argv", [str(path), "--purge-before-transformer", "--no-purge-after-run"])
    args = function()
    assert args.purge_before_transformer is True
    assert args.purge_after_run is False
    assert args.purge_before_run is None
    assert getattr(args, "purge_after_text_encoder", None) is None


def test_aliases_preserve_destinations_defaults_and_units():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--show-ddo-profile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vmm-resident-weight-budget-gb", type=float, default=4)
    parser.add_argument("--vmm-staging-buffer-gb", type=float, default=.625)
    parser.add_argument("--ddo-backend", default="vmm")
    parser.add_argument("--ddo-preset", default="auto")
    args = parse_common_arguments(parser, ["--show-profile", "--ddo-vmm-resident-weight-budget-gb", "4",
                                          "--ddo-vmm-total-staging-gb", "1.5"])
    assert args.show_ddo_profile is True
    assert args.vmm_resident_weight_budget_gb == 4
    assert args.vmm_staging_buffer_gb == .5
    assert args.preset == "auto"  # Existing auxiliary selector is not the transformer selector.


def test_existing_linear_preset_alias_and_negative_switches():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parse_common_arguments(parser, ["--ddo-preset", "low_ram_safe", "--no-local-files-only"])
    assert args.preset == "low_ram_safe"
    assert args.ddo_backend == "linear"
    assert args.local_files_only is False


@pytest.mark.parametrize("explicit", [None, False, True])
def test_purge_passes_public_override_without_environment_replacement(explicit, monkeypatch):
    native = Mock(return_value={"purged": False})
    monkeypatch.setattr("diffusers_dynamic_offloader.maybe_purge_windows_standby_cache", native)
    args = SimpleNamespace(purge_before_transformer=explicit)
    settings = object()
    purge_for_runner(args, settings, "before_transformer", record_event=None)
    assert native.call_args.args == (settings, "before_transformer")
    assert native.call_args.kwargs["enabled"] is explicit
    assert "environ" not in native.call_args.kwargs


@pytest.mark.parametrize("backend,preset,expected", [("linear", "low_ram_safe", "low_ram_safe"),
                                                     ("vmm", "nvme_baseline", "one_shot_fast")])
def test_settings_precedence_never_mutates_process_environment(backend, preset, expected):
    source = {"DDO_PRESET": "off", "OTHER": "kept"}
    args = SimpleNamespace(explicit_ddo_preset=True, ddo_backend=backend, ddo_preset=preset)
    assert runner_settings_environ(args, source) == {"DDO_PRESET": expected, "OTHER": "kept"}
    assert source["DDO_PRESET"] == "off"
    args.explicit_ddo_preset = False
    assert runner_settings_environ(args, source) == source


def test_pipeline_and_transformer_sources_have_distinct_canonical_names():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled-model", default="pipeline")
    parser.add_argument("--model-path", default=None)
    args = parse_common_arguments(parser, ["--pipeline-path", "base", "--transformer-path", "quantized"])
    assert args.distilled_model == "base"
    assert args.model_path == "quantized"


def test_unstaged_control_rejects_an_unavailable_encoder_boundary(monkeypatch):
    function = parser_body(ROOT / "scripts/qwen_image_21/run_diffusers_leaf.py")
    monkeypatch.setattr("sys.argv", ["runner", "--purge-after-text-encoder"])
    with pytest.raises(SystemExit):
        function()


def test_canonical_image_linear_selector_reaches_legacy_settings_destination():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="one_shot_fast")
    parser.add_argument("--ddo-preset", default="auto")
    parser.add_argument("--ddo-backend", default="vmm")
    parser.add_argument("--vmm-staging-buffer-gb", type=float, default=.625)
    args = parse_common_arguments(parser, ["--ddo-backend", "linear", "--ddo-preset", "low_ram_safe"])
    assert args.preset == "low_ram_safe"


def test_cuda_remains_uninitialized():
    import torch
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("level,calls", [(0, 0), (1, 0), (2, 1)])
def test_report_detail_is_shared_and_does_not_change_snapshot_objects(level, calls):
    tracker = SimpleNamespace(record_ddo_metrics=Mock())
    report = object()
    record_runner_report(SimpleNamespace(metrics_level=level), tracker, report, "transformer")
    assert tracker.record_ddo_metrics.call_count == calls
    if calls:
        tracker.record_ddo_metrics.assert_called_once_with(report, "transformer")
