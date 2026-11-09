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


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    """Keep optional TorchAO/Triton import probes inside the CPU test contract."""
    import torch

    def refuse_initialization(*args, **kwargs):
        raise AssertionError("Parser/reporting CPU tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", refuse_initialization)


@pytest.mark.parametrize("backend_name,preset", [
    ("vmm", "auto"), ("vmm", "nvme_baseline"), ("linear", "auto"),
])
def test_backend_reporting_preserves_base_settings(backend_name, preset):
    from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
    from scripts.helpers.runner_controls import backend_reporting_fields

    backend = resolve_dynamic_offload_backend(backend_name, preset, running_on_wsl=False)
    settings = SimpleNamespace(requested_preset="one_shot_fast", effective_preset="one_shot_fast")
    before = vars(settings).copy()
    fields = backend_reporting_fields(backend, settings)
    assert fields["ddo_backend"] == backend_name
    assert fields["dynamic_offload_requested_preset"] == preset
    assert fields["dynamic_offload_effective_preset"] == backend.effective_preset
    assert fields["dynamic_offload_base_effective_preset"] == "one_shot_fast"
    assert vars(settings) == before


def parser_body(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = dict(argparse=argparse, math=math, os=os, sys=sys, Path=Path, ROOT=ROOT,
                     parse_common_arguments=parse_common_arguments, __doc__="CPU parser contract",
                     add_lora_arguments=add_lora_arguments, validate_lora_arguments=validate_lora_arguments,
                     parse_optional_float=lambda value: None if value in (None, "") else float(value))
    for node in tree.body:
        # Parser helpers shared by the runners (CPU-only modules) are imported as
        # the runner imports them; other top-level imports and code do not run.
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scripts.helpers"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
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


def test_qwen_vmm_single_frame_default_and_explicit_original_decoder():
    parse = parser_body(ROOT / "scripts/qwen_image_21/run_dynamic_modular_vmm.py")
    assert parse([]).vae_no_temporal_cache
    assert parse(["--vae-no-temporal-cache"]).vae_no_temporal_cache
    assert not parse(["--no-vae-no-temporal-cache"]).vae_no_temporal_cache
    assert parse(["--vae-tiling"]).vae_tiling
    with pytest.raises(SystemExit):
        parse(["--no-vae-no-temporal-cache", "--vae-alias-residuals"])


def test_linear_video_recorded_computation_defaults_and_opt_outs():
    parse = parser_body(ROOT / "scripts/ltx2_video/run_dynamic_modular_distilled.py")
    args = parse(["--audio", "conditioning.mp3", "--use-audio-condition"])
    assert (args.feed_forward_chunk_size, args.normalization_chunk_size) == (1024, 1024)
    assert args.phase_modulation and args.compact_timesteps and args.inplace_residuals
    assert args.reuse_rotary_embeddings
    assert args.audio == "conditioning.mp3" and args.use_audio_condition
    args = parse(["--feed-forward-chunk-size", "0", "--normalization-chunk-size", "0",
                  "--no-phase-modulation", "--no-compact-timesteps", "--no-inplace-residuals"])
    assert (args.feed_forward_chunk_size, args.normalization_chunk_size) == (0, 0)
    assert not (args.phase_modulation or args.compact_timesteps or args.inplace_residuals)
    with pytest.raises(SystemExit):
        parse(["--normalization-chunk-size", "-1"])


def test_shared_video_timestep_admission_preserves_nonuniform_masks():
    import torch
    from scripts.helpers.ltx2.timestep_layout import compact_timestep_repeat

    mask = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.]).reshape(1, 8, 1)
    layout = dict(latent_frames=2, latent_height=2, latent_width=2)
    assert compact_timestep_repeat(mask, **layout) == 4
    mask[0, 1, 0] = 1
    assert compact_timestep_repeat(mask, **layout) == 1
    assert compact_timestep_repeat(mask, **layout, patch_size_t=2) == 1


@pytest.mark.parametrize("preset,ceiling,expected,policy,pin", [
    ("auto", None, "one_shot_fast", "balanced", True),
    ("one_shot_fast", "3", "one_shot_fast", "balanced", True),
    ("low_ram_safe", None, "low_ram_safe", "off", False),
])
def test_linear_video_preset_and_ceiling_are_distinct(preset, ceiling, expected, policy, pin, capsys):
    from diffusers_dynamic_offloader import DynamicOffloadSettings
    from scripts.ltx2_video.run_dynamic_modular_distilled import print_linear_memory_policy

    parse = parser_body(ROOT / "scripts/ltx2_video/run_dynamic_modular_distilled.py")
    args = parse([] if preset == "auto" else ["--ddo-preset", preset])
    env = {} if ceiling is None else {"DDO_MAX_RESIDENT_MODULE_BUDGET_GB": ceiling}
    settings = DynamicOffloadSettings.from_env(
        execution_device="cpu", default_preset=args.ddo_preset,
        environ=runner_settings_environ(args, env), running_on_wsl=False)
    assert settings.effective_preset == expected
    assert settings.config.auto_budget_policy == policy
    assert settings.config.pin_cpu_memory is pin
    if ceiling is not None:
        assert settings.config.max_resident_module_budget_gb == 3
        assert settings.config.resident_module_budget_gb == 0
    if expected == "low_ram_safe":
        assert settings.config.resident_module_budget_gb == 3
    print_linear_memory_policy(settings)
    output = capsys.readouterr().out
    assert f"effective={expected}" in output
    assert ("ceiling is not a target" if policy == "balanced" else "fixed resident-module budget=3.00 GiB") in output


@pytest.mark.parametrize("path", RUNNERS, ids=lambda path: str(path.relative_to(ROOT)))
def test_every_runner_accepts_the_same_explicit_lifecycle_switches(path, monkeypatch):
    function = parser_body(path)
    monkeypatch.setattr("sys.argv", [str(path), "--purge-before-transformer", "--no-purge-after-run"])
    args = function()
    assert args.purge_before_transformer is True
    assert args.purge_after_run is False
    assert args.purge_before_run is None
    assert getattr(args, "purge_after_text_encoder", None) is None


@pytest.mark.parametrize("path", RUNNERS, ids=lambda path: str(path.relative_to(ROOT)))
@pytest.mark.parametrize("mode", ("rss", "full"))
def test_every_runner_accepts_and_forwards_memory_monitor(path, mode, monkeypatch):
    function = parser_body(path)
    monkeypatch.setattr("sys.argv", [str(path), "--memory-monitor", mode])
    assert function().memory_monitor == mode
    tree = ast.parse(path.read_text(encoding="utf-8"))
    trackers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "RunTracker"]
    for tracker in trackers:
        value = next(keyword.value for keyword in tracker.keywords if keyword.arg == "memory_monitor")
        assert isinstance(value, ast.Attribute) and value.attr == "memory_monitor"


def test_shared_monitor_preserves_legacy_default_and_rejects_unknown_mode():
    assert parse_common_arguments(argparse.ArgumentParser(), []).memory_monitor == "full"
    with pytest.raises(SystemExit):
        parse_common_arguments(argparse.ArgumentParser(), ["--memory-monitor", "unknown"])


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


PUBLIC_IMAGE_VMM = [ROOT / "scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py",
                    ROOT / "scripts/qwen_image_21/run_dynamic_modular_vmm.py"]
PUBLIC_LTX_LINEAR = [ROOT / "scripts/ltx2_image" / name for name in (
    "run_dynamic_modular_distilled.py",
    "run_dynamic_modular_sdnq_distilled.py")]
PUBLIC_QWEN_OTHER = [ROOT / "scripts/qwen_image_21" / name for name in (
    "run_dynamic_minimal.py", "run_dynamic_modular.py",
    "run_dynamic_modular_sdnq.py", "run_diffusers_leaf.py")]
PUBLIC_VIDEO = [ROOT / "scripts/ltx2_video" / name for name in (
    "run_dynamic_minimal.py", "run_dynamic_modular_distilled.py",
    "run_dynamic_modular_sdnq_distilled.py", "run_dynamic_modular_vmm_distilled.py")]

LINEAR_PROFILE_RUNNERS = [PUBLIC_LTX_LINEAR[0], PUBLIC_VIDEO[1], PUBLIC_QWEN_OTHER[1]]


@pytest.mark.parametrize("path", LINEAR_PROFILE_RUNNERS)
@pytest.mark.parametrize("build", [False, True])
def test_modular_linear_profile_selection_and_creation(path, build, monkeypatch):
    arguments = ["--ddo-profile", "workload-specific-name"]
    if build:
        arguments.append("--build-ddo-profile")
    monkeypatch.setattr(sys, "argv", [str(path), *arguments])
    args = parser_body(path)()
    assert args.ddo_profile == "workload-specific-name"
    assert args.build_ddo_profile is build


@pytest.mark.parametrize("path", LINEAR_PROFILE_RUNNERS)
@pytest.mark.parametrize("arguments", [
    ["--build-ddo-profile"], ["--ddo-profile", " "],
    ["--ddo-profile", "name", "--ddo-preset", "off"],
    ["--ddo-profile", "name", "--ddo-preset", "diffusers_leaf_offload_compat"],
])
def test_linear_profile_invalid_requests_fail_before_model_loading(path, arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path), *arguments])
    with pytest.raises(SystemExit):
        parser_body(path)()


# The native VMM entries (image and video) expose --ddo-profile for the VMM
# residency profile (tests/test_image_vmm_profile_cpu.py, test_vmm_profile_cpu.py).
@pytest.mark.parametrize("path", [PUBLIC_LTX_LINEAR[1], *PUBLIC_VIDEO[::2]])
def test_profile_restoration_does_not_expose_other_routes(path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path), "--ddo-profile", "name"])
    with pytest.raises(SystemExit):
        parser_body(path)()


@pytest.mark.parametrize("path", LINEAR_PROFILE_RUNNERS)
def test_linear_profile_wiring_and_decode_boundary(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    attaches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "enable_offload"
                and any(item.arg == "profile_name" for item in node.keywords)]
    assert len(attaches) == 1
    keywords = {item.arg: item.value for item in attaches[0].keywords}
    args = SimpleNamespace(ddo_profile="same-profile", build_ddo_profile=True)
    namespace = dict(args=args, name="transformer")
    for keyword, expected in (("profile_name", "same-profile"), ("build_profile", True)):
        assert eval(compile(ast.Expression(keywords[keyword]), str(path), "eval"), namespace) == expected
    sessions = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                and any(isinstance(item.context_expr, ast.Call)
                        and isinstance(item.context_expr.func, ast.Attribute)
                        and item.context_expr.func.attr == "profile_run" for item in node.items)]
    assert len(sessions) == 1
    assert any(isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name) and node.func.id in ("denoise_pipe", "denoise", "run_stage")
        or isinstance(node.func, ast.Attribute) and node.func.attr == "cpu"
    ) for node in ast.walk(sessions[0]))
    stops = [node for node in ast.walk(tree) if isinstance(node, ast.If)
             and isinstance(node.test, ast.Attribute) and node.test.attr == "build_ddo_profile"
             and any(isinstance(child, ast.Return) for child in node.body)]
    assert len(stops) == 1
    assert stops[0].lineno > sessions[0].lineno


@pytest.mark.parametrize("path", PUBLIC_IMAGE_VMM + PUBLIC_VIDEO[-1:])
def test_public_native_vocabulary_and_total_capacity(path, monkeypatch, capsys):
    function = parser_body(path)
    monkeypatch.setattr(sys, "argv", [str(path), "--ddo-vmm-resident-weight-budget-gb", "4",
        "--ddo-vmm-total-staging-gb", "1.5", "--ddo-vmm-source-cache-gb", "16",
        "--ddo-vmm-source-registered-gb", "0", "--show-profile", "--no-purge-before-transformer"])
    args = function()
    assert args.purge_before_transformer is False
    assert args.ddo_vmm_source_cache_gb == 16
    if "ltx2_image" in str(path):
        assert args.vmm_resident_weight_budget_gb == 4
        assert args.vmm_staging_buffer_gb * 3 == 1.5
        assert args.show_ddo_profile is True
        assert args.ddo_vmm_diagnostic_adapter is False
        assert args.ddo_public_native_runtime is None
    else:
        from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
        config = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).build_config_from_args(args)
        assert config.vmm_resident_weight_budget_gb == 4
        assert config.vmm_staging_buffer_gb == 1.5
        assert args.show_profile is True
    monkeypatch.setattr(sys, "argv", [str(path), "--help"])
    with pytest.raises(SystemExit) as exit_info: function()
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    import re
    exposed = set(re.findall(r"--ddo-[a-z-]+", help_text))
    assert exposed == {"--ddo-backend", "--ddo-preset", "--ddo-vmm-resident-weight-budget-gb",
                       "--ddo-vmm-total-staging-gb", "--ddo-vmm-source-cache-gb", "--ddo-vmm-source-registered-gb",
                       "--ddo-vmm-execution-backing-reclaim", "--ddo-profile"}


@pytest.mark.parametrize("path", PUBLIC_IMAGE_VMM)
@pytest.mark.parametrize("option", ["--preset", "--vmm-staging-buffer-gb", "--ddo-vmm-staging-buffer-gb",
                                  "--ddo-public-native-runtime", "--show-ddo-profile", "--ddo-vmm-diagnostic-adapter"])
def test_public_native_rejects_previous_and_diagnostic_names(path, option, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path), option])
    with pytest.raises(SystemExit): parser_body(path)()


@pytest.mark.parametrize("path", PUBLIC_IMAGE_VMM + PUBLIC_LTX_LINEAR + PUBLIC_QWEN_OTHER + PUBLIC_VIDEO)
def test_public_native_help_options_have_documentation(path, monkeypatch, capsys):
    """Keep the documented inventory aligned with the actual exposed parser."""
    import re

    page = ("ltx2-image-parameters.md" if "ltx2_image" in str(path) else
            "ltx2-video-parameters.md" if "ltx2_video" in str(path) else "qwen-image-21-parameters.md")
    documented = "\n".join((ROOT / "docs/user" / name).read_text(encoding="utf-8")
                           for name in (page, "runner-controls.md"))
    exposed = set()
    original_parse = argparse.ArgumentParser.parse_args

    def inspect_options(parser, *args, **kwargs):
        exposed.update(option for action in parser._actions for option in action.option_strings
                       if option.startswith("--"))
        return original_parse(parser, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", inspect_options)
    monkeypatch.setattr(sys, "argv", [str(path), "--help"])
    with pytest.raises(SystemExit) as result:
        parser_body(path)()
    assert result.value.code == 0
    capsys.readouterr()
    positive = {"--" + option[5:] if option.startswith("--no-") else option for option in exposed}
    missing = positive - set(re.findall(r"--[a-z][a-z0-9-]*", documented))
    assert not missing, sorted(missing)


@pytest.mark.parametrize("path", PUBLIC_LTX_LINEAR)
def test_ltx_linear_canonical_controls_preserve_runtime_destinations(path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path), "--ddo-backend", "linear", "--ddo-preset",
        "low_ram_safe", "--custom-blocks", "owner/code", "--custom-blocks-revision", "fixed",
        "--lora", "adapter.safetensors", "--lora-scale", "0.5"])
    args = parser_body(path)()
    assert args.preset == args.ddo_preset == "low_ram_safe"
    assert args.explicit_ddo_preset
    assert runner_settings_environ(args, {})["DDO_PRESET"] == "low_ram_safe"
    assert args.custom_blocks == "owner/code"
    assert args.custom_blocks_revision == "fixed"
    assert args.lora == ["adapter.safetensors"]
    assert args.lora_scale == [0.5]
    assert args.build_ddo_profile is False


@pytest.mark.parametrize("path", PUBLIC_LTX_LINEAR)
@pytest.mark.parametrize("arguments", [
    ["--preset", "auto"], ["--pipeline-path", "model"], ["--show-ddo-profile"],
    ["--build-ddo-profile"], ["--ddo-profile-steps", "2"],
    ["--fake-prompt"], ["--generation-repeats", "2"], ["--transformer-prepare-repeats", "2"],
    ["--resident-module-budget-gb", "1"], ["--ddo-backend", "vmm"],
    ["--ddo-vmm-diagnostic-backing"], ["--custom-block", "owner/code"],
])
def test_ltx_linear_rejects_aliases_diagnostics_and_unimplemented_routes(path, arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path), *arguments])
    with pytest.raises(SystemExit):
        parser_body(path)()


@pytest.mark.parametrize("option", ["--soft-lora", "--crisp-lora", "--text-encoder-dynamic-offload"])
def test_ltx_sdnq_rejects_duplicate_lora_and_unimplemented_encoder_control(option, monkeypatch):
    path = PUBLIC_LTX_LINEAR[-1]
    monkeypatch.setattr(sys, "argv", [str(path), option])
    with pytest.raises(SystemExit):
        parser_body(path)()


@pytest.mark.parametrize("path", PUBLIC_QWEN_OTHER)
@pytest.mark.parametrize("arguments", [
    ["--pipeline-path", "model"], ["--transformer-path", "model"],
    ["--resident-module-budget-gb", "1"], ["--pin-weight-budget-gb", "1"],
    ["--ddo-gpu-timing"], ["--ddo-vmm-diagnostic-backing"], ["--wid", "512"],
])
def test_other_qwen_entries_reject_old_and_diagnostic_controls(path, arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path), *arguments])
    with pytest.raises(SystemExit):
        parser_body(path)()


@pytest.mark.parametrize("path", PUBLIC_QWEN_OTHER[1:-1])
def test_other_qwen_runtime_selection_keeps_original_destination(path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path), "--ddo-backend", "linear",
                                     "--ddo-preset", "low_ram_safe"])
    args = parser_body(path)()
    assert args.ddo_preset == "low_ram_safe"
    assert args.explicit_ddo_preset
    assert runner_settings_environ(args, {})["DDO_PRESET"] == "low_ram_safe"


def documented_image_commands():
    """Parse published recipes against real parsers without importing runners."""
    import re
    import shlex

    commands = []
    for page in (ROOT / "docs").rglob("*.md"):
        for block in re.findall(r"```(?:powershell|bash|sh)\n(.*?)```", page.read_text(encoding="utf-8"), re.S):
            # PowerShell argument arrays ($name = @("--a", "b", ...)) splatted as @name.
            arrays = {name: shlex.split(body.replace(",", " "))
                      for name, body in re.findall(r"\$(\w+)\s*=\s*@\((.*?)\)", block, re.S)}
            block = re.sub(r"[`\\]\r?\n", " ", block)
            for line in block.splitlines():
                if not re.search(r"python\s+-B\s+scripts/(?:ltx2_image|ltx2_video|qwen_image_21)/", line):
                    continue
                tokens = shlex.split(line)
                arguments = [part for token in tokens[3:]
                             for part in (arrays[token[1:]] if token.startswith("@") and token[1:] in arrays
                                          else [token])]
                commands.append((page.relative_to(ROOT).as_posix(), tokens[2], arguments))
    return commands


@pytest.mark.parametrize("page,runner,arguments", documented_image_commands())
def test_documented_image_recipe_matches_public_parser(page, runner, arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", [runner, *arguments])
    if "--help" in arguments:
        with pytest.raises(SystemExit) as result:
            parser_body(ROOT / runner)()
        assert result.value.code == 0, page
    else:
        parser_body(ROOT / runner)()


@pytest.mark.parametrize("level,calls", [(0, 0), (1, 0), (2, 1)])
def test_report_detail_is_shared_and_does_not_change_snapshot_objects(level, calls):
    tracker = SimpleNamespace(record_ddo_metrics=Mock())
    report = object()
    record_runner_report(SimpleNamespace(metrics_level=level), tracker, report, "transformer")
    assert tracker.record_ddo_metrics.call_count == calls
    if calls:
        tracker.record_ddo_metrics.assert_called_once_with(report, "transformer")


@pytest.mark.parametrize("path", PUBLIC_VIDEO)
@pytest.mark.parametrize("arguments", [
    ["--pipeline-path", "model"], ["--transformer-path", "model"],
    ["--build-ddo-profile"],
    ["--trace-transformer-blocks"], ["--ddo-vmm-library-path", "library"],
    ["--ddo-vmm-staging-buffer-gb", "1.5"], ["--ddo-vmm-page-mb", "32"],
    ["--ddo-vmm-native-transport"], ["--ddo-vmm-diagnostic-backing"],
    ["--fused-rms-modulation"], ["--fused-rotary-pair"], ["--wid", "1280"],
])
def test_public_video_rejects_aliases_and_diagnostics(path, arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path), *arguments])
    with pytest.raises(SystemExit):
        parser_body(path)()


@pytest.mark.parametrize("path", PUBLIC_VIDEO)
def test_public_video_preserves_inputs_decode_and_reporting_defaults(path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path)])
    args = parser_body(path)()
    assert (args.width, args.height, args.seconds, args.frame_rate, args.steps, args.seed) == (1280, 704, 5, 24, 8, 43)
    assert args.audio is None and not args.use_audio_condition
    assert args.decode_device == "cpu" and not args.stream_video_decode
    assert args.memory_monitor == "rss" and args.show_metrics and args.save_metrics
    monkeypatch.setattr(sys, "argv", [str(path), "--audio", "condition.mp3", "--use-audio-condition",
        "--decode-device", "cuda:0", "--stream-video-decode", "--custom-blocks", "owner/blocks"])
    args = parser_body(path)()
    assert args.audio == "condition.mp3" and args.use_audio_condition
    assert args.stream_video_decode and args.custom_blocks == "owner/blocks"


@pytest.mark.parametrize("preset,expected", [("auto", 1.5), ("nvme_baseline", .75)])
def test_video_total_staging_uses_backend_conversion(preset, expected, monkeypatch):
    from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
    path = PUBLIC_VIDEO[-1]
    monkeypatch.setattr(sys, "argv", [str(path), "--ddo-preset", preset, "--ddo-vmm-total-staging-gb", "1.5"])
    args = parser_body(path)()
    config = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).build_config_from_args(args)
    assert config.vmm_staging_buffer_gb == expected
    assert not args.fused_rms_modulation and not args.fused_rotary_pair


def test_video_native_capacity_defaults_and_diagnostic_separation(monkeypatch):
    path = PUBLIC_VIDEO[-1]
    monkeypatch.setattr(sys, "argv", [str(path)])
    args = parser_body(path)()
    # Omitted: DDO resolves it from a VMM residency profile or the structural floor.
    assert args.ddo_vmm_resident_weight_budget_gb is None
    assert args.ddo_vmm_total_staging_gb is None
    assert args.ddo_vmm_staging_buffer_gb == .625
    assert args.ddo_vmm_source_cache_gb == args.ddo_vmm_source_registered_gb == 0
    assert not args.show_profile
    diagnostic = ROOT / "scripts/ltx2_video/run_dynamic_modular_diagnostics_distilled.py"
    monkeypatch.setattr(sys, "argv", [str(diagnostic), "--ddo-backend", "vmm", "--fused-rms-modulation",
        "--fused-rotary-pair", "--ddo-vmm-library-path", "isolated.dll"])
    args = parser_body(diagnostic)()
    assert args.fused_rms_modulation and args.fused_rotary_pair
    assert args.ddo_vmm_library_path == "isolated.dll"


@pytest.mark.parametrize("path", PUBLIC_IMAGE_VMM + PUBLIC_VIDEO[-1:])
@pytest.mark.parametrize("environment", ["0", "1"])
@pytest.mark.parametrize("switch,expected", [(None, False), ("--ddo-vmm-execution-backing-reclaim", True),
    ("--no-ddo-vmm-execution-backing-reclaim", False)])
def test_reclaim_is_cli_opt_in_independent_of_environment(path, environment, switch, expected, monkeypatch):
    from diffusers_dynamic_offloader import DynamicOffloadConfig
    from scripts.helpers.runner_controls import apply_execution_backing_reclaim
    monkeypatch.setenv("DDO_VMM_EXECUTION_BACKING_RECLAIM", environment)
    monkeypatch.setattr(sys, "argv", [str(path), *([] if switch is None else [switch])])
    args = parser_body(path)()
    assert args.ddo_vmm_execution_backing_reclaim is expected
    original = DynamicOffloadConfig(execution_mode="vmm_fast_runtime", vmm_execution_backing_reclaim=environment == "1")
    config = apply_execution_backing_reclaim(original, args)
    assert config.vmm_execution_backing_reclaim is expected
    assert original.vmm_execution_backing_reclaim == (environment == "1")
    assert os.environ["DDO_VMM_EXECUTION_BACKING_RECLAIM"] == environment
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "apply_execution_backing_reclaim" for node in ast.walk(tree))


@pytest.mark.parametrize("path", PUBLIC_IMAGE_VMM + PUBLIC_VIDEO[-1:])
def test_reclaim_enabled_rejects_nvme_route_before_execution(path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(path), "--ddo-preset", "nvme_baseline", "--ddo-vmm-execution-backing-reclaim"])
    with pytest.raises(SystemExit):
        parser_body(path)()


@pytest.mark.parametrize("model", ["ltx2_image", "qwen_image_21", "ltx2_video"])
def test_minimal_native_configuration_and_reclaim(model, monkeypatch):
    from scripts.helpers.runner_controls import minimal_vmm_config
    path = ROOT / "scripts" / model / "run_dynamic_minimal.py"
    monkeypatch.setenv("DDO_VMM_EXECUTION_BACKING_RECLAIM", "1")
    monkeypatch.setattr(sys, "argv", [str(path)])
    args = parser_body(path)()
    config = minimal_vmm_config(args, "cuda:0")
    assert args.ddo_backend == "vmm" and args.ddo_preset == "auto"
    assert config.execution_mode == "vmm_fast_runtime" and config.vmm_native_transport
    assert not config.vmm_execution_backing_reclaim
    monkeypatch.setattr(sys, "argv", [str(path), "--ddo-vmm-resident-weight-budget-gb", "1.5",
        "--ddo-vmm-total-staging-gb", "1.5", "--ddo-vmm-source-cache-gb", "16",
        "--ddo-vmm-source-registered-gb", "0", "--ddo-vmm-execution-backing-reclaim"])
    config = minimal_vmm_config(parser_body(path)(), "cuda:0")
    assert config.vmm_resident_weight_budget_gb == config.vmm_staging_buffer_gb == 1.5
    assert config.vmm_source_cache_gb == 16 and config.vmm_source_registered_gb == 0
    assert config.vmm_execution_backing_reclaim
    for selection in (["--ddo-backend", "linear"], ["--ddo-preset", "nvme_baseline"]):
        monkeypatch.setattr(sys, "argv", [str(path), *selection])
        with pytest.raises(SystemExit):
            parser_body(path)()


@pytest.mark.parametrize("model", ["ltx2_image", "qwen_image_21", "ltx2_video"])
def test_minimal_bootstraps_before_loading_or_parsing(model, monkeypatch):
    """Run the real entry prefix with a bootstrap sentinel, before any model import."""
    path = ROOT / "scripts" / model / "run_dynamic_minimal.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class BootstrapReached(Exception):
        pass
    def bootstrap():
        raise BootstrapReached
    monkeypatch.setattr("ddo_vmm_bootstrap.ensure_vmm_fast_process", bootstrap)
    monkeypatch.setattr(sys, "argv", [str(path)])
    namespace = {"__name__": "__main__", "__file__": str(path), "sys": sys}
    if model == "ltx2_image":
        boundary = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.If))
        body = tree.body[:boundary + 1]
    else:
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        namespace["parse_args"] = lambda: pytest.fail("Parser ran before native bootstrap")
        body = [main, ast.Expr(value=ast.Call(func=ast.Name(id="main", ctx=ast.Load()), args=[], keywords=[]))]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    with pytest.raises(BootstrapReached):
        exec(compile(module, str(path), "exec"), namespace)


@pytest.mark.parametrize("show", [True, False])
def test_ltx_minimal_progress_does_not_query_unsupported_allocator_stats(show, capsys):
    path = ROOT / "scripts/ltx2_image/run_dynamic_minimal.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    callback = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "denoise_callback")
    unsupported = Mock(side_effect=RuntimeError("CUDAPluggableAllocator does not yet support getDeviceStats"))
    namespace = {"time": SimpleNamespace(perf_counter=lambda: 12.0), "STEPS": 8,
                 "SHOW_DENOISE_STEPS": show, "DEVICE": "cuda:0",
                 "torch": SimpleNamespace(cuda=SimpleNamespace(memory_allocated=unsupported, memory_reserved=unsupported))}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(path), "exec"), namespace)
    function = namespace["denoise_callback"]
    function.start_time = 10.0
    function.last_time = 10.0
    payload = {"latents": object()}
    assert function(None, 0, None, payload) is payload
    unsupported.assert_not_called()
    output = capsys.readouterr().out
    if show:
        assert "step 1/8 elapsed=2.0000s" in output
        assert "torch_alloc=n/a torch_reserved=n/a" in output
    else:
        assert output == ""
