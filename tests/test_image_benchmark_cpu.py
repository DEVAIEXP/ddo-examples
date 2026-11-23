"""Exercise image plans and session recovery without launching a runner."""

import argparse
import importlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest

from benchmarks import image_suite as suite
from benchmarks import run_benchmark as video_benchmark


def args(recipe, family="runtime", case=None):
    return argparse.Namespace(recipe=recipe, family=family, case=case,
                              checkpoint=None, custom_blocks=None, nvme_resident_gib=1.5)


@pytest.mark.parametrize("recipe,platform,family,case,disabled", [
    ("qwen-image21", "wsl", "runtime", "official-leaf", True),
    ("qwen-image21", "wsl", "runtime", "wsl-compat", False),
    ("qwen-image21", "wsl", "runtime", "vmm-fast", False),
    ("qwen-image21", "windows", "runtime", "official-leaf", False),
    ("qwen-image21", "linux", "runtime", "official-leaf", False),
    ("qwen-image21", "wsl", "attention", "cudnn-sdpa", False),
    ("ltx23-image", "wsl", "runtime", "official-leaf", False),
])
def test_qwen_wsl_leaf_allocator_exception_is_recipe_local(recipe, platform, family, case, disabled):
    options = args(recipe, family, case)
    options.platform = platform
    plan = suite.build_plan(options)
    assert "PYTORCH_ALLOC_CONF" not in plan["environment_overrides"]
    for stage in plan["cases"][case]["stages"]:
        value = stage.get("environment", {}).get("PYTORCH_ALLOC_CONF")
        assert value == ("expandable_segments:False" if disabled else None)


def test_linux_video_leaf_preserves_control_recipe(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU plan initialized CUDA"))
    from scripts.ltx2_video.run_dynamic_modular_distilled import parse_args

    monkeypatch.setattr(video_benchmark, "current_platform", lambda: "linux")
    options = argparse.Namespace(
        platform="linux", recipe=video_benchmark.PRESSURE_RECIPE, case="official-leaf", family="runtime",
        vram_gib=8, checkpoint=tmp_path, custom_blocks=tmp_path, fixtures_dir=tmp_path, profile_file=None,
    )
    plan = video_benchmark.video_runtime_plan(options)
    assert list(plan["cases"]) == ["official-leaf"]
    case = plan["cases"]["official-leaf"]
    assert case["profile"] is None and case["preset"] is None
    assert [stage["name"] for stage in case["stages"]] == ["measure"]
    stage = case["stages"][0]
    assert set(stage["environment"].values()) == {"0"}
    parsed = parse_args(stage["command"][4:])
    assert (parsed.seconds, parsed.width, parsed.height, parsed.steps, parsed.seed) == (8, 1280, 704, 8, 43)
    assert parsed.video_decoder == "exact-streaming" and parsed.stream_video_decode
    assert parsed.video_self_attention == "default" and parsed.decode_device == "cuda:0"
    assert parsed.feed_forward_chunk_size == parsed.normalization_chunk_size == 0
    assert not any((parsed.phase_modulation, parsed.compact_timesteps,
                    parsed.inplace_residuals, parsed.reuse_rotary_embeddings))
    assert "--no-stream-offload" not in stage["command"]
    options.case = "nvme-baseline"
    with pytest.raises(ValueError, match="not available"):
        video_benchmark.video_runtime_plan(options)


@pytest.mark.parametrize(("case_name", "preset"), [("one-shot-fast", "one_shot_fast"),
                                                  ("low-ram-safe", "low_ram_safe")])
def test_linux_video_linear_calibrates_before_measurement(monkeypatch, tmp_path, case_name, preset):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU plan initialized CUDA"))
    from scripts.ltx2_video.run_dynamic_modular_distilled import parse_args

    monkeypatch.setattr(video_benchmark, "current_platform", lambda: "linux")
    options = argparse.Namespace(
        platform="linux", recipe=video_benchmark.PRESSURE_RECIPE, case=case_name, family="runtime",
        vram_gib=8, checkpoint=tmp_path, custom_blocks=tmp_path, fixtures_dir=tmp_path, profile_file=None,
    )
    plan = video_benchmark.video_runtime_plan(options)
    case = plan["cases"][case_name]
    assert case["routes"] == {"transformer": "linear_runtime"}
    assert case["preset"] == preset and case["dispatcher"] is None
    assert [stage["name"] for stage in case["stages"]] == ["profile-1", "measure"]
    for stage in case["stages"]:
        parsed = parse_args(stage["command"][4:])
        assert parsed.ddo_profile == f"ltx23-video-1280x704-8s-8steps-{case_name}"
        assert parsed.build_ddo_profile is (stage["name"] == "profile-1")
        assert (parsed.ddo_backend, parsed.ddo_preset) == ("linear", preset)
        assert (parsed.seconds, parsed.width, parsed.height, parsed.steps, parsed.seed) == (8, 1280, 704, 8, 43)
        assert parsed.video_decoder == "exact-streaming" and parsed.video_self_attention == "default"
        assert not any((parsed.phase_modulation, parsed.compact_timesteps,
                        parsed.inplace_residuals, parsed.reuse_rotary_embeddings))
        assert parsed.feed_forward_chunk_size == parsed.normalization_chunk_size == 0
        assert parsed.release_pytorch_host_cache_before_vae is (video_benchmark.sys.platform == "linux"
                                                               and "microsoft" not in platform.release().lower())
        assert stage["environment"]["DDO_RESIDENT_MODULE_BUDGET_GB"] == "0"
        assert not any(flag.startswith("--purge-") for flag in stage["command"])
        assert all(value == "0" for key, value in stage["environment"].items() if "PURGE" in key)


def test_linux_video_vmm_fast_profiles_native_routes_without_wsl_limits(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU parser initialized CUDA"))
    from scripts.ltx2_video.run_dynamic_modular_vmm_distilled import parse_args

    monkeypatch.setattr(video_benchmark, "current_platform", lambda: "linux")
    options = argparse.Namespace(
        platform="linux", recipe=video_benchmark.PRESSURE_RECIPE, case="vmm-fast", family="runtime",
        vram_gib=8, checkpoint=tmp_path, custom_blocks=tmp_path, fixtures_dir=tmp_path, profile_file=None,
    )
    plan = video_benchmark.video_runtime_plan(options)
    case = plan["cases"]["vmm-fast"]
    assert case["routes"] == {"transformer": "vmm_fast_runtime", "encoder": "native_vmm", "vae": "resident"}
    assert case["dispatcher"] is True
    assert [stage["name"] for stage in case["stages"]] == ["profile-1", "profile-2", "measure"]
    for stage in case["stages"]:
        command = stage["command"]
        parsed = parse_args(command[4:])
        assert parsed.ddo_profile == "ltx23-video-1280x704-8s-8steps-vmm-fast"
        assert parsed.steps == (8 if stage["name"] == "measure" else 3)
        assert "--ddo-vmm-resident-weight-budget-gb" not in command
        assert parsed.ddo_vmm_total_staging_gb == 1.5
        assert parsed.ddo_vmm_source_cache_gb == parsed.ddo_vmm_source_registered_gb == 24.625
        assert parsed.text_encoder_native_vmm and parsed.text_encoder_vmm_resident_weight_budget_gb == 4
        assert not parsed.vae_leaf_offload and not parsed.ddo_vmm_execution_backing_reclaim
        assert parsed.video_decoder == "exact-streaming" and parsed.video_self_attention == "default"
        assert parsed.feed_forward_chunk_size == parsed.normalization_chunk_size == 0
        assert not any((parsed.phase_modulation, parsed.compact_timesteps,
                        parsed.inplace_residuals, parsed.reuse_rotary_embeddings))
        assert set(stage["environment"].values()) == {"0"}


def test_linux_video_vmm_verifier_rejects_wsl_registration_and_missing_dispatcher():
    case = {"routes": {"transformer": "vmm_fast_runtime"}, "preset": "vmm_fast", "profile": "test"}
    config = {
        "dynamic_offload_execution_mode": "vmm_fast_runtime", "dynamic_offload_effective_preset": "vmm_fast",
        "dynamic_offload_vmm_source_cache_gb": 24.625, "dynamic_offload_vmm_source_registered_gb": 24.625,
        "dynamic_offload_vmm_source_dispatcher": True, "dynamic_offload_vmm_execution_backing_reclaim": False,
    }
    runtime = {"fast_source_cache_source_dispatcher_enabled": 1}
    report = {
        "runner": "run_dynamic_modular_vmm_distilled", "dtype": "torch.bfloat16",
        "transformer_weight_format_requested": "bf16", "num_frames": 193,
        "requested_controls": {"ddo_vmm_total_staging_gb": 1.5, "text_encoder_native_vmm": True,
                               "vae_leaf_offload": False},
        "text_encoder_route_requested": "native",
        "dynamic_offload_reports": {
            "transformer": {"configuration": config, "runtime": {"vmm": runtime}},
            "text_encoder": {"configuration": {
                "dynamic_offload_execution_mode": "vmm_fast_runtime",
                "dynamic_offload_vmm_resident_weight_budget_gb": 4.0,
                "dynamic_offload_vmm_source_dispatcher": True}},
        },
        "ddo_profile": {"name": "test", "status": "saved", "build_stage": "floor_measured"},
    }
    plan = {"platform": "linux", "expected_frames": 193}
    video_benchmark.verify_video_vmm(plan, case, "profile-1", report, [])
    config["dynamic_offload_vmm_source_registered_gb"] = 0.5
    with pytest.raises(ValueError, match="configuration"):
        video_benchmark.verify_video_vmm(plan, case, "profile-1", report, [])
    config["dynamic_offload_vmm_source_registered_gb"] = 24.625
    runtime["fast_source_cache_source_dispatcher_enabled"] = 0
    with pytest.raises(ValueError, match="active source dispatcher"):
        video_benchmark.verify_video_vmm(plan, case, "profile-1", report, [])


@pytest.mark.parametrize(("case_name", "backend"), [("cudnn-sdpa", "cudnn-only"),
                                                  ("flash-attn2", "flash-attn2"),
                                                  ("sol-attn", "sol-attn")])
def test_linux_video_attention_keeps_runtime_recipe_and_profile(monkeypatch, tmp_path, case_name, backend):
    monkeypatch.setattr(video_benchmark, "current_platform", lambda: "linux")
    options = argparse.Namespace(
        platform="linux", recipe=video_benchmark.PRESSURE_RECIPE, case="vmm-fast", family="runtime",
        vram_gib=8, checkpoint=tmp_path, custom_blocks=tmp_path, fixtures_dir=tmp_path, profile_file=None,
        expandable_segments=False,
    )
    control = video_benchmark.video_runtime_plan(options)
    options.case, options.family = case_name, "attention"
    plan = video_benchmark.video_attention_plan(options, {"identity": {"plan": control}})
    selected = plan["cases"][case_name]
    assert [stage["name"] for stage in selected["stages"]] == ["operator-check", "measure"]
    assert selected["profile"] == control["cases"]["vmm-fast"]["profile"]
    assert selected["routes"] == control["cases"]["vmm-fast"]["routes"]
    reference = control["cases"]["vmm-fast"]["stages"][-1]
    for stage in selected["stages"]:
        command = stage["command"].copy()
        assert command[command.index("--video-self-attention") + 1] == backend
        command[command.index("--video-self-attention") + 1] = "default"
        if stage["name"] == "operator-check":
            command.remove("--trace-attention-operators")
        assert command == reference["command"]
        assert stage["environment"] == reference["environment"]


def sol_dispatch_report(trace=True):
    return {
        "requested_controls": {"trace_attention_operators": trace},
        "video_self_attention": {
            "requested_backend": "sol-attn",
            "sol_policy": {"backend": "triton", "tau": 1.0, "thresh_type": "diag"},
            "layers": [{"index": i, "backend": "flash-attn2" if i == 0 else "sol-attn",
                        "calls": 8, "fallback_calls": 0} for i in range(48)],
        },
        "attention_environment": {"pytorch_attention_operators": {
            "flash_attn::_flash_attn_forward": 8, "ddo::sol_attn_triton": 376,
        } if trace else None},
    }


@pytest.mark.parametrize("failure", [None, "revision", "dirty", "missing", "extra"])
def test_sol_dependency_identity_is_cpu_only_and_rejects_unreviewed_source(monkeypatch, tmp_path, failure):
    import hashlib
    from importlib import metadata
    from kernels import sol_loader

    package = tmp_path / 'sol_attn'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'interface.py').write_text('source = 1\n')
    manifest = {'schema': 1, 'revision': video_benchmark.SOL_ATTN_REVISION,
                'files': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                          for p in package.iterdir()}}
    if failure == 'revision':
        manifest['revision'] = 'wrong'
    (tmp_path / 'sol_attn_source.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(sol_loader, 'ROOT', tmp_path)
    monkeypatch.setattr(metadata, 'version', lambda name: '3.7.1')
    if failure == 'dirty':
        (package / 'interface.py').write_text('source = 2\n')
    elif failure == 'missing':
        (package / 'interface.py').unlink()
    elif failure == 'extra':
        (package / 'extra.py').write_text('')
    fake_suite = SimpleNamespace()
    if failure:
        with pytest.raises(ValueError):
            video_benchmark.video_sol_dependency_identity(fake_suite)
    else:
        before = video_benchmark.video_sol_dependency_identity(fake_suite)
        (package / 'interface.py').write_bytes(b'source = 1\r\n')
        after = video_benchmark.video_sol_dependency_identity(fake_suite)
        assert before['revision'] == after['revision']
        assert after['source_kind'] == 'bundled'
        assert before['triton_version'] == '3.7.1'
        assert before['triton_distribution'] == 'triton'
        assert len(before['files']) == 2


@pytest.mark.parametrize('platform,available,expected', [
    ('win32', {'triton': '3.7.1', 'triton-windows': '3.7.1.post27'}, 'triton'),
    ('win32', {'triton-windows': '3.7.1.post27'}, 'triton-windows'),
    ('linux', {'triton': '3.7.1'}, 'triton'),
    ('linux', {'triton-windows': '3.7.1.post27'}, None),
    ('win32', {}, None),
])
def test_triton_provider_fallback_is_windows_only(monkeypatch, platform, available, expected):
    from importlib import metadata

    calls = []
    def version(name):
        calls.append(name)
        if name not in available:
            raise metadata.PackageNotFoundError(name)
        return available[name]
    monkeypatch.setattr(video_benchmark.sys, 'platform', platform)
    monkeypatch.setattr(metadata, 'version', version)
    if expected is None:
        with pytest.raises(metadata.PackageNotFoundError):
            video_benchmark.video_triton_dependency_identity()
    else:
        assert video_benchmark.video_triton_dependency_identity() == {
            'triton_distribution': expected, 'triton_version': available[expected]}
    assert calls == (['triton', 'triton-windows'] if platform == 'win32' and 'triton' not in available
                     else ['triton'])


@pytest.mark.parametrize("trace", [True, False])
def test_sol_dispatch_accepts_only_planned_mixed_routes(trace):
    report = sol_dispatch_report(trace)
    case = {"attention_backend": "sol-attn", "attention_operator": "ddo::sol_attn_triton"}
    video_benchmark.verify_video_attention_dispatch(case, report, trace)
    report["video_self_attention"]["layers"][0]["backend"] = "sol-attn"
    with pytest.raises(ValueError, match="layer policy"):
        video_benchmark.verify_video_attention_dispatch(case, report, trace)


@pytest.mark.parametrize("failure", ["fallback", "missing", "calls", "marker", "flash", "tau"])
def test_sol_dispatch_rejects_incomplete_or_altered_evidence(failure):
    report = sol_dispatch_report()
    layers = report["video_self_attention"]["layers"]
    operators = report["attention_environment"]["pytorch_attention_operators"]
    if failure == "fallback":
        layers[3]["fallback_calls"] = 1
    elif failure == "missing":
        layers.pop()
    elif failure == "calls":
        layers[1]["calls"] = 7
    elif failure == "marker":
        operators["ddo::sol_attn_triton"] = 375
    elif failure == "flash":
        operators["flash_attn::_flash_attn_forward"] = 0
    else:
        report["video_self_attention"]["sol_policy"]["tau"] = 0.5
    with pytest.raises(ValueError):
        video_benchmark.verify_video_attention_dispatch(
            {"attention_backend": "sol-attn", "attention_operator": "ddo::sol_attn_triton"}, report, True)


@pytest.mark.parametrize("case_name", ["one-shot-fast", "low-ram-safe"])
def test_linux_video_linear_reuses_successful_zero_budget_profile(monkeypatch, tmp_path, case_name):
    monkeypatch.setattr(video_benchmark, "current_platform", lambda: "linux")
    profile = tmp_path / "ddo-capacity-v1-test.json"
    payload = {
        "schema_version": 1, "profile_key": profile.stem,
        "context": {"name": f"ltx23-video-1280x704-8s-8steps-{case_name}",
                    "execution_mode": "linear_runtime", "model_class": "LTX2VideoTransformer3DModel"},
        "observations": [{"status": "success"}],
        "recommendation": {"profile_resident_module_budget_gb": 0.0},
    }
    profile.write_text(json.dumps(payload))
    options = argparse.Namespace(
        platform="linux", recipe=video_benchmark.PRESSURE_RECIPE, case=case_name, family="runtime",
        vram_gib=8, checkpoint=tmp_path, custom_blocks=tmp_path, fixtures_dir=tmp_path, profile_file=profile,
        expandable_segments=case_name == "low-ram-safe",
    )
    plan = video_benchmark.video_runtime_plan(options)
    case = plan["cases"][case_name]
    assert [stage["name"] for stage in case["stages"]] == ["measure"]
    assert case["profile"] == payload["context"]["name"]
    assert "--build-ddo-profile" not in case["stages"][0]["command"]
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "inherited-invalid-setting")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "inherited-invalid-setting")
    env = suite.child_environment(plan)
    assert env.get("PYTORCH_ALLOC_CONF") == ("expandable_segments:True" if options.expandable_segments else None)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    payload["recommendation"]["profile_resident_module_budget_gb"] = 3.0
    profile.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="zero-budget"):
        video_benchmark.video_runtime_plan(options)


@pytest.mark.parametrize(("system", "case_name"), [("windows", "low-ram-safe"), ("wsl", "low-ram-safe"),
                                                   ("linux", "vmm-fast"), ("linux", "one-shot-fast")])
def test_video_expandable_diagnostic_rejects_other_routes(system, case_name):
    options = argparse.Namespace(platform=system, case=case_name, expandable_segments=True)
    with pytest.raises(ValueError, match="Linux Low RAM Safe"):
        video_benchmark.video_runtime_plan(options)


@pytest.mark.parametrize("reuse_final_profile", [False, True])
def test_windows_video_vmm_resident_budget_comes_from_profile(monkeypatch, tmp_path, reuse_final_profile):
    from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
    from scripts.ltx2_video.run_dynamic_modular_vmm_distilled import parse_args

    monkeypatch.setattr(video_benchmark, "current_platform", lambda: "windows")
    shell = SimpleNamespace(shell32=SimpleNamespace(IsUserAnAdmin=lambda: True))
    monkeypatch.setattr(video_benchmark, "ctypes", SimpleNamespace(windll=shell))
    profile = None
    if reuse_final_profile:
        profile = tmp_path / "ddo-vmm-residency-v1-test.json"
        profile.write_text(json.dumps({
            "context": {"name": "reviewed-profile"}, "profile_key": profile.stem,
            "recommendation": {
                "vmm_build_stage": "final", "vmm_resident_weight_budget_bytes": 675282944,
            },
        }), encoding="utf-8")
    options = argparse.Namespace(
        platform="windows", recipe=video_benchmark.PRESSURE_RECIPE, case="vmm-fast", family="runtime",
        vram_gib=8, checkpoint=tmp_path, custom_blocks=tmp_path, fixtures_dir=tmp_path, profile_file=profile,
    )
    plan = video_benchmark.video_runtime_plan(options)
    case = plan["cases"]["vmm-fast"]
    expected_stages = ["measure"] if profile else ["profile-1", "profile-2", "measure"]
    assert [stage["name"] for stage in case["stages"]] == expected_stages
    for stage in case["stages"]:
        argv = stage["command"][4:]
        assert "--ddo-vmm-resident-weight-budget-gb" not in argv
        parsed = parse_args(argv)
        route = resolve_dynamic_offload_backend(parsed.ddo_backend, parsed.ddo_preset)
        assert route.build_config_from_args(parsed).vmm_resident_weight_budget_gb == 0
        assert parsed.ddo_profile == case["profile"]
        assert parsed.ddo_vmm_total_staging_gb == 1.5


def test_ltx_flash_is_scoped_to_self_attention(monkeypatch):
    from diffusers.models import attention_dispatch
    from scripts.helpers.image_benchmark_attention import (
        add_image_attention_arguments, configure_image_attention,
    )

    parser = argparse.ArgumentParser()
    add_image_attention_arguments(parser)
    selected = parser.parse_args(["--image-self-attention", "flash-attn2"])
    monkeypatch.setattr(attention_dispatch, "_check_attention_backend_requirements", lambda backend: None)
    self_processor = SimpleNamespace(_attention_backend=None)
    cross_processor = SimpleNamespace(_attention_backend=None)
    transformer = SimpleNamespace(transformer_blocks=[SimpleNamespace(
        attn1=SimpleNamespace(processor=self_processor),
        attn2=SimpleNamespace(processor=cross_processor),
    )])
    metrics = {}
    configure_image_attention(transformer, parser.parse_args([]), metrics)
    assert self_processor._attention_backend is cross_processor._attention_backend is None
    configure_image_attention(transformer, selected, metrics)
    assert self_processor._attention_backend == "flash"
    assert cross_processor._attention_backend == "native"
    assert metrics["transformer_attention_scope"] == "self"


def test_qwen_attention_plan_preserves_joint_mask():
    recipe = "qwen-image21"
    plan = suite.build_plan(args(recipe, "attention"))
    assert set(plan["cases"]) == {"official-leaf", "cudnn-sdpa"}
    with pytest.raises(ValueError, match="padding mask"):
        suite.build_plan(args(recipe, "attention", "flash-attn2"))


def test_qwen_wsl_leaf_control_disables_streaming_and_verifies_metrics(tmp_path):
    from scripts.qwen_image_21 import run_diffusers_leaf

    recipe = suite.RECIPES["qwen-image21"]
    selected = args("qwen-image21")
    selected.platform = "wsl"
    cases = suite.build_plan(selected)["cases"]
    assert tuple(cases) == suite.WSL_RUNTIME
    control = cases["official-leaf"]
    (stage,) = control["stages"]
    assert stage["name"] == "measure" and control["profile"] is None
    parsed = run_diffusers_leaf.parse_args(stage["command"][4:])
    assert (parsed.width, parsed.height, parsed.steps, parsed.seed) == (1024, 1024, 20, 42)
    assert parsed.no_stream_offload and parsed.vae_leaf_offload and parsed.image_attention == "default"
    assert not any(flag.startswith("--purge-") for flag in stage["command"])
    assert not run_diffusers_leaf.parse_args([]).no_stream_offload

    metrics = tmp_path / "metrics.json"
    report = {
        "workload": {"width": 1024, "height": 1024, "steps_requested": 20, "seed": 42},
        "steps": isolated_phase_rows("denoise", "decode"),
        "offload": {"offload_type": "leaf_level", "use_stream": False},
        "transformer_attention_requested": "native", "vae_decode_route": "diffusers_leaf",
        "total_elapsed_sec": 10, "global_peak_vram_absolute_gb": 2, "global_peak_ram_gb": 2,
        "memory_monitor": {"ram_metric": "rss"},
    }
    metrics.write_text(json.dumps(report))
    Image.new("RGB", recipe["size"]).save(tmp_path / "output.png")
    assert len(suite.verify_stage(recipe, control, "measure", tmp_path, tmp_path)) == 2
    report["offload"]["use_stream"] = True
    metrics.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="non-streamed Leaf"):
        suite.verify_stage(recipe, control, "measure", tmp_path, tmp_path)


@pytest.mark.parametrize(("platform", "recipe"), [
    ("wsl", "ltx23-image"), ("wsl", "qwen-image21"),
    ("windows", "ltx23-image"), ("windows", "qwen-image21"),
    ("linux", "ltx23-image"), ("linux", "qwen-image21"),
])
def test_vmm_attention_uses_verified_runtime_control(tmp_path, platform, recipe):
    selected = args(recipe, "runtime", "vmm-fast")
    selected.platform = platform
    if platform != "wsl" and recipe == "qwen-image21":
        selected.vae_model_path = tmp_path / "vae"
    runtime = suite.build_plan(selected)
    base = runtime["cases"]["vmm-fast"]
    selected.family, selected.case, selected.attention_runtime = "attention", None, "vmm-fast"
    attention = suite.build_plan(selected)
    ltx = recipe == "ltx23-image"
    if platform != "wsl" and not ltx:
        assert base["routes"]["vae"] == base["routes"]["encoder"] == "native_vmm"
        for stage in base["stages"]:
            command = stage["command"]
            assert command[command.index("--vae-model-path") + 1] == str(selected.vae_model_path)
            assert command[command.index("--vae-vmm-resident-weight-budget-gb") + 1] == "0.375"
            assert "--no-vae-no-temporal-cache" in command
            assert not any(flag.startswith("--purge-") for flag in command)
    assert tuple(attention["cases"]) == (("cudnn-sdpa", "flash-attn2") if ltx else ("cudnn-sdpa",))
    for name, variant in attention["cases"].items():
        diagnostic, measurement = variant["stages"]
        assert [diagnostic["name"], measurement["name"]] == ["operator-check", "measure"]
        original = base["stages"][-1]["command"]
        if ltx:
            assert measurement["command"] == original + ["--image-self-attention", suite.ATTENTION[name]]
        else:
            changed = original.index("--image-joint-attention") + 1
            assert original[changed] == "default"
            assert measurement["command"] == original[:changed] + ["cudnn"] + original[changed + 1:]
        assert diagnostic["command"] == measurement["command"] + ["--trace-transformer-attention"]
        assert variant["routes"] == base["routes"]

    recommendation = {"vmm_resident_weight_budget_bytes": 2327838720}
    profile, metrics, png = (tmp_path / name for name in ("profile.json", "metrics.json", "output.png"))
    profile.write_text(json.dumps({"recommendation": recommendation}))
    metrics.write_text(json.dumps({"transformer_attention_requested": "native", "ddo_profile": {
        "name": base["profile"], "status": "applied", "build_stage": "final", "recommendation": recommendation,
    }}))
    Image.new("RGB", suite.RECIPES[recipe]["size"]).save(png)
    old = {"plan": runtime, "packages": {"flash-attn": None}, "runner_sha256": {
        "benchmarks/image_suite.py": "old", "scripts/runner.py": "same",
    }}
    current = {"plan": attention, "packages": {"flash-attn": "2.8.3.post2"}, "runner_sha256": {
        "benchmarks/image_suite.py": "new", "scripts/runner.py": "same",
    }}
    manifest = tmp_path / "manifest.json"
    suite.write_state(manifest, {"identity": old, "cases": {"vmm-fast": {
        "status": "completed", "artifacts": {str(file): suite.digest(file) for file in (metrics, png)},
        "profile_receipt": {str(profile): suite.digest(profile)},
    }}})
    assert suite.verified_vmm_attention_reference(manifest, current)["resident_budget_bytes"] == 2327838720
    current["runner_sha256"]["scripts/runner.py"] = "changed"
    with pytest.raises(ValueError, match="does not match"):
        suite.verified_vmm_attention_reference(manifest, current)
    current["runner_sha256"]["scripts/runner.py"] = "same"
    profile.write_text("{}")
    with pytest.raises(ValueError, match="missing or has changed"):
        suite.verified_vmm_attention_reference(manifest, current)


def test_export_profile_rekeys_and_preserves_conflicting_recommendation(tmp_path):
    from diffusers_dynamic_offloader.profiles import (
        dynamic_offload_profile_path, load_dynamic_offload_profile,
    )

    context = {"name": "old", "model_class": "Model", "execution_mode": "linear_runtime",
               "identity_version": 2}
    source = dynamic_offload_profile_path(context, tmp_path)
    record = {"schema_version": 1, "context": context, "profile_key": source.stem,
              "recommendation": {"profile_resident_module_budget_gb": 5.5}}
    source.write_text(json.dumps(record), encoding="utf-8")
    store = tmp_path / "store"
    target = suite.export_profile(source, "new", store)
    assert target != source
    assert suite.export_profile(source, "new", store) == target
    assert load_dynamic_offload_profile({**context, "name": "new"}, store)["recommendation"] == record["recommendation"]
    assert json.loads(source.read_text()) == record
    record["recommendation"]["profile_resident_module_budget_gb"] = 4
    source.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="different profile"):
        suite.export_profile(source, "new", store)


@pytest.mark.parametrize(("recipe", "family", "platform", "case_name"), [
    (recipe, family, "windows", None)
    for recipe in suite.RECIPES for family in ("runtime", "attention")
] + [("qwen-image21", "runtime", "wsl", case) for case in ("wsl-compat", "vmm-fast")]
  + [("ltx23-image", "runtime", "linux", case)
     for case in ("official-leaf", "one-shot-fast", "low-ram-safe", "vmm-fast")]
  + [("qwen-image21", "runtime", "linux", case)
     for case in ("official-leaf", "one-shot-fast", "low-ram-safe", "vmm-fast")])
def test_commands_match_runner_parsers(recipe, family, platform, case_name, monkeypatch, tmp_path):
    import sys
    import torch

    def no_cuda(*_args, **_kwargs):
        pytest.fail("Benchmark parser initialized CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    monkeypatch.setattr(sys, "argv", ["parser-test", "--help"])
    selected = args(recipe, family, case_name)
    selected.platform = platform
    selected.checkpoint = tmp_path
    (tmp_path / "vae").mkdir()
    selected.expandable_segments = platform == "linux" and recipe == "qwen-image21" and case_name != "vmm-fast"
    if platform in ("windows", "linux") and recipe == "qwen-image21":
        selected.vae_model_path = tmp_path
    plan = suite.build_plan(selected)
    if selected.expandable_segments:
        assert suite.child_environment(plan)["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    if platform == "linux":
        case = plan["cases"][case_name]
        if case_name == "official-leaf":
            assert case["profile"] is None and case["require_streamed_leaf"]
        elif case_name == "vmm-fast":
            assert case["profile"] and case["dispatcher"]
            assert case["routes"] == {
                "transformer": "vmm_fast_runtime", "encoder": "native_vmm", "vae": "native_vmm",
            }
            assert [stage["name"] for stage in case["stages"]] == ["profile-1", "profile-2", "measure"]
        else:
            assert case["profile"] and [stage["name"] for stage in case["stages"]] == ["profile-1", "measure"]
        command = case["stages"][0]["command"]
        assert "--no-stream-offload" not in command
        assert not any("purge" in flag for flag in command if flag.startswith("--"))
    for case in plan["cases"].values():
        for stage in case["stages"]:
            command = stage["command"]
            runner = Path(command[3])
            path = runner.relative_to(suite.ROOT).with_suffix("")
            if recipe == "ltx23-image" and runner.stem == "run_diffusers_leaf":
                path = path.with_name("run_dynamic_modular_distilled")
                kwargs = {"official_leaf": True}
            else:
                kwargs = {}
            module = importlib.import_module(".".join(path.parts))
            monkeypatch.setattr(sys, "argv", command[3:] + ["--output-dir", "unused"])
            parsed = module.parse_args(**kwargs)
            assert parsed.width == suite.RECIPES[recipe]["size"][0]
            assert parsed.vae_leaf_offload == (case["routes"]["vae"] == "diffusers_leaf")
            if case["routes"]["vae"] == "native_vmm":
                expected_vae = (Path(plan["checkpoint"]) / "vae" if recipe == "ltx23-image"
                                else selected.vae_model_path)
                assert parsed.vae_vmm_offload and parsed.vae_model_path == expected_vae
            assert getattr(parsed, "ddo_vmm_execution_backing_reclaim", False) is False
            if case["dispatcher"]:
                assert parsed.text_encoder_native_vmm is (platform != "wsl")
            if platform == "wsl":
                assert parsed.image_attention == "default"
                assert not parsed.vae_no_temporal_cache
                assert parsed.steps == (3 if case["dispatcher"] and stage["name"] != "measure" else 20)
                assert not any(flag.startswith("--purge-") for flag in command)
                if case["dispatcher"]:
                    for flag, value in (
                        ("--ddo-vmm-total-staging-gb", "0.625"),
                        ("--ddo-vmm-source-cache-gb", "16"),
                        ("--ddo-vmm-source-registered-gb", "0.5"),
                    ):
                        assert command[command.index(flag) + 1] == value
                else:
                    assert parsed.ddo_preset == "wsl_compat"
                    assert parsed.ddo_profile == "qwen-image21-1024x1024-20steps-wsl-compat"
                    assert parsed.build_ddo_profile is (stage["name"] == "profile-1")
                    assert stage["environment"] == {"DDO_RESIDENT_MODULE_BUDGET_GB": "0"}
            monkeypatch.setattr(sys, "argv", ["parser-test", "--help"])


def test_environment_removes_policy_overrides(monkeypatch, tmp_path):
    for key in ("DDO_VMM_SOURCE_DISPATCHER", "DDO_VMM_LIBRARY_PATH", "PYTORCH_CUDA_ALLOC_CONF",
                "DIFFUSERS_ATTN_BACKEND", "HF_MODULES_CACHE"):
        monkeypatch.setenv(key, "override")
    env = suite.child_environment(suite.build_plan(args("ltx23-image")), tmp_path)
    assert env["DIFFUSERS_ATTN_BACKEND"] == "native"
    assert env["DDO_PROFILE_DIR"] == str(tmp_path / "profiles")
    assert env["HF_MODULES_CACHE"] == str(tmp_path / "hf-modules")
    assert "DDO_VMM_SOURCE_DISPATCHER" not in env
    assert "DDO_VMM_LIBRARY_PATH" not in env
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env


def test_ltx_low_ram_pre_vae_purge_changes_only_measurement():
    cases = suite.build_plan(args("ltx23-image"))["cases"]
    control = cases["low-ram-safe"]
    variant = cases[suite.LOW_RAM_VAE_PURGE]
    assert variant["profile"] == control["profile"]
    assert variant["stages"][0] == control["stages"][0]
    assert variant["stages"][1]["command"] == control["stages"][1]["command"] + ["--purge-before-vae"]
    assert suite.LOW_RAM_VAE_PURGE not in suite.build_plan(args("qwen-image21"))["cases"]


def test_component_control_uses_vmm_fast_with_leaf_vae():
    control = suite.build_plan(args("ltx23-image", "components"))["cases"]["vmm-fast"]
    assert control["routes"]["vae"] == "diffusers_leaf"
    assert control["dispatcher"] is True


def test_windows_ltx_vmm_fast_uses_native_vae_and_two_purges():
    case = suite.build_plan(args("ltx23-image", case="vmm-fast"))["cases"]["vmm-fast"]
    assert case["hold"] is None
    assert case["routes"]["encoder"] == case["routes"]["vae"] == "native_vmm"
    for stage in case["stages"]:
        command = stage["command"]
        assert "--text-encoder-native-vmm" in command
        assert "--vae-vmm-offload" in command
        assert "--vae-leaf-offload" not in command
        assert command[command.index("--vae-vmm-resident-weight-budget-gb") + 1] == "1.5"
        assert {flag for flag in command if flag.startswith("--purge-")} == {
            "--purge-before-run", "--purge-before-transformer",
        }


def test_nvme_component_control_uses_fixed_budget_and_leaf_components():
    control = suite.build_plan(args("ltx23-image", "components", "nvme-baseline"))["cases"]["nvme-baseline"]
    official = suite.build_plan(args("ltx23-image", case="nvme-baseline"))["cases"]["nvme-baseline"]
    assert official["hold"]
    assert control == {**official, "hold": None}
    assert control["profile"] is None
    assert control["dispatcher"] is None
    assert control["routes"] == {"transformer": "vmm_runtime", "encoder": "diffusers_leaf", "vae": "diffusers_leaf"}
    stage, = control["stages"]
    assert stage["name"] == "measure"
    command = stage["command"]
    assert command[command.index("--ddo-vmm-resident-weight-budget-gb") + 1] == "1.5"


def isolated_phase_rows(*names):
    return [{"name": name, "elapsed_sec": 1, "peak_vram_absolute_gb": 2,
             "peak_ram_gb": 3, "ram_metric": "rss",
             "memory_monitor": {"ram_metric": "rss", "samples": 2, "error_type": None}}
            for name in names]


def write_leaf_result(output):
    report = {
        "status": "completed", "width": 1280, "height": 704, "seed": 43, "num_inference_steps": 8,
        "steps": isolated_phase_rows("denoise", "Decode VAE"),
        "total_elapsed_sec": 10, "global_peak_vram_absolute_gb": 4, "global_peak_ram_gb": 8,
        "memory_monitor": {"ram_metric": "rss"}, "vae_decode_route": "diffusers_leaf",
        "offload": {"offload_type": "leaf_level"},
    }
    (output / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    Image.new("RGB", (1280, 704)).save(output / "image.png")


@pytest.mark.parametrize("no_purge", [False, True])
def test_purge_override_requires_matching_receipt(tmp_path, no_purge):
    selection = args("ltx23-image", case="official-leaf")
    selection.purge_before_transformer_only = not no_purge
    selection.no_purge = no_purge
    case = suite.build_plan(selection)["cases"]["official-leaf"]
    recipe = suite.RECIPES["ltx23-image"]
    assert "--no-purge-before-run" in case["stages"][-1]["command"]
    assert ("--no-purge-before-transformer" in case["stages"][-1]["command"]) == no_purge
    write_leaf_result(tmp_path)
    path = tmp_path / "metrics.json"
    report = json.loads(path.read_text())
    report["purge_windows_standby_before_run"] = {"purged": True}
    report["purge_windows_standby_before_transformer_1"] = {"purged": True}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Expected successful purges"):
        suite.verify_stage(recipe, case, "measure", tmp_path, tmp_path)
    report["purge_windows_standby_before_run"] = None
    if no_purge:
        report["purge_windows_standby_before_transformer_1"] = None
    path.write_text(json.dumps(report))
    assert suite.verify_stage(recipe, case, "measure", tmp_path, tmp_path)


@pytest.mark.parametrize("transformer_only", [False, True])
def test_one_shot_calibration_and_measurement_use_same_purge_policy(transformer_only):
    selection = args("ltx23-image", case="one-shot-fast")
    selection.purge_before_transformer_only = transformer_only
    for stage in suite.build_plan(selection)["cases"]["one-shot-fast"]["stages"]:
        command = stage["command"]
        assert "--no-purge-before-run" in command
        assert ("--purge-before-transformer" in command) == transformer_only


@pytest.mark.parametrize("recipe", suite.RECIPES)
def test_low_ram_uses_named_profile_without_a_fixed_budget_override(recipe):
    selection = args(recipe, case="low-ram-safe")
    for stage in suite.build_plan(selection)["cases"]["low-ram-safe"]["stages"]:
        assert "--ddo-profile" in stage["command"]
        assert "DDO_RESIDENT_MODULE_BUDGET_GB" not in stage.get("environment", {})


def test_failure_resume_and_skip_completed(tmp_path):
    plan = suite.build_plan(args("ltx23-image", case="official-leaf"))
    manifest = tmp_path / "manifest.json"
    state = {"identity": {"plan": plan}, "cases": {"official-leaf": {"status": "pending"}}}
    with pytest.raises(suite.subprocess.CalledProcessError):
        suite.execute_session(manifest, state, lambda *_: 1)
    first_output = Path(state["cases"]["official-leaf"]["active_output"])
    assert first_output.is_dir()

    def run(command, output, env):
        assert output != first_output
        write_leaf_result(output)
        return 0

    with pytest.raises(RuntimeError, match="still running"):
        def active(_):
            raise RuntimeError("still running")

        suite.execute_session(manifest, suite.read_json(manifest), run, check_child=active)
    saved = suite.read_json(manifest)
    suite.execute_session(manifest, saved, run, check_child=lambda path: path == first_output)
    assert saved["cases"]["official-leaf"]["status"] == "completed"
    suite.execute_session(manifest, saved, lambda *_: pytest.fail("Completed case ran twice"))
    next(Path(path) for path in saved["cases"]["official-leaf"]["artifacts"] if path.endswith(".png")).unlink()
    with pytest.raises(ValueError, match="artifacts changed"):
        suite.execute_session(manifest, saved, lambda *_: pytest.fail("Unverified case ran"))


def test_reuse_requires_verified_artifacts_and_same_identity(tmp_path):
    runtime = suite.build_plan(args("ltx23-image", case="official-leaf"))
    attention = suite.build_plan(args("ltx23-image", "attention", "official-leaf"))
    old = {"identity": {"plan": runtime, "dll": "reviewed"}, "cases": {
        "official-leaf": {"status": "completed", "artifacts": {"image": "digest"}}}}
    new = {"identity": {"plan": attention, "dll": "reviewed"}, "cases": {"official-leaf": {}}}
    with patch.object(suite, "receipts_valid", return_value=True):
        suite.reusable_measurements(new, [old])
    assert "measure" in new["cases"]["official-leaf"]["stages"]
    new["identity"]["dll"] = "changed"
    new["cases"]["official-leaf"] = {}
    with patch.object(suite, "receipts_valid", return_value=True):
        suite.reusable_measurements(new, [old])
    assert not new["cases"]["official-leaf"]
