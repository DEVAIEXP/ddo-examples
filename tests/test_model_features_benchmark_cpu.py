"""Cumulative recipes preserve their control budget, evidence and resume boundary."""

import argparse
import importlib
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks import image_suite as suite, model_features_suite as features, run_benchmark as video


def arguments(tmp_path, recipe=features.VIDEO, case=None):
    (tmp_path / "bf16-vae").mkdir(exist_ok=True)
    return argparse.Namespace(platform="windows", recipe=recipe, family="model-features", case=case,
                              checkpoint=tmp_path, custom_blocks=tmp_path, fixtures_dir=tmp_path,
                              vae_model_path=tmp_path / "bf16-vae", vram_gib=8)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    import torch
    def forbidden(*args, **kwargs):
        pytest.fail("CPU feature preparation attempted to initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(video, "current_platform", lambda: "windows")
    if hasattr(video.ctypes, "windll"):
        # Preserve kernel32 for dependency imports on the real Windows host.
        monkeypatch.setattr(video.ctypes.windll.shell32, "IsUserAnAdmin", lambda: True)
    else:
        monkeypatch.setattr(video.ctypes, "windll", SimpleNamespace(
            shell32=SimpleNamespace(IsUserAnAdmin=lambda: True)), raising=False)


@pytest.mark.parametrize("recipe,count", [(features.VIDEO, 10), (features.QWEN, 9)])
def test_cumulative_queue_and_defaults(tmp_path, recipe, count):
    plan = features.build_plan(suite, video, arguments(tmp_path, recipe))
    assert tuple(plan["cases"]) == features.case_names(recipe)
    assert len(plan["cases"]) == count
    cases = list(plan["cases"].values())
    assert [s["name"] for s in cases[0]["stages"]] == ["profile-1", "profile-2", "measure"]
    previous = cases[0]
    for name, case in list(plan["cases"].items())[1:]:
        if recipe == features.VIDEO:
            assert [s["name"] for s in case["stages"]] == ["profile-1", "profile-2", "measure"]
            assert case["feature_fixed_budget_case"] == features.CONTROL
            assert case["profile"] != previous["profile"]
        else:
            assert [s["name"] for s in case["stages"]] == ["measure"]
            assert case["feature_profile_case"] == features.CONTROL
            assert case["profile"] == previous["profile"]
        assert case["routes"] == previous["routes"] and case["dispatcher"] is True
        parent = plan["cases"][case["previous_case"]]
        changed = [k for k in case["model_features"] if case["model_features"][k] != parent["model_features"][k]]
        assert changed == [case["added_feature"]]
        previous = case
    for case in cases:
        for stage in case["stages"]:
            command = stage["command"]
            assert "--build-ddo-profile" in command if stage["name"].startswith("profile-") else "--build-ddo-profile" not in command
            assert command[command.index("--ddo-profile") + 1] == case["profile"]
            assert "--trace-attention-operators" not in command
            flag = "--video-self-attention" if recipe == features.VIDEO else "--image-joint-attention"
            assert command[command.index(flag) + 1] == "default"
            assert "--compile-repeated-blocks" not in command
    if recipe == features.VIDEO:
        assert cases[-1]["model_features"]["feed_forward_chunk_size"] == 1024
        assert cases[-1]["model_features"]["normalization_chunk_size"] == 1024
        assert "--no-reuse-rotary-embeddings" in cases[0]["stages"][-1]["command"]
        assert "--no-reuse-rotary-embeddings" not in cases[1]["stages"][-1]["command"]
    else:
        assert cases[0]["routes"]["vae"] == "native_vmm"
        assert cases[-1]["model_features"]["vae_band_rows"] is None
        assert cases[-1]["model_features"]["vae_tiling"] is True


def test_selected_case_keeps_prerequisites_and_platform_scope(tmp_path):
    args = arguments(tmp_path, case="compact-timesteps")
    assert list(features.build_plan(suite, video, args)["cases"]) == [features.CONTROL, "rope-reuse", "compact-timesteps"]
    args.platform = "wsl"
    with pytest.raises(ValueError, match="Windows"):
        features.build_plan(suite, video, args)
    with pytest.raises(ValueError, match="LTX Image"):
        features.case_names("ltx23-image")


def test_linux_video_queue_preserves_native_routes_and_excludes_purge(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.platform = "linux"
    monkeypatch.setattr(video, "current_platform", lambda: "linux")
    plan = features.build_plan(suite, video, args)
    monkeypatch.setattr(sys, "argv", ["CPU-parser-check", "--help"])
    parse = importlib.import_module("scripts.ltx2_video.run_model_features_distilled").parse_args
    assert len(plan["cases"]) == 10
    assert plan["cases"][features.CONTROL]["feature_control_exact_budget"] is True
    for name, case in plan["cases"].items():
        assert case["routes"] == {"transformer": "vmm_fast_runtime", "encoder": "native_vmm", "vae": "resident"}
        if name != features.CONTROL:
            assert case["feature_fixed_budget_case"] == features.CONTROL
        for stage in case["stages"]:
            command = stage["command"]
            assert not any(flag.startswith("--purge-") for flag in command)
            assert not any(value == "1" for key, value in stage["environment"].items() if "PURGE" in key)
            parsed = parse(command[4:])
            assert parsed.text_encoder_native_vmm is True
            assert parsed.text_encoder_vmm_resident_weight_budget_gb == 4
            assert parsed.ddo_vmm_total_staging_gb == 1.5
            assert parsed.ddo_vmm_source_cache_gb == parsed.ddo_vmm_source_registered_gb == 24.625
            assert parsed.video_self_attention == "default"
            assert parsed.video_decoder == "exact-streaming"
            assert parsed.vae_leaf_offload is False
            for key, value in case["model_features"].items():
                assert getattr(parsed, key) == value


def test_windows_control_keeps_its_existing_profile_policy(tmp_path):
    plan = features.build_plan(suite, video, arguments(tmp_path))
    assert "feature_control_exact_budget" not in plan["cases"][features.CONTROL]
    assert "--purge-before-transformer" in plan["cases"][features.CONTROL]["stages"][-1]["command"]
    args = arguments(tmp_path, features.QWEN)
    args.platform = "wsl"
    with pytest.raises(ValueError, match="Windows or native Linux"):
        features.build_plan(suite, video, args)


def report_for(case):
    expected = case["model_features"]
    report = {"requested_controls": {**expected, "benchmark_feature_entrypoint": True}}
    for field, metric in (("reuse_rotary_embeddings", "rotary_reuse_prepared"),
                          ("compact_timesteps", "compact_timesteps_executed"),
                          ("inplace_residuals", "inplace_residuals_executed"),
                          ("phase_modulation", "phase_modulation_configuration_applied"),
                          ("frame_modulation", "frame_modulation_executed")):
        report[metric] = expected[field]
    report["execution_ops_submissions"] = {"rms_modulate_submissions": int(expected["fused_rms_modulation"])}
    report["video_self_attention"] = {"fused_rotary_pair_submissions": int(expected["fused_rotary_pair"])}
    for _, field, _ in features.VIDEO_CHUNKS:
        report[field + "_configuration_applied"] = expected[field]
    return report


def test_features_require_application_and_submission_evidence(tmp_path):
    plan = features.build_plan(suite, video, arguments(tmp_path))
    for case in plan["cases"].values():
        features.verify_features(features.VIDEO, case, report_for(case))
    final = plan["cases"]["normalization-1024"]
    for metric in ("rotary_reuse_prepared", "phase_modulation_configuration_applied",
                   "execution_ops_submissions", "video_self_attention", "normalization_chunk_size_configuration_applied"):
        report = report_for(final)
        del report[metric]
        with pytest.raises(ValueError):
            features.verify_features(features.VIDEO, final, report)
    case = plan["cases"][features.CONTROL]
    report = report_for(case)
    report["requested_controls"]["phase_modulation"] = True
    with pytest.raises(ValueError):
        features.verify_features(features.VIDEO, case, report)


def test_qwen_requires_decode_evidence_but_not_during_calibration(tmp_path):
    plan = features.build_plan(suite, video, arguments(tmp_path, features.QWEN))
    case = plan["cases"]["bands-256"]
    report = {"executed_controls": {**case["model_features"], "vae_class": "QwenImage21ImageVAE", "vae_band_blocks": 2}}
    features.verify_features(features.QWEN, case, report)
    features.verify_features(features.QWEN, case, {}, calibration=True)
    for field in (*case["model_features"], "vae_class", "vae_band_blocks"):
        changed = deepcopy(report)
        del changed["executed_controls"][field]
        with pytest.raises(ValueError):
            features.verify_features(features.QWEN, case, changed)


@pytest.mark.parametrize("recipe,module", [
    (features.VIDEO, "scripts.ltx2_video.run_model_features_distilled"),
    (features.QWEN, "scripts.qwen_image_21.run_dynamic_modular_vmm"),
])
def test_every_command_passes_the_actual_runner_parser(tmp_path, monkeypatch, recipe, module):
    monkeypatch.setattr(sys, "argv", ["CPU-parser-check", "--help"])
    parser = importlib.import_module(module).parse_args
    plan = features.build_plan(suite, video, arguments(tmp_path, recipe))
    for case in plan["cases"].values():
        for stage in case["stages"]:
            parsed = parser(stage["command"][4:])
            for key, value in case["model_features"].items():
                assert getattr(parsed, key) == value
            assert parsed.ddo_backend == "vmm" and parsed.ddo_preset == "auto"
            assert parsed.ddo_profile == case["profile"]


def test_compute_entry_preserves_public_runner_and_excludes_compilation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["CPU-parser-check", "--help"])
    public = importlib.import_module("scripts.ltx2_video.run_dynamic_modular_vmm_distilled")
    explicit = importlib.import_module("scripts.ltx2_video.run_model_features_distilled")
    with pytest.raises(SystemExit):
        public.parse_args(["--fused-rms-modulation"])
    with pytest.raises(SystemExit):
        explicit.parse_args(["--compile-repeated-blocks"])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {str(path): suite.digest(path)}


def test_shared_profile_requires_complete_control_and_matching_budget(tmp_path):
    profile = tmp_path / "profiles/profile.json"
    receipt = write_json(profile, {"context": {"name": "control"}, "recommendation": {
        "vmm_build_stage": "final", "vmm_resident_weight_budget_bytes": 1024}})
    metrics = tmp_path / "metrics.json"
    artifacts = write_json(metrics, {"ddo_profile": {"status": "applied", "shape_check": "match",
                                                    "recommendation": {"vmm_resident_weight_budget_bytes": 1024}}})
    state = {"identity": {"plan": {"cases": {features.CONTROL: {"profile": "control"}}}},
             "cases": {features.CONTROL: {"status": "completed", "artifacts": artifacts, "profile_receipt": receipt}}}
    assert features.control_profile(state, features.CONTROL)["resident_budget_bytes"] == 1024
    state["cases"][features.CONTROL]["status"] = "pending"
    with pytest.raises(ValueError):
        features.control_profile(state, features.CONTROL)
    state["cases"][features.CONTROL]["status"] = "completed"
    profile.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        features.control_profile(state, features.CONTROL)


def test_resume_skips_control_and_preserves_failed_feature_attempt(tmp_path, monkeypatch):
    plan = features.build_plan(suite, video, arguments(tmp_path, features.QWEN, case="single-frame"))
    control = tmp_path / "control.json"
    artifacts = write_json(control, {})
    receipt = write_json(tmp_path / "shared-profile.json", {"context": {
        "name": plan["cases"][features.CONTROL]["profile"]}, "recommendation": {
        "vmm_build_stage": "final", "vmm_resident_weight_budget_bytes": 1024}})
    state = {"identity": {"plan": plan}, "cases": {
        features.CONTROL: {"status": "completed", "artifacts": artifacts},
        "single-frame": {"status": "pending"}}}
    monkeypatch.setattr(features, "control_profile", lambda *_: {"profile_receipt": receipt, "resident_budget_bytes": 1024})
    calls = []
    def failed(command, output, env):
        calls.append(output)
        assert env["DDO_PROFILE_DIR"] == str(control.parent)
        return 1
    manifest = tmp_path / "manifest.json"
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            suite.execute_session(manifest, state, failed, check_child=lambda _: None)
    assert len(calls) == 2 and calls[0] != calls[1] and all(p.is_dir() for p in calls)
    assert state["cases"][features.CONTROL]["status"] == "completed"


def test_own_profile_keeps_control_budget_and_rejects_shape_mismatch(tmp_path, monkeypatch):
    plan = features.build_plan(suite, video, arguments(tmp_path, case="rope-reuse"))
    case = plan["cases"]["rope-reuse"]
    receipt = write_json(tmp_path / "profile.json", {"context": {"name": case["profile"]},
        "recommendation": {"vmm_build_stage": "final", "vmm_resident_weight_budget_bytes": 4 * 2**30}})
    monkeypatch.setattr(features, "control_profile", lambda *_: {"resident_budget_bytes": 671088640})
    effective, command = features.fixed_budget_measurement({}, case, {"profile_receipt": receipt},
                                                           case["stages"][-1]["command"])
    flag = "--ddo-vmm-resident-weight-budget-gb"
    assert command.count(flag) == 1 and command[command.index(flag) + 1] == "0.625"
    assert "feature_fixed_budget_bytes" not in case
    report = {"ddo_profile": {"name": case["profile"], "status": "explicit", "shape_check": "match",
                              "resident_budget_gb": 0.625},
              "dynamic_offload_reports": {"transformer": {"configuration": {
                  "dynamic_offload_vmm_resident_weight_budget_gb": 0.625}}}}
    features.verify_fixed_budget(effective, report)
    for key, value in (("shape_check", "mismatch"), ("resident_budget_gb", 4), ("status", "applied")):
        invalid = deepcopy(report)
        invalid["ddo_profile"][key] = value
        with pytest.raises(ValueError):
            features.verify_fixed_budget(effective, invalid)
    Path(next(iter(receipt))).write_text("changed")
    with pytest.raises(ValueError):
        features.fixed_budget_measurement({}, case, {"profile_receipt": receipt}, command)


def test_linux_control_and_addition_use_exact_bytes_and_reject_rounding(tmp_path, monkeypatch):
    args = arguments(tmp_path, case="rope-reuse")
    args.platform = "linux"
    monkeypatch.setattr(video, "current_platform", lambda: "linux")
    plan = features.build_plan(suite, video, args)
    control = plan["cases"][features.CONTROL]
    budget = 675282944
    receipt = write_json(tmp_path / "control-profile.json", {"context": {"name": control["profile"]},
        "recommendation": {"vmm_build_stage": "final", "vmm_resident_weight_budget_bytes": budget}})
    effective, command = features.fixed_budget_measurement({}, control, {"profile_receipt": receipt},
                                                          control["stages"][-1]["command"])
    flag = "--ddo-vmm-resident-weight-budget-gb"
    assert float(command[command.index(flag)+1]) * 2**30 == budget
    report = {"ddo_profile": {"name": control["profile"], "status": "explicit", "shape_check": "match",
        "resident_budget_gb": budget/2**30},
        "dynamic_offload_reports": {"transformer": {"configuration": {
            "dynamic_offload_vmm_resident_weight_budget_gb": budget/2**30}}}}
    features.verify_fixed_budget(effective, report)
    artifacts = write_json(tmp_path / "control-metrics.json", report)
    state = {"identity": {"plan": plan}, "cases": {features.CONTROL: {
        "status": "completed", "profile_receipt": receipt, "artifacts": artifacts}}}
    assert features.control_profile(state, features.CONTROL)["resident_budget_bytes"] == budget
    addition = plan["cases"]["rope-reuse"]
    own = write_json(tmp_path / "addition-profile.json", {"context": {"name": addition["profile"]},
        "recommendation": {"vmm_build_stage": "final", "vmm_resident_weight_budget_bytes": 4*2**30}})
    added, added_command = features.fixed_budget_measurement(state, addition, {"profile_receipt": own},
                                                            addition["stages"][-1]["command"])
    assert added["feature_fixed_budget_bytes"] == effective["feature_fixed_budget_bytes"] == budget
    assert added_command[added_command.index(flag)+1] == command[command.index(flag)+1]
    report["ddo_profile"]["resident_budget_gb"] = round(budget/2**30, 4)
    write_json(tmp_path / "control-metrics.json", report)
    state["cases"][features.CONTROL]["artifacts"] = {str(tmp_path / "control-metrics.json"): suite.digest(tmp_path / "control-metrics.json")}
    with pytest.raises(ValueError, match="control budget"):
        features.control_profile(state, features.CONTROL)


def test_controller_applies_exact_budget_to_linux_control_before_completion(tmp_path, monkeypatch):
    args = arguments(tmp_path, case=features.CONTROL)
    args.platform = "linux"
    monkeypatch.setattr(video, "current_platform", lambda: "linux")
    plan = features.build_plan(suite, video, args)
    control = plan["cases"][features.CONTROL]
    budget = 675282944
    receipt = write_json(tmp_path / "profile.json", {"context": {"name": control["profile"]},
        "recommendation": {"vmm_build_stage": "final", "vmm_resident_weight_budget_bytes": budget}})
    state = {"identity": {"plan": plan}, "cases": {features.CONTROL: {"status": "pending",
        "profile_receipt": receipt, "profile_build_stage": "final", "stages": {"profile-1": receipt}}}}
    commands = []
    def fail(command, output, env):
        commands.append(command)
        return 1
    manifest = tmp_path / "manifest.json"
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            suite.execute_session(manifest, state, fail, check_child=lambda _: None)
    assert len(commands) == 2
    for command in commands:
        assert float(command[command.index("--ddo-vmm-resident-weight-budget-gb")+1]) * 2**30 == budget
        assert "--build-ddo-profile" not in command
    attempts = list((tmp_path / features.CONTROL).glob("measure-*"))
    assert len(attempts) == 2


@pytest.mark.parametrize("budget", [None, True, 0, -1, 1.25])
def test_linux_control_rejects_unavailable_or_invalid_byte_budget(tmp_path, monkeypatch, budget):
    args = arguments(tmp_path, case=features.CONTROL)
    args.platform = "linux"
    monkeypatch.setattr(video, "current_platform", lambda: "linux")
    control = features.build_plan(suite, video, args)["cases"][features.CONTROL]
    receipt = write_json(tmp_path / "profile.json", {"context": {"name": control["profile"]},
        "recommendation": {"vmm_build_stage": "final", "vmm_resident_weight_budget_bytes": budget}})
    with pytest.raises(ValueError, match="positive exact byte budget"):
        features.fixed_budget_measurement({}, control, {"profile_receipt": receipt},
                                         control["stages"][-1]["command"])


def test_linux_preflight_selects_library_hash_and_never_launches_child(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(video, "current_platform", lambda: "linux")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for name in ("man_guitar.png", "diff_vocal_song.mp3"):
        (fixtures / name).write_bytes(b"CPU identity fixture")
    monkeypatch.setattr(video, "REFERENCE_FIXTURE_SHA256", {
        p.name: suite.digest(p) for p in fixtures.iterdir()})
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (tools / name).write_bytes(b"CPU executable identity fixture")
    monkeypatch.setattr(features.shutil, "which", lambda name, path=None: str(tools / name))
    calls = []
    def identity(plan, expected):
        calls.append((plan, expected))
        return {"plan": plan, "dll_sha256": expected}
    monkeypatch.setattr(suite, "execution_identity", identity)
    monkeypatch.setattr(sys, "argv", ["CPU-preflight", "--platform", "linux", "--recipe", features.VIDEO,
        "--family", "model-features", "--fixtures-dir", str(fixtures), "--ffmpeg-bin", str(tools),
        "--expected-library-sha256", "a"*64, "--preflight"])
    features.main(suite, video, lambda *_: pytest.fail("Preflight launched a child"))
    result = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 and calls[0][1] == "a"*64
    assert result["plan"]["platform"] == "linux"
    assert len(result["fixture_sha256"]) == len(result["ffmpeg_sha256"]) == 2


def test_controller_injects_fixed_budget_only_into_measurement(tmp_path, monkeypatch):
    plan = features.build_plan(suite, video, arguments(tmp_path, case="rope-reuse"))
    case = plan["cases"]["rope-reuse"]
    receipt = write_json(tmp_path / "profiles/profile.json", {"context": {"name": case["profile"]},
                         "recommendation": {"vmm_build_stage": "final"}})
    artifacts = write_json(tmp_path / "control.json", {})
    state = {"identity": {"plan": plan}, "cases": {
        features.CONTROL: {"status": "completed", "artifacts": artifacts},
        "rope-reuse": {"status": "pending", "profile_receipt": receipt, "profile_build_stage": "final",
                       "stages": {"profile-1": receipt}}}}
    monkeypatch.setattr(features, "control_profile", lambda *_: {"resident_budget_bytes": 671088640})
    calls = []
    def failed(command, output, env):
        calls.append(command)
        assert env["DDO_PROFILE_DIR"] == str(tmp_path / "rope-reuse/profiles")
        return 1
    with pytest.raises(subprocess.CalledProcessError):
        suite.execute_session(tmp_path / "manifest.json", state, failed)
    assert len(calls) == 1 and "--build-ddo-profile" not in calls[0]
    flag = "--ddo-vmm-resident-weight-budget-gb"
    assert calls[0][calls[0].index(flag) + 1] == "0.625"
    assert flag not in case["stages"][-1]["command"] or case["stages"][-1]["command"][
        case["stages"][-1]["command"].index(flag) + 1] == "0"


@pytest.mark.parametrize("identical", [True, False])
def test_video_summary_compares_mp4_hash_to_control_and_previous(tmp_path, identical):
    plan = features.build_plan(suite, video, arguments(tmp_path, case="rope-reuse"))
    metrics = {"total_elapsed_sec": 20, "global_peak_vram_absolute_gb": 6,
               "global_peak_ram_gb": 30, "steps": [{"name": "denoise", "elapsed_sec": 10}]}
    old = write_json(tmp_path / "old.json", metrics)
    old_video = tmp_path / "old.mp4"
    old_video.write_bytes(b"container fixture")
    old[str(old_video)] = suite.digest(old_video)
    metrics["total_elapsed_sec"] = 18
    current = write_json(tmp_path / "new.json", metrics)
    new_video = tmp_path / "new.mp4"
    new_video.write_bytes(b"container fixture" if identical else b"different container")
    current[str(new_video)] = suite.digest(new_video)
    state = {"identity": {"plan": plan}, "cases": {features.CONTROL: {
        "status": "completed", "artifacts": old, "summary": suite.measurement_summary(old)}}}
    result = features.feature_summary(state, "rope-reuse", current)["model_features"]
    for key in ("previous", "control"):
        assert result[key]["delta"]["output"]["mp4_sha256_identical"] is identical
        assert result[key]["delta"]["output"]["mp4_sha256"] == suite.digest(new_video)
        assert result[key]["delta"]["output"]["reference_mp4_sha256"] == suite.digest(old_video)
        assert result[key]["delta"]["total_sec"] == -2


@pytest.mark.parametrize("recipe", [features.VIDEO, features.QWEN])
def test_cli_preview_dispatches_without_initializing_cuda(tmp_path, recipe):
    result = subprocess.run([sys.executable, "-I", "-B", "-c",
        "import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); "
        "runpy.run_module('benchmarks.run_benchmark',run_name='__main__')",
        str(suite.ROOT), "--platform", "windows", "--recipe", recipe,
        "--family", "model-features", "--list-cases"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == list(features.case_names(recipe))


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_qwen_nine_cases_preserve_old_combinations_and_reset_branches(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(sys, "argv", ["CPU-parser", "--help"])
    args = arguments(tmp_path, features.QWEN)
    args.platform = platform
    plan = features.build_plan(suite, video, args)
    parser = importlib.import_module("scripts.qwen_image_21.run_dynamic_modular_vmm").parse_args
    assert len(plan["cases"]) == 9
    assert sum(len(case["stages"]) for case in plan["cases"].values()) == 11
    assert plan["cases"][features.CONTROL]["feature_control_exact_budget"] is True
    expected = {
        features.CONTROL: (False, False, False, None, False),
        "single-frame": (True, False, False, None, False),
        "residual-alias": (True, True, False, None, False),
        "direct-padding": (True, True, True, None, False),
        "bands-256": (True, True, True, 256, False),
        "bands-128": (True, True, True, 128, False),
        "bands-256-no-direct-padding": (True, True, False, 256, False),
        "bands-128-no-direct-padding": (True, True, False, 128, False),
        "tiling-only": (False, False, False, None, True),
    }
    fields = ("vae_no_temporal_cache", "vae_alias_residuals", "vae_direct_padding", "vae_band_rows", "vae_tiling")
    for name, case in plan["cases"].items():
        assert tuple(case["model_features"][key] for key in fields) == expected[name]
        assert case["routes"] == {"transformer": "vmm_fast_runtime", "encoder": "native_vmm", "vae": "native_vmm"}
        if name != features.CONTROL:
            assert case["feature_fixed_budget_case"] == case["feature_profile_case"] == features.CONTROL
        for stage in case["stages"]:
            command = stage["command"]
            parsed = parser(command[4:])
            assert tuple(getattr(parsed, key) for key in fields) == expected[name]
            assert parsed.vae_vmm_offload is True and parsed.vae_vmm_resident_weight_budget_gb == 0.375
            assert parsed.text_encoder_native_vmm is True
            assert parsed.ddo_vmm_total_staging_gb == 0.625
            assert parsed.ddo_vmm_source_cache_gb == parsed.ddo_vmm_source_registered_gb == 16
            assert not any(flag.startswith("--purge-") for flag in command)
        actual = {**case["model_features"], "vae_band_blocks": 2 if expected[name][3] else None,
                  "vae_class": "QwenImage21ImageVAE" if expected[name][0] else "AutoencoderKLQwenImage21"}
        features.verify_features(features.QWEN, case, {"executed_controls": actual})
    assert plan["cases"]["bands-128"]["previous_case"] == "direct-padding"
    for name in ("bands-256-no-direct-padding", "bands-128-no-direct-padding"):
        assert plan["cases"][name]["previous_case"] == "residual-alias"
    assert plan["cases"]["tiling-only"]["previous_case"] == features.CONTROL
    actual["vae_class"] = "QwenImage21ImageVAE"
    with pytest.raises(ValueError, match="decoder class"):
        features.verify_features(features.QWEN, plan["cases"]["tiling-only"], {"executed_controls": actual})


def qwen_feature_report(case, budget):
    phases = [{"name": name, "elapsed_sec": latency, "peak_vram_absolute_gb": 2,
               "peak_ram_gb": 4, "ram_metric": "rss", "memory_monitor": {
                   "ram_metric": "rss", "samples": 1, "error_type": None}}
              for name, latency in (("denoise", 5), ("decode", 1))]
    return {"status": "completed", "workload": {"width": 1024, "height": 1024,
        "seed": 42, "steps_requested": 20}, "transformer_attention_requested": "native",
        "total_elapsed_sec": 10, "global_peak_vram_absolute_gb": 2, "global_peak_ram_gb": 4,
        "memory_monitor": {"ram_metric": "rss"}, "standby_purges": {}, "steps": phases,
        "ddo_profile": {"name": case["profile"], "status": "explicit", "shape_check": "match",
                        "resident_budget_gb": budget/2**30},
        "dynamic_offload_reports": {name: {"route": "dynamic_offload", "configuration": {
            "dynamic_offload_execution_mode": "vmm_fast_runtime",
            "dynamic_offload_vmm_resident_weight_budget_gb": budget/2**30},
            "runtime": {"vmm": {"fast_source_cache_source_dispatcher_enabled": 1}}}
            for name in ("transformer", "text_encoder", "vae")},
        "executed_controls": {**case["model_features"], "vae_offload": "native_vmm_component",
            "vae_class": "QwenImage21ImageVAE" if case["model_features"]["vae_no_temporal_cache"] else "AutoencoderKLQwenImage21",
            "vae_band_blocks": 2 if case["model_features"]["vae_band_rows"] else None}}


def test_qwen_explicit_shared_profile_completes_and_resume_skips_completed(tmp_path, monkeypatch):
    from PIL import Image
    args = arguments(tmp_path, features.QWEN, case="single-frame")
    args.platform = "linux"
    plan = features.build_plan(suite, video, args)
    control = plan["cases"][features.CONTROL]
    budget = 675282944
    receipt = write_json(tmp_path / "profiles/profile.json", {"context": {"name": control["profile"]},
        "recommendation": {"vmm_build_stage": "final", "vmm_resident_weight_budget_bytes": budget}})
    control_metrics = tmp_path / "control.json"
    artifacts = write_json(control_metrics, qwen_feature_report(control, budget))
    picture = tmp_path / "control.png"
    Image.new("RGB", (1024, 1024), "white").save(picture)
    artifacts[str(picture)] = suite.digest(picture)
    state = {"identity": {"plan": plan}, "cases": {features.CONTROL: {
        "status": "completed", "artifacts": artifacts, "profile_receipt": receipt,
        "summary": suite.measurement_summary(artifacts)}, "single-frame": {"status": "pending"}}}
    calls = []
    def run(command, output, env):
        calls.append(command)
        assert "--build-ddo-profile" not in command
        assert float(command[command.index("--ddo-vmm-resident-weight-budget-gb")+1])*2**30 == budget
        assert env["DDO_PROFILE_DIR"] == str(tmp_path / "profiles")
        write_json(output / "metrics.json", qwen_feature_report(plan["cases"]["single-frame"], budget))
        Image.new("RGB", (1024, 1024), "white").save(output / "image.png")
        return 0
    def verify(case, stage, output, profiles):
        features.verify_features(features.QWEN, case, suite.read_json(output / "metrics.json"))
        return suite.verify_stage(suite.RECIPES[features.QWEN], case, stage, output, profiles)
    manifest = tmp_path / "manifest.json"
    suite.execute_session(manifest, state, run, verify=verify)
    assert state["cases"]["single-frame"]["status"] == "completed"
    assert state["cases"]["single-frame"]["profile_receipt"] == receipt
    suite.execute_session(manifest, state, run, verify=verify)
    assert len(calls) == 1
    changed = deepcopy(plan["cases"]["single-frame"])
    with pytest.raises(ValueError, match="shared feature profile changed"):
        features.fixed_budget_measurement(state, changed, {"profile_receipt": artifacts}, calls[0])


def test_qwen_branch_summary_compares_explicit_parent_and_control(tmp_path):
    from PIL import Image
    plan = features.build_plan(suite, video, arguments(tmp_path, features.QWEN))
    state = {"identity": {"plan": plan}, "cases": {}}
    all_artifacts = {}
    for name in (features.CONTROL, "residual-alias", "bands-256-no-direct-padding"):
        case = plan["cases"][name]
        artifacts = write_json(tmp_path / name / "metrics.json", qwen_feature_report(case, 675282944))
        picture = tmp_path / name / "image.png"
        Image.new("RGB", (1024, 1024), "white").save(picture)
        artifacts[str(picture)] = suite.digest(picture)
        all_artifacts[name] = artifacts
        state["cases"][name] = {"status": "completed", "artifacts": artifacts,
                                 "summary": suite.measurement_summary(artifacts)}
    result = features.feature_summary(state, "bands-256-no-direct-padding",
                                     all_artifacts["bands-256-no-direct-padding"])["model_features"]
    assert result["parent"]["case"] == "residual-alias"
    assert result["control"]["case"] == features.CONTROL
    assert "previous" not in result


def test_qwen_linux_preflight_hashes_vae_source_without_loading_tensors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(video, "current_platform", lambda: "linux")
    source = tmp_path / "bf16-vae"
    source.mkdir()
    (source / "config.json").write_text('{}')
    (source / "diffusion_pytorch_model.safetensors").write_bytes(b"hash-only CPU fixture")
    monkeypatch.setattr(suite, "execution_identity", lambda plan, expected: {
        "plan": plan, "dll_sha256": expected})
    monkeypatch.setattr(sys, "argv", ["CPU-Qwen-preflight", "--platform", "linux", "--recipe", features.QWEN,
        "--family", "model-features", "--vae-model-path", str(source),
        "--expected-library-sha256", "b"*64, "--preflight"])
    features.main(suite, video, lambda *_: pytest.fail("CPU preflight launched inference"))
    result = json.loads(capsys.readouterr().out)
    assert len(result["plan"]["cases"]) == 9
    assert result["dll_sha256"] == "b"*64
    assert result["vae_source"] == {str(p): suite.digest(p) for p in source.iterdir()}
