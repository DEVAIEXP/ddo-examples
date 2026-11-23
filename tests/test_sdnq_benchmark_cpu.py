"""Exercise saved-INT4 benchmark contracts without model loading or CUDA."""

import argparse
import ast
import json
import subprocess
from pathlib import Path
import sys

from PIL import Image
import pytest

from benchmarks import image_suite as suite
from benchmarks import sdnq_suite as sdnq


def options(**changes):
    return argparse.Namespace(**{"platform": "windows", "recipe": "ltx23-image", "family": "precision",
                                 "case": None, "checkpoint": None, "custom_blocks": None,
                                 "sdnq_model_path": None, **changes})


def write_profile(path, case, **recommendation):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"context": {"name": case["profile"], "execution_mode": "sdnq_runtime"},
        "recommendation": {"profile_resident_module_budget_gb": 2, **recommendation}}))


def fixture(case, stage="measure"):
    leaf = case["profile"] is None
    return {"width": 1280, "height": 704, "seed": 43, "num_inference_steps": 8,
            "active_inference_steps": 8, "sdnq_enabled": True, "sdnq_bits": 4,
            "sdnq_text_encoder_bits": None, "text_encoder_kind": "original", "dtype": "torch.bfloat16",
            "guidance_scale": 1, "guidance_rescale": 0.7, "pag_enabled": False, "lora_enabled": False,
            "offload_backend": "diffusers_leaf" if leaf else "ddo_sdnq_runtime", "vae_decode_route": "resident",
            "sdnq_model_path": case["sdnq_checkpoint"], "ddo_preset": case["preset"], "build_ddo_profile": False,
            "validate_ddo_profile": False,
            "transformer_attention_requested": "native", "memory_monitor": {"ram_metric": "rss"},
            "denoise_step_times": [1.0] * 8, "total_elapsed_sec": 20, "global_peak_vram_absolute_gb": 6,
            "global_peak_ram_gb": 10, "ddo_profile": {"name": case["profile"], "status": "applied", "resident_budget_gb": 2},
            "dynamic_offload_reports": {"transformer": {"route": "diffusers_group_offload" if leaf else "dynamic_offload",
                "configuration": {"dynamic_offload_execution_mode": "plan" if leaf else "sdnq_runtime"}}},
            "events": [{"name": "setup_text_encoder_group_offload"}, {"name": "load_vae_to_cuda"}]}


def test_pair_plan_and_real_parser(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized"))
    from scripts.ltx2_image.run_dynamic_modular_sdnq_distilled import parse_args
    plan = sdnq.build_plan(suite, options())
    assert list(plan["cases"]) == list(sdnq.CASES)
    for case in plan["cases"].values():
        for stage in case["stages"]:
            monkeypatch.setattr(sys, "argv", stage["command"][3:])
            args = parse_args()
            assert args.bits == 4 and args.text_encoder_bits is None
            assert args.local_files_only and args.image_attention == "default"
            assert args.ddo_profile == case["profile"]
            assert not args.purge_before_run and not args.purge_before_transformer
            assert args.build_ddo_profile == (stage["name"] == "profile-1")
            assert args.validate_ddo_profile == (stage["name"] == "profile-2")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("change", ["runtime", "profile", "purge", "png", "bits", "timings", "vae"])
def test_measurement_rejects_changed_or_incomplete_evidence(tmp_path, change):
    case = sdnq.build_plan(suite, options())["cases"][sdnq.CASES[1]]
    data = fixture(case)
    write_profile(tmp_path / "profiles/profile.json", case)
    Image.new("RGB", (1280, 704)).save(tmp_path / "image.png")
    def verify():
        (tmp_path / "metrics.json").write_text(json.dumps(data))
        return sdnq.verify_stage(suite, case, "measure", tmp_path, tmp_path / "profiles")
    assert verify()
    if change == "runtime": data["dynamic_offload_reports"]["transformer"]["configuration"]["dynamic_offload_execution_mode"] = "linear_runtime"
    elif change == "profile": data["ddo_profile"]["name"] = "BF16-profile"
    elif change == "purge": data["standby_purge"] = {"purged": True}
    elif change == "png": (tmp_path / "image.png").write_bytes(b"broken")
    elif change == "bits": data["sdnq_bits"] = 8
    elif change == "timings": data["denoise_step_times"].pop()
    else: data["vae_decode_route"] = "leaf"
    with pytest.raises((ValueError, OSError)): verify()


def test_profile_contract_and_calibration_does_not_decode(tmp_path):
    case = sdnq.build_plan(suite, options())["cases"][sdnq.CASES[1]]
    data = fixture(case)
    data.update(build_ddo_profile=True, active_inference_steps=2, denoise_step_times=[1, 1],
                events=[{"name": "setup_text_encoder_group_offload"}])
    saved = tmp_path / "profiles/profile.json"
    write_profile(saved, case)
    data["ddo_profile"].update(status="saved", path=str(saved), build_stage="final",
                               profile_measurement_kind="sdnq_baseline_safe")
    metrics = tmp_path / "metrics.json"; metrics.write_text(json.dumps(data))
    assert sdnq.verify_stage(suite, case, "profile-1", tmp_path, saved.parent)
    saved.write_text(json.dumps({"context": {"execution_mode": "vmm_fast_runtime"}}))
    with pytest.raises(ValueError, match="BF16 or VMM"):
        sdnq.verify_stage(suite, case, "profile-1", tmp_path, saved.parent)


def test_pair_resume_and_output_comparison(tmp_path):
    plan = sdnq.build_plan(suite, options())
    # Shared profile build/export contracts are exercised separately.
    for case in plan["cases"].values():
        case["profile"] = None
        case["stages"] = [case["stages"][-1]]
    state = {"identity": {"plan": plan}, "cases": {name: {"status": "pending"} for name in plan["cases"]}}
    calls = []
    def run(command, output, env):
        calls.append(output)
        if len(calls) == 2: return 1
        Image.new("RGB", (1280, 704)).save(output / "image.png")
        (output / "metrics.json").write_text(json.dumps({"total_elapsed_sec": 1,
            "global_peak_vram_absolute_gb": 1, "global_peak_ram_gb": 1}))
        return 0
    def verify(case, stage, output, profiles):
        return {str(p): suite.digest(p) for p in output.iterdir()}
    path = tmp_path / "manifest.json"
    with pytest.raises(subprocess.CalledProcessError): suite.execute_session(path, state, run, verify=verify)
    assert state["cases"][sdnq.CASES[0]]["status"] == "completed"
    failed = calls[-1]
    suite.execute_session(path, state, run, verify=verify, check_child=lambda _: None)
    assert len(calls) == 3 and calls[-1] != failed and failed.exists()
    assert state["cases"][sdnq.CASES[1]]["summary"]["output_difference"]["rgb_identical"]


def test_pending_probe_must_finish_before_measurement(tmp_path):
    case = sdnq.build_plan(suite, options())["cases"][sdnq.CASES[1]]
    data = fixture(case)
    saved = tmp_path / "profiles/profile.json"
    write_profile(saved, case, sdnq_probe_resident_budget_gb=2)
    Image.new("RGB", (1280, 704)).save(tmp_path / "image.png")
    def verify():
        (tmp_path / "metrics.json").write_text(json.dumps(data))
        return sdnq.verify_stage(suite, case, "measure", tmp_path, saved.parent)
    with pytest.raises(ValueError, match="still pending"): verify()
    write_profile(saved, case)
    assert verify()
    data["ddo_profile"]["resident_budget_gb"] = 3
    with pytest.raises(ValueError, match="budget differs"): verify()


def test_runner_releases_owning_result_before_decode():
    source = suite.ROOT / "scripts/ltx2_image/run_dynamic_modular_sdnq_distilled.py"
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    reset = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "transformer_offload" for target in node.targets)
                 and isinstance(node.value, ast.Constant) and node.value.value is None)
    cleanup = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Call)
                   and any(isinstance(arg, ast.Constant) and arg.value == "cleanup_before_vae_decode" for arg in node.args))
    assert reset < cleanup
