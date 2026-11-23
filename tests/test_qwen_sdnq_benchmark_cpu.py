"""Validate Qwen packed-runtime recipes and acceptance without loading a model."""

import ast
import json
import sys

from PIL import Image
import pytest

from benchmarks import image_suite as suite
from benchmarks import sdnq_suite as sdnq
from benchmarks import qwen_sdnq_suite as qwen
from tests.test_sdnq_benchmark_cpu import options, write_profile


def plan():
    return sdnq.build_plan(suite, options(recipe="qwen-image21"))


def fixture(case, stage="measure"):
    leaf = case["profile"] is None
    calibration = stage != "measure"
    count = 2 if calibration else 20
    return {"status": "completed", "variant": "base", "model": case["checkpoint"],
            "transformer_source": case["sdnq_checkpoint"], "backend": "linear",
            "requested_preset": case["preset"], "transformer_runtime": "plan" if leaf else "sdnq_runtime",
            "executed_steps": count, "transformer_attention_requested": "native",
            "build_ddo_profile": stage == "profile-1", "validate_ddo_profile": stage == "profile-2",
            "workload": {"width": 1024, "height": 1024, "width_actual": 1024, "height_actual": 1024,
                         "steps_requested": 20, "seed": 42, "prompt": case["prompt"], "dtype": "bfloat16",
                         "transformer_storage": "saved_sdnq", "true_cfg_scale": 1, "use_kv_cache": True,
                         "vae_tiling": False},
            "quantization_recipe": {"quant_method": "sdnq", "weights_dtype": "uint4"},
            "quantized_backend": {"sdnq_module_count": 10},
            "requested_controls": {"text_encoder_leaf_offload": True, "vae_leaf_offload": True},
            "memory_monitor": {"ram_metric": "rss"},
            "denoise_callbacks": [{"step": i + 1, "callback_interval_sec": 1} for i in range(count)],
            "denoise_through_cpu_readback_sec": count,
            "dynamic_offload_reports": {key: {"route": "dynamic_offload" if key == "transformer" and not leaf else "diffusers_group_offload",
                                                "configuration": {"dynamic_offload_execution_mode": "sdnq_runtime"}}
                                        for key in ("text_encoder", "transformer", "vae")},
            "steps": [{"name": key} for key in ("encode", "denoise", "cleanup_transformer") +
                      (() if calibration else ("load_vae", "decode", "cleanup_vae", "save_image"))],
            "executed_controls": {"vae_offload": "diffusers_leaf"},
            "total_elapsed_sec": 30, "global_peak_vram_absolute_gb": 7, "global_peak_ram_gb": 20,
            "ddo_profile": None if leaf else {"name": case["profile"], "status": "applied", "resident_budget_gb": 2}}


def test_real_parser_and_equal_auxiliary_routes(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from scripts.qwen_image_21.run_dynamic_modular_sdnq import parse_args
    value = plan()
    assert value["sdnq_revision"] == qwen.REVISION
    for case in value["cases"].values():
        assert case["routes"]["encoder"] == case["routes"]["vae"] == "diffusers_leaf"
        for stage in case["stages"]:
            args = parse_args(stage["command"][4:])
            assert (args.width, args.height, args.steps, args.seed) == (1024, 1024, 20, 42)
            assert args.text_encoder_leaf_offload and args.vae_leaf_offload
            assert args.ddo_profile == case["profile"]
            assert not args.vae_no_temporal_cache and not args.vae_tiling
            assert args.purge_before_transformer is False and args.purge_before_run is False
            assert args.build_ddo_profile == (stage["name"] == "profile-1")
            assert args.validate_ddo_profile == (stage["name"] == "profile-2")
    with pytest.raises(SystemExit): parse_args(["--validate-ddo-profile"])
    with pytest.raises(SystemExit): parse_args(["--ddo-preset", "diffusers_leaf_offload_compat", "--ddo-profile", "wrong"])
    assert not torch.cuda.is_initialized()


def test_omitted_purge_flags_override_legacy_installed_preset(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized"))
    from scripts.qwen_image_21.run_dynamic_modular_sdnq import parse_args
    from scripts.helpers.runner_controls import purge_for_runner
    import diffusers_dynamic_offloader as ddo
    observed = []
    def legacy_policy(*args, enabled=None, **kwargs):
        observed.append(enabled)
        return {"purged": True if enabled is None else enabled}
    monkeypatch.setattr(ddo, "maybe_purge_windows_standby_cache", legacy_policy)
    args = parse_args(["--ddo-preset", "one_shot_fast"])
    for phase in ("before_run", "after_text_encoder", "before_transformer", "after_run"):
        assert purge_for_runner(args, phase)["purged"] is False
    assert observed == [False] * 4
    args = parse_args(["--purge-before-transformer"])
    assert purge_for_runner(args, "before_transformer")["purged"] is True


@pytest.mark.parametrize("change", ["encoder", "runtime", "purge", "steps", "feature", "png", "budget", "pending"])
def test_measurement_requires_complete_matched_evidence(tmp_path, change):
    case = plan()["cases"][sdnq.CASES[1]]
    data = fixture(case)
    saved = tmp_path / "profiles/profile.json"
    write_profile(saved, case)
    Image.new("RGB", (1024, 1024)).save(tmp_path / "image.png")
    def verify():
        (tmp_path / "metrics.json").write_text(json.dumps(data))
        return sdnq.verify_stage(suite, case, "measure", tmp_path, saved.parent)
    assert verify()
    if change == "encoder": data["dynamic_offload_reports"]["text_encoder"]["route"] = "dynamic_offload"
    elif change == "runtime": data["transformer_runtime"] = "linear_runtime"
    elif change == "purge": data["standby_purges"] = {"before_run": {"requested": True}}
    elif change == "steps": data["denoise_callbacks"].pop()
    elif change == "feature": data["requested_controls"]["vae_no_temporal_cache"] = True
    elif change == "png": (tmp_path / "image.png").write_bytes(b"broken")
    elif change == "budget": data["ddo_profile"]["resident_budget_gb"] = 3
    else: write_profile(saved, case, sdnq_probe_resident_budget_gb=2)
    with pytest.raises((ValueError, OSError)): verify()


def test_calibration_stops_before_decode_and_uses_own_profile(tmp_path):
    case = plan()["cases"][sdnq.CASES[1]]
    data = fixture(case, "profile-1")
    saved = tmp_path / "profiles/profile.json"
    write_profile(saved, case, sdnq_probe_resident_budget_gb=2)
    data["ddo_profile"].update(status="saved", path=str(saved), build_stage="probe_pending",
                               profile_measurement_kind="sdnq_baseline_probe_pending")
    def verify():
        (tmp_path / "metrics.json").write_text(json.dumps(data))
        return sdnq.verify_stage(suite, case, "profile-1", tmp_path, saved.parent)
    assert verify()
    data["steps"].append({"name": "load_vae"})
    with pytest.raises(ValueError, match="decoded"): verify()


def test_runner_retires_transformer_owners_before_early_return_and_decode():
    tree = ast.parse((suite.ROOT / "scripts/qwen_image_21/run_dynamic_modular_sdnq.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    deleted = next(n.lineno for n in ast.walk(main) if isinstance(n, ast.Delete)
                   and {"result", "transformer", "prepared", "denoise"}.issubset(
                       {v.id for v in n.targets if isinstance(v, ast.Name)}))
    early = next(n.lineno for n in ast.walk(main) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == "profile_only"
                 and any(isinstance(v, ast.Return) for v in n.body))
    load = next(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)
                and any(isinstance(v, ast.Constant) and v.value == "load_vae" for v in n.args))
    assert deleted < early < load
