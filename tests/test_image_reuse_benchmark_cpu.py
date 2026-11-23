"""Validate the repeated-process recipe without running model or CUDA work."""

import argparse
import ast
import gc
import json
from pathlib import Path
import sys
import weakref
from types import SimpleNamespace

from PIL import Image
import pytest

from benchmarks import image_suite as suite


def test_previous_transformer_is_collected_before_next_preparation():
    """Exercise the runner loop with a cyclic model and an owning offload result."""
    path = suite.ROOT / "scripts/ltx2_image/run_dynamic_modular_distilled.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "prepare_repeat_index")
    refs = []

    class Model:
        def __init__(self):
            self.cycle = self

    def prepare(index):
        if index:
            assert refs[0]() is None, "Previous result retained the retired transformer"
        model = Model()
        refs.append(weakref.ref(model))
        return model, object(), SimpleNamespace(module=model)

    def cleanup(*_):
        gc.collect()
        assert refs[-1]() is None, "Cleanup still sees the retired model"

    namespace = dict(TRANSFORMER_PREPARE_REPEATS=2, dynamic_offload_enabled=True,
                     load_and_prepare_transformer=prepare, remove_dynamic_offload=lambda _: None,
                     cleanup_runtime_state=cleanup, record_event=None,
                     transformer_prepare_event_name=lambda *args: args)
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), namespace)
    assert refs[1]() is namespace["transformer"] is namespace["transformer_offload"].module


def options(**changes):
    return argparse.Namespace(**{"platform": "windows", "recipe": "ltx23-image", "family": "reuse",
                                 "case": "warm-process", "checkpoint": None, "custom_blocks": None,
                                 "nvme_resident_gib": 1.5, **changes})


def selected_case():
    return suite.build_plan(options())["cases"]["warm-process"]


def report(case):
    data = {"width": 1280, "height": 704, "seed": 43, "num_inference_steps": 8,
            "dynamic_offload_execution_mode": "linear_runtime", "dynamic_offload_effective_preset": "warm_process",
            "text_encoder_group_offload": True, "text_encoder_dynamic_offload": False,
            "dtype": "torch.bfloat16", "guidance_scale": 1, "guidance_rescale": 0.7, "pag_enabled": False,
            "ddo_profile": {"name": case["profile"], "status": "applied"},
            "generation_repeats": 2, "transformer_prepare_repeats": 2,
            "transformer_attention_requested": "native", "dynamic_offload_cache_plan": True,
            "dynamic_offload_cache_pinned_weights": True, "vae_decode_route": "resident",
            "denoise_step_times_by_repeat": [[0.1] * 8, [0.2] * 8],
            "total_elapsed_sec": 30, "global_peak_vram_absolute_gb": 7,
            "global_peak_ram_gb": 28, "memory_monitor": {"ram_metric": "rss"}, "events": []}
    for i in (1, 2):
        for name, latency in ((f"load_transformer_prepare_{i}", 1),
                              (f"build_dynamic_offload_plan_prepare_{i}", 2),
                              (f"denoise_modular_pipe_call_repeat_{i}", 3),
                              (f"prepare_through_cpu_readback_repeat_{i}", 4)):
            data["events"].append({"name": name, "elapsed_sec": latency})
    return data


def test_reuse_plan_calibrates_once_then_repeats_without_policy_overrides():
    case = selected_case()
    assert case["preset"] == "warm_process" and case["dispatcher"] is None
    assert case["routes"] == {"transformer": "linear_runtime", "encoder": "diffusers_leaf", "vae": "resident"}
    assert case["profile"] == "ltx23-image-1280x704-8steps-warm-process"
    assert [stage["name"] for stage in case["stages"]] == ["profile-1", "measure"]
    for stage in case["stages"]:
        cmd = stage["command"]
        assert "--vae-leaf-offload" not in cmd
        assert not any(flag.startswith("--purge-") for flag in cmd)
        assert not any("budget" in flag for flag in cmd)
        assert ("--warm-process-reuse" in cmd) == (stage["name"] == "measure")
    assert "warm-process" not in suite.build_plan(options(family="runtime", case=None))["cases"]


@pytest.mark.parametrize("changes", [{"platform": "linux"}, {"platform": "wsl"},
                                     {"recipe": "qwen-image21"}, {"purge_before_transformer_only": True}])
def test_reject_unprepared_reuse_variants(changes):
    with pytest.raises(ValueError):
        suite.build_plan(options(**changes))


def test_real_runner_parser_preserves_defaults_and_limits_reuse(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU test initialized CUDA"))
    from scripts.ltx2_image.run_dynamic_modular_distilled import parse_args

    for stage in selected_case()["stages"]:
        monkeypatch.setattr(sys, "argv", stage["command"][3:])
        parsed = parse_args()
        expected = 1 if stage["name"] == "profile-1" else 2
        assert parsed.generation_repeats == parsed.transformer_prepare_repeats == expected
        assert parsed.ddo_preset == "warm_process" and not parsed.vae_leaf_offload
    for extra in ([], ["--ddo-preset", "one_shot_fast"],
                  ["--ddo-preset", "warm_process", "--ddo-profile", "reuse", "--build-ddo-profile"]):
        monkeypatch.setattr(sys, "argv", ["runner", "--warm-process-reuse", *extra])
        with pytest.raises(SystemExit):
            parse_args()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("change", ["count", "event", "duplicate", "profile", "cache", "attention", "steps", "nonfinite", "route"])
def test_reuse_verifier_rejects_incomplete_or_changed_measurements(tmp_path, change):
    case = selected_case(); data = report(case)
    Image.new("RGB", (1280, 704)).save(tmp_path / "image.png")
    def verify():
        (tmp_path / "metrics.json").write_text(json.dumps(data))
        return suite.verify_stage(suite.RECIPES["ltx23-image"], case, "measure", tmp_path, tmp_path)
    assert verify()
    if change == "count": data["generation_repeats"] = 1
    elif change == "event": data["events"].pop()
    elif change == "duplicate": data["events"].append(data["events"][0])
    elif change == "profile": data["ddo_profile"]["name"] = "one-shot-fast"
    elif change == "cache": data["dynamic_offload_cache_pinned_weights"] = False
    elif change == "attention": data["transformer_attention_requested"] = "flash"
    elif change == "steps": data["denoise_step_times_by_repeat"][1].pop()
    elif change == "nonfinite": data["events"][0]["elapsed_sec"] = float("nan")
    else: data["vae_decode_route"] = "diffusers_leaf"
    with pytest.raises(ValueError): verify()


def test_reuse_resume_preserves_failed_attempt_and_skips_verified_result(tmp_path, monkeypatch):
    plan = suite.build_plan(options()); case = plan["cases"]["warm-process"]
    # Profile file handling is covered by the shared controller tests; isolate repeat receipts here.
    case["profile"] = None
    case["stages"] = [case["stages"][-1]]
    state = {"identity": {"plan": plan}, "cases": {"warm-process": {"status": "pending"}}}
    path = tmp_path / "manifest.json"; attempts = []
    def run(command, output, env):
        attempts.append(output)
        data = report(case)
        if len(attempts) == 1: data["generation_repeats"] = 1
        (output / "metrics.json").write_text(json.dumps(data))
        Image.new("RGB", (1280, 704)).save(output / "image.png")
    with pytest.raises(ValueError, match="generation count"):
        suite.execute_session(path, state, run)
    assert attempts[0].is_dir()
    state = suite.read_json(path)
    suite.execute_session(path, state, run, check_child=lambda _: None)
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    summary = state["cases"]["warm-process"]["summary"]
    assert summary["reuse"]["decoded_outputs"] == 1
    assert len(summary["reuse"]["repeats"]) == 2
    assert summary["reuse"]["cache_reuse_observed"] is None
    suite.execute_session(path, state, lambda *_: pytest.fail("Completed reuse ran again"))
