"""Qwen saved-INT4 recipes and evidence checks; no model imports in the controller."""

import argparse
import math
from pathlib import Path
import sys

if __package__:
    from .sdnq_suite import CASES, verify_profile
else:
    from sdnq_suite import CASES, verify_profile

MODEL = "OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit"
REVISION = "59d4c3610379d91fb023c4d0b6527c76755491c4"
PROFILE = "qwen-image21-1024x1024-20steps-sdnq-int4"


def build_plan(suite, args):
    if args.platform != "windows":
        raise ValueError("This SDNQ pair is prepared only for Windows.")
    base = argparse.Namespace(**{**vars(args), "family": "runtime", "case": "one-shot-fast",
                                "nvme_resident_gib": 1.5})
    plan = suite.build_plan(base)
    checkpoint = args.sdnq_model_path or Path("F:/hf_cache/hub") / (
        "models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    command = [sys.executable, "-B", "-u", str(suite.ROOT / "scripts/qwen_image_21/run_dynamic_modular_sdnq.py"),
               "--auxiliary-model", plan["checkpoint"], "--model-path", str(checkpoint),
               "--custom-blocks", plan["custom_blocks"], "--local-files-only", "--variant", "base",
               "--prompt", suite.RECIPES[args.recipe]["prompt"], "--width", "1024", "--height", "1024",
               "--steps", "20", "--seed", "42", "--image-joint-attention", "default",
               "--text-encoder-leaf-offload", "--vae-leaf-offload", "--memory-monitor", "rss",
               "--metrics-level", "2", "--save-metrics"]
    cases = {}
    for name in CASES:
        if args.case and args.case != name:
            continue
        leaf = name == CASES[0]
        preset = "diffusers_leaf_offload_compat" if leaf else "one_shot_fast"
        cmd = command + ["--ddo-preset", preset]
        stages = []
        if not leaf:
            cmd += ["--ddo-profile", PROFILE]
            stages = [{"name": "profile-1", "command": cmd + ["--build-ddo-profile"]},
                      {"name": "profile-2", "command": cmd + ["--validate-ddo-profile"]}]
        stages.append({"name": "measure", "command": cmd})
        cases[name] = {"recipe": args.recipe, "preset": preset, "profile": None if leaf else PROFILE,
                       "hold": None, "dispatcher": None, "sdnq_checkpoint": str(checkpoint),
                       "checkpoint": plan["checkpoint"], "custom_blocks": plan["custom_blocks"],
                       "prompt": suite.RECIPES[args.recipe]["prompt"],
                       "routes": {"transformer": "official_diffusers_leaf" if leaf else "sdnq_runtime",
                                  "encoder": "diffusers_leaf", "vae": "diffusers_leaf"}, "stages": stages}
        if not leaf and not args.case:
            cases[name]["png_reference_case"] = CASES[0]
    plan.update(family="precision", cases=cases, sdnq_checkpoint=str(checkpoint), sdnq_revision=REVISION)
    return plan


def verify_stage(suite, case, stage, output, profile_dir):
    path = suite.metrics_file(output)
    report = suite.read_json(path)
    leaf = case["profile"] is None
    calibration = stage.startswith("profile-")
    count = 2 if calibration else 20
    finite = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
    expected = {"status": "completed", "variant": "base", "model": case["checkpoint"],
                "transformer_source": case["sdnq_checkpoint"], "backend": "linear",
                "requested_preset": case["preset"], "transformer_runtime": "plan" if leaf else "sdnq_runtime",
                "executed_steps": count, "transformer_attention_requested": "native",
                "build_ddo_profile": stage == "profile-1", "validate_ddo_profile": stage == "profile-2"}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Qwen SDNQ recipe mismatch: {key}")
    workload = report.get("workload", {})
    for key, value in {"width": 1024, "height": 1024, "width_actual": 1024, "height_actual": 1024,
                       "steps_requested": 20, "seed": 42, "prompt": case["prompt"], "dtype": "bfloat16",
                       "transformer_storage": "saved_sdnq", "true_cfg_scale": 1.0,
                       "use_kv_cache": True, "vae_tiling": False}.items():
        if workload.get(key) != value:
            raise ValueError(f"Qwen SDNQ workload mismatch: {key}")
    recipe = report.get("quantization_recipe", {})
    if recipe.get("quant_method") != "sdnq" or recipe.get("weights_dtype") != "uint4":
        raise ValueError("Expected saved SDNQ INT4 weights.")
    if not report.get("quantized_backend", {}).get("sdnq_module_count", 0):
        raise ValueError("Packed SDNQ modules were not observed.")
    controls = report.get("requested_controls", {})
    if not controls.get("text_encoder_leaf_offload") or not controls.get("vae_leaf_offload"):
        raise ValueError("Both auxiliary components must explicitly use Leaf.")
    for key in ("vae_tiling", "vae_no_temporal_cache", "vae_alias_residuals", "vae_direct_padding", "compile"):
        if controls.get(key):
            raise ValueError(f"Unexpected model optimization: {key}")
    if report.get("memory_monitor", {}).get("ram_metric") != "rss":
        raise ValueError("Expected RSS monitoring.")
    callbacks = report.get("denoise_callbacks", [])
    if len(callbacks) != count or any(row.get("step") != i + 1 or not finite(row.get("callback_interval_sec"))
                                      for i, row in enumerate(callbacks)):
        raise ValueError("Incomplete Qwen SDNQ step timings.")
    if not finite(report.get("denoise_through_cpu_readback_sec")):
        raise ValueError("Missing denoise completion timing.")
    reports = report.get("dynamic_offload_reports", {})
    for component in ("text_encoder", "transformer") + (() if calibration else ("vae",)):
        value = reports.get(component, {})
        route = "dynamic_offload" if component == "transformer" and not leaf else "diffusers_group_offload"
        if value.get("route") != route:
            raise ValueError(f"Unexpected component route: {component}")
        if component == "transformer" and not leaf and value.get("configuration", {}).get("dynamic_offload_execution_mode") != "sdnq_runtime":
            raise ValueError("The packed SDNQ runtime was not observed.")
    for key in ("purged", "requested"):
        if any(value is True for value in suite.walk_field(report.get("standby_purges", {}), key)):
            raise ValueError("The SDNQ pair must not request standby purge.")
    verify_profile(suite, case, stage, report, profile_dir)
    artifacts = [path]
    pngs = list(output.glob("*.png"))
    phases = {row.get("name") for row in report.get("steps", [])}
    if calibration:
        if pngs or phases.intersection({"load_vae", "decode", "save_image"}):
            raise ValueError("Calibration unexpectedly decoded an image.")
    else:
        from PIL import Image
        if not {"encode", "denoise", "decode", "save_image", "cleanup_transformer", "cleanup_vae"}.issubset(phases):
            raise ValueError("Incomplete Qwen phase metrics.")
        if report.get("executed_controls", {}).get("vae_offload") != "diffusers_leaf":
            raise ValueError("Expected Leaf VAE decode.")
        if len(pngs) != 1:
            raise ValueError("Expected one Qwen PNG.")
        with Image.open(pngs[0]) as image:
            if image.size != (1024, 1024):
                raise ValueError("Unexpected output image shape.")
            image.verify()
        artifacts += pngs
        for key in ("total_elapsed_sec", "global_peak_vram_absolute_gb", "global_peak_ram_gb"):
            if not finite(report.get(key)):
                raise ValueError(f"Missing Qwen SDNQ metric: {key}")
    return {str(p): suite.digest(p) for p in artifacts}
