"""Saved image INT4 comparisons using the shared resumable controller."""

import argparse
import os
from importlib import metadata
import json
import math
from pathlib import Path
import sys

REVISION = "e00d7c057f1693aa0e4bb47583e7e7ca71ccd214"
MODEL = "elismasilva/ltx2.3-image-distilled-1.1-sdnq-int4"
CASES = ("sdnq-int4-leaf", "sdnq-int4-ddo")
PROFILE = "ltx23-image-1280x704-8steps-sdnq-int4"


def build_plan(suite, args):
    if args.recipe == "ltx23-video-8s":
        if __package__:
            from .video_sdnq_suite import build_plan as video_plan
        else:
            from video_sdnq_suite import build_plan as video_plan
        return video_plan(suite, args)
    if args.recipe == "qwen-image21":
        if __package__:
            from .qwen_sdnq_suite import build_plan as qwen_plan
        else:
            from qwen_sdnq_suite import build_plan as qwen_plan
        return qwen_plan(suite, args)
    if args.platform != "windows":
        raise ValueError("This SDNQ pair is prepared only for Windows.")
    base = argparse.Namespace(**{**vars(args), "family": "runtime", "case": "one-shot-fast",
                                "nvme_resident_gib": 1.5})
    plan = suite.build_plan(base)
    checkpoint = args.sdnq_model_path or suite.hub_snapshot(MODEL, REVISION)
    command = [sys.executable, "-B", "-u", str(suite.ROOT / "scripts/ltx2_image/run_dynamic_modular_sdnq_distilled.py"),
               "--model-path", plan["checkpoint"], "--sdnq-model-path", str(checkpoint),
               "--custom-blocks", plan["custom_blocks"], "--local-files-only", "--bits", "4",
               "--prompt", suite.RECIPES[args.recipe]["prompt"], "--width", "1280", "--height", "704",
               "--steps", "8", "--seed", "43", "--guidance-scale", "1", "--guidance-rescale", "0.7",
               "--image-self-attention", "default", "--text-encoder-group-offload", "--memory-monitor", "rss",
               "--metrics-level", "2", "--save-metrics", "--no-purge-before-run",
               "--no-purge-before-transformer", "--no-purge-after-text-encoder", "--no-purge-after-run"]
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
        cases[name] = {"preset": preset, "profile": None if leaf else PROFILE, "hold": None,
                       "dispatcher": None, "routes": {"transformer": "official_diffusers_leaf" if leaf else "sdnq_runtime",
                                                       "encoder": "diffusers_leaf", "vae": "resident"},
                       "stages": stages, "sdnq_checkpoint": str(checkpoint)}
        if not leaf and not args.case:
            cases[name]["png_reference_case"] = CASES[0]
    plan.update(family="precision", cases=cases, sdnq_checkpoint=str(checkpoint), sdnq_revision=REVISION)
    return plan


def verify_stage(suite, case, stage, output, profile_dir):
    if case.get("recipe") == "ltx23-video-8s":
        if __package__:
            from .video_sdnq_suite import verify_stage as video_verify
        else:
            from video_sdnq_suite import verify_stage as video_verify
        return video_verify(suite, case, stage, output, profile_dir)
    if case.get("recipe") == "qwen-image21":
        if __package__:
            from .qwen_sdnq_suite import verify_stage as qwen_verify
        else:
            from qwen_sdnq_suite import verify_stage as qwen_verify
        return qwen_verify(suite, case, stage, output, profile_dir)
    path = suite.metrics_file(output)
    report = suite.read_json(path)
    leaf = case["preset"] == "diffusers_leaf_offload_compat"
    calibration = stage.startswith("profile-")
    expected_steps = 2 if calibration else 8
    expected = {"width": 1280, "height": 704, "seed": 43, "num_inference_steps": 8,
                "active_inference_steps": expected_steps, "sdnq_enabled": True, "sdnq_bits": 4,
                "sdnq_text_encoder_bits": None, "text_encoder_kind": "original", "dtype": "torch.bfloat16",
                "guidance_scale": 1, "guidance_rescale": 0.7, "pag_enabled": False, "lora_enabled": False,
                "offload_backend": "diffusers_leaf" if leaf else "ddo_sdnq_runtime",
                "vae_decode_route": "resident", "sdnq_model_path": case["sdnq_checkpoint"],
                "ddo_preset": case["preset"], "build_ddo_profile": stage == "profile-1",
                "validate_ddo_profile": stage == "profile-2",
                "transformer_attention_requested": "native"}
    for key, value in expected.items():
        if key not in report or report[key] != value:
            raise ValueError(f"SDNQ recipe mismatch: {key}")
    if report.get("memory_monitor", {}).get("ram_metric") != "rss":
        raise ValueError("Expected RSS monitoring.")
    timings = report.get("denoise_step_times")
    if not isinstance(timings, list) or len(timings) != expected_steps:
        raise ValueError("Incomplete SDNQ step timings.")
    finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    if not all(map(finite, timings)):
        raise ValueError("Invalid SDNQ step timings.")
    component = report.get("dynamic_offload_reports", {}).get("transformer", {})
    if component.get("route") != ("diffusers_group_offload" if leaf else "dynamic_offload"):
        raise ValueError("Unexpected SDNQ transformer route.")
    if not leaf and component.get("configuration", {}).get("dynamic_offload_execution_mode") != "sdnq_runtime":
        raise ValueError("The dedicated packed SDNQ runtime was not observed.")
    for value in suite.walk_field(report, "purged"):
        if value is True:
            raise ValueError("The SDNQ comparison must not purge standby memory.")
    events = report.get("events", [])
    if not any(event.get("name") == "setup_text_encoder_group_offload" for event in events):
        raise ValueError("Leaf text encoding was not observed.")
    profile = report.get("ddo_profile") or {}
    artifacts = [path]
    if not leaf:
        if profile.get("name") != case["profile"] or profile.get("status") != ("saved" if calibration else "applied"):
            raise ValueError("The SDNQ-specific profile was not saved or applied.")
        if not finite(profile.get("resident_budget_gb")):
            raise ValueError("Missing SDNQ residency budget.")
        if calibration:
            saved = Path(profile["path"])
            if saved.parent.resolve() != profile_dir.resolve():
                raise ValueError("Profile escaped the session directory.")
            record = suite.read_json(saved)
            if record.get("context", {}).get("execution_mode") != "sdnq_runtime":
                raise ValueError("A BF16 or VMM profile cannot stand in for SDNQ.")
            if record.get("context", {}).get("name") != case["profile"]:
                raise ValueError("The saved SDNQ profile name differs.")
            kinds = {"sdnq_baseline_probe_pending", "sdnq_baseline_safe"} if stage == "profile-1" else {"sdnq_probe"}
            if profile.get("profile_measurement_kind") not in kinds:
                raise ValueError("Unexpected SDNQ calibration stage.")
        else:
            candidates = [p for p in profile_dir.glob("*.json")
                          if suite.read_json(p).get("context", {}).get("name") == case["profile"]]
            if len(candidates) != 1:
                raise ValueError("Expected one dedicated saved SDNQ profile.")
            record = suite.read_json(candidates[0])
            if record.get("context", {}).get("execution_mode") != "sdnq_runtime":
                raise ValueError("A BF16 or VMM profile cannot stand in for SDNQ.")
        recommendation = record.get("recommendation", {})
        budget = recommendation.get("profile_resident_module_budget_gb")
        if not finite(budget) or abs(budget - profile["resident_budget_gb"]) > 0.000051:
            raise ValueError("The observed SDNQ budget differs from its saved recommendation.")
        pending = "sdnq_probe_resident_budget_gb" in recommendation
        if calibration and profile.get("build_stage") != ("probe_pending" if pending else "final"):
            raise ValueError("The SDNQ calibration completion state differs.")
        if stage != "profile-1" and pending:
            raise ValueError("The SDNQ probe is still pending.")
    pngs = list(output.glob("*.png"))
    if calibration:
        if pngs or any(event.get("name") == "load_vae_to_cuda" for event in events):
            raise ValueError("Calibration unexpectedly decoded an image.")
    else:
        from PIL import Image
        if len(pngs) != 1:
            raise ValueError("Expected one SDNQ PNG.")
        with Image.open(pngs[0]) as image:
            if image.size != (1280, 704):
                raise ValueError("Unexpected output image shape.")
            image.verify()
        artifacts += pngs
        for key in ("total_elapsed_sec", "global_peak_vram_absolute_gb", "global_peak_ram_gb"):
            if not finite(report.get(key)):
                raise ValueError(f"Missing SDNQ metric: {key}")
        if not any(event.get("name") == "load_vae_to_cuda" for event in events):
            raise ValueError("Resident VAE loading was not observed.")
    return {str(p): suite.digest(p) for p in artifacts}


def verify_profile(suite, case, stage, report, profile_dir):
    """Validate a packed-runtime profile without accepting a dense/VMM substitute."""
    if case["profile"] is None:
        if report.get("ddo_profile"):
            raise ValueError("Leaf must not use a capacity profile.")
        return
    profile = report.get("ddo_profile") or {}
    calibration = stage.startswith("profile-")
    if profile.get("name") != case["profile"] or profile.get("status") != ("saved" if calibration else "applied"):
        raise ValueError("The SDNQ-specific profile was not saved or applied.")
    finite = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
    if not finite(profile.get("resident_budget_gb")):
        raise ValueError("Missing SDNQ residency budget.")
    if calibration:
        saved = Path(profile["path"])
        if saved.parent.resolve() != profile_dir.resolve():
            raise ValueError("Profile escaped the session directory.")
        kinds = {"sdnq_baseline_probe_pending", "sdnq_baseline_safe"} if stage == "profile-1" else {"sdnq_probe"}
        if profile.get("profile_measurement_kind") not in kinds:
            raise ValueError("Unexpected SDNQ calibration stage.")
    else:
        candidates = [p for p in profile_dir.glob("*.json")
                      if suite.read_json(p).get("context", {}).get("name") == case["profile"]]
        if len(candidates) != 1:
            raise ValueError("Expected one dedicated saved SDNQ profile.")
        saved = candidates[0]
    record = suite.read_json(saved)
    if record.get("context", {}).get("execution_mode") != "sdnq_runtime" or record.get("context", {}).get("name") != case["profile"]:
        raise ValueError("A BF16 or VMM profile cannot stand in for SDNQ.")
    recommendation = record.get("recommendation", {})
    budget = recommendation.get("profile_resident_module_budget_gb")
    if not finite(budget) or abs(budget - profile["resident_budget_gb"]) > 0.000051:
        raise ValueError("The observed SDNQ budget differs from its saved recommendation.")
    pending = "sdnq_probe_resident_budget_gb" in recommendation
    if calibration and profile.get("build_stage") != ("probe_pending" if pending else "final"):
        raise ValueError("The SDNQ calibration completion state differs.")
    if stage != "profile-1" and pending:
            raise ValueError("The SDNQ probe is still pending.")


def main(suite, run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows",), default="windows")
    parser.add_argument("--recipe", choices=("ltx23-image", "qwen-image21", "ltx23-video-8s"), required=True)
    parser.add_argument("--family", choices=("precision",), required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--checkpoint", type=Path, help=suite.CHECKPOINT_HELP)
    parser.add_argument("--sdnq-model-path", type=Path,
                        help="Optional saved INT4 snapshot. By default the pinned revision is used from the "
                             "Hugging Face cache and downloaded by --preflight or --execute when missing.")
    parser.add_argument("--custom-blocks", type=Path)
    parser.add_argument("--ffmpeg-bin", type=Path)
    parser.add_argument("--output-root", type=Path, default=suite.OUTPUT)
    parser.add_argument("--session-action", choices=("continue", "restart"))
    parser.add_argument("--expected-dll-sha256",
                        help="Optional Windows DLL SHA-256 to require; defaults to the installed wheel record.")
    parser.add_argument("--list-cases", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_plan(suite, args)
    if args.list_cases:
        if args.preflight or args.execute:
            parser.error("--list-cases cannot be used with --execute or --preflight.")
        rows = [(name, suite.describe_case(case)) for name, case in plan["cases"].items()]
        if args.recipe == "ltx23-video-8s" and not args.case:
            if __package__:
                from .run_benchmark import FP8_CASE, video_case_description
            else:
                from run_benchmark import FP8_CASE, video_case_description
            rows.append((FP8_CASE, video_case_description(FP8_CASE, "precision")))
        suite.print_cases(rows)
        return
    if not (args.preflight or args.execute):
        print(json.dumps(plan, indent=2))
        return
    if sys.platform != "win32":
        parser.error("Use the installed Windows RC5 environment.")
    identity = suite.execution_identity(
        plan, args.expected_dll_sha256.lower() if args.expected_dll_sha256 else None)
    # Image SDNQ repositories also contain encoders and a VAE that these runners
    # load from the base checkpoint, so only the packed transformer is fetched.
    suite.ensure_hub_snapshot(plan["sdnq_checkpoint"], "transformer/config.json",
                              allow_patterns=None if args.recipe == "ltx23-video-8s" else ["transformer/*"])
    if args.recipe == "ltx23-video-8s":
        if __package__:
            from .video_sdnq_suite import prepare_media_identity
        else:
            from video_sdnq_suite import prepare_media_identity
        identity["media"] = prepare_media_identity(suite, plan, args)
    identity["packages"]["sdnq"] = metadata.version("sdnq")
    identity["packages"]["triton-windows"] = metadata.version("triton-windows")
    checkpoint = Path(plan["sdnq_checkpoint"])
    config = checkpoint / "transformer/config.json"
    if checkpoint.name != plan["sdnq_revision"] or not config.is_file() or not any((checkpoint / "transformer").glob("*.safetensors")):
        raise ValueError("Expected the reviewed saved INT4 checkpoint snapshot.")
    identity["sdnq_config_sha256"] = suite.digest(config)
    if args.preflight:
        print(json.dumps(identity, indent=2))
        return
    suite.run_session(parser, args, identity, run,
                      verify=lambda *stage: verify_stage(suite, *stage))
