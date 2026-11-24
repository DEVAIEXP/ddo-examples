"""Plan or capture one approved pipeline benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from uuid import uuid4


BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
RECIPE = "ltx23-video-5s"
PRESSURE_RECIPE = "ltx23-video-8s"
CASE = "sdpa-baseline"
OFFICIAL_LEAF_CASE = "official-leaf"
PRESSURE_CASES = (OFFICIAL_LEAF_CASE, "one-shot-fast")
VIDEO_RUNTIME_CASES = (OFFICIAL_LEAF_CASE, "one-shot-fast", "low-ram-safe", "vmm-fast")
WSL_VIDEO_RUNTIME_CASES = (OFFICIAL_LEAF_CASE, "wsl-compat", "vmm-fast")
LINUX_VIDEO_RUNTIME_CASES = (OFFICIAL_LEAF_CASE, "one-shot-fast", "low-ram-safe", "vmm-fast")
VIDEO_ATTENTION_CASES = {
    "cudnn-sdpa": ("cudnn-only", "aten::_scaled_dot_product_cudnn_attention"),
    "flash-attn2": ("flash-attn2", "flash_attn::_flash_attn_forward"),
    "sol-attn": ("sol-attn", "ddo::sol_attn_triton"),
}
SOL_ATTN_REVISION = "781e8341fb3b29d3aea5d50265772e9619340b66"
FP8_CASE = "fp8-weight-only"
VIDEO_BLOCKS_REVISION = "ae42dd1a04346f2f1e4ab11c209ea4af5357dd6e"
NORMALIZATION_CASE = "sdpa-normalization-1024"
REFERENCE_CASE = "sdpa-reference-5s"
FAMILIES = ("runtime", "model-features", "attention")
RUNTIME_VARIANTS = {
    OFFICIAL_LEAF_CASE: ("linear", "diffusers_leaf_offload_compat"),
    "one-shot-fast": ("linear", "one_shot_fast"),
    "low-ram-safe": ("linear", "low_ram_safe"),
    "wsl-compat": ("linear", "wsl_compat"),
    "vmm-fast": ("vmm", "auto"),
    "nvme-baseline": ("vmm", "nvme_baseline"),
    "vmm-fast-reclaim": ("vmm", "auto"),
}
CASES = {
    CASE: {"families": FAMILIES, "role": "control", "control_case": CASE},
    **{case: {"families": ("runtime",), "role": "control" if case == OFFICIAL_LEAF_CASE else "comparison",
              "control_case": CASE if case == "vmm-fast-reclaim" else OFFICIAL_LEAF_CASE,
              "status": "available" if case == OFFICIAL_LEAF_CASE else "preview-only"}
       for case in RUNTIME_VARIANTS},
    NORMALIZATION_CASE: {"families": ("model-features",), "role": "isolated", "control_case": CASE},
    REFERENCE_CASE: {"families": (), "role": "diagnostic", "control_case": None},
}
SNAPSHOT = "f070fb11c2af41e90ad9012d1f495b5131753a4d"
CAPACITIES = (8, 12, 16, 24, 32, 40, 80)
PURGES = ("before-run", "after-text-encoder", "before-transformer", "after-run")
REFERENCE_FIXTURE_SHA256 = {
    "man_guitar.png": "213964e961f05ee39a9a21ac07ec93f146e73ab135b79cda57f773928a81db90",
    "diff_vocal_song.mp3": "b4fe4ce9572aa43c893c39708d19ad94e37daddad69d9254541da661abe04436",
}


def current_platform():
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "linux":
        release = Path("/proc/sys/kernel/osrelease").read_text().lower()
        return "wsl" if "microsoft" in release else "linux"
    raise ValueError(f"Unsupported platform: {sys.platform}")


def disabled_purge_environment(purges):
    # The installed RC5 wheel can still enable purges through preset defaults.
    return {f"DDO_PURGE_WINDOWS_STANDBY_{phase.upper().replace('-', '_')}": "0"
            for phase, enabled in purges.items() if not enabled}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows", "linux", "wsl"))
    parser.add_argument("--recipe", choices=(RECIPE, PRESSURE_RECIPE), default=RECIPE)
    parser.add_argument("--case", choices=CASES, default=CASE)
    parser.add_argument("--family", choices=FAMILIES, help="Comparison family for this run.")
    parser.add_argument("--list-cases", action="store_true", help="List available plans without running them.")
    parser.add_argument("--vram-gib", type=int, choices=CAPACITIES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fixtures-dir", type=Path, default=BENCHMARK_DIR / "inputs" / RECIPE,
                        help="Local conditioning files. See docs/benchmarks/reproduction.md.")
    parser.add_argument("--custom-blocks", type=Path,
                        default=ROOT / "modular_diffusers/custom_blocks/ltx2_video")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "benchmarks")
    parser.add_argument("--execute", action="store_true", help="Run the reviewed plan once.")
    args = parser.parse_args()
    if args.recipe == PRESSURE_RECIPE and not args.list_cases and args.case not in PRESSURE_CASES:
        parser.error(f"The 8-second recipe supports these cases: {', '.join(PRESSURE_CASES)}.")
    if args.list_cases:
        if args.execute:
            parser.error("--list-cases cannot be used with --execute.")
        print(json.dumps({"recipe": args.recipe, "cases": {
            case: metadata for case, metadata in CASES.items()
            if (args.family is None or args.family in metadata["families"])
            and (args.recipe != PRESSURE_RECIPE or case in PRESSURE_CASES)
        }}, indent=2))
        parser.exit()
    if args.platform is None:
        parser.error("--platform is required to plan or execute a run.")
    if args.case in RUNTIME_VARIANTS and (args.platform != "windows"
            or (args.execute and args.case != OFFICIAL_LEAF_CASE)):
        parser.error("This runtime case is available only as a Windows preview pending review.")
    families = CASES[args.case]["families"]
    if args.family is not None and args.family not in families:
        parser.error(f"{args.case} is not available in the {args.family} family.")
    if args.family is None and families:
        args.family = families[0]
    if args.vram_gib is None:
        try:
            args.vram_gib = int(input(f"Nominal VRAM in GiB {CAPACITIES} [8]: ") or "8") if sys.stdin.isatty() else 8
        except ValueError:
            parser.error("VRAM capacity must be an integer.")
    if args.vram_gib not in CAPACITIES:
        parser.error(f"Choose a VRAM capacity from {CAPACITIES}.")
    return args


def build_plan(args, *, video_decoder="streaming-tiles", profile_managed=False):
    platform = current_platform()
    if platform != args.platform:
        raise ValueError(f"Requested platform {args.platform} does not match {platform}.")
    elevated = bool(ctypes.windll.shell32.IsUserAnAdmin()) if platform == "windows" else None
    official_leaf = args.case == OFFICIAL_LEAF_CASE
    seconds = 8 if args.recipe == PRESSURE_RECIPE else 5
    frame_rate = 24
    purges = {phase: not official_leaf and bool(elevated) and phase in ("before-run", "before-transformer")
              for phase in PURGES}
    reference = args.case == REFERENCE_CASE
    if reference:
        if platform != "windows" or not elevated or args.vram_gib != 8:
            raise ValueError("The reference requires elevated Windows and the 8 GiB profile.")
        purges["before-run"] = False
    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import constants

        checkpoint = Path(constants.HF_HUB_CACHE) / "models--OzzyGT--LTX-2.3-Distilled-1.1/snapshots" / SNAPSHOT
    image = (args.fixtures_dir / "man_guitar.png").resolve()
    audio = (args.fixtures_dir / "diff_vocal_song.mp3").resolve()
    resident = 1.5 * args.vram_gib / 8
    backend, preset = RUNTIME_VARIANTS.get(args.case, ("vmm", "auto"))
    runner = ("run_diffusers_leaf.py" if official_leaf else "run_dynamic_modular_distilled.py"
              if backend == "linear" else "run_dynamic_modular_vmm_distilled.py")
    normalization_chunk = "1024" if args.case in (NORMALIZATION_CASE, REFERENCE_CASE) else "0"
    command = [
        sys.executable, "-B", "-u", str(ROOT / "scripts/ltx2_video" / runner),
    ]
    if not official_leaf:
        command.extend(("--ddo-backend", backend, "--ddo-preset", preset))
    if backend == "vmm":
        cache_budget = "0" if preset == "nvme_baseline" else "24.625"
        if args.case == "vmm-fast" and platform == "wsl":
            command.extend(("--ddo-vmm-total-staging-gb", "0.375",
                            "--ddo-vmm-source-cache-gb", cache_budget,
                            "--ddo-vmm-source-registered-gb", "0.5", "--no-text-encoder-native-vmm"))
        else:
            if not profile_managed:
                command.extend(("--ddo-vmm-resident-weight-budget-gb", str(resident)))
            command.extend((
                "--ddo-vmm-total-staging-gb", "1.5",
                "--ddo-vmm-source-cache-gb", cache_budget, "--ddo-vmm-source-registered-gb", cache_budget,
            ))
            if args.case == "vmm-fast" and platform == "linux":
                command.extend(("--text-encoder-native-vmm", "--text-encoder-vmm-resident-weight-budget-gb", "4"))
    command.extend((
        "--distilled-model", str(checkpoint.resolve()), "--custom-blocks", str(args.custom_blocks.resolve()),
        "--local-files-only", "--image", str(image), "--audio", str(audio), "--use-audio-condition",
        "--prompt", "the man plays the guitar and sings on an old western town",
        "--width", "1280", "--height", "704", "--seconds", str(seconds), "--frame-rate", str(frame_rate),
        "--steps", "8", "--seed", "43", "--video-self-attention", "default",
        "--feed-forward-chunk-size", "1024" if reference else "0",
        "--normalization-chunk-size", normalization_chunk,
        "--reuse-rotary-embeddings" if reference else "--no-reuse-rotary-embeddings",
    ))
    if backend == "vmm":
        command.append("--ddo-vmm-execution-backing-reclaim" if args.case == "vmm-fast-reclaim"
                       else "--no-ddo-vmm-execution-backing-reclaim")
    command.extend((
        "--stream-video-decode", "--video-decoder", video_decoder, "--decode-device", "cuda:0",
        "--memory-monitor", "rss",
    ))
    if backend == "vmm":
        command.extend(("--metrics-level", "2"))
    else:
        command.extend(("--no-phase-modulation", "--no-compact-timesteps", "--no-inplace-residuals"))
    command.append("--save-metrics")
    if reference:
        command.extend(("--phase-modulation", "--compact-timesteps", "--inplace-residuals"))
    if not official_leaf:
        command.extend(f"--purge-{phase}" for phase, enabled in purges.items() if enabled)
    return {
        "platform": platform, "recipe": args.recipe, "case": args.case, "windows_elevated": elevated,
        "expected_frames": seconds * frame_rate + 1,
        "family": args.family, "case_role": CASES[args.case]["role"],
        "control_case": CASES[args.case]["control_case"],
        "nominal_vram_gib": args.vram_gib, "resident_budget_gib": resident if backend == "vmm" else None,
        "budget_status": ("not-applicable" if official_leaf else "preset-managed" if backend == "linear" else
                          "reference" if args.vram_gib == 8 else "unvalidated"),
        "purges_requested": purges, "environment_overrides": {
            "DIFFUSERS_ATTN_BACKEND": "native", **disabled_purge_environment(purges),
        },
        "working_directory": str(ROOT), "command": command,
        "fixtures": [str(image), str(audio)],
        "checkpoint": str(checkpoint.resolve()), "custom_blocks": str(args.custom_blocks.resolve()),
    }


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(directory, plan):
    (directory / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def relay_output(source, log, console):
    with source:
        while chunk := source.read1(8192):
            log.write(chunk)
            log.flush()
            if console is not None:
                try:
                    console.write(chunk)
                    console.flush()
                except OSError:
                    console = None


def run_with_live_logs(command, directory, env):
    with (directory / "stdout.txt").open("wb") as stdout, (directory / "stderr.txt").open("wb") as stderr:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
            with ThreadPoolExecutor(max_workers=2) as executor:
                streams = (
                    executor.submit(relay_output, child.stdout, stdout, getattr(sys.stdout, "buffer", None)),
                    executor.submit(relay_output, child.stderr, stderr, getattr(sys.stderr, "buffer", None)),
                )
                for stream in streams:
                    stream.result()
            return child.wait()


def execute(plan, directory):
    command = plan["command"]
    plan["status"] = "running"
    write_manifest(directory, plan)
    try:
        for path in (plan["checkpoint"], plan["custom_blocks"]):
            if not Path(path).is_dir():
                raise FileNotFoundError(f"Missing local benchmark input directory: {path}")
        missing_fixtures = [path for path in plan["fixtures"] if not Path(path).is_file()]
        if missing_fixtures:
            raise FileNotFoundError("Missing conditioning files. Supply --fixtures-dir with both files "
                                    f"listed in docs/benchmarks/reproduction.md: {missing_fixtures}")
        plan["fixture_sha256"] = {path: digest(path) for path in plan["fixtures"]}
        plan["reference_fixtures_match"] = {
            Path(path).name: sha256 for path, sha256 in plan["fixture_sha256"].items()
        } == REFERENCE_FIXTURE_SHA256
        write_manifest(directory, plan)
        env = {**os.environ, **plan["environment_overrides"]}
        plan["exit_code"] = run_with_live_logs(command, directory, env)
        if plan["exit_code"]:
            raise subprocess.CalledProcessError(plan["exit_code"], command)
        metrics = json.loads((directory / "metrics/modular_run.json").read_text(encoding="utf-8"))
        expected_frames = plan["expected_frames"]
        if metrics.get("status") != "completed" or metrics.get("decoded_media", {}).get("frames") != expected_frames:
            raise ValueError(f"The runner did not report a completed {expected_frames}-frame result.")
        if plan["case"] == OFFICIAL_LEAF_CASE:
            if metrics.get("runner") != "run_diffusers_leaf" or metrics.get("offload") != {
                "backend": "diffusers", "offload_type": "leaf_level",
                "use_stream": True, "low_cpu_mem_usage": True,
            }:
                raise ValueError("The runner did not report the planned official Diffusers leaf route.")
            decode_route = metrics.get("decode_route", {})
            if (decode_route.get("device") != "cuda:0" or not decode_route.get("stream_video_decode")
                    or decode_route.get("completed_stages") != ["video", "audio_mel", "audio"]):
                raise ValueError("The runner did not complete the planned streaming CUDA decode.")
        elif metrics.get("effective_preset") != "vmm_fast":
            raise ValueError("The runner did not report the planned VMM Fast route.")
        plan["purges_observed"] = [
            event["name"] for event in metrics.get("events", [])
            if event.get("name", "").startswith("purge_windows_standby_")
        ]
        expected = {f"purge_windows_standby_{phase.replace('-', '_')}"
                    for phase, enabled in plan["purges_requested"].items() if enabled}
        if set(plan["purges_observed"]) != expected:
            raise ValueError("Observed Windows purge events differ from the plan.")
        attention = metrics.get("attention_environment", {})
        if (attention.get("diffusers_dispatch_default") != "native"
                or attention.get("video_self_attention_processor_overrides")
                or attention.get("pytorch_attention_operators") is not None):
            raise ValueError("The runner did not report the planned untraced native attention route.")
        plan["nvidia_driver_versions"] = metrics.get(
            "nvidia_driver_versions", attention.get("nvidia_driver_versions")
        )
        plan["output_sha256"] = digest(metrics["output_path"])
        denoise = next(step for step in metrics["steps"] if step["name"] == "denoise")
        decode = next((item for item in metrics.get("events", []) + metrics["steps"]
                       if item["name"] == "decode_and_save"), {})
        row = {
            "platform": plan["platform"], "recipe": plan["recipe"], "case": plan["case"],
            "family": plan["family"], "control_case": plan["control_case"],
            "nominal_vram_gib": plan["nominal_vram_gib"], "resident_budget_gib": plan["resident_budget_gib"],
            "budget_status": plan["budget_status"],
            "nvidia_driver_versions": json.dumps(plan["nvidia_driver_versions"]),
            "denoise_sec": denoise["elapsed_sec"], "decode_and_save_sec": decode.get("elapsed_sec"),
            "total_sec": metrics["total_elapsed_sec"],
            "peak_vram_absolute_gib": metrics.get("global_peak_vram_absolute_gb"),
            "peak_ram_rss_gib": metrics.get("global_peak_ram_gb"),
        }
        with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        plan["status"] = "completed"
    except BaseException as error:
        plan.update(status="failed", error=str(error))
        raise
    finally:
        write_manifest(directory, plan)


def video_runtime_plan(args):
    expandable = getattr(args, "expandable_segments", False)
    if expandable and (args.platform != "linux" or args.case != "low-ram-safe"):
        raise ValueError("Video expandable segments require the Linux Low RAM Safe diagnostic case.")
    available = (LINUX_VIDEO_RUNTIME_CASES if args.platform == "linux" else
                 WSL_VIDEO_RUNTIME_CASES if args.platform == "wsl" else VIDEO_RUNTIME_CASES)
    if args.case and args.case not in available:
        raise ValueError(f"{args.case} is not available for video on {args.platform}.")
    names = (args.case,) if args.case else available
    cases = {}
    for name in names:
        fast = name == "vmm-fast"
        single = build_plan(argparse.Namespace(**{**vars(args), "case": name}),
                            video_decoder="exact-streaming", profile_managed=fast)
        command = single["command"]
        leaf = name == OFFICIAL_LEAF_CASE
        saved_linear = args.platform == "linux" and name in ("one-shot-fast", "low-ram-safe")
        saved_profile = getattr(args, "profile_file", None) if fast or saved_linear else None
        purge_transformer = not leaf and args.platform == "windows"
        profile = None
        stages = []
        if leaf and args.platform == "wsl":
            command += ["--no-stream-offload"]
        elif not leaf:
            command = [flag for flag in command if not flag.startswith("--purge-")]
            reader = (lambda source: read_linear_video_profile(source, name)) if saved_linear else read_video_profile
            profile = (reader(saved_profile)["context"]["name"] if saved_profile else
                       f"ltx23-video-1280x704-8s-8steps-{name}")
            if purge_transformer:
                command += ["--purge-before-transformer"]
            command += ["--ddo-profile", profile]
        environment = disabled_purge_environment({
            phase: purge_transformer and phase == "before-transformer" for phase in (*PURGES, "before-vae")
        })
        if not leaf and not fast:
            environment["DDO_RESIDENT_MODULE_BUDGET_GB"] = "0"
        if profile and not saved_profile:
            build = command + ["--build-ddo-profile"]
            if fast:
                build[build.index("--steps") + 1] = "3"
            stages.extend({"name": f"profile-{index}", "command": build, "environment": environment}
                          for index in range(1, 3 if fast else 2))
        stages.append({"name": "measure", "command": command, "environment": environment})
        route = "vmm_fast_runtime" if fast else "official_diffusers_leaf" if leaf else "linear_runtime"
        cases[name] = {
            "stages": stages, "preset": "vmm_fast" if fast else None if leaf else RUNTIME_VARIANTS[name][1],
            "routes": {"transformer": route},
            "profile": profile, "dispatcher": True if fast else None, "hold": None,
        }
        if fast and args.platform == "linux":
            cases[name]["routes"].update(encoder="native_vmm", vae="resident")
        if saved_profile:
            cases[name]["saved_profile"] = str(saved_profile.resolve())
    return {
        "schema": 1, "platform": args.platform, "recipe": PRESSURE_RECIPE, "family": "runtime",
        "nominal_vram_gib": args.vram_gib, "expected_frames": single["expected_frames"],
        "checkpoint": single["checkpoint"], "checkpoint_revision": SNAPSHOT,
        "custom_blocks": single["custom_blocks"], "blocks_revision": VIDEO_BLOCKS_REVISION,
        "fixtures": single["fixtures"], "cases": cases,
        "environment_overrides": {"DIFFUSERS_ATTN_BACKEND": "native", "HF_HUB_OFFLINE": "1",
                                  "PYTHONNOUSERSITE": "1",
                                  **({"PYTORCH_ALLOC_CONF": "expandable_segments:True"} if expandable else {})},
    }


def video_fp8_plan(args):
    """Keep the current SDPA workload and calibrate the saved FP8 payload separately."""
    if not args.fp8_model_path:
        raise ValueError("Saved FP8 requires --fp8-model-path pointing to the converted model root.")
    base_args = argparse.Namespace(**{**vars(args), "family": "runtime", "case": "vmm-fast"})
    plan = video_runtime_plan(base_args)
    base = plan["cases"]["vmm-fast"]
    profile = base["profile"] + "-fp8-weight-only"
    from scripts.helpers.model_sources import source_argument
    source = source_argument(args.fp8_model_path)
    stages = []
    for stage in base["stages"]:
        command = stage["command"].copy()
        command[command.index("--ddo-profile") + 1] = profile
        command += ["--model-path", source, "--transformer-fp8-weight-only"]
        stages.append({**stage, "command": command})
    plan["family"] = "precision"
    plan["cases"] = {FP8_CASE: {**base, "profile": profile, "stages": stages,
                                "fp8_model_path": source, "comparison_control": "vmm-fast"}}
    return plan


def video_fp8_identity(source):
    """Record the saved payload without conversion or CUDA initialization."""
    from importlib.metadata import version

    folder = Path(source) / "transformer"
    config_path = folder / "config.json"
    index_path = folder / "diffusion_pytorch_model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    quantization = config.get("quantization_config", {})
    settings = quantization.get("quant_type", {}).get("default", {})
    data = settings.get("_data", {})
    if (quantization.get("quant_method") != "torchao" or settings.get("_type") != "Float8WeightOnlyConfig"
            or data.get("weight_dtype", {}).get("_data") != "float8_e4m3fn"
            or data.get("granularity", {}).get("_type") != "PerRow"):
        raise ValueError("Expected the supported saved TorchAO per-row E4M3 weight-only payload.")
    index = json.loads(index_path.read_text())
    names = sorted(set(index.get("weight_map", {}).values()))
    if not names or any(Path(name).name != name or not name.endswith('.safetensors') for name in names):
        raise ValueError("Invalid saved FP8 shard index.")
    paths = [config_path, index_path, *(folder / name for name in names)]
    report = folder / "conversion_report.json"
    if report.is_file():
        paths.append(report)
    return {"torchao_version": version("torchao"), "files": {str(p.resolve()): digest(p) for p in paths},
            "stats": {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}}


def read_video_profile(path):
    profile = json.loads(path.read_text(encoding="utf-8"))
    context = profile.get("context", {})
    recommendation = profile.get("recommendation", {})
    budget = recommendation.get("vmm_resident_weight_budget_bytes")
    if (not isinstance(context.get("name"), str) or not context["name"]
            or profile.get("profile_key") != path.stem
            or recommendation.get("vmm_build_stage") != "final"
            or type(budget) is not int or budget <= 0):
        raise ValueError("Expected a final DDO VMM residency profile.")
    return profile


def read_linear_video_profile(path, case="one-shot-fast"):
    profile = json.loads(path.read_text(encoding="utf-8"))
    context = profile.get("context", {})
    observations = profile.get("observations", [])
    budget = profile.get("recommendation", {}).get("profile_resident_module_budget_gb")
    if (profile.get("schema_version") != 1 or profile.get("profile_key") != path.stem
            or case not in ("one-shot-fast", "low-ram-safe")
            or context.get("name") != f"ltx23-video-1280x704-8s-8steps-{case}"
            or context.get("execution_mode") != "linear_runtime"
            or context.get("model_class") != "LTX2VideoTransformer3DModel"
            or type(budget) not in (int, float) or budget != 0
            or not any(item.get("status") == "success" for item in observations)):
        raise ValueError("Expected the successful zero-budget Linux linear video capacity profile.")
    return profile


def verified_linear_video_profile(suite, source, identity):
    cases = identity["plan"]["cases"]
    if len(cases) != 1 or next(iter(cases)) not in ("one-shot-fast", "low-ram-safe"):
        raise ValueError("Select exactly one reviewed Linux linear video case for profile reuse.")
    case = next(iter(cases))
    profile = read_linear_video_profile(source, case)
    manifest = source.parents[2] / "manifest.json"
    previous = suite.read_json(manifest)
    old = previous["identity"]
    record = previous["cases"][case]
    receipt = {str(source): digest(source)}
    calibration = record.get("stages", {}).get("profile-1", {})
    command = cases[case]["stages"][0]["command"]
    old_command = old["plan"]["cases"][case]["stages"][0]["command"]
    if (any(old.get(key) != identity.get(key) for key in ("python", "packages", "gpu", "dll_sha256"))
            or any(old["plan"].get(key) != identity["plan"].get(key)
                   for key in ("platform", "recipe", "checkpoint", "blocks_revision", "fixtures"))
            or [flag for flag in old_command if flag != "--build-ddo-profile"] != command
            or record.get("profile_receipt") != receipt or not suite.receipts_valid(receipt)
            or not calibration or not suite.receipts_valid(calibration)):
        raise ValueError("The existing linear calibration does not match this recipe and environment.")
    metrics = suite.read_json(suite.metrics_file(Path(next(iter(calibration))).parent.parent))
    saved = metrics.get("ddo_profile") or {}
    if (metrics.get("status") != "completed" or saved.get("status") != "saved"
            or saved.get("name") != profile["context"]["name"] or saved.get("resident_budget_gb") != 0):
        raise ValueError("The reference did not complete the selected linear capacity calibration.")
    return {"kind": "linear", "profile_receipt": receipt, "resident_budget_gb": 0,
            "manifest": str(manifest), "manifest_sha256": digest(manifest)}


def verify_video_vmm(plan, case, stage, report, purges):
    transformer = report.get("dynamic_offload_reports", {}).get("transformer", {})
    configuration = transformer.get("configuration", {})
    runtime = transformer.get("runtime", {}).get("vmm", {})
    controls = report["requested_controls"]
    windows = plan["platform"] == "windows"
    native_encoder = plan["platform"] != "wsl"
    expected_purges = ["purge_windows_standby_before_transformer"] if windows else []
    expected = {
        "dynamic_offload_execution_mode": case["routes"]["transformer"],
        "dynamic_offload_effective_preset": case["preset"],
        "dynamic_offload_vmm_source_cache_gb": 24.625,
        "dynamic_offload_vmm_source_registered_gb": 24.625 if native_encoder else 0.5,
        "dynamic_offload_vmm_source_dispatcher": True,
        "dynamic_offload_vmm_execution_backing_reclaim": False,
    }
    if (report.get("runner") != "run_dynamic_modular_vmm_distilled" or purges != expected_purges
            or report.get("dtype") != "torch.bfloat16"
            or report.get("transformer_weight_format_requested") != (
                "torchao_fp8_weight_only" if case.get("fp8_model_path") else "bf16")
            or report.get("num_frames") != plan["expected_frames"]
            or controls.get("ddo_vmm_total_staging_gb") != (1.5 if native_encoder else 0.375)
            or controls.get("text_encoder_native_vmm") is not native_encoder
            or (native_encoder and report.get("text_encoder_route_requested") != "native")
            or any(configuration.get(key) != value for key, value in expected.items())):
        raise ValueError("The VMM Fast route or configuration differs from the plan.")
    if plan["platform"] == "linux":
        encoder = report.get("dynamic_offload_reports", {}).get("text_encoder", {})
        encoder_config = encoder.get("configuration", {})
        if (encoder_config.get("dynamic_offload_execution_mode") != "vmm_fast_runtime"
                or encoder_config.get("dynamic_offload_vmm_resident_weight_budget_gb") != 4.0
                or encoder_config.get("dynamic_offload_vmm_source_dispatcher") is not True
                or controls.get("vae_leaf_offload") is not False):
            raise ValueError("The Linux video native encoder or resident VAE route differs from the plan.")
    if runtime.get("fast_source_cache_source_dispatcher_enabled") != 1:
        raise ValueError("The VMM Fast transformer did not report an active source dispatcher.")
    profile = report.get("ddo_profile") or {}
    calibration = stage.startswith("profile-")
    build_stages = ("floor_measured", "final") if stage == "profile-1" else ("final",)
    fixed_feature = not calibration and "feature_fixed_budget_bytes" in case
    if (profile.get("name") != case["profile"]
            or profile.get("status") != ("saved" if calibration else "explicit" if fixed_feature else "applied")
            or (not fixed_feature and profile.get("build_stage") not in build_stages)):
        raise ValueError(f"VMM profile {case['profile']!r} was not saved or applied. "
                         f"Reported status: {profile.get('status')!r}, "
                         f"resident budget: {profile.get('resident_budget_gb')!r} GiB.")
    if not calibration and (profile.get("shape_check") != "match"
                            or profile.get("resident_budget_gb") is None
                            or profile["resident_budget_gb"]
                            != configuration.get("dynamic_offload_vmm_resident_weight_budget_gb")):
        raise ValueError("VMM Fast did not execute with the profile shape and budget.")
    if fixed_feature:
        from benchmarks.model_features_suite import verify_fixed_budget
        verify_fixed_budget(case, report)


def verify_video_stage(plan, case, stage, output, profile_dir):
    metrics_path = output / "metrics" / "modular_run.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    controls = report.get("requested_controls", {})
    startup_allocator = plan.get("environment_overrides", {}).get("PYTORCH_ALLOC_CONF")
    if startup_allocator is not None and report.get("pytorch_allocator_startup_environment", {}).get(
            "PYTORCH_ALLOC_CONF") != startup_allocator:
        raise ValueError("The requested PyTorch allocator configuration was not present before runner imports.")
    fast = case["routes"]["transformer"] == "vmm_fast_runtime"
    calibration = stage.startswith("profile-")
    attention_backend = case.get("attention_backend", "default")
    trace = stage == "operator-check"
    expected = {
        "distilled_model": plan["checkpoint"], "custom_blocks": plan["custom_blocks"],
        "image": plan["fixtures"][0], "audio": plan["fixtures"][1],
        "prompt": "the man plays the guitar and sings on an old western town",
        "local_files_only": True,
        "width": 1280, "height": 704, "seconds": 8, "frame_rate": 24, "steps": 3 if fast and calibration else 8,
        "seed": 43, "use_audio_condition": True,
        "video_self_attention": attention_backend,
        "feed_forward_chunk_size": 0, "normalization_chunk_size": 0,
        "reuse_rotary_embeddings": False, "phase_modulation": False,
        "compact_timesteps": False, "inplace_residuals": False,
        "stream_video_decode": True, "video_decoder": "exact-streaming", "decode_device": "cuda:0",
    }
    expected.update(case.get("model_features", {}))
    attention = report.get("attention_environment", {})
    if (report.get("status") != "completed" or any(controls.get(k) != v for k, v in expected.items())
            or attention.get("diffusers_dispatch_default") != "native"
            or attention.get("video_self_attention_processor_overrides")
            or (not trace and attention.get("pytorch_attention_operators") is not None)):
        raise ValueError("The video report differs from the planned workload or attention settings.")
    if case.get("fp8_model_path"):
        config = report.get("dynamic_offload_reports", {}).get("transformer", {}).get("configuration", {})
        if (controls.get("transformer_fp8_weight_only") is not True
                or controls.get("model_path") != case["fp8_model_path"]
                or controls.get("compile_fp8_linears") is not False
                or controls.get("fp8_feed_forward_reuse") is not False
                or config.get("dynamic_offload_vmm_float8_weight_only") is not True):
            raise ValueError("The saved FP8 route differs from the uncompiled weight-only plan.")
    if "attention_operator" in case:
        verify_video_attention_dispatch(case, report, trace)
    purges = [event["name"] for event in report.get("events", [])
              if event.get("name", "").startswith("purge_windows_standby_")]
    release_expected = plan["platform"] == "linux" and case["routes"]["transformer"] == "linear_runtime"
    if controls.get("release_pytorch_host_cache_before_vae", False) is not release_expected:
        raise ValueError("The linear video host-cache release policy differs from the plan.")
    if release_expected and not calibration:
        cleanup = report.get("dynamic_offload_reports", {}).get("host_cache_before_vae", {})
        observation = cleanup.get("setup", {}).get("host_cache_release", {})
        events = [event for event in report.get("events", [])
                  if event.get("name") == "release_pytorch_host_cache_before_vae"]
        if (cleanup.get("route") != "pytorch_host_cache_release"
                or observation.get("status") != "released" or len(events) != 1):
            raise ValueError("The planned pre-VAE PyTorch host-cache release was not recorded.")
    if fast:
        verify_video_vmm(plan, case, stage, report, purges)
    elif case["preset"]:
        configuration = report.get("dynamic_offload_reports", {}).get("transformer", {}).get("configuration", {})
        wsl_compat = case["preset"] == "wsl_compat"
        expected_purges = (["purge_windows_standby_before_transformer"]
                           if plan["platform"] == "windows" else [])
        observed_budget = configuration.get("dynamic_offload_resident_module_budget_gb")
        if (report.get("runner") != "run_dynamic_modular_distilled"
                or configuration.get("dynamic_offload_execution_mode") != "linear_runtime"
                or configuration.get("dynamic_offload_effective_preset") != case["preset"]
                or (not wsl_compat and observed_budget != 0)
                or purges != expected_purges):
            raise ValueError("The linear transformer route, budget or purge differs from the plan.")
        profile = report.get("ddo_profile") or {}
        if (profile.get("name") != case["profile"]
                or profile.get("status") != ("saved" if calibration else "applied")):
            raise ValueError("The planned linear profile was not saved or applied.")
        if wsl_compat:
            budget = profile.get("resident_budget_gb")
            if (configuration.get("dynamic_offload_effective_pin_cpu_memory") is not False
                    or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0
                    or observed_budget != (0 if calibration else budget)):
                raise ValueError("WSL linear execution did not use the planned profile budget and pinning policy.")
    elif (report.get("runner") != "run_diffusers_leaf"
          or report.get("offload") != {"backend": "diffusers", "offload_type": "leaf_level",
                                       "use_stream": plan["platform"] != "wsl", "low_cpu_mem_usage": True}
          or purges):
        raise ValueError("The official Leaf route or purge policy differs from the plan.")
    if calibration:
        saved = Path(report["ddo_profile"]["path"]).resolve()
        if not saved.is_relative_to(profile_dir.resolve()) or not saved.is_file():
            raise ValueError("The saved profile is missing or outside the session.")
        return {str(metrics_path): digest(metrics_path)}
    if fast:
        media = report.get("decoded_media", {})
        decoded = (report.get("vae_decode_route") == "none"
                   and media.get("frames") == plan["expected_frames"] and media.get("audio_finite") is True)
    else:
        decode = report.get("decode_route", {})
        decoded = (decode.get("device") == "cuda:0" and decode.get("stream_video_decode") is True
                   and decode.get("video_decoder") == expected["video_decoder"]
                   and decode.get("completed_stages") == ["video", "audio_mel", "audio"])
    if not decoded or report.get("memory_monitor", {}).get("ram_metric") != "rss":
        raise ValueError("The planned streaming decode or RSS monitoring was not recorded.")
    phases = {step["name"]: step["elapsed_sec"] for step in report["steps"]}
    decode_seconds = next((event.get("elapsed_sec") for event in report.get("events", [])
                           if event["name"] == "decode_and_save"), phases.get("decode_and_save"))
    values = [report.get(key) for key in ("total_elapsed_sec", "global_peak_vram_absolute_gb",
                                         "global_peak_ram_gb")]
    values += [phases.get("denoise"), decode_seconds]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("The video report is missing required phase or pipeline metrics.")
    videos = list(output.glob("*.mp4"))
    if len(videos) != 1 or videos[0].resolve() != Path(report["output_path"]).resolve():
        raise ValueError("Expected exactly one reported MP4 in the attempt directory.")
    probe = subprocess.check_output([
        plan["ffprobe"], "-v", "error", "-show_entries",
        "stream=codec_type,width,height,nb_frames", "-of", "json", str(videos[0]),
    ], text=True, timeout=30)
    streams = json.loads(probe)["streams"]
    video = [stream for stream in streams if stream["codec_type"] == "video"]
    if (len(video) != 1 or (video[0]["width"], video[0]["height"]) != (1280, 704)
            or video[0].get("nb_frames") != str(plan["expected_frames"])
            or not any(stream["codec_type"] == "audio" for stream in streams)):
        raise ValueError("The MP4 does not contain the expected video and audio streams.")
    return {str(path): digest(path) for path in (metrics_path, videos[0])}


def verify_video_attention_dispatch(case, report, trace):
    backend = case["attention_backend"]
    routes = report.get("video_self_attention", {})
    layers = routes.get("layers", [])
    operators = report.get("attention_environment", {}).get("pytorch_attention_operators") or {}
    if (routes.get("requested_backend") != backend
            or report.get("requested_controls", {}).get("trace_attention_operators") is not trace
            or len(layers) != 48 or [layer.get("index") for layer in layers] != list(range(48))
            or any(layer.get("calls") != 8 or layer.get("fallback_calls") != 0 for layer in layers)):
        raise ValueError(f"The requested {backend} video attention did not execute all planned calls.")
    expected = ["flash-attn2" if backend == "sol-attn" and index == 0 else backend
                for index in range(48)]
    if [layer.get("backend") for layer in layers] != expected:
        raise ValueError("The video attention layer policy differs from the plan.")
    calls = 376 if backend == "sol-attn" else 384
    if backend == "sol-attn":
        policy = routes.get("sol_policy", {})
        if (policy.get("backend"), policy.get("tau"), policy.get("thresh_type")) != ("triton", 1.0, "diag"):
            raise ValueError("The Sol-Attn kernel policy differs from the plan.")
        if trace and operators.get("flash_attn::_flash_attn_forward", 0) < 8:
            raise ValueError("The operator trace did not confirm FlashAttention in video block 0.")
    if trace and operators.get(case["attention_operator"], 0) < calls:
        raise ValueError(f"The operator trace did not confirm the {backend} video calls.")


def video_triton_dependency_identity():
    """Record the installed Triton provider without importing its runtime."""
    from importlib import metadata

    provider = "triton"
    try:
        version = metadata.version(provider)
    except metadata.PackageNotFoundError:
        if sys.platform != "win32":
            raise
        provider = "triton-windows"
        version = metadata.version(provider)
    return {"triton_distribution": provider, "triton_version": version}


def video_sol_dependency_identity(suite):
    """Inspect the optional source without importing Torch or initializing CUDA."""
    from kernels.sol_loader import source_identity

    identity = source_identity()
    if identity['revision'] != SOL_ATTN_REVISION:
        raise ValueError('The bundled Sol-Attn revision differs from the benchmark.')
    return {**identity, **video_triton_dependency_identity()}


def video_attention_plan(args, previous):
    control = previous["identity"]["plan"]
    if control["family"] != "runtime" or control["recipe"] != PRESSURE_RECIPE:
        raise ValueError("Use the completed LTX video 8-second VMM Fast runtime manifest.")
    saved = control["cases"]["vmm-fast"].get("saved_profile")
    runtime_args = argparse.Namespace(**{**vars(args), "family": "runtime", "case": "vmm-fast",
                                         "checkpoint": Path(control["checkpoint"]),
                                         "custom_blocks": Path(control["custom_blocks"]),
                                         "fixtures_dir": Path(control["fixtures"][0]).parent,
                                         "profile_file": Path(saved) if saved else None})
    plan = video_runtime_plan(runtime_args)
    case = plan["cases"]["vmm-fast"]
    measure = case["stages"][-1]
    if measure != control["cases"]["vmm-fast"]["stages"][-1]:
        raise ValueError("The planned video control differs from the measured runtime recipe.")
    command = measure["command"].copy()
    backend, operator = VIDEO_ATTENTION_CASES[args.case]
    command[command.index("--video-self-attention") + 1] = backend
    plan["family"] = "attention"
    plan["cases"] = {args.case: {
        **case, "attention_backend": backend, "attention_operator": operator,
        "stages": [
            {**measure, "name": "operator-check", "command": command + ["--trace-attention-operators"]},
            {**measure, "command": command},
        ],
    }}
    return plan


def verified_video_attention_reference(suite, manifest, previous, manifest_sha, identity):
    def comparable(value):
        result = {key: item for key, item in value.items()
                  if key not in ("plan", "video_saved_profile", "sol_attn_dependency")}
        updated = {"benchmarks/run_benchmark.py", "benchmarks/image_suite.py",
                   "scripts/helpers/ltx2/video_self_attention.py"}
        result["runner_sha256"] = {Path(key).as_posix(): item for key, item in result["runner_sha256"].items()
                                   if Path(key).as_posix() not in updated}
        return result

    def workload(plan):
        return {key: item for key, item in plan.items() if key not in ("family", "cases")}

    old = previous["identity"]
    record = previous["cases"]["vmm-fast"]
    if (record.get("status") != "completed" or comparable(old) != comparable(identity)
            or workload(old["plan"]) != workload(identity["plan"])):
        raise ValueError("The video attention environment differs from the completed SDPA reference.")
    artifacts = record.get("artifacts", {})
    receipt = record.get("profile_receipt", {})
    metrics = [Path(path) for path in artifacts if Path(path).suffix == ".json"]
    if (len(metrics) != 1 or len([path for path in artifacts if Path(path).suffix == ".mp4"]) != 1
            or len(receipt) != 1 or not suite.receipts_valid(artifacts) or not suite.receipts_valid(receipt)):
        raise ValueError("The SDPA reference metrics, MP4 or profile is missing or has changed.")
    report = suite.read_json(metrics[0])
    applied = report.get("ddo_profile") or {}
    saved = suite.read_json(Path(next(iter(receipt))))
    budget = saved["recommendation"]["vmm_resident_weight_budget_bytes"]
    if (report.get("requested_controls", {}).get("video_self_attention") != "default"
            or applied.get("status") != "applied" or applied.get("build_stage") != "final"
            or applied.get("name") != old["plan"]["cases"]["vmm-fast"]["profile"]
            or applied.get("shape_check") != "match" or not isinstance(budget, int) or budget <= 0
            or (applied.get("recommendation") or {}).get("vmm_resident_weight_budget_bytes") != budget
            or digest(manifest) != manifest_sha):
        raise ValueError("The reference did not apply the saved final SDPA profile.")
    return {"manifest": str(manifest), "manifest_sha256": manifest_sha, "artifacts": artifacts,
            "profile_receipt": receipt, "resident_budget_bytes": budget}


def video_case_description(name, family):
    """Summarize a video case without building its platform-specific plan."""
    if family == "attention":
        return f"vmm / auto, transformer vmm_fast_runtime, attention {VIDEO_ATTENTION_CASES[name][0]}"
    if name == FP8_CASE:
        return "vmm / auto, transformer vmm_fast_runtime, saved FP8 weight-only transformer"
    if name == OFFICIAL_LEAF_CASE:
        return "Diffusers Leaf reference (not a DDO preset)"
    backend, preset = RUNTIME_VARIANTS[name]
    transformer = ("linear_runtime" if backend == "linear" else
                   "vmm_runtime" if preset == "nvme_baseline" else "vmm_fast_runtime")
    return f"{backend} / {preset}, transformer {transformer}"


def video_runtime_main(suite):
    parser = argparse.ArgumentParser(description="Run the resumable LTX video runtime benchmark.")
    parser.add_argument("--recipe", choices=(PRESSURE_RECIPE,), required=True)
    parser.add_argument("--platform", choices=("windows", "linux", "wsl"), default="windows")
    parser.add_argument("--family", choices=("runtime", "attention", "precision"), default="runtime")
    parser.add_argument("--case", choices=tuple(dict.fromkeys((*VIDEO_RUNTIME_CASES, *WSL_VIDEO_RUNTIME_CASES,
                                                            *VIDEO_ATTENTION_CASES, FP8_CASE))))
    parser.add_argument("--attention-runtime", choices=("vmm-fast",))
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--fp8-model-path", type=Path, help="Saved FP8 local root or Hub model ID; precision case only.")
    parser.add_argument("--profile-file", type=Path,
                        help="Existing final Windows VMM Fast profile or calibrated Linux linear video profile.")
    parser.add_argument("--expandable-segments", action="store_true",
                        help="Diagnostic allocator configuration before imports; Linux Low RAM Safe video only.")
    parser.add_argument("--vram-gib", type=int, choices=(8,), default=8)
    parser.add_argument("--checkpoint", type=Path, help=suite.CHECKPOINT_HELP)
    parser.add_argument("--custom-blocks", type=Path)
    parser.add_argument("--fixtures-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks")
    parser.add_argument("--session-action", choices=("continue", "restart"))
    parser.add_argument("--expected-dll-sha256",
                        help="Optional Windows DLL SHA-256 to require; defaults to the installed wheel record.")
    parser.add_argument("--expected-library-sha256",
                        help="Optional Linux or WSL library SHA-256 to require; defaults to the installed wheel record.")
    parser.add_argument("--ffmpeg-bin", type=Path)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Validate identities and prerequisite plans without CUDA.")
    args = parser.parse_args()
    wsl = args.platform == "wsl"
    attention = args.family == "attention"
    precision = args.family == "precision"
    if precision and not args.fp8_model_path:
        args.fp8_model_path = Path("elismasilva/ltx2.3_video_fp8_distilled_transformer")
    if precision and (args.case != FP8_CASE or not args.fp8_model_path):
        parser.error("Select --family precision --case fp8-weight-only --fp8-model-path <converted-root>.")
    if args.fp8_model_path and not precision:
        parser.error("--fp8-model-path requires the precision family.")
    linear_profile = args.platform == "linux" and not attention and args.case in ("one-shot-fast", "low-ram-safe")
    if args.expandable_segments and (args.platform != "linux" or attention or args.case != "low-ram-safe"):
        parser.error("--expandable-segments requires the Linux Low RAM Safe video runtime case.")
    if args.platform == "linux" and attention and not args.list_cases and args.case not in VIDEO_ATTENTION_CASES:
        parser.error("Select a reviewed Linux video attention comparison.")
    if args.profile_file and not (linear_profile or
                                 (args.platform == "windows" and not attention and args.case == "vmm-fast")):
        parser.error("--profile-file requires Windows VMM Fast or a Linux linear video runtime.")
    if not attention and (args.attention_runtime or args.reference_manifest):
        parser.error("Reference options require the attention family.")
    available = ((FP8_CASE,) if precision else tuple(VIDEO_ATTENTION_CASES) if attention else
                 LINUX_VIDEO_RUNTIME_CASES if args.platform == "linux" else
                 WSL_VIDEO_RUNTIME_CASES if wsl else VIDEO_RUNTIME_CASES)
    if args.case is not None and args.case not in available:
        parser.error(f"Select a video runtime case for {args.platform}: {', '.join(available)}.")
    if args.list_cases:
        if args.execute or args.preflight:
            parser.error("--list-cases cannot be used with --execute or --preflight.")
        suite.print_cases((name, video_case_description(name, args.family)) for name in available)
        return
    control_plan = None
    if attention:
        if not args.case or args.attention_runtime != "vmm-fast":
            parser.error("Select one attention case with --attention-runtime vmm-fast.")
        if args.reference_manifest:
            if args.checkpoint or args.custom_blocks or args.fixtures_dir:
                parser.error("An explicit reference manifest supplies checkpoint, blocks and fixtures.")
            previous = suite.read_json(args.reference_manifest.resolve())
        else:
            control_args = argparse.Namespace(**{**vars(args), "family": "runtime", "case": "vmm-fast",
                                                "attention_runtime": None})
            control_args.custom_blocks = args.custom_blocks or ROOT / "modular_diffusers/custom_blocks/ltx2_video"
            control_args.fixtures_dir = args.fixtures_dir or BENCHMARK_DIR / "inputs" / RECIPE
            control_plan = video_runtime_plan(control_args)
            previous = {"identity": {"plan": control_plan}}
        plan = video_attention_plan(args, previous)
    else:
        args.custom_blocks = args.custom_blocks or ROOT / "modular_diffusers/custom_blocks/ltx2_video"
        args.fixtures_dir = args.fixtures_dir or BENCHMARK_DIR / "inputs" / RECIPE
        plan = video_fp8_plan(args) if precision else video_runtime_plan(args)
    if args.preflight and args.execute:
        parser.error("Choose either preflight or execution.")
    if not args.execute and not args.preflight:
        print(json.dumps({"attention": plan, "prerequisite": control_plan,
                          "prerequisite_status": "resolve-compatible-control-before-execution"}
                         if control_plan else plan, indent=2))
        return
    expected_hash = args.expected_dll_sha256 if args.platform == "windows" else args.expected_library_sha256
    if (args.execute and args.platform == "windows" and any(case["preset"] for case in plan["cases"].values())
            and not ctypes.windll.shell32.IsUserAnAdmin()):
        parser.error("Use an elevated Windows terminal for the transformer standby purge.")
    fixture_hashes = {Path(path).name: digest(path) for path in plan["fixtures"]}
    if fixture_hashes != REFERENCE_FIXTURE_SHA256:
        parser.error("Conditioning files differ from the reviewed video fixtures.")
    ffmpeg_bin = args.ffmpeg_bin or os.getenv("DDO_RUNNER_FFMPEG_BIN")
    search_path = os.pathsep.join((str(Path(ffmpeg_bin).resolve()), os.environ.get("PATH", ""))) if ffmpeg_bin else None
    tools = {name: shutil.which(name, path=search_path) for name in ("ffmpeg", "ffprobe")}
    if not all(tools.values()):
        parser.error("Install FFmpeg with ffprobe or pass --ffmpeg-bin.")
    plan["ffprobe"] = str(Path(tools["ffprobe"]).resolve())
    plan["environment_overrides"]["DDO_RUNNER_FFMPEG_BIN"] = str(Path(tools["ffmpeg"]).resolve().parent)
    if control_plan is not None:
        control_plan["ffprobe"] = plan["ffprobe"]
        control_plan["environment_overrides"]["DDO_RUNNER_FFMPEG_BIN"] = plan["environment_overrides"]["DDO_RUNNER_FFMPEG_BIN"]
    identity = suite.execution_identity(plan, expected_hash.lower() if expected_hash else None)
    identity.update(fixture_sha256=fixture_hashes, ffmpeg_sha256={
        name: digest(path) for name, path in tools.items()
    })
    if precision:
        print("Verifying saved FP8 shard hashes before timing; no conversion is performed.", flush=True)
        identity["saved_fp8_payload"] = video_fp8_identity(plan["cases"][FP8_CASE]["fp8_model_path"])
    if args.profile_file and linear_profile:
        identity["video_linear_saved_profile"] = verified_linear_video_profile(
            suite, args.profile_file.resolve(), identity)
    elif args.profile_file:
        source = args.profile_file.resolve()
        profile_hash = digest(source)
        profile = read_video_profile(source)
        context = profile["context"]
        transport = context.get("transport", {})
        if (context.get("native_library_sha256") != identity["dll_sha256"]
                or context.get("platform") != "win32"
                or context.get("execution_mode") != "vmm_fast_runtime"
                or context.get("component") != "transformer"
                or context.get("device_memory_mib") != args.vram_gib * 1024
                or (transport.get("staging_gb"), transport.get("source_cache_gb"),
                    transport.get("source_registered_gb")) != (1.5, 24.625, 24.625)
                or digest(source) != profile_hash):
            parser.error("The final profile does not match the Windows VMM Fast configuration.")
        identity["video_saved_profile"] = {
            "profile_receipt": {str(source): profile_hash},
            "resident_budget_bytes": profile["recommendation"]["vmm_resident_weight_budget_bytes"],
        }
    if attention:
        if args.case in ("flash-attn2", "sol-attn") and not identity["packages"]["flash-attn"]:
            parser.error("Install the compatible FlashAttention 2 wheel in this environment.")
        if args.case == "sol-attn":
            identity["sol_attn_dependency"] = video_sol_dependency_identity(suite)
        def validate_reference(path, target_identity):
            return verified_video_attention_reference(suite, path, suite.read_json(path), digest(path), target_identity)

        reference = suite.ensure_attention_reference(
            parser, args, identity, control_plan, validate_reference, run_with_live_logs,
            verify=lambda *stage: verify_video_stage(control_plan, *stage))
        if reference is not None:
            identity["vmm_attention_reference"] = reference
    if args.preflight:
        print(json.dumps(identity, indent=2))
        return
    def verify(*stage):
        artifacts = verify_video_stage(plan, *stage)
        if precision:
            verify_fp8_files()
        if args.case == "sol-attn":
            if video_sol_dependency_identity(suite) != identity["sol_attn_dependency"]:
                raise ValueError("Sol-Attn source changed during the benchmark.")
            report = suite.read_json(suite.metrics_file(stage[2]))
            policy = report.get("video_self_attention", {}).get("sol_policy", {})
            if policy.get("interface") != identity["sol_attn_dependency"]["interface"]:
                raise ValueError("The runner imported a different Sol-Attn source.")
        return artifacts

    def verify_fp8_files():
        for name, observed in identity["saved_fp8_payload"]["stats"].items():
            stat = Path(name).stat()
            if [stat.st_size, stat.st_mtime_ns] != observed:
                raise ValueError("The saved FP8 payload changed during the benchmark.")

    def run(command, directory, env):
        if precision:
            verify_fp8_files()
        if args.case == "sol-attn" and video_sol_dependency_identity(suite) != identity["sol_attn_dependency"]:
            raise ValueError("Sol-Attn source changed before benchmark execution.")
        return run_with_live_logs(command, directory, env)

    suite.run_session(parser, args, identity, run, verify=verify, start_if_missing=attention)


def main():
    sys.path.insert(0, str(ROOT))
    from scripts.helpers.runner_logging import configure_runner_logging

    configure_runner_logging()
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--recipe")
    selector.add_argument("--family")
    selector.add_argument("--case")
    selected, _ = selector.parse_known_args()
    if __package__:
        from . import image_suite
    else:
        import image_suite

    if selected.family == "model-features" and selected.recipe in (PRESSURE_RECIPE, *image_suite.RECIPES):
        if __package__:
            from . import model_features_suite
        else:
            import model_features_suite
        model_features_suite.main(image_suite, sys.modules[__name__], run_with_live_logs)
        return

    if selected.recipe == PRESSURE_RECIPE:
        if selected.family == "precision" and selected.case in (None, "sdnq-int4-leaf", "sdnq-int4-ddo"):
            if __package__:
                from . import sdnq_suite
            else:
                import sdnq_suite
            sdnq_suite.main(image_suite, run_with_live_logs)
            return
        video_runtime_main(image_suite)
        return
    if selected.recipe in image_suite.RECIPES:
        if selected.family == "precision":
            if __package__:
                from . import sdnq_suite
            else:
                import sdnq_suite
            sdnq_suite.main(image_suite, run_with_live_logs)
            return
        image_suite.main(run_with_live_logs)
        return
    args = parse_args()
    plan = build_plan(args)
    parent = args.output_root.resolve() / args.platform / args.recipe
    prefix = f"{args.case}-{datetime.now():%Y%m%d-%H%M%S}-"
    if args.execute:
        parent.mkdir(parents=True, exist_ok=True)
        directory = parent / f"{prefix}{uuid4().hex}"
        # Keep the output parent's access permissions for benchmark reports.
        directory.mkdir()
    else:
        directory = parent / f"{prefix}<run-id>"
    plan["command"] += ["--output-dir", str(directory)]
    print(json.dumps(plan, indent=2), flush=True)
    render = subprocess.list2cmdline if args.platform == "windows" else shlex.join
    print(render(plan["command"]), flush=True)
    if args.execute:
        print(f"Capture directory: {directory}", flush=True)
        execute(plan, directory)
        print(directory / "summary.csv")


if __name__ == "__main__":
    main()
