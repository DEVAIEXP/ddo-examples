"""Plan or capture one approved pipeline benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4


BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
RECIPE = "ltx23-video-5s"
CASE = "sdpa-baseline"
OFFICIAL_LEAF_CASE = "official-leaf"
NORMALIZATION_CASE = "sdpa-normalization-1024"
REFERENCE_CASE = "sdpa-reference-5s"
FAMILIES = ("runtime", "model-features", "attention")
RUNTIME_VARIANTS = {
    OFFICIAL_LEAF_CASE: ("linear", "diffusers_leaf_offload_compat"),
    "one-shot-fast": ("linear", "one_shot_fast"),
    "low-ram-safe": ("linear", "low_ram_safe"),
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows", "linux", "wsl"))
    parser.add_argument("--recipe", choices=(RECIPE,), default=RECIPE)
    parser.add_argument("--case", choices=CASES, default=CASE)
    parser.add_argument("--family", choices=FAMILIES, help="Comparison family for this run.")
    parser.add_argument("--list-cases", action="store_true", help="List available plans without running them.")
    parser.add_argument("--vram-gib", type=int, choices=CAPACITIES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fixtures-dir", type=Path, default=BENCHMARK_DIR / "inputs" / RECIPE,
                        help="Local conditioning files. See docs/benchmarks/recording.md.")
    parser.add_argument("--custom-blocks", type=Path,
                        default=ROOT / "modular_diffusers/custom_blocks/ltx2_video")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "benchmarks")
    parser.add_argument("--execute", action="store_true", help="Run the reviewed plan once.")
    args = parser.parse_args()
    if args.list_cases:
        if args.execute:
            parser.error("--list-cases cannot be used with --execute.")
        print(json.dumps({"recipe": args.recipe, "cases": {
            case: metadata for case, metadata in CASES.items()
            if args.family is None or args.family in metadata["families"]
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


def build_plan(args):
    platform = current_platform()
    if platform != args.platform:
        raise ValueError(f"Requested platform {args.platform} does not match {platform}.")
    elevated = bool(ctypes.windll.shell32.IsUserAnAdmin()) if platform == "windows" else None
    official_leaf = args.case == OFFICIAL_LEAF_CASE
    purges = {phase: not official_leaf and bool(elevated) and phase in ("before-run", "before-transformer")
              for phase in PURGES}
    reference = args.case == REFERENCE_CASE
    if reference:
        if platform != "windows" or not elevated or args.vram_gib != 8:
            raise ValueError("The reference requires elevated Windows and the 8 GiB profile.")
        purges["before-run"] = False
    checkpoint = args.checkpoint
    if checkpoint is None:
        cache = Path(os.environ.get("HF_HOME", r"F:\hf_cache" if platform == "windows" else Path.home() / ".cache/huggingface"))
        checkpoint = cache / "hub/models--OzzyGT--LTX-2.3-Distilled-1.1/snapshots" / SNAPSHOT
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
        command.extend((
            "--ddo-vmm-resident-weight-budget-gb", str(resident),
            "--ddo-vmm-total-staging-gb", "1.5",
            "--ddo-vmm-source-cache-gb", cache_budget, "--ddo-vmm-source-registered-gb", cache_budget,
        ))
    command.extend((
        "--distilled-model", str(checkpoint.resolve()), "--custom-blocks", str(args.custom_blocks.resolve()),
        "--local-files-only", "--image", str(image), "--audio", str(audio), "--use-audio-condition",
        "--prompt", "the man plays the guitar and sings on an old western town",
        "--width", "1280", "--height", "704", "--seconds", "5", "--frame-rate", "24",
        "--steps", "8", "--seed", "43", "--video-self-attention", "default",
        "--feed-forward-chunk-size", "1024" if reference else "0",
        "--normalization-chunk-size", normalization_chunk,
        "--reuse-rotary-embeddings" if reference else "--no-reuse-rotary-embeddings",
    ))
    if backend == "vmm":
        command.append("--ddo-vmm-execution-backing-reclaim" if args.case == "vmm-fast-reclaim"
                       else "--no-ddo-vmm-execution-backing-reclaim")
    command.extend((
        "--stream-video-decode", "--decode-device", "cuda:0",
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
        command.extend(f"--{'' if enabled else 'no-'}purge-{phase}" for phase, enabled in purges.items())
    return {
        "platform": platform, "recipe": args.recipe, "case": args.case, "windows_elevated": elevated,
        "family": args.family, "case_role": CASES[args.case]["role"],
        "control_case": CASES[args.case]["control_case"],
        "nominal_vram_gib": args.vram_gib, "resident_budget_gib": resident if backend == "vmm" else None,
        "budget_status": ("not-applicable" if official_leaf else "preset-managed" if backend == "linear" else
                          "reference" if args.vram_gib == 8 else "unvalidated"),
        "purges_requested": purges, "environment_overrides": {"DIFFUSERS_ATTN_BACKEND": "native"},
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
                                    f"listed in docs/benchmarks/recording.md: {missing_fixtures}")
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
        if metrics.get("status") != "completed" or metrics.get("decoded_media", {}).get("frames") != 121:
            raise ValueError("The runner did not report a completed 121-frame result.")
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
        plan["nvidia_driver_versions"] = attention.get("nvidia_driver_versions")
        plan["output_sha256"] = digest(metrics["output_path"])
        denoise = next(step for step in metrics["steps"] if step["name"] == "denoise")
        decode = next((step for step in metrics["steps"] if step["name"] == "decode_and_save"), {})
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
        print(json.dumps(row, indent=2), flush=True)
        plan["status"] = "completed"
    except BaseException as error:
        plan.update(status="failed", error=str(error))
        raise
    finally:
        write_manifest(directory, plan)


def main():
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
