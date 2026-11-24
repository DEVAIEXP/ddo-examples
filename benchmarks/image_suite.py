"""RC5 image recipes, evidence inventory and resumable capture."""

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "benchmarks"
RUNTIME = ("official-leaf", "one-shot-fast", "low-ram-safe", "nvme-baseline", "vmm-fast")
WSL_RUNTIME = ("official-leaf", "wsl-compat", "vmm-fast")
LINUX_RUNTIME = ("official-leaf", "one-shot-fast", "low-ram-safe", "vmm-fast")
LOW_RAM_VAE_PURGE = "low-ram-safe-pre-vae-purge"
ATTENTION = {"official-leaf": "default", "cudnn-sdpa": "cudnn", "flash-attn2": "flash-attn2"}
RECIPES = {
    "ltx23-image": {
        "folder": "ltx2_image", "model": "elismasilva/ltx2.3-image-distilled-1.1",
        "snapshot": "5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9",
        "blocks": "6e0e04e9305a05040237456be6bc35635cd630d7",
        "size": (1280, 704), "steps": 8, "seed": 43,
        "prompt": "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, "
                  "with bright colors, realistic fur detail, and playful viral-pet energy.",
    },
    "qwen-image21": {
        "folder": "qwen_image_21", "model": "Qwen/Qwen-Image-2.1",
        "snapshot": "d26bb61231c349cf6b7896fa83353113880e1ba3",
        "blocks": "cd1efe4ad857b3e15fe5cb675842b4283750b805",
        "size": (1024, 1024), "steps": 20, "seed": 42,
        "prompt": "A capybara wearing a wizard hat, reading by candlelight, oil painting",
    },
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hub_snapshot(repo_id, revision):
    """Locate a pinned snapshot in the standard Hugging Face hub cache.

    The cache follows huggingface_hub: HF_HUB_CACHE, then HF_HOME/hub, then the
    user default.
    """
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision


def ensure_hub_snapshot(path, required, allow_patterns=None):
    """Download a pinned hub-cache snapshot whose required file is missing.

    Only paths with the hub-cache snapshot layout are downloaded, at their
    pinned revision. Other local folders are never modified.
    """
    path = Path(path)
    if (path / required).is_file():
        return
    parts = path.parts
    if len(parts) < 3 or parts[-2] != "snapshots" or not parts[-3].startswith("models--"):
        return
    repo_id = parts[-3][len("models--"):].replace("--", "/", 1)
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} at revision {parts[-1]} into {path.parents[2]}", flush=True)
    snapshot_download(repo_id=repo_id, revision=parts[-1], cache_dir=path.parents[2],
                      allow_patterns=allow_patterns)


CHECKPOINT_HELP = ("Optional local checkpoint snapshot. By default the pinned revision is used from the "
                   "Hugging Face cache and downloaded by --preflight or --execute when missing.")
ATTENTION_FLAGS = ("--image-self-attention", "--image-joint-attention", "--video-self-attention")


def describe_case(case):
    """Summarize what one planned case runs, from its routes and measured command."""
    command = case["stages"][-1]["command"]

    def value(flag):
        return command[command.index(flag) + 1] if flag in command else None

    routes = case.get("routes", {})
    if routes.get("transformer") == "official_diffusers_leaf":
        parts = ["Diffusers Leaf reference (not a DDO preset)"]
    else:
        backend = value("--ddo-backend")
        parts = [f"{backend} / {case['preset']}" if backend else f"preset {case['preset']}",
                 f"transformer {routes.get('transformer')}"]
    if case.get("sdnq_checkpoint"):
        parts.append("saved SDNQ INT4")
    if "--transformer-fp8-weight-only" in command:
        parts.append("saved FP8 weight-only transformer")
    attention = next((value(flag) for flag in ATTENTION_FLAGS if value(flag) not in (None, "default")), None)
    if attention:
        parts.append(f"attention {attention}")
    if routes.get("vae"):
        parts.append(f"VAE {routes['vae']}")
    if "--purge-before-vae" in command:
        parts.append("purge before VAE")
    if "--warm-process-reuse" in command:
        parts.append("two generations in one process")
    return ", ".join(parts)


def print_cases(rows):
    """Print one case name and its summary per line."""
    rows = list(rows)
    width = max(len(name) for name, _ in rows)
    for name, description in rows:
        print(f"{name:<{width}}  {description}")


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def metrics_file(directory):
    paths = sorted([*directory.glob("metrics.json"), *directory.glob("metrics/*.json")])
    if len(paths) != 1:
        raise ValueError(f"Expected one metrics JSON in {directory}, found {len(paths)}")
    return paths[0]


def inventory(recipe):
    """Keep original artifacts untouched and explain why a run needs review."""
    expected = RECIPES[recipe]
    rows = []
    for folder in sorted((OUTPUT / "windows" / recipe).glob("rc5-*")):
        if not folder.is_dir():
            continue
        row = {"directory": str(folder), "classification": "pending-new-run"}
        try:
            path = metrics_file(folder)
            report = read_json(path)
            workload = report.get("workload", report)
            pngs = list(folder.glob("*.png"))
            row.update(metrics=str(path), metrics_sha256=digest(path),
                       shape=[workload.get("width"), workload.get("height")],
                       profile=report.get("ddo_profile"), png_sha256={p.name: digest(p) for p in pngs},
                       companion_files=[p.name for p in folder.iterdir() if p.is_file() and p.suffix != ".png"])
            if row["shape"] == list(expected["size"]) and (
                    pngs and report.get("status", "completed") == "completed" or report.get("ddo_profile")):
                row.update(classification="diagnostic-evidence",
                           reason="Raw report has no matching reviewed suite identity.")
            else:
                row["reason"] = "Different workload, failed execution or incomplete artifacts."
        except (ValueError, OSError) as error:
            row["reason"] = str(error)
        rows.append(row)
    return rows


def vmm_attention_command(command, recipe, variant):
    if recipe == "ltx23-image":
        return command + ["--image-self-attention", ATTENTION[variant]]
    if variant != "cudnn-sdpa":
        raise ValueError("Qwen VMM Fast attention supports only cuDNN SDPA.")
    command = command.copy()
    command[command.index("--image-joint-attention") + 1] = ATTENTION[variant]
    return command


def build_plan(args):
    from scripts.helpers.model_sources import source_argument
    reuse = args.family == "reuse"
    if reuse and (args.recipe != "ltx23-image" or getattr(args, "platform", "windows") != "windows"):
        raise ValueError("The reuse recipe is prepared only for Windows LTX Image.")
    if reuse and (getattr(args, "purge_before_transformer_only", False)
                  or getattr(args, "attention_runtime", None)
                  or getattr(args, "component_reference_manifest", None)):
        raise ValueError("The reuse recipe preserves default SDPA and disables purge and component overrides.")
    if getattr(args, "attention_runtime", None) == "vmm-fast":
        runtime_args = argparse.Namespace(**vars(args))
        runtime_args.family, runtime_args.case, runtime_args.attention_runtime = "runtime", "vmm-fast", None
        plan = build_plan(runtime_args)
        base = plan["cases"]["vmm-fast"]
        variants = (args.case,) if args.case else (
            ("cudnn-sdpa", "flash-attn2") if args.recipe == "ltx23-image" else ("cudnn-sdpa",)
        )
        plan["family"] = "attention"
        plan["cases"] = {}
        for variant in variants:
            command = vmm_attention_command(base["stages"][-1]["command"], args.recipe, variant)
            plan["cases"][variant] = {**base, "stages": [
                {"name": "operator-check", "command": command + ["--trace-transformer-attention"]},
                {"name": "measure", "command": command},
            ]}
        return plan
    recipe = RECIPES[args.recipe]
    ltx = args.recipe == "ltx23-image"
    platform = getattr(args, "platform", "windows")
    wsl = platform == "wsl"
    checkpoint = (args.checkpoint or hub_snapshot(recipe["model"], recipe["snapshot"])).resolve()
    blocks = (args.custom_blocks or ROOT / "modular_diffusers/custom_blocks" / recipe["folder"]).resolve()
    component_control = args.family == "components"
    component_cases = ("vmm-fast", "nvme-baseline")
    cases = component_cases if component_control else (
        tuple(case for case in ATTENTION if ltx or case != "flash-attn2")
        if args.family == "attention" else RUNTIME
    )
    if ltx and args.family == "runtime":
        cases = (*RUNTIME[:3], LOW_RAM_VAE_PURGE, *RUNTIME[3:])
    if wsl and args.family == "runtime":
        cases = WSL_RUNTIME
    if platform == "linux" and args.family == "runtime":
        cases = LINUX_RUNTIME
    if reuse:
        cases = ("warm-process",)
    if args.case:
        if args.family == "attention" and args.case == "flash-attn2" and not ltx:
            raise ValueError("Qwen joint attention uses a text padding mask that the FlashAttention 2 backend cannot preserve.")
        if args.case not in cases:
            raise ValueError(f"{args.case} is not in the {args.family} family")
        cases = (args.case,)
    plans = {}
    for case in cases:
        leaf = case in ATTENTION
        fast = case == "vmm-fast"
        qwen_vae = getattr(args, "vae_model_path", None)
        native_vae = fast and not wsl and not component_control and (ltx or qwen_vae is not None)
        nvme = case == "nvme-baseline"
        pre_vae_purge = case == LOW_RAM_VAE_PURGE
        low_ram = case == "low-ram-safe" or pre_vae_purge
        backend = "vmm" if fast or nvme else "linear"
        preset = "low_ram_safe" if low_ram else {
            "vmm-fast": "auto", "nvme-baseline": "nvme_baseline",
        }.get(case, case.replace("-", "_"))
        suffix = "_distilled" if ltx else ""
        runner = "run_diffusers_leaf" if leaf else "run_dynamic_modular" + ("_vmm" if backend == "vmm" else "") + suffix
        command = [
            sys.executable, "-B", "-u", str(ROOT / "scripts" / recipe["folder"] / (runner + ".py")),
            "--model-path", str(checkpoint), "--local-files-only", "--prompt", recipe["prompt"],
            "--width", str(recipe["size"][0]), "--height", str(recipe["size"][1]),
            "--steps", str(recipe["steps"]), "--seed", str(recipe["seed"]),
            "--memory-monitor", "rss", "--metrics-level", "2", "--save-metrics",
        ]
        if native_vae:
            vae_path = checkpoint / "vae" if ltx else source_argument(qwen_vae)
            command += ["--vae-vmm-offload", "--vae-model-path", str(vae_path),
                        "--vae-vmm-resident-weight-budget-gb", "1.5" if ltx else "0.375"]
        elif reuse:
            pass  # Omit --vae-leaf-offload to use the runner's resident VAE path.
        else:
            command += ["--vae-leaf-offload"]
        if ltx or not leaf:
            command += ["--custom-blocks", str(blocks)]
        if ltx:
            command += ["--guidance-scale", "1", "--guidance-rescale", "0.7", "--no-pag"]
        if leaf:
            flag = "--image-self-attention" if ltx else "--image-joint-attention"
            command += [flag, ATTENTION[case]]
            if wsl:
                command += ["--no-stream-offload"]
        else:
            command += ["--ddo-backend", backend, "--ddo-preset", preset]
            if wsl and not ltx and backend == "linear":
                command += ["--image-joint-attention", "default"]
        if backend == "vmm":
            command += ["--no-ddo-vmm-execution-backing-reclaim", "--text-encoder-native-vmm" if fast and not wsl
                        else "--no-text-encoder-native-vmm"]
            if not ltx:
                command += ["--no-vae-no-temporal-cache"]
                if wsl or native_vae or (platform == "linux" and component_control and fast):
                    command += ["--image-joint-attention", "default"]
            if nvme:
                command += ["--ddo-vmm-resident-weight-budget-gb", str(args.nvme_resident_gib),
                            "--ddo-vmm-total-staging-gb", "1.5",
                            "--ddo-vmm-source-cache-gb", "0", "--ddo-vmm-source-registered-gb", "0"]
            else:
                staging, source = ("1.5", "24.625") if ltx else ("0.625", "16")
                command += ["--ddo-vmm-total-staging-gb", "0.375" if ltx and wsl else staging,
                            "--ddo-vmm-source-cache-gb", source,
                            "--ddo-vmm-source-registered-gb", "0.5" if wsl else source]
        if not leaf:
            width, height = recipe["size"]
            profile_case = "low-ram-safe" if low_ram else case
            if wsl and fast:
                profile_case += "-wsl"
            profile = f"{args.recipe}-{width}x{height}-{recipe['steps']}steps-{profile_case}" if not nvme else None
            if profile:
                command += ["--ddo-profile", profile]
        else:
            profile = None
        transformer_only = getattr(args, "purge_before_transformer_only", False)
        no_purge = (native_vae and not ltx) or getattr(args, "no_purge", False) or (
            (leaf or reuse or (ltx and (case == "one-shot-fast" or low_ram))) and not transformer_only
        )
        if platform == "windows":
            command += ["--no-purge-before-run" if transformer_only or no_purge else "--purge-before-run",
                        "--no-purge-before-transformer" if no_purge else "--purge-before-transformer"]
            if not (ltx and leaf):
                command += ["--no-purge-after-run"]
            if not leaf:
                command += ["--no-purge-after-text-encoder"]

        stages = []
        if profile and not getattr(args, "component_reference_manifest", None):
            build = command + ["--build-ddo-profile"]
            if fast:
                build = build.copy()
                build[build.index("--steps") + 1] = "3"
            stages.extend({"name": f"profile-{index + 1}", "command": build}
                          for index in range(2 if fast else 1))
        if leaf and args.family == "attention":
            stages.append({"name": "operator-check", "command": command + ["--trace-transformer-attention"]})
        stages.append({"name": "measure", "command": command + (["--purge-before-vae"] if pre_vae_purge else [])})
        if reuse:
            stages[-1]["command"] += ["--warm-process-reuse", "--image-self-attention", "default"]
        if case == "wsl-compat" or reuse:
            # RC5 treats the positive preset budget as an override of the saved profile.
            for stage in stages:
                stage["environment"] = {"DDO_RESIDENT_MODULE_BUDGET_GB": "0"}
        if wsl and not ltx and args.family == "runtime" and case == "official-leaf":
            # Keep this tested WSL control exception out of shared runner defaults.
            for stage in stages:
                stage.setdefault("environment", {})["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"
        plans[case] = {
            "stages": stages, "preset": preset if not leaf else None,
            "routes": {"transformer": "official_diffusers_leaf" if leaf else
                       "vmm_fast_runtime" if fast else "vmm_runtime" if nvme else "linear_runtime",
                       "encoder": "native_vmm" if fast and not wsl else "diffusers_leaf",
                       "vae": "native_vmm" if native_vae else "resident" if reuse else "diffusers_leaf"},
            "dispatcher": True if fast else None,
            "profile": profile,
            "required_phase_memory": (["denoise", "Decode VAE" if ltx else "decode"]
                                      if not reuse else []),
            "hold": ("Complete the encoder/VAE route study before accepting this runtime row."
                     if backend == "vmm" and not component_control and not wsl and not native_vae else None),
        }
        if platform == "linux" and leaf:
            plans[case]["require_streamed_leaf"] = True
        if reuse:
            plans[case]["reuse"] = {"preparations": 2, "generations": 2, "decoded_outputs": 1,
                                    "total_scope": "two preparations, two generations and final decode/save"}
    allocator_environment = (
        {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
        if getattr(args, "expandable_segments", False) else {}
    )
    return {"schema": 1, "platform": platform, "recipe": args.recipe, "nominal_vram_gib": 8,
            "family": args.family, "checkpoint": str(checkpoint), "custom_blocks": str(blocks),
            "checkpoint_revision": recipe["snapshot"], "blocks_revision": recipe["blocks"],
            "environment_overrides": {"DIFFUSERS_ATTN_BACKEND": "native", "HF_HUB_OFFLINE": "1",
                                      "PYTHONNOUSERSITE": "1", **allocator_environment}, "cases": plans}


def child_environment(plan, directory=None):
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("DDO_", "DIFFUSERS_", "PYTHON", "TORCH", "CUDA_"))
           and key.upper() not in {"HF_MODULES_CACHE", "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF",
                                   "PYTORCH_ALLOC_CONF"}}
    env.update(plan["environment_overrides"])
    if directory:
        env.update(DDO_PROFILE_DIR=str(directory / "profiles"),
                   HF_MODULES_CACHE=str(directory / "hf-modules"))
    return env


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


SOURCE_DISPATCHER_EXPORTS = {"ddo_vmm_source_dispatcher_enabled", "ddo_vmm_source_cache_select_dispatcher"}


def native_exports(path):
    """Read exported symbol names from a PE (Windows) or ELF (Linux) file without loading it."""
    import struct

    data = Path(path).read_bytes()
    if data[:2] == b"MZ":
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            raise ValueError(f"Not a PE file: {path}")
        sections, optional_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
        optional = pe + 24
        directories = optional + (112 if struct.unpack_from("<H", data, optional)[0] == 0x20B else 96)
        export_rva = struct.unpack_from("<I", data, directories)[0]
        # Section header fields: VirtualSize, VirtualAddress, SizeOfRawData, PointerToRawData.
        table = [struct.unpack_from("<IIII", data, optional + optional_size + 40 * index + 8)
                 for index in range(sections)]

        def offset(rva):
            for virtual_size, virtual_address, raw_size, raw_pointer in table:
                if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
                    return rva - virtual_address + raw_pointer
            raise ValueError(f"RVA outside the PE sections: {rva:#x}")

        if not export_rva:
            return set()
        directory = offset(export_rva)
        count, names = struct.unpack_from("<I", data, directory + 24)[0], struct.unpack_from("<I", data, directory + 32)[0]
        result = set()
        for index in range(count):
            start = offset(struct.unpack_from("<I", data, offset(names) + 4 * index)[0])
            result.add(data[start:data.index(b"\0", start)].decode("ascii"))
        return result
    if data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1:
        section_offset = struct.unpack_from("<Q", data, 0x28)[0]
        entry_size, count = struct.unpack_from("<HH", data, 0x3A)
        headers = [struct.unpack_from("<IIQQQQIIQQ", data, section_offset + entry_size * index)
                   for index in range(count)]
        result = set()
        for kind, _flags, _address, start, size, link, entry in (
                (h[1], h[2], h[3], h[4], h[5], h[6], h[9]) for h in headers):
            if kind != 11:  # SHT_DYNSYM
                continue
            strings = headers[link][4]
            for symbol in range(start, start + size, entry):
                name, info, _other, section_index = struct.unpack_from("<IBBH", data, symbol)
                if name and section_index and info >> 4 in (1, 2):  # defined GLOBAL or WEAK
                    result.add(data[strings + name:data.index(b"\0", strings + name)].decode("ascii"))
        return result
    raise ValueError(f"Unsupported native library format: {path}")


def execution_identity(plan, expected_dll=None):
    """Record the execution identity; download a missing pinned checkpoint first.

    Without expected_dll, the installed library must match its own wheel
    artifact record. An explicit expected_dll must also match.
    """
    from scripts.helpers.model_sources import resolve_plan_sources
    ensure_hub_snapshot(plan["checkpoint"], "model_index.json")
    resolve_plan_sources(plan)
    import ddo_vmm_bootstrap as bootstrap
    from packaging.version import Version

    packages = ("diffusers-dynamic-offloader", "ddo-vmm-allocator", "torch", "diffusers",
                "transformers", "huggingface-hub", "flash-attn")
    versions = {}
    for name in packages:
        try:
            distribution = metadata.distribution(name)
            versions[name] = distribution.version
            if name in packages[:2] and (Version(distribution.version) < Version("0.1.0rc5")
                    or json.loads(distribution.read_text("direct_url.json") or "{}")
                    .get("dir_info", {}).get("editable")):
                raise ValueError("Use installed DDO and allocator wheels at RC5 or later.")
        except metadata.PackageNotFoundError:
            if name != "flash-attn":
                raise
            versions[name] = None
    if bootstrap._default_allocator_root() is not None:
        raise ValueError("The allocator must be loaded from the installed environment.")
    dll = bootstrap._discover_library(None, bootstrap._installed_allocator_root())
    if dll is None:
        raise ValueError("The installed allocator wheel does not provide a native library.")
    artifact = read_json(dll.parent / "artifact.json")
    actual = digest(dll)
    if (expected_dll is not None and actual != expected_dll) or actual != artifact.get("sha256"):
        raise ValueError("Packaged native library identity differs from the reviewed artifact.")
    # Some ABI reports list only export counts; read the verified binary's own table then.
    listed = artifact.get("abi", {}).get("required_exports")
    exports = set(listed) if isinstance(listed, list) else native_exports(dll)
    if not SOURCE_DISPATCHER_EXPORTS <= exports:
        raise ValueError("The packaged native library does not export the source dispatcher ABI.")
    checkpoint = Path(plan["checkpoint"])
    if checkpoint.name != plan["checkpoint_revision"] or not (checkpoint / "model_index.json").is_file():
        raise ValueError(f"Expected the reviewed checkpoint snapshot: {checkpoint}")
    for case in plan["cases"].values():
        if case["routes"].get("vae") != "native_vmm":
            continue
        command = case["stages"][-1]["command"]
        vae = Path(command[command.index("--vae-model-path") + 1])
        if not (vae / "config.json").is_file() or not any(vae.glob("*.safetensors")):
            raise ValueError(f"Expected native VAE source files in the reviewed checkpoint: {vae}")
    blocks = Path(plan["custom_blocks"])
    needs_blocks = any("--custom-blocks" in case["stages"][-1]["command"] for case in plan["cases"].values())
    if needs_blocks and (git(blocks, "rev-parse", "HEAD") != plan["blocks_revision"]
                         or git(blocks, "status", "--porcelain")):
        raise ValueError("Custom blocks must be at the reviewed clean revision.")
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip().splitlines()
    if len(driver) != 1 or not 8000 <= int(driver[0].rsplit(",", 1)[-1]) <= 8192:
        raise ValueError("This recipe requires the reviewed single 8 GiB GPU configuration.")
    return {"plan": plan, "python": sys.version, "packages": versions, "gpu": driver[0],
            "dll": str(dll), "dll_sha256": actual, "examples_revision": git(ROOT, "rev-parse", "HEAD"),
            "runner_sha256": {str(p.relative_to(ROOT)): digest(p)
                              for folder in (ROOT / "benchmarks", ROOT / "scripts")
                              for p in sorted(folder.rglob("*.py"))}}


def walk_field(value, name):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == name:
                yield child
            yield from walk_field(child, name)
    elif isinstance(value, list):
        for child in value:
            yield from walk_field(child, name)


def verify_phase_memory(report, names):
    """Refuse a new measurement whose isolated phase telemetry is incomplete."""
    for name in names:
        matches = [step for step in report.get("steps", []) if step.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one separately monitored phase: {name}")
        step = matches[0]
        monitor = step.get("memory_monitor", {})
        if (step.get("status", "completed") != "completed" or step.get("ram_metric") != "rss"
                or monitor.get("ram_metric") != "rss" or monitor.get("error_type") is not None
                or monitor.get("samples", 0) < 1):
            raise ValueError(f"Incomplete phase memory telemetry: {name}")
        for key in ("elapsed_sec", "peak_vram_absolute_gb", "peak_ram_gb"):
            value = step.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Missing or invalid {name} metric: {key}")


def verify_stage(recipe, case, stage, output, profile_dir):
    path = metrics_file(output)
    report = read_json(path)
    if report.get("status", "completed") != "completed":
        raise ValueError("Runner reported an incomplete execution.")
    workload = report.get("workload", report)
    if ((workload.get("width"), workload.get("height")) != recipe["size"]
            or workload.get("seed") != recipe["seed"]):
        raise ValueError("The report does not match the requested shape or seed.")
    transformer = report.get("dynamic_offload_reports", {}).get("transformer", {})
    executed_mode = (transformer.get("configuration", {}).get("dynamic_offload_execution_mode")
                     or report.get("dynamic_offload_execution_mode"))
    if case["preset"] and executed_mode != case["routes"]["transformer"]:
        raise ValueError(f"Unexpected transformer runtime: {executed_mode!r}")
    if not case["preset"] and report.get("offload", {}).get("offload_type") != "leaf_level":
        raise ValueError("Runner did not report official leaf-level offload.")
    if any(value is True for value in walk_field(report, "dynamic_offload_vmm_execution_backing_reclaim")):
        raise ValueError("Execution-backing reclaim was enabled.")
    if "standby_purges" in report:
        purged_phases = {
            phase for phase, entry in report["standby_purges"].items()
            if (entry.get("result") or {}).get("purged") is True
        }
    else:
        purged_phases = {
            key.removeprefix("purge_windows_standby_").removesuffix("_1")
            for key, value in report.items()
            if key.startswith("purge_windows_standby_") and isinstance(value, dict) and value.get("purged") is True
        }
    stage_command = next(item["command"] for item in case["stages"] if item["name"] == stage)
    if case.get("require_streamed_leaf"):
        if report.get("offload", {}).get("use_stream") is not True:
            raise ValueError("LTX Image did not report streamed Leaf offload.")
    if recipe["folder"] == "qwen_image_21" and "--no-stream-offload" in stage_command:
        if report.get("offload", {}).get("use_stream") is not False:
            raise ValueError("Qwen did not report non-streamed Leaf offload.")
        if report.get("transformer_attention_requested") != "native":
            raise ValueError("Qwen did not report default SDPA for the runtime control.")
    flag = "--image-joint-attention"
    if case["preset"] and flag in stage_command:
        selected = stage_command[stage_command.index(flag) + 1]
        expected_backend = {"default": "native", "cudnn": "_native_cudnn"}[selected]
        if report.get("transformer_attention_requested") != expected_backend:
            raise ValueError(f"Qwen did not report the requested joint attention backend: {selected}.")
        if case["routes"]["encoder"] == "diffusers_leaf":
            encoder = report.get("dynamic_offload_reports", {}).get("text_encoder", {})
            setup = encoder.get("setup", {})
            if (encoder.get("route") != "diffusers_group_offload"
                    or setup.get("offload_type") != "leaf_level" or setup.get("use_stream") is not False):
                raise ValueError("Qwen did not report non-streamed Leaf offload for the text encoder.")
    expected_purges = {
        flag.removeprefix("--purge-").replace("-", "_")
        for flag in stage_command if flag.startswith("--purge-")
    }
    if purged_phases != expected_purges:
        raise ValueError(f"Expected successful purges {sorted(expected_purges)}; observed {sorted(purged_phases)}.")
    artifacts = [path]
    if stage.startswith("profile-"):
        profile = report.get("ddo_profile") or {}
        expected_stage = {"floor_measured", "final"} if stage == "profile-1" else {"final"}
        if profile.get("status") != "saved" or profile.get("name") != case["profile"]:
            raise ValueError("Runner did not save the requested profile.")
        if case["dispatcher"] and profile.get("build_stage") not in expected_stage:
            raise ValueError("VMM calibration did not reach the required stage.")
        path = Path(profile["path"]).resolve()
        if not path.is_relative_to(profile_dir.resolve()) or not path.is_file():
            raise ValueError("Saved profile is outside the case profile directory.")
    else:
        from PIL import Image

        verify_phase_memory(report, case.get("required_phase_memory", []))

        images = list(output.glob("*.png"))
        if len(images) != 1:
            raise ValueError("Expected exactly one PNG.")
        with Image.open(images[0]) as image:
            if image.format != "PNG" or image.size != recipe["size"]:
                raise ValueError("Output PNG differs from the requested workload.")
            image.load()
        for field in ("total_elapsed_sec", "global_peak_vram_absolute_gb", "global_peak_ram_gb"):
            value = report.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Missing or invalid metric: {field}")
        if report.get("memory_monitor", {}).get("ram_metric") != "rss":
            raise ValueError("The run did not record RSS memory metrics.")
        steps = workload.get("steps_requested", workload.get("num_inference_steps"))
        if steps != recipe["steps"]:
            raise ValueError(f"Unexpected step count: {steps!r}")
        if "feature_fixed_budget_bytes" in case:
            if __package__:
                from .model_features_suite import verify_fixed_budget
            else:
                from model_features_suite import verify_fixed_budget
            verify_fixed_budget(case, report)
        elif case["profile"] and (report.get("ddo_profile") or {}).get("status") != "applied":
            raise ValueError("The selected profile was not applied.")
        vae = report.get("vae_decode_route") or report.get("executed_controls", {}).get("vae_offload")
        if case["routes"]["vae"] == "native_vmm":
            reports = report.get("dynamic_offload_reports", {})
            for component in ("text_encoder", "vae"):
                info = reports.get(component, {})
                if (info.get("route") != "dynamic_offload"
                        or info.get("configuration", {}).get("dynamic_offload_execution_mode") != "vmm_fast_runtime"):
                    raise ValueError(f"Expected native VMM for {component}.")
            expected_vae = "native_vmm_component" if recipe["folder"] == "qwen_image_21" else "dynamic_offload"
            if vae != expected_vae:
                raise ValueError(f"Expected the native VMM VAE, observed {vae!r}.")
        elif case["routes"]["vae"] == "resident":
            if vae != "resident":
                raise ValueError(f"Expected the resident VAE, observed {vae!r}.")
        elif vae not in ("diffusers_leaf", "diffusers_group_offload"):
            raise ValueError(f"Expected the Leaf VAE, observed {vae!r}.")
        if case["preset"] == "wsl_compat" and recipe["folder"] == "qwen_image_21":
            setup = report.get("dynamic_offload_reports", {}).get("vae", {}).get("setup", {})
            if setup.get("offload_type") != "leaf_level" or setup.get("use_stream") is not False:
                raise ValueError("Qwen did not report non-streamed Leaf offload for the VAE.")
        artifacts.append(images[0])
        flag = "--image-self-attention"
        if case["dispatcher"] and flag in stage_command:
            selected = stage_command[stage_command.index(flag) + 1]
            expected_backend = {"default": "native", "cudnn": "_native_cudnn", "flash-attn2": "flash"}[selected]
            if report.get("transformer_attention_requested") != expected_backend:
                raise ValueError(f"Runner did not report the requested attention backend: {selected}.")
        if stage == "operator-check":
            requested = report.get("transformer_attention_requested")
            scope = "self" if recipe["folder"] == "ltx2_image" else "joint"
            if report.get("transformer_attention_scope") != scope:
                raise ValueError(f"Expected the {scope} attention scope.")
            operators = report.get("transformer_attention_operators", [])
            expected = {"native": ("scaled_dot_product_attention",), "_native_cudnn": ("cudnn_attention",),
                        "flash": ("flash_attn",)}.get(requested, ())
            if not any(token in operator for operator in operators for token in expected):
                raise ValueError(f"Could not confirm {requested} from operators: {operators}")
            if scope == "self" and requested == "flash" and not any(
                "scaled_dot_product_attention" in operator for operator in operators
            ):
                raise ValueError("Could not confirm SDPA alongside FlashAttention 2.")
    if case.get("reuse"):
        verify_reuse_report(report, case, stage, recipe)
    if case["dispatcher"] and not any(walk_field(report, "fast_source_cache_source_dispatcher_enabled")):
        raise ValueError("Native source dispatcher activation was not observed.")
    return {str(p): digest(p) for p in artifacts}


def reuse_measurements(report):
    """Read recorded repeat intervals without inferring cache hits or subtracting totals."""
    result = []
    for index in (1, 2):
        names = {"load_sec": f"load_transformer_prepare_{index}",
                 "attachment_sec": f"build_dynamic_offload_plan_prepare_{index}",
                 "denoise_sec": f"denoise_modular_pipe_call_repeat_{index}",
                 "prepare_through_cpu_readback_sec": f"prepare_through_cpu_readback_repeat_{index}"}
        row = {"repeat": index}
        for field, name in names.items():
            events = [event for event in report.get("events", []) if event.get("name") == name]
            if len(events) != 1:
                raise ValueError(f"Missing or duplicate reuse event: {name}")
            value = events[0].get("elapsed_sec")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid reuse latency: {name}")
            row[field] = value
        result.append(row)
    return result


def verify_reuse_report(report, case, stage, recipe):
    if (report.get("dynamic_offload_effective_preset") != "warm_process"
            or report.get("text_encoder_group_offload") is not True
            or report.get("text_encoder_dynamic_offload") is not False):
        raise ValueError("The reuse recipe requires warm_process and a Leaf text encoder.")
    if (report.get("dtype") != "torch.bfloat16" or report.get("guidance_scale") != 1
            or report.get("guidance_rescale") != 0.7 or report.get("pag_enabled") is not False):
        raise ValueError("The reuse report changed BF16/guidance/PAG controls.")
    profile = report.get("ddo_profile") or {}
    if profile.get("name") != case["profile"]:
        raise ValueError("The reuse report selected another preset's profile.")
    expected = 1 if stage.startswith("profile-") else 2
    if report.get("generation_repeats") != expected or report.get("transformer_prepare_repeats") != expected:
        raise ValueError("Unexpected reuse preparation or generation count.")
    if stage.startswith("profile-"):
        return
    if report.get("transformer_attention_requested") != "native":
        raise ValueError("The reuse report did not preserve default SDPA.")
    if report.get("dynamic_offload_cache_plan") is not True or report.get("dynamic_offload_cache_pinned_weights") is not True:
        raise ValueError("The reuse preset did not enable the plan and pinned-weight caches.")
    rows = reuse_measurements(report)
    timings = report.get("denoise_step_times_by_repeat")
    if (not isinstance(timings, list) or len(timings) != 2
            or any(not isinstance(row, list) or len(row) != recipe["steps"] for row in timings)):
        raise ValueError("The reuse report lacks step timings for both generations.")
    for row in timings:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in row):
            raise ValueError("Invalid reuse step latency.")
    for row in rows:
        if row["prepare_through_cpu_readback_sec"] < row["denoise_sec"]:
            raise ValueError("The reuse readback interval does not cover denoise.")


def measurement_summary(artifacts):
    path = next(Path(path) for path in artifacts if Path(path).suffix == ".json")
    report = read_json(path)
    result = {
        "total_sec": report["total_elapsed_sec"],
        "peak_gpu_absolute_gib": report["global_peak_vram_absolute_gb"],
        "peak_ram_rss_gib": report["global_peak_ram_gb"],
        "nvidia_driver_versions": report.get("nvidia_driver_versions"),
        "phases": report.get("steps", []),
        "purges": report.get("standby_purges", {
            key: value for key, value in report.items() if key.startswith("purge_windows_standby_")
        }),
    }
    if report.get("generation_repeats") == report.get("transformer_prepare_repeats") == 2:
        result["reuse"] = {"repeats": reuse_measurements(report), "decoded_outputs": 1,
                           "total_scope": "two preparations, two generations and final decode/save",
                           "cache_reuse_observed": None,
                           "cache_observation_scope": "Enabled cache configuration does not prove cache hits."}
    return result


def compare_pngs(reference, result):
    from PIL import Image, ImageChops, ImageStat

    def png(artifacts):
        return next(path for path in artifacts if Path(path).suffix == ".png")

    with Image.open(png(reference)) as left, Image.open(png(result)) as right:
        difference = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
        return {"rgb_identical": difference.getbbox() is None,
                "mean_absolute_8bit_per_channel": ImageStat.Stat(difference).mean,
                "max_absolute_8bit_per_channel": [high for _, high in difference.getextrema()]}


def verified_vmm_attention_reference(path, identity):
    """Match the runtime control and its applied profile before starting a child."""
    manifest_sha256 = digest(path)
    previous = read_json(path)
    old = previous["identity"]
    plan = identity["plan"]
    old_plan = old["plan"]
    control = old_plan["cases"].get("vmm-fast", {})
    record = previous["cases"].get("vmm-fast", {})

    def comparable(environment):
        result = {key: value for key, value in environment.items() if key != "plan"}
        result["packages"] = {key: value for key, value in result["packages"].items() if key != "flash-attn"}
        result["runner_sha256"] = {key: value for key, value in result["runner_sha256"].items()
                                   if key != "benchmarks/image_suite.py"}
        return result

    def workload(value):
        return {key: item for key, item in value.items() if key not in ("family", "cases")}

    if (old_plan["family"] != "runtime" or comparable(old) != comparable(identity)
            or workload(old_plan) != workload(plan) or record.get("status") != "completed"):
        raise ValueError("The SDPA reference does not match the reviewed runtime environment.")
    for name, case in plan["cases"].items():
        if (any(case[key] != control.get(key) for key in ("routes", "profile", "dispatcher", "preset"))
                or case["stages"][-1]["command"]
                != vmm_attention_command(control["stages"][-1]["command"], plan["recipe"], name)):
            raise ValueError("The attention variant differs from the SDPA runtime configuration.")
    artifacts, profile = record.get("artifacts", {}), record.get("profile_receipt", {})
    metrics = [Path(file) for file in artifacts if Path(file).suffix == ".json"]
    if (len(metrics) != 1 or len([file for file in artifacts if Path(file).suffix == ".png"]) != 1
            or len(profile) != 1 or not receipts_valid(artifacts) or not receipts_valid(profile)):
        raise ValueError("The SDPA reference metrics, PNG or profile is missing or has changed.")
    report = read_json(metrics[0])
    applied = report.get("ddo_profile") or {}
    saved = read_json(Path(next(iter(profile))))
    recommendation = saved["recommendation"]
    budget = recommendation.get("vmm_resident_weight_budget_bytes")
    if (report.get("transformer_attention_requested") != "native" or applied.get("status") != "applied"
            or applied.get("name") != control["profile"] or applied.get("build_stage") != "final"
            or not isinstance(budget, int) or budget <= 0
            or (applied.get("recommendation") or {}).get("vmm_resident_weight_budget_bytes") != budget
            or digest(path) != manifest_sha256):
        raise ValueError("The reference does not establish SDPA with the saved final resident budget.")
    return {"manifest": str(path), "manifest_sha256": manifest_sha256, "artifacts": artifacts,
            "profile_receipt": profile, "resident_budget_bytes": budget}


def verified_vae_leaf_reference(path, identity):
    previous = read_json(path)
    old_plan = previous["identity"]["plan"]
    plan = identity["plan"]
    case = plan["cases"]["vmm-fast"]
    native = old_plan["cases"]["vmm-fast"]
    command = native["stages"][-1]["command"].copy()
    if native["routes"] != {
            "transformer": "vmm_fast_runtime", "encoder": "native_vmm", "vae": "native_vmm"}:
        raise ValueError("The VAE control requires a completed native Qwen VMM Fast reference.")
    command.remove("--vae-vmm-offload")
    for flag in ("--vae-model-path", "--vae-vmm-resident-weight-budget-gb"):
        index = command.index(flag)
        del command[index:index + 2]
    command.insert(command.index("--custom-blocks"), "--vae-leaf-offload")
    if (case["stages"] != [{"name": "measure", "command": command}]
            or case["routes"] != {**native["routes"], "vae": "diffusers_leaf"}
            or case["profile"] != native["profile"]):
        raise ValueError("Only the VAE route may change in this component comparison.")
    attention_plan = {**old_plan, "family": "attention", "cases": {
        "cudnn-sdpa": {**native, "stages": [{"name": "measure", "command":
            vmm_attention_command(native["stages"][-1]["command"], plan["recipe"], "cudnn-sdpa")}],
        },
    }}
    if {k: v for k, v in plan.items() if k not in ("family", "cases")} != {
            k: v for k, v in old_plan.items() if k not in ("family", "cases")}:
        raise ValueError("The component workload differs from its native reference.")
    return verified_vmm_attention_reference(path, {**identity, "plan": attention_plan})


def reusable_measurements(state, previous_states):
    """A runtime Leaf measurement can also serve as the attention control."""
    identity = state["identity"]
    plan = identity["plan"]
    common = {k: v for k, v in identity.items() if k != "plan"}
    recipe = {k: v for k, v in plan.items() if k not in ("family", "cases")}
    for previous in previous_states:
        old_identity = previous["identity"]
        old_plan = old_identity["plan"]
        if ({k: v for k, v in old_identity.items() if k != "plan"} != common
                or {k: v for k, v in old_plan.items() if k not in ("family", "cases")} != recipe):
            continue
        for name, case in plan["cases"].items():
            record = state["cases"][name]
            old = previous["cases"].get(name, {})
            old_case = old_plan["cases"].get(name, {})
            if (record.get("stages") or old.get("status") != "completed" or case["profile"]
                    or old_case.get("routes") != case["routes"] or old_case["stages"][-1] != case["stages"][-1]
                    or not receipts_valid(old.get("artifacts", {}))):
                continue
            record["stages"] = {"measure": old["artifacts"]}
            record["reused_measurement"] = old["artifacts"]


@contextmanager
def session_lock(path):
    with path.open("a+b") as stream:
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another controller owns this session.") from error
        yield


def check_abandoned_child(path):
    """The bootstrap child repeats the attempt path in its command line."""
    if sys.platform == "linux":
        target = os.fsencode(path)
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                arguments = (entry / "cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError as error:
                raise RuntimeError("Cannot prove the previous attempt has exited.") from error
            if target in arguments or b"--output-dir=" + target in arguments:
                raise RuntimeError("Cannot prove the previous attempt has exited.")
        return
    env = dict(os.environ, DDO_BENCHMARK_ATTEMPT=str(path))
    code = ("$ErrorActionPreference='Stop'; $p=$env:DDO_BENCHMARK_ATTEMPT; "
            "$active=Get-CimInstance Win32_Process | Where-Object { "
            "$_.CommandLine -and $_.CommandLine.Contains($p) }; if ($active) { exit 7 }")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", code],
                            env=env, timeout=20, check=False)
    if result.returncode:
        raise RuntimeError("Cannot prove the previous attempt has exited.")


def receipts_valid(receipts):
    try:
        return bool(receipts) and all(digest(Path(path)) == sha for path, sha in receipts.items())
    except OSError:
        return False


def export_profile(source, name, destination=None):
    """Copy a verified recommendation to the runner's profile store."""
    from diffusers_dynamic_offloader.profiles import (
        dynamic_offload_profile_key,
        dynamic_offload_profile_path,
        load_dynamic_offload_profile,
    )

    record = read_json(source)
    context = record["context"]
    if load_dynamic_offload_profile(context, source.parent) != record:
        raise ValueError(f"Invalid source profile: {source}")
    context = {**context, "name": name}
    record.update(context=context, profile_key=dynamic_offload_profile_key(context))
    target = dynamic_offload_profile_path(context, destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
        json.dump(record, output, indent=2, sort_keys=True)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = load_dynamic_offload_profile(context, target.parent)
            if existing is None or existing.get("recommendation") != record["recommendation"]:
                raise ValueError(f"A different profile already exists at {target}.")
    finally:
        temporary.unlink()
    return target


def execute_session(path, state, run, *, check_child=check_abandoned_child, verify=None):
    plan = state["identity"]["plan"]
    cases = state["cases"]
    total = len(plan["cases"])
    completed = sum(item["status"] == "completed" for item in cases.values())
    print(f"Benchmark batch: {plan['recipe']} / {plan['family']}\n"
          f"Cases: {total} total | {completed} completed | {total - completed} remaining\n"
          f"Session: {path.parent}", flush=True)
    attention_reference = state["identity"].get("vmm_attention_reference")
    external_profile = (attention_reference or state["identity"].get("video_saved_profile")
                      or state["identity"].get("video_linear_saved_profile")
                      or state["identity"].get("vae_component_reference"))
    for case_index, (case_id, case) in enumerate(plan["cases"].items(), start=1):
        record = cases[case_id]
        if record["status"] == "completed":
            if not receipts_valid(record["artifacts"]):
                raise ValueError(f"Completed artifacts changed for {case_id}. Start a new session.")
            print(f"Case {case_index}/{total}: {case_id} | Already completed; skipping.", flush=True)
            continue
        if record["status"] == "running":
            check_child(Path(record["active_output"]))
        if case["hold"]:
            raise RuntimeError(f"{case_id}: {case['hold']}")
        reused_profile = external_profile
        if case.get("feature_profile_case"):
            if __package__:
                from .model_features_suite import control_profile
            else:
                from model_features_suite import control_profile
            reused_profile = control_profile(state, case["feature_profile_case"])
        completed = sum(item["status"] == "completed" for item in cases.values())
        directory = path.parent / case_id
        stages = record.setdefault("stages", {})
        profile_dir = directory / "profiles"
        if "profile_receipt" in record and not receipts_valid(record["profile_receipt"]):
            raise ValueError(f"Profile changed for {case_id}. Start a new session.")
        for stage_index, stage in enumerate(case["stages"], start=1):
            name = stage["name"]
            progress = f"Case {case_index}/{total}: {case_id} | Stage {stage_index}/{len(case['stages'])}: {name}"
            if name == "profile-2" and record.get("profile_build_stage") == "final":
                print(f"{progress} | Skipped: profile already final.", flush=True)
                continue
            if name in stages:
                if not receipts_valid(stages[name]):
                    raise ValueError(f"Stage artifacts changed for {case_id}/{name}.")
                print(f"{progress} | Already verified; skipping.", flush=True)
                continue
            output = directory / f"{name}-{uuid4().hex}"
            output.mkdir(parents=True)
            record.update(status="running", active_output=str(output))
            write_state(path, state)
            command = stage["command"] + ["--output-dir", str(output)]
            print(f"{progress} | Completed cases: {completed}/{total}\nOutput: {output}", flush=True)
            try:
                effective_case = case
                if name == "measure" and (case.get("feature_fixed_budget_case")
                                          or case.get("feature_control_exact_budget")):
                    if __package__:
                        from .model_features_suite import fixed_budget_measurement
                    else:
                        from model_features_suite import fixed_budget_measurement
                    effective_case, command = fixed_budget_measurement(state, case, record, command)
                env = child_environment(plan, directory)
                env.update(stage.get("environment", {}))
                if reused_profile:
                    receipt = reused_profile["profile_receipt"]
                    if not receipts_valid(receipt):
                        raise ValueError("The reference profile changed before execution.")
                    profile_dir = Path(next(iter(receipt))).parent
                    env["DDO_PROFILE_DIR"] = str(profile_dir)
                if (case["preset"] == "low_ram_safe"
                        and state["identity"].get("packages", {}).get("diffusers-dynamic-offloader") == "0.1.0rc5"):
                    env["DDO_RESIDENT_MODULE_BUDGET_GB"] = "0"
                if case["routes"]["transformer"] == "vmm_runtime":
                    # NVMe loads the arena directly without the VMM Fast bootstrap.
                    env["DDO_VMM_LIBRARY_PATH"] = state["identity"]["dll"]
                exit_code = run(command, output, env)
                if exit_code:
                    raise subprocess.CalledProcessError(exit_code, command)
                if "feature_fixed_budget_bytes" in effective_case:
                    checked_case, _ = fixed_budget_measurement(state, case, record, command)
                    if checked_case["feature_fixed_budget_bytes"] != effective_case["feature_fixed_budget_bytes"]:
                        raise ValueError("The feature control budget changed during measurement.")
                if verify is None:
                    stages[name] = verify_stage(RECIPES[plan["recipe"]], case, name, output, profile_dir)
                else:
                    stages[name] = verify(effective_case, name, output, profile_dir)
                if reused_profile:
                    profile = read_json(metrics_file(output)).get("ddo_profile") or {}
                    if reused_profile.get("kind") == "linear":
                        valid_profile = (profile.get("name") == case["profile"]
                                         and profile.get("status") == "applied"
                                         and profile.get("resident_budget_gb") == reused_profile["resident_budget_gb"])
                    elif "feature_fixed_budget_bytes" in effective_case:
                        valid_profile = (profile.get("name") == case["profile"]
                                         and profile.get("status") == "explicit"
                                         and profile.get("shape_check") == "match"
                                         and effective_case["feature_fixed_budget_bytes"] == reused_profile["resident_budget_bytes"]
                                         and profile.get("resident_budget_gb") == reused_profile["resident_budget_bytes"] / 2**30)
                    else:
                        valid_profile = (profile.get("name") == case["profile"] and profile.get("shape_check") == "match"
                                         and (profile.get("recommendation") or {}).get("vmm_resident_weight_budget_bytes")
                                         == reused_profile["resident_budget_bytes"])
                    if not valid_profile or not receipts_valid(receipt):
                        stages.pop(name, None)
                        raise ValueError("The stage did not preserve the verified profile and budget.")
                    record["profile_receipt"] = receipt
                    record["profile_build_stage"] = profile.get("build_stage")
                if name.startswith("profile-"):
                    profile = read_json(metrics_file(output))["ddo_profile"]
                    saved = Path(profile["path"])
                    record["profile_receipt"] = {str(saved): digest(saved)}
                    record["profile_build_stage"] = profile.get("build_stage")
                write_state(path, state)
            except BaseException as error:
                # Preserve the active attempt for the process check on resume.
                record["error"] = str(error)
                write_state(path, state)
                print(f"{progress} | Failed. {total - completed} cases remain. "
                      f"Attempt preserved at {output}", flush=True)
                raise
        summary = measurement_summary(stages["measure"])
        if plan["family"] == "model-features":
            if __package__:
                from .model_features_suite import feature_summary
            else:
                from model_features_suite import feature_summary
            summary.update(feature_summary(state, case_id, stages["measure"]))
        if case.get("png_reference_case"):
            reference = cases[case["png_reference_case"]]
            if reference.get("status") != "completed" or not receipts_valid(reference.get("artifacts", {})):
                raise ValueError("The paired PNG control is incomplete or changed.")
            summary["output_difference"] = compare_pngs(reference["artifacts"], stages["measure"])
        component_reference = state["identity"].get("vae_component_reference")
        if component_reference:
            if not receipts_valid(component_reference["artifacts"]):
                raise ValueError("The native VAE control artifacts changed during execution.")
            summary["output_difference"] = compare_pngs(component_reference["artifacts"], stages["measure"])
        if attention_reference:
            reference_artifacts = attention_reference["artifacts"]
            if not receipts_valid(reference_artifacts):
                raise ValueError("The reference artifacts changed during execution.")
            reference_video = [sha for file, sha in reference_artifacts.items() if Path(file).suffix == ".mp4"]
            if reference_video:
                measured_video = [sha for file, sha in stages["measure"].items() if Path(file).suffix == ".mp4"]
                summary["mp4_sha256_matches_reference"] = measured_video == reference_video
            else:
                summary["output_difference"] = compare_pngs(reference_artifacts, stages["measure"])
        elif plan["family"] == "attention" and case_id != "official-leaf":
            reference = cases.get("official-leaf")
            if reference and reference["status"] == "completed":
                summary["output_difference"] = compare_pngs(reference["artifacts"], stages["measure"])
        record.update(status="completed", artifacts=stages["measure"], summary=summary)
        record.pop("error", None)
        if case["profile"] and not reused_profile:
            source = Path(next(iter(record["profile_receipt"])))
            try:
                exported = export_profile(source, case["profile"])
                record["exported_profile"] = {"name": case["profile"], "path": str(exported)}
                print(f"Reusable DDO profile: --ddo-profile {case['profile']} ({exported})", flush=True)
            except (OSError, ValueError) as error:
                record["profile_export_error"] = str(error)
                print(f"Profile kept in the session: {source}. Export unavailable: {error}", flush=True)
        write_state(path, state)
        completed = sum(item["status"] == "completed" for item in cases.values())
        print(f"Case {case_index}/{total}: {case_id} | Completed. "
              f"{completed}/{total} cases completed; {total - completed} remaining.", flush=True)
    print(f"Benchmark batch finished: {total}/{total} cases completed.", flush=True)


def main(run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows", "linux", "wsl"), default="windows")
    parser.add_argument("--recipe", required=True, choices=RECIPES)
    parser.add_argument("--family", choices=("runtime", "attention", "components", "reuse"), default="runtime")
    parser.add_argument("--case", choices=sorted(set(RUNTIME) | set(WSL_RUNTIME) | ATTENTION.keys() | {LOW_RAM_VAE_PURGE, "warm-process"}))
    parser.add_argument("--attention-runtime", choices=("vmm-fast",), help="Use the reviewed VMM Fast control.")
    parser.add_argument("--reference-manifest", type=Path, help="Completed SDPA runtime session manifest.")
    parser.add_argument("--component-reference-manifest", type=Path,
                        help="Completed native Linux Qwen VMM Fast session for the Leaf VAE comparison.")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--vram-gib", type=int, choices=(8,), default=8)
    parser.add_argument("--checkpoint", type=Path, help=CHECKPOINT_HELP)
    parser.add_argument("--custom-blocks", type=Path)
    parser.add_argument("--vae-model-path", type=Path, help="BF16 VAE local directory or Hub model ID for native Qwen VMM Fast.")
    parser.add_argument("--nvme-resident-gib", type=float, default=1.5)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--session-action", choices=("continue", "restart"))
    parser.add_argument("--expandable-segments", action="store_true",
                        help="Explicit PyTorch expandable segments for Linux Qwen linear and Leaf cases.")
    purge_policy = parser.add_mutually_exclusive_group()
    purge_policy.add_argument("--purge-before-transformer-only", action="store_true",
                              help="Use only the transformer purge for the selected image runtime comparison.")
    purge_policy.add_argument("--no-purge", action="store_true",
                              help="Run the selected image comparison without standby cache purges.")
    parser.add_argument("--expected-dll-sha256",
                        help="Optional Windows DLL SHA-256 to require; defaults to the installed wheel record.")
    parser.add_argument("--expected-library-sha256",
                        help="Optional Linux or WSL library SHA-256 to require; defaults to the installed wheel record.")
    parser.add_argument("--preflight", action="store_true", help="Check package and input identity without a runner.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.component_reference_manifest and (
            args.platform != "linux" or args.recipe != "qwen-image21"
            or args.family != "components" or args.case != "vmm-fast"
            or args.vae_model_path is not None):
        parser.error("--component-reference-manifest requires the Linux Qwen VMM Fast Leaf VAE component case.")
    if args.expandable_segments and (
            args.platform != "linux" or args.recipe != "qwen-image21"
            or args.family != "runtime"
            or args.case not in ("official-leaf", "one-shot-fast", "low-ram-safe")):
        parser.error("--expandable-segments requires a Linux Qwen Leaf or linear runtime case.")
    native_qwen = args.platform in ("windows", "linux") and args.recipe == "qwen-image21"
    qwen_runtime_case = args.family == "runtime" and args.case in (None, "vmm-fast")
    if (native_qwen and (qwen_runtime_case or args.attention_runtime)
            and args.vae_model_path is None):
        args.vae_model_path = Path("elismasilva/qwen-image-2.1-bf16-vae")
    if args.vae_model_path is not None and not (native_qwen and (qwen_runtime_case or args.attention_runtime)):
        parser.error("--vae-model-path requires native Linux or Windows Qwen VMM Fast runtime or attention.")
    if args.attention_runtime:
        if args.family != "attention":
            parser.error("--attention-runtime requires the attention family.")
        if native_qwen and args.vae_model_path is None:
            parser.error("Native Qwen VMM Fast attention requires the reference's BF16 --vae-model-path.")
        supported = ("cudnn-sdpa", "flash-attn2") if args.recipe == "ltx23-image" else ("cudnn-sdpa",)
        if args.case and args.case not in supported:
            parser.error(f"Select from {', '.join(supported)}.")
    elif args.reference_manifest:
        parser.error("--reference-manifest requires --attention-runtime vmm-fast.")
    if args.platform == "linux":
        linux_runtime = (args.family == "runtime"
                         and (args.case in LINUX_RUNTIME or (args.list_cases and args.case is None)))
        linux_attention = args.family == "attention" and args.attention_runtime == "vmm-fast"
        linux_ltx = args.recipe == "ltx23-image" and (linux_runtime or linux_attention)
        linux_component = args.family == "components" and args.component_reference_manifest
        linux_qwen = args.recipe == "qwen-image21" and (linux_runtime or linux_attention or linux_component)
        if not (linux_ltx or linux_qwen):
            parser.error("Select a reviewed Linux image runtime case or attention on VMM Fast.")
        if args.inventory:
            parser.error("The historical image inventory covers Windows only.")
        if args.purge_before_transformer_only or args.no_purge:
            parser.error("Purge options are Windows-only.")
    if args.platform == "wsl":
        if args.recipe == "ltx23-image":
            if args.family not in ("runtime", "attention"):
                parser.error("WSL LTX Image supports the runtime and attention families.")
            if args.family == "runtime" and args.case and args.case not in WSL_RUNTIME:
                parser.error("Select official-leaf, wsl-compat or vmm-fast for WSL runtime.")
        elif not args.attention_runtime:
            if args.family != "runtime" or (args.case and args.case not in WSL_RUNTIME):
                parser.error("Select official-leaf, wsl-compat or vmm-fast for WSL Qwen Image runtime.")
        if args.inventory:
            parser.error("The historical image inventory covers Windows only.")
        if args.purge_before_transformer_only or args.no_purge:
            parser.error("Purge options are Windows-only.")
    ltx_purge_control = args.recipe == "ltx23-image" and args.case in (
        "official-leaf", "one-shot-fast", "low-ram-safe"
    )
    qwen_purge_control = args.recipe == "qwen-image21" and (
        (args.purge_before_transformer_only and args.case == "official-leaf")
        or (args.no_purge and args.case in ("one-shot-fast", "low-ram-safe"))
    )
    qwen_vmm_control = (
        args.recipe == "qwen-image21" and args.family == "components"
        and args.case in ("nvme-baseline", "vmm-fast")
        and (args.no_purge or args.component_reference_manifest)
    )
    if (args.purge_before_transformer_only or args.no_purge) and (
        args.family != "runtime" or not (ltx_purge_control or qwen_purge_control)
    ) and not qwen_vmm_control:
        parser.error("A purge override requires a supported explicit image runtime case.")
    if args.family == "components" and not args.list_cases and (
        not args.case or (args.recipe == "qwen-image21" and not qwen_vmm_control)
    ):
        parser.error("Select one explicit component case. Qwen requires nvme-baseline or vmm-fast with --no-purge.")
    if args.inventory or args.list_cases:
        if args.execute or args.preflight:
            parser.error("Inventory and case listing are read-only.")
        if args.inventory:
            print(json.dumps(inventory(args.recipe), indent=2))
        else:
            print_cases((name, describe_case(case)) for name, case in build_plan(args)["cases"].items())
        return
    if not math.isfinite(args.nvme_resident_gib) or args.nvme_resident_gib <= 0:
        parser.error("--nvme-resident-gib must be positive and finite.")
    plan = build_plan(args)
    control_plan = None
    if args.attention_runtime and not args.reference_manifest:
        control_args = argparse.Namespace(**{**vars(args), "family": "runtime", "case": "vmm-fast",
                                            "attention_runtime": None, "reference_manifest": None})
        control_plan = build_plan(control_args)
    if args.preflight and args.execute:
        parser.error("Choose either preflight or execution.")
    if args.preflight or args.execute:
        if args.platform != "windows":
            is_wsl = (sys.platform == "linux"
                      and "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower())
            expected_hash = args.expected_library_sha256
            if sys.platform != "linux" or is_wsl != (args.platform == "wsl"):
                parser.error(f"{args.platform} requires a matching {args.platform} interpreter.")
        else:
            expected_hash = args.expected_dll_sha256
            if sys.platform != "win32":
                parser.error("Windows requires a Windows interpreter.")
    if args.preflight or args.execute:
        identity = execution_identity(plan, expected_hash.lower() if expected_hash else None)
        if args.component_reference_manifest:
            identity["vae_component_reference"] = verified_vae_leaf_reference(
                args.component_reference_manifest.resolve(), identity)
        if args.attention_runtime:
            if "flash-attn2" in plan["cases"] and not identity["packages"]["flash-attn"]:
                parser.error("Install the FlashAttention 2 wheel before running this selection.")
            reference = ensure_attention_reference(
                parser, args, identity, control_plan, verified_vmm_attention_reference, run)
            if reference is not None:
                identity["vmm_attention_reference"] = reference
    if args.preflight:
        print(json.dumps(identity, indent=2))
        return
    if not args.execute:
        print(json.dumps({"attention": plan, "prerequisite": control_plan,
                          "prerequisite_status": "resolve-compatible-control-before-execution"}
                         if control_plan else plan, indent=2))
        return
    needs_purge = any(
        flag.startswith("--purge-")
        for case in plan["cases"].values() for stage in case["stages"] for flag in stage["command"]
    )
    if needs_purge and not ctypes.windll.shell32.IsUserAnAdmin():
        parser.error("Run from an elevated Windows terminal to enable standby cache purging.")
    if any(case["hold"] for case in plan["cases"].values()):
        parser.error("This selection includes cases held for component validation. Select a reviewed case.")
    run_session(parser, args, identity, run,
                inventory_rows=(lambda: inventory(args.recipe)) if args.platform == "windows" else list,
                start_if_missing=bool(args.attention_runtime))


def ensure_attention_reference(parser, args, identity, control_plan, validate, run, *, verify=None):
    """Resolve one matched SDPA prerequisite; never launch a child during preflight."""
    if args.reference_manifest:
        return validate(args.reference_manifest.resolve(), identity)
    parent = args.output_root.resolve() / args.platform / args.recipe / "sessions"
    for path in sorted(parent.glob("*/manifest.json"), reverse=True):
        try:
            candidate = read_json(path)
            if (candidate.get("identity", {}).get("plan", {}).get("family") != "runtime"
                    or candidate.get("cases", {}).get("vmm-fast", {}).get("status") != "completed"):
                continue
            reference = validate(path, identity)
        except (ValueError, OSError, KeyError, TypeError) as error:
            print(f"SDPA candidate not reused: {path}: {error}", flush=True)
            continue
        print(f"Reusing verified SDPA control and profile: {path}", flush=True)
        return reference
    if control_plan is None:
        raise ValueError("The attention selection has no SDPA prerequisite plan.")
    print("Attention prerequisite: VMM Fast SDPA calibration and measurement are required.", flush=True)
    print(json.dumps({"prerequisite": control_plan, "then_attention": identity["plan"]}, indent=2), flush=True)
    if not args.execute:
        print("Preflight only: prerequisite pending; no benchmark was executed.", flush=True)
        return None
    if any(flag.startswith("--purge-") for case in control_plan["cases"].values()
           for stage in case["stages"] for flag in stage["command"]):
        if args.platform != "windows" or not ctypes.windll.shell32.IsUserAnAdmin():
            parser.error("Run from an elevated Windows terminal for the planned standby purge.")
    control_identity = {key: value for key, value in identity.items()
                        if key not in ("plan", "vmm_attention_reference", "sol_attn_dependency")}
    control_identity["plan"] = control_plan
    control_args = argparse.Namespace(**{**vars(args), "family": "runtime", "case": "vmm-fast",
                                        "attention_runtime": None, "reference_manifest": None})
    manifest = run_session(parser, control_args, control_identity, run,
                           verify=verify, start_if_missing=True)
    return validate(manifest, identity)


def run_session(parser, args, identity, run, *, inventory_rows=list, verify=None, start_if_missing=False):
    plan = identity["plan"]
    parent = args.output_root.resolve() / plan["platform"] / args.recipe / "sessions"
    parent.mkdir(parents=True, exist_ok=True)
    with session_lock(parent / ".lock"):
        paths = sorted(parent.glob("*/manifest.json"), reverse=True)
        states = [(path, read_json(path)) for path in paths]
        for _, previous in states:
            for record in previous["cases"].values():
                if record["status"] == "running":
                    check_abandoned_child(Path(record["active_output"]))
        match = next(((path, state) for path, state in states if state["identity"] == identity), None)
        action = args.session_action
        if match and action is None:
            records = match[1]["cases"].values()
            done = sum(record["status"] == "completed" for record in records)
            failed = sum("error" in record and record["status"] != "completed" for record in records)
            print(f"Completed: {done}. Failed: {failed}. Remaining: {len(records) - done}.")
            if not sys.stdin.isatty():
                parser.error("Choose --session-action continue or restart.")
            action = {"c": "continue", "r": "restart"}.get(input("[C]ontinuar / [R]odar tudo de novo: ").lower())
            if not action:
                parser.error("Choose C or R.")
        if action == "continue" and not match and not start_if_missing:
            parser.error("No compatible session to continue.")
        if match and action == "continue":
            path, state = match
        else:
            folder = parent / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex}"
            folder.mkdir()
            path = folder / "manifest.json"
            state = {"identity": identity, "legacy_inventory": inventory_rows(),
                     "cases": {case: {"status": "pending"} for case in plan["cases"]}}
            if not match and args.family == "attention" and action != "restart" and not identity.get("vmm_attention_reference"):
                reusable_measurements(state, [previous for _, previous in states])
            write_state(path, state)
        execute_session(path, state, run, verify=verify)
        print(f"[{len(state['cases'])}/{len(state['cases'])}] Complete. Manifest: {path}", flush=True)
        return path
