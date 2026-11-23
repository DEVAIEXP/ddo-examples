"""Resolve explicit local folders or Hub model IDs before measured execution."""

from pathlib import Path
import re

QWEN_BF16_VAE = "elismasilva/qwen-image-2.1-bf16-vae"
LTX_VIDEO_FP8 = "elismasilva/ltx2.3_video_fp8_distilled_transformer"


def source_argument(source):
    """Keep Hub IDs portable and local directories absolute in command previews."""
    path = Path(source)
    if path.is_dir():
        return str(path.resolve())
    name = path.as_posix()
    if re.fullmatch(r"[\w.-]+/[\w.-]+", name) and not name.startswith("."):
        return name
    return str(path.resolve())


def resolve_model_source(source, *, local_files_only=False):
    """Use the standard Hub cache; offline mode only accepts an existing snapshot."""
    value = source_argument(source)
    if Path(value).is_dir():
        return Path(value)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value):
        raise ValueError(f"Model directory does not exist: {source}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=value, local_files_only=local_files_only))


def resolve_plan_sources(plan):
    """Freeze auxiliary Hub sources to local snapshots before identity checks."""
    resolved = {}
    for case in plan["cases"].values():
        for stage in case["stages"]:
            command = stage["command"]
            flags = ["--vae-model-path"]
            if "--transformer-fp8-weight-only" in command:
                flags.append("--model-path")
            for flag in flags:
                if flag not in command:
                    continue
                index = command.index(flag) + 1
                value = command[index]
                if value not in resolved:
                    resolved[value] = str(resolve_model_source(value))
                command[index] = resolved[value]
        if case.get("fp8_model_path") in resolved:
            case["fp8_model_path"] = resolved[case["fp8_model_path"]]
    hub_sources = {key: value for key, value in resolved.items() if key != value}
    if hub_sources:
        plan["hub_model_sources"] = hub_sources


def resolve_vae_source(source=None, *, model_source, local_files_only=False):
    """Use an explicit standalone VAE or the selected pipeline's VAE subfolder."""
    root = resolve_model_source(source or model_source, local_files_only=local_files_only)
    vae = root / "vae" if (root / "vae").is_dir() else root
    if not (vae / "config.json").is_file():
        raise ValueError(f"VAE source has no config.json: {vae}")
    return vae
