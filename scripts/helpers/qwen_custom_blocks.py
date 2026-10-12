"""Trusted custom-code loading only; runners retain visible phase orchestration."""

from importlib import import_module
from pathlib import Path

DEFAULT_REPOSITORY = "elismasilva/qwen_image_2.1_custom_blocks"


def add_custom_blocks_arguments(parser):
    parser.add_argument(
        "--custom-blocks",
        default=DEFAULT_REPOSITORY,
        help="Trusted Hub code repository or local custom-block checkout.",
    )
    parser.add_argument("--custom-blocks-revision", help="Hub code revision (prefer a published commit SHA).")


def load_custom_blocks(args):
    """Resolve code once, then obtain every stage from the same trusted namespace."""
    from diffusers import ModularPipelineBlocks
    from huggingface_hub import snapshot_download

    source = Path(args.custom_blocks)
    local = source.is_dir()
    if not local:
        # Snapshot paths avoid dotted repository IDs becoming Python package names.
        source = Path(
            snapshot_download(
                repo_id=args.custom_blocks,
                revision=args.custom_blocks_revision,
                allow_patterns=["*.py", "modular_config.json"],
                local_files_only=args.local_files_only,
            )
        )
    elif args.custom_blocks_revision:
        raise ValueError("--custom-blocks-revision applies only to Hub repositories")
    if not (source / "modular_config.json").is_file():
        raise FileNotFoundError(
            f"Custom blocks are missing modular_config.json in {source}. "
            "Publish the code first or use --custom-blocks with the local checkout."
        )
    blocks = ModularPipelineBlocks.from_pretrained(
        str(source),
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    module = import_module(type(blocks).__module__)
    required = (
        "QwenImage21TextEncoderStep",
        "QwenImage21PrepareStep",
        "QwenImage21TurboPrepareStep",
        "QwenImage21DenoiseStep",
        "QwenImage21DecodeStep",
        "QwenImage21ImageVAE",
    )
    for name in required:
        if not hasattr(module, name):
            raise ImportError(f"Trusted Qwen custom blocks do not export {name}")
    metadata = {
        "repository": args.custom_blocks,
        "requested_revision": args.custom_blocks_revision,
        "resolved_revision": source.name if not local and source.parent.name == "snapshots" else None,
        "resolved_path": str(source),
        "source": "local" if local else "hub_snapshot",
        "trust_remote_code": True,
    }
    return module, metadata
