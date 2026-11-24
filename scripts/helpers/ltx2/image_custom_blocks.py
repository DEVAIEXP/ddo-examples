"""Resolve one trusted LTX image code source for every model-specific stage."""

from importlib import import_module
from pathlib import Path

DEFAULT_REPOSITORY = "elismasilva/ltx2.3_image_custom_blocks"
# Optional development checkout prepared by scripts/setup_custom_blocks.py.
LOCAL_CHECKOUT = Path(__file__).resolve().parents[3] / "modular_diffusers/custom_blocks/ltx2_image"
REQUIRED_EXPORTS = (
    "LTX2ImageDistilledBlocks", "LTX2ImageTextEncoderStep", "LTX2ImageConnectorStep",
    "LTX2ImageDenoiseStep", "LTX2ImagePrepareLatentsStep",
    "LTX2ImageTransformer2DModel", "LTX2ImageTextConnectors",
)


def add_custom_blocks_arguments(parser):
    parser.add_argument("--custom-blocks", default=DEFAULT_REPOSITORY,
                        help="Trusted Hub code repository or local LTX image checkout.")
    parser.add_argument("--custom-blocks-revision", help="Hub code revision; rejected for a local checkout.")


def load_custom_blocks(args):
    """Load components and phases from the same resolved executable namespace."""
    from diffusers import ModularPipelineBlocks
    from huggingface_hub import snapshot_download

    source = Path(args.custom_blocks)
    local = source.is_dir()
    if not local:
        if source.is_absolute():
            raise FileNotFoundError(f"Custom-block directory does not exist: {source}")
        source = Path(snapshot_download(
            repo_id=args.custom_blocks, revision=args.custom_blocks_revision,
            allow_patterns=["*.py", "modular_config.json"], local_files_only=args.local_files_only,
        ))
    elif args.custom_blocks_revision:
        raise ValueError("--custom-blocks-revision applies only to Hub repositories")
    if not (source / "modular_config.json").is_file():
        raise FileNotFoundError(f"Custom blocks are missing modular_config.json in {source}")
    blocks = ModularPipelineBlocks.from_pretrained(
        str(source), trust_remote_code=True, local_files_only=args.local_files_only,
    )
    module = import_module(type(blocks).__module__)
    for name in REQUIRED_EXPORTS:
        if not hasattr(module, name):
            raise ImportError(f"Trusted LTX image custom blocks do not export {name}")
    metadata = {
        "repository": args.custom_blocks, "requested_revision": args.custom_blocks_revision,
        "resolved_revision": source.name if not local and source.parent.name == "snapshots" else None,
        "resolved_path": str(source.resolve()), "source": "local" if local else "hub_snapshot",
        "trust_remote_code": True,
    }
    return module, metadata
