"""Save a standalone dense component in BF16, without CUDA or model forwards.

The source directory and any existing destination are never overwritten.
This is checkpoint preparation, not DDO runtime residency or quantization logic.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_component(source, output):
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError("Destination must be separate from the source component")
    config = source / "config.json"
    weights = source / "diffusion_pytorch_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise ValueError("Require config.json and diffusion_pytorch_model.safetensors")
    if len(list(source.glob("*.safetensors"))) != 1 or list(source.glob("*.index.json")):
        raise ValueError("Only a standalone unsharded dense component is supported")
    # Fail early on malformed config; its contents are preserved, not rewritten.
    json.loads(config.read_text(encoding="utf-8"))
    before_hash = sha256(weights)
    tensors = {}
    source_dtypes = {}
    with safe_open(weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():  # noqa: SIM118 -- safe_open is not a dictionary iterable.
            value = handle.get_tensor(key)
            if value.is_floating_point() and value.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(f"Unsupported source dtype for {key}: {value.dtype}")
            source_dtypes[str(value.dtype)] = source_dtypes.get(str(value.dtype), 0) + 1
            tensors[key] = value.to(torch.bfloat16) if value.is_floating_point() else value
    if not tensors:
        raise ValueError("Empty component checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ddo-bf16-", dir=output.parent) as temporary:
        staging = Path(temporary) / "component"
        staging.mkdir()
        target = staging / weights.name
        save_file(tensors, target, metadata=metadata)
        # Check saved values against the source cast, not just against the dict
        # submitted to save_file. No generator, inference or CUDA is involved.
        with (
            safe_open(weights, framework="pt", device="cpu") as original,
            safe_open(target, framework="pt", device="cpu") as saved,
        ):
            if set(original.keys()) != set(saved.keys()):
                raise RuntimeError("Saved parameter keys differ")
            for key in original.keys():  # noqa: SIM118 -- safe_open requires its keys() API.
                value = original.get_tensor(key)
                expected = value.to(torch.bfloat16) if value.is_floating_point() else value
                actual = saved.get_tensor(key)
                if expected.dtype != actual.dtype or not torch.equal(expected, actual):
                    raise RuntimeError(f"Saved source-cast equality failed: {key}")
        if sha256(weights) != before_hash:
            raise RuntimeError("Source changed during conversion")
        shutil.copy2(config, staging / config.name)
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE"):
            if (source / name).is_file():
                shutil.copy2(source / name, staging / name)
        report = {
            "schema_version": 1,
            "source_directory": str(source),
            "source_sha256": before_hash,
            "output_sha256": sha256(target),
            "source_dtypes": source_dtypes,
            "tensor_count": len(tensors),
            "output_dtype": "bfloat16_for_floating_tensors",
            "torch_version": torch.__version__,
            "source_cast_equality": "exact",
            "source_file_bytes": weights.stat().st_size,
            "output_file_bytes": target.stat().st_size,
            "cuda_initialized": torch.cuda.is_initialized(),
        }
        if report["cuda_initialized"]:
            raise RuntimeError("Checkpoint preparation must not initialize CUDA")
        (staging / "conversion.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Destination appeared during conversion: {output}")
        staging.rename(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Standalone dense component directory.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; no overwrite.")
    args = parser.parse_args()
    print(json.dumps(convert_component(args.source, args.output), indent=2))
