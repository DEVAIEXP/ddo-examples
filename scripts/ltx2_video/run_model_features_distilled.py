"""Explicit cumulative compute experiments using the unchanged VMM sampling phases.

The ordinary VMM entry keeps experimental fusion flags hidden. This benchmark
entry exposes only the selected RMS/rotary callbacks, without enabling tracing,
compilation, alternate allocators or changing any sampling defaults.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.helpers.pytorch_allocator import configure_pytorch_allocator

configure_pytorch_allocator()


def parse_args(argv=None):
    from scripts.ltx2_video.run_dynamic_modular_vmm_distilled import parse_args as parse_vmm
    return parse_vmm(argv, entrypoint="model-features")


def main():
    from ddo_vmm_bootstrap import ensure_vmm_fast_process
    if not any(flag in sys.argv[1:] for flag in ("--help", "-h", "--dry-run")):
        ensure_vmm_fast_process()
    from scripts.ltx2_video.run_dynamic_modular_vmm_distilled import run, describe_plan
    args = parse_args()
    if args.dry_run:
        import json
        print(json.dumps(describe_plan(args), indent=2))
        return
    os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".hf_modules_modular_video"))
    ffmpeg = os.getenv("DDO_RUNNER_FFMPEG_BIN", "")
    handle = os.add_dll_directory(ffmpeg) if os.name == "nt" and os.path.isdir(ffmpeg) else None
    try:
        run(args)
    finally:
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    main()
