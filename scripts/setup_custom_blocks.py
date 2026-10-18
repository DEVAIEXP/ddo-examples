"""Clone custom blocks from main or an explicit revision; preserve existing clones."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = (
    ("ltx2_image", "https://huggingface.co/elismasilva/ltx2.3_image_custom_blocks"),
    ("ltx2_video", "https://huggingface.co/elismasilva/ltx2.3_video_custom_blocks"),
    ("qwen_image_21", "https://huggingface.co/elismasilva/qwen_image_2.1_custom_blocks"),
)

def prepare(revision=None, models=None):
    base = ROOT / "modular_diffusers" / "custom_blocks"
    base.mkdir(parents=True, exist_ok=True)
    for name, url in BLOCKS:
        if models and name not in models:
            continue
        target = base / name
        if target.exists():
            origin = subprocess.check_output(
                ["git", "-C", str(target), "remote", "get-url", "origin"], text=True).strip()
            head = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
            if origin.rstrip("/").removesuffix(".git") != url.rstrip("/").removesuffix(".git"):
                raise RuntimeError(f"Existing {target} has a different origin; inspect manually.")
            if revision:
                expected = subprocess.check_output(
                    ["git", "-C", str(target), "rev-parse", revision], text=True).strip()
                if head != expected:
                    raise RuntimeError(f"Existing {target} differs from requested revision; inspect manually.")
            print(f"{name}: preserving existing checkout at {head}; no fetch, pull or reset")
            continue
        else:
            subprocess.run(["git", "clone", "--branch", "main", url, str(target)], check=True)
            if revision:
                subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
        head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        print(f"{name}: {head}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=[name for name, _ in BLOCKS],
                        help="Prepare only this model (repeatable); default: all three.")
    parser.add_argument("--revision", help="Optional code revision; use --model to pin one repository.")
    args = parser.parse_args()
    prepare(revision=args.revision, models=args.model)
