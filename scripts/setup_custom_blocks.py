"""Clone reviewed custom blocks without modifying existing repositories."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = (
    ("ltx2_image", "https://huggingface.co/elismasilva/ltx2.3_image_custom_blocks",
     "a6e6f865dc6f8decd7ff681e91969bf4b07e9f21"),
    ("ltx2_video", "https://huggingface.co/elismasilva/ltx2.3_video_custom_blocks",
     "f5c712e93c52283cb101da26d4a4544c808666bf"),
)

def prepare():
    base = ROOT / "modular_diffusers" / "custom_blocks"
    base.mkdir(parents=True, exist_ok=True)
    for name, url, revision in BLOCKS:
        target = base / name
        if target.exists():
            origin = subprocess.check_output(
                ["git", "-C", str(target), "remote", "get-url", "origin"], text=True).strip()
            head = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
            if origin.rstrip("/") != url.rstrip("/") or head != revision:
                raise RuntimeError(f"Existing {target} differs from reviewed source; inspect manually.")
        else:
            subprocess.run(["git", "clone", url, str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
        print(f"{name}: {revision}")

if __name__ == "__main__":
    prepare()
