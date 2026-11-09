"""Run the staged LTX 2.3 image pipeline with official Diffusers leaf offload."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ltx2_image.run_dynamic_modular_distilled import main


if __name__ == "__main__":
    main(official_leaf=True)
