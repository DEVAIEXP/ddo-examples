"""Historical video entry point; retained for existing reproduction commands."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ltx2_video.run_dynamic_modular_diagnostics_distilled import *  # Legacy import compatibility.

if __name__ == "__main__":
    main()
