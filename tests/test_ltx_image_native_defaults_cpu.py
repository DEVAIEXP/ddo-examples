"""Resolve runner defaults without importing Torch or starting CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("backend,preset,diagnostic,explicit,expected", [
    ("vmm", "auto", False, None, True),
    ("vmm", "vmm_fast", False, None, True),
    ("vmm", "nvme_baseline", False, None, False),
    ("linear", "auto", False, None, False),
    ("vmm", "auto", True, None, False),
    ("vmm", "auto", False, False, False),
    ("vmm", "auto", False, True, True),
])
def test_public_native_default_preserves_other_routes(backend, preset, diagnostic, explicit, expected):
    path = Path(__file__).parents[1] / "scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    default = next(n for n in main.body if isinstance(n, ast.If))
    args = SimpleNamespace(ddo_backend=backend, ddo_preset=preset,
                           ddo_vmm_diagnostic_adapter=diagnostic, ddo_public_native_runtime=explicit)
    exec(compile(ast.Module(body=[default], type_ignores=[]), str(path), "exec"), {"args": args})
    assert args.ddo_public_native_runtime is expected
