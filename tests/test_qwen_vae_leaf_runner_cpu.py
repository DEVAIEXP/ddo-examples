"""Runner component routing contracts: no CUDA, downloads or bootstrap execution."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["run_dynamic_modular.py", "run_dynamic_modular_vmm.py"])
def test_vae_policy_is_owned_by_ddo_and_attached_after_local_modules(name):
    source = (ROOT / "scripts/qwen_image_21" / name).read_text()
    tree = ast.parse(source)
    assert "--vae-leaf-offload" in source
    assert "--vae-resident" in source
    assert "vae.enable_group_offload(" not in source
    assert "vae.to(device)" not in source
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    vae_attach = [
        n
        for n in calls
        if isinstance(n.func, ast.Name)
        and n.func.id == "enable_offload"
        and any(
            k.arg == "component" and isinstance(k.value, ast.Constant) and k.value.value == "vae" for k in n.keywords
        )
    ]
    assert vae_attach
    local = [
        n
        for n in calls
        if isinstance(n.func, ast.Attribute)
        and n.func.attr in ("enable_single_frame_decode", "enable_residual_aliasing", "enable_direct_padding")
    ]
    assert all(n.lineno < min(call.lineno for call in vae_attach) for n in local)
    assert "vae_result.metrics_snapshot(" in source
    assert "vae_result.cleanup_on_error()" in source
