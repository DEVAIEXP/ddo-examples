"""Explicit runner opt-in and staged device contracts, without CUDA."""

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/qwen_image_21/run_dynamic_modular.py"


def runner():
    spec = importlib.util.spec_from_file_location("qwen_linear_contract", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_preserved_and_opt_in_available():
    module = runner()
    assert not module.parse_args([]).vae_linear_offload
    assert module.parse_args(["--vae-linear-offload"]).vae_linear_offload
    with pytest.raises(SystemExit):
        module.parse_args(["--vae-linear-offload", "--vae-leaf-offload"])
    with pytest.raises(SystemExit):
        module.parse_args(["--vae-linear-offload", "--ddo-preset", "diffusers_leaf_offload_compat"])


def test_candidate_uses_public_lifecycle_and_explicit_device():
    source = RUNNER.read_text()
    tree = ast.parse(source)
    route = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "vae_route" for target in n.targets)
    )
    assert isinstance(route, ast.IfExp)
    assert route.test.attr == "vae_linear_offload"
    assert route.body.value == "linear_convolution"
    attach = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "enable_offload"
        and any(
            k.arg == "component" and isinstance(k.value, ast.Constant) and k.value.value == "vae" for k in n.keywords
        )
    )
    assert next(k.value for k in attach.keywords if k.arg == "preset").attr == "ddo_preset"
    assert "DynamicOffloadConfig(" not in source
    assert "vae_result.cleanup_on_error()" in source
    assert "vae_result.metrics_snapshot()" in source
    assert "remove_dynamic_offload(vae)" in source
    call = next(
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "decode"
    )
    assert "execution_device" in {keyword.arg for keyword in call.keywords}


def test_local_decode_preserves_default_device_inference():
    source = (ROOT / "modular_diffusers/custom_blocks/qwen_image_21/decoders.py").read_text()
    assert 'InputParam("execution_device", default=None)' in source
    assert "block.execution_device or components._execution_device" in source
