"""CPU checkpoint preparation and native VAE runner option contracts."""

import argparse
import ast
import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.helpers.convert_safetensors_component_bf16 import convert_component

ROOT = Path(__file__).resolve().parents[1]


def source_checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"channels": 2}\n')
    save_file({"weight": torch.randn(2, 3), "index": torch.arange(3)}, source / "diffusion_pytorch_model.safetensors")
    return source


def test_dense_conversion_preserves_source_config_and_cast_values(tmp_path):
    source = source_checkpoint(tmp_path)
    before = (source / "diffusion_pytorch_model.safetensors").read_bytes()
    output = tmp_path / "bf16"
    report = convert_component(source, output)
    saved = load_file(output / "diffusion_pytorch_model.safetensors")
    original = load_file(source / "diffusion_pytorch_model.safetensors")
    assert torch.equal(saved["weight"], original["weight"].to(torch.bfloat16))
    assert torch.equal(saved["index"], original["index"])
    assert (source / "diffusion_pytorch_model.safetensors").read_bytes() == before
    assert (output / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert json.loads((output / "conversion.json").read_text()) == report
    assert report["source_cast_equality"] == "exact" and not report["cuda_initialized"]


def test_conversion_refuses_overwrite_and_source_descendant(tmp_path):
    source = source_checkpoint(tmp_path)
    with pytest.raises(FileExistsError):
        convert_component(source, source)
    with pytest.raises(ValueError, match="separate"):
        convert_component(source, source / "converted")
    output = tmp_path / "bf16"
    output.mkdir()
    with pytest.raises(FileExistsError):
        convert_component(source, output)
    assert list(output.iterdir()) == []


def test_native_vae_runner_keeps_explicit_capacity_and_public_lifecycle():
    source = (ROOT / "scripts/qwen_image_21/run_dynamic_modular_vmm.py").read_text()
    tree = ast.parse(source)
    assert "--vae-vmm-offload" in source
    assert "args.vae_vmm_resident_weight_budget_gb" in source
    assert "vae_config = replace(" in source
    assert "vmm_component_tensor_ownership=True" in source
    assert 'vae_loaded.inspect_vmm_component(component="vae")' in source
    assert "vae_result.metrics_snapshot(collect_completed=True)" in source
    assert "vae_result.cleanup_on_error()" in source
    assert "remove_dynamic_offload(vae)" in source
    assert "mutually exclusive" in source
    assert "config.vmm_resident_weight_budget_gb =" not in source
    native_flag = next(
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and any(isinstance(arg, ast.Constant) and arg.value == "--vae-vmm-offload" for arg in call.args)
    )
    assert any(k.arg == "action" and k.value.value == "store_true" for k in native_flag.keywords)


def parse_runner_args(argv):
    tree = ast.parse((ROOT / "scripts/qwen_image_21/run_dynamic_modular_vmm.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = {"argparse": argparse, "math": math, "Path": Path, "ROOT": ROOT}
    # Extract argument parsing only: do not import/execute the runner bootstrap.
    exec(compile(ast.Module(body=[function], type_ignores=[]), "runner_args", "exec"), namespace)  # noqa: S102 -- Trusted local parser AST only.
    return namespace["parse_args"](argv)


def test_vae_options_keep_defaults_and_transformer_budget(tmp_path):
    defaults = parse_runner_args([])
    assert not defaults.vae_vmm_offload and not defaults.vae_leaf_offload
    assert defaults.vae_vmm_resident_weight_budget_gb is None
    native = parse_runner_args(
        ["--vae-vmm-offload", "--vae-model-path", str(tmp_path), "--vae-vmm-resident-weight-budget-gb", "0.375"]
    )
    assert native.vae_vmm_resident_weight_budget_gb == 0.375
    assert native.ddo_vmm_resident_weight_budget_gb == defaults.ddo_vmm_resident_weight_budget_gb
    with pytest.raises(SystemExit):
        parse_runner_args(["--vae-vmm-offload", "--vae-leaf-offload"])
    with pytest.raises(SystemExit):
        parse_runner_args(["--vae-vmm-offload", "--ddo-preset", "nvme_baseline"])


@pytest.mark.parametrize("budget", ["nan", "inf", "0", "-1"])
def test_vae_options_refuse_invalid_native_budget(tmp_path, budget):
    with pytest.raises(SystemExit):
        parse_runner_args(
            ["--vae-vmm-offload", "--vae-model-path", str(tmp_path), "--vae-vmm-resident-weight-budget-gb", budget]
        )
