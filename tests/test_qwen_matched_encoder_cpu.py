"""Explicit encoder controls preserve defaults; no model or CUDA execution."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def encoder_options():
    source = Path(__file__).parents[1] / "scripts/qwen_image_21/run_dynamic_modular.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "text_encoder_offload_options"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name], tree


@pytest.mark.parametrize("preset", ["auto", "one_shot_fast", "diffusers_leaf_offload_compat", "wsl_compat"])
def test_default_encoder_policy_is_unchanged(preset):
    options, _ = encoder_options()
    assert options(SimpleNamespace(ddo_preset=preset, text_encoder_preset=None)) == {"preset": preset}


@pytest.mark.parametrize("preset", ["one_shot_fast", "diffusers_leaf_offload_compat"])
def test_explicit_encoder_control_is_identical_for_transformer_routes(preset):
    options, _ = encoder_options()
    args = SimpleNamespace(ddo_preset=preset, text_encoder_preset="diffusers_leaf_offload_compat")
    assert options(args) == {
        "preset": "diffusers_leaf_offload_compat",
        "group_offload": True,
        "dynamic_offload": False,
        "offload_type": "leaf_level",
    }


def test_encoder_control_is_used_only_at_encoder_attachment():
    _, tree = encoder_options()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    encoder_call = next(
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "enable_offload"
        and any(keyword.arg == "component" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "text_encoder" for keyword in node.keywords)
    )
    assert any(
        keyword.arg is None and isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == "text_encoder_offload_options"
        for keyword in encoder_call.keywords
    )
    transformer_call = next(
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "enable_offload"
        and any(keyword.arg == "component" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "transformer" for keyword in node.keywords)
    )
    assert any(keyword.arg == "preset" and isinstance(keyword.value, ast.Attribute)
               and keyword.value.attr == "ddo_preset" for keyword in transformer_call.keywords)
