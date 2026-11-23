"""Allocator environment defaults and entry-point ordering without Torch imports."""

import ast
from pathlib import Path

import pytest

from scripts.helpers.pytorch_allocator import configure_pytorch_allocator


@pytest.mark.parametrize("initial", [
    {},
    {"PYTORCH_ALLOC_CONF": "expandable_segments:False"},
    {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
    {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128"},
    {"PYTORCH_ALLOC_CONF": ""},
    {"PYTORCH_CUDA_ALLOC_CONF": ""},
    {"PYTORCH_ALLOC_CONF": "backend:native", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"},
])
def test_default_is_idempotent_and_explicit_configuration_wins(initial):
    values = dict(initial, UNRELATED="preserved")
    expected = dict(values)
    if not initial:
        expected["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    report = configure_pytorch_allocator(values)
    assert values == expected
    assert report == {name: values.get(name) for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}
    assert configure_pytorch_allocator(values) == report
    assert values == expected


RUNNERS = sorted((Path(__file__).parents[1] / "scripts").glob("*/run_*.py"))


@pytest.mark.parametrize("path", RUNNERS, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_all_public_runners_configure_before_bootstrap_and_framework_imports(path):
    tree = ast.parse(path.read_text())
    calls = [node for node in tree.body if
             (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name) and node.value.func.id == "configure_pytorch_allocator")
             or (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name) and node.value.func.id == "configure_pytorch_allocator")]
    assert len(calls) == 1
    configured = calls[0].lineno
    owning = {"torch", "diffusers", "transformers", "sdnq", "diffusers_dynamic_offloader", "ddo_vmm_bootstrap"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in owning for alias in node.names):
                assert node.lineno > configured
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in owning:
                assert node.lineno > configured
            if node.module and node.module.startswith("scripts.") and node.module != "scripts.helpers.pytorch_allocator":
                assert node.lineno > configured
    assert 'setdefault("PYTORCH_CUDA_ALLOC_CONF"' not in path.read_text()


def test_helper_has_no_framework_import_or_eager_configuration():
    path = Path(__file__).parents[1] / "scripts/helpers/pytorch_allocator.py"
    tree = ast.parse(path.read_text())
    imports = [node for node in tree.body if isinstance(node, ast.Import)]
    assert [alias.name for node in imports for alias in node.names] == ["os"]
    assert not any(isinstance(node, (ast.Expr, ast.Assign)) for node in tree.body[2:])
