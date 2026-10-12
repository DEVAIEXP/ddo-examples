"""Qwen SDNQ orchestration contracts; no weights, downloads or CUDA execution."""

import ast
from pathlib import Path

import pytest

from scripts.qwen_image_21 import run_dynamic_modular_sdnq as runner


def test_default_checkpoint_is_pinned_int4():
    args = runner.parse_args([])
    assert runner.transformer_source(args) == runner.SDNQ_MODEL
    assert runner.transformer_revision(args) == runner.SDNQ_REVISION
    assert len(runner.SDNQ_REVISION) == 40
    assert (args.width, args.height, args.steps, args.seed) == (512, 512, 20, 42)
    assert args.auxiliary_model == "Qwen/Qwen-Image-2.1"


def test_override_does_not_inherit_another_repository_revision():
    args = runner.parse_args(["--model-path", "another/checkpoint"])
    assert runner.transformer_revision(args) is None
    args.model_revision = "fixed-revision"
    assert runner.transformer_revision(args) == "fixed-revision"


@pytest.mark.parametrize("preset", ["auto", "one_shot_fast", "low_ram_safe", "wsl_compat", "warm_process"])
def test_packed_runtime_not_dense_or_native(preset, monkeypatch):
    monkeypatch.setenv("DDO_BACKEND", "vmm")
    monkeypatch.setenv("DDO_PRESET", "vmm_fast")
    settings = runner.sdnq_settings(runner.parse_args(["--ddo-preset", preset]), "cpu")
    assert settings.execution_mode == "sdnq_runtime"
    assert settings.enabled and not settings.plan


def test_leaf_control_is_plan_mode():
    args = runner.parse_args(["--ddo-preset", "diffusers_leaf_offload_compat"])
    settings = runner.sdnq_settings(args, "cpu")
    assert settings.execution_mode == "plan"
    assert settings.enabled and settings.plan


@pytest.mark.parametrize("argv", [["--ddo-preset", "vmm_fast"], ["--variant", "turbo"]])
def test_rejects_unvalidated_routes(argv):
    with pytest.raises(SystemExit):
        runner.parse_args(argv)


def test_visible_lifecycle_preserves_backend_and_shared_reporting():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "RunnerReporting(args, metrics, device)" in source
    assert "run.purge_before_transformer()" in source
    assert 'backend.get("sdnq_module_count")' in source
    assert 'recipe.get("quant_method") != "sdnq"' in source
    assert "ensure_vmm_fast_process" not in source
    assert "torch.compile" not in source
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            assert node.attr != "forward"
    for component in ("text_encoder", "vae"):
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "enable_offload"
            and any(
                keyword.arg == "component"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == component
                for keyword in node.keywords
            )
        ]
        assert len(calls) == 1
        assert any(
            keyword.arg == "preset"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "one_shot_fast"
            for keyword in calls[0].keywords
        )
