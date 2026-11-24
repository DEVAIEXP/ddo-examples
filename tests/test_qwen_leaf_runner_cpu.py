"""The Qwen Leaf control runs the staged custom-block workflow with official Diffusers offload."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAF = ROOT / "scripts/qwen_image_21/run_diffusers_leaf.py"
STAGED_PHASES = [
    "metadata", "load_text_encoder", "attach_text_encoder", "encode", "cleanup_text_encoder",
    "purge_before_transformer", "load_transformer", "attach_transformer", "prepare", "denoise",
    "cleanup_transformer", "load_vae", "attach_vae", "decode", "cleanup_vae", "save_image",
]


def phases(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.args[0].value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "phase"
            and node.args and isinstance(node.args[0], ast.Constant)], tree


def test_leaf_control_measures_every_component_phase_in_order():
    names, _ = phases(LEAF)
    assert names == STAGED_PHASES


def test_leaf_control_shares_the_ddo_runners_text_encoder_phases():
    leaf, _ = phases(LEAF)
    for runner in ("run_dynamic_modular.py", "run_dynamic_modular_vmm.py"):
        other, _ = phases(ROOT / "scripts/qwen_image_21" / runner)
        shared = ["load_text_encoder", "attach_text_encoder", "encode", "cleanup_text_encoder"]
        assert [name for name in other if name in shared] == [name for name in leaf if name in shared]


def test_leaf_control_uses_custom_blocks_and_official_offload_without_ddo_runtime():
    source = LEAF.read_text(encoding="utf-8")
    _, tree = phases(LEAF)
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "load_custom_blocks" in calls and "apply_group_offloading" in calls
    assert not calls & {"enable_offload", "remove_dynamic_offload", "from_pretrained_with_dynamic_offload"}
    # Only the pipeline config is read; the full conventional pipeline is never built.
    assert "QwenImage21Pipeline.from_pretrained" not in source
    assert "QwenImage21Pipeline.load_config" in source
    for step in ("QwenImage21TextEncoderStep", "QwenImage21PrepareStep", "QwenImage21DenoiseStep",
                 "QwenImage21DecodeStep"):
        assert f"{step}().init_pipeline()" in source


def test_leaf_control_releases_intermediates_like_the_ddo_runners():
    source = LEAF.read_text(encoding="utf-8")
    assert "value.cpu() if isinstance(value, torch.Tensor)" in source
    assert "del text_encoder, processor, prompt_pipe" in source
    assert "del encoded, prepare" in source
    assert "del transformer, prepared, denoise, scheduler, timer, callback" in source
    assert "del decode, vae" in source
    assert source.count("flush()") == 3
