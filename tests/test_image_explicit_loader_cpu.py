"""Image checkpoint loading must not require fork-only AutoModel exports."""

import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
# Components each runner loads through DDO with an explicit, fork-free class.
EXPLICIT_LOADERS = {"Gemma3ForConditionalGeneration", "LTX2ImageTransformer2DModel", "AutoencoderKLLTX2Video"}


@pytest.mark.parametrize(
    "runner",
    [
        "run_dynamic_minimal.py",
        "run_dynamic_modular_distilled.py",
        "run_dynamic_modular_vmm_distilled.py",
    ],
)
def test_image_runners_select_custom_transformer(runner):
    tree = ast.parse((ROOT / "scripts/ltx2_image" / runner).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "from_pretrained_with_dynamic_offload"
    ]
    loaders = []
    for node in calls:
        selected = [kw.value for kw in node.keywords if kw.arg == "model_loader"]
        assert len(selected) == 1 and isinstance(selected[0], ast.Name)
        loaders.append(selected[0].id)
    assert set(loaders) <= EXPLICIT_LOADERS
    assert "LTX2ImageTransformer2DModel" in loaders


def test_saved_custom_checkpoint_loads_without_auto_model(tmp_path, monkeypatch):
    import diffusers

    # Scoped to this test so collection does not expose the clones as plain packages.
    monkeypatch.syspath_prepend(str(ROOT / "modular_diffusers"))
    from custom_blocks.ltx2_image.transformer_ltx2_image import LTX2ImageTransformer2DModel
    from diffusers_dynamic_offloader import from_pretrained_with_dynamic_offload

    def forbidden(*args, **kwargs):
        raise AssertionError("AutoModel must not resolve a fork-only checkpoint class")

    monkeypatch.setattr(diffusers.AutoModel, "from_pretrained", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    model = LTX2ImageTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        num_layers=1,
    ).eval()
    component = tmp_path / "transformer"
    model.save_pretrained(component)
    loaded = from_pretrained_with_dynamic_offload(
        tmp_path,
        model_loader=LTX2ImageTransformer2DModel,
        subfolder="transformer",
        local_files_only=True,
        low_cpu_mem_usage=True,
        apply_dynamic=False,
    )
    assert type(loaded.module) is LTX2ImageTransformer2DModel
    assert loaded.module.config.in_channels == 4
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.module.state_dict()[name])
    assert not torch.cuda.is_initialized()
