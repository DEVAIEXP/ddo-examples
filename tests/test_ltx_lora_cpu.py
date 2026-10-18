"""Unfused adapter loading and base-storage contracts, without CUDA/network."""

import argparse
import ast
from pathlib import Path

import pytest
import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import PeftAdapterMixin
from safetensors.torch import save_file

from scripts.helpers.ltx2.lora import add_lora_arguments, validate_lora_arguments, load_ltx_lora


class TinyTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin):
    @register_to_config
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)

    def forward(self, value):
        return self.proj(value)


def arguments(values):
    parser = argparse.ArgumentParser()
    add_lora_arguments(parser)
    args = parser.parse_args(values)
    validate_lora_arguments(parser, args)
    return args


def test_disabled_lora_does_not_touch_model_or_reports():
    metrics = {}
    load_ltx_lora(None, arguments([]), metrics, None)
    assert metrics == {}


@pytest.mark.parametrize("values", [["--lora-scale", "nan"], ["--lora-scale", "0.8"],
                                    ["--lora-weight-name", "x.safetensors"],
                                    ["--lora", "first", "--lora", "second", "--lora-scale", "0.8"]])
def test_invalid_arguments(values):
    with pytest.raises(SystemExit):
        arguments(values)


def test_actual_peft_adapter_changes_output_and_preserves_base_storage(tmp_path):
    from diffusers_dynamic_offloader import enable_offload, remove_dynamic_offload

    model = TinyTransformer().eval()
    original_weight = model.proj.weight
    storage = original_weight.data_ptr()
    base = original_weight.detach().clone()
    value = torch.ones(2, 4)
    a, b = torch.ones(2, 4), torch.ones(3, 2) * 0.25
    path = tmp_path / "adapter.safetensors"
    save_file({"transformer.proj.lora_A.weight": a, "transformer.proj.lora_B.weight": b}, path)
    args = arguments(["--lora", str(path), "--lora-scale", "0.8"])
    metrics, events = {}, []
    load_ltx_lora(model, args, metrics, lambda *a, **k: events.append((a, k)))
    assert model.proj.base_layer.weight is original_weight
    assert model.proj.base_layer.weight.data_ptr() == storage
    torch.testing.assert_close(model.proj.base_layer.weight, base, rtol=0, atol=0)
    expected = torch.nn.functional.linear(value, base) + 0.8 * torch.nn.functional.linear(
        torch.nn.functional.linear(value, a), b)
    with torch.inference_mode():
        torch.testing.assert_close(model(value), expected)
    assert metrics["lora_adapters"][0]["target_count"] == 1
    assert metrics["lora_adapters"][0]["fused"] is False
    assert events[0][0][0] == "load_transformer_lora"
    result = enable_offload(model, preset="one_shot_fast", execution_device="cpu", offload_device="cpu",
                            use_environment=False)
    try:
        with torch.inference_mode():
            torch.testing.assert_close(model(value), expected)
    finally:
        remove_dynamic_offload(model)
    assert not torch.cuda.is_initialized()


def test_unmatched_checkpoint_fails_before_model_mutation(tmp_path):
    model = TinyTransformer()
    path = tmp_path / "bad.safetensors"
    save_file({"transformer.missing.lora_A.weight": torch.ones(2, 4)}, path)
    with pytest.raises(ValueError, match="unmatched"):
        load_ltx_lora(model, arguments(["--lora", str(path)]), {}, None)
    assert type(model.proj) is torch.nn.Linear


def test_all_ltx_entries_expose_and_load_shared_adapter():
    root = Path(__file__).resolve().parents[1]
    for family in ("ltx2_image", "ltx2_video"):
        for path in (root / "scripts" / family).glob("run_dynamic*.py"):
            text = path.read_text(encoding="utf-8")
            ast.parse(text)
            assert "add_lora_arguments(parser)" in text, path
            assert "validate_lora_arguments(parser, args)" in text, path
            assert "load_ltx_lora(" in text, path


def test_every_ltx_parser_accepts_the_same_lora_arguments(monkeypatch):
    from test_runner_controls_cpu import RUNNERS, parser_body
    for path in RUNNERS:
        if path.parent.name not in ("ltx2_image", "ltx2_video"):
            continue
        monkeypatch.setattr("sys.argv", [str(path), "--lora", "example/adapter", "--lora-scale", "0.8"])
        args = parser_body(path)()
        assert args.lora == ["example/adapter"]
        assert args.lora_scale == [0.8]


def test_two_actual_adapters_remain_active_with_independent_scales(tmp_path):
    model = TinyTransformer().eval()
    base = model.proj.weight.detach().clone()
    pointer = model.proj.weight.data_ptr()
    value = torch.ones(2, 4)
    paths, pairs = [], []
    for index, multiplier in enumerate((0.25, 0.5)):
        a, b = torch.ones(2, 4), torch.full((3, 2), multiplier)
        path = tmp_path / f"adapter{index}.safetensors"
        save_file({"transformer.proj.lora_A.weight": a, "transformer.proj.lora_B.weight": b}, path)
        paths.append(path)
        pairs.append((a, b))
    args = arguments(["--lora", str(paths[0]), "--lora-scale", "0.8",
                      "--lora", str(paths[1]), "--lora-scale", "0.5"])
    metrics = {}
    load_ltx_lora(model, args, metrics, lambda *a, **k: None)
    expected = torch.nn.functional.linear(value, base)
    for scale, (a, b) in zip((0.8, 0.5), pairs):
        expected = expected + scale * torch.nn.functional.linear(torch.nn.functional.linear(value, a), b)
    with torch.inference_mode():
        torch.testing.assert_close(model(value), expected)
    assert set(model.proj.active_adapters) == {"runner_lora", "runner_lora_2"}
    assert model.proj.base_layer.weight.data_ptr() == pointer
    torch.testing.assert_close(model.proj.base_layer.weight, base, rtol=0, atol=0)
    assert [r["scale"] for r in metrics["lora_adapters"]] == [0.8, 0.5]
    assert len({r["adapter_name"] for r in metrics["lora_adapters"]}) == 2
    assert not torch.cuda.is_initialized()
