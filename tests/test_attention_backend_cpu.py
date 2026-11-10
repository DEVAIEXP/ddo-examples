"""SDPA backend priority helper and image runner options (CPU, no models)."""
import argparse
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.helpers.attention_backend import add_attention_backend_argument, sdpa_backend_priority  # noqa: E402


def test_default_records_the_request_and_changes_nothing():
    metrics = {}
    with sdpa_backend_priority("default", metrics) as record:
        assert record == {"requested": "default", "applied": False}
    assert metrics["attention_backend"]["applied"] is False


def test_cudnn_priority_applies_inside_the_block_and_keeps_results_on_cpu():
    metrics = {}
    q = k = v = torch.randn(1, 2, 8, 4)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    with sdpa_backend_priority("cudnn", metrics) as record:
        # CPU inputs are not cuDNN eligible: PyTorch falls back to the math backend.
        inside = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert record["applied"] is True and record["priority"][:2] == ["FLASH_ATTENTION", "CUDNN_ATTENTION"]
    torch.testing.assert_close(inside, reference)
    assert metrics["attention_backend"] is record


def test_unknown_backend_and_parser_choices():
    with pytest.raises(argparse.ArgumentTypeError):
        with sdpa_backend_priority("flash3"):
            pass
    parser = add_attention_backend_argument(argparse.ArgumentParser())
    assert parser.parse_args([]).attention_backend == "default"
    assert parser.parse_args(["--attention-backend", "cudnn"]).attention_backend == "cudnn"
    with pytest.raises(SystemExit):
        parser.parse_args(["--attention-backend", "flash3"])


def load_qwen(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner", "--help"])
    return __import__("scripts.qwen_image_21.run_dynamic_modular_vmm", fromlist=["parse_args"])


def test_qwen_band_decode_options(monkeypatch):
    qwen = load_qwen(monkeypatch)
    args = qwen.parse_args(["--vae-no-temporal-cache", "--vae-band-rows", "256"])
    assert (args.vae_band_rows, args.vae_band_blocks) == (256, 2)
    assert qwen.parse_args([]).vae_band_rows is None
    # Single-frame decode is the Qwen VMM default (375ecd6); bands need it.
    assert qwen.parse_args(["--vae-band-rows", "256"]).vae_band_rows == 256
    for bad in (["--no-vae-no-temporal-cache", "--vae-band-rows", "256"],
                ["--vae-no-temporal-cache", "--vae-tiling", "--vae-band-rows", "256"],
                ["--vae-no-temporal-cache", "--vae-band-rows", "0"],
                ["--vae-band-blocks", "3"]):
        with pytest.raises(SystemExit):
            qwen.parse_args(bad)


def test_attention_backend_is_part_of_the_auto_profile_name(monkeypatch):
    qwen = load_qwen(monkeypatch)
    default = qwen.parse_args(["--ddo-profile", "auto"]).ddo_profile
    cudnn = qwen.parse_args(["--ddo-profile", "auto", "--attention-backend", "cudnn"]).ddo_profile
    assert default != cudnn and default.startswith("qwen-image21-1024x1024-")
