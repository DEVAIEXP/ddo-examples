"""CPU parity/ownership gates for the local untiled image VAE; no CUDA proof."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular_diffusers"))
from custom_blocks.qwen_image_21 import QwenImage21ImageVAE
from diffusers import AutoencoderKLQwenImage21


@pytest.mark.parametrize("patch_size", [None, 2])
@pytest.mark.parametrize("slicing", [False, True])
def test_local_decode_exact_parity_and_no_cache(patch_size, slicing):
    torch.manual_seed(42)
    config = {
        "base_dim": 16,
        "decoder_base_dim": 16,
        "z_dim": 64,
        "dim_mult": [1, 1, 1, 1, 1],
        "num_res_blocks": 1,
        "patch_size": patch_size,
    }
    native = AutoencoderKLQwenImage21(**config).eval()
    local = QwenImage21ImageVAE(**config).eval()
    local.load_state_dict(native.state_dict(), strict=True)
    local.enable_single_frame_decode()
    if slicing:
        native.enable_slicing()
        local.enable_slicing()
    z = torch.randn(2 if slicing else 1, 64, 1, 2, 2)
    cache_counts = []

    def observe(module, inputs, output):
        cache_counts.append(sum(isinstance(v, torch.Tensor) for v in local._feat_map))

    handle = local.decoder.register_forward_hook(observe)
    with torch.inference_mode():
        reference = native.decode(z).sample
        for _ in range(2):
            actual = local.decode(z, return_dict=False)[0]
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    handle.remove()
    assert cache_counts and all(count == 0 for count in cache_counts)
    assert all(v is None for v in local._feat_map)
    assert set(local.state_dict()) == set(native.state_dict())


def test_native_checkpoint_loads_without_key_conversion(tmp_path):
    native = AutoencoderKLQwenImage21(
        base_dim=16, decoder_base_dim=16, dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1
    ).eval()
    native.save_pretrained(tmp_path)
    local = QwenImage21ImageVAE.from_pretrained(tmp_path, local_files_only=True).eval()
    assert not getattr(local, "single_frame_decode_enabled", False)
    for key, value in native.state_dict().items():
        torch.testing.assert_close(local.state_dict()[key], value, atol=0, rtol=0)


def test_default_and_tiling_delegate_to_native(monkeypatch):
    local = QwenImage21ImageVAE(base_dim=16, decoder_base_dim=16, dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1)
    calls = []
    sentinel = object()

    def native_decode(self, z, return_dict=True):
        calls.append((z.shape, return_dict))
        return sentinel

    monkeypatch.setattr(AutoencoderKLQwenImage21, "_decode", native_decode)
    z = torch.zeros(1, 64, 1, 2, 2)
    assert local._decode(z) is sentinel
    local.enable_single_frame_decode()
    local.enable_tiling()
    assert local._decode(z, return_dict=False) is sentinel
    local.disable_tiling()
    assert local._decode(z.expand(1, 64, 2, 2, 2)) is sentinel
    local.disable_single_frame_decode()
    assert local._decode(z) is sentinel
    assert len(calls) == 4


def test_failure_clears_cache_without_masking_error(monkeypatch):
    local = QwenImage21ImageVAE(base_dim=16, decoder_base_dim=16, dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1)
    local.enable_single_frame_decode()
    original = RuntimeError("decoder failure")

    def fail(*args, **kwargs):
        local._feat_map[0] = torch.ones(1)
        raise original

    monkeypatch.setattr(local.decoder, "forward", fail)
    with pytest.raises(RuntimeError) as caught, torch.inference_mode():
        local.decode(torch.zeros(1, 64, 1, 2, 2))
    assert caught.value is original
    assert all(v is None for v in local._feat_map)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_residual_alias_exact_parity_and_shared_weights(dtype):
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import QwenImage21RMS_norm

    torch.manual_seed(8)
    vae = (
        QwenImage21ImageVAE(base_dim=16, decoder_base_dim=16, dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1)
        .to(dtype)
        .eval()
    )
    vae.enable_single_frame_decode()
    z = torch.randn(1, 64, 1, 2, 2, dtype=dtype)
    original_blocks = list(vae.decoder.up_blocks)
    identities = {key: id(value) for key, value in vae.named_parameters()}
    norm_forward = QwenImage21RMS_norm.forward
    with torch.inference_mode():
        reference = vae.decode(z).sample
        vae.enable_residual_aliasing()
        assert identities == {key: id(value) for key, value in vae.named_parameters()}
        assert norm_forward is QwenImage21RMS_norm.forward
        for before, after in zip(original_blocks, vae.decoder.up_blocks):
            assert before.resnets is after.resnets
            assert before.upsampler is after.upsampler
            assert before.avg_shortcut is after.avg_shortcut
            block_input = torch.randn(1, before.in_dim, 1, 2, 2, dtype=dtype)
            unchanged = block_input.clone()
            torch.testing.assert_close(
                after(block_input, first_chunk=True), before(block_input, first_chunk=True), atol=0, rtol=0
            )
            torch.testing.assert_close(block_input, unchanged, atol=0, rtol=0)
        for _ in range(2):
            torch.testing.assert_close(vae.decode(z).sample, reference, atol=0, rtol=0)
        vae.disable_residual_aliasing()
        torch.testing.assert_close(vae.decode(z).sample, reference, atol=0, rtol=0)


def test_residual_alias_requires_explicit_cacheless_route():
    vae = QwenImage21ImageVAE(base_dim=16, decoder_base_dim=16, dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1).eval()
    with pytest.raises(ValueError, match="explicit single-frame"):
        vae.enable_residual_aliasing()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kernel,stride", [(1, 1), (3, 1), (3, 2)])
def test_direct_padding_convolution_exact_parity(dtype, kernel, stride):
    from custom_blocks.qwen_image_21.vae import QwenImage21DirectPaddingConv
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import QwenImage21CausalConv3d

    torch.manual_seed(12)
    native = QwenImage21CausalConv3d(8, 12, kernel, stride=stride, padding=kernel // 2).to(dtype).eval()
    candidate = QwenImage21DirectPaddingConv(native)
    assert candidate.weight is native.weight
    assert candidate.bias is native.bias
    candidate.direct_padding_active = True
    x = torch.randn(1, 8, 1, 8, 8, dtype=dtype).transpose(-1, -2)
    unchanged = x.clone()
    with torch.inference_mode():
        torch.testing.assert_close(candidate(x), native(x), atol=0, rtol=0)
    torch.testing.assert_close(x, unchanged, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_direct_padding_full_decode_exact_parity(dtype):
    from custom_blocks.qwen_image_21.vae import QwenImage21DirectPaddingConv

    torch.manual_seed(8)
    vae = (
        QwenImage21ImageVAE(base_dim=16, decoder_base_dim=16, dim_mult=[1, 1, 1, 1, 1], num_res_blocks=1)
        .to(dtype)
        .eval()
    )
    vae.enable_single_frame_decode()
    vae.enable_residual_aliasing()
    z = torch.randn(1, 64, 1, 2, 2, dtype=dtype)
    parameters = {key: id(value) for key, value in vae.named_parameters()}
    with torch.inference_mode():
        reference = vae.decode(z).sample
        vae.enable_direct_padding()
        assert parameters == {key: id(value) for key, value in vae.named_parameters()}
        for _ in range(2):
            torch.testing.assert_close(vae.decode(z).sample, reference, atol=0, rtol=0)
            assert all(
                not m.direct_padding_active for m in vae.modules() if isinstance(m, QwenImage21DirectPaddingConv)
            )
        vae.disable_direct_padding()
        torch.testing.assert_close(vae.decode(z).sample, reference, atol=0, rtol=0)
