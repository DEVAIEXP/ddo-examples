"""CPU parity gates for the opt-in band decode of the local image VAE; no CUDA proof."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular_diffusers"))
from custom_blocks.qwen_image_21 import QwenImage21ImageVAE
from diffusers import AutoencoderKLQwenImage21

# Same structure as the checkpoint (five levels, two residual blocks), smaller widths.
CONFIG = {"base_dim": 8, "decoder_base_dim": 8, "z_dim": 64, "dim_mult": [1, 2, 4, 4, 4], "num_res_blocks": 2}


def pair(**overrides):
    torch.manual_seed(7)
    config = {**CONFIG, **overrides}
    native = AutoencoderKLQwenImage21(**config).eval()
    local = QwenImage21ImageVAE(**config).eval()
    local.load_state_dict(native.state_dict(), strict=True)
    local.enable_single_frame_decode()
    return native, local


@pytest.mark.parametrize("patch_size", [None, 2])
@pytest.mark.parametrize("banded_blocks", [1, 2, 4])
@pytest.mark.parametrize("rows", [1, 24, 40, 4096])
def test_band_decode_matches_the_whole_image_decode(patch_size, banded_blocks, rows):
    native, local = pair(patch_size=patch_size)
    local.enable_band_decode(rows=rows, banded_blocks=banded_blocks)
    z = torch.randn(1, 64, 1, 7, 3)  # Odd, non-square latent: bands of uneven size.
    with torch.inference_mode():
        reference = native.decode(z).sample
        actual = local.decode(z).sample
    assert actual.shape == reference.shape
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)


def test_band_decode_composes_with_residual_aliasing_and_direct_padding():
    native, local = pair()
    local.enable_residual_aliasing()
    local.enable_direct_padding()
    local.enable_band_decode(rows=16, banded_blocks=2)
    z = torch.randn(1, 64, 1, 5, 2)
    with torch.inference_mode():
        torch.testing.assert_close(local.decode(z).sample, native.decode(z).sample, atol=1e-5, rtol=1e-5)


def test_halo_covers_the_banded_path_and_a_short_halo_breaks_parity():
    native, local = pair()
    local.enable_band_decode(rows=8, banded_blocks=2)
    # The last up block does not upsample: two banded blocks double the rows once.
    assert local.band_decode_halo >= 2 and local.band_decode_factor == 2
    z = torch.randn(1, 64, 1, 6, 2)
    with torch.inference_mode():
        reference = native.decode(z).sample
        local.band_decode_halo = 0
        broken = local.decode(z).sample
    assert not torch.allclose(broken, reference, atol=1e-5, rtol=1e-5)


def test_band_decode_lowers_the_largest_banded_activation():
    _, local = pair()
    z = torch.randn(1, 64, 1, 32, 2)
    largest = []

    def observe(module, inputs, output):
        largest.append(output.numel())

    # Only the banded (highest-resolution) blocks shrink; earlier blocks run whole.
    handles = [block.register_forward_hook(observe) for block in local.decoder.up_blocks[-2:]]
    with torch.inference_mode():
        local.decode(z)
        whole = max(largest)
        largest.clear()
        local.enable_band_decode(rows=32, banded_blocks=2)
        local.decode(z)
        banded = max(largest)
    for handle in handles:
        handle.remove()
    assert banded < whole / 2


def test_band_decode_requires_the_explicit_single_frame_route_and_valid_settings():
    vae = QwenImage21ImageVAE(**CONFIG).eval()
    with pytest.raises(ValueError, match="single-frame"):
        vae.enable_band_decode()
    vae.enable_single_frame_decode()
    for rows in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            vae.enable_band_decode(rows=rows)
    for blocks in (0, len(vae.decoder.up_blocks) + 1, 1.0):
        with pytest.raises(ValueError, match="Banded blocks"):
            vae.enable_band_decode(banded_blocks=blocks)
    vae.enable_band_decode()
    vae.disable_band_decode()
    assert vae.band_decode_rows is None
