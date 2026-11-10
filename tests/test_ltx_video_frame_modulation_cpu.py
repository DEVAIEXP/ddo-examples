"""Bit-exact CPU parity of per-frame modulation in the LTX video custom blocks (no CUDA)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular_diffusers"))
from custom_blocks.ltx2_video.transformer_ltx2_video import LTX2VideoTransformerBlock  # noqa: E402

DIM, AUDIO_DIM, FRAMES, PER_FRAME, AUDIO_TOKENS, TEXT = 32, 16, 3, 5, 4, 6


def block(adaln):
    torch.manual_seed(3)
    module = LTX2VideoTransformerBlock(
        dim=DIM, num_attention_heads=2, attention_head_dim=16, cross_attention_dim=DIM,
        audio_dim=AUDIO_DIM, audio_num_attention_heads=2, audio_attention_head_dim=8,
        audio_cross_attention_dim=AUDIO_DIM, video_gated_attn=True, video_cross_attn_adaln=adaln,
        audio_gated_attn=True, audio_cross_attn_adaln=adaln, rope_type="split")
    return module.eval()


def inputs(adaln):
    torch.manual_seed(5)
    video_params = 9 if adaln else 6
    audio_params = 9 if adaln else 6
    return dict(
        hidden_states=torch.randn(1, FRAMES * PER_FRAME, DIM),
        audio_hidden_states=torch.randn(1, AUDIO_TOKENS, AUDIO_DIM),
        encoder_hidden_states=torch.randn(1, TEXT, DIM),
        audio_encoder_hidden_states=torch.randn(1, TEXT, AUDIO_DIM),
        temb=torch.randn(1, FRAMES, video_params * DIM),  # Compact: one row per frame.
        temb_audio=torch.randn(1, AUDIO_TOKENS, audio_params * AUDIO_DIM),
        temb_ca_scale_shift=torch.randn(1, 1, 4 * DIM),  # use_cross_timestep: per sample.
        temb_ca_audio_scale_shift=torch.randn(1, 1, 4 * AUDIO_DIM),
        temb_ca_gate=torch.randn(1, 1, DIM),
        temb_ca_audio_gate=torch.randn(1, 1, AUDIO_DIM),
        timestep_token_repeat=PER_FRAME,
    )


def run(module, kwargs, *, frame, phase, inplace=False, fused=None):
    module.frame_modulation = frame
    module.phase_modulation = phase
    module.inplace_residuals = inplace
    module.rms_modulate = fused
    cloned = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
    with torch.inference_mode():
        return module(**cloned)


def cpu_fused(norm, x, scale, shift):
    # Same contract as the fused CUDA kernel: rank-3, token dimension 1 or matching.
    assert x.ndim == scale.ndim == shift.ndim == 3 and scale.shape[0] == x.shape[0]
    assert scale.shape[1] in (1, x.shape[1]) and scale.stride(-1) == 1
    return norm(x) * (1 + scale) + shift


@pytest.mark.parametrize("adaln", [False, True])
@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_frame_modulation_is_bit_exact(adaln, phase, inplace):
    module, kwargs = block(adaln), inputs(adaln)
    reference = run(module, kwargs, frame=False, phase=phase, inplace=inplace)
    actual = run(module, kwargs, frame=True, phase=phase, inplace=inplace)
    for got, expected in zip(actual, reference):
        assert torch.equal(got, expected)


@pytest.mark.parametrize("adaln", [False, True])
def test_frame_modulation_with_a_fused_callback_is_bit_exact(adaln):
    module, kwargs = block(adaln), inputs(adaln)
    reference = run(module, kwargs, frame=False, phase=True, fused=cpu_fused)
    actual = run(module, kwargs, frame=True, phase=True, fused=cpu_fused)
    for got, expected in zip(actual, reference):
        assert torch.equal(got, expected)


def test_frame_modulation_never_expands_parameters_per_token():
    module, kwargs = block(True), inputs(True)
    original = torch.Tensor.repeat_interleave
    calls = []

    def observed(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    torch.Tensor.repeat_interleave = observed
    try:
        run(module, kwargs, frame=False, phase=True)
        expanded = len(calls)
        calls.clear()
        run(module, kwargs, frame=True, phase=True)
    finally:
        torch.Tensor.repeat_interleave = original
    assert expanded > 0 and calls == []


def test_profile_name_changes_only_when_frame_modulation_is_enabled():
    from types import SimpleNamespace

    from scripts.helpers.ltx2.vmm_profile import VIDEO_ACTIVATION_OPTIONS, workload_profile_name

    base = dict({name: None for name in VIDEO_ACTIVATION_OPTIONS}, width=1920, height=1056)
    default = workload_profile_name(SimpleNamespace(**base), 241)
    assert workload_profile_name(SimpleNamespace(**base, frame_modulation=False), 241) == default
    assert workload_profile_name(SimpleNamespace(**base, frame_modulation=True), 241) != default


def test_frame_modulation_is_inactive_without_compact_timesteps_and_checks_geometry():
    module, kwargs = block(False), inputs(False)
    dense = dict(kwargs, temb=kwargs["temb"].repeat_interleave(PER_FRAME, dim=1), timestep_token_repeat=1)
    reference = run(module, dense, frame=False, phase=True)
    actual = run(module, dense, frame=True, phase=True)
    for got, expected in zip(actual, reference):
        assert torch.equal(got, expected)
    bad = dict(kwargs, temb=kwargs["temb"][:, :2])
    with pytest.raises(ValueError, match="frames \\* token_repeat"):
        run(module, bad, frame=True, phase=True)
