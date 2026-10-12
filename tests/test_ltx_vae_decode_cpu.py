"""LTX decode routing and streaming ownership checks; no CUDA execution."""

import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model", ["ltx2_image", "ltx2_video"])
def test_leaf_decode_is_opt_in_and_uses_public_ddo(model):
    source = (ROOT / "scripts" / model / "run_dynamic_modular_vmm_distilled.py").read_text()
    ast.parse(source)
    assert '--vae-leaf-offload' in source
    assert 'use_stream=False, low_cpu_mem_usage=False' in source
    assert 'component="vae"' in source
    assert 'cleanup_on_error()' in source
    assert 'vae_decode_route' in source
    assert '.enable_group_offload(' not in source


def streaming_helper():
    path = ROOT / "modular_diffusers/custom_blocks/ltx2_video/decoders.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode_video_streaming_tiles")
    namespace = {"torch": torch}
    # Execute only the trusted local helper, without importing model loaders.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace[node.name]


class FakeVAE:
    temporal_compression_ratio = 2
    tile_sample_min_num_frames = 8
    tile_sample_stride_num_frames = 4
    dtype = torch.float32
    device = torch.device("cpu")
    use_tiling = True
    use_framewise_decoding = True

    def decode(self, tile, timestep, return_dict=False):
        assert tile.device.type == "cpu"
        assert not self.use_tiling and not self.use_framewise_decoding
        return (tile.repeat_interleave(2, dim=2)[:, :, :-1],)

    def blend_t(self, previous, row, overlap):
        return row


@pytest.mark.parametrize("frames", [2, 7])
def test_explicit_execution_device_preserves_streaming_output(frames):
    helper = streaming_helper()
    latent = torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1)
    reference = helper(FakeVAE(), latent, None)
    vae = FakeVAE()
    # Parameters do not describe the compute device once offload is attached.
    vae.device = torch.device("meta")
    actual = helper(vae, latent, None, execution_device="cpu")
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert vae.use_tiling and vae.use_framewise_decoding


def test_streaming_restores_flags_after_decode_failure():
    vae = FakeVAE()
    def fail(*args, **kwargs):
        raise RuntimeError("decode failed")
    vae.decode = fail
    with pytest.raises(RuntimeError, match="decode failed"):
        streaming_helper()(vae, torch.zeros(1, 1, 7, 1, 1), None, execution_device="cpu")
    assert vae.use_tiling and vae.use_framewise_decoding


def test_video_phase_monitor_is_not_nested():
    source = (ROOT / "scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py").read_text()
    assert 'step_start("decode_and_save")' not in source
    assert 'scope="aggregate_wall_time_not_an_additional_phase"' in source
    for prefix in ("load_decode_", "decode_", "cleanup_decode_"):
        assert f'step_start("{prefix}" + name)' in source
        assert f'step_end("{prefix}" + name,' in source
