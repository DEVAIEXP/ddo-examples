"""Exercise the runner's decode loop without importing CUDA-owning modules."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ("run_dynamic_modular_distilled.py", "run_dynamic_modular_sdnq_distilled.py", "run_dynamic_minimal.py")


@pytest.mark.parametrize("runner", VIDEO)
def test_denoise_sampling_stops_before_transformer_retirement(runner):
    source = (ROOT / "scripts/ltx2_video" / runner).read_text(encoding="utf-8")
    stop = source.index('tracker.step_end("denoise", start)')
    cleanup = source.index('tracker.step_start("cleanup_transformer")')
    retire = source.index('finish_phase("cleanup_transformer", cleanup_start, pipe)')
    assert stop < cleanup < retire
    assert 'finish_phase("denoise", start, pipe)' not in source


@pytest.mark.parametrize("runner", VIDEO)
@pytest.mark.parametrize("streaming", [False, True])
def test_decode_phase_excludes_load_and_cleanup(runner, streaming):
    tree = ast.parse((ROOT / "scripts/ltx2_video" / runner).read_text(encoding="utf-8"))
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and ast.unparse(node.target) == "(name, decoder_block, component_names)")
    clock = [0]
    records, active, calls = {}, [], []

    def start(name):
        assert not active, "Memory sampling phases must not nest"
        active.append(name)
        return clock[0]

    def end(name, beginning):
        assert active.pop() == name
        records[name] = clock[0] - beginning

    def load(*args, **kwargs):
        clock[0] += 10
        return SimpleNamespace(vae=SimpleNamespace(enable_tiling=lambda: None),
                               vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000)),
                               to=lambda _: None, unload_components=lambda _: clock.__setitem__(0, clock[0] + 5))

    def decode(pipe, state, device):
        calls.append(active[-1])
        clock[0] += 20
        state.update(videos=["frames"], audio=["audio"])

    names = ["video", "audio_mel", "audio"] if streaming else ["video", "audio"]
    ns = dict(tracker=SimpleNamespace(step_start=start, step_end=end),
              decoder_stages=[(name, None, []) for name in names],
              load_phase=load, run_stage=decode, state={}, decoded={},
              args=SimpleNamespace(decode_device="cuda:0", stream_video_decode=streaming),
              metrics={"decode_route": {"completed_stages": []}},
              gc=SimpleNamespace(collect=lambda: None),
              torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(ROOT / runner), "exec"), ns)
    assert not active
    assert calls == ["decode_" + name for name in names]
    assert ns["metrics"]["decode_route"]["completed_stages"] == names
    for name in names:
        assert records["load_decode_" + name] == 10
        assert records["decode_" + name] == 20
        assert records["cleanup_decode_" + name] == 5


@pytest.mark.parametrize("runner", VIDEO)
def test_image_load_decode_cleanup_are_disjoint(runner):
    source = (ROOT / "scripts/ltx2_image" / runner).read_text(encoding="utf-8")
    tree = ast.parse(source)
    # This scope must exclude both weight loading and resource retirement.
    start = source.index('step_start("Decode VAE")')
    end = source.index('step_end("Decode VAE", t0)', start)
    assert source.index('step_end("Load VAE for decode", t0)') < start
    assert start < source.index('image = decode_pipe(', start) < end
    assert end < source.index('step_start("Cleanup VAE")', end)
    assert end < source.index('del decode_pipe, vae, image_latent', end)
    assert not any(isinstance(n, ast.Call) and ast.unparse(n.func).endswith("step_start")
                   and n.args and isinstance(n.args[0], ast.Constant)
                   and n.args[0].value == "Pass 2: Decode VAE" for n in ast.walk(tree))
