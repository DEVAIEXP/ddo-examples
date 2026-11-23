"""Saved FP8 keeps the current eight-second SDPA workload without model features."""

import argparse
import json
from pathlib import Path

import pytest

from benchmarks import run_benchmark as video


@pytest.mark.parametrize('platform', ['windows', 'linux', 'wsl'])
def test_fp8_changes_only_saved_weights_and_profile(monkeypatch, tmp_path, platform):
    from types import SimpleNamespace

    monkeypatch.setattr(video, 'current_platform', lambda: platform)
    monkeypatch.setattr(video.ctypes, 'windll', SimpleNamespace(
        shell32=SimpleNamespace(IsUserAnAdmin=lambda: True)), raising=False)
    args = argparse.Namespace(platform=platform, recipe=video.PRESSURE_RECIPE, family='runtime',
                              case='vmm-fast', checkpoint=tmp_path, custom_blocks=tmp_path,
                              fixtures_dir=tmp_path, profile_file=None, expandable_segments=False,
                              vram_gib=8, fp8_model_path=tmp_path / 'fp8')
    original = video.video_runtime_plan(args)
    args.family, args.case = 'precision', video.FP8_CASE
    selected = video.video_fp8_plan(args)
    base = original['cases']['vmm-fast']
    case = selected['cases'][video.FP8_CASE]
    assert case['routes'] == base['routes'] and case['dispatcher'] is True
    assert case['profile'] != base['profile'] and case['profile'].endswith('-fp8-weight-only')
    assert [v['name'] for v in case['stages']] == ['profile-1', 'profile-2', 'measure']
    for expected, actual in zip(base['stages'], case['stages']):
        command = actual['command'].copy()
        assert command[-3:] == ['--model-path', str(args.fp8_model_path.resolve()), '--transformer-fp8-weight-only']
        command = command[:-3]
        command[command.index('--ddo-profile') + 1] = base['profile']
        assert command == expected['command']
        assert actual['environment'] == expected['environment']
        assert command[command.index('--seconds') + 1] == '8'
        assert command[command.index('--video-self-attention') + 1] == 'default'
        assert '--use-audio-condition' in command
        assert '--no-reuse-rotary-embeddings' in command
        assert '--compile-fp8-linears' not in command and '--fp8-feed-forward-reuse' not in command


def saved_payload(tmp_path):
    folder = tmp_path / 'transformer'
    folder.mkdir()
    config = {'quantization_config': {'quant_method': 'torchao', 'quant_type': {'default': {
        '_type': 'Float8WeightOnlyConfig', '_data': {
            'weight_dtype': {'_data': 'float8_e4m3fn'}, 'granularity': {'_type': 'PerRow'}}}}}}
    (folder / 'config.json').write_text(json.dumps(config))
    (folder / 'diffusion_pytorch_model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'weight': 'shard.safetensors'}}))
    (folder / 'shard.safetensors').write_bytes(b'CPU fixture payload')
    return folder


def test_saved_fp8_identity_hashes_payload_without_loading_tensors(tmp_path):
    folder = saved_payload(tmp_path)
    original = video.video_fp8_identity(tmp_path)
    assert len(original['files']) == 3 and original['torchao_version']
    (folder / 'shard.safetensors').write_bytes(b'Changed fixture payload')
    assert video.video_fp8_identity(tmp_path) != original


@pytest.mark.parametrize('failure', ['quantization', 'path', 'missing'])
def test_saved_fp8_identity_rejects_wrong_format_or_shards(tmp_path, failure):
    folder = saved_payload(tmp_path)
    if failure == 'quantization':
        (folder / 'config.json').write_text('{}')
    elif failure == 'path':
        (folder / 'diffusion_pytorch_model.safetensors.index.json').write_text(
            json.dumps({'weight_map': {'weight': '../outside.safetensors'}}))
    else:
        (folder / 'shard.safetensors').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        video.video_fp8_identity(tmp_path)
