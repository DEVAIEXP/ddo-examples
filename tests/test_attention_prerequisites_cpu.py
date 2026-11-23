"""Attention prerequisites must be explicit, resumable and independent of CUDA."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import image_suite as suite
from benchmarks import run_benchmark as video


def selection(tmp_path, platform, recipe, execute=True):
    options = argparse.Namespace(platform=platform, recipe=recipe, family='attention',
                                 case='sol-attn' if recipe == video.PRESSURE_RECIPE else 'cudnn-sdpa',
                                 attention_runtime='vmm-fast', reference_manifest=None,
                                 output_root=tmp_path, execute=execute, session_action='continue')
    control = {'platform': platform, 'recipe': recipe, 'family': 'runtime',
               'cases': {'vmm-fast': {'stages': [
                   {'name': name, 'command': ['python', 'runner.py'], 'environment': {}}
                   for name in ('profile-1', 'profile-2', 'measure')]}}}
    identity = {'plan': {**control, 'family': 'attention', 'cases': {options.case: {}}},
                'packages': {'torch': 'test'}, 'sol_attn_dependency': {'revision': 'reviewed'}}
    return options, control, identity


def control_file(options, name='control'):
    path = options.output_root / options.platform / options.recipe / 'sessions' / name / 'manifest.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'identity': {'plan': {'family': 'runtime'}},
                                'cases': {'vmm-fast': {'status': 'completed'}}}))
    return path


@pytest.mark.parametrize('platform', ['windows', 'linux', 'wsl'])
@pytest.mark.parametrize('recipe', ['ltx23-image', 'qwen-image21', video.PRESSURE_RECIPE])
def test_isolated_attention_runs_only_its_missing_control(monkeypatch, tmp_path, platform, recipe):
    args, plan, identity = selection(tmp_path, platform, recipe)
    original = json.dumps(identity, sort_keys=True)
    calls = []

    def run_session(parser, selected, actual_identity, run, **kwargs):
        calls.append(selected.case)
        assert selected.family == 'runtime' and selected.case == 'vmm-fast'
        assert actual_identity['plan'] == plan and 'sol_attn_dependency' not in actual_identity
        assert kwargs['start_if_missing'] is True
        return control_file(args)

    def validate(path, target):
        assert target is identity and suite.read_json(path)['cases']['vmm-fast']['status'] == 'completed'
        return {'manifest': str(path), 'profile_receipt': {'verified': 'hash'}}

    monkeypatch.setattr(suite, 'run_session', run_session)
    result = suite.ensure_attention_reference(argparse.ArgumentParser(), args, identity, plan, validate, None)
    assert calls == ['vmm-fast'] and result['profile_receipt']
    assert args.family == 'attention' and json.dumps(identity, sort_keys=True) == original


@pytest.mark.parametrize('platform', ['windows', 'linux', 'wsl'])
def test_reuses_only_verified_control_without_launching_a_runner(monkeypatch, tmp_path, platform):
    args, plan, identity = selection(tmp_path, platform, video.PRESSURE_RECIPE)
    accepted = control_file(args, 'older')
    rejected = control_file(args, 'newer')
    monkeypatch.setattr(suite, 'run_session', lambda *a, **k: pytest.fail('unexpected execution'))

    def validate(path, target):
        if path == rejected:
            raise ValueError('Driver, output hash or profile mismatch')
        return {'manifest': str(path)}

    result = suite.ensure_attention_reference(argparse.ArgumentParser(), args, identity, plan, validate, None)
    assert result['manifest'] == str(accepted)


@pytest.mark.parametrize('platform', ['windows', 'linux', 'wsl'])
def test_preflight_with_missing_control_does_not_run_or_create_session(monkeypatch, tmp_path, platform):
    args, plan, identity = selection(tmp_path, platform, 'ltx23-image', execute=False)
    monkeypatch.setattr(suite, 'run_session', lambda *a, **k: pytest.fail('preflight launched execution'))
    result = suite.ensure_attention_reference(argparse.ArgumentParser(), args, identity, plan, None, None)
    assert result is None and list(tmp_path.iterdir()) == []


def test_explicit_invalid_reference_is_not_silently_replaced(monkeypatch, tmp_path):
    args, plan, identity = selection(tmp_path, 'linux', 'qwen-image21')
    args.reference_manifest = control_file(args)
    monkeypatch.setattr(suite, 'run_session', lambda *a, **k: pytest.fail('invalid explicit reference replaced'))
    def invalid(*args):
        raise ValueError('Changed reference artifact')
    with pytest.raises(ValueError, match='Changed reference'):
        suite.ensure_attention_reference(argparse.ArgumentParser(), args, identity, plan, invalid, None)


def test_failed_control_stops_before_attention(monkeypatch, tmp_path):
    args, plan, identity = selection(tmp_path, 'linux', video.PRESSURE_RECIPE)
    def failed(*args, **kwargs):
        raise RuntimeError('control failed; preserve attempt')
    monkeypatch.setattr(suite, 'run_session', failed)
    with pytest.raises(RuntimeError, match='control failed'):
        suite.ensure_attention_reference(argparse.ArgumentParser(), args, identity, plan,
                                         lambda *a: pytest.fail('failed control accepted'), None)


def test_continue_can_create_next_dependency_session_but_runtime_stays_strict(monkeypatch, tmp_path):
    args, plan, identity = selection(tmp_path, 'linux', 'ltx23-image')
    identity['plan'] = plan
    args.family = 'runtime'
    def finish(path, state, run, **kwargs):
        state['cases']['vmm-fast']['status'] = 'completed'
        suite.write_state(path, state)
    monkeypatch.setattr(suite, 'execute_session', finish)
    with pytest.raises(SystemExit):
        suite.run_session(argparse.ArgumentParser(), args, identity, None)
    path = suite.run_session(argparse.ArgumentParser(), args, identity, None, start_if_missing=True)
    assert suite.read_json(path)['cases']['vmm-fast']['status'] == 'completed'
    assert suite.run_session(argparse.ArgumentParser(), args, identity, None, start_if_missing=True) == path


@pytest.mark.parametrize('platform', ['windows', 'linux', 'wsl'])
def test_video_cli_previews_missing_control_without_a_reference(monkeypatch, tmp_path, capsys, platform):
    monkeypatch.setattr(video, 'current_platform', lambda: platform)
    monkeypatch.setattr(video.ctypes, 'windll', SimpleNamespace(
        shell32=SimpleNamespace(IsUserAnAdmin=lambda: True)), raising=False)
    monkeypatch.setattr('sys.argv', ['run_benchmark.py', '--platform', platform, '--recipe', video.PRESSURE_RECIPE,
                                  '--family', 'attention', '--case', 'sol-attn', '--attention-runtime', 'vmm-fast',
                                  '--checkpoint', str(tmp_path)])
    video.video_runtime_main(suite)
    result = json.loads(capsys.readouterr().out)
    assert list(result['attention']['cases']) == ['sol-attn']
    control = result['prerequisite']['cases']
    assert list(control) == ['vmm-fast']
    assert [v['name'] for v in control['vmm-fast']['stages']] == ['profile-1', 'profile-2', 'measure']


@pytest.mark.parametrize('platform', ['windows', 'linux', 'wsl'])
@pytest.mark.parametrize('recipe', ['ltx23-image', 'qwen-image21'])
def test_image_cli_previews_missing_control_without_a_reference(monkeypatch, tmp_path, capsys, platform, recipe):
    command = ['run_benchmark.py', '--platform', platform, '--recipe', recipe, '--family', 'attention',
               '--case', 'cudnn-sdpa', '--attention-runtime', 'vmm-fast', '--checkpoint', str(tmp_path)]
    if recipe == 'qwen-image21' and platform != 'wsl':
        command += ['--vae-model-path', str(tmp_path)]
    monkeypatch.setattr('sys.argv', command)
    suite.main(lambda *a: pytest.fail('preview launched a runner'))
    result = json.loads(capsys.readouterr().out)
    assert list(result['attention']['cases']) == ['cudnn-sdpa']
    assert list(result['prerequisite']['cases']) == ['vmm-fast']
