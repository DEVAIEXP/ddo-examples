from pathlib import Path
import importlib
import sys
from types import SimpleNamespace

import pytest

from scripts.helpers.model_sources import resolve_model_source, resolve_plan_sources, source_argument
from scripts.helpers.model_sources import resolve_vae_source


@pytest.mark.parametrize('module,flag,expected', [
    ('scripts.qwen_image_21.run_dynamic_modular_vmm', '--vae-vmm-offload',
     'elismasilva/qwen-image-2.1-bf16-vae'),
    ('scripts.qwen_image_21.run_dynamic_modular_vmm_diagnostics', '--vae-vmm-offload',
     'elismasilva/qwen-image-2.1-bf16-vae'),
    ('scripts.ltx2_video.run_dynamic_modular_vmm_distilled', '--transformer-fp8-weight-only',
     'elismasilva/ltx2.3_video_fp8_distilled_transformer'),
    ('scripts.ltx2_video.run_dynamic_modular_diagnostics_distilled', '--transformer-fp8-weight-only',
     'elismasilva/ltx2.3_video_fp8_distilled_transformer'),
])
def test_optional_precision_sources_default_only_when_selected(monkeypatch, module, flag, expected):
    monkeypatch.setattr(sys, 'argv', ['parser-test', '--help'])
    parse = importlib.import_module(module).parse_args
    options = ['--ddo-backend', 'vmm', flag]
    if flag == '--vae-vmm-offload':
        options += ['--vae-vmm-resident-weight-budget-gb', '0.375']
    args = parse(options)
    value = args.vae_model_path if flag == '--vae-vmm-offload' else args.model_path
    assert str(value).replace('\\', '/') == expected


def test_vae_uses_selected_pipeline_or_explicit_local_override(tmp_path):
    pipeline = tmp_path / 'pipeline'
    vae = pipeline / 'vae'
    vae.mkdir(parents=True)
    (vae / 'config.json').write_text('{}')
    assert resolve_vae_source(model_source=pipeline) == vae
    standalone = tmp_path / 'standalone'
    standalone.mkdir()
    (standalone / 'config.json').write_text('{}')
    assert resolve_vae_source(standalone, model_source=pipeline) == standalone


def test_local_source_does_not_contact_hub(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda **kwargs: pytest.fail('Local path contacted Hub')))
    assert resolve_model_source(tmp_path) == tmp_path.resolve()


def test_hub_source_respects_offline_and_default_cache(tmp_path, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    assert resolve_model_source(Path('elismasilva/qwen-image-2.1-bf16-vae'), local_files_only=True) == tmp_path
    assert calls == [{'repo_id': 'elismasilva/qwen-image-2.1-bf16-vae', 'local_files_only': True}]


def test_missing_absolute_source_is_not_a_hub_id(tmp_path):
    with pytest.raises(ValueError):
        resolve_model_source(tmp_path / 'missing')


def test_plan_resolves_once_and_freezes_fp8_metadata(tmp_path, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    source = 'elismasilva/ltx2.3_video_fp8_distilled_transformer'
    command = ['runner', '--model-path', source, '--transformer-fp8-weight-only', '--local-files-only']
    case = {'stages': [{'command': command.copy()}, {'command': command.copy()}], 'fp8_model_path': source}
    plan = {'cases': {'fp8': case}}
    resolve_plan_sources(plan)
    assert len(calls) == 1
    assert case['fp8_model_path'] == str(tmp_path)
    assert all(stage['command'][2] == str(tmp_path) for stage in case['stages'])
    assert plan['hub_model_sources'] == {source: str(tmp_path)}
    resolve_plan_sources(plan)
    assert len(calls) == 1
