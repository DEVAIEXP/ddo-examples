"""Check bundled source, package isolation and runner wiring without CUDA."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
from kernels import sol_loader

ROOT = Path(__file__).resolve().parents[1]


def test_source_identity_has_no_gpu_import_side_effects():
    code = """
import sys
from kernels.sol_loader import source_identity
identity = source_identity()
assert identity['source_kind'] == 'bundled'
assert len(identity['files']) == 57
assert 'torch' not in sys.modules and 'triton' not in sys.modules
assert 'sol_attn' not in sys.modules
"""
    subprocess.run([sys.executable, '-I', '-B', '-c',
                    f'import sys; sys.path.insert(0, {str(ROOT)!r});' + code], check=True)


@pytest.fixture
def fixture_source(tmp_path, monkeypatch):
    package = tmp_path / 'sol_attn'
    package.mkdir()
    (package / '__init__.py').write_text('from .interface import value\n')
    (package / 'interface.py').write_text('value = 43\n')
    (package / 'nested.py').write_text('from sol_attn.interface import value\n')
    manifest = {'schema': 1, 'revision': sol_loader.REVISION,
                'files': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                          for p in package.iterdir()}}
    (tmp_path / 'sol_attn_source.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(sol_loader, 'ROOT', tmp_path)
    original = {k: v for k, v in sys.modules.items() if k == 'sol_attn' or k.startswith('sol_attn.')}
    for key in original:
        monkeypatch.delitem(sys.modules, key)
    yield package
    for key in list(sys.modules):
        if key == 'sol_attn' or key.startswith('sol_attn.'):
            sys.modules.pop(key)
    sys.modules.update(original)


def test_loader_uses_bundled_package_without_search_path_changes(fixture_source):
    before = sys.path.copy()
    interface = sol_loader.load_interface()
    assert interface.value == 43
    assert sol_loader.load_interface() is interface
    from sol_attn.nested import value
    assert value == 43
    assert sys.path == before


def test_loader_rejects_already_loaded_external_source(fixture_source, monkeypatch):
    external = ModuleType('sol_attn')
    external.__file__ = str(fixture_source.parent / 'external/__init__.py')
    monkeypatch.setitem(sys.modules, 'sol_attn', external)
    with pytest.raises(ImportError, match='external'):
        sol_loader.load_interface()
    assert sys.modules['sol_attn'] is external


def test_failed_import_does_not_leave_partial_package(fixture_source, monkeypatch):
    def fail(module):
        raise ModuleNotFoundError('fixture dependency unavailable')
    import importlib.machinery
    monkeypatch.setattr(importlib.machinery.SourceFileLoader, 'exec_module', lambda self, module: fail(module))
    with pytest.raises(ModuleNotFoundError):
        sol_loader.load_interface()
    assert not any(k == 'sol_attn' or k.startswith('sol_attn.') for k in sys.modules)


@pytest.mark.parametrize('entry', sorted((ROOT / 'scripts/ltx2_video').glob('*.py')))
def test_every_public_video_entry_uses_shared_attention_or_shared_entry(entry):
    if entry.name == '__init__.py':
        return
    tree = ast.parse(entry.read_text(encoding='utf-8-sig'))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert ('scripts.helpers.ltx2.video_self_attention' in imports
            or 'scripts.ltx2_video.run_dynamic_modular_distilled' in imports
            or 'scripts.ltx2_video.run_dynamic_modular_vmm_distilled' in imports
            or 'scripts.ltx2_video.run_dynamic_modular_diagnostics_distilled' in imports)


def test_real_bundled_interface_import_does_not_initialize_cuda():
    code = """
import torch
def refuse(*args, **kwargs):
    raise AssertionError('CUDA is prohibited')
torch.cuda._lazy_init = refuse
from kernels.sol_loader import load_interface, source_identity
interface = load_interface()
assert interface.__file__ == source_identity()['interface']
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, '-I', '-B', '-c',
                    f'import sys; sys.path.insert(0, {str(ROOT)!r});' + code], check=True)
