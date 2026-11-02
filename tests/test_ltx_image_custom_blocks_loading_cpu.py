"""Trusted LTX code-source selection without models, downloads or CUDA."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.helpers.ltx2 import image_custom_blocks as loader


@pytest.fixture
def dependencies(monkeypatch):
    import torch
    import diffusers
    import huggingface_hub

    def forbid(*args, **kwargs):
        raise AssertionError("CUDA is forbidden in code-loading CPU tests")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    exports = SimpleNamespace(**{name: object() for name in loader.REQUIRED_EXPORTS})
    factory = Mock(return_value=SimpleNamespace())
    snapshot = Mock()
    monkeypatch.setattr(diffusers.ModularPipelineBlocks, "from_pretrained", factory)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(loader, "import_module", lambda _: exports)
    return exports, factory, snapshot


def arguments(source, revision=None):
    return SimpleNamespace(custom_blocks=str(source), custom_blocks_revision=revision, local_files_only=True)


def test_local_source_is_used_without_download(tmp_path, dependencies):
    exports, factory, snapshot = dependencies
    (tmp_path / "modular_config.json").write_text("{}")
    module, metadata = loader.load_custom_blocks(arguments(tmp_path))
    assert module is exports
    snapshot.assert_not_called()
    factory.assert_called_once_with(str(tmp_path), trust_remote_code=True, local_files_only=True)
    assert metadata["source"] == "local"
    assert metadata["resolved_path"] == str(tmp_path.resolve())


def test_hub_revision_and_offline_mode_are_preserved(tmp_path, dependencies):
    _, factory, snapshot = dependencies
    code = tmp_path / "snapshots" / ("a" * 40)
    code.mkdir(parents=True)
    (code / "modular_config.json").write_text("{}")
    snapshot.return_value = str(code)
    _, metadata = loader.load_custom_blocks(arguments("owner/image-code", "fixed-sha"))
    snapshot.assert_called_once_with(repo_id="owner/image-code", revision="fixed-sha",
                                    allow_patterns=["*.py", "modular_config.json"], local_files_only=True)
    factory.assert_called_once_with(str(code), trust_remote_code=True, local_files_only=True)
    assert metadata["resolved_revision"] == "a" * 40
    assert metadata["requested_revision"] == "fixed-sha"


def test_local_revision_is_rejected(tmp_path, dependencies):
    with pytest.raises(ValueError, match="only to Hub"):
        loader.load_custom_blocks(arguments(tmp_path, "main"))
    dependencies[1].assert_not_called()


def test_missing_local_directory_does_not_become_hub_request(tmp_path, dependencies):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        loader.load_custom_blocks(arguments(tmp_path / "missing"))
    dependencies[2].assert_not_called()


def test_missing_config_is_rejected(tmp_path, dependencies):
    with pytest.raises(FileNotFoundError, match="modular_config.json"):
        loader.load_custom_blocks(arguments(tmp_path))
    dependencies[1].assert_not_called()


@pytest.mark.parametrize("name", loader.REQUIRED_EXPORTS)
def test_partial_source_cannot_mix_with_other_components(name, tmp_path, dependencies):
    delattr(dependencies[0], name)
    (tmp_path / "modular_config.json").write_text("{}")
    with pytest.raises(ImportError, match=name):
        loader.load_custom_blocks(arguments(tmp_path))


@pytest.mark.parametrize("name", ["run_dynamic_minimal.py", "run_dynamic_modular_distilled.py",
    "run_dynamic_modular_sdnq_distilled.py", "run_dynamic_modular_vmm_distilled.py"])
def test_public_runner_uses_selected_namespace_for_all_custom_classes(name):
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/ltx2_image" / name).read_text()
    tree = ast.parse(source)
    assert "from custom_blocks.ltx2_image" not in source
    assert "load_custom_blocks(args)" in source
    assert '"custom_blocks": custom_blocks_metadata' in source
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    bound = {node.targets[0].id: node.value.attr for node in main.body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name)
             and node.value.value.id == "custom"}
    assert all(bound.get(name) == name for name in loader.REQUIRED_EXPORTS)


def test_real_local_custom_code_loads_without_cuda(tmp_path, monkeypatch):
    import torch
    import diffusers.utils.dynamic_modules_utils as dynamic
    import huggingface_hub

    def forbid(*args, **kwargs):
        raise AssertionError("No CUDA or download allowed")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbid)
    monkeypatch.setattr(dynamic, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    module, metadata = loader.load_custom_blocks(arguments(loader.DEFAULT_SOURCE))
    assert all(hasattr(module, name) for name in loader.REQUIRED_EXPORTS)
    assert metadata["source"] == "local"
    assert not torch.cuda.is_initialized()
