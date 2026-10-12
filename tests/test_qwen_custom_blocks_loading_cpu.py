"""Trusted code routing contracts; no model downloads or CUDA execution."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.helpers import qwen_custom_blocks as loader


@pytest.fixture
def dependencies(monkeypatch):
    import diffusers
    import huggingface_hub

    exports = SimpleNamespace(
        **{
            name: object()
            for name in (
                "QwenImage21TextEncoderStep",
                "QwenImage21PrepareStep",
                "QwenImage21TurboPrepareStep",
                "QwenImage21DenoiseStep",
                "QwenImage21DecodeStep",
                "QwenImage21ImageVAE",
            )
        }
    )
    factory = Mock(return_value=SimpleNamespace())
    snapshot = Mock()
    monkeypatch.setattr(diffusers.ModularPipelineBlocks, "from_pretrained", factory)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(loader, "import_module", lambda _: exports)
    return exports, factory, snapshot


def test_local_uses_trusted_loader_without_download(tmp_path, dependencies):
    exports, factory, snapshot = dependencies
    (tmp_path / "modular_config.json").write_text("{}")
    module, metadata = loader.load_custom_blocks(
        SimpleNamespace(
            custom_blocks=str(tmp_path),
            custom_blocks_revision=None,
            local_files_only=True,
        )
    )
    assert module is exports
    snapshot.assert_not_called()
    factory.assert_called_once_with(str(tmp_path), trust_remote_code=True, local_files_only=True)
    assert metadata["source"] == "local"
    assert metadata["resolved_revision"] is None


def test_hub_snapshot_records_revision_and_offline_mode(tmp_path, dependencies):
    _, factory, snapshot = dependencies
    code = tmp_path / "snapshots" / ("a" * 40)
    code.mkdir(parents=True)
    (code / "modular_config.json").write_text("{}")
    snapshot.return_value = str(code)
    _, metadata = loader.load_custom_blocks(
        SimpleNamespace(
            custom_blocks=loader.DEFAULT_REPOSITORY,
            custom_blocks_revision="published-sha",
            local_files_only=True,
        )
    )
    snapshot.assert_called_once_with(
        repo_id=loader.DEFAULT_REPOSITORY,
        revision="published-sha",
        allow_patterns=["*.py", "modular_config.json"],
        local_files_only=True,
    )
    assert metadata["resolved_revision"] == "a" * 40
    factory.assert_called_once_with(str(code), trust_remote_code=True, local_files_only=True)


def test_local_rejects_hub_revision(tmp_path, dependencies):
    with pytest.raises(ValueError, match="only to Hub"):
        loader.load_custom_blocks(
            SimpleNamespace(
                custom_blocks=str(tmp_path),
                custom_blocks_revision="main",
                local_files_only=True,
            )
        )


def test_missing_entry_config_is_actionable(tmp_path, dependencies):
    with pytest.raises(FileNotFoundError, match="Publish the code first"):
        loader.load_custom_blocks(
            SimpleNamespace(
                custom_blocks=str(tmp_path),
                custom_blocks_revision=None,
                local_files_only=True,
            )
        )


def test_missing_export_refuses_partial_package(tmp_path, dependencies):
    exports, _, _ = dependencies
    del exports.QwenImage21ImageVAE
    (tmp_path / "modular_config.json").write_text("{}")
    with pytest.raises(ImportError, match="QwenImage21ImageVAE"):
        loader.load_custom_blocks(
            SimpleNamespace(
                custom_blocks=str(tmp_path),
                custom_blocks_revision=None,
                local_files_only=True,
            )
        )


@pytest.mark.parametrize(
    "runner",
    [
        "run_dynamic_minimal.py",
        "run_dynamic_modular.py",
        "run_dynamic_modular_vmm.py",
    ],
)
def test_runners_load_all_stages_from_trusted_code(runner):
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/qwen_image_21" / runner).read_text()
    ast.parse(source)
    assert "from custom_blocks.qwen_image_21" not in source
    assert "add_custom_blocks_arguments(parser)" in source
    assert "load_custom_blocks(args)" in source
    assert '"custom_blocks": custom_blocks_metadata' in source
    assert "custom.QwenImage21TurboPrepareStep" in source
