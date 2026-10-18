"""Custom-code setup must never update an existing checkout implicitly."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


spec = importlib.util.spec_from_file_location(
    "setup_blocks", Path(__file__).parents[1] / "scripts/setup_custom_blocks.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_new_clone_uses_main(tmp_path):
    with patch.object(setup, "ROOT", tmp_path), patch.object(setup.subprocess, "run") as run, \
            patch.object(setup.subprocess, "check_output", return_value="abc\n"):
        setup.prepare(models=["qwen_image_21"])
    assert run.call_count == 1
    assert run.call_args.args[0][:4] == ["git", "clone", "--branch", "main"]


def test_existing_checkout_is_preserved(tmp_path):
    (tmp_path / "modular_diffusers/custom_blocks/ltx2_image").mkdir(parents=True)
    with patch.object(setup, "ROOT", tmp_path), patch.object(setup.subprocess, "run") as run, \
            patch.object(setup.subprocess, "check_output", side_effect=[setup.BLOCKS[0][1], "new-head"]):
        setup.prepare(models=["ltx2_image"])
    run.assert_not_called()


def test_explicit_revision_mismatch_refuses_mutation(tmp_path):
    (tmp_path / "modular_diffusers/custom_blocks/ltx2_image").mkdir(parents=True)
    with patch.object(setup, "ROOT", tmp_path), patch.object(setup.subprocess, "run") as run, \
            patch.object(setup.subprocess, "check_output", side_effect=[setup.BLOCKS[0][1], "new", "old"]):
        with pytest.raises(RuntimeError, match="requested revision"):
            setup.prepare(revision="old", models=["ltx2_image"])
    run.assert_not_called()
