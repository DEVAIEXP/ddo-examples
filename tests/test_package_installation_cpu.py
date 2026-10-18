"""User packages and developer editables must remain separate installation paths."""

from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_flash_attention_is_optional_and_matches_windows_torch_stack():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert not any(dep.startswith("flash-attn") for dep in metadata["project"]["dependencies"])
    requirements = metadata["project"]["optional-dependencies"]["flash-attn"]
    assert len(requirements) == 1
    assert "flash_attn-2.9.1%2Bcu132torch2.12.1cxx11abiTRUE-cp312-cp312-win_amd64.whl" in requirements[0]
    assert "sys_platform == 'win32' and python_version == '3.12'" in requirements[0]


def test_default_sources_do_not_select_local_ddo():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sources = metadata["tool"]["uv"]["sources"]
    assert "diffusers-dynamic-offloader" not in sources
    assert "ddo-vmm-allocator" not in sources
    assert not any("path" in source or "editable" in source for source in sources.values())
    assert sources["torch"]["index"] == "pytorch-cu132"
    assert sources["torchaudio"] == {"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}
    assert "diffusers-dynamic-offloader[vmm]==0.1.0rc2" in metadata["project"]["dependencies"]
    assert "ddo-vmm-allocator==0.1.0rc2" in metadata["project"]["dependencies"]


def test_constraints_preserve_core_versions_without_ddo_source_bindings():
    lines = (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines()
    assert "diffusers==0.41.0" in lines
    assert "torch==2.12.1+cu132" in lines
    assert "sdnq==0.2.7" in lines
    assert "torchaudio==2.11.0+cpu ; sys_platform == 'linux'" in lines
    requirements = [line for line in lines if line and not line.startswith("#")]
    assert all(" @ " not in line and not line.startswith("-e") for line in requirements)
    assert not any(line.startswith(("diffusers-dynamic-offloader", "ddo-vmm-allocator")) for line in requirements)


def test_user_guide_selects_wheels_and_developer_guide_selects_editables():
    user = (ROOT / "docs/user/setup.md").read_text(encoding="utf-8")
    developer = (ROOT / "docs/developer/setup.md").read_text(encoding="utf-8")
    assert '--constraint "$examples/constraints.txt" -r "$examples/pyproject.toml"' in user
    assert 'git clone https://github.com/DEVAIEXP/ddo-examples.git' in user
    assert 'Set-Location -LiteralPath $examples' in user
    assert 'cd "$examples"' in user
    assert 'uv venv --managed-python --python 3.12.12 .venv-packages' in user
    assert 'https://test.pypi.org' not in user
    assert '--index-url https://pypi.org/simple/' in user
    assert '--find-links' not in user
    assert 'C:/path/to/' not in user
    assert "<environment-python>" not in user
    assert "-e ../diffusers-dynamic-offloader" not in user
    assert "-e ../diffusers-dynamic-offloader" in developer
    assert "--no-deps" in developer
    assert "uv sync --locked" not in developer


def test_windows_ffmpeg_setup_documents_existing_video_runner_setting():
    user = (ROOT / "docs/user/setup.md").read_text(encoding="utf-8")
    assert "$env:DDO_RUNNER_FFMPEG_BIN" in user
    assert "shared" in user
    assert "Could not load libtorchcodec" in user
    assert "CUDA initialized: False" in user
    for name in (
        "run_dynamic_minimal.py", "run_dynamic_modular_distilled.py",
        "run_dynamic_modular_sdnq_distilled.py",
        "run_dynamic_modular_vmm_distilled.py",
        "run_dynamic_modular_diagnostics_distilled.py",
    ):
        source = (ROOT / "scripts/ltx2_video" / name).read_text(encoding="utf-8")
        assert 'os.getenv("DDO_RUNNER_FFMPEG_BIN", "")' in source
