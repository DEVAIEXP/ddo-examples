"""VMM residency profile and text-encoder options of the image VMM runners (CPU, no models)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(module_name, monkeypatch):
    # "--help" in argv skips the fresh-process bootstrap executed at import.
    monkeypatch.setattr(sys, "argv", ["runner", "--help"])
    return __import__(module_name, fromlist=["parse_args"])


def ltx_image(monkeypatch, argv):
    pytest.importorskip("ddo_vmm_allocator")  # This runner imports the allocator adapter directly.
    module = load("scripts.ltx2_image.run_dynamic_modular_vmm_distilled", monkeypatch)
    monkeypatch.setattr(sys, "argv", ["runner", *argv])
    return module.parse_args()


def qwen(monkeypatch, argv):
    return load("scripts.qwen_image_21.run_dynamic_modular_vmm", monkeypatch).parse_args(argv)


@pytest.mark.parametrize("parse, prefix", [(ltx_image, "ltx23-image-1280x704-"), (qwen, "qwen-image21-1024x1024-")])
def test_profile_options_and_omitted_budget(monkeypatch, parse, prefix):
    default = parse(monkeypatch, [])
    assert default.ddo_profile is None and default.build_ddo_profile is False
    assert getattr(default, "vmm_resident_weight_budget_gb", getattr(default, "ddo_vmm_resident_weight_budget_gb", 0)) is None
    built = parse(monkeypatch, ["--ddo-profile", "auto", "--build-ddo-profile"])
    assert built.ddo_profile.startswith(prefix) and built.build_ddo_profile is True
    with pytest.raises(SystemExit):
        parse(monkeypatch, ["--build-ddo-profile"])
    with pytest.raises(SystemExit):
        parse(monkeypatch, ["--ddo-preset", "nvme_baseline", "--ddo-profile", "auto"])


@pytest.mark.parametrize("parse, default", [(ltx_image, True), (qwen, False)])
def test_native_text_encoder_default_and_its_own_ceiling(monkeypatch, parse, default):
    assert parse(monkeypatch, []).text_encoder_native_vmm is default  # Qwen: leaf until measured.
    native = parse(monkeypatch, ["--text-encoder-native-vmm"])
    assert native.text_encoder_native_vmm is True and native.text_encoder_vmm_resident_weight_budget_gb == 4.0
    assert parse(monkeypatch, ["--no-text-encoder-native-vmm"]).text_encoder_native_vmm is False


def test_native_encoder_config_enables_native_transport():
    # Component ownership is refused without native transport (Qwen's generic config has it off).
    from diffusers_dynamic_offloader import DynamicOffloadConfig
    from scripts.helpers.text_encoder_route import native_text_encoder_config

    config = native_text_encoder_config(DynamicOffloadConfig(vmm_native_transport=False), 4.0)
    assert config.vmm_native_transport is True and config.vmm_component_tensor_ownership is True
    assert config.vmm_resident_weight_budget_gb == 4.0 and config.profile_name is None


def test_ltx_image_keeps_leaf_with_explicit_compatibility_or_other_routes(monkeypatch):
    assert ltx_image(monkeypatch, ["--text-encoder-group-offload"]).text_encoder_native_vmm is False
    assert ltx_image(monkeypatch, ["--ddo-preset", "nvme_baseline"]).text_encoder_native_vmm is False


def test_ltx_image_builds_three_steps_and_keeps_the_nvme_budget(monkeypatch):
    assert ltx_image(monkeypatch, ["--ddo-profile", "auto", "--build-ddo-profile"]).ddo_profile_steps == 3
    assert ltx_image(monkeypatch, ["--ddo-preset", "nvme_baseline"]).vmm_resident_weight_budget_gb == 2.0
    assert ltx_image(monkeypatch, ["--ddo-vmm-resident-weight-budget-gb", "4"]).vmm_resident_weight_budget_gb == 4.0
