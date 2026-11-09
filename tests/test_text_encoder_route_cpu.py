"""Text-encoder route defaults of the LTX video runners (CPU only, no model loading)."""
import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.helpers.ltx2 import text_encoder  # noqa: E402


def parse(argv, backend="vmm", preset="auto"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddo-backend", default=backend)
    parser.add_argument("--ddo-preset", default=preset)
    text_encoder.add_text_encoder_arguments(parser)
    args = parser.parse_args(argv)
    return text_encoder.resolve_text_encoder_route(parser, args)


def test_native_route_defaults_to_native_encoder_with_measured_budget():
    args = parse([])
    assert args.text_encoder_native_vmm is True
    assert args.text_encoder_vmm_resident_weight_budget_gb == text_encoder.DEFAULT_TEXT_ENCODER_VMM_BUDGET_GB == 4.0


def test_no_flag_restores_leaf_and_rejects_an_encoder_budget():
    args = parse(["--no-text-encoder-native-vmm"])
    assert args.text_encoder_native_vmm is False
    assert args.text_encoder_vmm_resident_weight_budget_gb is None
    assert text_encoder.build_text_encoder_config(args, object()) is None
    with pytest.raises(SystemExit):
        parse(["--no-text-encoder-native-vmm", "--text-encoder-vmm-resident-weight-budget-gb", "4"])


@pytest.mark.parametrize("backend, preset", [("linear", "auto"), ("vmm", "nvme_baseline")])
def test_other_routes_default_to_leaf_and_refuse_native(backend, preset):
    assert parse([], backend, preset).text_encoder_native_vmm is False
    with pytest.raises(SystemExit):
        parse(["--text-encoder-native-vmm"], backend, preset)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_explicit_budget_is_refused(value):
    with pytest.raises(SystemExit):
        parse(["--text-encoder-vmm-resident-weight-budget-gb", value])


def test_explicit_budget_and_component_config():
    from diffusers_dynamic_offloader import DynamicOffloadConfig

    args = parse(["--text-encoder-vmm-resident-weight-budget-gb", "6"])
    base = DynamicOffloadConfig(execution_mode="vmm_fast_runtime", vmm_resident_weight_budget_gb=0.625)
    config = text_encoder.build_text_encoder_config(args, base)
    assert config.vmm_resident_weight_budget_gb == 6.0
    assert config.vmm_component_tensor_ownership is True and config.vmm_page_mb == 2
    assert base.vmm_resident_weight_budget_gb == 0.625  # Transformer ceiling unchanged.


@pytest.mark.parametrize("module_name, extra", [
    ("scripts.ltx2_video.run_dynamic_modular_vmm_distilled", []),
    ("scripts.ltx2_video.run_dynamic_modular_diagnostics_distilled", ["--ddo-backend", "vmm"]),
    ("scripts.ltx2_video.run_dynamic_minimal", []),
])
def test_video_runners_resolve_the_native_encoder_default(module_name, extra):
    module = __import__(module_name, fromlist=["parse_args"])
    args = module.parse_args([*extra, "--dry-run"])
    assert args.text_encoder_native_vmm is True
    assert args.text_encoder_vmm_resident_weight_budget_gb == 4.0
    assert module.parse_args([*extra, "--no-text-encoder-native-vmm", "--dry-run"]).text_encoder_native_vmm is False
