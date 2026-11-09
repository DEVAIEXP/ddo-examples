"""VMM residency profile options of the LTX video runners (CPU only, no model loading)."""
import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.helpers.ltx2 import vmm_profile  # noqa: E402


def parse(argv, backend="vmm", preset="auto", **defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddo-backend", default=backend)
    parser.add_argument("--ddo-preset", default=preset)
    for name, value in dict(width=1280, height=704, feed_forward_chunk_size=0, normalization_chunk_size=0,
                            video_self_attention="cudnn", **defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), default=value, type=type(value))
    vmm_profile.add_vmm_profile_arguments(parser)
    args = parser.parse_args(argv)
    vmm_profile.resolve_vmm_profile(parser, args, 193)
    return args


def test_auto_name_carries_the_workload_shape_and_activation_options():
    name = parse(["--ddo-profile", "auto"]).ddo_profile
    assert name.startswith("ltx23-video-1280x704-193f-")
    assert parse(["--ddo-profile", "auto", "--width", "1920", "--height", "1056"]).ddo_profile.startswith(
        "ltx23-video-1920x1056-193f-")
    chunked = parse(["--ddo-profile", "auto", "--feed-forward-chunk-size", "1024"]).ddo_profile
    assert chunked != name  # Activation-relevant options select another profile.


def test_explicit_name_and_build_validation():
    assert parse(["--ddo-profile", "my-run"]).ddo_profile == "my-run"
    assert parse([]).ddo_profile is None
    for argv in (["--build-ddo-profile"], ["--ddo-profile", " "]):
        with pytest.raises(SystemExit):
            parse(argv)


@pytest.mark.parametrize("backend, preset", [("linear", "auto"), ("vmm", "nvme_baseline")])
def test_profiles_require_the_native_fast_route(backend, preset):
    with pytest.raises(SystemExit):
        parse(["--ddo-profile", "auto"], backend, preset)


@pytest.mark.parametrize("module_name, extra", [
    ("scripts.ltx2_video.run_dynamic_modular_vmm_distilled", []),
    ("scripts.ltx2_video.run_dynamic_modular_diagnostics_distilled", ["--ddo-backend", "vmm"]),
])
def test_vmm_video_runners_accept_profile_options(module_name, extra):
    module = __import__(module_name, fromlist=["parse_args"])
    args = module.parse_args([*extra, "--ddo-profile", "auto", "--build-ddo-profile", "--steps", "1", "--dry-run"])
    assert args.build_ddo_profile is True
    assert args.ddo_profile.startswith("ltx23-video-1280x704-")
    assert args.ddo_vmm_resident_weight_budget_gb is None
