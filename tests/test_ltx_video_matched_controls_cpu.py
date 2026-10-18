"""Equal model-math controls without native allocator opt-in; no CUDA execution."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest


def validator():
    path = Path(__file__).parents[1] / "scripts/ltx2_video/run_dynamic_modular_diagnostics_distilled.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "validate_allocator_controls")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def controls(**overrides):
    values = dict(fused_rms_modulation=False, fused_rotary_pair=False,
                  release_pytorch_host_cache_before_transformer=False,
                  detach_transformer_auxiliary_cpu_sources=False, inspect_host_memory_phases=False,
                  fp8_group_offload_control=False, feed_forward_chunk_size=1024,
                  normalization_chunk_size=1024, phase_modulation=True,
                  compact_timesteps=True, inplace_residuals=True)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("native", [False, True])
def test_equal_model_controls_are_allowed(native):
    validator()(controls(), native)


@pytest.mark.parametrize("name", ["fused_rms_modulation", "fused_rotary_pair",
                                 "release_pytorch_host_cache_before_transformer",
                                 "detach_transformer_auxiliary_cpu_sources", "inspect_host_memory_phases"])
def test_allocator_experiments_remain_native(name):
    with pytest.raises(ValueError, match="Native video controls"):
        validator()(controls(**{name: True}), False)
    validator()(controls(**{name: True}), True)


def test_existing_explicit_fp8_control_exception_is_preserved():
    validator()(controls(fp8_group_offload_control=True,
                         release_pytorch_host_cache_before_transformer=True), False)


def test_decoder_reports_separate_phases_without_nested_monitor():
    path = Path(__file__).parents[1] / "scripts/ltx2_video/run_dynamic_modular_diagnostics_distilled.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'step_start("decode_and_save")' not in source
    assert 'scope="aggregate_wall_time_not_an_additional_phase"' in source
    for prefix in ("load_decode_", "decode_", "cleanup_decode_"):
        assert f'step_start("{prefix}" + name)' in source
        assert f'step_end("{prefix}" + name,' in source


def test_before_vae_cache_release_is_explicit_and_disabled_by_default():
    from scripts.ltx2_video.run_dynamic_modular_diagnostics_distilled import parse_args
    assert parse_args([]).release_pytorch_host_cache_before_vae is False
    assert parse_args(["--release-pytorch-host-cache-before-vae"]).release_pytorch_host_cache_before_vae is True


@pytest.mark.parametrize("enabled", [False, True])
def test_before_vae_release_runs_after_caller_owners_are_deleted(enabled):
    path = Path(__file__).parents[1] / "scripts/ltx2_video/run_dynamic_modular_diagnostics_distilled.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    release = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                   and node.test.attr == "release_pytorch_host_cache_before_vae")
    deletion = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.Delete)
                    and {target.id for target in node.targets if isinstance(target, ast.Name)}
                    == {"result", "transformer", "scheduler"})
    assert deletion.lineno < release.lineno

    class Model:
        pass

    model = Model()
    reference = weakref.ref(model)
    calls = []

    def free_cache(**options):
        gc.collect()
        assert reference() is None
        assert options == {"enabled": True, "phase": "before_vae", "device": "cuda:0"}
        calls.append("release")
        return "public_report"

    def report(value, name):
        assert value == "public_report" and name == "host_cache_before_vae"
        calls.append("report")

    namespace = dict(args=SimpleNamespace(release_pytorch_host_cache_before_vae=enabled),
                     result=SimpleNamespace(module=model), transformer=model, scheduler=object(),
                     release_pytorch_host_cache=free_cache, report=report, device="cuda:0")
    del model
    exec(compile(ast.Module(body=[deletion, release], type_ignores=[]), str(path), "exec"), namespace)
    assert calls == (["release", "report"] if enabled else [])
