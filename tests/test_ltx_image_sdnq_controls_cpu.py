"""Static contracts for SDNQ comparison controls; never import model code."""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/ltx2_image/run_dynamic_modular_sdnq_distilled.py"


def test_leaf_control_never_selects_dense_or_vmm_runtime():
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'leaf_control = args.preset == "diffusers_leaf_offload_compat"' in source
    assert 'execution_mode="plan" if leaf_control else "sdnq_runtime"' in source
    assert 'route="diffusers_group_offload" if leaf_control else "dynamic_offload"' in source
    assert 'plan=leaf_control' in source


def test_purge_follows_connector_cleanup_before_weight_load():
    source = SOURCE.read_text(encoding="utf-8")
    cleanup = source.index('cleanup_runtime_state(record_event, "cleanup_after_connectors")')
    purge = source.index('purge_result = maybe_purge_windows_standby_cache(')
    load = source.index('transformer = LTX2ImageTransformer2DModel.from_pretrained(')
    assert cleanup < purge < load
    assert 'run_metrics["standby_purge"]' in source


def test_reports_environment_and_shared_monitor():
    source = SOURCE.read_text(encoding="utf-8")
    assert 'run_metrics["environment"] = environment_metadata()' in source
    assert 'memory_monitor=args.memory_monitor' in source
    assert '"offload_backend": "diffusers_leaf" if leaf_control else "ddo_sdnq_runtime"' in source
