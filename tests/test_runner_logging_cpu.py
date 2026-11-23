"""Narrow dependency-notice filtering before benchmark controller imports."""

import ast
import logging
from pathlib import Path

from scripts.helpers.runner_logging import _keep_record


def test_only_known_enum_deprecations_are_hidden():
    for name in ("KernelPreference", "ScaleCalculationMode"):
        message = (f"<enum '{name}'> is an Enum subclass and is now natively supported by torch.compile "
                   "as an opaque value type. Calling register_constant() on Enum subclasses is deprecated "
                   "and will be an error in a future release.")
        assert not _keep_record(logging.LogRecord("torch.utils._pytree", logging.WARNING, "", 0, message, (), None))
        assert _keep_record(logging.LogRecord("torch.utils._pytree", logging.ERROR, "", 0, message, (), None))
    assert _keep_record(logging.LogRecord("torch.utils._pytree", logging.WARNING, "", 0,
                                         "An unrelated warning", (), None))
    assert _keep_record(logging.LogRecord("benchmark", logging.INFO, "", 0,
                                         "Reusable DDO profile: --ddo-profile example", (), None))


def test_controller_installs_filter_before_dispatch_and_profile_export():
    path = Path(__file__).parents[1] / "benchmarks/run_benchmark.py"
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    configure = next(node for node in calls if node.func.id == "configure_runner_logging")
    dispatch = next(node for node in calls if node.func.id == "video_runtime_main")
    assert configure.lineno < dispatch.lineno
