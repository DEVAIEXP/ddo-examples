"""Execute the runner's cleanup slice with fake owners, without Torch or CUDA."""

import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest


def cleanup_slice():
    path = Path(__file__).parents[1] / "scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    end = next(i for i, node in enumerate(main.body)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name)
               and node.value.func.id == "cleanup_before_vae_decode")
    start = max(i for i, node in enumerate(main.body[:end]) if isinstance(node, ast.If))
    return compile(ast.Module(body=main.body[start:end + 1], type_ignores=[]), str(path), "exec")


class Model:
    pass


@pytest.mark.parametrize("diagnostic,dynamic", [(False, True), (False, False), (True, True)])
def test_all_result_owners_are_released_before_vae_cleanup(diagnostic, dynamic):
    calls = []
    model = Model()
    reference = weakref.ref(model)

    class Result:
        def __init__(self, module):
            self.module = module

        def close(self):
            assert reference() is not None
            calls.append("close")

    def remove(module):
        assert module is reference()
        calls.append("remove")

    def cleanup(record_event):
        gc.collect()
        assert reference() is None, "A runner owner still retains the transformer before VAE loading"
        calls.append("cleanup")

    namespace = {
        "args": SimpleNamespace(ddo_vmm_diagnostic_adapter=diagnostic),
        "dynamic_offload_enabled": dynamic,
        "transformer": model,
        "transformer_offload": Result(model),
        "dynamic_offload_hook": None,
        "prepare_pipe": object(), "denoise_pipe": object(), "scheduler": object(),
        "remove_dynamic_offload": remove,
        "cleanup_before_vae_decode": cleanup,
        "record_event": None,
    }
    del model
    exec(cleanup_slice(), namespace)
    assert calls == (["close"] if diagnostic else ["remove"] if dynamic else []) + ["cleanup"]
    assert "transformer_offload" not in namespace


@pytest.mark.parametrize("diagnostic,dynamic", [(False, True), (False, False), (True, True)])
@pytest.mark.parametrize("repeats", [1, 2])
def test_prepare_repeat_releases_all_owners_before_next_load(diagnostic, dynamic, repeats):
    path = Path(__file__).parents[1] / "scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(node for node in main.body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "prepare_repeat_index")
    code = compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec")
    references = []
    calls = []

    class Result:
        def __init__(self, module):
            self.module = module

        def close(self):
            assert self.module is references[-1]()
            calls.append("close")

    def load(index):
        if references:
            assert references[-1]() is None, "Previous transformer survives into the next load"
        model = Model()
        model.cycle = model
        references.append(weakref.ref(model))
        calls.append("load")
        return model, SimpleNamespace(module=model), Result(model)

    def remove(model):
        assert model is references[-1]()
        calls.append("remove")

    def cleanup(record_event, name):
        gc.collect()
        assert references[-1]() is None, "A runner owner survives preparation cleanup"
        calls.append("cleanup")

    namespace = {
        "TRANSFORMER_PREPARE_REPEATS": repeats,
        "args": SimpleNamespace(ddo_vmm_diagnostic_adapter=diagnostic),
        "dynamic_offload_enabled": dynamic,
        "load_and_prepare_transformer": load,
        "remove_dynamic_offload": remove,
        "cleanup_runtime_state": cleanup,
        "record_event": None,
        "transformer_prepare_event_name": lambda name, index: name,
    }
    exec(code, namespace)
    retirement = ["close"] if diagnostic else ["remove"] if dynamic else []
    assert calls == (["load"] + retirement + ["cleanup", "load"] if repeats == 2 else ["load"])
    assert references[-1]() is namespace["transformer"]
    assert namespace["transformer_offload"].module is namespace["transformer"]
    assert namespace["dynamic_offload_hook"].module is namespace["transformer"]
    namespace.clear()
    gc.collect()
    assert references[-1]() is None
