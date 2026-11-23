"""Set the ordinary PyTorch allocator default before CUDA-owning imports."""

import os


def pytorch_allocator_environment(environ=None):
    values = os.environ if environ is None else environ
    return {name: values.get(name) for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}


def configure_pytorch_allocator(environ=None):
    """Preserve either explicit spelling; otherwise request expandable segments.

    This only sets the environment inherited by PyTorch and bootstrap children.
    It does not import Torch, initialize CUDA or configure the DDO native pool.
    """
    values = os.environ if environ is None else environ
    if "PYTORCH_ALLOC_CONF" not in values and "PYTORCH_CUDA_ALLOC_CONF" not in values:
        values["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return pytorch_allocator_environment(values)
