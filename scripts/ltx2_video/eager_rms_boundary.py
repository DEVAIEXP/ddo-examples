"""Experimental opaque boundaries around installed RMSNorm implementations."""
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import List, Optional

import torch
from diffusers.models.normalization import RMSNorm


@torch.library.custom_op("ltx_execution_bench::eager_rmsnorm", mutates_args=(),
                         device_types=("cpu", "cuda"))
def eager_rmsnorm(inputs: torch.Tensor, weight: Optional[torch.Tensor],
                  bias: Optional[torch.Tensor], eps: float, chunk_size: int) -> torch.Tensor:
    """Call the installed eager implementation; do not reimplement its math."""
    state = SimpleNamespace(weight=weight, bias=bias, eps=eps)
    if chunk_size > 0 and inputs.ndim == 3 and inputs.shape[1] > chunk_size:
        return torch.cat([RMSNorm.forward(state, part)
                          for part in inputs.split(chunk_size, dim=1)], dim=1)
    return RMSNorm.forward(state, inputs)


@eager_rmsnorm.register_fake
def fake_eager_rmsnorm(inputs, weight, bias, eps, chunk_size):
    dtype = inputs.dtype if weight is None else torch.promote_types(inputs.dtype, weight.dtype)
    if bias is not None and weight is not None:
        dtype = torch.promote_types(dtype, bias.dtype)
    chunked = chunk_size > 0 and inputs.ndim == 3 and inputs.shape[1] > chunk_size
    return torch.empty_like(inputs, dtype=dtype,
                            memory_format=torch.contiguous_format if chunked else torch.preserve_format)


@dataclass
class EagerRmsHandle:
    bindings: list

    def validate_removal(self):
        for module, wrapper, original, had_local in self.bindings:
            if module.forward is not wrapper:
                raise RuntimeError("Retire later RMSNorm wrappers before restoring eager boundaries")

    def remove(self):
        self.validate_removal()
        for module, wrapper, original, had_local in self.bindings:
            if had_local:
                module.forward = original
            else:
                del module.forward
        self.bindings.clear()

    def counts(self):
        return {
            "diffusers": sum(type(module) is RMSNorm for module, *_ in self.bindings),
            "pytorch": sum(type(module) is torch.nn.RMSNorm for module, *_ in self.bindings),
        }


@torch.library.custom_op("ltx_execution_bench::eager_torch_rmsnorm", mutates_args=(),
                         device_types=("cpu", "cuda"))
def eager_torch_rmsnorm(inputs: torch.Tensor, weight: Optional[torch.Tensor],
                        normalized_shape: List[int], eps: Optional[float]) -> torch.Tensor:
    """Keep the installed PyTorch module's forward, shape and epsilon semantics."""
    if weight is not None and weight.dtype != inputs.dtype:
        raise ValueError("Diagnostic PyTorch RMSNorm requires matching input/weight dtypes")
    state = SimpleNamespace(normalized_shape=tuple(normalized_shape), weight=weight, eps=eps)
    return torch.nn.RMSNorm.forward(state, inputs)


@eager_torch_rmsnorm.register_fake
def fake_eager_torch_rmsnorm(inputs, weight, normalized_shape, eps):
    if weight is not None and weight.dtype != inputs.dtype:
        raise ValueError("Diagnostic PyTorch RMSNorm requires matching input/weight dtypes")
    return torch.empty_like(inputs, memory_format=torch.preserve_format)


def install_eager_rms_boundaries(model, chunk_size=0, *, include_torch=False):
    """Replace standard RMSNorm compute only, preserving live parameter inputs.

    The caller supplies the already-selected normalization chunk size. Existing
    chunk wrappers are restored on removal; no parameters or modules are replaced.
    """
    if type(chunk_size) is not int or chunk_size < 0:
        raise ValueError("RMSNorm chunk size must be a non-negative integer")
    types = (RMSNorm, torch.nn.RMSNorm) if include_torch else (RMSNorm,)
    modules = [module for module in model.modules() if type(module) in types]
    if not modules or any(module.training for module in modules):
        raise ValueError("Eager RMSNorm boundaries require standard eval-mode RMSNorm modules")
    handle = EagerRmsHandle([])
    try:
        for module in modules:
            original, had_local = module.forward, "forward" in vars(module)

            def forward(self, hidden_states):
                return eager_rmsnorm(hidden_states, self.weight, self.bias, float(self.eps), chunk_size)

            def torch_forward(self, hidden_states):
                # PyTorch norms were not token-chunked by the original host.
                return eager_torch_rmsnorm(hidden_states, self.weight, list(self.normalized_shape), self.eps)

            wrapper = MethodType(torch_forward if type(module) is torch.nn.RMSNorm else forward, module)
            module.forward = wrapper
            handle.bindings.append((module, wrapper, original, had_local))
    except BaseException:
        handle.remove()
        raise
    return handle
