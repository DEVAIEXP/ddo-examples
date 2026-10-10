"""Explicit host-side token-wise activation chunking for standard Diffusers layers.

This does not manage weight placement, residency, streams or allocator budgets.
It is an opt-in video experiment, not a replacement for a DDO runtime.
"""

from dataclasses import dataclass
from types import MethodType

import torch
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import RMSNorm


@dataclass
class FeedForwardChunking:
    """Own reversible per-instance forward wrappers without changing model classes."""

    chunk_size: int
    bindings: list

    @property
    def module_count(self):
        return len(self.bindings)

    def remove(self):
        for module, wrapper in self.bindings:
            if module.forward is not wrapper:
                raise RuntimeError("Token-wise wrapper changed before removal")
        for module, _wrapper in self.bindings:
            del module.forward
        self.bindings.clear()


def enable_feed_forward_chunking(model, chunk_size):
    """Chunk the token axis of unchanged, dense FeedForward instances in eval mode.

    Only token-wise standard Diffusers layers are admitted. Original forwards
    remain responsible for projections, activations and kwargs. Attention and
    temporal context are untouched. Inference-only: dropout/training/quantized or
    modified forwards are refused rather than silently changing their behavior.
    """
    return _enable_tokenwise_chunking(model, chunk_size, FeedForward)


def enable_normalization_chunking(model, chunk_size):
    """Bound FP32 RMSNorm temporaries without changing channel-wise reductions."""
    return _enable_tokenwise_chunking(model, chunk_size, RMSNorm)


def _enable_tokenwise_chunking(model, chunk_size, module_type):
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("Token-wise chunk size must be a positive integer")
    modules = [module for module in model.modules() if type(module) is module_type]
    if not modules:
        raise ValueError(f"No standard Diffusers {module_type.__name__} modules found")
    for module in modules:
        if any(child.training for child in module.modules()) or "forward" in vars(module):
            raise ValueError("Token-wise chunking requires unchanged eval-mode forwards")
        for child in module.modules():
            if isinstance(child, torch.nn.Linear) and type(child) is not torch.nn.Linear:
                raise ValueError("Feed-forward chunking does not support custom linear backends")
            if hasattr(child, "sdnq_dequantizer") or hasattr(child, "quantization_method"):
                raise ValueError("Feed-forward chunking does not support quantized backends")
    bindings = []
    for module in modules:
        original = module.forward

        def chunked(self, hidden_states, *args, _original=original, **kwargs):
            if any(child.training for child in self.modules()) or torch.is_grad_enabled():
                raise RuntimeError("Token-wise chunking is inference-only")
            if hidden_states.ndim != 3:
                # Some normalization call sites use other ranks. Preserve the
                # original calculation there; only the rank-3 token axis is split.
                if module_type is RMSNorm:
                    return _original(hidden_states, *args, **kwargs)
                raise ValueError("Token-wise chunking expects [batch, tokens, channels]")
            if hidden_states.shape[1] <= chunk_size:
                return _original(hidden_states, *args, **kwargs)
            outputs = [
                _original(part, *args, **kwargs)
                for part in hidden_states.split(chunk_size, dim=1)
            ]
            return torch.cat(outputs, dim=1)

        wrapper = MethodType(chunked, module)
        module.forward = wrapper
        bindings.append((module, wrapper))
    return FeedForwardChunking(chunk_size, bindings)
