"""Model-agnostic PyTorch SDPA backend priority for a denoise region.

Diffusers' default ("native") attention dispatch calls
``torch.nn.functional.scaled_dot_product_attention``. Windows PyTorch wheels
lack flash attention, so PyTorch's default order selects memory-efficient
attention; on the LTX 2.3 video recipe preferring cuDNN attention reduced
attention time from 17.4 to 11.6 s per step. ``cudnn`` sets the priority
flash > cuDNN > memory-efficient > math for the wrapped region only; inputs
cuDNN does not accept (for example some masks) fall back to the next backend.
This changes kernels, not model math, but outputs can differ numerically from
the default backend, so compare outputs before adopting it.
"""
import argparse
from contextlib import contextmanager

ATTENTION_BACKENDS = ("default", "cudnn")


def add_attention_backend_argument(parser):
    parser.add_argument(
        "--attention-backend", choices=ATTENTION_BACKENDS, default="default",
        help="PyTorch SDPA backend priority during denoising: default keeps PyTorch's order; cudnn "
             "prefers flash > cuDNN > memory-efficient > math (opt-in; outputs may differ numerically).")
    return parser


def _validate(backend):
    if backend not in ATTENTION_BACKENDS:
        raise argparse.ArgumentTypeError(f"Unsupported attention backend: {backend!r}")


@contextmanager
def sdpa_backend_priority(backend, metrics=None):
    """Apply the requested SDPA priority inside the block; restore PyTorch's state after."""
    _validate(backend)
    record = {"requested": backend, "applied": False}
    if metrics is not None:
        metrics["attention_backend"] = record
    if backend == "default":
        yield record
        return
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    order = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
             SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    record["priority"] = [backend.name for backend in order]
    record["cudnn_sdp_enabled"] = bool(torch.backends.cuda.cudnn_sdp_enabled())
    with sdpa_kernel(order, set_priority=True):
        record["applied"] = True
        yield record
