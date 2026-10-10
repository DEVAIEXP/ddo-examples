"""Opt-in regional compilation configuration, not native residency policy."""
from dataclasses import dataclass


@dataclass
class RegionalCompileHandle:
    bindings: list
    rms_handle: object = None

    def remove(self):
        """Restore compute forwards only after native wrappers have retired."""
        if self.rms_handle is not None:
            self.rms_handle.validate_removal()
        for module, compiled, original, had_local, local in self.bindings:
            if module.forward is not compiled:
                raise RuntimeError("Retire native forward boundaries before removing compilation")
        for module, compiled, original, had_local, local in self.bindings:
            if had_local:
                module.forward = local
            else:
                del module.forward
        self.bindings.clear()
        if self.rms_handle is not None:
            self.rms_handle.remove()
            self.rms_handle = None


def configure_regional_compile(model, *, eager_rmsnorm=False, normalization_chunk_size=0,
                               eager_torch_rmsnorm=False):
    """Compile compute forwards before native attach, not module call wrappers.

    Lazy compilation errors propagate; no graph relaxation or eager fallback.
    Native attach must wrap these forwards afterwards and retire first.
    """
    if model.training:
        raise ValueError("Regional compilation diagnostics require an eval model")
    if eager_torch_rmsnorm and not eager_rmsnorm:
        raise ValueError("PyTorch norm boundaries require the explicit eager RMSNorm candidate")
    names = tuple(getattr(model, "_repeated_blocks", ()))
    blocks = [module for module in model.modules() if type(module).__name__ in names]
    if not names or not blocks:
        raise ValueError("No declared repeated blocks found")
    if any(module._compiled_call_impl is not None for module in blocks):
        raise ValueError("Cannot mix compiled module calls with compute-only compilation")
    if any("forward" in vars(module) for module in blocks):
        raise ValueError("Configure compute-only compilation before forward wrappers are installed")
    import torch
    options = {"triton.cudagraphs": False, "emulate_precision_casts": True}
    handle = RegionalCompileHandle([])
    try:
        if eager_rmsnorm:
            from eager_rms_boundary import install_eager_rms_boundaries
            handle.rms_handle = install_eager_rms_boundaries(
                model, normalization_chunk_size, include_torch=eager_torch_rmsnorm)
        for module in blocks:
            original = module.forward
            compiled = torch.compile(original, backend="inductor", fullgraph=True,
                                     dynamic=False, options=options.copy())
            handle.bindings.append((module, compiled, original, False, None))
            module.forward = compiled
    except BaseException:
        handle.remove()
        raise
    report = {
        "requested": True,
        "configuration_applied": True,
        "denoise_completed": False,
        "block_count": len(blocks),
        "block_classes": list(names),
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": options,
        "boundary": "compute_forward_before_native_attach",
        "module_call_compilation": False,
        "normalization_boundary": "opaque_installed_eager_rmsnorm" if eager_rmsnorm else "inductor",
        "opaque_rmsnorm_count": len(handle.rms_handle.bindings) if handle.rms_handle else 0,
        "opaque_rmsnorm_types": handle.rms_handle.counts() if handle.rms_handle else {},
        "opaque_torch_rmsnorm_requested": eager_torch_rmsnorm,
        "opaque_rmsnorm_chunk_size": normalization_chunk_size if eager_rmsnorm else None,
        "timing_note": "Lazy compilation cost is included in the first denoise interval",
    }
    return report, handle
