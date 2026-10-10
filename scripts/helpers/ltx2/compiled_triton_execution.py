"""Host-only opt-in binding for the existing Triton kernels under compilation."""
import hashlib
from pathlib import Path


class CompiledTritonHandle:
    def __init__(self, model, blocks, attentions, rms_callback, rotary_callback):
        self.model, self.blocks, self.attentions = model, blocks, attentions
        self.rms_callback, self.rotary_callback = rms_callback, rotary_callback

    def remove(self):
        if (any(block.rms_modulate is not self.rms_callback for block in self.blocks) or
                any(module.processor._rotary_pair is not self.rotary_callback or
                    module.processor._norm_rotary_pair is not None for module in self.attentions)):
            raise RuntimeError("Compiled Triton callback changed before retirement")
        self.model.set_execution_ops()
        self.blocks, self.attentions = [], []


def configure_compiled_triton(model):
    """Install compute callbacks only; retire after native completion and compilation."""
    from .ltx_transformer_types import resolve_ltx_attention_types
    LTX2Attention, LTX2AudioVideoAttnProcessor, LTX2PerturbedAttnProcessor = resolve_ltx_attention_types(model)
    blocks = list(model.transformer_blocks)
    attentions = [module for module in model.modules() if isinstance(module, LTX2Attention)]
    if model.training or not blocks or not attentions:
        raise ValueError("Compiled Triton requires an eval model with execution callback bindings")
    if any(block.rms_modulate is not None for block in blocks):
        raise ValueError("Compiled Triton requires unused modulation callbacks")
    for module in attentions:
        if (type(module.processor) not in (LTX2AudioVideoAttnProcessor, LTX2PerturbedAttnProcessor) or
                module.processor._rotary_pair is not None or module.processor._norm_rotary_pair is not None):
            raise ValueError("Compiled Triton requires standard processors and unused rotary callbacks")
    from ddo_vmm_allocator import triton_execution, triton_execution_compile
    operations = triton_execution_compile.CompilableTritonExecutionOps()
    rms_callback, rotary_callback = operations.rms_modulate, operations.rotary_pair
    model.set_execution_ops(rms_modulate=rms_callback, rotary_pair=rotary_callback)
    report = dict(configuration_applied=True, denoise_completed=False, callback_retired=False,
                  block_count=len(blocks), attention_count=len(attentions),
                  source_sha256=hashlib.sha256(Path(triton_execution.__file__).read_bytes()).hexdigest(),
                  wrapper_sha256=hashlib.sha256(Path(triton_execution_compile.__file__).read_bytes()).hexdigest(),
                  normalization_backend="original_norm_forward", modulation_backend="original_triton",
                  rotary_backend="original_triton", submission_counts=None,
                  boundary="opaque_compute_only", residency_policy_changed=False)
    return report, CompiledTritonHandle(model, blocks, attentions, rms_callback, rotary_callback)
