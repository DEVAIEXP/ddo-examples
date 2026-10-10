"""Host glue for the independent experimental Triton Q/K compute candidate."""
import torch
import hashlib
from pathlib import Path


class IndependentQkRopeOps:
    def __init__(self):
        from ddo_vmm_allocator import triton_qk_rope
        self.ops = triton_qk_rope

    def norm_rotary_pair(self, norm_q, norm_k, query, key, qfreqs, kfreqs, layout, heads):
        if layout != "split" or type(norm_q) is not torch.nn.RMSNorm or type(norm_k) is not torch.nn.RMSNorm:
            raise ValueError("Independent Q/K candidate requires original PyTorch RMSNorm and split RoPE")
        if norm_q.training or norm_k.training or torch.is_grad_enabled():
            raise ValueError("Independent Q/K candidate requires eval/no-grad execution")
        if norm_q.weight is None or norm_k.weight is None:
            raise ValueError("Independent Q/K candidate requires live affine weights")
        qcos, qsin = (table.transpose(1, 2).contiguous() for table in qfreqs)
        if (qfreqs[0] is kfreqs[0] and qfreqs[1] is kfreqs[1] and
                query.shape == key.shape and norm_q.eps == norm_k.eps):
            return self.ops.rms_norm_split_rope_qk(
                query.contiguous(), key.contiguous(), norm_q.weight, norm_k.weight,
                qcos, qsin, heads, norm_q.eps)
        kcos, ksin = (table.transpose(1, 2).contiguous() for table in kfreqs)
        return (
            self.ops.rms_norm_split_rope(query.contiguous(), norm_q.weight, qcos, qsin, heads, norm_q.eps),
            self.ops.rms_norm_split_rope(key.contiguous(), norm_k.weight, kcos, ksin, heads, norm_k.eps),
        )


class IndependentQkRopeHandle:
    def __init__(self, model, attentions, callback):
        self.model, self.attentions, self.callback = model, attentions, callback

    def remove(self):
        if any(module.processor._norm_rotary_pair is not self.callback for module in self.attentions):
            raise RuntimeError("Independent Q/K callback changed before retirement")
        self.model.set_execution_ops()
        self.attentions = []


def configure_independent_qk_rope(model):
    """Install optional compute only; preserve parameters and native ownership."""
    from .ltx_transformer_types import resolve_ltx_attention_types
    LTX2Attention, LTX2AudioVideoAttnProcessor, LTX2PerturbedAttnProcessor = resolve_ltx_attention_types(model)
    attentions = [module for module in model.modules() if isinstance(module, LTX2Attention)]
    if model.training or not attentions:
        raise ValueError("Independent Q/K requires an eval LTX model")
    if any(getattr(block, "rms_modulate", None) is not None for block in model.transformer_blocks):
        raise ValueError("Independent Q/K must not mix modulation callbacks")
    for module in attentions:
        if type(module.processor) not in (LTX2AudioVideoAttnProcessor, LTX2PerturbedAttnProcessor):
            raise ValueError("Independent Q/K requires standard processors")
        if module.processor._rotary_pair is not None or module.processor._norm_rotary_pair is not None:
            raise ValueError("Independent Q/K requires unused callback bindings")
        if module.rope_type != "split" or module.heads != 32:
            raise ValueError("Independent Q/K requires the gated split/32-head geometry")
        for norm in (module.norm_q, module.norm_k):
            if (type(norm) is not torch.nn.RMSNorm or norm.training or norm.eps != 1e-6 or
                    norm.normalized_shape not in ((2048,), (4096,)) or norm.weight is None or
                    norm.weight.dtype != torch.bfloat16):
                raise ValueError("Independent Q/K requires gated original BF16 PyTorch norms")
    operations = IndependentQkRopeOps()
    callback = operations.norm_rotary_pair
    model.set_execution_ops(norm_rotary_pair=callback)
    from ddo_vmm_allocator import triton_qk_rope
    report = dict(configuration_applied=True, denoise_completed=False, callback_retired=False,
                  attention_count=len(attentions), dependency="independent_triton_qk_prototype",
                  source_sha256=hashlib.sha256(Path(triton_qk_rope.__file__).read_bytes()).hexdigest(),
                  normalization_backend="pytorch_qk_arithmetic_grouped_fp32_fma",
                  boundary="live_qk_weights_before_native_compute",
                  submission_counts=None,
                  submission_counts_note="Unavailable: no Python counters inside compute",
                  dispatch_note="Pair only for identical frequency tensors and Q/K shapes; otherwise two singles")
    return report, IndependentQkRopeHandle(model, attentions, callback)
