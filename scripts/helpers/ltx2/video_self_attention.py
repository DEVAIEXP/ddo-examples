from time import perf_counter
from contextlib import contextmanager, nullcontext
from copy import copy
from types import FunctionType


class VideoSelfAttention:
    def __init__(self, original, load, rotary_pair=None):
        self.original = original
        self.load = load
        self.rotary_pair = rotary_pair
        self.kernel = None
        self.selected = 0
        self.fallback = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, query_rotary_emb=None, key_rotary_emb=None,
                 perturbation_mask=None, all_perturbed=None):
        args = (attn, hidden_states, encoder_hidden_states, attention_mask,
                query_rotary_emb, key_rotary_emb)
        if type(self.original).__name__ == 'LTX2PerturbedAttnProcessor':
            args += (perturbation_mask, all_perturbed)
        elif perturbation_mask is not None or all_perturbed is not None:
            raise TypeError('Unexpected perturbation arguments for the standard processor')
        if (encoder_hidden_states is not None or attention_mask is not None
                or self.original._parallel_config is not None
                or perturbation_mask is not None or all_perturbed):
            self.fallback += 1
            return self.original(*args)

        method = type(self.original).__call__
        scope = dict(method.__globals__)
        dense = scope['dispatch_attention_fn']

        def dispatch(q, k, v, **options):
            if self.load is None:
                return dense(q, k, v, **options)
            if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape
                    or any(t.dtype != q.dtype or t.device != q.device for t in (k, v))
                    or any(size <= 0 for size in q.shape)):
                self.fallback += 1
                return dense(q, k, v, **options)
            if self.kernel is None:
                self.kernel = self.load()
            result = self.kernel(q, k, v)
            if result.shape != q.shape or result.dtype != q.dtype or result.device != q.device:
                raise ValueError('Invalid optional attention output')
            self.selected += 1
            return result

        scope['dispatch_attention_fn'] = dispatch
        processor = copy(self.original) if self.rotary_pair is not None else self.original
        if self.rotary_pair is not None:
            processor._rotary_pair = self.rotary_pair
        if hasattr(processor, 'attention_dispatch_override'):
            if processor is self.original:
                processor = copy(processor)
            processor.attention_dispatch_override = dispatch
            return processor(*args)
        invoke = FunctionType(method.__code__, scope, argdefs=method.__defaults__)
        return invoke(processor, *args)


@contextmanager
def install(model, layers, processor_type, load, *, rotary_pair=None):
    blocks = model.transformer_blocks
    if len(set(layers)) != len(layers) or any(type(i) is not int or not 0 <= i < len(blocks) for i in layers):
        raise ValueError('Invalid video layer selection')
    targets = [blocks[i].attn1 for i in layers]
    if any(type(target.processor) is not processor_type for target in targets):
        raise TypeError('Unsupported video self-attention processor')
    installed = []
    try:
        for target in targets:
            if rotary_pair is not None and (
                not hasattr(target.processor, '_rotary_pair')
                or target.processor._rotary_pair is not None
                or getattr(target.processor, '_norm_rotary_pair', None) is not None
            ):
                raise ValueError('Rotary fusion requires an unused rotary callback.')
            wrapper = VideoSelfAttention(target.processor, load, rotary_pair)
            installed.append((target, wrapper))
            target.set_processor(wrapper)
        yield tuple(wrapper for _, wrapper in installed)
    finally:
        for target, wrapper in reversed(installed):
            target.set_processor(wrapper.original)

def add_video_self_attention_argument(parser):
    parser.add_argument('--video-self-attention', choices=('default', 'flash-attn2', 'sol-attn', 'cudnn', 'cudnn-only'),
                        default='default', help='Backend for eligible video self-attention only.')
    parser.add_argument('--trace-attention-operators', action='store_true',
                        help='Profile PyTorch attention operators during denoising for diagnostics.')


def record_attention_adjusted_total(metrics):
    finalized = metrics.get('attention_environment', {}).get('attention_profiler_finalize_sec')
    if finalized is not None:
        metrics['total_elapsed_excluding_attention_profiler_finalize_sec'] = round(
            metrics['total_elapsed_sec'] - finalized, 4
        )


@contextmanager
def video_self_attention(transformer, backend, metrics, *, fused_rotary=False, trace_operators=False):
    from diffusers.models import attention_dispatch

    registry = getattr(attention_dispatch, '_AttentionBackendRegistry', None)
    active = getattr(registry, 'get_active_backend', None)
    default_backend = active()[0] if active is not None else None
    overrides = {}
    for index, block in enumerate(getattr(transformer, 'transformer_blocks', ())):
        processor = getattr(getattr(block, 'attn1', None), 'processor', None)
        override = getattr(processor, '_attention_backend', None)
        if override is not None:
            overrides[str(index)] = str(getattr(override, 'value', override))
    metrics.setdefault('attention_environment', {}).update({
        'diffusers_dispatch_default': getattr(default_backend, 'value', default_backend),
        'video_self_attention_processor_overrides': overrides,
        'pytorch_attention_operators': None,
    })
    dispatch = metrics['attention_environment']['diffusers_dispatch_default']
    print(f"  [attention] requested={backend}, Diffusers default={dispatch or 'unavailable'}, "
          f"processor overrides={overrides}, operator trace={'on' if trace_operators else 'off'}", flush=True)
    if trace_operators:
        from torch.profiler import ProfilerActivity, profile

        trace = profile(activities=[ProfilerActivity.CPU])
    else:
        trace = nullcontext()
    with _configure_video_self_attention(transformer, backend, metrics, fused_rotary=fused_rotary,
                                         trace_operators=trace_operators):
        with trace:
            yield
            finalize_start = perf_counter() if trace_operators else None
        if trace_operators:
            operators = {
                event.key: event.count for event in trace.key_averages()
                if (event.key.startswith('aten::') and 'attention' in event.key)
                or event.key == 'flash_attn::_flash_attn_forward'
                or event.key == 'ddo::sol_attn_triton'
            }
            elapsed = perf_counter() - finalize_start
            metrics['attention_environment'].update(
                pytorch_attention_operators=operators, attention_profiler_finalize_sec=elapsed
            )
            print(f"  [attention] Profiler finalization: {elapsed:.4f}s", flush=True)
            print(f"  [attention] PyTorch operators observed during denoising: {operators}", flush=True)


def cudnn_sdpa(q, k, v, *, strict=False):
    """PyTorch SDPA with cuDNN attention preferred, for (batch, tokens, heads, dim) inputs.

    The priority order is flash, cuDNN, memory-efficient, math. Builds without
    flash attention (for example Windows wheels) therefore select cuDNN when the
    inputs are eligible and otherwise fall back to the remaining PyTorch backends.
    Strict mode requires cuDNN and propagates unsupported-kernel errors.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    order = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
             SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    if strict:
        order = [SDPBackend.CUDNN_ATTENTION]
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    with sdpa_kernel(order, set_priority=True):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2)


@contextmanager
def _configure_video_self_attention(transformer, backend, metrics, *, fused_rotary=False,
                                    trace_operators=False):
    """Apply the selected backend only to eligible video self-attention."""
    if backend == 'default' and not fused_rotary:
        metrics['video_self_attention'] = {'requested_backend': backend}
        yield
        return

    from contextlib import ExitStack

    if backend in ('flash-attn2', 'sol-attn'):
        from flash_attn import flash_attn_func

    def flash(q, k, v):
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

    def cudnn_only(q, k, v):
        return cudnn_sdpa(q, k, v, strict=True)

    kernels = {'flash-attn2': flash, 'cudnn': cudnn_sdpa, 'cudnn-only': cudnn_only}
    if backend == 'sol-attn':
        import torch
        from kernels.sol_loader import load_interface

        sol_interface = load_interface()
        get_sol_attn_backend = sol_interface.get_sol_attn_backend
        sol_attn = sol_interface.sol_attn

        if get_sol_attn_backend(0) != 'triton':
            raise RuntimeError('This Sol-Attn option requires the Triton backend')

        def sparse(q, k, v):
            if q.shape[-1] != 128 or q.dtype != torch.bfloat16 or q.device.type != 'cuda':
                raise ValueError(
                    'Sol-Attn video self-attention requires CUDA BF16 BTHD tensors with head dimension 128.'
                )
            marker = torch.profiler.record_function('ddo::sol_attn_triton') if trace_operators else nullcontext()
            with marker:
                return sol_attn(*(t.contiguous() for t in (q, k, v)), tau=1.0, thresh_type='diag')

        kernels['sol-attn'] = sparse

    execution_ops = None
    if fused_rotary:
        from ddo_vmm_allocator.triton_execution import TritonExecutionOps

        execution_ops = TritonExecutionOps()

    routes = []
    try:
        with ExitStack() as stack:
            for index, block in enumerate(transformer.transformer_blocks):
                processor = type(block.attn1.processor)
                if processor.__name__ not in ('LTX2AudioVideoAttnProcessor', 'LTX2PerturbedAttnProcessor'):
                    raise TypeError(f'Unsupported video attention processor: {processor.__name__}')
                selected = 'flash-attn2' if backend == 'sol-attn' and index == 0 else backend
                kernel = kernels.get(selected)
                rotary_pair = (
                    execution_ops.rotary_pair
                    if execution_ops is not None and (backend != 'sol-attn' or selected == 'sol-attn')
                    else None
                )
                load = None if selected == 'default' else lambda kernel=kernel: kernel
                (route,) = stack.enter_context(
                    install(transformer, (index,), processor, load, rotary_pair=rotary_pair)
                )
                routes.append((index, selected, route))
            yield
    finally:
        # Record dispatch counts also when denoising stops with an exception
        # (failure or a deliberate early stop), not only on normal completion.
        metrics['video_self_attention'] = {
            'requested_backend': backend,
            'layers': [{'index': index, 'backend': selected, 'calls': route.selected,
                        'fallback_calls': route.fallback} for index, selected, route in routes],
        }
        if backend == 'sol-attn':
            from pathlib import Path

            metrics['video_self_attention']['sol_policy'] = {
                'backend': 'triton', 'tau': 1.0, 'thresh_type': 'diag',
                'interface': str(Path(sol_interface.__file__).resolve()),
            }
        if execution_ops is not None:
            metrics['video_self_attention']['fused_rotary_pair_submissions'] = execution_ops.rotary_pair_submissions
