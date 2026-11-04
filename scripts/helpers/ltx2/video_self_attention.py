from contextlib import contextmanager
from copy import copy
from types import FunctionType


class VideoSelfAttention:
    def __init__(self, original, load):
        self.original = original
        self.load = load
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
        if hasattr(self.original, 'attention_dispatch_override'):
            processor = copy(self.original)
            processor.attention_dispatch_override = dispatch
            return processor(*args)
        invoke = FunctionType(method.__code__, scope, argdefs=method.__defaults__)
        return invoke(self.original, *args)


@contextmanager
def install(model, layers, processor_type, load):
    blocks = model.transformer_blocks
    if len(set(layers)) != len(layers) or any(type(i) is not int or not 0 <= i < len(blocks) for i in layers):
        raise ValueError('Invalid video layer selection')
    targets = [blocks[i].attn1 for i in layers]
    if any(type(target.processor) is not processor_type for target in targets):
        raise TypeError('Unsupported video self-attention processor')
    installed = []
    try:
        for target in targets:
            wrapper = VideoSelfAttention(target.processor, load)
            installed.append((target, wrapper))
            target.set_processor(wrapper)
        yield tuple(wrapper for _, wrapper in installed)
    finally:
        for target, wrapper in reversed(installed):
            target.set_processor(wrapper.original)

def add_video_self_attention_argument(parser):
    parser.add_argument('--video-self-attention', choices=('default', 'flash-attn2', 'sol-attn'),
                        default='default', help='Backend for eligible video self-attention only.')


@contextmanager
def video_self_attention(transformer, backend, metrics):
    """Apply the selected backend only to eligible video self-attention."""
    if backend == 'default':
        yield
        return

    from contextlib import ExitStack

    from flash_attn import flash_attn_func

    def flash(q, k, v):
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

    if backend == 'sol-attn':
        import torch

        from sol_attn.interface import get_sol_attn_backend, sol_attn

        if get_sol_attn_backend(0) != 'triton':
            raise RuntimeError('This Sol-Attn option requires the Triton backend')

        def sparse(q, k, v):
            if q.shape[-1] != 128 or q.dtype != torch.bfloat16 or q.device.type != 'cuda':
                raise ValueError(
                    'Sol-Attn video self-attention requires CUDA BF16 BTHD tensors with head dimension 128.'
                )
            return sol_attn(*(t.contiguous() for t in (q, k, v)), tau=1.0, thresh_type='diag')

    routes = []
    with ExitStack() as stack:
        for index, block in enumerate(transformer.transformer_blocks):
            processor = type(block.attn1.processor)
            if processor.__name__ not in ('LTX2AudioVideoAttnProcessor', 'LTX2PerturbedAttnProcessor'):
                raise TypeError(f'Unsupported video attention processor: {processor.__name__}')
            selected = 'sol-attn' if backend == 'sol-attn' and index else 'flash-attn2'
            kernel = sparse if selected == 'sol-attn' else flash
            (route,) = stack.enter_context(install(transformer, (index,), processor, lambda kernel=kernel: kernel))
            routes.append((selected, route))
        yield
    metrics['video_self_attention'] = {
        'requested_backend': backend,
        'layers': [{'index': index, 'backend': selected, 'calls': route.selected,
                    'fallback_calls': route.fallback} for index, (selected, route) in enumerate(routes)],
    }
