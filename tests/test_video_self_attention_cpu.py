"""CPU contracts for the LTX video self-attention backend option."""
import argparse
import sys
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace
from contextlib import nullcontext

import torch

from scripts.helpers.ltx2 import video_self_attention as vsa


def dispatch_attention_fn(*args, **kwargs):
    raise AssertionError('The CPU Sol fixture must use the selected dispatch override')


class VideoSelfAttentionCpu(unittest.TestCase):
    def test_sol_mixed_routes_and_restoration_without_cuda(self):
        class LTX2AudioVideoAttnProcessor:
            _parallel_config = None
            attention_dispatch_override = None

            def __call__(self, attn, hidden_states, *args):
                return self.attention_dispatch_override(hidden_states, hidden_states, hidden_states)

        class Attention:
            def __init__(self):
                self.processor = LTX2AudioVideoAttnProcessor()

            def set_processor(self, value):
                self.processor = value

        tensor = SimpleNamespace(ndim=4, shape=(1, 33, 32, 128), dtype=torch.bfloat16,
                                 device=SimpleNamespace(type='cuda'))
        tensor.contiguous = lambda: tensor
        sparse_calls, flash_calls = [], []
        flash = ModuleType('flash_attn')
        flash.flash_attn_func = lambda q, k, v, **kw: flash_calls.append(kw) or q
        package = ModuleType('sol_attn')
        interface = ModuleType('sol_attn.interface')
        interface.__file__ = __file__
        interface.get_sol_attn_backend = lambda device: 'triton'
        interface.sol_attn = lambda q, k, v, **kw: sparse_calls.append(kw) or q
        package.interface = interface
        model = SimpleNamespace(transformer_blocks=[SimpleNamespace(attn1=Attention()) for _ in range(48)])
        originals = [block.attn1.processor for block in model.transformer_blocks]
        metrics = {}
        with patch.dict(sys.modules, {'flash_attn': flash, 'sol_attn': package,
                                      'sol_attn.interface': interface}), \
                patch('kernels.sol_loader.load_interface', return_value=interface), \
                patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA prohibited')), \
                patch.object(torch.profiler, 'record_function', side_effect=lambda name: nullcontext()) as marker:
            with self.assertRaisesRegex(RuntimeError, 'planned stop'):
                with vsa._configure_video_self_attention(model, 'sol-attn', metrics, trace_operators=True):
                    for block in model.transformer_blocks:
                        for _ in range(8):
                            self.assertIs(block.attn1.processor(block.attn1, tensor), tensor)
                    raise RuntimeError('planned stop')
        self.assertEqual(len(flash_calls), 8)
        self.assertEqual(len(sparse_calls), 376)
        self.assertEqual(marker.call_count, 376)
        self.assertTrue(all(call.args == ('ddo::sol_attn_triton',) for call in marker.call_args_list))
        self.assertTrue(all(v == {'tau': 1.0, 'thresh_type': 'diag'} for v in sparse_calls))
        self.assertTrue(all(block.attn1.processor is original
                            for block, original in zip(model.transformer_blocks, originals)))
        self.assertEqual(metrics['video_self_attention']['layers'][0]['backend'], 'flash-attn2')
        self.assertEqual(metrics['video_self_attention']['sol_policy']['backend'], 'triton')

    def test_choices_include_cudnn(self):
        parser = argparse.ArgumentParser()
        vsa.add_video_self_attention_argument(parser)
        self.assertEqual(parser.parse_args([]).video_self_attention, 'default')
        self.assertEqual(parser.parse_args(['--video-self-attention', 'cudnn']).video_self_attention, 'cudnn')
        with self.assertRaises(SystemExit):
            parser.parse_args(['--video-self-attention', 'cudnn-all'])

    def test_cudnn_sdpa_matches_reference_layout(self):
        # Inputs arrive as (batch, tokens, heads, dim); the result keeps that layout.
        generator = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(2, 5, 3, 8, generator=generator) for _ in range(3))
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        result = vsa.cudnn_sdpa(q, k, v)
        self.assertEqual(result.shape, q.shape)
        torch.testing.assert_close(result, expected)

    def test_cudnn_sdpa_sets_priority(self):
        entered = []
        original = vsa.cudnn_sdpa.__globals__.get('sdpa_kernel')
        self.assertIsNone(original)  # Imported lazily inside the function.
        from torch.nn import attention

        real = attention.sdpa_kernel

        def recording(backends, set_priority=False):
            entered.append((list(backends), set_priority))
            return real(backends, set_priority=set_priority)

        q = torch.zeros(1, 2, 1, 4)
        with patch.object(attention, 'sdpa_kernel', recording):
            vsa.cudnn_sdpa(q, q, q)
        S = attention.SDPBackend
        self.assertEqual(entered, [([S.FLASH_ATTENTION, S.CUDNN_ATTENTION, S.EFFICIENT_ATTENTION, S.MATH], True)])

    def test_cudnn_does_not_import_flash_attn(self):
        transformer = type('T', (), {'transformer_blocks': []})()
        metrics = {}
        # flash_attn import would fail; the cudnn route must not need it.
        with patch.dict(sys.modules, {'flash_attn': None}):
            with vsa._configure_video_self_attention(transformer, 'cudnn', metrics):
                pass
        self.assertEqual(metrics['video_self_attention'], {'requested_backend': 'cudnn', 'layers': []})
        with patch.dict(sys.modules, {'flash_attn': None}), self.assertRaises(ImportError):
            with vsa._configure_video_self_attention(transformer, 'flash-attn2', metrics):
                pass


    def test_metrics_recorded_when_denoise_raises(self):
        LTX2AudioVideoAttnProcessor = type('LTX2AudioVideoAttnProcessor', (), {})

        class Attention:
            def __init__(self):
                self.processor = LTX2AudioVideoAttnProcessor()

            def set_processor(self, processor):
                self.processor = processor

        block = type('Block', (), {})()
        block.attn1 = Attention()
        original = block.attn1.processor
        transformer = type('T', (), {})()
        transformer.transformer_blocks = [block]
        metrics = {}

        class PlannedStop(RuntimeError):
            pass

        with self.assertRaises(PlannedStop):
            with vsa._configure_video_self_attention(transformer, 'cudnn', metrics):
                self.assertIsInstance(block.attn1.processor, vsa.VideoSelfAttention)
                raise PlannedStop('stop after a step')
        self.assertIs(block.attn1.processor, original)
        self.assertEqual(metrics['video_self_attention'], {
            'requested_backend': 'cudnn',
            'layers': [{'index': 0, 'backend': 'cudnn', 'calls': 0, 'fallback_calls': 0}],
        })


if __name__ == '__main__':
    unittest.main()
