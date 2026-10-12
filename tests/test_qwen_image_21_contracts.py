"""CPU-only Qwen 2.1 contracts using real native models and schedulers.

These checks do not establish checkpoint/encoder parity or CUDA acceptance.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modular_diffusers"))

from custom_blocks.qwen_image_21 import QwenImage21DenoiseStep, QwenImage21PrepareStep, QwenImage21TurboPrepareStep
from diffusers import ClassifierFreeGuidance, FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor, QwenImage21KVCache


@pytest.fixture
def components():
    torch.manual_seed(0)
    transformer = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=8,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    ).eval()
    transformer.set_attn_processor(QwenImage21AttnProcessor())
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    return transformer, scheduler


def prepare(components, *, turbo=False, sample_sigmas=None, copies=1, execution_device=None):
    transformer, scheduler = components
    block = QwenImage21TurboPrepareStep() if turbo else QwenImage21PrepareStep()
    pipe = block.init_pipeline()
    pipe.update_components(transformer=transformer, scheduler=scheduler, sample_sigmas=sample_sigmas)
    torch.manual_seed(1)
    return pipe(
        execution_device=execution_device,
        prompt_embeds=torch.randn(1, 4, 8),
        prompt_embeds_mask=torch.ones(1, 4, dtype=torch.long),
        image_pad_mask=torch.zeros(1, 4, dtype=torch.bool),
        height=32,
        width=32,
        num_images_per_prompt=copies,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(42),
        output=[
            "latents",
            "timesteps",
            "img_shapes",
            "prompt_embeds",
            "prompt_embeds_mask",
            "image_pad_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "negative_image_pad_mask",
        ],
    )


class TestQwenImage21Contracts:
    def test_prepare_compute_device_is_independent_of_weight_storage(self, components):
        transformer, scheduler = components
        transformer.to("meta")
        state = prepare((transformer, scheduler), execution_device="cpu")
        for name in ("latents", "prompt_embeds", "image_pad_mask", "timesteps"):
            assert state[name].device == torch.device("cpu")
        assert transformer.device == torch.device("meta")

    def test_native_noise_layout_and_expansion(self, components):
        state = prepare(components, copies=2)
        expected = torch.randn(2, 4, 2, 2, generator=torch.Generator().manual_seed(42)).flatten(2).transpose(1, 2)
        torch.testing.assert_close(state["latents"], expected, atol=0, rtol=0)
        assert state["image_pad_mask"].shape == (2, 5)
        assert state["image_pad_mask"][:, -1].all()
        assert state["prompt_embeds_mask"] is None
        torch.testing.assert_close(state["prompt_embeds"][0], state["prompt_embeds"][1])

    def test_turbo_requires_saved_schedule(self, components):
        with pytest.raises(ValueError, match="Turbo requires"):
            prepare(components, turbo=True)

    def test_turbo_preserves_saved_sigmas(self, components):
        state = prepare(components, turbo=True, sample_sigmas=[1.0, 0.7, 0.3])
        reference = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        slope = (1.15 - 0.5) / (4096 - 256)
        reference.set_timesteps(sigmas=[1.0, 0.7, 0.3], device="cpu", mu=4 * slope + 0.5 - slope * 256)
        torch.testing.assert_close(state["timesteps"], reference.timesteps, atol=0, rtol=0)
        assert len(state["timesteps"]) == 3

    @pytest.mark.parametrize("cache_enabled", [False, True])
    @pytest.mark.parametrize("cfg_scale", [1.0, 2.0])
    def test_loop_matches_native_reference_and_cache_is_call_local(self, components, cache_enabled, cfg_scale):
        transformer, scheduler = components
        state = prepare(components)
        if cfg_scale > 1:
            state["negative_prompt_embeds"] = state["prompt_embeds"] * 0.5
            state["negative_image_pad_mask"] = state["image_pad_mask"].clone()
        reference_scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
        reference_scheduler.set_timesteps(
            sigmas=[1.0, 0.5],
            device="cpu",
            mu=4 * ((1.15 - 0.5) / (4096 - 256)) + 0.5 - ((1.15 - 0.5) / (4096 - 256)) * 256,
        )
        reference_scheduler.set_begin_index(0)
        expected = state["latents"].clone()
        cache = QwenImage21KVCache(1) if cache_enabled else None
        negative_cache = QwenImage21KVCache(1) if cache_enabled and cfg_scale > 1 else None
        with torch.inference_mode():
            for i, t in enumerate(state["timesteps"]):

                def predict(embeds, mask, kv, sample=expected, timestep=t, step=i):
                    return transformer(
                        hidden_states=sample,
                        timestep=timestep.expand(sample.shape[0]).to(sample.dtype) / 1000,
                        encoder_hidden_states=embeds,
                        encoder_hidden_states_mask=None,
                        img_shapes=state["img_shapes"],
                        img_mask=mask,
                        kv_cache=kv,
                        kv_cache_mode=("extract" if step == 0 else "cached") if cache_enabled else None,
                        return_dict=False,
                    )[0][:, -sample.shape[1] :]

                pred = predict(state["prompt_embeds"], state["image_pad_mask"], cache)
                if cfg_scale > 1:
                    neg = predict(state["negative_prompt_embeds"], state["negative_image_pad_mask"], negative_cache)
                    pred = neg + cfg_scale * (pred - neg)
                expected = reference_scheduler.step(pred, t, expected, return_dict=False)[0]
        pipe = QwenImage21DenoiseStep().init_pipeline()
        pipe.update_components(
            transformer=transformer, scheduler=scheduler, guider=ClassifierFreeGuidance(guidance_scale=cfg_scale)
        )
        for _ in range(2):
            # A new generation resets the scheduler through the native public
            # preparation path. set_begin_index alone does not reset step_index.
            state = prepare(components)
            if cfg_scale > 1:
                state["negative_prompt_embeds"] = state["prompt_embeds"] * 0.5
                state["negative_image_pad_mask"] = state["image_pad_mask"].clone()
            callbacks = []
            result = pipe(
                **state,
                use_kv_cache=cache_enabled,
                callback_on_step_end=lambda p, i, t, data, callbacks=callbacks: callbacks.append(i),
                output="latents",
            )
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
            assert callbacks == [0, 1]

    def test_native_decoder_matches_direct_decode(self):
        from custom_blocks.qwen_image_21 import QwenImage21DecodeStep
        from diffusers import AutoencoderKLQwenImage21

        torch.manual_seed(5)
        vae = AutoencoderKLQwenImage21(
            base_dim=16,
            decoder_base_dim=16,
            z_dim=64,
            dim_mult=[1, 1, 1, 1, 1],
            num_res_blocks=1,
        ).eval()
        latent = torch.randn(1, 4, 64)
        pipe = QwenImage21DecodeStep().init_pipeline()
        pipe.update_components(vae=vae)
        with torch.inference_mode():
            unpacked = latent.transpose(1, 2).reshape(1, 64, 1, 2, 2)
            mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
            reference = vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
            expected = (reference / 2 + 0.5).clamp(0, 1)
            result = pipe(latents=latent, height=32, width=32, output_type="pt", output="images")
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        assert result.shape[-2:] == (32, 32)
