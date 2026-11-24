"""Staged Qwen-Image 2.1 control using official Diffusers leaf offload.

The same modular workflow and phases as the DDO runners: trusted custom blocks
for text encoding, preparation, denoising and decoding, each component loaded,
attached, run and released in its own phase. Official Diffusers leaf-level
group offloading replaces DDO; no DDO offload runtime or allocator is installed.
Shared reporting and optional DDO timing/Windows lifecycle helpers are
instrumentation only.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.helpers.pytorch_allocator import configure_pytorch_allocator

configure_pytorch_allocator()


from scripts.helpers.runner_logging import configure_runner_logging, quiet_qwen_processor_doc_notices

configure_runner_logging()

from scripts.helpers.runner_controls import parse_common_arguments
from scripts.helpers.image_benchmark_attention import add_image_attention_arguments, configure_image_attention


def parse_args(argv=None):
    from scripts.helpers.qwen_custom_blocks import add_custom_blocks_arguments
    from scripts.helpers.runner_reporting import add_reporting_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_reporting_arguments(parser)
    parser.add_argument("--model-path", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--prompt", default="A capybara wearing a wizard hat, reading by candlelight, oil painting")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    add_custom_blocks_arguments(parser)
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--vae-leaf-offload", action="store_true")
    parser.add_argument("--no-stream-offload", action="store_true",
                        help="Use non-streamed Leaf for the text encoder and transformer.")
    add_image_attention_arguments(parser, qwen=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qwen21_base_diffusers_leaf_20steps")
    return parse_common_arguments(parser, argv, canonical_image=True,
                                  purge_phases=("before_run", "before_transformer", "after_run"))


def main():
    args = parse_args()
    import diffusers
    import torch

    from scripts.helpers.qwen_custom_blocks import load_custom_blocks

    custom, custom_blocks_metadata = load_custom_blocks(args)
    QwenImage21DecodeStep = custom.QwenImage21DecodeStep
    QwenImage21DenoiseStep = custom.QwenImage21DenoiseStep
    QwenImage21PrepareStep = custom.QwenImage21PrepareStep
    QwenImage21TextEncoderStep = custom.QwenImage21TextEncoderStep
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from diffusers.hooks.group_offloading import apply_group_offloading
    from diffusers_dynamic_offloader.timing import DynamicOffloadExecutionTimer
    from huggingface_hub import snapshot_download
    with quiet_qwen_processor_doc_notices():
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    from scripts.helpers.inference_utils import flush
    from scripts.helpers.runner_reporting import RunnerReporting

    model = args.model_path
    loading = {"local_files_only": args.local_files_only}
    device, dtype = "cuda:0", torch.bfloat16
    offload = {
        "offload_type": "leaf_level",
        "use_stream": not args.no_stream_offload,
        "record_stream": False,
        "low_cpu_mem_usage": not args.no_stream_offload,
        "non_blocking": False,
    }

    def attach_leaf(module, **options):
        """Official Diffusers leaf-level group offloading on one component."""
        apply_group_offloading(module, onload_device=torch.device(device), offload_device=torch.device("cpu"),
                               **options)

    metrics = {
        "runner": Path(__file__).stem,
        "variant": "base",
        "model": model,
        "custom_blocks": custom_blocks_metadata,
        "backend": "official_diffusers_leaf",
        "ddo_enabled": False,
        "offload": offload,
        "events": [],
        "steps": [],
        "denoise_callbacks": [],
        "workload": {
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "steps_requested": args.steps,
            "seed": args.seed,
            "dtype": "bfloat16",
            "true_cfg_scale": 1.0,
            "use_kv_cache": True,
            "vae_tiling": args.vae_tiling,
        },
        "diffusers_source": diffusers.__file__,
    }

    with RunnerReporting(args, metrics, device) as run:
        with run.phase("metadata"):
            metadata = QwenImage21Pipeline.load_config(model, **loading)
            sample_sigmas = metadata.get("sample_sigmas")
            model_root = Path(model)
            if not model_root.is_dir():
                model_root = Path(snapshot_download(model, allow_patterns=["processor/*"], **loading))
            metrics["resolved_model_root"] = str(model_root)
            metrics["sample_sigmas"] = sample_sigmas

        with run.phase("load_text_encoder"):
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                model,
                subfolder="text_encoder",
                torch_dtype=dtype,
                **loading,
            ).eval()
        with quiet_qwen_processor_doc_notices():
            processor = Qwen3VLProcessor.from_pretrained(model_root / "processor", **loading)
        with run.phase("attach_text_encoder"):
            attach_leaf(text_encoder, **offload)
            prompt_pipe = QwenImage21TextEncoderStep().init_pipeline()
            prompt_pipe.update_components(text_encoder=text_encoder, processor=processor)
        with run.phase("encode"):
            encoded = prompt_pipe(
                prompt=args.prompt,
                execution_device=device,
                output=["prompt_embeds", "prompt_embeds_mask", "image_pad_mask"],
            )
            # Keep only downstream tensors, on CPU, before releasing encoder.
            encoded = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
        with run.phase("cleanup_text_encoder"):
            del text_encoder, processor, prompt_pipe
            flush()

        with run.phase("purge_before_transformer"):
            run.purge_before_transformer()

        with run.phase("load_transformer"):
            transformer = QwenImage21Transformer2DModel.from_pretrained(
                model,
                subfolder="transformer",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **loading,
            ).eval()
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler", **loading)
            configure_image_attention(transformer, args, metrics, qwen=True)
        with run.phase("attach_transformer"):
            attach_leaf(transformer, **offload)
        with run.phase("prepare"):
            prepare = QwenImage21PrepareStep().init_pipeline()
            prepare.update_components(transformer=transformer, scheduler=scheduler, sample_sigmas=sample_sigmas)
            prepared = prepare(
                **encoded,
                execution_device=device,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                generator=torch.Generator().manual_seed(args.seed),
            )
            # Drop CPU conditioning and the preparation pipeline before denoise.
            del encoded, prepare
            height, width = prepared.get("height"), prepared.get("width")
            metrics["executed_steps"] = len(prepared.get("timesteps"))
            metrics["workload"].update(height_actual=height, width_actual=width)
            denoise = QwenImage21DenoiseStep().init_pipeline()
            denoise.update_components(transformer=transformer, scheduler=scheduler)

        with run.phase("denoise"):
            last_callback = time.perf_counter()
            started = last_callback
            timer = DynamicOffloadExecutionTimer(enabled=args.ddo_gpu_timing, device=torch.device(device))

            def callback(pipeline, index, timestep, values, execution_timer=timer):
                nonlocal last_callback
                execution_timer.mark_step()
                now = time.perf_counter()
                elapsed = now - last_callback
                metrics["denoise_callbacks"].append(
                    {
                        "step": index + 1,
                        "callback_interval_sec": elapsed,
                        "scope": "asynchronous_host_callback",
                    }
                )
                last_callback = now
                if args.show_denoise_steps:
                    print(f"Step {index + 1}: {elapsed:.3f} s (host callback)", flush=True)
                return values

            with timer:
                # CPU readback is the completion boundary; no per-step waits.
                latent = denoise(state=prepared, callback_on_step_end=callback, output="latents").cpu()
            metrics["denoise_through_cpu_readback_sec"] = time.perf_counter() - started
            # Resolve the final event only once, when explicitly requested.
            metrics["execution_timing"] = timer.snapshot(wait=args.ddo_gpu_timing)
        with run.phase("cleanup_transformer"):
            # Remove all state/component references, not just the module variable.
            del transformer, prepared, denoise, scheduler, timer, callback
            flush()

        with run.phase("load_vae"):
            vae = AutoencoderKLQwenImage21.from_pretrained(
                model,
                subfolder="vae",
                torch_dtype=dtype,
                **loading,
            ).eval()
            if args.vae_tiling:
                vae.enable_tiling()
        with run.phase("attach_vae"):
            if args.vae_leaf_offload:
                attach_leaf(vae, offload_type="leaf_level", use_stream=False)
            else:
                vae.to(device)
            metrics["vae_decode_route"] = "diffusers_leaf" if args.vae_leaf_offload else "resident"
            metrics["executed_controls"] = {
                "vae_offload": metrics["vae_decode_route"],
                "vae_offload_use_stream": False,
                "vae_tiling": bool(vae.use_tiling),
                "vae_class": type(vae).__name__,
            }
            decode = QwenImage21DecodeStep().init_pipeline()
            decode.update_components(vae=vae)
        with run.phase("decode"):
            image = decode(latents=latent, height=height, width=width, execution_device=device, output="images")[0]
            # Native postprocessing already materializes a CPU image.
            del latent
        with run.phase("cleanup_vae"):
            del decode, vae
            flush()
        with run.phase("save_image"):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            image.save(args.output_dir / "image.png")
            del image


if __name__ == "__main__":
    main()
