"""Phase-staged Qwen-Image 2.1 BF16 linear/compatibility runner with shared reporting."""

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    from scripts.helpers.runner_reporting import add_reporting_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddo-preset", default="auto")
    parser.add_argument("--resident-module-budget-gb", type=float)
    parser.add_argument("--pin-weight-budget-gb", type=float)
    add_reporting_arguments(parser)
    parser.add_argument("--variant", choices=("base", "turbo"), default="base")
    parser.add_argument("--model-path")
    parser.add_argument("--prompt", default="A capybara wearing a wizard hat, reading by candlelight, oil painting")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25, help="Base only; Turbo uses saved checkpoint sigmas.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    from scripts.helpers.qwen_custom_blocks import add_custom_blocks_arguments

    add_custom_blocks_arguments(parser)
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument(
        "--vae-resident", action="store_true", help="Override the preset VAE policy with resident decode."
    )
    parser.add_argument(
        "--vae-linear-offload", action="store_true", help="Experimental DDO eager convolution offload for the VAE."
    )
    parser.add_argument(
        "--vae-leaf-offload",
        action="store_true",
        help="Explicit leaf override; auto/one_shot_fast/low_ram_safe already select leaf without stream.",
    )
    parser.add_argument(
        "--vae-no-temporal-cache", action="store_true", help="Opt-in local untiled single-frame decode."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qwen_image_21_linear")
    parser.add_argument(
        "--vae-alias-residuals",
        action="store_true",
        help="Opt-in local residual input reuse; requires --vae-no-temporal-cache.",
    )
    parser.add_argument(
        "--vae-direct-padding",
        action="store_true",
        help="Opt-in local convolution padding; requires --vae-no-temporal-cache.",
    )
    args = parser.parse_args(argv)
    if sum((args.vae_linear_offload, args.vae_leaf_offload, args.vae_resident)) > 1:
        parser.error("VAE linear, leaf and resident overrides are mutually exclusive")
    if args.vae_linear_offload and args.ddo_preset not in (
        "auto",
        "one_shot_fast",
        "low_ram_safe",
        "wsl_compat",
        "warm_process",
    ):
        parser.error("--vae-linear-offload requires a linear transformer preset")
    if args.vae_direct_padding and not args.vae_no_temporal_cache:
        parser.error("--vae-direct-padding requires --vae-no-temporal-cache")
    if args.vae_alias_residuals and not args.vae_no_temporal_cache:
        parser.error("--vae-alias-residuals requires --vae-no-temporal-cache")
    return args


def main():
    args = parse_args()
    import diffusers
    import torch

    from scripts.helpers.qwen_custom_blocks import load_custom_blocks

    custom, custom_blocks_metadata = load_custom_blocks(args)
    QwenImage21DecodeStep = custom.QwenImage21DecodeStep
    QwenImage21DenoiseStep = custom.QwenImage21DenoiseStep
    QwenImage21ImageVAE = custom.QwenImage21ImageVAE
    QwenImage21PrepareStep = custom.QwenImage21PrepareStep
    QwenImage21TextEncoderStep = custom.QwenImage21TextEncoderStep
    QwenImage21TurboPrepareStep = custom.QwenImage21TurboPrepareStep
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from diffusers_dynamic_offloader import (
        enable_offload,
        remove_dynamic_offload,
    )
    from huggingface_hub import snapshot_download
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    from scripts.helpers.inference_utils import flush
    from scripts.helpers.runner_reporting import RunnerReporting

    model = args.model_path or ("Qwen/Qwen-Image-2.1-Turbo" if args.variant == "turbo" else "Qwen/Qwen-Image-2.1")
    loading = {"local_files_only": args.local_files_only}
    device, dtype = "cuda:0", torch.bfloat16
    overrides = {}
    if args.resident_module_budget_gb is not None:
        overrides["resident_module_budget_gb"] = args.resident_module_budget_gb
    if args.pin_weight_budget_gb is not None:
        overrides["max_pin_weight_budget_gb"] = args.pin_weight_budget_gb
    route_description = {"backend": "linear", "requested_preset": args.ddo_preset}
    metrics = {
        "runner": Path(__file__).stem,
        "variant": args.variant,
        "model": model,
        "custom_blocks": custom_blocks_metadata,
        **route_description,
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
        "validation": "Reporting/lifecycle revision: CUDA revalidation pending",
    }

    with RunnerReporting(args, metrics, device) as run:
        with run.phase("metadata"):
            metadata = QwenImage21Pipeline.load_config(model, **loading)
            sample_sigmas = metadata.get("sample_sigmas")
            if args.variant == "turbo" and not sample_sigmas:
                raise ValueError("Turbo checkpoint is missing sample_sigmas")
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
            processor = Qwen3VLProcessor.from_pretrained(model_root / "processor", **loading)
        with run.phase("attach_text_encoder"):
            text_result = enable_offload(
                text_encoder,
                component="text_encoder",
                preset=args.ddo_preset,
                execution_device=device,
                use_environment=False,
                record_event=run.tracker.record_event,
            )
            prompt_pipe = QwenImage21TextEncoderStep().init_pipeline()
            prompt_pipe.update_components(text_encoder=text_result.module, processor=processor)
        with text_result.cleanup_on_error(), run.phase("encode"):
            encoded = prompt_pipe(
                prompt=args.prompt,
                execution_device=device,
                output=["prompt_embeds", "prompt_embeds_mask", "image_pad_mask"],
            )
            # Keep only downstream tensors, on CPU, before releasing encoder.
            encoded = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
            run.report(text_result.metrics_snapshot(), "text_encoder")
        with run.phase("cleanup_text_encoder"):
            remove_dynamic_offload(text_encoder)
            del text_encoder, text_result, processor, prompt_pipe
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
        with run.phase("attach_transformer"):
            result = enable_offload(
                transformer,
                component="transformer",
                preset=args.ddo_preset,
                **overrides,
                execution_device=device,
                use_environment=False,
                record_event=run.tracker.record_event,
            )
        with result.cleanup_on_error():
            with run.phase("prepare"):
                prepare = (
                    QwenImage21TurboPrepareStep() if args.variant == "turbo" else QwenImage21PrepareStep()
                ).init_pipeline()
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
                timer = result.execution_timer(enabled=args.ddo_gpu_timing)

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
            with run.phase("collect_transformer"):
                run.report(
                    result.metrics_snapshot(
                        collect_completed=True,
                        execution_timing=timer if args.ddo_gpu_timing else None,
                    ),
                    "transformer",
                )
        with run.phase("cleanup_transformer"):
            remove_dynamic_offload(transformer)
            # Remove all state/component references, not just the module variable.
            del result, transformer, prepared, denoise, scheduler, timer, callback
            flush()

        with run.phase("load_vae"):
            vae_class = QwenImage21ImageVAE if args.vae_no_temporal_cache else AutoencoderKLQwenImage21
            vae = vae_class.from_pretrained(
                model,
                subfolder="vae",
                torch_dtype=dtype,
                **loading,
            ).eval()
            if args.vae_tiling:
                vae.enable_tiling()
            if args.vae_no_temporal_cache:
                vae.enable_single_frame_decode()
            if args.vae_alias_residuals:
                vae.enable_residual_aliasing()
            if args.vae_direct_padding:
                vae.enable_direct_padding()
            # DDO resolves the component policy; the host only passes explicit overrides.
            vae_route = (
                "linear_convolution"
                if args.vae_linear_offload
                else "diffusers_group_offload"
                if args.vae_leaf_offload
                else "none"
                if args.vae_resident
                else None
            )
            vae_result = enable_offload(
                vae,
                component="vae",
                preset=args.ddo_preset,
                route=vae_route,
                execution_device=device,
                use_environment=False,
                record_event=run.tracker.record_event,
            )
            metrics["executed_controls"] = {
                "vae_offload": {
                    "diffusers_group_offload": "diffusers_leaf",
                    "dynamic_offload": "ddo_linear_convolution",
                    "none": "resident",
                }[vae_result.route],
                "vae_offload_use_stream": False,
                "vae_tiling": bool(vae.use_tiling),
                "vae_no_temporal_cache": args.vae_no_temporal_cache and not vae.use_tiling,
                "vae_class": type(vae).__name__,
                "vae_alias_residuals": args.vae_alias_residuals and not vae.use_tiling,
                "vae_direct_padding": args.vae_direct_padding and not vae.use_tiling,
            }
            decode = QwenImage21DecodeStep().init_pipeline()
            decode.update_components(vae=vae)
        with vae_result.cleanup_on_error() if vae_result is not None else nullcontext(), run.phase("decode"):
            image = decode(latents=latent, height=height, width=width, execution_device=device, output="images")[0]
            # Native postprocessing already materializes a CPU image.
            del latent
        if vae_result is not None:
            with run.phase("collect_vae"):
                run.report(vae_result.metrics_snapshot(), "vae")
        with run.phase("cleanup_vae"):
            if vae_result is not None:
                remove_dynamic_offload(vae)
            del decode, vae, vae_result
            flush()
        with run.phase("save_image"):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            image.save(args.output_dir / "image.png")
            del image


if __name__ == "__main__":
    main()
