"""Phase-staged Qwen-Image 2.1 BF16 native VMM runner with shared reporting."""

import argparse
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

# Allocator bootstrap remains before Torch, DDO or CUDA-owning imports.
from ddo_vmm_bootstrap import ensure_vmm_fast_process, vmm_fast_requested

if not any(flag in sys.argv[1:] for flag in ("--help", "-h")) and vmm_fast_requested(default_backend="vmm"):
    ensure_vmm_fast_process()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from scripts.helpers.runner_controls import parse_common_arguments, purge_for_runner, runner_settings_environ, record_runner_report

def parse_args(argv=None):
    from diffusers_dynamic_offloader import add_dynamic_offload_arguments

    from scripts.helpers.runner_reporting import add_reporting_arguments

    parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
    add_dynamic_offload_arguments(parser)
    parser.set_defaults(ddo_backend="vmm")
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
        "--clear-cublas-workspaces-before-vae",
        action="store_true",
        help="Experimental single-GPU workspace cleanup; requires --release-native-cache-before-vae.",
    )
    parser.add_argument(
        "--release-native-cache-before-vae",
        action="store_true",
        help="Explicit idle native pool trim after transformer cleanup; requires a supporting DLL.",
    )
    parser.add_argument(
        "--vae-leaf-offload",
        action="store_true",
        help="Explicit leaf override; native vmm_fast already selects leaf without stream.",
    )
    parser.add_argument(
        "--vae-vmm-offload", action="store_true", help="Experimental native component ownership for a saved BF16 VAE."
    )
    parser.add_argument(
        "--vae-resident", action="store_true", help="Override the preset VAE policy with resident decode."
    )
    parser.add_argument(
        "--vae-model-path", type=Path, help="Standalone BF16 VAE directory; required for --vae-vmm-offload."
    )
    parser.add_argument(
        "--vae-vmm-resident-weight-budget-gb",
        type=float,
        help="Explicit VAE GPU weight ceiling, independent of the transformer.",
    )
    parser.add_argument(
        "--vae-no-temporal-cache", action="store_true", help="Opt-in local untiled single-frame decode."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qwen_image_21_vmm")
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
    args = parse_common_arguments(parser, argv)
    if sum((args.vae_vmm_offload, args.vae_leaf_offload, args.vae_resident)) > 1:
        parser.error("VAE native, leaf and resident overrides are mutually exclusive")
    if args.vae_vmm_offload:
        if args.vae_leaf_offload:
            parser.error("--vae-vmm-offload and --vae-leaf-offload are mutually exclusive")
        if args.ddo_backend != "vmm" or args.ddo_preset != "auto":
            parser.error("--vae-vmm-offload requires the native vmm/auto route")
        if args.vae_model_path is None or not args.vae_model_path.is_dir():
            parser.error("--vae-vmm-offload requires an existing --vae-model-path directory")
        budget = args.vae_vmm_resident_weight_budget_gb
        if budget is None or not math.isfinite(budget) or budget <= 0:
            parser.error("--vae-vmm-offload requires a finite positive --vae-vmm-resident-weight-budget-gb")
    elif args.vae_model_path is not None or args.vae_vmm_resident_weight_budget_gb is not None:
        parser.error("VAE model path and VMM budget require --vae-vmm-offload")
    if args.clear_cublas_workspaces_before_vae and not args.release_native_cache_before_vae:
        parser.error("--clear-cublas-workspaces-before-vae requires --release-native-cache-before-vae")
    if args.vae_direct_padding and not args.vae_no_temporal_cache:
        parser.error("--vae-direct-padding requires --vae-no-temporal-cache")
    if args.vae_alias_residuals and not args.vae_no_temporal_cache:
        parser.error("--vae-alias-residuals requires --vae-no-temporal-cache")
    return args


def main():
    args = parse_args()
    if args.ddo_backend != "vmm":
        raise ValueError("Use run_dynamic_modular.py for linear/compatibility routes.")
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
        from_pretrained_with_dynamic_offload,
        release_native_vmm_idle_cache,
        remove_dynamic_offload,
        resolve_dynamic_offload_backend,
    )
    from huggingface_hub import snapshot_download
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    from scripts.helpers.inference_utils import flush
    from scripts.helpers.runner_reporting import RunnerReporting

    model = args.model_path or ("Qwen/Qwen-Image-2.1-Turbo" if args.variant == "turbo" else "Qwen/Qwen-Image-2.1")
    loading = {"local_files_only": args.local_files_only}
    device, dtype = "cuda:0", torch.bfloat16
    route = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset)
    config = route.build_config_from_args(args)
    route_description = {"backend": route.backend, "preset": route.effective_preset}
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
                preset="diffusers_leaf_offload_compat",
                group_offload=True,
                dynamic_offload=False,
                offload_type="leaf_level",
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
            loaded = from_pretrained_with_dynamic_offload(
                model,
                model_loader=QwenImage21Transformer2DModel,
                subfolder="transformer",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                dynamic_offload_config=config,
                apply_dynamic=False,
                **loading,
            )
            transformer = loaded.module
            run.report(loaded.inspect_vmm_component(component="transformer"), "transformer_preflight")
            del loaded
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler", **loading)
        with run.phase("attach_transformer"):
            result = enable_offload(
                transformer,
                component="transformer",
                config=config,
                route="dynamic_offload",
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
            vae_settings = replace(result.settings, effective_preset=route.effective_preset)
            remove_dynamic_offload(transformer)
            # Remove all state/component references, not just the module variable.
            del result, transformer, prepared, denoise, scheduler, timer, callback
            flush()

        if args.release_native_cache_before_vae:
            with run.phase("release_native_cache_before_vae"):
                release_report = release_native_vmm_idle_cache(
                    enabled=True,
                    phase="before_vae",
                    device=device,
                    clear_cublas_workspaces=args.clear_cublas_workspaces_before_vae,
                )
                run.report(release_report, "device_cache_release")
                if not release_report.setup["device_cache_release"]["trim_executed"]:
                    raise RuntimeError("Selected allocator cannot execute the requested idle trim")

        with run.phase("load_vae"):
            vae_class = QwenImage21ImageVAE if args.vae_no_temporal_cache else AutoencoderKLQwenImage21
            vae_result = None
            if args.vae_vmm_offload:
                # The transformer config is unchanged; VAE capacity is explicit.
                vae_config = replace(
                    config,
                    vmm_component_tensor_ownership=True,
                    vmm_execution_auxiliary_ownership=False,
                    vmm_float8_weight_only=False,
                    vmm_module_pattern=".*",
                    vmm_resident_weight_budget_gb=args.vae_vmm_resident_weight_budget_gb,
                )
                vae_loaded = from_pretrained_with_dynamic_offload(
                    args.vae_model_path,
                    model_loader=vae_class,
                    torch_dtype=dtype,
                    dynamic_offload_config=vae_config,
                    apply_dynamic=False,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                vae = vae_loaded.module.eval()
            else:
                vae = vae_class.from_pretrained(
                    model,
                    subfolder="vae",
                    torch_dtype=dtype,
                    **loading,
                )
            if args.vae_tiling:
                vae.enable_tiling()
            if args.vae_no_temporal_cache:
                vae.enable_single_frame_decode()
            if args.vae_alias_residuals:
                vae.enable_residual_aliasing()
            if args.vae_direct_padding:
                vae.enable_direct_padding()
            # Configure local modules before attaching either offload route.
            if args.vae_vmm_offload:
                run.report(vae_loaded.inspect_vmm_component(component="vae"), "vae_preflight")
                del vae_loaded
            else:
                vae_result = enable_offload(
                    vae,
                    component="vae",
                    settings=vae_settings,
                    route="diffusers_group_offload" if args.vae_leaf_offload else "none" if args.vae_resident else None,
                    execution_device=device,
                    use_environment=False,
                    record_event=run.tracker.record_event,
                )
            metrics["executed_controls"] = {
                "vae_offload": (
                    "native_vmm_component"
                    if args.vae_vmm_offload
                    else "diffusers_leaf"
                    if vae_result.route == "diffusers_group_offload"
                    else "resident"
                ),
                "vae_offload_use_stream": False,
                "vae_tiling": bool(vae.use_tiling),
                "vae_no_temporal_cache": args.vae_no_temporal_cache and not vae.use_tiling,
                "vae_class": type(vae).__name__,
                "vae_alias_residuals": args.vae_alias_residuals and not vae.use_tiling,
                "vae_direct_padding": args.vae_direct_padding and not vae.use_tiling,
            }
            decode = QwenImage21DecodeStep().init_pipeline()
            decode.update_components(vae=vae)
        if args.vae_vmm_offload:
            with run.phase("attach_vae"):
                vae_result = enable_offload(
                    vae,
                    component="vae",
                    config=vae_config,
                    route="dynamic_offload",
                    execution_device=device,
                    use_environment=False,
                    record_event=run.tracker.record_event,
                )
            metrics["executed_controls"].update(
                vae_offload="native_vmm_component",
                vae_model_path=str(args.vae_model_path.resolve()),
                vae_vmm_resident_weight_budget_gb=args.vae_vmm_resident_weight_budget_gb,
            )
        with vae_result.cleanup_on_error() if vae_result is not None else nullcontext():
            with run.phase("decode"):
                image = decode(latents=latent, height=height, width=width, execution_device=device, output="images")[0]
                # Native postprocessing already materializes a CPU image.
                del latent
            if vae_result is not None:
                with run.phase("collect_vae"):
                    run.report(vae_result.metrics_snapshot(collect_completed=True), "vae")
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
