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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


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
    from isolation import Isolation
    isolation = Isolation(args.output_dir)
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

        isolation.snapshot("after_text_encoder_cleanup")
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

            import json
            import os
            import threading
            from observer import Native, Observer, ObservationStop

            if not hasattr(transformer, "transformer_blocks") or len(transformer.transformer_blocks) < 2:
                raise RuntimeError("Expected Qwen block sequence unavailable")
            def execution_context():
                return (threading.get_ident(), torch.cuda.current_device(),
                        int(torch.cuda.current_stream().cuda_stream))
            observer = Observer(Native(os.environ["DDO_VMM_LIBRARY_PATH"]), execution_context,
                                deadline=int(os.environ["SZ_OBSERVATION_STARTED_NS"])/1e9+1800)
            from cycle import Cycle
            observer.cycle = Cycle(isolation, observer.deadline)
            isolation.snapshot("before_observed_blocks")
            handles = []
            def enter_block(index):
                def enter(module, inputs):
                    observer.check()
                    if index in observer.blocks:
                        raise RuntimeError("Repeated observed block")
                    observer.block = index
                    observer.blocks.append(index)
                return enter
            def first_output(module, inputs, output):
                observer.select(output)
                isolation.snapshot("after_block_0")
            def second_output(module, inputs, output):
                if observer.blocks != [0, 1] or observer.target is None:
                    raise RuntimeError("Expected two-block observation incomplete")
                if not observer.cycle.closed:
                    raise RuntimeError("Actual cycle did not close before block 1")
                isolation.snapshot("after_block_1")
                raise ObservationStop("Two declared Qwen blocks observed; no further model work")
            with run.phase("owner_access_two_blocks"):
                try:
                    handles += [transformer.transformer_blocks[index].register_forward_pre_hook(enter_block(index))
                                for index in (0,1)]
                    handles += [transformer.transformer_blocks[0].register_forward_hook(first_output),
                                transformer.transformer_blocks[1].register_forward_hook(second_output)]
                    try:
                        with observer:
                            denoise(state=prepared, output="latents")
                        raise RuntimeError("Expected observation stop missing")
                    except ObservationStop as stop:
                        if getattr(stop, "__notes__", None):
                            raise RuntimeError("Observation stop acquired failure notes") from stop
                except BaseException as error:
                    observer.failure = repr(error)
                    raise
                finally:
                    for handle in handles:
                        handle.remove()
                    observer.save(args.output_dir)
            with run.phase("collect_transformer"):
                run.report(result.metrics_snapshot(collect_completed=True), "transformer")
        with run.phase("cleanup_transformer"):
            remove_dynamic_offload(transformer)
            del result, transformer, prepared, denoise, scheduler, first_output, second_output, handles, handle
            flush()
        isolation.snapshot("after_transformer_cleanup")
        summary = observer.summary()
        summary["codec"] = observer.cycle.summary()
        summary.update(expected_stop=True, cleanup_returned=True, decode_executed=False,
                       completed_steps=0, close_scope="DDO removal returned; final driver release unavailable")
        raw = json.dumps(summary, indent=2).encode()
        if len(raw) > 262144:
            raise RuntimeError("Rendered observation ceiling")
        (args.output_dir/"observation.json").write_bytes(raw)
        print(json.dumps(dict(event="owner_access", status=summary["status"],
                              queries=summary["queries"], accesses=summary["accesses"])), flush=True)


if __name__ == "__main__":
    main()
