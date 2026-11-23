"""Didactic, phase-staged Qwen-Image 2.1 BF16 example using DDO native VMM."""

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.helpers.pytorch_allocator import configure_pytorch_allocator

configure_pytorch_allocator()


from scripts.helpers.runner_logging import configure_runner_logging, quiet_qwen_processor_doc_notices

configure_runner_logging()

from scripts.helpers.runner_controls import parse_common_arguments, purge_for_runner, runner_settings_environ, record_runner_report, minimal_vmm_config
from scripts.helpers.image_benchmark_attention import add_image_attention_arguments, configure_image_attention

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_image_attention_arguments(parser, qwen=True)
    parser.add_argument("--variant", choices=("base", "turbo"), default="base")
    parser.add_argument("--model-path", help="Override the checkpoint, keeping the explicit variant contract.")
    parser.add_argument("--prompt", default="A capybara wearing a wizard hat, reading by candlelight, oil painting")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25, help="Base only; Turbo uses checkpoint sample_sigmas.")
    parser.add_argument("--seed", type=int, default=42)
    from scripts.helpers.runner_controls import add_minimal_vmm_arguments
    add_minimal_vmm_arguments(parser)
    parser.add_argument("--local-files-only", action="store_true")
    from scripts.helpers.qwen_custom_blocks import add_custom_blocks_arguments

    add_custom_blocks_arguments(parser)
    parser.add_argument("--vae-tiling", action="store_true", help="Opt-in native tiling; inspect output for seams.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qwen_image_21_minimal")
    return parse_common_arguments(parser, canonical_image=True)


def main():
    from ddo_vmm_bootstrap import ensure_vmm_fast_process
    if not any(flag in sys.argv[1:] for flag in ("--help", "-h")):
        ensure_vmm_fast_process()
    args = parse_args()
    import torch

    from scripts.helpers.qwen_custom_blocks import load_custom_blocks

    custom, custom_blocks_metadata = load_custom_blocks(args)
    QwenImage21DecodeStep = custom.QwenImage21DecodeStep
    QwenImage21DenoiseStep = custom.QwenImage21DenoiseStep
    QwenImage21PrepareStep = custom.QwenImage21PrepareStep
    QwenImage21TextEncoderStep = custom.QwenImage21TextEncoderStep
    QwenImage21TurboPrepareStep = custom.QwenImage21TurboPrepareStep
    from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
    from diffusers_dynamic_offloader import enable_offload, remove_dynamic_offload, from_pretrained_with_dynamic_offload
    from huggingface_hub import snapshot_download
    with quiet_qwen_processor_doc_notices():
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    from scripts.helpers.inference_utils import RunTracker, flush

    model = args.model_path or ("Qwen/Qwen-Image-2.1-Turbo" if args.variant == "turbo" else "Qwen/Qwen-Image-2.1")
    device, dtype = "cuda:0", torch.bfloat16
    transformer_config = minimal_vmm_config(args, device)
    loading = {"local_files_only": args.local_files_only}
    # Pipeline-level saved sigmas are required for Turbo. Read metadata, not the conventional pipeline instance.
    from diffusers import QwenImage21Pipeline

    checkpoint_config = QwenImage21Pipeline.load_config(model, **loading)
    sample_sigmas = checkpoint_config.get("sample_sigmas")
    if args.variant == "turbo" and not sample_sigmas:
        raise ValueError("Turbo checkpoint is missing sample_sigmas")
    metrics = {
        "runner": Path(__file__).stem,
        "variant": args.variant,
        "model": model,
        "custom_blocks": custom_blocks_metadata,
        "status": "running",
        "validation": "candidate; no real-checkpoint acceptance",
        "events": [],
        "steps": [],
    }
    tracker = RunTracker(device, metrics, memory_monitor=args.memory_monitor)
    purge_for_runner(args, "before_run", preset=args.ddo_preset, record_event=tracker.record_event)
    phase = tracker.step_start("encode")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model,
        subfolder="text_encoder",
        torch_dtype=dtype,
        **loading,
    ).eval()
    # Resolve the component directory explicitly: Transformers tokenizer config
    # fallback can otherwise query the pipeline repository root in offline mode.
    model_root = Path(model)
    if not model_root.is_dir():
        model_root = Path(snapshot_download(model, allow_patterns=["processor/*"], **loading))
    with quiet_qwen_processor_doc_notices():
        processor = Qwen3VLProcessor.from_pretrained(model_root / "processor", **loading)
    text_result = enable_offload(
        text_encoder,
        component="text_encoder",
        preset=args.ddo_preset,
        execution_device=device,
        use_environment=False,
        record_event=tracker.record_event,
    )
    prompt_pipe = QwenImage21TextEncoderStep().init_pipeline()
    prompt_pipe.update_components(text_encoder=text_result.module, processor=processor)
    try:
        with text_result.cleanup_on_error():
            encoded = prompt_pipe(
                prompt=args.prompt,
                execution_device=device,
                output=["prompt_embeds", "prompt_embeds_mask", "image_pad_mask"],
            )
            encoded = {
                name: value.cpu() if isinstance(value, torch.Tensor) else value for name, value in encoded.items()
            }
            record_runner_report(args, tracker, text_result.metrics_snapshot(), "text_encoder")
    finally:
        remove_dynamic_offload(text_encoder)
    del text_encoder, text_result, processor, prompt_pipe
    gc.collect()
    flush()
    tracker.step_end("encode", phase)
    purge_for_runner(args, "after_text_encoder", preset=args.ddo_preset, record_event=tracker.record_event)

    phase = tracker.step_start("transformer")
    purge_for_runner(args, "before_transformer", preset=args.ddo_preset, record_event=tracker.record_event)
    loaded = from_pretrained_with_dynamic_offload(
        model,
        model_loader=QwenImage21Transformer2DModel,
        apply_dynamic=False,
        dynamic_offload_config=transformer_config,
        subfolder="transformer",
        torch_dtype=dtype,
        **loading,
    )
    transformer = loaded.module.eval()
    configure_image_attention(transformer, args, metrics, qwen=True)
    record_runner_report(args, tracker, loaded.metrics_snapshot(), "transformer_load")
    del loaded
    result = enable_offload(
        transformer,
        config=transformer_config,
        route="dynamic_offload",
        component="transformer",
        preset=args.ddo_preset,
        execution_device=device,
        use_environment=False,
        record_event=tracker.record_event,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler", **loading)
    prepare = (QwenImage21TurboPrepareStep() if args.variant == "turbo" else QwenImage21PrepareStep()).init_pipeline()
    prepare.update_components(transformer=result.module, scheduler=scheduler, sample_sigmas=sample_sigmas)
    denoise = QwenImage21DenoiseStep().init_pipeline()
    denoise.update_components(transformer=result.module, scheduler=scheduler)
    try:
        with result.cleanup_on_error():
            prepared = prepare(
                **encoded,
                execution_device=device,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                generator=torch.Generator().manual_seed(args.seed),
            )
            latent = denoise(state=prepared, output="latents").cpu()
            record_runner_report(args, tracker, result.metrics_snapshot(), "transformer")
            height, width = prepared.get("height"), prepared.get("width")
    finally:
        remove_dynamic_offload(transformer)
    del result, transformer, prepare, prepared, denoise, scheduler, encoded
    gc.collect()
    flush()
    tracker.step_end("transformer", phase)

    phase = tracker.step_start("decode")
    vae = AutoencoderKLQwenImage21.from_pretrained(model, subfolder="vae", torch_dtype=dtype, **loading).to(device)
    if args.vae_tiling:
        vae.enable_tiling()
    decode = QwenImage21DecodeStep().init_pipeline()
    decode.update_components(vae=vae)
    image = decode(latents=latent, height=height, width=width, output="images")[0]
    tracker.step_end("decode", phase)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_dir / "image.png")
    purge_for_runner(args, "after_run", preset=args.ddo_preset, record_event=tracker.record_event)
    metrics.update(status="completed", total_elapsed_sec=tracker.total_elapsed())
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
