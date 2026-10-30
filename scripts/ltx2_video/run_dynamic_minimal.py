"""Readable, staged LTX 2.3 video example: modular blocks + linear DDO.

Read main() from top to bottom: text, connectors, image/audio conditioning,
prepare, denoise, trim and decode. Model math belongs to the trusted blocks;
only telemetry is shared with other runners. No native or kernel experiments.
"""
import argparse
import gc
from importlib import import_module
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REVISION = "f5c712e93c52283cb101da26d4a4544c808666bf"


from scripts.helpers.runner_controls import parse_common_arguments, purge_for_runner, runner_settings_environ, record_runner_report
from scripts.helpers.ltx2.lora import add_lora_arguments, validate_lora_arguments, load_ltx_lora

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddo-preset", default="auto", choices=(
        "auto", "one_shot_fast", "low_ram_safe", "wsl_compat", "warm_process",
        "diffusers_offload_compat", "diffusers_leaf_offload_compat", "off"))
    parser.add_argument("--distilled-model", default="OzzyGT/LTX-2.3-Distilled-1.1")
    parser.add_argument("--custom-blocks", default="elismasilva/ltx2.3_video_custom_blocks")
    parser.add_argument("--custom-blocks-revision")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt", default="the man plays the guitar and sings on an old western town")
    parser.add_argument("--image", default="https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/man_guitar.png")
    parser.add_argument("--audio")
    parser.add_argument("--use-audio-condition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/ddo_modular_video"))
    parser.add_argument("--show-profile", action="store_true")
    parser.add_argument("--memory-monitor", choices=("rss", "full"), default="rss")
    parser.add_argument("--stream-video-decode", action="store_true",
                        help="Use temporal GPU video streaming and separate audio VAE/vocoder stages.")
    parser.add_argument("--decode-device", choices=("cpu", "cuda:0"), default="cpu",
                        help="Explicit decode device; CPU fallback preserves the existing default.")
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    add_lora_arguments(parser)
    args = parse_common_arguments(parser, argv)
    validate_lora_arguments(parser, args)
    if (args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32 or
            not math.isfinite(args.seconds) or args.seconds <= 0 or
            not math.isfinite(args.frame_rate) or args.frame_rate <= 0 or not 1 <= args.steps <= 8):
        parser.error("Use dimensions divisible by 32, positive finite duration/fps and 1-8 steps.")
    if args.use_audio_condition and not args.audio:
        parser.error("--use-audio-condition requires --audio.")
    if args.stream_video_decode and args.decode_device != "cuda:0":
        parser.error("Streaming video decode requires --decode-device cuda:0.")
    return args


def move_tensors(value, device):
    """Carry only stage outputs across devices, never model weights."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {name: move_tensors(item, device) for name, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def phase_pipeline(block, device, checkpoint=None, local_files_only=False):
    """Specify execution device without moving the entire offloaded transformer."""
    import torch
    from diffusers.modular_pipelines.ltx2.modular_pipeline import LTX2ModularPipeline
    class PhasePipeline(LTX2ModularPipeline):
        @property
        def _execution_device(self):
            return torch.device(device)
    options = {"local_files_only": local_files_only} if checkpoint else {}
    return PhasePipeline(blocks=block, pretrained_model_name_or_path=checkpoint, **options)


def run_stage(pipe, state, device, retain=None):
    """Run a declared modular stage and keep its outputs on CPU between phases.

    kwargs_type carries coordinate metadata used by multimodal blocks. The
    denoise call runs once, so this does not copy latents between denoise steps.
    """
    import torch
    inputs = {}
    for field in pipe.blocks.inputs:
        if field.name in state:
            inputs[field.name] = move_tensors(state[field.name], device)
        if field.kwargs_type and field.kwargs_type in state:
            inputs[field.kwargs_type] = move_tensors(state[field.kwargs_type], device)
    with torch.inference_mode():
        output = pipe(**inputs)
    for field in pipe.blocks.intermediate_outputs:
        if field.name and (retain is None or field.name in retain):
            state[field.name] = move_tensors(output.get(field.name), "cpu")
            if field.kwargs_type:
                state.setdefault(field.kwargs_type, {})[field.name] = state[field.name]


def main():
    # Parsing/help/dry-run do not load code, weights or initialize CUDA.
    args = parse_args()
    frames = round(args.seconds * args.frame_rate) // 8 * 8 + 1
    if args.dry_run:
        print(json.dumps({"runner": Path(__file__).stem, "backend": "linear",
                          "preset": args.ddo_preset, "width": args.width, "height": args.height,
                          "num_frames": frames, "steps": args.steps, "seed": args.seed,
                          "decode_device": args.decode_device, "stream_video_decode": args.stream_video_decode,
                          "phases": ["text_encoder", "connectors", "video_encoder", "audio_encoder",
                                     "prepare", "denoise", "trim", "decode"]}, indent=2))
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".hf_modules_modular_video"))
    import torch
    from diffusers import ModularPipelineBlocks
    try:
        from diffusers.guiders import LTX2Guidance
    except ImportError:
        from diffusers.modular_pipelines.ltx2.guider import LTX2Guidance
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    from diffusers.utils import load_image, encode_video
    from transformers import Gemma3ForConditionalGeneration
    from diffusers_dynamic_offloader import (
        DynamicOffloadSettings, enable_offload, from_pretrained_with_dynamic_offload,
        remove_dynamic_offload, clear_dynamic_offload_pinned_tensor_cache,
        maybe_purge_windows_standby_cache,
    )
    from scripts.helpers.inference_utils import RunTracker

    # 1. Load trusted modular code, separately from model weights.
    source = args.custom_blocks
    if not Path(source).is_dir():
        from huggingface_hub import snapshot_download
        # Resolve the snapshot first to avoid the pinned Diffusers dotted-ID bug.
        revision = args.custom_blocks_revision
        if revision is None and source == "elismasilva/ltx2.3_video_custom_blocks":
            revision = REVISION
        source = snapshot_download(repo_id=source, revision=revision,
                                   allow_patterns=["*.py", "modular_config.json"],
                                   local_files_only=args.local_files_only)
    blocks = ModularPipelineBlocks.from_pretrained(source, trust_remote_code=True,
                                                   local_files_only=args.local_files_only)
    if type(blocks).__name__ != "LTX2VideoDistilledBlocks":
        raise ValueError("Use the reviewed distilled video block set.")
    exports = import_module(type(blocks).__module__)
    stages = blocks.sub_blocks
    transformer_type = next(spec.type_hint for spec in blocks.expected_components if spec.name == "transformer")
    device, dtype = "cuda:0", torch.bfloat16
    settings = DynamicOffloadSettings.from_env(execution_device=device, offload_device="cpu",
                                              default_preset=args.ddo_preset, environ=runner_settings_environ(args))
    metrics = {"runner": Path(__file__).stem, "events": [], "steps": [], "requested_controls": vars(args).copy()}
    tracker = RunTracker(device, metrics, show_metrics=args.show_metrics, memory_monitor=args.memory_monitor)
    active = []
    pipe = None
    output_dir = Path(args.output_dir)
    state = dict(prompt=args.prompt, max_sequence_length=256, width=args.width, height=args.height,
                 num_frames=frames, frame_rate=args.frame_rate, num_inference_steps=args.steps,
                 num_videos_per_prompt=1, sigmas=list(DISTILLED_SIGMA_VALUES[:args.steps]),
                 generator=torch.Generator(device=device).manual_seed(args.seed), dtype=dtype,
                 use_cross_timestep=True, seed=args.seed)
    ffmpeg_bin = os.getenv("DDO_RUNNER_FFMPEG_BIN", "")
    dll_directory = os.add_dll_directory(ffmpeg_bin) if os.name == "nt" and os.path.isdir(ffmpeg_bin) else None

    def load_phase(block, names=(), phase_device=device):
        phase = phase_pipeline(block, phase_device, args.distilled_model, args.local_files_only)
        for name in names:
            phase.load_components(names=name, pretrained_model_name_or_path=args.distilled_model,
                                  subfolder=name, torch_dtype=dtype, local_files_only=args.local_files_only)
            if getattr(phase, name) is None:
                raise RuntimeError(f"Failed to load {name}")
        return phase

    def attach(module, name, group=False):
        result = enable_offload(module, settings=settings, component=name,
                                route="diffusers_group_offload" if group else None,
                                execution_device=device, offload_device="cpu", offload_type="leaf_level",
                                low_cpu_mem_usage=True, use_stream=True, record_stream=False,
                                record_event=tracker.record_event)
        active.append((name, result))
        if result.should_move_to_execution_device:
            module.to(device)
        return result

    def finish_phase(name, start, phase):
        for component, result in active:
            with result.cleanup_on_error():
                report = result.metrics_snapshot(collect_completed=True)
                record_runner_report(args, tracker, report, component)
                if args.show_profile and not args.show_metrics:
                    print(report.format_profile())
                if result.route == "dynamic_offload":
                    remove_dynamic_offload(result.module)
        active.clear()
        clear_dynamic_offload_pinned_tensor_cache()
        phase.unload_components(list(phase.components))
        tracker.step_end(name, start)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        purge_for_runner(args, settings, "before_run", record_event=tracker.record_event)
        # 2. Encode text, then release encoder weights before loading connectors.
        start = tracker.step_start("text_encoder")
        pipe = load_phase(stages["text_encoder"])
        pipe.load_components(names="tokenizer", local_files_only=args.local_files_only)
        encoder = Gemma3ForConditionalGeneration.from_pretrained(
            args.distilled_model, subfolder="text_encoder", torch_dtype=dtype,
            low_cpu_mem_usage=True, local_files_only=args.local_files_only)
        pipe.update_components(text_encoder=encoder)
        attach(encoder, "text_encoder", group=True)
        del encoder
        run_stage(pipe, state, device)
        finish_phase("text_encoder", start, pipe)
        pipe = None
        purge_for_runner(args, settings, "after_text_encoder", record_event=tracker.record_event)

        # 3. Connect text embeddings to the video/audio transformer inputs.
        start = tracker.step_start("connectors")
        pipe = load_phase(stages["connectors"], ["connectors"])
        pipe.load_components(names="tokenizer", local_files_only=args.local_files_only)
        pipe.tokenizer.padding_side = "left"
        attach(pipe.connectors, "connectors", group=True)
        run_stage(pipe, state, device)
        for name in ("prompt_embeds", "negative_prompt_embeds", "prompt_attention_mask", "negative_prompt_attention_mask"):
            state.pop(name, None)
        finish_phase("connectors", start, pipe)
        pipe = None

        # 4. Encode the source image and optional conditioning audio separately.
        start = tracker.step_start("video_encoder")
        pipe = load_phase(stages["video_encoder"], ["vae"])
        pipe.vae.enable_tiling()
        pipe.vae.use_framewise_encoding = True
        attach(pipe.vae, "vae", group=True)
        state["image_conditions"] = [exports.LTX2ImageCondition(load_image(args.image))] if args.image else []
        run_stage(pipe, state, device)
        state.pop("image_conditions", None)
        finish_phase("video_encoder", start, pipe)
        pipe = None

        start = tracker.step_start("audio_encoder")
        pipe = load_phase(stages["audio_encoder"], ["audio_vae"])
        attach(pipe.audio_vae, "audio_vae", group=True)
        state["audio_conditions"] = []
        if args.use_audio_condition:
            import torchaudio
            waveform, sample_rate = torchaudio.load(args.audio)
            target_rate = pipe.audio_vae.config.sample_rate
            if sample_rate != target_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
            state["audio_conditions"] = [exports.LTX2AudioCondition(
                waveform[:, :round(args.seconds * target_rate)], end_time=args.seconds)]
            del waveform
        run_stage(pipe, state, device)
        state.pop("audio_conditions", None)
        finish_phase("audio_encoder", start, pipe)
        pipe = None

        # 5. Load the CPU transformer; prepare uses configuration, not its weights.
        start = tracker.step_start("load_and_prepare_transformer")
        loaded = from_pretrained_with_dynamic_offload(
            args.distilled_model, model_loader=transformer_type, subfolder="transformer",
            torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=args.local_files_only,
            apply_dynamic=False, dynamic_offload_config=settings.config if settings.enabled else None)
        transformer = loaded.module
        load_ltx_lora(transformer, args, metrics, tracker.record_event)
        record_runner_report(args, tracker, loaded.metrics_snapshot(), "transformer_load")
        del loaded
        pipe = load_phase(stages["prepare"], ["vae", "audio_vae", "scheduler"])
        pipe.update_components(transformer=transformer)
        run_stage(pipe, state, device)
        scheduler = pipe.scheduler
        rotary_pipe = phase_pipeline(exports.LTX2VideoPrepareRotaryStep(), device)
        rotary_pipe.update_components(transformer=transformer)
        run_stage(rotary_pipe, state, device, retain={"rotary_embeddings"})
        rotary_pipe.unload_components(["transformer"])
        del rotary_pipe
        pipe.unload_components(["transformer", "vae", "audio_vae", "scheduler"])
        pipe = None
        gc.collect()
        tracker.step_end("load_and_prepare_transformer", start)

        # 6. DDO owns offload; the modular denoise block owns the sampling loop.
        start = tracker.step_start("attach_transformer")
        pipe = load_phase(stages["denoise"])
        pipe.update_components(transformer=transformer, scheduler=scheduler,
                               guider=LTX2Guidance(guidance_scale=1.0, guidance_rescale=0.7),
                               audio_guider=exports.LTX2VideoAudioGuidance(guidance_scale=1.0))
        result = attach(transformer, "transformer")
        tracker.step_end("attach_transformer", start)
        purge_for_runner(args, settings, "before_transformer", record_event=tracker.record_event)
        start = tracker.step_start("denoise")
        with result.cleanup_on_error():
            run_stage(pipe, state, device, retain={"latents", "audio_latents"})
            torch.cuda.synchronize(device)
            finish_phase("denoise", start, pipe)
        del result, transformer, scheduler
        pipe = None
        keep = {"latents", "audio_latents", "base_token_count", "audio_ref_num_tokens", "audio_num_frames",
                "height", "width", "num_frames", "dtype", "batch_size", "generator", "seed"}
        state = {name: value for name, value in state.items() if name in keep}
        gc.collect()
        torch.cuda.empty_cache()
        run_stage(load_phase(stages["trim"], phase_device="cpu"), state, "cpu")

        # 7. Match the VMM decode contract; transformer offload is independent.
        start = tracker.step_start("decode_and_save")
        decoded = {}
        decoder_stages = [("video", stages["decode"].sub_blocks["video"], ["vae"]),
                          ("audio", stages["decode"].sub_blocks["audio"], ["audio_vae", "vocoder"])]
        if args.stream_video_decode:
            decoder_stages = [("video", exports.LTX2VideoStreamingDecoderStep(), ["vae"]),
                              ("audio_mel", exports.LTX2VideoAudioMelDecoderStep(), ["audio_vae"]),
                              ("audio", exports.LTX2VideoVocoderStep(), ["vocoder"])]
        metrics["decode_route"] = {"device": args.decode_device,
                                   "stream_video_decode": args.stream_video_decode,
                                   "planned_stages": [name for name, _, _ in decoder_stages],
                                   "completed_stages": []}
        for name, decoder_block, component_names in decoder_stages:
            pipe = load_phase(decoder_block, component_names, phase_device=args.decode_device)
            if name == "video":
                pipe.vae.enable_tiling()
                pipe.vae.use_framewise_decoding = True
            elif name == "audio":
                audio_rate = pipe.vocoder.config.output_sampling_rate
            pipe.to(args.decode_device)
            state["output_type"] = "pil"
            input_device = "cpu" if args.stream_video_decode and name == "video" else args.decode_device
            run_stage(pipe, state, input_device)
            if name != "audio_mel":
                decoded[name] = state.pop("videos" if name == "video" else "audio")
            if args.stream_video_decode and name == "audio":
                state.pop("generated_mel_spectrograms", None)
            pipe.unload_components(component_names)
            metrics["decode_route"]["completed_stages"].append(name)
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()
        video, audio = decoded["video"][0], decoded["audio"][0].float().cpu()
        if len(video) != frames or any(frame.size != (args.width, args.height) for frame in video):
            raise RuntimeError("Decoded frame count/resolution mismatch")
        if audio.numel() == 0 or not bool(torch.isfinite(audio).all()):
            raise RuntimeError("Decoded audio is empty or non-finite")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ltx23_modular_{args.width}x{args.height}_{args.seconds}s_seed{args.seed}.mp4"
        encode_video(video, fps=args.frame_rate, output_path=str(output_path), audio=audio, audio_sample_rate=audio_rate)
        tracker.step_end("decode_and_save", start)
        metrics.update(status="completed", output_path=str(output_path))
        purge_for_runner(args, settings, "after_run", record_event=tracker.record_event)
        print(f"Saved: {output_path}")
    except BaseException as error:
        metrics.update(status="failed", error=repr(error))
        for _, result in active:
            with result.cleanup_on_error():
                raise error
        raise
    finally:
        tracker.monitor.stop()
        metrics.update(total_elapsed_sec=round(tracker.total_elapsed(), 4),
                       global_peak_vram_gb=tracker.global_peak_vram, global_peak_ram_gb=tracker.global_peak_ram)
        if args.save_metrics:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metrics").mkdir(exist_ok=True)
            (output_dir / "metrics/modular_run.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if dll_directory is not None:
            dll_directory.close()


if __name__ == "__main__":
    main()
