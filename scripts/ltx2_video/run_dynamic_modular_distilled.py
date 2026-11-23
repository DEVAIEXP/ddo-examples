"""Staged LTX 2.3 video runner for linear DDO and compatibility presets.

Read main() from top to bottom: text, connectors, image/audio conditioning,
prepare, denoise, trim and decode. Model math belongs to the trusted blocks;
Computation defaults match the recorded BF16 linear recipe. No native or kernel experiments.
"""
import argparse
from contextlib import nullcontext
import gc
from importlib import import_module
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.helpers.pytorch_allocator import configure_pytorch_allocator

PYTORCH_ALLOCATOR_STARTUP_ENVIRONMENT = configure_pytorch_allocator()


from scripts.helpers.runner_logging import configure_runner_logging

configure_runner_logging()

from scripts.helpers.runner_controls import parse_common_arguments, purge_for_runner, runner_settings_environ, record_runner_report
from scripts.helpers.ltx2.lora import add_lora_arguments, validate_lora_arguments, load_ltx_lora
from scripts.helpers.ltx2.timestep_layout import compact_timestep_repeat
from scripts.helpers.ltx2.host_cache import default_host_cache_release, release_host_cache_before_vae

from scripts.helpers.ltx2.video_decoder import add_video_decoder_argument, streaming_video_decoder_step, validate_video_decoder
from scripts.helpers.ltx2.video_self_attention import (
    add_video_self_attention_argument,
    record_attention_adjusted_total,
    video_self_attention,
)


def parse_args(argv=None, *, official_leaf=False):
    parser = argparse.ArgumentParser(description=__doc__)
    if not official_leaf:
        parser.add_argument("--ddo-backend", choices=("linear",), default="linear")
        parser.add_argument("--ddo-preset", default="auto", choices=(
            "auto", "one_shot_fast", "low_ram_safe", "wsl_compat", "warm_process",
            "diffusers_offload_compat", "diffusers_leaf_offload_compat", "off"))
    parser.add_argument("--distilled-model", default="OzzyGT/LTX-2.3-Distilled-1.1")
    parser.add_argument("--model-path", help="BF16 transformer override; other components use --distilled-model.")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--memory-monitor", choices=("full", "rss"), default="rss")
    parser.add_argument("--reuse-rotary-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feed-forward-chunk-size", type=int, default=1024,
                        help="Token chunk size for feed-forward layers; 0 disables chunking.")
    parser.add_argument("--normalization-chunk-size", type=int, default=1024,
                        help="Token chunk size for RMSNorm temporaries; 0 disables chunking.")
    parser.add_argument("--phase-modulation", action=argparse.BooleanOptionalAction, default=True,
                        help="Use phase-local modulation buffers in the trusted transformer.")
    parser.add_argument("--compact-timesteps", action=argparse.BooleanOptionalAction, default=True,
                        help="Compact spatially uniform timesteps; unsupported layouts remain dense.")
    parser.add_argument("--inplace-residuals", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse model-owned residual states during inference.")
    if not official_leaf:
        parser.add_argument("--ddo-profile", help="Named transformer workload profile.")
        parser.add_argument("--build-ddo-profile", action="store_true")
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
    if not official_leaf:
        parser.add_argument("--show-profile", action="store_true")
    parser.add_argument("--stream-video-decode", action="store_true",
                        help="Use temporal GPU video streaming and separate audio VAE/vocoder stages.")
    add_video_decoder_argument(parser)
    parser.add_argument("--decode-device", choices=("cpu", "cuda:0"), default="cpu",
                        help="Explicit decode device; CPU fallback preserves the existing default.")
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    add_lora_arguments(parser)
    add_video_self_attention_argument(parser)
    if official_leaf:
        parser.add_argument("--no-stream-offload", action="store_true",
                            help="Use non-streamed Diffusers Leaf offload for model weights.")
        parser.set_defaults(output_dir=str(ROOT / "outputs/ltx23_diffusers_leaf"),
                            feed_forward_chunk_size=0, normalization_chunk_size=0,
                            reuse_rotary_embeddings=False, phase_modulation=False,
                            compact_timesteps=False, inplace_residuals=False,
                            stream_video_decode=True, decode_device="cuda:0",
                            build_ddo_profile=False)
        args = parser.parse_args(argv)
    else:
        parser.add_argument("--purge-before-vae", action="store_true",
                            help="Purge the Windows standby cache before loading the video VAE.")
        parser.add_argument("--release-pytorch-host-cache-before-vae", action=argparse.BooleanOptionalAction,
                            default=None, help="Release unused PyTorch pinned host cache after transformer cleanup. "
                            "Defaults to on for native Linux DDO linear presets; off for Windows, WSL and compatibility routes.")
        args = parse_common_arguments(parser, argv, canonical_video=True, workload_profiles=True)
        if args.release_pytorch_host_cache_before_vae is None:
            args.release_pytorch_host_cache_before_vae = default_host_cache_release(args.ddo_preset)
    validate_lora_arguments(parser, args)
    if (args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32 or
            not math.isfinite(args.seconds) or args.seconds <= 0 or
            not math.isfinite(args.frame_rate) or args.frame_rate <= 0 or not 1 <= args.steps <= 8):
        parser.error("Use dimensions divisible by 32, positive finite duration/fps and 1-8 steps.")
    if args.use_audio_condition and not args.audio:
        parser.error("--use-audio-condition requires --audio.")
    validate_video_decoder(parser, args)
    if args.stream_video_decode and args.decode_device != "cuda:0":
        parser.error("Streaming video decode requires --decode-device cuda:0.")
    if args.max_sequence_length <= 0:
        parser.error("Use a positive text sequence length.")
    if min(args.feed_forward_chunk_size, args.normalization_chunk_size) < 0:
        parser.error("Chunk sizes must be nonnegative.")
    if args.build_ddo_profile and not args.ddo_profile:
        parser.error("--build-ddo-profile requires --ddo-profile NAME.")
    return args


def print_linear_memory_policy(settings):
    """Distinguish preset policy and configured limits from the resolved plan."""
    config = settings.config
    print(f"  [dynamic-offload] preset: requested={settings.requested_preset}, "
          f"effective={settings.effective_preset}, execution={config.execution_mode}.", flush=True)
    if not settings.enabled or config.execution_mode != "linear_runtime":
        return
    if config.auto_budget_policy != "off":
        cap = config.max_resident_module_budget_gb
        cap_label = f"{cap:.2f} GiB" if cap > 0 else "no explicit cap"
        print(f"  [dynamic-offload] memory policy: automatic ({config.auto_budget_policy}), "
              f"resident ceiling={cap_label}; ceiling is not a target. "
              "Full CPU pinning may select a zero resident-module budget when RAM permits.", flush=True)
    else:
        print(f"  [dynamic-offload] initial configuration (before workload profile): resident-module budget="
              f"{config.resident_module_budget_gb:.2f} GiB, "
              f"CPU pinning requested={config.pin_cpu_memory}. "
              "A saved workload profile may override this value. This budget does not limit total GPU memory.", flush=True)


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


def main(*, official_leaf=False):
    # Parsing/help/dry-run do not load code, weights or initialize CUDA.
    args = parse_args(official_leaf=official_leaf)
    runner_name = "run_diffusers_leaf" if official_leaf else Path(__file__).stem
    frames = round(args.seconds * args.frame_rate) // 8 * 8 + 1
    if args.dry_run:
        print(json.dumps({"runner": runner_name, "backend": "diffusers" if official_leaf else "linear",
                          "preset": "leaf_level" if official_leaf else args.ddo_preset,
                          "width": args.width, "height": args.height,
                          "num_frames": frames, "steps": args.steps, "seed": args.seed,
                          "decode_device": args.decode_device, "stream_video_decode": args.stream_video_decode, "video_decoder": args.video_decoder,
                          "feed_forward_chunk_size": args.feed_forward_chunk_size,
                          "normalization_chunk_size": args.normalization_chunk_size,
                          "phase_modulation": args.phase_modulation, "compact_timesteps": args.compact_timesteps,
                          "inplace_residuals": args.inplace_residuals,
                          "phases": ["text_encoder", "connectors", "video_encoder", "audio_encoder",
                                     "prepare", "denoise", "trim", "decode"]}, indent=2))
        return

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
    from scripts.helpers.inference_utils import RunTracker

    # 1. Load trusted modular code, separately from model weights.
    source = args.custom_blocks
    if not Path(source).is_dir():
        from huggingface_hub import snapshot_download
        # Resolve the snapshot first to avoid the pinned Diffusers dotted-ID bug.
        source = snapshot_download(repo_id=source, revision=args.custom_blocks_revision,
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
    metrics = {"runner": runner_name, "events": [], "steps": [], "requested_controls": vars(args).copy(),
               "pytorch_allocator_startup_environment": PYTORCH_ALLOCATOR_STARTUP_ENVIRONMENT.copy()}
    if official_leaf:
        from diffusers.hooks.group_offloading import apply_group_offloading
        metrics["offload"] = {"backend": "diffusers", "offload_type": "leaf_level",
                              "use_stream": not args.no_stream_offload, "low_cpu_mem_usage": True}
    else:
        from diffusers_dynamic_offloader import (
            DynamicOffloadSettings, enable_offload, from_pretrained_with_dynamic_offload,
            remove_dynamic_offload, clear_dynamic_offload_pinned_tensor_cache,
        )
        settings = DynamicOffloadSettings.from_env(execution_device=device, offload_device="cpu",
                                                  default_preset=args.ddo_preset, environ=runner_settings_environ(args))
        print_linear_memory_policy(settings)
    tracker = RunTracker(device, metrics, show_metrics=args.show_metrics, memory_monitor=args.memory_monitor)
    metrics.setdefault("attention_environment", {})["nvidia_driver_versions"] = metrics.get("nvidia_driver_versions")
    active = []
    pipe = None
    output_dir = Path(args.output_dir)
    state = dict(prompt=args.prompt, max_sequence_length=args.max_sequence_length, width=args.width, height=args.height,
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
        if official_leaf:
            apply_group_offloading(module, onload_device=torch.device(device), offload_device=torch.device("cpu"),
                                   offload_type="leaf_level", low_cpu_mem_usage=True,
                                   use_stream=metrics["offload"]["use_stream"], record_stream=False)
            return None
        result = enable_offload(module, settings=settings, component=name,
                                route="diffusers_group_offload" if group else None,
                                execution_device=device, offload_device="cpu", offload_type="leaf_level",
                                low_cpu_mem_usage=True, use_stream=True, record_stream=False,
                                profile_name=(args.ddo_profile or "") if name == "transformer" else "",
                                build_profile=args.build_ddo_profile if name == "transformer" else False,
                                record_event=tracker.record_event)
        active.append((name, result))
        if name == "transformer" and result.route == "dynamic_offload":
            configuration = result.metrics_snapshot().as_dict()["configuration"]
            budget = configuration.get("dynamic_offload_resident_module_budget_gb")
            if budget is not None:
                print(f"  [dynamic-offload] transformer effective resident-module budget={budget:.2f} GiB "
                      "(after workload profile resolution). This budget does not limit total GPU memory.", flush=True)
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
        if not official_leaf:
            clear_dynamic_offload_pinned_tensor_cache()
        phase.unload_components(list(phase.components))
        tracker.step_end(name, start)
        gc.collect()
        torch.cuda.empty_cache()

    def purge(phase):
        if not official_leaf:
            purge_for_runner(args, settings, phase, record_event=tracker.record_event)

    try:
        purge("before_run")
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
        purge("after_text_encoder")

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
        load_options = dict(subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True,
                            local_files_only=args.local_files_only)
        model_path = args.model_path or args.distilled_model
        if official_leaf:
            transformer = transformer_type.from_pretrained(model_path, **load_options)
        else:
            loaded = from_pretrained_with_dynamic_offload(
                model_path, model_loader=transformer_type, apply_dynamic=False,
                dynamic_offload_config=settings.config if settings.enabled else None, **load_options)
            transformer = loaded.module
        load_ltx_lora(transformer, args, metrics, tracker.record_event)
        if not official_leaf:
            record_runner_report(args, tracker, loaded.metrics_snapshot(), "transformer_load")
            del loaded
        pipe = load_phase(stages["prepare"], ["vae", "audio_vae", "scheduler"])
        pipe.update_components(transformer=transformer)
        run_stage(pipe, state, device)
        scheduler = pipe.scheduler
        if args.compact_timesteps:
            import inspect
            if "timestep_token_repeat" not in inspect.signature(type(transformer).forward).parameters:
                raise RuntimeError("Compact timesteps require the trusted transformer with timestep_token_repeat support")
            repeat = compact_timestep_repeat(
                state["conditioning_mask"],
                latent_frames=(state["num_frames"] - 1) // pipe.vae.temporal_compression_ratio + 1,
                latent_height=args.height // pipe.vae.spatial_compression_ratio,
                latent_width=args.width // pipe.vae.spatial_compression_ratio,
                patch_size=transformer.config.patch_size,
                patch_size_t=transformer.config.patch_size_t,
            )
            state["video_timestep_token_repeat"] = repeat
            metrics["compact_timesteps_executed"] = repeat > 1
            metrics["video_timestep_token_repeat"] = repeat
        if args.reuse_rotary_embeddings:
            rotary_pipe = phase_pipeline(exports.LTX2VideoPrepareRotaryStep(), device)
            rotary_pipe.update_components(transformer=transformer)
            run_stage(rotary_pipe, state, device, retain={"rotary_embeddings"})
            rotary_pipe.unload_components(["transformer"])
            del rotary_pipe
        pipe.unload_components(["transformer", "vae", "audio_vae", "scheduler"])
        pipe = None
        gc.collect()
        tracker.step_end("load_and_prepare_transformer", start)

        # 6. Attach offload before the modular sampling loop.
        start = tracker.step_start("attach_transformer")
        pipe = load_phase(stages["denoise"])
        pipe.update_components(transformer=transformer, scheduler=scheduler,
                               guider=LTX2Guidance(guidance_scale=1.0, guidance_rescale=0.7),
                               audio_guider=exports.LTX2VideoAudioGuidance(guidance_scale=1.0))
        chunk_handles = []
        if args.phase_modulation:
            transformer.enable_phase_modulation()
            metrics["phase_modulation_executed"] = True
        if args.inplace_residuals:
            transformer.enable_inplace_residuals()
            metrics["inplace_residuals_executed"] = True
        if args.feed_forward_chunk_size:
            from scripts.helpers.execution.feed_forward_chunking import enable_feed_forward_chunking
            chunk_handles.append(enable_feed_forward_chunking(transformer, args.feed_forward_chunk_size))
        if args.normalization_chunk_size:
            from scripts.helpers.execution.feed_forward_chunking import enable_normalization_chunking
            chunk_handles.append(enable_normalization_chunking(transformer, args.normalization_chunk_size))
        result = attach(transformer, "transformer")
        tracker.step_end("attach_transformer", start)
        purge("before_transformer")
        start = tracker.step_start("denoise")
        cleanup = result.cleanup_on_error if result is not None else nullcontext
        profile = result.profile_run() if result is not None else nullcontext()
        with cleanup(), profile as profile_report, \
                video_self_attention(
                    transformer, args.video_self_attention, metrics,
                    trace_operators=args.trace_attention_operators):
            run_stage(pipe, state, device, retain={"latents", "audio_latents"})
            torch.cuda.synchronize(device)
        if profile_report is not None:
            metrics["ddo_profile"] = profile_report
        tracker.step_end("denoise", start)
        metrics["denoise_phase_scope"] = "sampling_loop_including_weight_transfers_excluding_transformer_cleanup"
        cleanup_start = tracker.step_start("cleanup_transformer")
        with cleanup():
            finish_phase("cleanup_transformer", cleanup_start, pipe)
        for handle in chunk_handles:
            handle.remove()
        chunk_handles.clear()
        del cleanup, profile, result, transformer, scheduler
        pipe = None
        if args.build_ddo_profile:
            metrics.update(status="completed", operation="capacity_profile_calibration")
            purge("after_run")
            print(f"Capacity profile calibration complete: {metrics.get('ddo_profile')}", flush=True)
            return
        keep = {"latents", "audio_latents", "base_token_count", "audio_ref_num_tokens", "audio_num_frames",
                "height", "width", "num_frames", "dtype", "batch_size", "generator", "seed"}
        state = {name: value for name, value in state.items() if name in keep}
        gc.collect()
        torch.cuda.empty_cache()
        run_stage(load_phase(stages["trim"], phase_device="cpu"), state, "cpu")

        if not official_leaf and args.release_pytorch_host_cache_before_vae:
            release_host_cache_before_vae(tracker, device)

        # 7. Match the VMM decode contract; transformer offload is independent.
        decode_and_save_start = time.time()
        if not official_leaf and args.purge_before_vae:
            purge("before_vae")
        decoded = {}
        decoder_stages = [("video", stages["decode"].sub_blocks["video"], ["vae"]),
                          ("audio", stages["decode"].sub_blocks["audio"], ["audio_vae", "vocoder"])]
        if args.stream_video_decode:
            decoder_stages = [("video", streaming_video_decoder_step(exports, args.video_decoder), ["vae"]),
                              ("audio_mel", exports.LTX2VideoAudioMelDecoderStep(), ["audio_vae"]),
                              ("audio", exports.LTX2VideoVocoderStep(), ["vocoder"])]
        metrics["decode_route"] = {"device": args.decode_device,
                                   "stream_video_decode": args.stream_video_decode, "video_decoder": args.video_decoder,
                                   "planned_stages": [name for name, _, _ in decoder_stages],
                                   "completed_stages": []}
        for name, decoder_block, component_names in decoder_stages:
            load_start = tracker.step_start("load_decode_" + name)
            pipe = load_phase(decoder_block, component_names, phase_device=args.decode_device)
            if name == "video":
                pipe.vae.enable_tiling()
                pipe.vae.use_framewise_decoding = True
            elif name == "audio":
                audio_rate = pipe.vocoder.config.output_sampling_rate
            pipe.to(args.decode_device)
            tracker.step_end("load_decode_" + name, load_start)
            state["output_type"] = "pil"
            input_device = "cpu" if args.stream_video_decode and name == "video" else args.decode_device
            decode_start = tracker.step_start("decode_" + name)
            run_stage(pipe, state, input_device)
            tracker.step_end("decode_" + name, decode_start)
            if name != "audio_mel":
                decoded[name] = state.pop("videos" if name == "video" else "audio")
            if args.stream_video_decode and name == "audio":
                state.pop("generated_mel_spectrograms", None)
            cleanup_start = tracker.step_start("cleanup_decode_" + name)
            pipe.unload_components(component_names)
            metrics["decode_route"]["completed_stages"].append(name)
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            tracker.step_end("cleanup_decode_" + name, cleanup_start)
        video, audio = decoded["video"][0], decoded["audio"][0].float().cpu()
        if len(video) != frames or any(frame.size != (args.width, args.height) for frame in video):
            raise RuntimeError("Decoded frame count/resolution mismatch")
        if audio.numel() == 0 or not bool(torch.isfinite(audio).all()):
            raise RuntimeError("Decoded audio is empty or non-finite")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ltx23_modular_{args.width}x{args.height}_{args.seconds}s_seed{args.seed}.mp4"
        save_start = tracker.step_start("save_video")
        encode_video(video, fps=args.frame_rate, output_path=str(output_path), audio=audio, audio_sample_rate=audio_rate)
        tracker.step_end("save_video", save_start)
        tracker.record_event("decode_and_save", time.time() - decode_and_save_start,
                             scope="aggregate_wall_time_not_an_additional_phase")
        metrics.update(status="completed", output_path=str(output_path))
        if official_leaf:
            metrics["decoded_media"] = dict(frames=len(video), width=args.width, height=args.height)
        purge("after_run")
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
        record_attention_adjusted_total(metrics)
        if args.save_metrics:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metrics").mkdir(exist_ok=True)
            (output_dir / "metrics/modular_run.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if dll_directory is not None:
            dll_directory.close()


if __name__ == "__main__":
    main()
