"""Explicit video compute/control experiments; sampling phases live in run()."""
import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    from diffusers_dynamic_offloader import add_dynamic_offload_arguments
    parser = argparse.ArgumentParser(description=__doc__, conflict_handler="resolve")
    add_dynamic_offload_arguments(parser)
    parser.add_argument("--ddo-vmm-library-path")
    parser.add_argument("--dry-run", action="store_true", help="Print the phase plan without weights, network or CUDA.")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--distilled-model", default="OzzyGT/LTX-2.3-Distilled-1.1")
    parser.add_argument("--model-path", help="Transformer override; auxiliary weights still use --distilled-model.")
    parser.add_argument("--transformer-fp8-weight-only", action="store_true",
                        help="Experimental saved TorchAO FP8 component; requires native VMM and --model-path.")
    parser.add_argument("--fp8-group-offload-control", action="store_true",
                        help="Explicit standard TorchAO/leaf group-offload control; requires linear/off and disables FP8 compute experiments.")
    parser.add_argument("--fp8-group-offload-stream", action=argparse.BooleanOptionalAction, default=True,
                        help="Stream uploads for the explicit FP8 group-control transformer only; auxiliary components are unchanged.")
    parser.add_argument("--compile-fp8-linears", action="store_true",
                        help="Experimental FP8-only Linear compilation with preserved BF16 rounding; no regional compilation.")
    parser.add_argument("--fp8-feed-forward-reuse", action="store_true",
                        help="Experimental bounded FP8 conversion reuse across one feed-forward's chunks.")
    parser.add_argument("--fp8-feed-forward-temporary-mb", type=float,
                        help="Explicit cap for retained converted weights, including pending CUDA events; MiB.")
    parser.add_argument("--custom-blocks", default="elismasilva/ltx2.3_video_custom_blocks",
                        help="Trusted Hugging Face block repository, or an explicit local development directory.")
    parser.add_argument("--custom-blocks-revision", help="Optional immutable custom-code Hub revision.")
    parser.add_argument("--prompt", default="the man plays the guitar and sings on an old western town")
    parser.add_argument("--image", default="https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/man_guitar.png")
    parser.add_argument("--audio", help="Explicit conditioning audio path; no input media is bundled.")
    parser.add_argument("--use-audio-condition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--text-encoder-native-vmm", action="store_true",
                        help="Opt in to staged native encoder ownership; requires an explicit encoder ceiling.")
    parser.add_argument("--text-encoder-vmm-resident-weight-budget-gb", type=float,
                        help="Separate encoder-only native weight ceiling; never changes the transformer ceiling.")
    parser.add_argument("--feed-forward-chunk-size", type=int, default=0)
    parser.add_argument("--normalization-chunk-size", type=int, default=0)
    parser.add_argument("--phase-modulation", action="store_true",
                        help="Opt into phase-local modulation buffers in the updated local Diffusers transformer.")
    parser.add_argument("--compact-timesteps", action="store_true",
                        help="Project one video timestep per spatially uniform frame; otherwise retain the dense path.")
    parser.add_argument("--inplace-residuals", action="store_true",
                        help="Reuse model-owned residual states in eval/no-grad mode; preserves separate multiply rounding.")
    parser.add_argument("--fused-rms-modulation", action="store_true",
                        help="Preserve the original RMSNorm forward and opt into experimental Triton modulation.")
    parser.add_argument("--fused-rotary-pair", action="store_true",
                        help="Opt into experimental paired Q/K Triton rotary application.")
    parser.add_argument("--compile-repeated-blocks", action="store_true",
                        help="Diagnostic regional Inductor compilation; no CUDA Graphs or eager fallback.")
    parser.add_argument("--compile-eager-rmsnorm", action="store_true",
                        help="Experimental opaque original RMSNorm boundary; requires regional compilation.")
    parser.add_argument("--compile-eager-torch-rmsnorm", action="store_true",
                        help="Also preserve PyTorch Q/K RMSNorm; requires --compile-eager-rmsnorm.")
    parser.add_argument("--external-qk-rope-library",
                        help="Opt-in pinned external benchmark kernel; requires both opaque norm flags and compilation.")
    parser.add_argument("--independent-qk-rope", action="store_true",
                        help="Opt-in independent grouped RMSNorm/split-RoPE Triton prototype; native route only.")
    parser.add_argument("--compile-triton-execution", action="store_true",
                        help="Isolated compiled comparison using the previous modulation/paired-RoPE kernels.")
    parser.add_argument("--reuse-rotary-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="Prepare generation-local RoPE once; requires the updated local Diffusers forward.")
    parser.add_argument("--release-pytorch-host-cache-before-transformer", action="store_true")
    parser.add_argument("--detach-transformer-auxiliary-cpu-sources", action="store_true")
    parser.add_argument("--inspect-host-memory-phases", action="store_true")
    parser.add_argument("--diagnose-denoise", action="store_true",
                        help="Opt-in native driver counters and per-step GPU/unit timing; diagnostic, not a performance benchmark.")
    parser.add_argument("--diagnose-operators", action="store_true",
                        help="Profile selected module invocations 0 and 2; may synchronize, requires --diagnose-denoise.")
    parser.add_argument("--diagnose-operator-module", default="transformer_blocks.0",
                        help="Host-selected module path for the optional operator diagnostic.")
    parser.add_argument("--output-type", choices=("video", "latent"), default="video")
    parser.add_argument("--stream-video-decode", action="store_true",
                        help="Opt in to temporal streaming video decode and staged audio VAE/vocoder.")
    parser.add_argument("--decode-device", choices=("cpu", "cuda:0"), default="cpu",
                        help="CPU is the conservative initial decode; streaming GPU parity is not yet established.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "ddo_modular_video"))
    parser.add_argument("--memory-monitor", choices=("full", "rss"), default="rss")
    parser.add_argument("--metrics-level", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.runner_entrypoint = "diagnostics"
    if args.compile_triton_execution:
        if not (args.compile_repeated_blocks and args.compile_eager_rmsnorm and args.compile_eager_torch_rmsnorm):
            parser.error("Compiled Triton requires regional compilation and both opaque norm flags.")
        if (args.fused_rms_modulation or args.fused_rotary_pair or
                args.independent_qk_rope or args.external_qk_rope_library):
            parser.error("Compiled Triton is a separate callback comparison; do not mix variants.")
    if args.independent_qk_rope:
        from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
        if not resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).requires_native_bootstrap:
            parser.error("Independent Q/K requires native vmm/auto.")
        if args.external_qk_rope_library or args.fused_rms_modulation or args.fused_rotary_pair:
            parser.error("Independent Q/K must not mix external or existing Triton callbacks.")
        if args.compile_repeated_blocks and not (args.compile_eager_rmsnorm and args.compile_eager_torch_rmsnorm):
            parser.error("Compiled independent Q/K requires both opaque norm flags.")
    if args.compile_eager_rmsnorm and not args.compile_repeated_blocks:
        parser.error("--compile-eager-rmsnorm requires --compile-repeated-blocks")
    if args.compile_eager_torch_rmsnorm and not args.compile_eager_rmsnorm:
        parser.error("--compile-eager-torch-rmsnorm requires --compile-eager-rmsnorm")
    if args.external_qk_rope_library and not (
            args.compile_repeated_blocks and args.compile_eager_rmsnorm and args.compile_eager_torch_rmsnorm):
        parser.error("External Q/K comparison requires regional compilation and both opaque norm flags.")
    if args.compile_repeated_blocks:
        from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
        if not resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).requires_native_bootstrap:
            parser.error("Regional compilation diagnostics require the native vmm/auto route.")
        if args.fused_rms_modulation or args.fused_rotary_pair:
            parser.error("Validate regional compilation separately from experimental Triton callbacks.")
    if args.diagnose_operators and (not args.diagnose_denoise or args.steps < 3):
        parser.error("Operator diagnostics require --diagnose-denoise and at least three steps.")
    if args.diagnose_denoise:
        from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
        if not resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).requires_native_bootstrap:
            parser.error("Denoise diagnostics require the native vmm/auto route and a separate diagnostic DLL.")
    if args.stream_video_decode and (args.output_type != "video" or args.decode_device != "cuda:0"):
        parser.error("Streaming video decode requires --output-type video --decode-device cuda:0.")
    if args.text_encoder_native_vmm:
        from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
        if not resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).requires_native_bootstrap:
            parser.error("Native text encoding requires the vmm/auto fast route.")
        if (args.text_encoder_vmm_resident_weight_budget_gb is None or
                not math.isfinite(args.text_encoder_vmm_resident_weight_budget_gb) or
                args.text_encoder_vmm_resident_weight_budget_gb <= 0):
            parser.error("Native text encoding requires an explicit positive finite --text-encoder-vmm-resident-weight-budget-gb.")
    elif args.text_encoder_vmm_resident_weight_budget_gb is not None:
        parser.error("The encoder-only ceiling requires --text-encoder-native-vmm.")
    if (args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32 or
            not math.isfinite(args.seconds) or args.seconds <= 0 or
            not math.isfinite(args.frame_rate) or args.frame_rate <= 0):
        parser.error("Use positive, finite duration/fps and dimensions divisible by 32.")
    if not 1 <= args.steps <= 8 or args.max_sequence_length <= 0:
        parser.error("The distilled runner supports 1–8 steps and a positive text sequence length.")
    if min(args.feed_forward_chunk_size, args.normalization_chunk_size) < 0:
        parser.error("Chunk sizes must be nonnegative.")
    # Do not silently reinterpret legacy experiments as this supported workflow.
    for name in ("LTX_VIDEO_ONLY", "LTX_USE_DIFFUSION_DECODER", "LTX_USE_SOFT_ENHANCE_LORA",
                 "LTX_SLIDING_WINDOW_LATENT_FRAMES", "LTX_REFERENCE_KEYFRAME_EVERY_LATENT_FRAMES"):
        if os.getenv(name, "0").strip() not in ("", "0", "false", "False"):
            parser.error(f"{name} is not supported by this modular runner; use the reference runner.")
    if args.compile_fp8_linears and not args.transformer_fp8_weight_only:
        parser.error("FP8 Linear compilation requires --transformer-fp8-weight-only.")
    if args.fp8_feed_forward_reuse:
        if not args.compile_fp8_linears or args.feed_forward_chunk_size <= 0:
            parser.error("FP8 feed-forward reuse requires compiled FP8 Linears and a positive chunk size.")
        cap = args.fp8_feed_forward_temporary_mb
        if cap is None or not math.isfinite(cap) or not math.isfinite(cap*1024**2) or cap*1024**2 < 1:
            parser.error("FP8 feed-forward reuse requires an explicit positive finite temporary MiB cap.")
    elif args.fp8_feed_forward_temporary_mb is not None:
        parser.error("The FP8 temporary cap requires --fp8-feed-forward-reuse.")
    if not args.fp8_group_offload_stream and not args.fp8_group_offload_control:
        parser.error("Disabling FP8 group streams requires --fp8-group-offload-control.")
    if args.fp8_group_offload_control:
        if not args.transformer_fp8_weight_only or args.ddo_backend != "linear" or args.ddo_preset != "off":
            parser.error("The FP8 group control requires --transformer-fp8-weight-only --ddo-backend linear --ddo-preset off.")
        if args.compile_fp8_linears or args.fp8_feed_forward_reuse:
            parser.error("The initial FP8 group control preserves backend compute; compile/reuse experiments are excluded.")
    if args.transformer_fp8_weight_only:
        from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
        if not args.fp8_group_offload_control and not resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).requires_native_bootstrap:
            parser.error("FP8 weight-only requires native vmm/auto.")
        if not args.model_path or not args.local_files_only:
            parser.error("FP8 requires --model-path pointing to the converted model root and --local-files-only.")
        if (args.compile_repeated_blocks or args.fused_rms_modulation or args.fused_rotary_pair
                or args.external_qk_rope_library or args.independent_qk_rope or args.compile_triton_execution
                or args.detach_transformer_auxiliary_cpu_sources):
            parser.error("FP8 excludes regional compilation, custom execution kernels and auxiliary source detachment; use the explicit Linear candidate only.")
    if args.use_audio_condition and not args.audio:
        parser.error("--use-audio-condition requires --audio pointing to an input file.")
    if args.external_qk_rope_library:
        parser.error("External benchmark kernels are not bundled with ddo-examples.")
    return args


def describe_plan(args):
    from diffusers_dynamic_offloader import resolve_dynamic_offload_backend
    route = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset)
    return {
        "runner": {
            "legacy": "ltx23_modular_vmm_video",
            "minimal": "run_dynamic_minimal",
            "linear": "run_dynamic_modular_distilled",
            "vmm": "run_dynamic_modular_vmm_distilled",
            "diagnostics": "run_dynamic_modular_diagnostics_distilled",
        }[args.runner_entrypoint], "experimental": True,
        "backend": route.backend, "effective_preset": route.effective_preset,
        "custom_blocks": args.custom_blocks, "custom_blocks_revision": args.custom_blocks_revision,
        "model": args.model_path or args.distilled_model,
        "auxiliary_model": args.distilled_model, "dtype": "torch.bfloat16",
        "transformer_weight_format_requested": "torchao_fp8_weight_only" if args.transformer_fp8_weight_only else "bf16",
        "compile_fp8_linears_requested": args.compile_fp8_linears,
        "fp8_group_offload_control_requested": args.fp8_group_offload_control,
        "fp8_group_offload_stream_requested": args.fp8_group_offload_stream if args.fp8_group_offload_control else None,
        "denoise_execution_mode_requested": denoise_execution_mode(args),
        "transformer_offload_route_requested": "diffusers_group_offload" if args.fp8_group_offload_control else route.backend,
        "fp8_feed_forward_reuse_requested": args.fp8_feed_forward_reuse,
        "fp8_feed_forward_temporary_mb": args.fp8_feed_forward_temporary_mb,
        "width": args.width, "height": args.height, "seconds": args.seconds,
        "frame_rate": args.frame_rate, "num_frames": round(args.seconds * args.frame_rate) // 8 * 8 + 1,
        "steps": args.steps, "seed": args.seed, "use_audio_condition": args.use_audio_condition,
        "feed_forward_chunk_size": args.feed_forward_chunk_size,
        "normalization_chunk_size": args.normalization_chunk_size,
        "phase_modulation": args.phase_modulation,
        "compact_timesteps_requested": args.compact_timesteps,
        "inplace_residuals_requested": args.inplace_residuals,
        "fused_rms_modulation_requested": args.fused_rms_modulation,
        "fused_rotary_pair_requested": args.fused_rotary_pair,
        "compile_repeated_blocks_requested": args.compile_repeated_blocks,
        "compile_eager_rmsnorm_requested": args.compile_eager_rmsnorm,
        "compile_eager_torch_rmsnorm_requested": args.compile_eager_torch_rmsnorm,
        "external_qk_rope_library_requested": args.external_qk_rope_library,
        "independent_qk_rope_requested": args.independent_qk_rope,
        "compile_triton_execution_requested": args.compile_triton_execution,
        "reuse_rotary_embeddings": args.reuse_rotary_embeddings,
        "text_encoder_route_requested": "native" if args.text_encoder_native_vmm else "diffusers_group_offload",
        "text_encoder_vmm_resident_weight_budget_gb": args.text_encoder_vmm_resident_weight_budget_gb,
        "output_type": args.output_type, "decode_device": args.decode_device,
        "stream_video_decode": args.stream_video_decode,
        "diagnose_denoise": args.diagnose_denoise,
        "diagnose_operators": args.diagnose_operators,
        "diagnose_operator_module": args.diagnose_operator_module if args.diagnose_operators else None,
        "phases": ["text_encoder", "connectors", "video_encoder", "audio_encoder",
                   "prepare", "denoise", "trim", "decode" if args.output_type == "video" else "save_latents"],
        "validation": "CPU contracts only; real-checkpoint parity and CUDA execution pending",
    }


def group_stream_for_component(args, component):
    """Scope the explicit control override to its transformer; preserve defaults."""
    if args.fp8_group_offload_control and component == "transformer":
        return args.fp8_group_offload_stream
    return True


def denoise_execution_mode(args):
    """Avoid the reproduced TorchAO group-hook inference-mode alias failure."""
    return "no_grad" if args.fp8_group_offload_control else "inference_mode"


def build_transformer_config(args, config):
    """Keep dense defaults; opt into native composite ownership explicitly."""
    if not args.transformer_fp8_weight_only or args.fp8_group_offload_control:
        return config
    return replace(config, vmm_float8_weight_only=True,
                   vmm_component_tensor_ownership=True, vmm_module_pattern=".*",
                   vmm_execution_auxiliary_ownership=False, profile_name=None, build_profile=False)


def build_text_encoder_config(args, transformer_config):
    """Select explicit component ownership without mutating transformer limits."""
    if not args.text_encoder_native_vmm:
        return None
    return replace(transformer_config,
                   vmm_resident_weight_budget_gb=args.text_encoder_vmm_resident_weight_budget_gb,
                   vmm_module_pattern=".*", vmm_page_mb=2,
                   vmm_component_tensor_ownership=True, vmm_component_buffer_limit_mb=1,
                   vmm_execution_auxiliary_ownership=False,
                   vmm_float8_weight_only=False,
                   profile_name=None, build_profile=False)


def load_text_encoder_for_phase(args, dtype, config, record_report):
    """Use DDO's public weight-aware loader only for the explicit native route."""
    from transformers import Gemma3ForConditionalGeneration
    from diffusers_dynamic_offloader import from_pretrained_with_dynamic_offload
    loading = dict(subfolder="text_encoder", torch_dtype=dtype,
                   low_cpu_mem_usage=True, local_files_only=args.local_files_only)
    if config is None:
        return Gemma3ForConditionalGeneration.from_pretrained(args.distilled_model, **loading)
    loaded = from_pretrained_with_dynamic_offload(
        args.distilled_model, model_loader=Gemma3ForConditionalGeneration,
        apply_dynamic=False, dynamic_offload_config=config, **loading)
    module = loaded.module.eval()
    record_report(loaded.inspect_vmm_component(component="text_encoder"), "text_encoder_preflight")
    return module


def load_transformer_for_phase(args, model_loader, dtype, config, enabled):
    """Use the standard Diffusers quantized loader for the explicit group control."""
    from diffusers_dynamic_offloader import DynamicOffloadLoadResult, from_pretrained_with_dynamic_offload
    loading = dict(subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True,
                   local_files_only=args.local_files_only)
    checkpoint = args.model_path or args.distilled_model
    if args.fp8_group_offload_control:
        module = model_loader.from_pretrained(checkpoint, **loading)
        validate_fp8_group_control_weights(module)
        return DynamicOffloadLoadResult(module=module)
    return from_pretrained_with_dynamic_offload(
        checkpoint, model_loader=model_loader, apply_dynamic=False,
        dynamic_offload_config=config if enabled else None, **loading)


def validate_fp8_group_control_weights(module):
    """Refuse a silently dense or differently quantized control before CUDA."""
    import torch
    from torchao.quantization import Float8Tensor
    weights = [child.weight for child in module.modules()
               if type(child) is torch.nn.Linear and type(child.weight) is Float8Tensor]
    if not weights or any(parameter.device.type != "cpu" for parameter in module.parameters()):
        raise ValueError("Group control requires CPU-loaded standard FP8 Linears")
    for weight in weights:
        if (weight.dtype != torch.bfloat16 or weight.qdata.dtype != torch.float8_e4m3fn
                or weight.scale.dtype != torch.float32 or weight.act_quant_kwargs is not None
                or weight.mm_config is not None or weight.block_size != [1, weight.shape[1]]
                or tuple(weight.scale.shape) != (weight.shape[0], 1)):
            raise ValueError("Group control requires the matching per-row E4M3 weight-only recipe")


def move_tensors(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def make_phase_pipeline(blocks, device, checkpoint=None, local_files_only=False):
    """Give configuration-only preparation an explicit execution device.

    A CPU-loaded native transformer must not be moved wholesale to CUDA merely
    to let the modular framework infer the device for noise or RoPE tensors.
    This host pipeline specialization changes device inference, not allocation,
    residency or the model forward.
    """
    import torch
    from diffusers.modular_pipelines.ltx2.modular_pipeline import LTX2ModularPipeline

    class PhasePipeline(LTX2ModularPipeline):
        @property
        def _execution_device(self):
            return torch.device(device)

    loading = {"local_files_only": local_files_only} if checkpoint is not None else {}
    return PhasePipeline(blocks=blocks, pretrained_model_name_or_path=checkpoint, **loading)


def invoke_stage(pipe, payload, device, retain=None, *, execution_mode="inference_mode", execution_report=None):
    """Pass declared fields only, retaining kwargs-type coordinate metadata."""
    import torch
    if execution_mode not in ("inference_mode", "no_grad"):
        raise ValueError("Unsupported phase execution mode")
    inputs = {}
    for param in pipe.blocks.inputs:
        if param.name in payload:
            inputs[param.name] = move_tensors(payload[param.name], device)
        if param.kwargs_type and param.kwargs_type in payload:
            inputs[param.kwargs_type] = move_tensors(payload[param.kwargs_type], device)
    context = torch.inference_mode() if execution_mode == "inference_mode" else torch.no_grad()
    with context:
        if execution_mode == "no_grad" and torch.is_inference_mode_enabled():
            raise RuntimeError("FP8 group no-grad control cannot run inside inference_mode")
        if execution_report is not None:
            execution_report.update(mode=execution_mode,
                                    inference_mode_enabled=torch.is_inference_mode_enabled(),
                                    grad_enabled=torch.is_grad_enabled())
        state = pipe(**inputs)
    for param in pipe.blocks.intermediate_outputs:
        if param.name and (retain is None or param.name in retain):
            value = move_tensors(state.get(param.name), "cpu")
            payload[param.name] = value
            if param.kwargs_type:
                payload.setdefault(param.kwargs_type, {})[param.name] = value
    return payload


def compact_timestep_repeat(mask, *, latent_frames, latent_height, latent_width, patch_size=1, patch_size_t=1):
    """Admit a known packed layout on CPU once, never synchronize CUDA in the denoise loop.

    Spatially varying masks, appended references and unsupported temporal patching
    retain the dense path. Mask equality is checked, not guessed from geometry.
    """
    import torch
    if mask.device.type != "cpu":
        raise ValueError("Compact timestep admission requires a CPU mask")
    if patch_size_t != 1 or latent_height % patch_size or latent_width % patch_size:
        return 1
    repeat = (latent_height // patch_size) * (latent_width // patch_size)
    if repeat <= 1 or mask.ndim != 3 or mask.shape[-1] != 1 or mask.shape[1] != latent_frames * repeat:
        return 1
    grouped = mask.reshape(mask.shape[0], latent_frames, repeat, 1)
    if not torch.isfinite(mask).all() or not torch.equal(grouped, grouped[:, :, :1].expand_as(grouped)):
        return 1
    return repeat


def load_custom_blocks(args):
    """Load one trusted code revision; local-files-only also applies to Hub code."""
    from diffusers import ModularPipelineBlocks

    options = dict(trust_remote_code=True, local_files_only=args.local_files_only)
    source = args.custom_blocks
    if not Path(source).is_dir():
        from huggingface_hub import snapshot_download

        # This Diffusers revision treats dots in Hub repo IDs as Python package
        # separators. Resolve code only to an immutable snapshot, then use its
        # standard trusted local loader. Never download weights in this step.
        revision = args.custom_blocks_revision
        if revision is None and source == "elismasilva/ltx2.3_video_custom_blocks":
            revision = "f5c712e93c52283cb101da26d4a4544c808666bf"
        source = snapshot_download(
            repo_id=source, revision=revision,
            allow_patterns=["*.py", "modular_config.json"],
            local_files_only=args.local_files_only,
        )
    return ModularPipelineBlocks.from_pretrained(source, **options)


def custom_block_exports(blocks):
    """Resolve helpers from the same cached code namespace as the block set."""
    from importlib import import_module

    name = type(blocks).__module__
    if not name.endswith(".modular_blocks_ltx2_video"):
        raise ValueError("Video blocks must expose the trusted video entry module")
    return import_module(name)


def resolve_local_transformer_loader(blocks):
    """Use the component class from the selected trusted custom-block package.

    This also preserves class identity when trust_remote_code loads a cached
    package rather than importing the checkout directly.
    """
    specs = [spec for spec in blocks.expected_components if spec.name == "transformer"]
    if len(specs) != 1:
        raise ValueError("Custom blocks must declare one transformer component")
    loader = specs[0].type_hint
    if loader is None or not loader.__module__.endswith(".transformer_ltx2_video"):
        raise ValueError("Modular video requires its independent local transformer class")
    return loader


def run(args):
    import torch
    from diffusers import ModularPipelineBlocks
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    from diffusers.utils import load_image, encode_video
    from transformers import Gemma3ForConditionalGeneration
    from diffusers_dynamic_offloader import (
        DynamicOffloadSettings, resolve_dynamic_offload_backend, enable_offload,
        from_pretrained_with_dynamic_offload, remove_dynamic_offload,
        clear_dynamic_offload_pinned_tensor_cache, maybe_purge_windows_standby_cache,
        release_pytorch_host_cache, inspect_host_memory,
    )
    from scripts.helpers.inference_utils import RunTracker
    try:
        from diffusers.guiders import LTX2Guidance
    except ImportError:
        from diffusers.modular_pipelines.ltx2.guider import LTX2Guidance

    device, dtype = "cuda:0", torch.bfloat16
    route = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset)
    native = route.requires_native_bootstrap
    native_controls = (args.feed_forward_chunk_size or args.normalization_chunk_size or args.phase_modulation or args.compact_timesteps or args.inplace_residuals or args.fused_rms_modulation or args.fused_rotary_pair or
                       args.release_pytorch_host_cache_before_transformer or
                       args.detach_transformer_auxiliary_cpu_sources or args.inspect_host_memory_phases)
    if native_controls and not native and not args.fp8_group_offload_control:
        raise ValueError("Native video controls require --ddo-backend vmm --ddo-preset auto.")
    settings = DynamicOffloadSettings.from_env(
        execution_device=device, offload_device="cpu",
        default_preset=args.ddo_preset if args.ddo_backend == "linear" else "one_shot_fast",
    )
    if args.ddo_backend == "vmm":
        settings = replace(settings, config=replace(route.build_config_from_args(args),
                           execution_device=device, offload_device="cpu", vmm_native_transport=native),
                           enabled=True, plan=False, requested_preset=route.requested_preset,
                           effective_preset=route.effective_preset,
                           requested_pin_cpu_memory=False, effective_pin_cpu_memory=False)
    metrics = {**describe_plan(args), "events": [], "steps": [], "requested_controls": vars(args).copy()}
    tracker = RunTracker(device, metrics, show_metrics=args.show_metrics, memory_monitor=args.memory_monitor)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_results = []
    current_pipe = None
    fp8_chunk_handle = None
    blocks = load_custom_blocks(args)
    exports = custom_block_exports(blocks)
    LTX2ImageCondition = exports.LTX2ImageCondition
    LTX2AudioCondition = exports.LTX2AudioCondition
    LTX2VideoAudioGuidance = exports.LTX2VideoAudioGuidance
    if type(blocks).__name__ != "LTX2VideoDistilledBlocks":
        raise ValueError("This runner requires the distilled video block set.")
    transformer_loader = resolve_local_transformer_loader(blocks)
    metrics["transformer_implementation"] = transformer_loader.__module__
    stages = blocks.sub_blocks
    payload = {
        "prompt": args.prompt, "max_sequence_length": args.max_sequence_length,
        "width": args.width, "height": args.height, "num_frames": metrics["num_frames"],
        "frame_rate": args.frame_rate, "num_inference_steps": args.steps,
        "num_videos_per_prompt": 1, "sigmas": list(DISTILLED_SIGMA_VALUES[:args.steps]),
        "generator": torch.Generator(device=device).manual_seed(args.seed), "dtype": dtype,
        "use_cross_timestep": True, "seed": args.seed,
    }

    def report(value, name):
        tracker.record_ddo_metrics(value, component=name)
        if args.show_profile and not args.show_metrics:
            print(value.format_profile())

    def inspect(phase, module=None):
        if args.inspect_host_memory_phases:
            report(inspect_host_memory(module, phase=phase), "host_memory_" + phase)

    def load_stage(name, names=()):
        pipe = make_phase_pipeline(stages[name], device, args.distilled_model, args.local_files_only)
        # Specs are resolved from each subfolder config; load no unrelated models.
        for component in names:
            pipe.load_components(names=component, pretrained_model_name_or_path=args.distilled_model,
                                 subfolder=component, torch_dtype=dtype,
                                 local_files_only=args.local_files_only)
            if getattr(pipe, component) is None:
                raise RuntimeError(f"Failed to load {name}/{component}; see component diagnostics.")
        return pipe

    def retire_results():
        for name, result in active_results:
            with result.cleanup_on_error():
                report(result.metrics_snapshot(collect_completed=True), name)
                if result.route == "dynamic_offload":
                    remove_dynamic_offload(result.module)
        active_results.clear()
        clear_dynamic_offload_pinned_tensor_cache()

    def attach(module, name, group=False, config=None):
        component_config = {"config": config} if config is not None else {}
        result = enable_offload(module, settings=settings, component=name,
                                route="diffusers_group_offload" if group else (
                                    "dynamic_offload" if args.ddo_backend == "vmm" else None),
                                execution_device=device, offload_device="cpu",
                                low_cpu_mem_usage=True, offload_type="leaf_level",
                                use_stream=group_stream_for_component(args, name), record_stream=False, record_event=tracker.record_event,
                                **component_config)
        active_results.append((name, result))
        if result.should_move_to_execution_device:
            module.to(device)
        return result

    def finish_phase(name, start, pipe):
        retire_results()
        pipe.unload_components(list(pipe.components))
        tracker.step_end(name, start)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        maybe_purge_windows_standby_cache(settings, "before_run", record_event=tracker.record_event)
        start = tracker.step_start("text_encoder")
        current_pipe = load_stage("text_encoder")
        current_pipe.load_components(names="tokenizer", local_files_only=args.local_files_only)
        if current_pipe.tokenizer is None:
            raise RuntimeError("Failed to load the checkpoint-declared tokenizer.")
        encoder_config = build_text_encoder_config(args, settings.config)
        current_pipe.update_components(text_encoder=load_text_encoder_for_phase(args, dtype, encoder_config, report))
        encoder_result = attach(current_pipe.text_encoder, "text_encoder",
                                group=encoder_config is None, config=encoder_config)
        with encoder_result.cleanup_on_error():
            invoke_stage(current_pipe, payload, device)
            finish_phase("text_encoder", start, current_pipe)
        del encoder_result, encoder_config
        current_pipe = None
        maybe_purge_windows_standby_cache(settings, "after_text_encoder", record_event=tracker.record_event)

        start = tracker.step_start("connectors")
        current_pipe = load_stage("connectors", ["connectors"])
        current_pipe.load_components(names="tokenizer", local_files_only=args.local_files_only)
        if current_pipe.tokenizer is None:
            raise RuntimeError("Failed to load the checkpoint-declared connector tokenizer.")
        current_pipe.tokenizer.padding_side = "left"
        attach(current_pipe.connectors, "connectors", group=True)
        invoke_stage(current_pipe, payload, device)
        # Text hidden states are no longer needed after connection.
        for key in ("prompt_embeds", "negative_prompt_embeds", "prompt_attention_mask", "negative_prompt_attention_mask"):
            payload.pop(key, None)
        finish_phase("connectors", start, current_pipe)
        current_pipe = None

        start = tracker.step_start("video_encoder")
        current_pipe = load_stage("video_encoder", ["vae"])
        current_pipe.vae.enable_tiling()
        current_pipe.vae.use_framewise_encoding = True
        attach(current_pipe.vae, "vae", group=True)
        payload["image_conditions"] = [LTX2ImageCondition(load_image(args.image))] if args.image else []
        invoke_stage(current_pipe, payload, device)
        payload.pop("image_conditions", None)
        finish_phase("video_encoder", start, current_pipe)
        current_pipe = None

        start = tracker.step_start("audio_encoder")
        current_pipe = load_stage("audio_encoder", ["audio_vae"])
        attach(current_pipe.audio_vae, "audio_vae", group=True)
        payload["audio_conditions"] = []
        if args.use_audio_condition:
            import torchaudio
            waveform, sample_rate = torchaudio.load(args.audio)
            target_rate = current_pipe.audio_vae.config.sample_rate
            if sample_rate != target_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
            waveform = waveform[:, :round(args.seconds * target_rate)]
            payload["audio_conditions"] = [LTX2AudioCondition(waveform, end_time=args.seconds)]
            del waveform
        invoke_stage(current_pipe, payload, device)
        payload.pop("audio_conditions", None)
        finish_phase("audio_encoder", start, current_pipe)
        current_pipe = None

        start = tracker.step_start("load_and_prepare_transformer")
        inspect("before_transformer_load")
        transformer_config = build_transformer_config(args, settings.config)
        loaded = load_transformer_for_phase(args, transformer_loader, dtype, transformer_config, settings.enabled)
        transformer = loaded.module
        report(loaded.metrics_snapshot(), "transformer_load")
        del loaded
        inspect("transformer_loaded", transformer)
        # The prepare stages use configuration/RoPE only; weights stay unattached.
        current_pipe = load_stage("prepare", ["vae", "audio_vae", "scheduler"])
        current_pipe.update_components(transformer=transformer)
        invoke_stage(current_pipe, payload, device)
        scheduler = current_pipe.scheduler
        if args.compact_timesteps:
            import inspect as python_inspect
            if "timestep_token_repeat" not in python_inspect.signature(type(transformer).forward).parameters:
                raise RuntimeError("Compact timesteps require the updated local Diffusers forward")
            repeat = compact_timestep_repeat(
                payload["conditioning_mask"],
                latent_frames=(payload["num_frames"] - 1) // current_pipe.vae.temporal_compression_ratio + 1,
                latent_height=args.height // current_pipe.vae.spatial_compression_ratio,
                latent_width=args.width // current_pipe.vae.spatial_compression_ratio,
                patch_size=transformer.config.patch_size,
                patch_size_t=transformer.config.patch_size_t,
            )
            payload["video_timestep_token_repeat"] = repeat
            metrics["compact_timesteps_executed"] = repeat > 1
            metrics["video_timestep_token_repeat"] = repeat
        if args.reuse_rotary_embeddings:
            import inspect as python_inspect
            LTX2VideoPrepareRotaryStep = exports.LTX2VideoPrepareRotaryStep
            if "rotary_embeddings" not in python_inspect.signature(type(transformer).forward).parameters:
                raise RuntimeError("RoPE reuse requires the updated local Diffusers; use --no-reuse-rotary-embeddings for the unchanged forward")
            rotary_pipe = make_phase_pipeline(LTX2VideoPrepareRotaryStep(), device)
            rotary_pipe.update_components(transformer=transformer)
            invoke_stage(rotary_pipe, payload, device, retain={"rotary_embeddings"})
            rotary_pipe.unload_components(["transformer"])
            del rotary_pipe
        current_pipe.unload_components(["transformer", "vae", "audio_vae", "scheduler"])
        current_pipe = None
        gc.collect()
        if args.release_pytorch_host_cache_before_transformer:
            report(release_pytorch_host_cache(enabled=True, phase="before_transformer", device=device), "host_cache_before_transformer")
        tracker.step_end("load_and_prepare_transformer", start)

        start = tracker.step_start("attach_transformer")
        current_pipe = load_stage("denoise")
        current_pipe.update_components(transformer=transformer, scheduler=scheduler,
                                       guider=LTX2Guidance(guidance_scale=1.0, guidance_rescale=0.7),
                                       audio_guider=LTX2VideoAudioGuidance(guidance_scale=1.0))
        chunk_handles = []
        if args.phase_modulation:
            transformer.enable_phase_modulation()
        if args.inplace_residuals:
            transformer.enable_inplace_residuals()
            metrics["inplace_residuals_executed"] = True
        if args.fused_rms_modulation or args.fused_rotary_pair:
            from ddo_vmm_allocator.triton_execution import TritonExecutionOps
            execution_ops = TritonExecutionOps()
            transformer.set_execution_ops(
                rms_modulate=execution_ops.rms_modulate if args.fused_rms_modulation else None,
                rotary_pair=execution_ops.rotary_pair if args.fused_rotary_pair else None,
            )
            metrics["execution_ops_requested"] = dict(
                rms_modulation=args.fused_rms_modulation, rotary_pair=args.fused_rotary_pair
            )
        if args.feed_forward_chunk_size:
            from scripts.helpers.execution.feed_forward_chunking import enable_feed_forward_chunking
            chunk_handles.append(enable_feed_forward_chunking(transformer, args.feed_forward_chunk_size))
        if args.normalization_chunk_size:
            from scripts.helpers.execution.feed_forward_chunking import enable_normalization_chunking
            chunk_handles.append(enable_normalization_chunking(transformer, args.normalization_chunk_size))
        external_qk_rope_handle = None
        independent_qk_rope_handle = None
        compiled_triton_handle = None
        if args.compile_triton_execution:
            from scripts.helpers.ltx2.compiled_triton_execution import configure_compiled_triton
            metrics["compiled_triton_execution"], compiled_triton_handle = configure_compiled_triton(transformer)
        if args.independent_qk_rope:
            from scripts.helpers.ltx2.independent_qk_rope_execution import configure_independent_qk_rope
            metrics["independent_qk_rope"], independent_qk_rope_handle = configure_independent_qk_rope(transformer)
        if args.external_qk_rope_library:
            from external_qk_rope_execution import configure_external_qk_rope
            metrics["external_qk_rope"], external_qk_rope_handle = configure_external_qk_rope(
                transformer, args.external_qk_rope_library)
        fp8_compile_handle = None
        if args.compile_fp8_linears:
            from diffusers_dynamic_offloader.float8_compile import configure_float8_linear_compile
            metrics["fp8_linear_compile"], fp8_compile_handle = configure_float8_linear_compile(transformer)
        if args.fp8_feed_forward_reuse:
            configure_float8_chunk_reuse = exports.configure_float8_chunk_reuse
            fp8_chunk_handle = configure_float8_chunk_reuse(
                transformer, temporary_cap_bytes=int(args.fp8_feed_forward_temporary_mb*1024**2),
                compile_compute=True, prior_compile_handle=fp8_compile_handle)
            metrics["fp8_feed_forward_reuse"] = dict(
                requested=True, configuration_applied=True, denoise_completed=False, callback_retired=False)
        regional_compile_handle = None
        if args.compile_repeated_blocks:
            from scripts.helpers.execution.regional_compile import configure_regional_compile
            metrics["regional_compile"], regional_compile_handle = configure_regional_compile(
                transformer, eager_rmsnorm=args.compile_eager_rmsnorm,
                normalization_chunk_size=args.normalization_chunk_size or 0,
                eager_torch_rmsnorm=args.compile_eager_torch_rmsnorm)
            if args.external_qk_rope_library or args.independent_qk_rope:
                metrics["regional_compile"]["normalization_execution_note"] = (
                    "Opaque norm counts are installed boundaries, not invocation counts; "
                    "attention with RoPE executes the explicitly selected fused Q/K normalization instead.")
        result = attach(transformer, "transformer", group=args.fp8_group_offload_control,
                        config=transformer_config if args.transformer_fp8_weight_only and not args.fp8_group_offload_control else None)
        with result.cleanup_on_error():
            if args.detach_transformer_auxiliary_cpu_sources:
                report(result.detach_auxiliary_cpu_sources(enabled=True), "transformer_auxiliary_sources")
            inspect("transformer_attached", transformer)
            tracker.step_end("attach_transformer", start)
            maybe_purge_windows_standby_cache(settings, "before_transformer", record_event=tracker.record_event)
            start = tracker.step_start("denoise")
            denoise_context_report = metrics.setdefault("denoise_execution_context_executed", {})
            if args.diagnose_denoise:
                with result.execution_timer(enabled=True, native_units=True) as execution_timer, result.allocation_diagnostics(enabled=True) as allocation_diagnostics, result.operator_diagnostics(enabled=args.diagnose_operators, module_path=args.diagnose_operator_module, output_dir=output_dir / "operator_traces") as operator_diagnostics, result.metrics_journal(output_dir / "metrics" / "denoise_snapshots", enabled=True) as metrics_journal:
                    metrics_journal.record()
                    def observe_step(index):
                        execution_timer.mark_step()
                        allocation_diagnostics.mark_step()
                        snapshot_path = metrics_journal.record(index)
                        print(f"  [diagnostic] step {index + 1} queued; partial snapshot: {snapshot_path}", flush=True)
                    payload["callback_on_step_end"] = observe_step
                    try:
                        invoke_stage(current_pipe, payload, device, retain={"latents", "audio_latents"},
                                     execution_mode=denoise_execution_mode(args), execution_report=denoise_context_report)
                    finally:
                        payload.pop("callback_on_step_end", None)
            else:
                invoke_stage(current_pipe, payload, device, retain={"latents", "audio_latents"},
                             execution_mode=denoise_execution_mode(args), execution_report=denoise_context_report)
            torch.cuda.synchronize(device)
            if args.compile_repeated_blocks:
                metrics["regional_compile"]["denoise_completed"] = True
            if args.external_qk_rope_library:
                metrics["external_qk_rope"]["denoise_completed"] = True
            if args.independent_qk_rope:
                metrics["independent_qk_rope"]["denoise_completed"] = True
            if args.compile_triton_execution:
                metrics["compiled_triton_execution"]["denoise_completed"] = True
            if args.diagnose_denoise:
                report(operator_diagnostics.attach(allocation_diagnostics.attach(result.metrics_snapshot(execution_timing=execution_timer))),
                       "transformer_diagnostics")
            tracker.step_end("denoise", start)
            inspect("after_generation", transformer)
            retire_results()
        if fp8_chunk_handle is not None:
            fp8_chunk_handle.remove()
            metrics["fp8_feed_forward_reuse"].update(denoise_completed=True, callback_retired=True)
        if fp8_compile_handle is not None:
            fp8_compile_handle.remove()
            metrics["fp8_linear_compile"].update(denoise_completed=True, callback_retired=True)
        if regional_compile_handle is not None:
            regional_compile_handle.remove()
        if compiled_triton_handle is not None:
            compiled_triton_handle.remove()
            metrics["compiled_triton_execution"]["callback_retired"] = True
        if external_qk_rope_handle is not None:
            external_qk_rope_handle.remove()
            metrics["external_qk_rope"]["callback_retired"] = True
        if independent_qk_rope_handle is not None:
            independent_qk_rope_handle.remove()
            metrics["independent_qk_rope"]["callback_retired"] = True
        current_pipe.unload_components(list(current_pipe.components))
        current_pipe = None
        for handle in chunk_handles:
            handle.remove()
        chunk_handles.clear()
        del result, transformer, scheduler
        # Keep only the generated states and decode metadata past native retirement.
        keep = {"latents", "audio_latents", "base_token_count", "audio_ref_num_tokens", "audio_num_frames",
                "height", "width", "num_frames", "dtype", "batch_size", "generator", "seed"}
        payload = {key: value for key, value in payload.items() if key in keep}
        gc.collect()
        torch.cuda.empty_cache()
        invoke_stage(load_stage("trim"), payload, "cpu")

        start = tracker.step_start("decode_and_save")
        stem = f"ltx23_modular_{args.width}x{args.height}_{args.seconds}s_seed{args.seed}"
        if args.output_type == "latent":
            output_path = output_dir / (stem + ".pt")
            torch.save({key: payload[key] for key in ("latents", "audio_latents", "audio_num_frames", "num_frames",
                                                     "width", "height", "seed") if key in payload}, output_path)
        else:
            decode_blocks = stages["decode"].sub_blocks
            decoded = {}
            audio_rate = None
            decoder_stages = [("video", decode_blocks["video"], ["vae"]),
                              ("audio", decode_blocks["audio"], ["audio_vae", "vocoder"])]
            if args.stream_video_decode:
                LTX2VideoStreamingDecoderStep = exports.LTX2VideoStreamingDecoderStep
                LTX2VideoAudioMelDecoderStep = exports.LTX2VideoAudioMelDecoderStep
                LTX2VideoVocoderStep = exports.LTX2VideoVocoderStep
                decoder_stages = [("video", LTX2VideoStreamingDecoderStep(), ["vae"]),
                                  ("audio_mel", LTX2VideoAudioMelDecoderStep(), ["audio_vae"]),
                                  ("audio", LTX2VideoVocoderStep(), ["vocoder"])]
            for name, decoder_block, component_names in decoder_stages:
                current_pipe = make_phase_pipeline(decoder_block, args.decode_device,
                                                   args.distilled_model, args.local_files_only)
                for component in component_names:
                    current_pipe.load_components(names=component, pretrained_model_name_or_path=args.distilled_model,
                                                 subfolder=component, torch_dtype=dtype,
                                                 local_files_only=args.local_files_only)
                    if getattr(current_pipe, component) is None:
                        raise RuntimeError(f"Failed to load decoder component {component}")
                if name == "video":
                    current_pipe.vae.enable_tiling()
                    current_pipe.vae.use_framewise_decoding = True
                elif name == "audio":
                    audio_rate = current_pipe.vocoder.config.output_sampling_rate
                current_pipe.to(args.decode_device)
                payload["output_type"] = "pil"
                input_device = "cpu" if args.stream_video_decode and name == "video" else args.decode_device
                invoke_stage(current_pipe, payload, input_device)
                if name != "audio_mel":
                    decoded[name] = payload.pop("videos" if name == "video" else "audio")
                if args.stream_video_decode and name == "audio":
                    payload.pop("generated_mel_spectrograms", None)
                current_pipe.unload_components(component_names)
                current_pipe = None
                gc.collect()
                torch.cuda.empty_cache()
            output_path = output_dir / (stem + ".mp4")
            frames, waveform = decoded["video"][0], decoded["audio"][0].float().cpu()
            if len(frames) != payload["num_frames"] or any(frame.size != (args.width, args.height) for frame in frames):
                raise RuntimeError("Decoded video does not match the requested frame count/resolution")
            if waveform.numel() == 0 or not bool(torch.isfinite(waveform).all()):
                raise RuntimeError("Decoded audio is empty or non-finite")
            metrics["decoded_media"] = dict(frames=len(frames), width=args.width, height=args.height,
                                           audio_shape=list(waveform.shape), audio_sample_rate=audio_rate,
                                           audio_finite=True)
            encode_video(frames, fps=args.frame_rate, output_path=str(output_path),
                         audio=waveform, audio_sample_rate=audio_rate)
        tracker.step_end("decode_and_save", start)
        metrics["status"] = "completed"
        metrics["output_path"] = str(output_path)
        maybe_purge_windows_standby_cache(settings, "after_run", record_event=tracker.record_event)
        print(f"Saved: {output_path}")
    except BaseException as error:
        metrics["status"], metrics["error"] = "failed", repr(error)
        # The error context retains failed native ownership when retirement is unsafe.
        for name, result in active_results:
            with result.cleanup_on_error():
                raise error
        raise
    finally:
        tracker.monitor.stop()
        if fp8_chunk_handle is not None:
            metrics["fp8_feed_forward_reuse"].update(fp8_chunk_handle.snapshot())
        if "execution_ops" in locals():
            metrics["execution_ops_submissions"] = execution_ops.snapshot()
        metrics["total_elapsed_sec"] = round(tracker.total_elapsed(), 4)
        metrics["global_peak_vram_gb"] = tracker.global_peak_vram
        metrics["global_peak_ram_gb"] = tracker.global_peak_ram
        if args.save_metrics:
            metrics_path = output_dir / "metrics" / "modular_run.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".hf_modules_modular_video"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from ddo_vmm_bootstrap import ensure_vmm_fast_process, vmm_fast_requested
    if not any(flag in sys.argv[1:] for flag in ("--help", "-h", "--dry-run")) and vmm_fast_requested():
        early = argparse.ArgumentParser(add_help=False)
        early.add_argument("--ddo-vmm-library-path")
        early_args, _ = early.parse_known_args()
        ensure_vmm_fast_process(library_path=early_args.ddo_vmm_library_path)
    args = parse_args()
    if args.dry_run:
        print(json.dumps(describe_plan(args), indent=2))
        return
    ffmpeg_bin = os.getenv("DDO_RUNNER_FFMPEG_BIN", "")
    dll_directory = os.add_dll_directory(ffmpeg_bin) if os.name == "nt" and os.path.isdir(ffmpeg_bin) else None
    try:
        run(args)
    finally:
        if dll_directory is not None:
            dll_directory.close()


if __name__ == "__main__":
    main()
