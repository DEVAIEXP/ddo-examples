"""VMM experiment runner for the LTX 2.3 distilled modular BF16 T2I blocks.

This is intentionally separate from ``run_dynamic_modular_distilled.py``. It
keeps the text encoder on Diffusers group offload by default and routes the dense
transformer through DDO's experimental CUDA virtual-memory runtime. Explicit
--text-encoder-native-vmm also enables the separately validated encoder path.
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Resolve the independent ignored custom-block clone before allocator bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modular_diffusers"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_MODULES_CACHE", str((Path(__file__).parent / ".hf_modules").resolve()))

# This is a VMM-only harness: install the native allocator in the relaunched
# child before importing Torch, Diffusers, or any CUDA-owning module.
from ddo_vmm_bootstrap import ensure_vmm_fast_process, vmm_fast_requested

if not any(flag in sys.argv[1:] for flag in ("--help", "-h")) and vmm_fast_requested(default_backend="vmm"):
    ensure_vmm_fast_process()

import torch
from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers_dynamic_offloader import (
    DynamicOffloadSettings,
    enable_offload,
    format_dynamic_offload_presets,
    from_pretrained_with_dynamic_offload,
    is_wsl_environment,
    maybe_purge_windows_standby_cache,
    remove_dynamic_offload,
    resolve_dynamic_offload_backend,
)
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from custom_blocks.ltx2_image import LTX2ImageDistilledBlocks, LTX2ImageTextEncoderStep
from custom_blocks.ltx2_image.transformer_ltx2_image import LTX2ImageTransformer2DModel
from custom_blocks.ltx2_image.connectors_ltx2_image import LTX2ImageTextConnectors
from custom_blocks.ltx2_image.modular_blocks_ltx2_image import (
    LTX2ImageConnectorStep,
    LTX2ImageDenoiseStep,
    LTX2ImagePrepareLatentsStep,
)
from scripts.helpers.inference_utils import RunTracker, flush

import ddo_vmm_allocator

# Diagnostic paths may use the installed checkout; normal execution uses DDO discovery.
DDO_VMM_ALLOCATOR_ROOT = Path(ddo_vmm_allocator.__file__).resolve().parents[2]
from ddo_vmm_allocator.model_arena_adapter import attach_model_linear_modules
from ddo_vmm_allocator.native import load_native_vmm


class DiagnosticVmmRuntime:
    route = "diagnostic_vmm"
    hook = None

    def __init__(self, adapter, store, transformer, staging_pool=None, registered_host=None, report=None) -> None:
        self.adapter = adapter
        self.store = store
        self.transformer = transformer
        self.staging_pool = staging_pool
        self.registered_host = registered_host
        self.report = report
        self.input_handle = transformer.register_forward_pre_hook(self._move_inputs, with_kwargs=True)

        # The VMM adapter owns block parameters.  Root parameters and buffers
        # outside those blocks remain ordinary tensors and must be CUDA-resident
        # for the transformer's auxiliary projections and embeddings.
        for parameter in transformer.parameters():
            if parameter.device.type == "cpu":
                parameter.data = parameter.data.to(device=DEVICE)
        for buffer in transformer.buffers():
            if buffer.device.type == "cpu":
                buffer.data = buffer.data.to(device=DEVICE)

    @staticmethod
    def _move_inputs(_module, args, kwargs):
        def move(value):
            if torch.is_tensor(value) and value.device.type == "cpu":
                return value.to(device=DEVICE)
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value

        return move(args), move(kwargs)

    def profile_run(self):
        return nullcontext(None)

    def metrics_snapshot(self):
        setup = {}
        if self.report is not None:
            setup = {
                "attached_modules": self.report.attached_modules,
                "attached_tensors": self.report.attached_tensors,
                "execution_units": self.report.execution_units,
                "payload_bytes": self.report.payload_bytes,
                "virtual_bytes": self.report.virtual_bytes,
                "page_bytes": self.report.page_bytes,
            }
        return self.store.metrics_snapshot(setup=setup)

    def close(self) -> None:
        self.input_handle.remove()
        self.adapter.close()
        self.store.collect_completed_events(wait=True)
        self.store.close()
        if self.staging_pool is not None:
            self.staging_pool.close()
        if self.registered_host is not None:
            self.registered_host.close()

RUNNING_ON_WSL = False
DEVICE = "cuda:0"
OFFLOAD_DEVICE = "cpu"
DTYPE = torch.bfloat16

MODEL_TAG = "distilled_modular_vmm"
MODEL_PATH = os.getenv("MODEL_PATH", r"elismasilva/ltx2.3-image-distilled-1.1")
LOCAL_FILES_ONLY = False
TEXT_ENCODER_LOW_CPU_MEM_USAGE = True
MODEL_LOW_CPU_MEM_USAGE = True
DYNAMIC_OFFLOAD_SETTINGS = None
REQUESTED_DYNAMIC_OFFLOAD_PRESET = "one_shot_fast"
DYNAMIC_OFFLOAD_PRESET = "one_shot_fast"


def preset_env(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    value = DYNAMIC_OFFLOAD_SETTINGS.preset_value(name)
    if value != "":
        return value
    return default


def parse_bool_preset_env(name: str, default: str = "0") -> bool:
    return preset_env(name, default).strip().lower() in {"1", "true", "yes", "on"}


TEXT_ENCODER_GROUP_OFFLOAD = True
TEXT_ENCODER_DYNAMIC_OFFLOAD = False
TEXT_ENCODER_NATIVE_VMM = False
TRANSFORMER_MEMORY_MANAGER = "off"
DYNAMIC_OFFLOAD_CONFIG = None
DYNAMIC_OFFLOAD_EXECUTION_MODE = "vmm_fast_runtime"
VMM_RESIDENT_WEIGHT_BUDGET_GB = 2.0
VMM_STAGING_BUFFER_GB = 0.625
VMM_WINDOWS_FILE_READ_CHUNK_MB = 2
DYNAMIC_OFFLOAD_PIN_CPU_MEMORY = None
DYNAMIC_OFFLOAD_SHOW_PROFILE = False
DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY = None
PRE_VAE_CLEANUP_REPEATS = 1
RESET_DYNAMIC_MEMORY_AFTER_RUN = False
METRICS_LEVEL = 1
SHOW_METRICS = True
SAVE_METRICS = False
SHOW_DENOISE_STEPS = True
WIDTH = 1280
HEIGHT = 704
SEED = 43
NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 1.0
GUIDANCE_RESCALE = 0.7
DECODE_TIMESTEP = 0.0
DECODE_NOISE_SCALE = None
PAG_ENABLED = False
PAG_SCALE = 0.2
PAG_APPLIED_LAYERS = [28]
FAKE_PROMPT_EMBEDS = False
GENERATION_REPEATS = 1
TRANSFORMER_PREPARE_REPEATS = 1

prompt = (
    "Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, "
    "with bright colors, realistic fur detail, and playful viral-pet energy."
)
negative_prompt = ""


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


from scripts.helpers.runner_controls import parse_common_arguments, purge_for_runner, runner_settings_environ, record_runner_report, backend_reporting_fields
from scripts.helpers.ltx2.lora import add_lora_arguments, validate_lora_arguments, load_ltx_lora

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=LOCAL_FILES_ONLY,
        help="Use only already cached model artifacts; useful for reproducible offline benchmarks.",
    )
    parser.add_argument("--preset", default="one_shot_fast")
    parser.add_argument(
        "--ddo-backend",
        choices=("linear", "vmm"),
        default="vmm",
        help="DDO backend; this VMM harness defaults to vmm.",
    )
    parser.add_argument(
        "--ddo-preset",
        default="auto",
        help="DDO backend preset; VMM accepts auto or nvme_baseline.",
    )
    parser.add_argument(
        "--ddo-public-native-runtime", action=argparse.BooleanOptionalAction, default=None,
        help="Use the public DDO native API (default for native fast unless the diagnostic adapter is selected).",
    )
    parser.add_argument(
        "--ddo-vmm-source-cache-gb", type=float, default=0.0,
        help="Native persistent host-storage ceiling in GiB; 0 disables retention (not WDDM GPU residency).",
    )
    parser.add_argument(
        "--ddo-vmm-source-registered-gb", type=float, default=0.0,
        help="CUDA registration ceiling within persistent storage; 0 uses its ceiling. Multiples of 64 MiB.",
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--negative-prompt", default=negative_prompt)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--guidance-rescale", type=float, default=GUIDANCE_RESCALE)
    parser.add_argument("--decode-timestep", type=float, default=DECODE_TIMESTEP)
    parser.add_argument("--decode-noise-scale", type=parse_optional_float, default=DECODE_NOISE_SCALE)
    parser.add_argument("--pag", action=argparse.BooleanOptionalAction, default=PAG_ENABLED)
    parser.add_argument("--pag-scale", type=float, default=PAG_SCALE)
    parser.add_argument("--pag-layers", default=",".join(str(layer) for layer in PAG_APPLIED_LAYERS))
    parser.add_argument("--fake-prompt", action=argparse.BooleanOptionalAction, default=FAKE_PROMPT_EMBEDS)
    parser.add_argument("--text-encoder-native-vmm", action="store_true",
                        help="Opt-in native encoder phase; unload before transformer loading. Requires public VMM runtime and real prompt.")
    parser.add_argument("--generation-repeats", type=int, default=GENERATION_REPEATS)
    parser.add_argument("--transformer-prepare-repeats", type=int, default=TRANSFORMER_PREPARE_REPEATS)
    parser.add_argument("--ddo-profile", default=None, help="Named DDO denoise workload profile.")
    parser.add_argument(
        "--build-ddo-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calibrate the named DDO profile with a zero-resident denoise run.",
    )
    parser.add_argument(
        "--ddo-profile-steps",
        type=int,
        default=2,
        help="Denoise steps used only by --build-ddo-profile (default: 2).",
    )
    parser.add_argument("--pre-vae-cleanup-repeats", type=int, default=None)
    parser.add_argument("--vae-leaf-offload", action="store_true",
                        help="Opt in to no-stream leaf VAE decode; preserve resident decode otherwise.")
    parser.add_argument("--output-dir", default="outputs/ltx_image_modular_vmm")
    parser.add_argument(
        "--vmm-resident-weight-budget-gb",
        type=float,
        default=VMM_RESIDENT_WEIGHT_BUDGET_GB,
        help="Maximum CUDA physical backing for mapped transformer weights (default: 2.0 GiB).",
    )
    parser.add_argument(
        "--vmm-staging-buffer-gb",
        type=float,
        default=VMM_STAGING_BUFFER_GB,
        help="Size of each bounded pinned staging slot; VMM uses two slots for current/next prefetch overlap (default: 0.625 GiB).",
    )
    parser.add_argument(
        "--vmm-windows-file-read-chunk-mb",
        type=int,
        default=VMM_WINDOWS_FILE_READ_CHUNK_MB,
        help="Chunk size for overlapped checkpoint reads into each VMM staging slot (default: 2 MiB).",
    )
    parser.add_argument(
        "--vmm-virtual-host-transport",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Windows VirtualAlloc staging slots registered once with CUDA for direct VMM H2D transfers.",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-register-source-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Register existing CPU weight storage once and DMA directly to device-backed VMM pages.",
    )
    parser.add_argument(
        "--vmm-host-cache-gb",
        type=float,
        default=0.0,
        help="Budget for persistent registered CPU weight pages; 0 retains bounded file transport only (default: 0).",
    )
    parser.add_argument(
        "--vmm-pin-source-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pin all CPU transformer source weights before denoise; off by default because setup is expensive.",
    )
    parser.add_argument(
        "--vmm-fault-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep successfully faulted VMM groups resident; use temporary CUDA storage for later groups under the VMM budget.",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-adapter",
        action="store_true",
        help="Use the DDO diagnostic VMM block adapter for the transformer instead of DDO runtime.",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-pressure-driven-residency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the native allocator and DXGI budget guard determine transformer residency; requires the fresh-process allocator launcher.",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-dxgi-budget-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Protect device/shared/system reserves using live DXGI budgets before native VMM materialization (Windows only).",
    )
    parser.add_argument(
        "--ddo-vmm-dxgi-local-headroom-gb",
        type=float,
        default=0.0,
        help="VRAM reserve retained below the WDDM local budget; 0 disables this individual limit.",
    )
    parser.add_argument(
        "--ddo-vmm-dxgi-shared-headroom-gb",
        type=float,
        default=4.0,
        help="Shared-memory reserve retained below the WDDM non-local budget (default: 4 GiB).",
    )
    parser.add_argument(
        "--ddo-vmm-dxgi-system-headroom-gb",
        type=float,
        default=4.0,
        help="Physical/commit RAM reserve retained before a native VMM mapping (default: 4 GiB).",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-library",
        default=str(DDO_VMM_ALLOCATOR_ROOT / "build" / "Release" / "ddo_vmm.dll"),
        help="Path to the DDO VMM native library.",
    )
    parser.add_argument(
        "--ddo-vmm-module-pattern",
        default=r"transformer_blocks\.\d+$",
        help="Regex selecting transformer blocks for the diagnostic adapter.",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-leaf-residency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use direct-parameter transformer modules as VMM groups for finer residency granularity.",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-backing",
        choices=("device", "host"),
        default="device",
        help="Physical VMM backing for diagnostic groups; host enables WDDM/shared-memory probing.",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-virtual-multiplier",
        type=float,
        default=1.0,
        help="Virtual model-address span multiplier over attached BF16 weight payload (default: 1).",
    )
    parser.add_argument(
        "--ddo-vmm-page-mb",
        type=int,
        default=32,
        help="Native VMM physical page size in MiB (default: 32).",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-pin-source",
        action="store_true",
        help="Pin CPU source weights and use non-blocking VMM copies (experimental; uses locked host memory).",
    )
    parser.add_argument(
        "--ddo-vmm-diagnostic-pin-source-gb",
        type=float,
        default=0.0,
        help="Upper bound for pinned CPU source weights; 0 means unlimited when --ddo-vmm-diagnostic-pin-source is set.",
    )
    parser.add_argument("--metrics-level", type=int, choices=(0, 1, 2), default=METRICS_LEVEL)
    parser.add_argument(
        "--ddo-gpu-timing", action="store_true",
        help="Opt-in DDO compute-stream event timing and completed-ticket collection after readback; public native route only.",
    )
    parser.add_argument(
        "--memory-monitor", choices=("full", "rss"), default="full",
        help="RAM sampling: full preserves USS on Windows; rss avoids repeated full-memory scans. RSS is not USS. GPU sampling is unchanged.",
    )
    parser.add_argument(
        "--show-ddo-profile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print the DDO transport/profile summary after denoise.",
    )
    parser.add_argument("--show-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-denoise-steps", action=argparse.BooleanOptionalAction, default=SHOW_DENOISE_STEPS)
    parser.add_argument("--ddo-native-timeline", action="store_true",
                        help="Opt-in native stream windows (not pure kernel/DMA activity); requires --ddo-gpu-timing.")
    parser.add_argument("--ddo-diagnostic-step-wait", action="store_true",
                        help="Diagnostic only: wait for each compute-step event; requires --ddo-native-timeline. Not a benchmark.")
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=MODEL_LOW_CPU_MEM_USAGE)
    parser.add_argument("--text-encoder-low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=TEXT_ENCODER_LOW_CPU_MEM_USAGE)
    parser.add_argument("--text-encoder-group-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--text-encoder-dynamic-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resident-module-budget-gb", type=float, default=None)
    parser.add_argument("--max-resident-module-budget-gb", type=float, default=None)
    parser.add_argument("--pin-weight-budget-gb", type=float, default=None)
    parser.add_argument("--reset-dynamic-memory-after-run", action=argparse.BooleanOptionalAction, default=RESET_DYNAMIC_MEMORY_AFTER_RUN)
    parser.add_argument("--print-presets", action="store_true")
    add_lora_arguments(parser)
    args = parse_common_arguments(parser)
    validate_lora_arguments(parser, args)
    return args


def build_run_slug(seed):
    pag_tag = f"pag{PAG_SCALE:g}_layers{'-'.join(map(str, PAG_APPLIED_LAYERS))}" if PAG_ENABLED else "nopag"
    return "_".join(
        [
            "ltx23_image",
            MODEL_TAG,
            "bf16",
            f"vmm{VMM_RESIDENT_WEIGHT_BUDGET_GB:g}gb",
            "text_encoder_native" if TEXT_ENCODER_NATIVE_VMM else "text_encoder_original",
            pag_tag,
            f"{WIDTH}x{HEIGHT}",
            f"steps{NUM_INFERENCE_STEPS}",
            f"seed{seed}",
        ]
    )


def cleanup_runtime_state(record_event, event_name: str, *, repeats: int = 1, collect_cuda_ipc: bool = False) -> None:
    if repeats <= 0:
        return

    before_free_gb = before_total_gb = None
    after_free_gb = after_total_gb = None
    if torch.cuda.is_available():
        before_free, before_total = torch.cuda.mem_get_info(DEVICE)
        before_free_gb = before_free / 1024**3
        before_total_gb = before_total / 1024**3

    event_t0 = time.time()
    for _ in range(repeats):
        flush()
        if collect_cuda_ipc and torch.cuda.is_available() and hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    if torch.cuda.is_available():
        after_free, after_total = torch.cuda.mem_get_info(DEVICE)
        after_free_gb = after_free / 1024**3
        after_total_gb = after_total / 1024**3

    record_event(
        event_name,
        time.time() - event_t0,
        repeats=repeats,
        collect_cuda_ipc=collect_cuda_ipc,
        before_free_vram_gb=None if before_free_gb is None else round(before_free_gb, 4),
        before_total_vram_gb=None if before_total_gb is None else round(before_total_gb, 4),
        after_free_vram_gb=None if after_free_gb is None else round(after_free_gb, 4),
        after_total_vram_gb=None if after_total_gb is None else round(after_total_gb, 4),
    )


def cleanup_before_vae_decode(record_event) -> None:
    cleanup_runtime_state(
        record_event,
        "cleanup_before_vae_decode",
        repeats=PRE_VAE_CLEANUP_REPEATS,
        collect_cuda_ipc=True,
    )


def denoise_progress_callback(components, step_index, timestep, callback_kwargs):
    timer = getattr(denoise_progress_callback, "execution_timer", None)
    if timer is not None:
        timer.mark_step()
    now = time.perf_counter()
    last_time = getattr(denoise_progress_callback, "last_time", now)
    start_time = getattr(denoise_progress_callback, "start_time", last_time)
    step_elapsed = now - last_time
    total_elapsed = now - start_time
    denoise_progress_callback.last_time = now
    denoise_progress_callback.step_times.append(step_elapsed)
    displayed_step = len(denoise_progress_callback.step_times)

    try:
        used_gb = torch.cuda.memory_allocated(DEVICE) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(DEVICE) / 1024**3
        allocator_metrics = f"torch_alloc={used_gb:.2f} GiB torch_reserved={reserved_gb:.2f} GiB"
    except RuntimeError as error:
        # PyTorch's pluggable allocator API currently does not implement
        # getDeviceStats.  Timing remains valid, but allocator counters are
        # unavailable for that opt-in process.
        if "CUDAPluggableAllocator" not in str(error):
            raise
        allocator_metrics = "torch_alloc=n/a torch_reserved=n/a"
    total_steps = len(denoise_progress_callback.timesteps)
    avg_elapsed = total_elapsed / displayed_step
    if SHOW_DENOISE_STEPS:
        print(
            f"  [denoise] step {displayed_step}/{total_steps} timestep={denoise_progress_callback.display_timesteps[step_index]:.4f} "
            f"elapsed={step_elapsed:.4f}s avg={avg_elapsed:.4f}s/it "
            f"{allocator_metrics}",
            flush=True,
        )
    return callback_kwargs


def main():
    global RUNNING_ON_WSL, DEVICE, MODEL_PATH, LOCAL_FILES_ONLY, TEXT_ENCODER_LOW_CPU_MEM_USAGE, MODEL_LOW_CPU_MEM_USAGE
    global DYNAMIC_OFFLOAD_SETTINGS, REQUESTED_DYNAMIC_OFFLOAD_PRESET, DYNAMIC_OFFLOAD_PRESET
    global TEXT_ENCODER_GROUP_OFFLOAD, TEXT_ENCODER_DYNAMIC_OFFLOAD, TEXT_ENCODER_NATIVE_VMM, TRANSFORMER_MEMORY_MANAGER
    global DYNAMIC_OFFLOAD_CONFIG, DYNAMIC_OFFLOAD_EXECUTION_MODE, DYNAMIC_OFFLOAD_PIN_CPU_MEMORY
    global VMM_RESIDENT_WEIGHT_BUDGET_GB, VMM_STAGING_BUFFER_GB, VMM_WINDOWS_FILE_READ_CHUNK_MB
    global DYNAMIC_OFFLOAD_SHOW_PROFILE, DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY
    global PRE_VAE_CLEANUP_REPEATS, RESET_DYNAMIC_MEMORY_AFTER_RUN
    global METRICS_LEVEL, SHOW_METRICS, SAVE_METRICS, SHOW_DENOISE_STEPS
    global WIDTH, HEIGHT, SEED, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, GUIDANCE_RESCALE
    global DECODE_TIMESTEP, DECODE_NOISE_SCALE, PAG_ENABLED, PAG_SCALE, PAG_APPLIED_LAYERS
    global FAKE_PROMPT_EMBEDS, GENERATION_REPEATS, TRANSFORMER_PREPARE_REPEATS
    global prompt, negative_prompt

    args = parse_args()
    if args.ddo_public_native_runtime is None:
        args.ddo_public_native_runtime = (
            args.ddo_backend == "vmm"
            and args.ddo_preset in ("auto", "vmm_fast")
            and not args.ddo_vmm_diagnostic_adapter
        )
    if args.text_encoder_native_vmm:
        if not args.ddo_public_native_runtime or args.fake_prompt:
            raise SystemExit("--text-encoder-native-vmm requires --ddo-public-native-runtime and real prompt")
        if args.text_encoder_group_offload is True or args.text_encoder_dynamic_offload is False:
            raise SystemExit("Native encoder cannot be combined with explicit encoder compatibility controls")
    TEXT_ENCODER_NATIVE_VMM = args.text_encoder_native_vmm
    if args.ddo_diagnostic_step_wait and not args.ddo_native_timeline:
        raise ValueError("--ddo-diagnostic-step-wait requires --ddo-native-timeline")
    if args.ddo_native_timeline and not args.ddo_gpu_timing:
        raise ValueError("--ddo-native-timeline requires --ddo-gpu-timing")
    if args.ddo_gpu_timing and not args.ddo_public_native_runtime:
        raise ValueError("--ddo-gpu-timing requires --ddo-public-native-runtime")
    try:
        ddo_backend = resolve_dynamic_offload_backend(
            args.ddo_backend,
            args.ddo_preset,
            running_on_wsl=is_wsl_environment(),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.ddo_public_native_runtime:
        if ddo_backend.execution_mode != "vmm_fast_runtime":
            raise SystemExit("--ddo-public-native-runtime requires --ddo-backend vmm --ddo-preset auto")
        if args.ddo_vmm_diagnostic_adapter:
            raise SystemExit("Choose the public DDO runtime or --ddo-vmm-diagnostic-adapter, not both")
        if args.vmm_host_cache_gb or args.vmm_pin_source_weights or args.ddo_vmm_diagnostic_register_source_weights:
            raise SystemExit("Public native source-cache execution must not retain/register loader CPU weights")
        if args.ddo_vmm_diagnostic_backing != "device" or args.ddo_vmm_diagnostic_leaf_residency or args.ddo_vmm_diagnostic_virtual_multiplier != 1.0:
            raise SystemExit("Public native runtime does not accept diagnostic backing/leaf/span overrides")
    elif args.ddo_vmm_source_cache_gb or args.ddo_vmm_source_registered_gb:
        raise SystemExit("--ddo-vmm-source-cache-gb requires --ddo-public-native-runtime in this harness")
    if not math.isfinite(args.ddo_vmm_source_cache_gb) or not 0 <= args.ddo_vmm_source_cache_gb <= 32:
        raise SystemExit("--ddo-vmm-source-cache-gb must be finite and between 0 and 32 GiB")
    if not math.isfinite(args.ddo_vmm_source_registered_gb) or not 0 <= args.ddo_vmm_source_registered_gb <= args.ddo_vmm_source_cache_gb:
        raise SystemExit("--ddo-vmm-source-registered-gb must be finite, non-negative and <= source storage ceiling")

    if args.build_ddo_profile and not args.ddo_profile:
        raise SystemExit("--build-ddo-profile requires --ddo-profile NAME")

    RUNNING_ON_WSL = is_wsl_environment()
    if args.print_presets:
        print(format_dynamic_offload_presets(default_preset=args.preset, running_on_wsl=RUNNING_ON_WSL))
        return

    DEVICE = args.device
    MODEL_PATH = args.model_path
    LOCAL_FILES_ONLY = args.local_files_only
    TEXT_ENCODER_LOW_CPU_MEM_USAGE = args.text_encoder_low_cpu_mem_usage
    MODEL_LOW_CPU_MEM_USAGE = args.low_cpu_mem_usage

    settings_environ = runner_settings_environ(args)
    if args.resident_module_budget_gb is not None:
        settings_environ["DDO_RESIDENT_MODULE_BUDGET_GB"] = str(args.resident_module_budget_gb)
    if args.max_resident_module_budget_gb is not None:
        settings_environ["DDO_MAX_RESIDENT_MODULE_BUDGET_GB"] = str(args.max_resident_module_budget_gb)
    if args.pin_weight_budget_gb is not None:
        settings_environ["DDO_PIN_WEIGHT_BUDGET_GB"] = str(args.pin_weight_budget_gb)
    if args.vmm_resident_weight_budget_gb <= 0:
        raise SystemExit("--vmm-resident-weight-budget-gb must be positive")
    if args.vmm_staging_buffer_gb <= 0:
        raise SystemExit("--vmm-staging-buffer-gb must be positive")
    if args.vmm_windows_file_read_chunk_mb <= 0:
        raise SystemExit("--vmm-windows-file-read-chunk-mb must be positive")
    if args.vmm_host_cache_gb < 0:
        raise SystemExit("--vmm-host-cache-gb must be non-negative")
    if args.ddo_vmm_diagnostic_pressure_driven_residency and not args.ddo_vmm_diagnostic_adapter:
        raise SystemExit("--ddo-vmm-diagnostic-pressure-driven-residency requires --ddo-vmm-diagnostic-adapter")
    if args.ddo_vmm_diagnostic_pressure_driven_residency and not os.environ.get("DDO_VMM_LIBRARY_PATH"):
        raise SystemExit(
            "--ddo-vmm-diagnostic-pressure-driven-residency requires the fresh-process allocator launcher"
        )
    if args.ddo_vmm_diagnostic_pressure_driven_residency and not args.ddo_vmm_diagnostic_dxgi_budget_guard:
        raise SystemExit(
            "--ddo-vmm-diagnostic-pressure-driven-residency requires "
            "--ddo-vmm-diagnostic-dxgi-budget-guard for Windows safety"
        )
    if args.ddo_vmm_diagnostic_dxgi_budget_guard and not args.ddo_vmm_diagnostic_adapter:
        raise SystemExit("--ddo-vmm-diagnostic-dxgi-budget-guard requires --ddo-vmm-diagnostic-adapter")
    for flag, value in (
        ("--ddo-vmm-dxgi-local-headroom-gb", args.ddo_vmm_dxgi_local_headroom_gb),
        ("--ddo-vmm-dxgi-shared-headroom-gb", args.ddo_vmm_dxgi_shared_headroom_gb),
        ("--ddo-vmm-dxgi-system-headroom-gb", args.ddo_vmm_dxgi_system_headroom_gb),
    ):
        if value < 0:
            raise SystemExit(f"{flag} must be non-negative")
    settings_environ["DDO_EXECUTION_MODE"] = ddo_backend.execution_mode
    settings_environ["DDO_VMM_RESIDENT_WEIGHT_BUDGET_GB"] = str(args.vmm_resident_weight_budget_gb)
    settings_environ["DDO_VMM_STAGING_BUFFER_GB"] = str(args.vmm_staging_buffer_gb)
    settings_environ["DDO_VMM_WINDOWS_FILE_READ_CHUNK_MB"] = str(args.vmm_windows_file_read_chunk_mb)
    settings_environ["DDO_VMM_VIRTUAL_HOST_TRANSPORT"] = "1" if args.vmm_virtual_host_transport else "0"
    settings_environ["DDO_VMM_HOST_CACHE_GB"] = str(args.vmm_host_cache_gb)
    if args.show_ddo_profile is not None:
        settings_environ["DDO_SHOW_PROFILE"] = "1" if args.show_ddo_profile else "0"
    settings_environ["DDO_VMM_PIN_SOURCE_WEIGHTS"] = "1" if args.vmm_pin_source_weights else "0"
    settings_environ["DDO_VMM_FAULT_FALLBACK"] = "1" if args.vmm_fault_fallback else "0"

    settings_kwargs = {
        "execution_device": DEVICE,
        "offload_device": OFFLOAD_DEVICE,
        "running_on_wsl": RUNNING_ON_WSL,
        "default_preset": args.preset,
        "environ": settings_environ,
    }

    DYNAMIC_OFFLOAD_SETTINGS = DynamicOffloadSettings.from_env(**settings_kwargs)
    if args.ddo_public_native_runtime:
        # The diagnostic adapter uses three slots of the host's staging value.
        # DDO accepts TOTAL capacity: preserve those bytes explicitly, not 1/3.
        public_config = replace(
            DYNAMIC_OFFLOAD_SETTINGS.config,
            execution_mode=ddo_backend.execution_mode,
            vmm_native_transport=True,
            vmm_staging_buffer_gb=args.vmm_staging_buffer_gb * 3,
            vmm_source_cache_gb=args.ddo_vmm_source_cache_gb,
            vmm_source_registered_gb=args.ddo_vmm_source_registered_gb,
            vmm_native_source_cache_slot_mb=0,
            vmm_native_source_cache_slots=0,
            vmm_module_pattern=args.ddo_vmm_module_pattern,
            vmm_page_mb=args.ddo_vmm_page_mb,
            vmm_dxgi_local_headroom_gb=args.ddo_vmm_dxgi_local_headroom_gb,
            vmm_dxgi_shared_headroom_gb=args.ddo_vmm_dxgi_shared_headroom_gb,
            vmm_dxgi_system_headroom_gb=args.ddo_vmm_dxgi_system_headroom_gb,
        )
        DYNAMIC_OFFLOAD_SETTINGS = replace(DYNAMIC_OFFLOAD_SETTINGS, config=public_config, enabled=True)
    REQUESTED_DYNAMIC_OFFLOAD_PRESET = DYNAMIC_OFFLOAD_SETTINGS.requested_preset
    DYNAMIC_OFFLOAD_PRESET = DYNAMIC_OFFLOAD_SETTINGS.effective_preset
    DYNAMIC_OFFLOAD_CONFIG = DYNAMIC_OFFLOAD_SETTINGS.config
    DYNAMIC_OFFLOAD_EXECUTION_MODE = DYNAMIC_OFFLOAD_CONFIG.execution_mode
    DYNAMIC_OFFLOAD_PIN_CPU_MEMORY = DYNAMIC_OFFLOAD_SETTINGS.requested_pin_cpu_memory
    DYNAMIC_OFFLOAD_SHOW_PROFILE = DYNAMIC_OFFLOAD_CONFIG.show_profile
    DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY = DYNAMIC_OFFLOAD_SETTINGS.effective_pin_cpu_memory
    VMM_RESIDENT_WEIGHT_BUDGET_GB = DYNAMIC_OFFLOAD_CONFIG.vmm_resident_weight_budget_gb
    VMM_STAGING_BUFFER_GB = DYNAMIC_OFFLOAD_CONFIG.vmm_staging_buffer_gb
    VMM_WINDOWS_FILE_READ_CHUNK_MB = DYNAMIC_OFFLOAD_CONFIG.vmm_windows_file_read_chunk_mb

    TEXT_ENCODER_GROUP_OFFLOAD = (
        args.text_encoder_group_offload
        if args.text_encoder_group_offload is not None
        else parse_bool_preset_env("DDO_RUNNER_TEXT_ENCODER_GROUP_OFFLOAD", "1")
    )
    TEXT_ENCODER_DYNAMIC_OFFLOAD = (
        args.text_encoder_dynamic_offload
        if args.text_encoder_dynamic_offload is not None
        else parse_bool_preset_env("DDO_RUNNER_TEXT_ENCODER_DYNAMIC_OFFLOAD")
    )
    TRANSFORMER_MEMORY_MANAGER = preset_env("DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER", "off").lower()
    PRE_VAE_CLEANUP_REPEATS = (
        args.pre_vae_cleanup_repeats
        if args.pre_vae_cleanup_repeats is not None
        else int(preset_env("DDO_RUNNER_PRE_VAE_CLEANUP_REPEATS", "3" if RUNNING_ON_WSL else "1"))
    )
    RESET_DYNAMIC_MEMORY_AFTER_RUN = args.reset_dynamic_memory_after_run
    METRICS_LEVEL = args.metrics_level
    SHOW_METRICS = args.show_metrics if args.show_metrics is not None else METRICS_LEVEL >= 1
    SAVE_METRICS = args.save_metrics if args.save_metrics is not None else METRICS_LEVEL >= 2
    SHOW_DENOISE_STEPS = args.show_denoise_steps
    WIDTH = args.width
    HEIGHT = args.height
    SEED = args.seed
    NUM_INFERENCE_STEPS = args.steps
    active_inference_steps = (
        max(1, min(NUM_INFERENCE_STEPS, args.ddo_profile_steps))
        if args.build_ddo_profile
        else NUM_INFERENCE_STEPS
    )
    GUIDANCE_SCALE = args.guidance_scale
    GUIDANCE_RESCALE = args.guidance_rescale
    DECODE_TIMESTEP = args.decode_timestep
    DECODE_NOISE_SCALE = args.decode_noise_scale
    PAG_ENABLED = args.pag
    PAG_SCALE = args.pag_scale
    PAG_APPLIED_LAYERS = [int(item.strip()) for item in args.pag_layers.split(",") if item.strip()]
    FAKE_PROMPT_EMBEDS = args.fake_prompt
    GENERATION_REPEATS = max(1, args.generation_repeats)
    TRANSFORMER_PREPARE_REPEATS = max(1, args.transformer_prepare_repeats)
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    seed = SEED or torch.randint(0, 2**32, (1,)).item()
    if not SEED:
        print(f"Using random seed: {seed}")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    run_slug = build_run_slug(seed)
    if args.ddo_public_native_runtime:
        run_slug += f"_ddo_public_cache{args.ddo_vmm_source_cache_gb:g}gb"
        if args.ddo_vmm_source_registered_gb:
            run_slug += f"_registered{args.ddo_vmm_source_registered_gb:g}gb"
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    run_metrics = {
        "run_slug": run_slug,
        "model_tag": MODEL_TAG,
        "model_path": MODEL_PATH,
        "width": WIDTH,
        "height": HEIGHT,
        "seed": seed,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "active_inference_steps": active_inference_steps,
        "ddo_profile_name": args.ddo_profile,
        "build_ddo_profile": args.build_ddo_profile,
        "generation_repeats": GENERATION_REPEATS,
        "transformer_prepare_repeats": TRANSFORMER_PREPARE_REPEATS,
        "guidance_scale": GUIDANCE_SCALE,
        "guidance_rescale": GUIDANCE_RESCALE,
        "vae_decode_timestep": DECODE_TIMESTEP,
        "vae_decode_noise_scale": DECODE_NOISE_SCALE,
        "pag_enabled": PAG_ENABLED,
        "pag_scale": PAG_SCALE if PAG_ENABLED else 0.0,
        "pag_applied_layers": PAG_APPLIED_LAYERS if PAG_ENABLED else None,
        "dtype": str(DTYPE),
        "text_encoder_low_cpu_mem_usage": TEXT_ENCODER_LOW_CPU_MEM_USAGE,
        "text_encoder_dynamic_offload": TEXT_ENCODER_DYNAMIC_OFFLOAD,
        "text_encoder_group_offload": TEXT_ENCODER_GROUP_OFFLOAD,
        "text_encoder_native_vmm": TEXT_ENCODER_NATIVE_VMM,
        "model_low_cpu_mem_usage": MODEL_LOW_CPU_MEM_USAGE,
        "running_on_wsl": RUNNING_ON_WSL,
        "reset_dynamic_memory_after_run": RESET_DYNAMIC_MEMORY_AFTER_RUN,
        "metrics_level": METRICS_LEVEL,
        "show_metrics": SHOW_METRICS,
        "save_metrics": SAVE_METRICS,
        "transformer_memory_manager": TRANSFORMER_MEMORY_MANAGER,
        "vmm_resident_weight_budget_gb": VMM_RESIDENT_WEIGHT_BUDGET_GB,
        "vmm_staging_buffer_gb": VMM_STAGING_BUFFER_GB,
        "vmm_windows_file_read_chunk_mb": VMM_WINDOWS_FILE_READ_CHUNK_MB,
        **DYNAMIC_OFFLOAD_SETTINGS.as_metrics(),
        **backend_reporting_fields(ddo_backend, DYNAMIC_OFFLOAD_SETTINGS),
        "pre_vae_cleanup_repeats": PRE_VAE_CLEANUP_REPEATS,
        "events": [],
        "steps": [],
    }

    print(
        f"Using DDO backend: {ddo_backend.backend}; preset: "
        f"{ddo_backend.requested_preset} -> {ddo_backend.effective_preset}; "
        f"runtime: {DYNAMIC_OFFLOAD_EXECUTION_MODE}; "
        f"base settings preset: {DYNAMIC_OFFLOAD_PRESET}",
        flush=True,
    )
    if DYNAMIC_OFFLOAD_PIN_CPU_MEMORY and not DYNAMIC_OFFLOAD_EFFECTIVE_PIN_CPU_MEMORY:
        print("  [dynamic-offload] disabling pinned CPU memory on WSL; set DDO_DISABLE_PIN_ON_WSL=0 to force it.", flush=True)
    text_encoder_route = (
        "dynamic_offload"
        if TEXT_ENCODER_NATIVE_VMM or TEXT_ENCODER_DYNAMIC_OFFLOAD
        else "group_offload"
        if TEXT_ENCODER_GROUP_OFFLOAD
        else "cuda_to"
    )
    transformer_route = "dynamic_offload" if DYNAMIC_OFFLOAD_SETTINGS.enabled else "standard"
    print(
        "  [runner] "
        f"text_encoder_route={text_encoder_route} "
        f"text_encoder_group_offload_requested={TEXT_ENCODER_GROUP_OFFLOAD} "
        f"text_encoder_dynamic_offload_requested={TEXT_ENCODER_DYNAMIC_OFFLOAD} "
        f"text_encoder_native_vmm={TEXT_ENCODER_NATIVE_VMM} "
        f"transformer_route={transformer_route} "
        f"vmm_weight_budget_gb={VMM_RESIDENT_WEIGHT_BUDGET_GB:g} "
        f"vmm_staging_buffer_gb={VMM_STAGING_BUFFER_GB:g} "
        f"vmm_windows_file_read_chunk_mb={VMM_WINDOWS_FILE_READ_CHUNK_MB} "
        f"preset={ddo_backend.effective_preset} "
        f"runtime={DYNAMIC_OFFLOAD_EXECUTION_MODE}",
        flush=True,
    )

    tracker = RunTracker(DEVICE, run_metrics, interval=0.1, show_metrics=SHOW_METRICS,
                         memory_monitor=args.memory_monitor)
    record_event = tracker.record_event
    step_start = tracker.step_start
    step_end = tracker.step_end

    run_metrics["purge_windows_standby_before_run"] = purge_for_runner(args,
        DYNAMIC_OFFLOAD_SETTINGS,
        "before_run",
        record_event=record_event,
    )

    t0 = step_start("Pass 0: Encode prompts")
    if FAKE_PROMPT_EMBEDS:
        prompt_embeds = torch.zeros((1, 1024, 188160), dtype=DTYPE, device=OFFLOAD_DEVICE)
        prompt_attention_mask = torch.ones((1, 1024), dtype=torch.long, device=OFFLOAD_DEVICE)
        record_event("fake_prompt_embeds", 0.0, shape=list(prompt_embeds.shape))
    else:
        event_t0 = time.time()
        if TEXT_ENCODER_NATIVE_VMM:
            text_encoder_load = from_pretrained_with_dynamic_offload(
                MODEL_PATH, model_loader=Gemma3ForConditionalGeneration,
                subfolder="text_encoder", torch_dtype=DTYPE,
                low_cpu_mem_usage=TEXT_ENCODER_LOW_CPU_MEM_USAGE,
                local_files_only=LOCAL_FILES_ONLY, apply_dynamic=False,
            )
            text_encoder = text_encoder_load.module.eval()
            record_runner_report(args, tracker, text_encoder_load.inspect_vmm_component(component="text_encoder"),
                                       "text_encoder_preflight")
            del text_encoder_load
        else:
            text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                subfolder="text_encoder",
                torch_dtype=DTYPE,
                low_cpu_mem_usage=TEXT_ENCODER_LOW_CPU_MEM_USAGE,
                local_files_only=LOCAL_FILES_ONLY,
            )
        record_event("load_text_encoder", time.time() - event_t0, source=MODEL_PATH)

        event_t0 = time.time()
        encoder_route_kwargs = {}
        if TEXT_ENCODER_NATIVE_VMM:
            encoder_route_kwargs = dict(
                route="dynamic_offload",
                config=replace(DYNAMIC_OFFLOAD_CONFIG, vmm_module_pattern=".*", vmm_page_mb=2,
                               vmm_component_tensor_ownership=True, vmm_component_buffer_limit_mb=1,
                               profile_name=None, build_profile=False),
            )
        text_encoder_offload = enable_offload(
            text_encoder,
            settings=DYNAMIC_OFFLOAD_SETTINGS,
            component="text_encoder",
            group_offload=TEXT_ENCODER_GROUP_OFFLOAD,
            dynamic_offload=TEXT_ENCODER_DYNAMIC_OFFLOAD,
            execution_device=DEVICE,
            offload_device=OFFLOAD_DEVICE,
            low_cpu_mem_usage=TEXT_ENCODER_LOW_CPU_MEM_USAGE,
            record_event=record_event,
            dynamic_event_name="setup_text_encoder_dynamic_offload",
            group_event_name="setup_text_encoder_group_offload",
            **encoder_route_kwargs,
        )
        text_encoder_dynamic_offload_hook = text_encoder_offload.hook
        run_metrics["text_encoder_offload_route"] = text_encoder_offload.route

        event_t0 = time.time()
        tokenizer = GemmaTokenizerFast.from_pretrained(
            MODEL_PATH, subfolder="tokenizer", local_files_only=LOCAL_FILES_ONLY
        )
        record_event("load_tokenizer", time.time() - event_t0, source=MODEL_PATH)

        event_t0 = time.time()
        prompt_pipe = LTX2ImageTextEncoderStep().init_pipeline()
        prompt_pipe.update_components(text_encoder=text_encoder, tokenizer=tokenizer)
        record_event("build_prompt_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

        event_t0 = time.time()
        try:
            with torch.inference_mode():
                prompt_state = prompt_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    output=["prompt_embeds", "prompt_attention_mask"],
                )
            if TEXT_ENCODER_NATIVE_VMM:
                record_runner_report(args, tracker, text_encoder_offload.metrics_snapshot(collect_completed=True), "text_encoder")
        finally:
            if TEXT_ENCODER_NATIVE_VMM:
                encoder_failure = sys.exc_info()[1]
                try:
                    remove_dynamic_offload(text_encoder)
                    text_encoder_dynamic_offload_hook = None
                except BaseException as cleanup_error:
                    if encoder_failure is None:
                        raise
                    encoder_failure.add_note(f"Native encoder cleanup also failed: {cleanup_error!r}")
        record_event("encode_prompt_call", time.time() - event_t0, classifier_free_guidance=False)

        prompt_embeds = prompt_state["prompt_embeds"].to(OFFLOAD_DEVICE)
        prompt_attention_mask = prompt_state["prompt_attention_mask"].to(OFFLOAD_DEVICE)
        if text_encoder_dynamic_offload_hook is not None:
            run_metrics["text_encoder_dynamic_offload_runtime_summary"] = text_encoder_dynamic_offload_hook.state.as_dict()
            if DYNAMIC_OFFLOAD_SHOW_PROFILE:
                text_encoder_dynamic_offload_hook.print_profile_summary()
            remove_dynamic_offload(text_encoder)
            text_encoder_dynamic_offload_hook = None
        del text_encoder_offload
        del prompt_state
        del prompt_pipe, text_encoder, tokenizer
        cleanup_runtime_state(record_event, "cleanup_after_text_encoder")
        run_metrics["purge_windows_standby_after_text_encoder"] = purge_for_runner(args,
            DYNAMIC_OFFLOAD_SETTINGS,
            "after_text_encoder",
            record_event=record_event,
        )

    if SHOW_METRICS:
        print(f"  prompt_embeds shape: {prompt_embeds.shape}")
    step_end("Pass 0: Encode prompts", t0)

    t0 = step_start(f"Pass 1: Generate at {WIDTH}x{HEIGHT}")

    event_t0 = time.time()
    connectors = LTX2ImageTextConnectors.from_pretrained(
        MODEL_PATH,
        subfolder="connectors",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
        local_files_only=LOCAL_FILES_ONLY,
    ).to(DEVICE)
    record_event("load_connectors_to_cuda", time.time() - event_t0, source=MODEL_PATH)

    event_t0 = time.time()
    connector_pipe = LTX2ImageConnectorStep().init_pipeline()
    connector_pipe.update_components(connectors=connectors)
    record_event("build_connector_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

    event_t0 = time.time()
    connector_state = connector_pipe(
        prompt_embeds=prompt_embeds.to(device=DEVICE, dtype=DTYPE),
        prompt_attention_mask=prompt_attention_mask.to(device=DEVICE),
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        do_classifier_free_guidance=False,
        pag_scale=PAG_SCALE if PAG_ENABLED else 0.0,
        output=[
            "connector_prompt_embeds",
            "connector_attention_mask",
            "batch_size",
            "transformer_batch_multiplier",
            "do_perturbed_attention_guidance",
        ],
    )
    record_event("connector_modular_pipe_call", time.time() - event_t0)

    connector_prompt_embeds = connector_state["connector_prompt_embeds"].to(OFFLOAD_DEVICE)
    connector_attention_mask = connector_state["connector_attention_mask"].to(OFFLOAD_DEVICE)
    latent_batch_size = connector_state["batch_size"]
    transformer_batch_multiplier = connector_state["transformer_batch_multiplier"]
    do_perturbed_attention_guidance = connector_state["do_perturbed_attention_guidance"]

    del prompt_embeds, prompt_attention_mask, connector_state
    del connector_pipe, connectors
    cleanup_runtime_state(record_event, "cleanup_after_connectors")

    dynamic_offload_enabled = DYNAMIC_OFFLOAD_SETTINGS.enabled
    dynamic_offload_config = DYNAMIC_OFFLOAD_CONFIG if dynamic_offload_enabled else None

    def transformer_prepare_event_name(name: str, repeat_index: int) -> str:
        if TRANSFORMER_PREPARE_REPEATS == 1:
            return name
        return f"{name}_prepare_{repeat_index + 1}"

    def load_and_prepare_transformer(repeat_index: int):
        transformer_load_kwargs = {
            "subfolder": "transformer",
            "torch_dtype": DTYPE,
            "low_cpu_mem_usage": MODEL_LOW_CPU_MEM_USAGE,
            "local_files_only": LOCAL_FILES_ONLY,
        }

        run_metrics[f"purge_windows_standby_before_transformer_{repeat_index + 1}"] = (
            purge_for_runner(args,
                DYNAMIC_OFFLOAD_SETTINGS,
                "before_transformer",
                record_event=record_event,
                event_name=transformer_prepare_event_name("purge_windows_standby_before_transformer", repeat_index),
            )
        )

        event_t0 = time.time()
        if MODEL_LOW_CPU_MEM_USAGE:
            transformer_load_kwargs["device_map"] = "cpu"
        # Use DDO's loader for every route.  With ``apply_dynamic=False`` it
        # only loads the model and retains immutable safetensor slice metadata;
        # the diagnostic runtime consumes no DDO hook or policy.
        transformer_load = from_pretrained_with_dynamic_offload(
            MODEL_PATH,
            model_loader=LTX2ImageTransformer2DModel,
            dynamic_offload_config=dynamic_offload_config,
            apply_dynamic=False,
            **transformer_load_kwargs,
        )
        prepared_transformer = transformer_load.module
        load_ltx_lora(prepared_transformer, args, run_metrics, record_event)
        prepared_dynamic_offload_hook = transformer_load.hook
        record_event(
            transformer_prepare_event_name("load_transformer", repeat_index),
            time.time() - event_t0,
            source=MODEL_PATH,
            low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
            device_map=transformer_load_kwargs.get("device_map"),
            loader="custom_blocks.LTX2ImageTransformer2DModel",
            resolved_class=prepared_transformer.__class__.__name__,
        )

        if args.ddo_vmm_diagnostic_adapter:
            event_t0 = time.time()
            library = load_native_vmm(Path(args.ddo_vmm_diagnostic_library))
            dxgi_memory_info = None
            if args.ddo_vmm_diagnostic_dxgi_budget_guard:
                if os.name != "nt":
                    raise SystemExit("--ddo-vmm-diagnostic-dxgi-budget-guard is available only on Windows")
                library.set_dxgi_budget_guard(
                    device_ordinal=0,
                    local_reserve_bytes=int(args.ddo_vmm_dxgi_local_headroom_gb * 2**30),
                    shared_reserve_bytes=int(args.ddo_vmm_dxgi_shared_headroom_gb * 2**30),
                    system_reserve_bytes=int(args.ddo_vmm_dxgi_system_headroom_gb * 2**30),
                )
                dxgi_memory_info = library.windows_memory_info(0)
            # The DDO native source owns its bounded registered-host slots.
            # Keep the established staging-size argument as the capacity
            # input; the harness does not construct or manage host buffers.
            staging_pool = None
            file_slices = getattr(prepared_transformer, "_ddo_safetensor_slices", None)
            if args.ddo_vmm_diagnostic_register_source_weights and args.vmm_host_cache_gb <= 0:
                raise ValueError(
                    "--ddo-vmm-diagnostic-register-source-weights requires --vmm-host-cache-gb > 0"
                )
            # DDO owns the model-address arena and block execution hooks. The
            # harness only supplies the component and resolved CLI settings.
            attach_report, store = attach_model_linear_modules(
                prepared_transformer,
                library,
                module_pattern=args.ddo_vmm_module_pattern,
                resident_budget_bytes=int(args.vmm_resident_weight_budget_gb * 2**30),
                page_size_bytes=int(args.ddo_vmm_page_mb * 2**20),
                device_ordinal=0,
                backing=args.ddo_vmm_diagnostic_backing,
                file_slices=file_slices,
                native_transport_slot_size_bytes=max(1, int(args.vmm_staging_buffer_gb * 2**30)),
                native_transport_slots=3,
                host_cache_bytes=(
                    int(args.vmm_host_cache_gb * 2**30)
                    if args.ddo_vmm_diagnostic_register_source_weights
                    else 0
                ),
            )
            transformer_offload = DiagnosticVmmRuntime(
                attach_report.adapter,
                store,
                prepared_transformer,
                staging_pool=staging_pool,
                report=attach_report,
            )
            store.reset_runtime_stats()
            print(
                f"  [ddo-vmm-diagnostic] execution_units={attach_report.execution_units} "
                f"attached_tensors={attach_report.attached_tensors} "
                f"payload_gb={attach_report.payload_bytes / 2**30:.4f} "
                f"active_budget_gb={args.vmm_resident_weight_budget_gb:g} "
                f"backing={args.ddo_vmm_diagnostic_backing} "
                f"file_transport={file_slices is not None} "
                "native_shard_transport=True "
                f"registered_host_gb={args.vmm_host_cache_gb if args.ddo_vmm_diagnostic_register_source_weights else 0:g} "
                f"dxgi_budget_guard={args.ddo_vmm_diagnostic_dxgi_budget_guard}",
                flush=True,
            )
            if dxgi_memory_info is not None:
                print(
                    "  [dxgi-budget] "
                    f"local={dxgi_memory_info.local_usage_bytes / 2**30:.2f}/"
                    f"{dxgi_memory_info.local_budget_bytes / 2**30:.2f} GiB "
                    f"shared={dxgi_memory_info.nonlocal_usage_bytes / 2**30:.2f}/"
                    f"{dxgi_memory_info.nonlocal_budget_bytes / 2**30:.2f} GiB "
                    f"reserves=local:{args.ddo_vmm_dxgi_local_headroom_gb:g},"
                    f"shared:{args.ddo_vmm_dxgi_shared_headroom_gb:g},"
                    f"system:{args.ddo_vmm_dxgi_system_headroom_gb:g} GiB",
                    flush=True,
                )
            record_event(
                transformer_prepare_event_name("build_ddo_vmm_diagnostic_adapter", repeat_index),
                time.time() - event_t0,
                execution_units=attach_report.attached_modules,
                tensors=attach_report.attached_tensors,
                pattern=args.ddo_vmm_module_pattern,
            )
            return prepared_transformer, None, transformer_offload

        if dynamic_offload_enabled and DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan" and TRANSFORMER_MEMORY_MANAGER != "off":
            raise ValueError(
                "Dynamic offload execution currently requires "
                "DDO_RUNNER_TRANSFORMER_MEMORY_MANAGER='off'. "
                "Use execution_mode='plan' with the block manager."
            )

        event_t0 = time.time()
        if TRANSFORMER_MEMORY_MANAGER != "off":
            raise ValueError(
                "The transformer block manager is no longer used by this runner. "
                "Use DDO_PRESET or transformer group offload."
            )
        transformer_offload = enable_offload(
            prepared_transformer,
            settings=DYNAMIC_OFFLOAD_SETTINGS,
            config=dynamic_offload_config,
            component="transformer",
            execution_device=DEVICE,
            offload_device=OFFLOAD_DEVICE,
            low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
            record_event=record_event,
            dynamic_event_name=transformer_prepare_event_name("build_dynamic_offload_plan", repeat_index),
            group_event_name=transformer_prepare_event_name("setup_transformer_group_offload", repeat_index),
            profile_name=args.ddo_profile or "",
            build_profile=args.build_ddo_profile,
        )
        prepared_dynamic_offload_hook = transformer_offload.hook
        if prepared_dynamic_offload_hook is not None and DYNAMIC_OFFLOAD_EXECUTION_MODE == "vmm_runtime":
            decisions = prepared_dynamic_offload_hook.state.planner_decisions
            print(
                "  [vmm] "
                f"execution_groups={decisions.get('vmm_execution_group_count', 0)} "
                f"max_group_weight_gb={decisions.get('vmm_execution_group_max_weight_gb', 0.0):.4f}",
                flush=True,
            )
        if transformer_offload.route == "dynamic_offload" and DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan":
            record_event(
                transformer_prepare_event_name("skip_transformer_to_cuda", repeat_index),
                time.time() - event_t0,
                reason=f"dynamic_offload_{DYNAMIC_OFFLOAD_EXECUTION_MODE}",
            )
        return prepared_transformer, prepared_dynamic_offload_hook, transformer_offload

    transformer = None
    dynamic_offload_hook = None
    transformer_offload = None
    for prepare_repeat_index in range(TRANSFORMER_PREPARE_REPEATS):
        if TRANSFORMER_PREPARE_REPEATS > 1:
            print(
                f"  Transformer prepare repeat {prepare_repeat_index + 1}/{TRANSFORMER_PREPARE_REPEATS}",
                flush=True,
            )
        transformer, dynamic_offload_hook, transformer_offload = load_and_prepare_transformer(prepare_repeat_index)
        if prepare_repeat_index + 1 < TRANSFORMER_PREPARE_REPEATS:
            if args.ddo_vmm_diagnostic_adapter:
                transformer_offload.close()
            elif dynamic_offload_enabled:
                remove_dynamic_offload(transformer)
            # The apply result owns the module even after hook removal.
            # Drop every runner owner before collecting and loading the next model.
            del transformer_offload
            del transformer
            dynamic_offload_hook = None
            cleanup_runtime_state(record_event, transformer_prepare_event_name("cleanup_after_transformer_prepare", prepare_repeat_index))

    with (transformer_offload.cleanup_on_error() if args.ddo_public_native_runtime else nullcontext()):
        event_t0 = time.time()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_PATH, subfolder="scheduler", local_files_only=LOCAL_FILES_ONLY
        )
        record_event("load_scheduler", time.time() - event_t0, source=MODEL_PATH)

        event_t0 = time.time()
        prepare_pipe = LTX2ImagePrepareLatentsStep().init_pipeline()
        prepare_pipe.update_components(transformer=transformer, scheduler=scheduler)
        record_event("build_prepare_latents_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

        event_t0 = time.time()
        denoise_pipe = LTX2ImageDenoiseStep().init_pipeline()
        denoise_pipe.update_components(transformer=transformer, scheduler=scheduler)
        record_event("build_denoise_modular_pipeline", time.time() - event_t0, model_path=MODEL_PATH)

        image_latent = None
        latent_height = None
        latent_width = None
        latent_channels = None
        dynamic_offload_metrics = None
        run_metrics["denoise_step_times_by_repeat"] = []
        run_metrics["generation_observations"] = []

        for repeat_index in range(GENERATION_REPEATS):
            repeat_suffix = "" if GENERATION_REPEATS == 1 else f"_repeat_{repeat_index + 1}"
            if GENERATION_REPEATS > 1:
                print(f"  Warm generation repeat {repeat_index + 1}/{GENERATION_REPEATS}", flush=True)

            generation_start = time.perf_counter()
            event_t0 = time.time()
            prepare_state = prepare_pipe(
                width=WIDTH,
                height=HEIGHT,
                num_inference_steps=active_inference_steps,
                batch_size=latent_batch_size,
                transformer_batch_multiplier=transformer_batch_multiplier,
                generator=generator,
                output=["latents", "timesteps", "latent_height", "latent_width", "in_channels", "video_rotary_emb"],
            )
            record_event(f"prepare_latents_modular_pipe_call{repeat_suffix}", time.time() - event_t0)

            denoise_progress_callback.timesteps = prepare_state["timesteps"]
            # Resolve display-only scalars before the measured denoise scope.
            # Converting a CUDA timestep in the callback drains the compute stream.
            denoise_progress_callback.display_timesteps = (
                prepare_state["timesteps"].detach().cpu().tolist() if SHOW_DENOISE_STEPS else []
            )
            denoise_progress_callback.start_time = time.perf_counter()
            denoise_progress_callback.last_time = denoise_progress_callback.start_time
            denoise_progress_callback.step_times = []
            print(f"  Starting denoise loop{repeat_suffix}", flush=True)
            if args.ddo_gpu_timing:
                # Public snapshot lets diagnostics isolate process-lifetime native
                # counters without teaching the runner their names or meanings.
                run_metrics[f"ddo_before_denoise{repeat_suffix}"] = (
                    transformer_offload.metrics_snapshot().as_dict()
                )
            event_t0 = time.time()
            execution_timer = transformer_offload.execution_timer(
                enabled=True, native_units=args.ddo_native_timeline,
                diagnostic_step_wait=args.ddo_diagnostic_step_wait,
            ) if args.ddo_gpu_timing else nullcontext()
            with transformer_offload.profile_run() as ddo_profile_report, execution_timer as ddo_timer:
                denoise_progress_callback.execution_timer = ddo_timer
                denoise_state = denoise_pipe(
                    latents=prepare_state["latents"],
                    timesteps=prepare_state["timesteps"],
                    connector_prompt_embeds=connector_prompt_embeds.to(device=DEVICE, dtype=DTYPE),
                    connector_attention_mask=connector_attention_mask.to(device=DEVICE),
                    latent_height=prepare_state["latent_height"],
                    latent_width=prepare_state["latent_width"],
                    video_rotary_emb=prepare_state["video_rotary_emb"],
                    batch_size=latent_batch_size,
                    transformer_batch_multiplier=transformer_batch_multiplier,
                    do_classifier_free_guidance=False,
                    do_perturbed_attention_guidance=do_perturbed_attention_guidance,
                    guidance_scale=GUIDANCE_SCALE,
                    guidance_rescale=GUIDANCE_RESCALE,
                    pag_scale=PAG_SCALE if PAG_ENABLED else 0.0,
                    pag_applied_layers=PAG_APPLIED_LAYERS if PAG_ENABLED else None,
                    callback_on_step_end=denoise_progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                    output="latents",
                )
            record_event(
                f"denoise_modular_pipe_call{repeat_suffix}",
                time.time() - event_t0,
                repeat_index=repeat_index,
            )
            run_metrics["denoise_step_times_by_repeat"].append(list(denoise_progress_callback.step_times))
            run_metrics["denoise_step_times"] = denoise_progress_callback.step_times
            if image_latent is not None:
                del image_latent
            image_latent = denoise_state.to(OFFLOAD_DEVICE)
            latent_height = prepare_state["latent_height"]
            latent_width = prepare_state["latent_width"]
            latent_channels = prepare_state["in_channels"]
            if SHOW_METRICS:
                print(f"  Image latent: {image_latent.shape}")
            del prepare_state, denoise_state
            # The existing CPU latent readback is the completion boundary. Observe
            # each generation here, not just the last timer after the loop. Do not
            # add a CUDA wait or reap tickets: final collection remains unchanged.
            generation_seconds = time.perf_counter() - generation_start
            generation_report = (
                transformer_offload.metrics_snapshot(execution_timing=ddo_timer).as_dict()
                if args.ddo_gpu_timing else None
            )
            run_metrics["generation_observations"].append({
                "repeat_index": repeat_index,
                "scope": "prepare_latents_through_cpu_latent_readback",
                "elapsed_seconds": generation_seconds,
                "ddo_report": generation_report,
            })

        run_metrics["ddo_profile"] = ddo_profile_report

        if dynamic_offload_enabled and DYNAMIC_OFFLOAD_EXECUTION_MODE != "plan":
            dynamic_offload_metrics = transformer_offload.metrics_snapshot(
                collect_completed=True, execution_timing=ddo_timer,
            ) if args.ddo_gpu_timing else transformer_offload.metrics_snapshot()
            record_runner_report(args, tracker, dynamic_offload_metrics)
            if DYNAMIC_OFFLOAD_SHOW_PROFILE and (not SHOW_METRICS or args.metrics_level < 2):
                print(dynamic_offload_metrics.format_profile(), flush=True)
        del connector_prompt_embeds, connector_attention_mask
    if args.ddo_vmm_diagnostic_adapter:
        transformer_offload.close()
        transformer_offload = None
    elif dynamic_offload_enabled:
        remove_dynamic_offload(transformer)
        dynamic_offload_hook = None
    # The apply result owns the module even after its hook is removed.
    # Release this final owner before collecting memory for resident VAE decode.
    del transformer_offload
    del prepare_pipe, denoise_pipe, transformer, scheduler
    cleanup_before_vae_decode(record_event)
    step_end(f"Pass 1: Generate at {WIDTH}x{HEIGHT}", t0)

    if args.build_ddo_profile:
        run_metrics["total_elapsed_sec"] = round(tracker.total_elapsed(), 4)
        run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
        run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)
        if SAVE_METRICS:
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = metrics_dir / f"{run_slug}_ddo_profile.json"
            metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
            print(f"  Metrics JSON: {metrics_path}")
        print(f"  DDO profile calibration complete: {ddo_profile_report}")
        return

    vae_phase_start = time.time()
    t0 = step_start("Load VAE for decode")

    event_t0 = time.time()
    vae = AutoencoderKLLTX2Video.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
        local_files_only=LOCAL_FILES_ONLY,
    )
    vae_offload = enable_offload(
        vae, component="vae", route="diffusers_group_offload" if args.vae_leaf_offload else "none",
        execution_device=DEVICE, offload_device="cpu", use_environment=False,
        offload_type="leaf_level", use_stream=False, low_cpu_mem_usage=False,
        record_event=record_event,
    )
    if vae_offload.should_move_to_execution_device:
        vae.to(DEVICE)
    run_metrics["vae_decode_route"] = vae_offload.route
    record_event("load_vae_for_decode", time.time() - event_t0, source=MODEL_PATH,
                 route=vae_offload.route)
    step_end("Load VAE for decode", t0)
    t0 = step_start("Decode VAE")

    event_t0 = time.time()
    decode_pipe = LTX2ImageDistilledBlocks().sub_blocks["decode"].init_pipeline()
    decode_pipe.update_components(vae=vae)
    with vae_offload.cleanup_on_error():
        image = decode_pipe(
            latents=image_latent.to(device=DEVICE, dtype=DTYPE),
            batch_size=latent_batch_size,
            latent_height=latent_height,
            latent_width=latent_width,
            in_channels=latent_channels,
            decode_timestep=DECODE_TIMESTEP,
            decode_noise_scale=DECODE_NOISE_SCALE,
            generator=generator,
            output_type="pil",
            output="images",
        )[0]
    record_event("vae_decode_modular_call", time.time() - event_t0)
    record_runner_report(args, tracker, vae_offload.metrics_snapshot(), "vae")
    step_end("Decode VAE", t0)
    t0 = step_start("Cleanup VAE")

    del decode_pipe, vae_offload, vae, image_latent
    cleanup_runtime_state(record_event, "cleanup_after_vae_decode")
    step_end("Cleanup VAE", t0)
    record_event("Pass 2: Decode VAE", time.time() - vae_phase_start,
                 scope="aggregate_wall_time_not_an_additional_phase")

    t0 = step_start("Save Image")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_slug}.png"
    image.save(output_path)
    print(f"  Image saved successfully to: {output_path}")
    step_end("Save Image", t0)

    total_time = tracker.total_elapsed()
    run_metrics["total_elapsed_sec"] = round(total_time, 4)
    run_metrics["global_peak_vram_gb"] = round(tracker.global_peak_vram, 4)
    run_metrics["global_peak_ram_gb"] = round(tracker.global_peak_ram, 4)

    metrics_path = metrics_dir / f"{run_slug}.json"
    if SAVE_METRICS:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {tracker.global_peak_vram:.2f} GB | Peak RAM: {tracker.global_peak_ram:.2f} GB")
    print(f"  Output: {output_path}")
    if SAVE_METRICS:
        print(f"  Metrics JSON: {metrics_path}")
    if RESET_DYNAMIC_MEMORY_AFTER_RUN:
        cleanup_runtime_state(record_event, "reset_dynamic_memory_after_run", collect_cuda_ipc=True)
        if SAVE_METRICS:
            metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
        print("  Dynamic memory state reset after run.")
    run_metrics["purge_windows_standby_after_run"] = purge_for_runner(args,
        DYNAMIC_OFFLOAD_SETTINGS,
        "after_run",
        record_event=record_event,
    )
    if SAVE_METRICS:
        metrics_path.write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    print("=" * 70)


if __name__ == "__main__":
    main()
