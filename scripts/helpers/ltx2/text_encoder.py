"""Text-encoder route selection shared by the LTX video runners.

On the native ``vmm/auto`` route the Gemma text encoder defaults to native
component ownership through DDO's weight-aware loader. A measured LTX 2.3
video 8 s pair on Windows (cold standby cache in both runs) showed the
native encoder phase at 11.2 s versus 20.8 s for Diffusers leaf group offload
with identical output; this is a recipe default, not a general claim.
``--no-text-encoder-native-vmm`` restores leaf group offload.
"""
import argparse
import math
from dataclasses import replace

# Encoder-only native weight ceiling. Component ownership needs two execution
# slots of the largest encoder component inside this ceiling (0.625 GiB is
# refused at attachment); 4 GiB is the measured value.
DEFAULT_TEXT_ENCODER_VMM_BUDGET_GB = 4.0


def add_text_encoder_arguments(parser):
    parser.add_argument(
        "--text-encoder-native-vmm", action=argparse.BooleanOptionalAction, default=None,
        help="Native text-encoder ownership. Default: on for the vmm/auto route, off otherwise; "
             "--no-text-encoder-native-vmm uses Diffusers leaf group offload.")
    parser.add_argument(
        "--text-encoder-vmm-resident-weight-budget-gb", type=float,
        help=f"Encoder-only native weight ceiling (default {DEFAULT_TEXT_ENCODER_VMM_BUDGET_GB:g} GiB "
             "with the native encoder); never changes the transformer ceiling.")


def resolve_text_encoder_route(parser, args):
    """Resolve the encoder default from the transformer route; reject conflicts."""
    from diffusers_dynamic_offloader import resolve_dynamic_offload_backend

    native_route = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).requires_native_bootstrap
    if args.text_encoder_native_vmm is None:
        args.text_encoder_native_vmm = native_route
    elif args.text_encoder_native_vmm and not native_route:
        parser.error("Native text encoding requires the vmm/auto fast route.")
    budget = args.text_encoder_vmm_resident_weight_budget_gb
    if args.text_encoder_native_vmm:
        if budget is None:
            args.text_encoder_vmm_resident_weight_budget_gb = DEFAULT_TEXT_ENCODER_VMM_BUDGET_GB
        elif not math.isfinite(budget) or budget <= 0:
            parser.error("--text-encoder-vmm-resident-weight-budget-gb must be positive and finite.")
    elif budget is not None:
        parser.error("The encoder-only ceiling requires the native text encoder.")
    return args


def build_text_encoder_config(args, base_config):
    """Select explicit component ownership without mutating transformer limits."""
    if not args.text_encoder_native_vmm:
        return None
    return replace(base_config,
                   vmm_resident_weight_budget_gb=args.text_encoder_vmm_resident_weight_budget_gb,
                   vmm_module_pattern=".*", vmm_page_mb=2,
                   vmm_component_tensor_ownership=True, vmm_component_buffer_limit_mb=1,
                   vmm_execution_auxiliary_ownership=False,
                   vmm_float8_weight_only=False,
                   profile_name=None, build_profile=False)


def load_text_encoder_for_phase(args, dtype, config, record_report):
    """Use DDO's public weight-aware loader only for the native encoder route."""
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
    if record_report is not None:
        record_report(loaded.inspect_vmm_component(component="text_encoder"), "text_encoder_preflight")
    return module
