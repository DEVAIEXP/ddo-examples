"""Text-encoder route selection shared by the native VMM runners (model-agnostic).

On ``vmm/auto`` a runner may load its text encoder through DDO's weight-aware
loader with native component ownership. Whether that is the default is decided
per entry from measurements; ``--[no-]text-encoder-native-vmm`` always wins.
The encoder ceiling is encoder-only and never the transformer's budget.
"""
import argparse
import math

# Component ownership needs two execution slots of the largest encoder
# component inside this ceiling (0.625 GiB was refused for Gemma 3); 4 GiB is
# the measured value for the LTX Gemma encoder.
DEFAULT_TEXT_ENCODER_VMM_BUDGET_GB = 4.0


def add_text_encoder_arguments(parser):
    parser.add_argument(
        "--text-encoder-native-vmm", action=argparse.BooleanOptionalAction, default=None,
        help="Native text-encoder ownership on the vmm/auto route (entry-specific default); "
             "--no-text-encoder-native-vmm uses Diffusers leaf group offload.")
    parser.add_argument(
        "--text-encoder-vmm-resident-weight-budget-gb", type=float,
        help=f"Encoder-only native weight ceiling (default {DEFAULT_TEXT_ENCODER_VMM_BUDGET_GB:g} GiB "
             "with the native encoder); never changes the transformer ceiling.")


def resolve_text_encoder_route(parser, args, *, native_default=True):
    """Resolve the encoder default from the transformer route; reject conflicts.

    ``native_default`` lets an entry keep leaf as its default on ``vmm/auto``
    until the native encoder is measured for that workload.
    """
    from diffusers_dynamic_offloader import resolve_dynamic_offload_backend

    native_route = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset).requires_native_bootstrap
    if args.text_encoder_native_vmm is None:
        args.text_encoder_native_vmm = native_route and native_default
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


def native_text_encoder_config(base_config, budget_gb):
    """Component-ownership config for a native encoder; transformer limits unchanged."""
    from dataclasses import replace

    # Component ownership requires native transport even when the transformer
    # route runs without it (generic CLI configs without a retained source).
    return replace(base_config, vmm_resident_weight_budget_gb=budget_gb, vmm_module_pattern=".*", vmm_page_mb=2,
                   vmm_native_transport=True,
                   vmm_component_tensor_ownership=True, vmm_component_buffer_limit_mb=1,
                   vmm_execution_auxiliary_ownership=False, vmm_float8_weight_only=False,
                   profile_name=None, build_profile=False)
