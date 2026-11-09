"""Text-encoder route selection for the LTX runners (Gemma 3 encoder).

On the native ``vmm/auto`` route the LTX video entries default to native
component ownership of the Gemma text encoder through DDO's weight-aware
loader. A measured LTX 2.3 video 8 s pair on Windows (cold standby cache in
both runs) showed the native encoder phase at 11.2 s versus 20.8 s for
Diffusers leaf group offload with identical output; this is a recipe default,
not a general claim. ``--no-text-encoder-native-vmm`` restores leaf group
offload. The route selection itself is the model-agnostic
``scripts/helpers/text_encoder_route.py``.
"""
from scripts.helpers.text_encoder_route import (  # noqa: F401  (public API of this module)
    DEFAULT_TEXT_ENCODER_VMM_BUDGET_GB,
    add_text_encoder_arguments,
    native_text_encoder_config,
    resolve_text_encoder_route,
)


def build_text_encoder_config(args, base_config):
    """Select explicit component ownership without mutating transformer limits."""
    if not args.text_encoder_native_vmm:
        return None
    return native_text_encoder_config(base_config, args.text_encoder_vmm_resident_weight_budget_gb)


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
