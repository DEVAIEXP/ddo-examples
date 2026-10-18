"""Explicit, unfused LTX adapters loaded before component offload hooks."""

import math
import time
from pathlib import Path
from types import SimpleNamespace


def add_lora_arguments(parser):
    parser.add_argument("--lora", action="append", help="Transformer LoRA source; repeat for multiple adapters.")
    parser.add_argument("--lora-weight-name", action="append", help="Filename per LoRA, in source order; 'auto' selects automatically.")
    parser.add_argument("--lora-revision", action="append", help="Hub revision per LoRA, in source order; 'auto' uses the default.")
    parser.add_argument("--lora-scale", type=float, action="append", help="Strength per LoRA, in source order; defaults to 1.0 for all.")


def validate_lora_arguments(parser, args):
    if any(not math.isfinite(scale) for scale in (args.lora_scale or [])):
        parser.error("--lora-scale must be finite")
    if not args.lora and (args.lora_weight_name or args.lora_revision or args.lora_scale):
        parser.error("LoRA filename, revision and scale require --lora")
    for name in ("lora_weight_name", "lora_revision", "lora_scale"):
        values = getattr(args, name)
        if values and len(values) != len(args.lora):
            parser.error(f"--{name.replace('_', '-')} must be supplied once per --lora, in source order")
    if args.lora and getattr(args, "transformer_fp8_weight_only", False):
        parser.error("Unfused LoRA with saved FP8 is not validated; use BF16 for this recipe")
    if args.lora and (getattr(args, "soft_lora", False) or getattr(args, "crisp_lora", False)):
        parser.error("Use --lora or the existing named LoRA controls, not both")


def load_ltx_lora(model, args, metrics, record_event):
    """Load all requested adapters before offload, keeping earlier ones active."""
    if not getattr(args, "lora", None):
        return
    active = []
    for index, source in enumerate(args.lora):
        options = dict(vars(args), lora=source)
        for name, default in (("lora_weight_name", None), ("lora_revision", None), ("lora_scale", 1.0)):
            values = getattr(args, name)
            value = values[index] if values else default
            options[name] = None if value == "auto" else value
        name = "runner_lora" if index == 0 else f"runner_lora_{index + 1}"
        _load_single_lora(model, SimpleNamespace(**options), metrics, record_event, name, active)
        active.append((name, options["lora_scale"]))


def _load_single_lora(model, args, metrics, record_event, adapter_name, active):
    """Preserve base Parameters and PEFT forwards; never fuse or move the model.

    DDO attaches afterwards to the base and adapter leaves. This helper does not
    create source descriptors, change native budgets or manage GPU residency.
    """
    if not getattr(args, "lora", None):
        return
    import torch
    from diffusers.loaders import LTX2LoraLoaderMixin
    from peft.tuners.tuners_utils import BaseTunerLayer

    started = time.perf_counter()
    if not active and any(isinstance(child, BaseTunerLayer) for child in model.modules()):
        raise ValueError("Load the requested LoRA once, before offload and other adapters")
    base_parameters = tuple(model.parameters())
    if any(parameter.device.type != "cpu" for parameter in base_parameters):
        raise ValueError("Load LoRA on the CPU transformer before attaching offload")
    source = args.lora
    options = dict(local_files_only=getattr(args, "local_files_only", False),
                   return_lora_metadata=True, use_safetensors=True)
    if Path(source).is_file():
        if args.lora_weight_name or args.lora_revision:
            raise ValueError("A local LoRA file cannot also select a filename or Hub revision")
        path = Path(source).resolve()
        source, options["weight_name"] = str(path.parent), path.name
    elif args.lora_weight_name:
        options["weight_name"] = args.lora_weight_name
    if args.lora_revision:
        options["revision"] = args.lora_revision
    state, metadata = LTX2LoraLoaderMixin.lora_state_dict(source, **options)
    if not state or any(not key.startswith("transformer.") for key in state):
        raise ValueError("This runner accepts transformer-only LoRAs; other components must not be silently ignored")
    modules = dict(model.named_modules())
    targets = set()
    for key in state:
        target, separator, suffix = key.removeprefix("transformer.").partition(".lora_")
        if not separator or suffix not in ("A.weight", "B.weight") or target not in modules:
            raise ValueError(f"Unsupported or unmatched LoRA tensor: {key}")
        if not torch.isfinite(state[key]).all().item():
            raise ValueError(f"Nonfinite LoRA tensor: {key}")
        targets.add(target)
    for target in targets:
        if not all(f"transformer.{target}.lora_{part}.weight" in state for part in ("A", "B")):
            raise ValueError(f"Incomplete LoRA pair: {target}")
    base_identity = {id(parameter): (parameter.data_ptr(), parameter._version) for parameter in base_parameters}
    model.load_lora_adapter(state, prefix="transformer", adapter_name=adapter_name,
                            metadata=metadata, low_cpu_mem_usage=True)
    selections = [*active, (adapter_name, args.lora_scale)]
    model.set_adapters([name for name, _ in selections], weights=[scale for _, scale in selections])
    model.eval().requires_grad_(False)
    layers = [(name, child) for name, child in model.named_modules()
              if isinstance(child, BaseTunerLayer) and adapter_name in child.lora_A]
    if {name for name, _ in layers} != targets:
        raise RuntimeError("Loaded LoRA targets do not match the requested checkpoint")
    if any(child.merged or adapter_name not in child.active_adapters or child.disable_adapters
           for _, child in layers):
        raise RuntimeError("The requested unfused LoRA is not active on every target")
    current_ids = {id(parameter) for parameter in model.parameters()}
    if any(id(p) not in current_ids or (p.data_ptr(), p._version) != base_identity[id(p)]
           for p in base_parameters):
        raise RuntimeError("LoRA loading changed base storage; refusing stale source associations")
    adapter_parameters = [p for p in model.parameters() if id(p) not in base_identity]
    report = dict(source=args.lora, weight_name=options.get("weight_name"), revision=args.lora_revision,
                  adapter_name=adapter_name, scale=args.lora_scale, fused=False,
                  target_count=len(targets), adapter_parameter_bytes=sum(p.numel()*p.element_size() for p in adapter_parameters),
                  base_storage_preserved=True, loaded_and_active=True,
                  validation_scope="load_and_activation")
    metrics.setdefault("lora_adapters", []).append(report)
    record_event("load_transformer_lora", time.perf_counter()-started, **report)
    print(f"LoRA active: {adapter_name}, scale={args.lora_scale:g}, targets={len(targets)}, unfused", flush=True)
