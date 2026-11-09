"""Shared CLI vocabulary and lifecycle calls, not model orchestration."""

import argparse
import math
import os
import sys
from dataclasses import replace


def _alias(parser, name, source):
    action = parser._option_string_actions.get(source)
    if action is None or name in parser._option_string_actions:
        return
    kwargs = dict(dest=action.dest, default=argparse.SUPPRESS,
                  help=f"Same control as {source}; retained compatibility spelling.")
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        kwargs["action"] = argparse.BooleanOptionalAction
    else:
        kwargs.update(type=action.type, choices=action.choices)
    parser.add_argument(name, **kwargs)


def _canonical_public_interface(parser, *, workload_profiles=False):
    """Expose one public vocabulary; diagnostics keep their own entries.

    Preserve internal destinations and defaults so this CLI migration does not
    change component orchestration, staging units or native allocation policy.
    """
    parser.allow_abbrev = False
    rename = {"--vmm-resident-weight-budget-gb": "--ddo-vmm-resident-weight-budget-gb",
              "--show-ddo-profile": "--show-profile",
              "--no-show-ddo-profile": "--no-show-profile"}
    public_ddo = {"--ddo-backend", "--ddo-preset", "--ddo-vmm-resident-weight-budget-gb",
                  "--ddo-vmm-total-staging-gb", "--ddo-vmm-source-cache-gb",
                  "--ddo-vmm-source-registered-gb", "--ddo-vmm-execution-backing-reclaim"}
    diagnostic = {"--preset", "--auxiliary-preset", "--pipeline-path", "--transformer-path",
                  "--fake-prompt", "--generation-repeats", "--transformer-prepare-repeats",
                  "--pre-vae-cleanup-repeats", "--resident-module-budget-gb",
                  "--max-resident-module-budget-gb", "--pin-weight-budget-gb",
                  "--reset-dynamic-memory-after-run", "--print-presets", "--build-ddo-profile",
                  "--trace-transformer-blocks"}
    if workload_profiles:
        public_ddo.update({"--ddo-profile", "--build-ddo-profile"})
        diagnostic.discard("--build-ddo-profile")
    for action in list(parser._actions):
        options = []
        for old in action.option_strings:
            if action.help and "retained compatibility spelling" in action.help:
                continue
            option = rename.get(old, old)
            positive = "--" + option[5:] if option.startswith("--no-") else option
            if positive in diagnostic or positive.startswith("--vmm-"):
                continue
            if positive.startswith("--ddo-") and positive not in public_ddo:
                continue
            options.append(option)
        if action.option_strings and not options:
            if action.default is not argparse.SUPPRESS:
                parser.set_defaults(**{action.dest: action.default})
            parser._remove_action(action)
            for group in parser._action_groups:
                if action in group._group_actions: group._group_actions.remove(action)
        for option in action.option_strings:
            if parser._option_string_actions.get(option) is action:
                parser._option_string_actions.pop(option)
        action.option_strings = options
        if any(option in public_ddo and option.endswith("-gb") for option in options):
            action.metavar = "GIB"
        if "--ddo-vmm-total-staging-gb" in options:
            action.help = "Total reusable VMM host-staging capacity in GiB; see the documented model defaults."
        for option in options: parser._option_string_actions[option] = action


def parse_common_arguments(parser, argv=None, *, purge_phases=("before_run", "after_text_encoder", "before_transformer", "after_run"), canonical_native=False, canonical_image=False, canonical_video=False, image_staging_slots=False, workload_profiles=False):
    """Normalize controls already supported by an entry; preserve old defaults.

    Minimal entries still expose only their implemented controls. Legacy option
    destinations remain intact so model orchestration does not need a dispatcher.
    """
    from diffusers_dynamic_offloader import add_dynamic_offload_lifecycle_arguments

    if workload_profiles:
        if "--ddo-profile" not in parser._option_string_actions:
            parser.add_argument("--ddo-profile", help="Named persistent transformer capacity profile.")
        if "--build-ddo-profile" not in parser._option_string_actions:
            parser.add_argument("--build-ddo-profile", action="store_true",
                                help="Calibrate and save --ddo-profile using a real denoise workload.")

    if "--memory-monitor" not in parser._option_string_actions:
        parser.add_argument(
            "--memory-monitor", choices=("rss", "full"), default="full",
            help="RSS monitoring or full USS monitoring; omitted preserves the existing full monitor.",
        )

    if "--purge-before-transformer" not in parser._option_string_actions:
        add_dynamic_offload_lifecycle_arguments(parser, phases=purge_phases)
    else:
        for phase in purge_phases:
            if phase != "before_transformer":
                parser.add_argument("--purge-" + phase.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    _alias(parser, "--ddo-preset", "--preset")
    if "--distilled-model" in parser._option_string_actions:
        _alias(parser, "--pipeline-path", "--distilled-model")
        _alias(parser, "--transformer-path", "--model-path")
    elif "--auxiliary-model" in parser._option_string_actions:
        _alias(parser, "--pipeline-path", "--auxiliary-model")
        _alias(parser, "--transformer-path", "--model-path")
    else:
        _alias(parser, "--pipeline-path", "--model-path")
        _alias(parser, "--transformer-path", "--sdnq-model-path")
    if "--ddo-backend" in parser._option_string_actions:
        _alias(parser, "--auxiliary-preset", "--preset")
    _alias(parser, "--show-profile", "--show-ddo-profile")
    _alias(parser, "--ddo-vmm-resident-weight-budget-gb", "--vmm-resident-weight-budget-gb")
    # Negative switches have the same meaning wherever the positive exists.
    for option in ("show-profile", "show-metrics", "save-metrics", "show-denoise-steps",
                   "local-files-only", "build-ddo-profile", "purge-before-transformer",
                   "vae-tiling", "vae-resident", "vae-leaf-offload", "vae-linear-offload"):
        action = parser._option_string_actions.get("--" + option)
        if action is not None and "--no-" + option not in parser._option_string_actions:
            parser.add_argument("--no-" + option, dest=action.dest, action="store_false",
                                default=argparse.SUPPRESS)
    if "--ddo-preset" in parser._option_string_actions and "--ddo-backend" not in parser._option_string_actions:
        parser.add_argument("--ddo-backend", choices=("linear",), default="linear",
                            help="This entry supports linear/compatibility routes only.")
    image_slots = image_staging_slots or "--vmm-staging-buffer-gb" in parser._option_string_actions
    if (canonical_native or canonical_image or canonical_video) and "--ddo-vmm-source-cache-gb" in parser._option_string_actions:
        parser.add_argument("--ddo-vmm-execution-backing-reclaim", action=argparse.BooleanOptionalAction,
                            default=False, help="Release idle execution backing under pressure in vmm/auto; disabled unless explicitly enabled.")
    if image_slots and "--ddo-vmm-total-staging-gb" not in parser._option_string_actions:
        parser.add_argument("--ddo-vmm-total-staging-gb", type=float, default=None,
                            help="Total staging capacity in GiB; omitted preserves legacy per-slot defaults.")
    if canonical_native or canonical_image or canonical_video:
        _canonical_public_interface(parser, workload_profiles=workload_profiles)
    args = parser.parse_args(argv)
    if canonical_image and hasattr(args, "preset"):
        args.ddo_preset = args.preset
    if workload_profiles:
        if args.ddo_profile is not None and not args.ddo_profile.strip():
            parser.error("--ddo-profile must be a nonempty name")
        if args.build_ddo_profile and not args.ddo_profile:
            parser.error("--build-ddo-profile requires --ddo-profile NAME")
        if args.ddo_profile and args.ddo_preset not in (
            "auto", "one_shot_fast", "low_ram_safe", "warm_process", "wsl_compat"
        ):
            parser.error("Workload profiles require a linear DDO preset or vmm/auto, "
                         "not a compatibility/off route or the NVMe baseline")
    if getattr(args, "ddo_vmm_execution_backing_reclaim", False) and (
        args.ddo_backend != "vmm" or args.ddo_preset != "auto"
    ):
        parser.error("--ddo-vmm-execution-backing-reclaim requires --ddo-backend vmm --ddo-preset auto")
    raw = sys.argv[1:] if argv is None else argv
    if getattr(args, "ddo_vmm_total_staging_gb", None) is not None and any(
        value.split("=", 1)[0] in ("--vmm-staging-buffer-gb", "--ddo-vmm-staging-buffer-gb") for value in raw
    ):
        parser.error("Use either total staging or a legacy staging value, not both")
    args.explicit_ddo_preset = any(value.split("=", 1)[0] in ("--ddo-preset", "--preset", "--auxiliary-preset") for value in raw)
    if image_slots and args.ddo_backend == "linear" and any(value.split("=", 1)[0] == "--ddo-preset" for value in raw):
        args.preset = args.ddo_preset
    if image_slots and args.ddo_vmm_total_staging_gb is not None:
        total = args.ddo_vmm_total_staging_gb
        if not math.isfinite(total) or total <= 0:
            parser.error("--ddo-vmm-total-staging-gb must be finite and positive")
        # The image harness preserves its historical three-slot native adapter.
        native = args.ddo_backend == "vmm" and args.ddo_preset == "auto"
        args.vmm_staging_buffer_gb = total / (3 if native else 2)
    return args


def apply_execution_backing_reclaim(config, args):
    """Apply the public CLI opt-in to DDO's public configuration, not process state."""
    enabled = getattr(args, "ddo_vmm_execution_backing_reclaim", None)
    if enabled is None:
        return config
    if enabled and config.execution_mode != "vmm_fast_runtime":
        raise ValueError("Execution backing reclaim requires vmm_fast_runtime")
    return replace(config, vmm_execution_backing_reclaim=enabled)


def add_minimal_vmm_arguments(parser):
    """Use DDO's public capacities for the minimal native examples."""
    from diffusers_dynamic_offloader import add_dynamic_offload_arguments

    native = argparse.ArgumentParser(add_help=False)
    add_dynamic_offload_arguments(native)
    for action in native._actions:
        if any(option.startswith("--ddo-") for option in action.option_strings):
            parser._add_action(action)
    parser.set_defaults(ddo_backend="vmm", ddo_preset="auto")
    parser._option_string_actions["--ddo-backend"].choices = ("vmm",)
    parser._option_string_actions["--ddo-backend"].help = "Native VMM backend for this minimal example."
    parser._option_string_actions["--ddo-preset"].choices = ("auto",)
    parser._option_string_actions["--ddo-preset"].help = "Native fast preset selected by vmm/auto."


def minimal_vmm_config(args, device):
    """Resolve native configuration through DDO without changing auxiliary routes."""
    from diffusers_dynamic_offloader import resolve_dynamic_offload_backend

    route = resolve_dynamic_offload_backend(args.ddo_backend, args.ddo_preset)
    return replace(apply_execution_backing_reclaim(route.build_config_from_args(args), args),
                   execution_device=device, offload_device="cpu", vmm_native_transport=True)


def runner_settings_environ(args, environ=None):
    """Keep explicit selector precedence local to the settings constructor."""
    values = dict(os.environ if environ is None else environ)
    if getattr(args, "explicit_ddo_preset", False):
        values["DDO_PRESET"] = (
            getattr(args, "preset", "one_shot_fast") if getattr(args, "ddo_backend", "linear") == "vmm"
            else getattr(args, "preset", getattr(args, "ddo_preset", "auto"))
        )
    return values


def backend_reporting_fields(backend, settings):
    """Distinguish the transformer selector from its base settings preset."""
    return {
        "ddo_backend": backend.backend,
        "dynamic_offload_requested_preset": backend.requested_preset,
        "dynamic_offload_effective_preset": backend.effective_preset,
        "dynamic_offload_base_requested_preset": settings.requested_preset,
        "dynamic_offload_base_effective_preset": settings.effective_preset,
    }


def record_runner_report(args, tracker, report, component=None):
    """Use one detail-level meaning without changing snapshot/ownership collection."""
    if getattr(args, "metrics_level", 2) >= 2:
        tracker.record_ddo_metrics(report, component)


def purge_for_runner(args, settings_or_phase=None, phase=None, **kwargs):
    """Pass explicit CLI policy to DDO without touching the process environment."""
    from diffusers_dynamic_offloader import maybe_purge_windows_standby_cache

    resolved_phase = phase if phase is not None else settings_or_phase
    kwargs["enabled"] = getattr(args, "purge_" + resolved_phase, None)
    return maybe_purge_windows_standby_cache(settings_or_phase, phase, **kwargs)
