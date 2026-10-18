"""Shared CLI vocabulary and lifecycle calls, not model orchestration."""

import argparse
import math
import os
import sys


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


def parse_common_arguments(parser, argv=None, *, purge_phases=("before_run", "after_text_encoder", "before_transformer", "after_run")):
    """Normalize controls already supported by an entry; preserve old defaults.

    Minimal entries still expose only their implemented controls. Legacy option
    destinations remain intact so model orchestration does not need a dispatcher.
    """
    from diffusers_dynamic_offloader import add_dynamic_offload_lifecycle_arguments

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
    image_slots = "--vmm-staging-buffer-gb" in parser._option_string_actions
    if image_slots:
        parser.add_argument("--ddo-vmm-total-staging-gb", type=float, default=None,
                            help="Total staging capacity in GiB; omitted preserves legacy per-slot defaults.")
    args = parser.parse_args(argv)
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


def runner_settings_environ(args, environ=None):
    """Keep explicit selector precedence local to the settings constructor."""
    values = dict(os.environ if environ is None else environ)
    if getattr(args, "explicit_ddo_preset", False):
        values["DDO_PRESET"] = (
            getattr(args, "preset", "one_shot_fast") if getattr(args, "ddo_backend", "linear") == "vmm"
            else getattr(args, "preset", getattr(args, "ddo_preset", "auto"))
        )
    return values


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
