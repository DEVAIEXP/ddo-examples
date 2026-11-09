"""VMM residency profile options shared by the native VMM runners.

DDO owns measurement, recommendation, identity (model, component, machine
RAM/VRAM, native library) and storage; it does not infer the workload shape.
Each runner names the profile from the arguments that change the activation
peak, so a profile built for one resolution or batch layout is never reused
for another.
"""
import hashlib
import json

PROFILE_AUTO = "auto"
BUILD_STEPS = 3


def add_vmm_profile_arguments(parser):
    parser.add_argument(
        "--ddo-profile",
        help="VMM residency profile name; 'auto' derives it from the workload shape and activation options. "
             "Without an explicit --ddo-vmm-resident-weight-budget-gb the saved budget is used.")
    parser.add_argument(
        "--build-ddo-profile", action="store_true",
        help=f"Measure the denoise and stop before decode. Run it twice with --steps {BUILD_STEPS}: the first "
             "build measures the structural floor, the second the hardware ceiling; DDO keeps the ceiling only "
             "if its gain is worth its memory and larger than the builds' timing noise.")


def workload_profile_name(prefix, shape, args, option_names):
    """Readable workload shape plus a digest of the options that change activations."""
    options = {name: getattr(args, name, None) for name in option_names}
    digest = hashlib.sha256(json.dumps(options, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{prefix}-{shape}-{digest}"


def resolve_vmm_profile(parser, args, auto_name, *, backend=None, preset=None):
    """Validate the profile options and resolve 'auto' to ``auto_name``; returns the name or None."""
    name = getattr(args, "ddo_profile", None)
    if name is not None and not name.strip():
        parser.error("--ddo-profile must be a nonempty name")
    if args.build_ddo_profile and not name:
        parser.error("--build-ddo-profile requires --ddo-profile NAME (or auto)")
    backend = args.ddo_backend if backend is None else backend
    preset = args.ddo_preset if preset is None else preset
    if name and (backend != "vmm" or preset != "auto"):
        parser.error("VMM residency profiles require --ddo-backend vmm --ddo-preset auto")
    if name == PROFILE_AUTO:
        name = auto_name
    args.ddo_profile = name
    return name
