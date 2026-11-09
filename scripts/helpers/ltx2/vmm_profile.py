"""VMM residency profile selection for the LTX video VMM runners.

DDO owns measurement, recommendation, identity (model, component, machine
RAM/VRAM, native library) and storage; it does not infer the workload shape.
The runner therefore names the profile from the arguments that change the
activation peak, so a 1280x704 profile is never reused at 1920x1056.
"""
import hashlib
import json

PROFILE_AUTO = "auto"


def add_vmm_profile_arguments(parser):
    parser.add_argument(
        "--ddo-profile",
        help="VMM residency profile name; 'auto' derives it from resolution, frames and activation options. "
             "Without an explicit --ddo-vmm-resident-weight-budget-gb the saved budget is used.")
    parser.add_argument(
        "--build-ddo-profile", action="store_true",
        help="Measure the denoise and stop before decode. Run it twice with --steps 2: the first build measures "
             "the structural floor, the second the hardware ceiling; DDO keeps the ceiling only if it is worth "
             "its memory.")


def workload_profile_name(args, num_frames):
    """Readable workload shape plus a digest of the options that change activations."""
    options = {name: getattr(args, name, None) for name in (
        "feed_forward_chunk_size", "normalization_chunk_size", "video_self_attention", "phase_modulation",
        "compact_timesteps", "inplace_residuals", "fused_rms_modulation", "fused_rotary_pair",
        "use_audio_condition", "max_sequence_length")}
    digest = hashlib.sha256(json.dumps(options, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"ltx23-video-{args.width}x{args.height}-{num_frames}f-{digest}"


def resolve_vmm_profile(parser, args, num_frames):
    """Validate the profile options and resolve 'auto'; returns the name or None."""
    name = getattr(args, "ddo_profile", None)
    if name is not None and not name.strip():
        parser.error("--ddo-profile must be a nonempty name")
    if args.build_ddo_profile and not name:
        parser.error("--build-ddo-profile requires --ddo-profile NAME (or auto)")
    if name and (args.ddo_backend != "vmm" or args.ddo_preset != "auto"):
        parser.error("VMM residency profiles require --ddo-backend vmm --ddo-preset auto")
    if name == PROFILE_AUTO:
        name = workload_profile_name(args, num_frames)
    args.ddo_profile = name
    return name
