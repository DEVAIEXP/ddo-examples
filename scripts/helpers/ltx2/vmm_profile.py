"""VMM residency profile selection for the LTX video VMM runners.

Thin LTX video binding of ``scripts/helpers/vmm_profile.py``: the automatic
name carries resolution, frames and the options that change the activation peak.
"""
from scripts.helpers import vmm_profile as _generic
from scripts.helpers.vmm_profile import PROFILE_AUTO, add_vmm_profile_arguments  # noqa: F401  (public API)

VIDEO_ACTIVATION_OPTIONS = (
    "feed_forward_chunk_size", "normalization_chunk_size", "video_self_attention", "phase_modulation",
    "compact_timesteps", "inplace_residuals", "fused_rms_modulation", "fused_rotary_pair",
    "use_audio_condition", "max_sequence_length")


def workload_profile_name(args, num_frames):
    """Readable workload shape plus a digest of the options that change activations."""
    return _generic.workload_profile_name("ltx23-video", f"{args.width}x{args.height}-{num_frames}f", args,
                                          VIDEO_ACTIVATION_OPTIONS)


def resolve_vmm_profile(parser, args, num_frames):
    """Validate the profile options and resolve 'auto'; returns the name or None."""
    return _generic.resolve_vmm_profile(parser, args, workload_profile_name(args, num_frames))
