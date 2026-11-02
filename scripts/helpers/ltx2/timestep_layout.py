"""CPU admission for compact video timesteps in the trusted LTX blocks."""


def compact_timestep_repeat(mask, *, latent_frames, latent_height, latent_width, patch_size=1, patch_size_t=1):
    """Use compact timesteps only for a known, spatially uniform packed layout."""
    import torch
    if mask.device.type != "cpu":
        raise ValueError("Compact timestep admission requires a CPU mask")
    if patch_size_t != 1 or latent_height % patch_size or latent_width % patch_size:
        return 1
    repeat = (latent_height // patch_size) * (latent_width // patch_size)
    if repeat <= 1 or mask.ndim != 3 or mask.shape[-1] != 1 or mask.shape[1] != latent_frames * repeat:
        return 1
    grouped = mask.reshape(mask.shape[0], latent_frames, repeat, 1)
    if not torch.isfinite(mask).all() or not torch.equal(grouped, grouped[:, :, :1].expand_as(grouped)):
        return 1
    return repeat
