"""Host-cache policy at the linear video runner's completed phase boundary."""

import platform
import sys


def default_host_cache_release(preset):
    """Enable native Linux cleanup for DDO linear presets; preserve compatibility routes."""
    return (sys.platform == "linux" and "microsoft" not in platform.release().lower()
            and preset not in {"off", "diffusers_offload_compat", "diffusers_leaf_offload_compat"})


def release_host_cache_before_vae(tracker, device):
    """Use the public DDO cleanup report after transformer owners are retired."""
    from diffusers_dynamic_offloader import release_pytorch_host_cache

    report = release_pytorch_host_cache(enabled=True, phase="before_vae", device=device)
    observation = report.as_dict()["setup"]["host_cache_release"]
    tracker.record_ddo_metrics(report, component="host_cache_before_vae")
    tracker.record_event("release_pytorch_host_cache_before_vae", observation["elapsed_seconds"],
                         status=observation["status"],
                         allocator_bytes_delta=observation.get("allocator_bytes_delta"))
