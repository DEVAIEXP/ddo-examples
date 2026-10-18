# Choose a runtime and preset

Choose the transformer route with `--ddo-backend` and `--ddo-preset`.
The runner's `--help` lists the choices it implements.

| Selection | Use |
| --- | --- |
| `linear / auto` | Default PyTorch-backed DDO route, equivalent to `one_shot_fast` |
| `linear / one_shot_fast` | Prioritize a single generation's speed |
| `linear / low_ram_safe` | More conservative residency; not a guarantee of the lowest measured RAM peak |
| `linear / warm_process` | Repeated use in a persistent process |
| `linear / wsl_compat` | WSL-oriented linear configuration |
| `linear / diffusers_leaf_offload_compat` | Official Diffusers leaf offload comparison |
| `vmm / auto` or `vmm / vmm_fast` | Experimental native allocator; requires a compatible native package |
| `vmm / nvme_baseline` | Protected NVMe-backed VMM route |

SDNQ has a dedicated runner and compatible loading route. Selecting a dense or
native preset does not make an arbitrary quantized checkpoint compatible.

## Match the hardware

With limited VRAM and ample system RAM, compare native VMM against linear DDO
for your model. Native VMM retains source data in RAM; it is not the low-RAM
choice merely because GPU weights are offloaded.

With limited RAM as well as VRAM, consider `low_ram_safe` or the NVMe route,
but check the measured RAM peak and staging requirements first. The published
tests do not certify operation on every low-memory machine.

For speed, use the recommendation beside the model's
[benchmark table](../benchmarks/README.md). No preset wins for every model,
precision or platform. WSL has separate registration limits and measurements.

## VAE choices

Transformer and VAE strategies can be combined. Where implemented,
`--vae-resident`, `--vae-leaf-offload` and `--vae-linear-offload` override
the preset's VAE choice. Native VAE needs its own explicit setup.

Without an override, auto/one_shot_fast/low_ram_safe/vmm_fast select no-stream
leaf for the VAE; warm_process/wsl_compat/nvme_baseline preserve resident staged
execution. A runner-specific decoder can impose additional requirements.

Offloading reduces weight residency, not necessarily activation memory.
If the VAE still does not fit, use spatial tiling or the supported temporal
streaming decoder. These choices can change decode time and output.
