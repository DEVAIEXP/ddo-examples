# Runtimes and presets

Choose the transformer route with `--ddo-backend` and `--ddo-preset`.
The runner's `--help` lists the choices it implements.

The combination of **backend** and **preset** determines the transformer
runtime. The backend selects the execution path; the preset selects its memory
and transfer settings. The runtime is the implementation that actually executes
the transformer. Encoder and VAE routes are chosen separately.

## Transformer runtimes

The mapping below applies to dense BF16/FP16 weights. Saved quantized weights
need a compatible loading route and backend forward.

| Runner parameters | Resolved preset | Executed runtime | Use |
| --- | --- | --- | --- |
| `--ddo-backend linear --ddo-preset auto` | `one_shot_fast` | `linear_runtime` | Default PyTorch-backed DDO route |
| `--ddo-backend linear --ddo-preset one_shot_fast` | `one_shot_fast` | `linear_runtime` | Prioritize a single generation's speed |
| `--ddo-backend linear --ddo-preset low_ram_safe` | `low_ram_safe` | `linear_runtime` | More conservative residency; not a guarantee of the lowest measured RAM peak |
| `--ddo-backend linear --ddo-preset wsl_compat` | `wsl_compat` | `linear_runtime` | WSL-oriented linear configuration |
| `--ddo-backend linear --ddo-preset warm_process` | `warm_process` | `linear_runtime` | Repeated use in a persistent process; compare two-generation workloads separately |
| `--ddo-backend vmm --ddo-preset auto` | `vmm_fast` | `vmm_fast_runtime` | VMM Fast; requires a compatible native package |
| `--ddo-backend vmm --ddo-preset nvme_baseline` | `nvme_baseline` | `vmm_runtime` | Experimental NVMe-backed alternative; may be discontinued in future releases |

`auto` depends on the backend: it selects `one_shot_fast` with `linear` and
VMM Fast (`vmm_fast`) with `vmm`. The benchmarks use the resolved preset name.
To reproduce a VMM Fast row, pass `--ddo-backend vmm --ddo-preset auto`;
`vmm_fast` is not an accepted public `--ddo-preset` value with the VMM backend.

### Compatibility and saved quantized weights

With the linear backend, `--ddo-preset diffusers_offload_compat` selects
Diffusers block group offload and `--ddo-preset diffusers_leaf_offload_compat`
selects Diffusers leaf group offload. Both preserve Diffusers hooks and forwards
through plan mode. Where supported, `off` leaves placement to the caller without
DDO hooks.

SDNQ has a dedicated runner: `--ddo-preset one_shot_fast` selects its packed
`sdnq_runtime`, preserving the SDNQ forward with normal CUDA allocation. Use the
model's SDNQ recipe; selecting a dense or native preset does not make an
arbitrary quantized checkpoint compatible.

In benchmark tables, the requested selector and executed runtime are separate
fields. A settings label alone does not prove which runtime ran: a native
runner can retain a `one_shot_fast` base-settings label while executing
`vmm_fast_runtime`. A standalone official Diffusers runner has no DDO preset.

## NVMe baseline status

`nvme_baseline` remains available as an experimental alternative for users
who want to evaluate disk-backed weight loading on their hardware. It may
be discontinued in future releases. For new native VMM workloads, start
with VMM Fast and consult the model's benchmark recommendations.

The published comparisons do not establish a speed advantage for
`nvme_baseline` over VMM Fast. Its usefulness on machines with less system
RAM still needs validation. The model guides retain commands for users who
want to test this route.

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

| VAE route | Meaning |
| --- | --- |
| Resident | Decoder weights remain on the execution GPU during decode |
| Leaf | Diffusers uploads decoder modules as needed; the default preset VAE route uses no-stream Leaf |
| DDO linear convolution | Explicit normal-allocator convolution adapter |
| Native VMM VAE | Explicit native decoder source and capacity setup, independent of the transformer |
| Tiling / temporal streaming | Strategies for limiting decoder working memory, rather than transformer presets |

A resident VAE can accompany a linear, VMM Fast or official Leaf transformer.
It does not mean the transformer is resident. Native bootstrap also changes
the process-wide CUDA allocator, so equal VAE routes alone do not establish
fully matched memory conditions.

Without an override, auto/one_shot_fast/low_ram_safe/vmm_fast select no-stream
leaf for the VAE; warm_process/wsl_compat/nvme_baseline preserve resident staged
execution. A runner-specific decoder can impose additional requirements.

Offloading reduces weight residency, not necessarily activation memory.
If the VAE still does not fit, use spatial tiling or the supported temporal
streaming decoder. These choices can change decode time and output.

## Read the benchmark comparisons

Model pages record the executed transformer runtime, requested selector and
component routes. Use their measured workload and platform when choosing a
preset. An unmeasured route is not a failed test or a performance result.
Missing measurements remain unavailable; a result with a different prompt,
checkpoint or timing scope is not a matched comparison.
