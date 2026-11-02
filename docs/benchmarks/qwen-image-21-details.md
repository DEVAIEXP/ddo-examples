# Qwen Image 2.1 — measurement details

[Back to comparison tables](qwen-image-21.md).
Scenario links in those tables open the sections below; no local report download
is required to read the results.

**Labels and units:** read the [labels and memory glossary](glossary.md) before
comparing runtime names, phase timings, tracked totals or memory peaks.

## Workload and measurement scope

Base BF16, 512×512, 20 steps, seed 42, capybara prompt, CFG 1 and KV cache. Shared official leaf text encoder; resident untiled VAE on Windows/Linux, untiled non-streamed leaf VAE on WSL.

Checkpoint: `Qwen/Qwen-Image-2.1`, snapshot
`d26bb61231c349cf6b7896fa83353113880e1ba3`. Official Diffusers 0.41.0
in the primary platform comparisons. The prompt is
"A capybara wearing a wizard hat, reading by candlelight, oil painting".
CFG 1, KV cache enabled. No compute compilation or local VAE math changes
in the primary 512-square cases. The separate 1024-square cases retain their
explicit local VAE/cleanup settings.

Text encoder load, attachment, `encode` and cleanup are separate monitored
phases. The text-encode number excludes the neighboring load/cleanup rows.
Denoise includes the runner's completion/CPU latent readback boundary.
Transformer attachment is separate from denoise. VAE loading is separate from
the `decode` phase.

All primary controls are staged modular runners. The official leaf control is
not the older monolithic pipeline whose encoder/denoise/decode could not be
separated. Baseline-relative console VRAM must not replace absolute driver peaks.

All displayed phase rows are existing observations, not new runs or inferred
phase estimates. Memory comes from each recorded phase's
`peak_vram_absolute_gb` and `peak_ram_gb` with `ram_metric=rss`.
Times use `elapsed_sec`; global peaks and tracked totals come from the same
report as that row. Values are rounded to two decimals for display only.
Missing fields remain unavailable. Phase tables show selected boundaries, not
an additive decomposition; do not add aggregates to their child events or sum
memory peaks. Host callbacks do not establish completed warm GPU timing.
See [metric definitions](glossary.md).

## Platform conditions

Windows comparisons recorded successful standby purge before the transformer;
phase-local purge can also be included in tracked totals. Linux and WSL do not
execute the Windows purge. WSL auxiliary leaf hooks are non-streamed and native
registration is bounded; those routes are not equivalent to Windows/Linux
streamed controls. Do not treat different platform observations as causal A/Bs.

Linux protected NVMe observations dated 2026-10-13 use driver 595.99.02 and a
fresh post-reboot/cache session, unlike the earlier route trios. This prevents
attributing their timing difference solely to the allocator. Primary cases are
single observations, not repeated statistical estimates or low-RAM certification.
The compared completed outputs matched within the recorded route sets; this is
not a universal quality or cross-platform byte-identity claim.

## BF16 scenarios

<a id="windows-leaf"></a>

### Windows / Official leaf

The `diffusers_leaf_offload_compat` control executes official Diffusers leaf hooks, not DDO dynamic denoising.


<a id="windows-linear"></a>

### Windows / `one_shot_fast`

`--ddo-backend linear --ddo-preset one_shot_fast` executes `linear_runtime`.


<a id="windows-low-ram"></a>

### Windows / `low_ram_safe`

`--ddo-backend linear --ddo-preset low_ram_safe` executes `linear_runtime`.


<a id="windows-native"></a>

### Windows / `vmm_fast` (reused observation)

`--ddo-backend vmm --ddo-preset auto` resolves to `vmm_fast`, executing `vmm_fast_runtime`.

Fixed transformer capacity: `4` GiB active weights; `640` MiB total staging. Retained source / registration ceilings: `16` / `16` GiB.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.
This is the reused official-package native observation, not a repeat alongside
the Windows leaf/linear runs. Matching workload fields and output do not prove
identical source-checkout or driver/cache state.

<a id="windows-nvme"></a>

### Windows / `nvme_baseline`

`--ddo-backend vmm --ddo-preset nvme_baseline` executes protected `vmm_runtime`.

Fixed transformer capacity: `4` GiB active weights; two `640` MiB staging slots. Retained source / registration ceilings: `0` / `0` GiB.
Normal PyTorch allocation plus native weight arena, not the process-wide native
Torch allocator. Zero persistent source cache is not zero OS file-cache usage.

<a id="linux-leaf"></a>

### Linux / Official leaf

The `diffusers_leaf_offload_compat` control executes official Diffusers leaf hooks, not DDO dynamic denoising.


<a id="linux-linear"></a>

### Linux / `one_shot_fast`

`--ddo-backend linear --ddo-preset one_shot_fast` executes `linear_runtime`.


<a id="linux-native"></a>

### Linux / `vmm_fast`

`--ddo-backend vmm --ddo-preset auto` resolves to `vmm_fast`, executing `vmm_fast_runtime`.

Fixed transformer capacity: `4` GiB active weights; `640` MiB total staging. Retained source / registration ceilings: `16` / `16` GiB.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

<a id="linux-nvme"></a>

### Linux / `nvme_baseline`

`--ddo-backend vmm --ddo-preset nvme_baseline` executes protected `vmm_runtime`.

Fixed transformer capacity: `4` GiB active weights; two `448` MiB staging slots. Retained source / registration ceilings: `0` / `0` GiB.
Normal PyTorch allocation plus native weight arena, not the process-wide native
Torch allocator. Zero persistent source cache is not zero OS file-cache usage.

<a id="wsl-leaf"></a>

### WSL / Official leaf (no stream)

The `diffusers_leaf_offload_compat` control executes official Diffusers leaf hooks, not DDO dynamic denoising.


<a id="wsl-linear"></a>

### WSL / `wsl_compat` (3 GiB override)

`--ddo-backend linear --ddo-preset wsl_compat` executes `linear_runtime`.

The resident-module budget is explicitly 3 GiB, CPU pinning is disabled and
automatic budgeting is off. This validates the override, not preset-default
capacity; a configured ceiling is not measured residency.

<a id="wsl-native"></a>

### WSL / `vmm_fast` (bounded)

`--ddo-backend vmm --ddo-preset auto` resolves to `vmm_fast`, executing `vmm_fast_runtime`.

Fixed transformer capacity: `4` GiB active weights; `384` MiB total staging. Retained source / registration ceilings: `16` / `0.5` GiB.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

<a id="wsl-nvme"></a>

### WSL / `nvme_baseline`

`--ddo-backend vmm --ddo-preset nvme_baseline` executes protected `vmm_runtime`.

Fixed transformer capacity: `4` GiB active weights; two `448` MiB staging slots. Retained source / registration ceilings: `0` / `0` GiB.
Normal PyTorch allocation plus native weight arena, not the process-wide native
Torch allocator. Zero persistent source cache is not zero OS file-cache usage.

## Windows 1024×1024 — separate VAE settings

Base BF16, 20 steps, seed 42; untiled leaf VAE. Native uses explicit cache trimming; this is not an allocator-only A/B.

<a id="windows-1024-linear"></a>

### Windows / `one_shot_fast` / leaf VAE

Same base model, prompt, 20 steps and seed 42 as the 512-square workload,
but resolution is 1024×1024. Leaf untiled VAE uses explicitly accepted local
options; do not attribute their effects to transformer offload alone.

<a id="windows-1024-native"></a>

### Windows / `vmm_fast` / leaf VAE + trim

Same base model, prompt, 20 steps and seed 42 as the 512-square workload,
but resolution is 1024×1024. Leaf untiled VAE uses explicitly accepted local
options; do not attribute their effects to transformer offload alone.
This case explicitly trims native idle cache before VAE. Native trim and
cuBLAS workspace clearing are not implied by selecting a VAE preset.

## Windows SDNQ INT4

Base 512×512, 20 steps, seed 42; BF16 auxiliaries, resident stock VAE. Output parity is within the INT4 pair, not a BF16 quality-equivalence claim.

<a id="int4-ddo"></a>

### Windows / SDNQ / `one_shot_fast`

Saved checkpoint `OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit`, snapshot
`59d4c3610379d91fb023c4d0b6527c76755491c4`, SDNQ 0.2.7 and official
Diffusers 0.41.0. BF16 auxiliaries and stock resident VAE remain unchanged.
`one_shot_fast` executes packed `sdnq_runtime`, preserving SDNQ math.
Normal PyTorch allocation, not native VMM. Outputs match within the INT4
pair; BF16-versus-INT4 quality equivalence was not established.

<a id="int4-leaf"></a>

### Windows / SDNQ / official leaf

Saved checkpoint `OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit`, snapshot
`59d4c3610379d91fb023c4d0b6527c76755491c4`, SDNQ 0.2.7 and official
Diffusers 0.41.0. BF16 auxiliaries and stock resident VAE remain unchanged.
`diffusers_leaf_offload_compat` executes official streamed leaf with SDNQ math.
Normal PyTorch allocation, not native VMM. Outputs match within the INT4
pair; BF16-versus-INT4 quality equivalence was not established.

## Reproduce this workload

Use the [runner guide](../user/qwen-image-21.md) and
[comparison controls](reproduction.md), retaining the model, schedule, inputs,
component routes and the scenario-specific capacities above. Runner defaults
or a different custom-block revision do not guarantee equivalent conditions.
Review trusted custom-block code and keep platform environments separate.
Package/build validation and raw engineering investigations are developer concerns,
not additional timing rows or prerequisites for interpreting these tables.
