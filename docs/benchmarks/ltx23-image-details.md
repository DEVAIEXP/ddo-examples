# LTX 2.3 image — measurement details

[Back to comparison tables](ltx23-image.md).
Scenario links in those tables open the sections below; no local report download
is required to read the results.

**Labels and units:** read the [labels and memory glossary](glossary.md) before
comparing runtime names, phase timings, tracked totals or memory peaks.

## Workload and measurement scope

Distilled BF16, 1280×704, 8 steps, seed 43, real prompt, CFG 1/rescale 0.7, no PAG. Shared official leaf text encoder and resident untiled VAE in the primary comparison.

Checkpoint: `elismasilva/ltx2.3-image-distilled-1.1`, snapshot
`5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9`. Official Diffusers 0.41.0
in the primary platform comparisons. Encoder setup differs in the separately
labeled native-encoder VAE pairs. No LoRAs, compute compilation, PAG or tiling
are added to the primary cases.

The text-encoder row is the monitored `Pass 0: Encode prompts`: loading,
tokenizer/pipeline setup, encoding and cleanup, plus any purge inside that phase.
It is not the pure `encode_prompt_call` duration. Connectors run in the
`Pass 1: Generate` aggregate together with transformer loading, attachment,
denoise and cleanup. Its memory peak cannot be relabeled as a denoise-only peak.

The overview's denoise wall is `denoise_modular_pipe_call`. VAE rows use
separate load/decode monitors where recorded. INT4 reports instead monitor
`Pass 2: Decode VAE` as a load/decode/cleanup aggregate: it must not be
compared to a pure decode time as if their scopes were identical.

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

### Windows / `vmm_fast`

`--ddo-backend vmm --ddo-preset auto` resolves to `vmm_fast`, executing `vmm_fast_runtime`.

Fixed transformer capacity: `4` GiB active weights; `1536` MiB total staging. Retained source / registration ceilings: `24.625` / `24.625` GiB.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

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

Fixed transformer capacity: `4` GiB active weights; `1536` MiB total staging. Retained source / registration ceilings: `24.625` / `24.625` GiB.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

<a id="linux-nvme"></a>

### Linux / `nvme_baseline`

`--ddo-backend vmm --ddo-preset nvme_baseline` executes protected `vmm_runtime`.

Fixed transformer capacity: `4` GiB active weights; two `576` MiB staging slots. Retained source / registration ceilings: `0` / `0` GiB.
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

Fixed transformer capacity: `4` GiB active weights; `384` MiB total staging. Retained source / registration ceilings: `24.625` / `0.5` GiB.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

## VAE strategies — separate native-encoder workload

1280×704, 8 steps, seed 43. These pairs use a native text encoder, unlike the shared-leaf-encoder comparisons above.

<a id="vae-windows-resident"></a>

### Windows / `vmm_fast` / resident VAE

Both encoder and transformer use native VMM. Only the independently selected
VAE route changes within each platform pair; outputs match within each pair.
These runs predate the primary shared-leaf-encoder cases and are a separate
VAE workload, not replacement transformer speed measurements.

<a id="vae-windows-leaf"></a>

### Windows / `vmm_fast` / leaf VAE

Both encoder and transformer use native VMM. Only the independently selected
VAE route changes within each platform pair; outputs match within each pair.
These runs predate the primary shared-leaf-encoder cases and are a separate
VAE workload, not replacement transformer speed measurements.

<a id="vae-linux-resident"></a>

### Linux / `vmm_fast` / resident VAE

Both encoder and transformer use native VMM. Only the independently selected
VAE route changes within each platform pair; outputs match within each pair.
These runs predate the primary shared-leaf-encoder cases and are a separate
VAE workload, not replacement transformer speed measurements.

<a id="vae-linux-leaf"></a>

### Linux / `vmm_fast` / leaf VAE

Both encoder and transformer use native VMM. Only the independently selected
VAE route changes within each platform pair; outputs match within each pair.
These runs predate the primary shared-leaf-encoder cases and are a separate
VAE workload, not replacement transformer speed measurements.

## Windows SDNQ INT4

Saved INT4 transformer, 1280×704, 8 steps, seed 43; resident VAE. Compare within this pair, not as an isolated BF16-versus-INT4 speedup.

<a id="int4-ddo"></a>

### Windows / SDNQ / `one_shot_fast`

Saved LTX image INT4 transformer; BF16 auxiliaries, stock resident VAE and
official Diffusers 0.41.0. Encoder/setup choices differ from the BF16 tables.
`one_shot_fast` executes packed `sdnq_runtime`, preserving SDNQ math.
Normal PyTorch allocation, not native VMM. Outputs match within the INT4
pair; BF16-versus-INT4 quality equivalence was not established.

<a id="int4-leaf"></a>

### Windows / SDNQ / official leaf

Saved LTX image INT4 transformer; BF16 auxiliaries, stock resident VAE and
official Diffusers 0.41.0. Encoder/setup choices differ from the BF16 tables.
`diffusers_leaf_offload_compat` executes official streamed leaf with SDNQ math.
Normal PyTorch allocation, not native VMM. Outputs match within the INT4
pair; BF16-versus-INT4 quality equivalence was not established.

<a id="windows-warm-process"></a>

## Windows / `warm_process` repeated workload

1280×704 BF16, real prompt, eight steps per generation, two transformer
preparations and two generations in one process. Shared streamed leaf encoder
and resident untiled VAE. Initial seed 43; the shared generator advances between
generations, so the final image need not equal a single-generation seed-43 image.
The plan and pinned-weight cache are reused. The total covers both generations
and final decode, not one image. No comparative repeated-preset ranking follows.

## Installed rc3 functional validation

The same local DDO/allocator rc3 Linux wheels completed separate native-Linux
and WSL image checks without rebuilding. Both used installed packages, Python
3.12.12, Torch 2.12.1+cu132 and Diffusers 0.41.0. The workload was the BF16
checkpoint/revision above, 1280 x 704, eight steps, seed 43, the runner's real
cat prompt, CFG 1/rescale 0.7 and no PAG. Each run included one generation,
one transformer preparation and resident VAE decode. Transformer runtime was
`vmm_fast_runtime`, requested through `vmm/auto` with base `one_shot_fast`;
the text encoder used official group offload and VAE route was `none`.

| Target | Weight capacity (GiB) | Total staging (GiB) | Source storage (GiB) | Source registration (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Native Linux | 4.00 | 1.50 | 24.63 | 24.63 |
| WSL2 | 4.00 | 0.38 | 24.63 | 0.50 |

The table rounds for display. Exact reproduction values are weights `4`,
source cache `24.625` and pages `32` MiB on both platforms;
`--ddo-vmm-total-staging-gb 1.5 --ddo-vmm-source-registered-gb 24.625` on
native Linux, and `--ddo-vmm-total-staging-gb 0.375 --ddo-vmm-source-registered-gb 0.5`
on WSL. Use `scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py`, local
model files, `--ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto`,
`--preset one_shot_fast`, `--vmm-resident-weight-budget-gb 4`,
`--ddo-vmm-source-cache-gb 24.625 --ddo-vmm-page-mb 32`, dimensions/steps/seed
above, one pre-VAE cleanup and level-2 metrics. No compression, activation
eviction, ETW, GPU timing or automatic execution-backing reclaim was enabled.
Windows standby purge is a no-op on these targets. The controller allowed
1,800 s with 10 s progress and 16 MiB per log stream, without retry or automatic
termination. The [runner guide](../user/ltx2-image.md) explains total staging.

| Target / observation | Denoise wall (s) | Runner tracked total (s) |
| --- | ---: | ---: |
| Native Linux rc3 | 14.65 | 33.26 |
| Native Linux retained rc2 first | 13.99 | 32.49 |
| Native Linux retained rc2 repeat | 12.98 | 27.41 |
| WSL rc3 | 89.01 | 278.12 |
| WSL retained rc2 | 97.44 | 271.72 |

Each rc3 run exited 0 after eight steps and decode. Its PNG and decoded RGB
matched the retained platform reference exactly. Available pre-close native
failure and quarantine counters were zero; decode and cleanup returned.
Final native counters, latent finiteness and final driver-memory release were
not established by the unchanged runner. Native Linux emitted a preserved
group-offload lazy-prefetch warning for unused vision-tower layers; this
text-prompt check does not validate image-conditioned vision-tower execution.

These single historical observations close selected functional checks, not
statistical performance acceptance. Platform capacities, host/cache conditions
and registration differ; do not use the table as a Linux-versus-WSL ranking or
a causal speed comparison. Neither full video nor arbitrary models are covered.
Rejected preparations and staging-mismatch captures were retained separately
and are excluded from these accepted observations. Package availability and
artifact delivery remain separate release gates.

## Reproduce this workload

Use the [runner guide](../user/ltx2-image.md) and
[comparison controls](reproduction.md), retaining the model, schedule, inputs,
component routes and the scenario-specific capacities above. Runner defaults
or a different custom-block revision do not guarantee equivalent conditions.
Review trusted custom-block code and keep platform environments separate.
Package/build validation and raw engineering investigations are developer concerns,
not additional timing rows or prerequisites for interpreting these tables.
