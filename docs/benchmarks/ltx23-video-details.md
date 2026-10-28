# LTX 2.3 video — measurement details

[Back to comparison tables](ltx23-video.md).
Scenario links in those tables open the sections below; no local report download
is required to read the results.

## Workload and measurement scope

Distilled BF16, 1280×704, 5 seconds at 24 fps (121 frames), 8 steps, seed 43; real guitar image/prompt, no input audio, generated stereo audio. Shared official leaf encoder/auxiliaries and resident temporal GPU VAE decode.

Checkpoint: `OzzyGT/LTX-2.3-Distilled-1.1`, BF16 snapshot
`f070fb11c2af41e90ad9012d1f495b5131753a4d`. Official Diffusers 0.41.0
in the primary platform comparisons. Feed-forward and normalization chunks
1024, phase modulation, compact timesteps, inplace residuals and RoPE reuse
are retained. No input audio, LoRAs, compilation or experimental kernels.
Output includes generated stereo audio. The separately labeled conditioned
WSL run and FP8/VAE pairs must not be substituted into this workload.

Text encoder, connectors, conditioning preparation, transformer load/prepare,
attachment and denoise are separately monitored `steps`. The text-encoder
phase contains its staged loading/encoding/cleanup, not only the model forward.
The denoise row is `steps[name=denoise]`; it is not a sum of completed
per-step GPU kernels.

Video VAE, audio VAE and vocoder are separate decode stages where the report
provides them. Older FP8 precision-pair reports expose `decode_and_save`
only: that aggregate includes more than video VAE execution. Temporal GPU
streaming uses CPU latent storage/stitching; it is not spatial VAE tiling.

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

Fixed transformer capacity: `1.5` GiB active weights; `1536` MiB total staging. Retained source / registration ceilings: `24.625` / `24.625` GiB.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

<a id="windows-nvme"></a>

### Windows / `nvme_baseline`

`--ddo-backend vmm --ddo-preset nvme_baseline` executes protected `vmm_runtime`.

Fixed transformer capacity: `1.5` GiB active weights; two `768` MiB staging slots. Retained source / registration ceilings: `0` / `0` GiB.
Normal PyTorch allocation plus native weight arena, not the process-wide native
Torch allocator. Zero persistent source cache is not zero OS file-cache usage.

<a id="linux-leaf"></a>

### Linux / Official leaf

The `diffusers_leaf_offload_compat` control executes official Diffusers leaf hooks, not DDO dynamic denoising.


<a id="linux-linear"></a>

### Linux / `one_shot_fast` + host-cache release

`--ddo-backend linear --ddo-preset one_shot_fast` executes `linear_runtime`.

This case explicitly releases idle PyTorch host cache before VAE loading.
That cleanup choice is part of the measured route, not an automatic new default.

<a id="linux-native"></a>

### Linux / `vmm_fast`

`--ddo-backend vmm --ddo-preset auto` resolves to `vmm_fast`, executing `vmm_fast_runtime`.

Fixed transformer capacity: `1.5` GiB active weights; `1536` MiB total staging. Retained source / registration ceilings: `24.625` / `24.625` GiB.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

<a id="linux-nvme"></a>

### Linux / `nvme_baseline`

`--ddo-backend vmm --ddo-preset nvme_baseline` executes protected `vmm_runtime`.

Fixed transformer capacity: `1.5` GiB active weights; two `768` MiB staging slots. Retained source / registration ceilings: `0` / `0` GiB.
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

Fixed transformer capacity: `1.5` GiB active weights; `384` MiB total staging. Retained source / registration ceilings: `24.625` / `0.5` GiB.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

## VAE strategies — separate native-encoder workload

Five seconds, 1280×704, 24 fps, 8 steps, seed 43. Native text encoder; temporal GPU streaming in both decoder routes.

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

## Saved FP8 weight-only — separate precision pairs

Five-second video; auxiliary components remain unchanged within each platform pair. Lower transformer storage does not imply faster denoising or lower peak GPU usage.

<a id="fp8-windows-bf16"></a>

### Windows / `vmm_fast` / BF16 control

Only transformer precision changes within the platform pair; auxiliary
weights, input and decoder route remain fixed. Saved TorchAO FP8 weight-only
retains logical BF16 math and backend dequantization, not native FP8 GEMM.
This separate Windows precision pair uses the earlier native-encoder route;
it is not the primary shared-encoder BF16 row. Decode/save is an aggregate.
FP8 does not match BF16 output byte-for-byte; perceptual quality was not scored.
These pairs do not establish a matched same-FP8 official-offload speed comparison.

<a id="fp8-windows"></a>

### Windows / `vmm_fast` / saved FP8

Only transformer precision changes within the platform pair; auxiliary
weights, input and decoder route remain fixed. Saved TorchAO FP8 weight-only
retains logical BF16 math and backend dequantization, not native FP8 GEMM.
This separate Windows precision pair uses the earlier native-encoder route;
it is not the primary shared-encoder BF16 row. Decode/save is an aggregate.
FP8 does not match BF16 output byte-for-byte; perceptual quality was not scored.
These pairs do not establish a matched same-FP8 official-offload speed comparison.

<a id="fp8-linux-bf16"></a>

### Linux / `vmm_fast` / BF16 control

Only transformer precision changes within the platform pair; auxiliary
weights, input and decoder route remain fixed. Saved TorchAO FP8 weight-only
retains logical BF16 math and backend dequantization, not native FP8 GEMM.
TorchAO 0.18.0, per-row E4M3 payload, native active/staging limits 1.5/1.5 GiB
and source/registration ceilings `24.625` GiB each. No FP8 Linear compilation,
conversion reuse or extra cache trimming.
FP8 does not match BF16 output byte-for-byte; perceptual quality was not scored.
These pairs do not establish a matched same-FP8 official-offload speed comparison.

<a id="fp8-linux"></a>

### Linux / `vmm_fast` / saved FP8

Only transformer precision changes within the platform pair; auxiliary
weights, input and decoder route remain fixed. Saved TorchAO FP8 weight-only
retains logical BF16 math and backend dequantization, not native FP8 GEMM.
TorchAO 0.18.0, per-row E4M3 payload, native active/staging limits 1.5/1.5 GiB
and source/registration ceilings `24.625` GiB each. No FP8 Linear compilation,
conversion reuse or extra cache trimming.
FP8 does not match BF16 output byte-for-byte; perceptual quality was not scored.
These pairs do not establish a matched same-FP8 official-offload speed comparison.

## WSL video with conditioning audio

Separate five-second workload with guitar image and input audio; not matched to the no-input-audio WSL comparison above.

<a id="wsl-conditioned"></a>

### WSL / `vmm_fast` / image + audio

Guitar image plus conditioning audio, five seconds, native encoder and
transformer; 121 frames, stereo 48 kHz output. Native active ceilings are
4 GiB for the encoder and 1.5 GiB for the transformer, with 384 MiB staging,
`24.625` GiB source storage and 512 MiB registration. This input/encoder
scope differs from the primary WSL no-input-audio trio.

## Residual CPU source detachment

Selected Windows validation on 2026-10-28 used distilled BF16 LTX 2.3, a
1920x1056 image with conditioning audio, 10.33 seconds at 24 fps, seed 2925624786
and the first two scheduler steps of an eight-step schedule. Both complete
transformer forwards contained 48 blocks; no third step or decode executed.
This is separate from the 1280x704 overview benchmark. The transformer executed
`vmm_fast_runtime` with an experimental execution-backing pressure DLL; this does
not promote that artifact or change an installed preset. VAE execution was absent.

The old weight-loading route and 1.5 GiB weight/staging budgets, 24.625 GiB
source/registration ceilings and 4 MiB allocator cache were retained. The single
loading intervention explicitly called the existing guarded
`detach_auxiliary_cpu_sources(enabled=True, max_copy_bytes=16 * 1024**2)` after
attachment and before the first forward. The model was exclusively owned and
in evaluation mode. Native readiness, CPU source ownership, alias/layout and
source provenance guards remained active. There was no compression or activation
eviction. The runner's existing option is
`--detach-transformer-auxiliary-cpu-sources`; this result does not enable it by
default or authorize arbitrary experimental runner controls.

The operation copied 7.88 MiB across 192 residual CPU groups and released their
large checkpoint mapping owners. Within that operation, process private commit
fell by 35.41 GiB. Full video/audio latents after scheduler step index one were
finite and bitwise identical to the preserved accepted reference, with zero
mismatched elements. Native/compute closure and scheduler restoration passed.

Windows process private commit and system commit below are **sampled peaks**,
in GiB, at 500 ms intervals. They are not RSS or local VRAM usage. System totals
include unrelated processes. Headroom is the sampled minimum of system commit
limit minus system commit. See [metric definitions](glossary.md).

| Observation scope | Peak process private commit (GiB) | Peak system commit (GiB) | Minimum system commit headroom (GiB) |
| --- | ---: | ---: | ---: |
| Full observed capture including loading | 48.08 | 71.96 | 23.96 |
| Two-step denoising interval | 42.07 | 59.18 | 36.74 |

The transient loading peak precedes detachment and remains. Correctness-capture
denoising time was 349.46 s, including final output readback, versus 352.25 s for
the earlier reference. This is not a controlled speed comparison or performance
acceptance. The demonstrated benefit is commit relief with selected exact-output
equivalence; no full eight-step/decode, generic CUDA stability, other model,
platform or installed-wheel acceptance is established.

Original evidence remains archived locally under
`storage-detach-two-step-enabled-4a110a9835f74117a631e804434657fa`, manifest
`06e66a9c4a8f87d6287d392c9d94877d9b6229d2e57652295a2d48469266c472`.
Reference: `execution-backing-two-step-numerical-enabled-35ad11099d7546e194864e11916f2dd9`.
These identities are archival evidence, not files required by public users.

## Reproduce this workload

Use the [runner guide](../user/ltx2-video.md) and
[comparison controls](reproduction.md), retaining the model, schedule, inputs,
component routes and the scenario-specific capacities above. Runner defaults
or a different custom-block revision do not guarantee equivalent conditions.
Review trusted custom-block code and keep platform environments separate.
Package/build validation and raw engineering investigations are developer concerns,
not additional timing rows or prerequisites for interpreting these tables.
