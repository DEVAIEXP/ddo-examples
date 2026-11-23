# LTX 2.3 image — measurement details

[Back to comparison tables](ltx23-image.md).
Scenario links in those tables open the sections below; no local report download
is required to read the results.

**Labels and units:** read the [labels and memory glossary](glossary.md) before
comparing runtime names, phase timings, tracked totals or memory peaks.

## Workload and measurement scope

Distilled BF16, 1280×704, 8 steps, seed 43, real prompt, CFG 1/rescale 0.7, no PAG. Component routes are listed per scenario.

Checkpoint: `elismasilva/ltx2.3-image-distilled-1.1`, snapshot
`5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9`. Official Diffusers 0.41.0
in the RC5 platform comparisons. Component routes are specified per scenario.
No LoRAs, compute compilation, PAG or tiling
are added to the primary cases.

The text-encoder row is the monitored `Pass 0: Encode prompts`: loading,
tokenizer/pipeline setup, encoding and cleanup, plus any purge inside that phase.
It is not the pure `encode_prompt_call` duration. Connectors run in the
`Pass 1: Generate` aggregate together with transformer loading, attachment,
denoise and cleanup. Its memory peak cannot be relabeled as a denoise-only peak.

The Windows Leaf and linear rows use an independently monitored `denoise`
interval around the same `denoise_modular_pipe_call`. It adds no CUDA
synchronization and excludes connector setup, latent readback and transformer
cleanup. Its sampled absolute GPU/RSS peaks can differ from the broader
generation aggregate. Other observations retain call latency and N/A for
unrecorded phase peaks. VAE rows use
separate load/decode monitors where recorded. A combined VAE phase must not
be compared to isolated decode compute as if their scopes were identical.

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

The Windows RC5 runtime overview uses standby purge at startup and before the
transformer for `vmm_fast` and NVMe. The other overview rows use no purge. Tracked totals
include any recorded purge time. Linux and WSL do not
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

The selected runtime row uses Leaf for the encoder and VAE with purge disabled.
Image decode is monitored separately from VAE loading, cleanup
and PNG saving. The selected Leaf and linear PNGs are byte-identical.
Denoise took 47.47 s, with sampled absolute GPU/RSS peaks of 1.60/28.76 GiB.
The full tracked run took 76.40 s and peaked at 7.12/28.76 GiB; those broader
peaks include connector setup and must not be substituted for denoise peaks.


<a id="windows-self-attention"></a>

### Windows / VMM Fast transformer self-attention

Generate the SDPA reference with `--platform windows --recipe ltx23-image
--family runtime --case vmm-fast`. Then select `--family attention
--attention-runtime vmm-fast` and pass the completed runtime manifest via
`--reference-manifest`. The `cudnn-sdpa` and `flash-attn2` cases reuse
the reference profile and change only `--image-self-attention`. Masked
cross-attention keeps default SDPA. All three cases use the Distilled BF16
checkpoint and the 1280×704, 8-step, seed-43 recipe. The reference's native
encoder, native VAE and two purge points apply to each case. Run from an
elevated PowerShell session to enable the purges.

The selected SDPA environment uses DDO and allocator 0.1.0rc5, Diffusers 0.41.0,
PyTorch 2.12.1+cu132, FlashAttention 2.9.1 and NVIDIA driver 617.42.
Both alternate measurements applied the same final 2.1921 GiB profile.
Separate diagnostics observed `aten::_scaled_dot_product_cudnn_attention`
and `flash_attn::_flash_attn_forward` for the respective variants, alongside
memory-efficient SDPA. The profiler was off during timed inference.
cuDNN took 37.11 s overall, 12.36 s to denoise and 0.66 s to decode;
FlashAttention 2 took 36.80 s overall, 12.35 s to denoise and 0.67 s to decode.
Both PNGs were valid 1280×704 images. They differed from SDPA, with mean
absolute RGB differences of 7.28 and 6.46 levels respectively on the 8-bit
scale. Visual quality remains unassessed. See the
[benchmark controller guide](reproduction.md).

<a id="windows-linear"></a>

### Windows / `one_shot_fast`

`--ddo-backend linear --ddo-preset one_shot_fast` executes `linear_runtime`.

The runtime row uses Leaf for the encoder and VAE and applies the 5.54 GiB
transformer residency profile.

The selected run took 48.55 s overall and 13.64 s to denoise. Denoise peaks
were 7.41 GiB absolute GPU usage and 22.13 GiB RSS; full-run RSS peaked at
27.20 GiB. Calibration time is outside the measured inference.


<a id="windows-low-ram"></a>

### Windows / `low_ram_safe`

`--ddo-backend linear --ddo-preset low_ram_safe` executes `linear_runtime`.

The runtime row uses Leaf for the encoder and VAE and applies the 4.04 GiB
transformer residency profile.

The selected run took 82.56 s overall and 39.38 s to denoise. Denoise peaks
were 5.83 GiB absolute GPU usage and 47.50 GiB RSS. The independently
monitored VAE decode took 0.96 s without standby purge.

<a id="windows-low-ram-pre-vae-purge"></a>

#### Windows / RC5 / `low_ram_safe` with Leaf VAE

The Windows image controller selects the two policies with
`--recipe ltx23-image --family runtime --case low-ram-safe` and
`--recipe ltx23-image --family runtime --case low-ram-safe-pre-vae-purge`.
The second case requires elevation and adds `--purge-before-vae` to inference
after transformer cleanup. Calibration runs with purges disabled in both cases.
Both calibrations produced an applied 4.0449 GiB resident budget with the same
profile name, on NVIDIA driver 617.42. The independently monitored decode
includes the latent transfer, VAE forward and conversion to an image, excluding
VAE loading, cleanup and PNG saving. The purge occurred before VAE loading,
took 2.51 s and was included in the full-run total. Decode took 0.96 s without
purge and 1.60 s with it; total latency was 82.56 s and 87.36 s respectively.
Denoise was nearly unchanged at 39.38 s and 39.37 s. Both PNGs were
byte-identical. The selected recipe leaves purge disabled.


<a id="windows-native"></a>

### Windows / `vmm_fast`

`--ddo-backend vmm --ddo-preset auto` resolves to `vmm_fast`, executing `vmm_fast_runtime`.

The runtime row uses the profiled 2.1921 GiB transformer residency budget and
1.5 GiB total staging. The encoder and VAE use native VMM via
`--text-encoder-native-vmm` and `--vae-vmm-offload`. The VAE uses the
checkpoint's `vae` subdirectory with its own 1.5 GiB residency budget.
Retained source and registration ceilings are each 24.625 GiB. The controller
confirmed the native source dispatcher and the startup and pre-transformer
purges. Both three-step calibration runs were outside the measured inference.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

The selected run took 37.02 s overall and 12.30 s to denoise (1.54 s/step).
Denoise peaks were 7.69 GiB absolute GPU usage and 27.38 GiB RSS. Independent
VAE decoding took 0.68 s, peaking at 5.83 GiB GPU and 4.52 GiB RSS.
Its PNG is byte-identical to the selected BF16 Leaf and linear outputs.

<a id="windows-nvme"></a>

### Windows / `nvme_baseline`

`--ddo-backend vmm --ddo-preset nvme_baseline` executes protected `vmm_runtime`.

The runtime row uses 1.5 GiB for transformer residency and 1.5 GiB total
staging. The encoder and VAE use Leaf. Retained source and registration
ceilings are both zero.
Normal PyTorch allocation plus native weight arena, not the process-wide native
Torch allocator. Zero persistent source cache is not zero OS file-cache usage.

The selected run took 69.54 s overall and 33.91 s to denoise. Independent
denoise peaks were 2.69 GiB absolute GPU usage and 6.20 GiB RSS. Decode took
1.02 s, peaking at 1.67 GiB GPU and 7.33 GiB RSS. Its PNG is byte-identical
to the selected BF16 Leaf and linear outputs.

<a id="linux-leaf"></a>

### Linux / Official leaf

The dedicated `run_diffusers_leaf.py` runner executes official Diffusers
Leaf hooks with streaming. Select `--platform linux --recipe ltx23-image
--family runtime --case official-leaf` in the benchmark controller.
The encoder and VAE use Leaf. The run uses default SDPA, BF16,
1280×704, eight steps, seed 43, CFG 1, rescale 0.7 and PAG disabled.
No calibration profile or standby purge is used.

The measured environment uses installed DDO and allocator 0.1.0rc5,
Torch 2.12.1+cu132, Diffusers 0.41.0 and NVIDIA driver 615.71.09 on the
RTX 3060 Ti 8 GiB. The controller verified the PNG and complete metrics.


<a id="linux-linear"></a>

### Linux / `one_shot_fast`

`--ddo-backend linear --ddo-preset one_shot_fast` executes `linear_runtime`.

Select `--platform linux --recipe ltx23-image --family runtime
--case one-shot-fast` with the same checkpoint and custom blocks as the
Leaf control. The saved profile selected a 5.6257 GiB resident-module
budget. Its eight-step calibration is outside the tracked inference total.
The encoder and VAE use Leaf, default SDPA is active and standby purge
is disabled. The output PNG matched the Leaf control by SHA-256.


<a id="linux-low-ram"></a>

### Linux / `low_ram_safe`

Select `--platform linux --recipe ltx23-image --family runtime
--case low-ram-safe` for the same RC5 workload and driver as the Leaf
control. The transformer uses `linear_runtime` and an applied 5.6843 GiB
resident-module profile. The eight-step calibration is outside the measured
inference total. The encoder and VAE use Leaf, with standby purge disabled.
The PNG matched the Leaf and One Shot Fast outputs by SHA-256.

Prompt encoding took 27.33 s and reached 45.42 GiB peak process RSS in this
run. Total time and process RSS exceeded both other measured Linux routes.

<a id="linux-native"></a>

### Linux / `vmm_fast`

`--ddo-backend vmm --ddo-preset auto` resolves to `vmm_fast`, executing `vmm_fast_runtime`.

Select `--platform linux --recipe ltx23-image --family runtime --case vmm-fast`
for the same RC5 workload and NVIDIA driver 615.71.09 as the Leaf control.
The final transformer profile selected 0.625 GiB resident weights and matched
the measured input shapes. Both three-step calibrations are outside the
tracked total. Total staging is 1.5 GiB, with source cache and registration
ceilings of 24.625 GiB each. The native source dispatcher was active.

The encoder and VAE both use native VMM. The VAE reads the checkpoint's
`vae` directory and has its own 1.5 GiB resident weight budget.
Execution-backing reclaim and standby purge are disabled. Default SDPA is
active. The output PNG matched all three Linux controls by SHA-256.
Native allocation is process-wide: encoder/VAE driver baselines can differ
even when their component offload policies are unchanged.

<a id="linux-native-attention"></a>

### Linux / VMM Fast self-attention

Select `--platform linux --recipe ltx23-image --family attention
--attention-runtime vmm-fast` with `--case cudnn-sdpa` or `--case flash-attn2`
and the completed default
SDPA runtime manifest supplied through `--reference-manifest`.
The controller reuses the verified final profile and native component
routes. Separate diagnostics recorded `aten::_scaled_dot_product_cudnn_attention`
and `flash_attn::_flash_attn_forward` with FlashAttention 2.8.3, alongside
SDPA for the other attention paths. The table uses separate measurements
with tracing disabled. Both PNGs were valid and their RGB pixels differed
from the SDPA control.

<a id="linux-nvme"></a>

### Linux / `nvme_baseline`

`--ddo-backend vmm --ddo-preset nvme_baseline` executes protected `vmm_runtime`.

Fixed transformer capacity: `4` GiB active weights; two `576` MiB staging slots. Retained source / registration ceilings: `0` / `0` GiB.
Normal PyTorch allocation plus native weight arena, not the process-wide native
Torch allocator. Zero persistent source cache is not zero OS file-cache usage.

<a id="wsl-leaf"></a>

### WSL / Official leaf (no stream)

The dedicated `run_diffusers_leaf.py` runner installs official Diffusers
leaf-level hooks, equivalent to the `diffusers_leaf_offload_compat` route.
The transformer and encoder use non-streamed Leaf, and the VAE uses Leaf.
The runtime control uses default SDPA and no DDO profile.

The selected run took 464.88 s overall, with absolute GPU/RSS peaks of
7.09/38.28 GiB. Denoise took 285.19 s, with peaks of 1.41/32.22 GiB;
isolated VAE decode took 4.43 s, with peaks of 1.55/6.51 GiB.


<a id="wsl-linear"></a>

### WSL / `wsl_compat`

`--ddo-backend linear --ddo-preset wsl_compat` executes `linear_runtime`.

The applied profile `ltx23-image-1280x704-8steps-wsl-compat` recommended
5.5762 GiB for resident modules. The benchmark controller clears the RC5
preset's resident-module override so the runner applies the profile. CPU
pinning is disabled, and the encoder and VAE use Leaf.

The selected run took 330.23 s overall, with absolute GPU/RSS peaks of
7.42/38.29 GiB. Denoise took 119.13 s, with peaks of 7.42/30.19 GiB;
isolated VAE decode took 4.40 s, with peaks of 1.55/6.54 GiB.

<a id="wsl-native"></a>

### WSL / `vmm_fast`

`--ddo-backend vmm --ddo-preset auto` resolves to `vmm_fast`, executing `vmm_fast_runtime`.

The applied profile `ltx23-image-1280x704-8steps-vmm-fast-wsl` selected
2,329,935,872 bytes (2.1699 GiB) for transformer weight residency. Total staging
is 384 MiB. Source cache and registration ceilings are 24.625 GiB and 0.5 GiB,
respectively. The native source dispatcher is active. The encoder and VAE
use Leaf. Native allocation is process-wide and can affect component driver
memory baselines.

The selected run took 286.57 s overall, with absolute GPU/RSS peaks of
7.73/37.57 GiB. Denoise took 83.87 s, with peaks of 7.73/30.47 GiB;
isolated VAE decode took 5.24 s, with peaks of 5.09/6.97 GiB.
The profile finalized and was applied with matching input shapes.

All three selected runtime cases completed with valid, byte-identical 1280×704
PNGs. They use default SDPA, driver 617.42 and no effective standby purge.
Each row's total, denoise, decode and detailed phase values come from that
same selected report. GPU peaks are sampled absolute usage and RAM peaks are
process RSS. Independent decode excludes VAE loading, cleanup and PNG saving.
Sequential WSL observations remain sensitive to Windows host activity and
available memory; this is not a repeated-run performance guarantee.

<a id="wsl-native-attention"></a>

### WSL / VMM Fast self-attention

The selected runtime default-SDPA measurement supplies the attention control.
Both variants applied its final 2.1699 GiB profile with the same checkpoint,
seed and component routes. Only transformer self-attention varies; masked
cross-attention remains default SDPA. Separate diagnostics confirmed
`aten::_scaled_dot_product_cudnn_attention` and
`flash_attn::_flash_attn_forward`; timed measurements have the profiler off.

cuDNN took 266.45 s overall and 76.45 s to denoise, with full-run absolute
GPU/RSS peaks of 7.74/37.78 GiB. Its denoise peaks were 7.74/30.51 GiB;
isolated decode took 4.56 s with peaks of 5.09/6.98 GiB.
FlashAttention 2 took 270.27 s overall and 79.86 s to denoise, with full-run
peaks of 7.76/37.99 GiB. Its denoise peaks were 7.76/30.47 GiB;
isolated decode took 5.34 s with peaks of 5.09/6.97 GiB.

Both cases completed with valid 1280×704 PNGs, differing from the SDPA PNG.
Visual quality remains unassessed. The sequential WSL observations do not
establish a repeatable speedup; Windows host activity and available memory
can affect timing independently of the attention mechanism.

<a id="windows-warm-process"></a>

## Windows / `warm_process`

This Windows RC5 observation uses the 8 GiB RTX 3060 Ti, driver 617.42,
Torch 2.12.1+cu132 and Diffusers 0.41.0. The BF16 checkpoint, custom blocks,
cat prompt, 1280×704 shape, eight steps, initial seed 43, guidance 1 and
rescale 0.7 match the runtime workload. PAG, model optimizations and purge
are disabled. The encoder uses Leaf and the final VAE is resident.

A dedicated calibration selected 5.5391 GiB of transformer residency.
Measurement performs two preparations first, then two generations with the
second transformer, decoding only the final latent. The first transformer's
owning offload result is released before cleanup and the next preparation.
The generator advances between generations. This total must not be compared
as one image or two independent complete pipelines.

The second preparation recorded plan and pinned-weight cache hits, including
18.52 GiB of cached pinned weights. Attachment took 10.77 s and 3.21 s;
denoise took 13.41 s and 13.30 s. The independent denoise peaks were
7.45 GiB and 7.41 GiB absolute GPU usage, with 22.13 GiB RSS in each interval.
Tracked total was 69.36 s, with 7.45 GiB peak absolute GPU usage and
27.98 GiB peak process RSS. The final isolated VAE decode took 0.31 s,
with peaks of 2.80 GiB absolute GPU usage and 22.17 GiB RSS; loading,
cleanup and PNG saving are outside that decode interval.
The controller completed with the profile applied and a verified 1280×704
PNG. Visual quality has not been assessed. See the
[recipe](reproduction.md#windows-ltx-image-process-reuse) and
[phase measurements](ltx23-image-phases.md#windows-warm-process).

## Windows SDNQ INT4

Measured on Windows with DDO/allocator RC5, RTX 3060 Ti 8 GiB, 64 GiB RAM,
NVIDIA driver 617.42, Torch 2.12.1+cu132, Diffusers 0.41.0, SDNQ 0.2.7 and
triton-windows 3.7.1.post27. The saved transformer is
`elismasilva/ltx2.3-image-distilled-1.1-sdnq-int4`, revision
`e00d7c057f1693aa0e4bb47583e7e7ca71ccd214`; original BF16 auxiliary weights
and reviewed image blocks match the BF16 workload. The benchmark cat prompt,
1280×704, eight steps, seed 43, guidance 1 and rescale 0.7 are fixed.
PAG, LoRA, compilation and standby purges are disabled. Default SDPA is
requested; no separate operator capture was performed for this INT4 pair.

The Leaf control uses official Diffusers leaf-level hooks with streaming,
selected through `diffusers_leaf_offload_compat` in the SDNQ runner.
DDO selects `linear/one_shot_fast` with the dedicated packed `sdnq_runtime`.
Neither replaces quantized backend forwards. Both use a Leaf BF16 encoder
and resident BF16 VAE. Native arena/source dispatcher settings are not applicable.

The dedicated SDNQ profile first measured a zero-resident baseline, proposed
6 GiB, then validated and reduced the recommendation to 4.8027 GiB. Both
calibrations used two steps without decode; the eight-step measurement
reported that final profile applied. Calibration time is outside inference.

Leaf denoise was 23.05 s, tracked total 52.13 s, peak absolute GPU
7.10 GiB and peak process RSS 25.99 GiB. DDO denoise was 10.79 s,
tracked total 26.31 s, peak absolute GPU 7.08 GiB and RSS 25.95 GiB.
Independent denoise peaks were 2.03 GiB GPU / 11.86 GiB RSS for Leaf and
7.07 GiB GPU / 6.59 GiB RSS for DDO. Isolated VAE decode took 0.23 s and
0.22 s respectively, with a 2.80 GiB GPU peak in both cases and RSS peaks
of 4.44 GiB and 6.63 GiB. These decode intervals exclude loading and saving.
The controller completed both cases and the verified 1280×704 PNGs are
byte-identical and RGB-identical. INT4 visual quality against BF16 has not been assessed.

Execution order affects interpretation: Leaf ran before the SDNQ calibration
processes and DDO measurement. Its unchanged Leaf prompt phase took
21.86 s versus 8.10 s later. The tracked-total difference includes these
warm-state effects and cannot be attributed entirely to transformer offloading.
No matched warm-state or repeated-run speed guarantee is established.
See the [recipe](reproduction.md#windows-ltx-image-sdnq-int4) and
[phase measurements](ltx23-image-phases.md#windows-sdnq-int4).

## Reproduce this workload

Use the [runner guide](../user/ltx2-image.md) and
[comparison controls](reproduction.md), retaining the model, schedule, inputs,
component routes and the scenario-specific capacities above. Runner defaults
or a different custom-block revision do not guarantee equivalent conditions.
Review trusted custom-block code and keep platform environments separate.
Package/build validation and raw engineering investigations are developer concerns,
not additional timing rows or prerequisites for interpreting these tables.

## Decode and save column


The latency columns use the same order as video: **Denoise**, **Average
(s/step)**, **Decode and save**, then **Tracked total**. Average divides denoise
wall latency by the recipe steps (8 for LTX Image, 20 for Qwen Image); it
retains any warmup/readback included in denoise. For images, Decode and save sums the recorded
VAE load/attachment, decode, collection/cleanup and PNG-save phases. It excludes
transformer cleanup and any pre-VAE purge. Some runners record a combined VAE
phase; its full recorded scope is retained. This is not isolated VAE compute.
N/A means that the required phases were not separately recorded, including
the monolithic Qwen Leaf generation; no estimate is subtracted from total.
See the phase measurements for the component timings. Denoise in Qwen includes
CPU latent readback. These phase intervals are already included in tracked total.
