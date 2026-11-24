# Qwen Image 2.1 — measurement details

[Back to comparison tables](qwen-image-21.md).
Scenario links in those tables open the sections below; no local report download
is required to read the results.

**Labels and units:** read the [labels and memory glossary](glossary.md) before
comparing runtime names, phase timings, tracked totals or memory peaks.

## Workload and measurement scope

Qwen Image 2.1 Base BF16, 1024×1024, 20 steps, seed 42, guidance 1 and
KV cache enabled. Checkpoint snapshot `d26bb61231c349cf6b7896fa83353113880e1ba3`.
The prompt is "A capybara wearing a wizard hat, reading by candlelight, oil painting".
See the [metric definitions](glossary.md) for sampled absolute GPU peaks,
process RSS and phase boundaries.

## Windows 1024×1024 — BF16 runtime control

### Windows 1024 Leaf control

Use the dedicated official Diffusers Leaf runner via
`benchmarks/run_benchmark.py --platform windows --recipe qwen-image21 --family runtime --case official-leaf`.
The recipe uses the capybara prompt, guidance 1, the default KV cache and
SDPA, streamed Leaf hooks for the encoder and transformer, non-streamed Leaf
for the untiled VAE, and no standby purge. The runner follows the same staged
custom-block phases as the DDO runners: the text encoder, transformer and VAE
are each loaded, attached, run and released separately, with official Diffusers
leaf-level offload on each component. See [benchmark reproduction](reproduction.md)
for execution setup.

On NVIDIA driver 617.42 with the RC5 wheels, the tracked total was 104.66 s,
with full-run peaks of 8.00 GiB absolute GPU usage and 18.97 GiB process RSS.
Text encoder loading, attachment, encoding and cleanup took 12.29 s (encoding
10.97 s), peaking at 2.28 GiB GPU and 18.97 GiB RSS. The sampling loop took
87.81 s (4.39 s/step), peaking at 2.13 GiB GPU and 16.99 GiB RSS. Decode took
2.16 s, peaking at 8.00 GiB GPU and 4.74 GiB RSS. All values come from this
one run, whose PNG matched the earlier Leaf observations byte for byte.

Each component is released before the next one loads, so the RSS peaks are
lower than the earlier run of the conventional pipeline, which kept every
component loaded until the end. Denoise covers the sampling loop through CPU
readback, including scheduler work and callbacks. Decode covers VAE execution
and image postprocessing; VAE loading, attachment, cleanup and PNG saving are
separate phases.

<a id="windows-1024-one-shot-fast"></a>

### Windows 1024 / `one_shot_fast`

Windows RC5, default SDPA, no standby purge. The transformer uses
`linear_runtime` with a calibrated 5.02 GiB resident budget. Encoder and
untiled VAE use Leaf. Select `--case one-shot-fast --no-purge` in the benchmark.

<a id="windows-1024-low-ram-safe"></a>

### Windows 1024 / `low_ram_safe`

Same workload and component routes, with a calibrated 4.75 GiB resident
budget. Select `--case low-ram-safe --no-purge`. Both linear PNGs match
the official Leaf PNG by SHA-256.

<a id="windows-1024-nvme-baseline"></a>

### Windows 1024 / `nvme_baseline`

The transformer uses `vmm_runtime` with a fixed 1.50 GiB resident weight
budget and a 1.50 GiB staging budget. Profile calibration is N/A. The
encoder and untiled VAE use Leaf. This Windows RC5 component control used
driver 617.42, default SDPA, and no standby purge. The PNG matched the
official Leaf control by SHA-256. Select
`--family components --case nvme-baseline --nvme-resident-gib 1.5 --no-purge`.

<a id="windows-1024-vmm-fast-leaf"></a>

### Windows 1024 / `vmm_fast` with Leaf VAE

The transformer uses `vmm/auto` (`vmm_fast_runtime`) with an applied 0.625 GiB
resident weight profile, 0.625 GiB staging, and 16 GiB each for source cache and
registration. The encoder uses native VMM, the source dispatcher was active,
and the untiled VAE uses Leaf. Default SDPA, standby purge disabled, and
execution-backing reclaim disabled. The PNG matched the official Leaf control
by SHA-256. Both three-step calibration runs are outside the tracked total.
This is the component control for evaluating the native VAE route.

Measured on the RTX 3060 Ti with NVIDIA driver 617.42, DDO and allocator
0.1.0rc5, Torch 2.12.1+cu132, Diffusers 0.41.0, and Transformers 5.15.1.
Select `--family components --case vmm-fast --no-purge` in the benchmark.

<a id="windows-1024-vmm-fast-native-vae"></a>

### Windows 1024 / `vmm_fast` with native VAE

This run calibrated a 0.625 GiB transformer profile and used the same native
VMM encoder route as the [Leaf VAE control](#windows-1024-vmm-fast-leaf).
The two three-step calibrations are outside the tracked total. The untiled VAE used native
VMM component ownership, an independent 0.375 GiB resident weight budget,
and a local BF16 copy converted from the checkpoint's FP32 VAE. The runner
uses `--vae-vmm-offload` with an explicit `--vae-model-path` and
`--vae-vmm-resident-weight-budget-gb 0.375`. Select
`--family runtime --case vmm-fast --vae-model-path elismasilva/qwen-image-2.1-bf16-vae`
in the benchmark controller. Single-frame VAE decode was disabled.

The sum of recorded VAE phases from load through cleanup was 3.02 s with
Leaf and 1.95 s with native VMM, including native attachment.

Both runs used default SDPA and disabled standby purge. The PNGs were
byte-identical. The load difference includes the prepared BF16 source.
Denoise ran before VAE loading.

<a id="windows-1024-joint-attention"></a>

### Windows 1024 / transformer joint attention on VMM Fast

The cuDNN case uses the [native VAE runtime control](#windows-1024-vmm-fast-native-vae)
and changes only transformer joint attention. Select
`--family attention --attention-runtime vmm-fast --case cudnn-sdpa`. Supply
the completed default SDPA manifest with `--reference-manifest` and the same
BF16 VAE with `--vae-model-path`. The controller reuses the verified final
transformer profile from the control. The operator check recorded
`aten::_scaled_dot_product_cudnn_attention` in a separate diagnostic run.
The table uses one measurement per case with tracing disabled. Both use
NVIDIA driver 617.42, no standby purge, and the same masked joint-attention
workload. Across the RGB channels, the PNGs differed by a mean absolute
1.20 levels on the 8-bit scale. Visual equivalence remains unverified.

FlashAttention 2 is excluded from this comparison because the available
backend cannot preserve the text padding mask used by Qwen joint attention.

## Linux 1024×1024 — BF16 runtime control

<a id="linux-1024-leaf-control"></a>

### Linux 1024 Leaf control

Select `--platform linux --recipe qwen-image21 --family runtime
--case official-leaf --expandable-segments` in the benchmark controller.
The dedicated official Leaf runner uses the same checkpoint, capybara
prompt, BF16, 1024×1024, 20 steps, seed 42, guidance 1 and default KV cache.
Transformer joint attention uses default SDPA. Encoder and transformer
use streamed Leaf hooks, and the untiled VAE uses non-streamed Leaf. The
runner follows the same staged custom-block phases as the DDO runners, with
official Diffusers leaf-level offload on each component.

The manifest records `PYTORCH_ALLOC_CONF=expandable_segments:True` explicitly.
The run completed all 20 steps and produced a verified 1024×1024 PNG,
byte-identical to the other Linux runtime outputs. Full-run latency was
70.23 s, with 7.36 GiB absolute GPU peak and 19.08 GiB process RSS peak.
Text encoder loading, attachment, encoding and cleanup took 6.10 s (encoding
5.07 s), peaking at 1.71 GiB GPU and 19.08 GiB RSS. Denoise took 60.44 s
(3.02 s/step), peaking at 1.44 GiB GPU and 17.14 GiB RSS. Decode took 1.54 s,
peaking at 7.36 GiB GPU and 5.07 GiB RSS.

Denoise covers the sampling loop through CPU readback, including scheduler
work and callbacks. Decode covers VAE execution and image postprocessing; VAE
loading, attachment, cleanup and PNG saving are separate phases. Each component
is released before the next one loads.

The environment uses installed DDO and allocator 0.1.0rc5, Torch
2.12.1+cu132, Diffusers 0.41.0 and NVIDIA driver 615.71.09. Standby purge,
single-frame decode and VAE tiling are disabled. No DDO profile is required.

<a id="linux-1024-one-shot-fast"></a>

### Linux 1024 / `one_shot_fast`

Select `--platform linux --recipe qwen-image21 --family runtime
--case one-shot-fast --expandable-segments` with the same checkpoint,
prompt and workload as the Linux Leaf control. The transformer uses
`linear_runtime` with an applied 5.2197 GiB resident-module profile.
The 20-step calibration is outside the tracked inference total.
Encoder and untiled VAE use Leaf. Default SDPA, standard KV cache and
PyTorch expandable segments are active, with standby purge disabled.
The output PNG matched the Leaf control by SHA-256.

The tracked total was 3.34 s shorter than the selected Leaf control, with
peak process RSS 12.15 GiB lower. These runs occurred in separate sessions;
the difference is not an isolated runtime effect.

<a id="linux-1024-low-ram-safe"></a>

### Linux 1024 / `low_ram_safe`

Select `--platform linux --recipe qwen-image21 --family runtime
--case low-ram-safe --expandable-segments`. The recipe, checkpoint,
default SDPA, Leaf encoder/VAE and allocator configuration match the
Linux One Shot Fast case. The transformer uses `linear_runtime` with an
applied 5.2706 GiB resident-module profile. Its 20-step calibration is
outside the tracked inference total. The PNG matched the Leaf and One Shot
Fast outputs by SHA-256.

The run took 71.60 s with 22.41 GiB peak RSS, exceeding One Shot Fast
on both metrics. Its VAE phase peaked at 2.95 GiB process RSS, while
the run-wide RSS peak occurred during text encoding.

<a id="linux-1024-vmm-fast"></a>

### Linux 1024 / `vmm_fast`

Select `--platform linux --recipe qwen-image21 --family runtime
--case vmm-fast` with the same BF16 VAE source used by the Windows native
case, passed through `--vae-model-path`. The transformer uses
`vmm/auto` (`vmm_fast_runtime`) and an applied final 0.625 GiB resident
weight profile with matching input shapes. Both three-step calibrations
are outside the tracked total. Staging is 0.625 GiB, and source cache and
registration ceilings are 16 GiB each.

Encoder and untiled VAE use native VMM. The VAE has its own 0.375 GiB
resident budget. The native source dispatcher was active. Execution-backing
reclaim, standby purge and single-frame VAE decode are disabled. The
process uses the native DDO allocator. The PNG matched all three Linux
Leaf and linear outputs by SHA-256.

<a id="linux-1024-vmm-fast-leaf"></a>

### Linux 1024 / `vmm_fast` / Leaf VAE

Select `--platform linux --recipe qwen-image21 --family components
--case vmm-fast` with `--component-reference-manifest` pointing to the completed
native VMM Fast runtime manifest. The controller verifies and reuses the final
0.625 GiB transformer profile. Encoder, transformer, default SDPA, source
budgets, checkpoint, prompt, seed and image dimensions match that reference.
Only the VAE route changes to `--vae-leaf-offload`; it loads from the main
checkpoint instead of the standalone BF16 native VAE source.

The untiled Leaf VAE uses the native DDO allocator installed for this VMM Fast
process. No single-frame optimization, reclaim or standby purge is enabled.
NVIDIA driver 615.71.09 and the RC5 environment match the native reference.
The run completed with an applied final profile and matching input shapes.
The verified PNG matched the native reference by SHA-256 and RGB pixels.
The 57.58 s tracked total includes a 47.19 s denoise/readback and a 1.74 s
VAE decode. This is one measurement per route; the total difference includes
variation in transformer time as well as the VAE route change.

<a id="linux-1024-joint-attention"></a>

### Linux 1024 / transformer joint attention on VMM Fast

Select `--platform linux --recipe qwen-image21 --family attention
--case cudnn-sdpa --attention-runtime vmm-fast`. Supply the completed
VMM Fast SDPA manifest with `--reference-manifest` and the same BF16 VAE
with `--vae-model-path`. The controller verifies the reference and reuses
its final 0.625 GiB profile, native component routes and transport budgets.
Only masked transformer joint attention changes.

The separate diagnostic recorded `aten::_scaled_dot_product_cudnn_attention`.
The timed measurement had tracing disabled. Both used NVIDIA driver
615.71.09 and no standby purge. The PNG was valid and differed from the
control by mean absolute values of 1.74, 1.28 and 0.69 levels across
the RGB channels on the 8-bit scale. Visual quality remains unassessed.
FlashAttention 2 is excluded because the available backend cannot preserve
the text padding mask used by this joint-attention workload.

## WSL 1024×1024 — BF16 runtime control

<a id="wsl-1024-leaf-control"></a>

### WSL 1024 Leaf control

The benchmark controller starts `run_diffusers_leaf.py` with
`--no-stream-offload --vae-leaf-offload --image-joint-attention default`.
The run uses BF16, 20 steps, seed 42, guidance 1, the standard KV cache,
and no VAE tiling or standby purge. The environment uses Diffusers `0.41.0`,
Torch `2.12.1+cu132`, installed DDO and allocator RC5, and NVIDIA driver `617.42`.
The runner follows the same staged custom-block phases as the DDO runners,
with official Diffusers leaf-level offload on each component, and uses no
calibration profile. The run completed all 20 steps and produced a verified
1024×1024 PNG, byte-identical to the other WSL runtime outputs.
It uses `PYTORCH_ALLOC_CONF=expandable_segments:False`; prefix a direct Bash
runner command with that assignment. The benchmark controller applies this
setting only to the WSL official Leaf runtime recipe. Shared defaults remain
unchanged.

The full run took 407.83 s, peaking at 8.00 GiB GPU and 22.39 GiB RSS.
Text encoder loading, attachment, encoding and cleanup took 84.88 s (encoding
82.88 s), peaking at 3.42 GiB GPU and 22.39 GiB RSS. Denoise took 312.09 s
(15.60 s/step), with 2.26 GiB GPU and 20.62 GiB RSS. Decode took 5.33 s, with
8.00 GiB GPU and 3.09 GiB RSS. Each component is released before the next one
loads; the transformer cleanup precedes VAE loading.

<a id="wsl-1024-wsl-compat"></a>

### WSL 1024 / `wsl_compat`

Select `--platform wsl --recipe qwen-image21 --family runtime --case wsl-compat`
with the same reviewed custom blocks as VMM Fast. The transformer uses
`linear/wsl_compat` (`linear_runtime`) with CPU pinning disabled and an applied
4.74 GiB resident-module budget. The controller clears the preset's fixed
resident-module override so the saved profile determines the budget. The
20-step calibration is outside the tracked total.

The encoder and untiled VAE use non-streamed Leaf. Default SDPA is active,
with no standby purge or single-frame VAE decode. The run used the same
GPU, driver, Python, and package versions as VMM Fast. Its output PNG
matched both other WSL runs by SHA-256.

<a id="wsl-1024-vmm-fast"></a>

### WSL 1024 / `vmm_fast`

Select `--platform wsl --recipe qwen-image21 --family runtime --case vmm-fast`.
Use a clean checkout of the custom blocks at
`cd1efe4ad857b3e15fe5cb675842b4283750b805` and pass it with `--custom-blocks`.
The transformer uses `vmm/auto` (`vmm_fast_runtime`) with the source dispatcher
active and an applied 4.34 GiB resident weight budget. Staging, source cache
and registration budgets were 0.625 GiB, 16 GiB and 0.5 GiB, respectively.
Both three-step profile calibrations took place outside the tracked total.

The encoder and untiled VAE use non-streamed Leaf. The run uses default
SDPA with execution-backing reclaim, standby purge and single-frame VAE
decode disabled. It used the same GPU, driver, DDO, allocator, Torch and
Diffusers versions as the WSL Leaf control, with Transformers 5.15.1 and
Python 3.12.12. The verified output PNG matched the Leaf control by SHA-256.

<a id="wsl-1024-joint-attention"></a>

### WSL 1024 / transformer joint attention on VMM Fast

Select `--platform wsl --recipe qwen-image21 --family attention --case cudnn-sdpa
--attention-runtime vmm-fast` and supply the completed VMM Fast default SDPA
manifest with `--reference-manifest`. The controller reuses the verified final
profile from the control. Only transformer joint attention changes.

The diagnostic recorded `aten::_scaled_dot_product_cudnn_attention`.
The timed run used the same profile and kept tracing disabled. Its PNG matched
the diagnostic PNG by SHA-256 and differed from the default SDPA control.
Mean absolute differences in 8-bit RGB values were 1.74, 1.28, and 0.69
per channel. Maximum differences were 182, 171, and 118, respectively.

## Reproduce this workload

Use the [runner guide](../user/qwen-image-21.md) and
[comparison controls](reproduction.md), retaining the model, schedule, inputs,
component routes and the scenario-specific capacities above. Runner defaults
or a different custom-block revision do not guarantee equivalent conditions.
Review trusted custom-block code and keep platform environments separate.
Package/build validation and raw engineering investigations are developer concerns,
not additional timing rows or prerequisites for interpreting these tables.

<a id="windows-sdnq-int4"></a>

## Windows SDNQ INT4

The RC5 pair uses saved `OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit` snapshot
`59d4c3610379d91fb023c4d0b6527c76755491c4` and BF16 Base auxiliaries from
`Qwen/Qwen-Image-2.1`, snapshot `d26bb61231c349cf6b7896fa83353113880e1ba3`.
Reviewed Qwen custom blocks are `cd1efe4ad857b3e15fe5cb675842b4283750b805`.
The saved recipe specifies unsigned four-bit weights, dynamic quantization
and Hadamard transforms; the runner preserves SDNQ's backend forward.

Workload: 1024×1024, 20 steps, seed 42, guidance 1, default KV cache,
default SDPA, and prompt “A capybara wearing a wizard hat, reading by
candlelight, oil painting”. Both use explicit no-stream Leaf encoder and
VAE routes. Transformer Leaf uses official streamed hooks; DDO uses
`linear/one_shot_fast` resolved to `sdnq_runtime`. No purge, compilation,
single-frame VAE or other model optimization is enabled.

Environment: Windows, RTX 3060 Ti 8 GiB, NVIDIA driver 617.42, installed
DDO/allocator 0.1.0rc5, Torch 2.12.1+cu132, Diffusers 0.41.0 and SDNQ 0.2.7.
The DDO-specific profile was built with a two-step baseline and a two-step
probe without decode, then applied at 6 GiB. Calibration is excluded from
inference totals. Source dispatcher is N/A for these normal-allocator routes.

Both measurements completed and their PNG RGB pixels were identical.
Results are single sequential observations, with Leaf before calibration
and DDO afterward; prompt and startup/cache effects affect total latency.
No comparison with BF16 output quality is established. Decode and save
sums VAE load, decode, collection/cleanup and PNG saving.
See the [phase table](qwen-image-21-phases.md#windows-sdnq-int4).

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

## Linux VAE feature study

The [nine Qwen VAE configurations](model-features.md#linux-qwen-vae-observations)
use Base snapshot `d26bb61231c349cf6b7896fa83353113880e1ba3`, custom blocks
`cd1efe4ad857b3e15fe5cb675842b4283750b805`, BF16, 1024×1024, 20 steps,
seed 42, guidance 1, KV cache and the same capybara prompt. Driver 615.71.09,
installed RC5, Torch 2.12.1+cu132 and Diffusers 0.41.0 match the Linux host.
Default SDPA, native VMM encoder and VAE, and the source dispatcher are fixed.
The VAE reads the same prepared BF16 source as the native runtime control;
the feature run does not convert it. Encoder residency is 4 GiB, VAE 0.375 GiB,
total staging 0.625 GiB and source cache/registration ceilings 16 GiB each.
No purge, reclaim, compilation or quantization is enabled.

Two three-step calibrations finalized the control transformer profile. All
nine measurements applied that intact profile with matching shapes and exactly
671,088,640 configured resident-weight bytes. Decoder features are applied
after denoising. The original tiling case disables single-frame, alias, direct
padding and band decode. Each alternative compares with its explicit parent
and the common control, not merely the previous execution in the queue.

All runs completed and their 1024-square PNGs and registered hashes verified.
The control, single-frame, alias and both band sizes without direct padding
are byte-identical. The three direct-padding variants match each other but
differ from the control; tiling has a third output. See the feature page for
pixel differences. The reviewer assessed both changed outputs (direct padding
and tiling) as visually good; this does not imply pixel equality. No positive final native
failure or quarantine value was observed in the recorded transformer counters.
Separate runtime/attention rows are not replaced by this feature control.
These are sequential single observations, not repeatability guarantees.

See [feature phase measurements](qwen-image-21-phases.md#linux-qwen-vae-features).

## Windows VAE feature study

The [nine-case Windows study](model-features.md#windows-qwen-vae-observations)
uses the same Base BF16 checkpoint, capybara prompt, 1024-square resolution,
20 steps, seed 42, guidance 1, standard KV cache and default SDPA as the runtime
recipe. Native encoder residency is 4 GiB, native BF16 VAE residency 0.375 GiB,
total staging 0.625 GiB and source cache/registration 16 GiB each. Dispatcher
was active; purge, reclaim, compilation and quantization were off.

The control alone built the final transformer profile in two three-step runs
without decode. All nine measurements applied that profile with matching shapes
and exactly 671,088,640 configured resident-weight bytes. Calibration is outside
inference latency. Driver 617.42, installed DDO/allocator RC5, Torch 2.12.1+cu132
and Diffusers 0.41.0 were fixed. Artifacts, complete phase memory, native counters,
PNG dimensions and comparisons with control and branch parents were verified.
No positive final failure or quarantine counters were observed in component
reports. Pixel equality and Windows visual acceptance are separate; see the
feature page for the per-branch results and pending visual review.
