# Benchmark overview

Compare measured runtime, latency and memory for LTX 2.3 image, LTX 2.3
video/audio and Qwen Image 2.1. Results are grouped by platform and workload;
no universal ranking is implied. Start with the Windows summaries below, then
open a model's results for native Linux, WSL, attention and precision comparisons.

| Model | Main comparison | Phase measurements | Measurement details |
| --- | --- | --- | --- |
| LTX 2.3 image | [Results and recommendations](ltx23-image.md) | [Phase measurements](ltx23-image-phases.md) | [Scenario details](ltx23-image-details.md) |
| LTX 2.3 video | [Results and recommendations](ltx23-video.md) | [Phase measurements](ltx23-video-phases.md) | [Scenario details](ltx23-video-details.md) |
| Qwen Image 2.1 | [Results and recommendations](qwen-image-21.md) | [Phase measurements](qwen-image-21-phases.md) | [Scenario details](qwen-image-21-details.md) |

**Before reading the tables:** see the [labels and memory glossary](glossary.md)
for runtime labels, denoise/decode scope, tracked totals and GPU/RAM peaks.

For Windows and native Linux measurements that add one model optimization per
case, see [cumulative model features](model-features.md). This family has separate
controls and does not replace the runtime comparisons below. Qwen Linux VAE
features reduced decode peak from 7.66 GiB to 2.77 GiB with bands 256 and no
direct padding, preserving the exact PNG. The Windows nine-case series is also
complete: the same branch reduced decode peak from 8.00 GiB to 3.20 GiB.
These are decoder-phase peaks, not full-run peaks. Output checks and
recommendations remain with each platform's feature results.

## Windows BF16 comparison

Selected RC5 observations on an **RTX 3060 Ti 8 GiB / 64 GiB RAM**, using
NVIDIA driver **617.42** and default PyTorch SDPA. Rows are selected individual
runs. The Qwen Leaf control followed a manual Windows standby cache purge;
its runner purge controls were disabled. Preset names identify the
transformer preset: `one_shot_fast` and `low_ram_safe` use `linear_runtime`,
`vmm_fast` uses `vmm_fast_runtime`, and `nvme_baseline` uses `vmm_runtime`.
The dedicated official Leaf control uses Diffusers hooks; its equivalent DDO
compatibility route is `diffusers_leaf_offload_compat`. See scenario details
for the actual component routes and profile or budget.

🟢 ↓ marks a lower value than the first row of the same table; 🔴 ↑ marks a
higher value. Equal displayed values have no arrow. Lower memory and shorter
latency are separate priorities, not a single ranking.

Times are seconds. GPU peaks are sampled **absolute driver usage**, RAM peaks
are process **RSS**, both in **GiB**, over the tracked run. Tracked total excludes
calibration, command startup and downloads. Denoise and other phase intervals can be nested;
do not sum them to reconstruct total. See [metric definitions](glossary.md).

**VAE decode** reports the recorded decoder interval, excluding VAE loading,
cleanup and file saving. For video it also excludes audio decoding. Conversion
or postprocessing inside the recorded call remains included. Modular Qwen
denoise includes CPU latent readback. See each model's measurements for exact
phase boundaries.

**Full run** includes all tracked phases, including loading, encoding,
denoising, decoding, cleanup, conversion and saving. The memory columns below
are full-run peaks, not denoising or decoder peaks. Open the model's tables for
separate phase peaks. Do not add phase peaks or nested timings.

### LTX video — 1280×704, 8 seconds, 193 frames, 8 steps

This series uses guitar image and audio conditioning, seed 43 and CUDA
exact-streaming decode. Official Leaf and VMM Fast use a resident video VAE;
the linear presets use Leaf VAE. DDO cases purge standby memory only before
the transformer; the Leaf control uses no purge. Chunking, fused/reused RoPE,
phase modulation, compact timesteps, in-place residuals and compilation are
disabled. See the [8-second measurements and route details](ltx23-video.md#windows-bf16-8-second-pressure-workload).

| Preset | Denoise (s) | Average (s/step) | Video VAE decode (s) | Full run latency (s) | Full run peak GPU (absolute, GiB) | Full run peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 363.33 | 45.42 | 19.12 | 421.12 | 8.00 | 41.34 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 335.02&nbsp;🟢&nbsp;↓ | 41.88&nbsp;🟢&nbsp;↓ | 19.23&nbsp;🔴&nbsp;↑ | 403.49&nbsp;🟢&nbsp;↓ | 8.00 | 50.35&nbsp;🔴&nbsp;↑ |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 419.30&nbsp;🔴&nbsp;↑ | 52.41&nbsp;🔴&nbsp;↑ | 18.87&nbsp;🟢&nbsp;↓ | 482.05&nbsp;🔴&nbsp;↑ | 8.00 | 42.36&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 321.35&nbsp;🟢&nbsp;↓ | 40.17&nbsp;🟢&nbsp;↓ | 18.79&nbsp;🟢&nbsp;↓ | 384.23&nbsp;🟢&nbsp;↓ | 8.00 | 33.08&nbsp;🟢&nbsp;↓ |

On VMM Fast, Sol-Attn took 211.65 s to denoise versus 321.35 s for its matched
SDPA control, with user-accepted visual quality. Earlier cuDNN and FlashAttention 2
measurements took 273.46 s and 274.68 s, with different code/environment identities. See the
[Windows video attention results](ltx23-video.md#windows-video-attention).

### LTX image — 1280×704, 8 steps

Distilled BF16, seed 43, CFG 1/rescale 0.7 and PAG disabled. VMM Fast uses
native VMM for the encoder and VAE; the other rows use Leaf for both.
VMM Fast and NVMe use standby purges at startup and before the transformer;
the other rows use no purge.

The separate [process reuse measurement](ltx23-image.md#windows-warm-process)
took 69.36 s for two preparations, two generations and one final decode/save.
It is not a single-image latency; the second attachment took 3.21 s versus
10.77 s for the first.

The separate [saved SDNQ INT4 pair](ltx23-image.md#windows-sdnq-int4)
completed in 52.13 s with Leaf and 26.31 s with DDO, producing identical
PNGs within the pair. These sequential observations include different prompt
warm states; they are not a matched warm-state speed guarantee.

| Preset | Denoise (s) | Average (s/step) | VAE decode (s) | Full run latency (s) | Full run peak GPU (absolute, GiB) | Full run peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Official Leaf](ltx23-image-details.md#windows-leaf) | 47.47 | 5.93 | 1.09 | 76.40 | 7.12 | 28.76 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 13.64&nbsp;🟢&nbsp;↓ | 1.70&nbsp;🟢&nbsp;↓ | 1.17&nbsp;🔴&nbsp;↑ | 48.55&nbsp;🟢&nbsp;↓ | 7.41&nbsp;🔴&nbsp;↑ | 27.20&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 39.38&nbsp;🟢&nbsp;↓ | 4.92&nbsp;🟢&nbsp;↓ | 0.96&nbsp;🟢&nbsp;↓ | 82.56&nbsp;🔴&nbsp;↑ | 7.08&nbsp;🟢&nbsp;↓ | 47.50&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 12.30&nbsp;🟢&nbsp;↓ | 1.54&nbsp;🟢&nbsp;↓ | 0.68&nbsp;🟢&nbsp;↓ | 37.02&nbsp;🟢&nbsp;↓ | 7.69&nbsp;🔴&nbsp;↑ | 27.38&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 33.91&nbsp;🟢&nbsp;↓ | 4.24&nbsp;🟢&nbsp;↓ | 1.02&nbsp;🟢&nbsp;↓ | 69.54&nbsp;🟢&nbsp;↓ | 7.10&nbsp;🟢&nbsp;↓ | 25.90&nbsp;🟢&nbsp;↓ |

VMM Fast had the shortest tracked total, and its denoise was 1.33 s shorter than
One Shot Fast. In the matched VMM Fast attention comparison, denoise times
differed by at most 0.06 s. FlashAttention 2 had the shortest tracked total
at 36.80 s, against 37.02 s for SDPA and 37.11 s for cuDNN. These single
observations do not establish a repeatable speedup; alternate outputs differed.
See [image results](ltx23-image.md#windows-bf16) and
[phase measurements](ltx23-image-phases.md#windows-bf16).
The VAE phases have different boundaries across routes; compare their
recorded load, decode and cleanup scope rather than treating them as identical
isolated decoder measurements.
### Qwen Image 2.1 — 1024×1024, 20 steps

| Preset | Denoise (s) | Average (s/step) | VAE decode (s) | Full run latency (s) | Full run peak GPU (absolute, GiB) | Full run peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 87.62 | 4.38 | 5.59 | 107.66 | 8.00 | 35.50 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 53.82&nbsp;🟢&nbsp;↓ | 2.69&nbsp;🟢&nbsp;↓ | 2.61&nbsp;🟢&nbsp;↓ | 66.08&nbsp;🟢&nbsp;↓ | 8.00 | 21.85&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 65.53&nbsp;🟢&nbsp;↓ | 3.28&nbsp;🟢&nbsp;↓ | 5.46&nbsp;🟢&nbsp;↓ | 85.58&nbsp;🟢&nbsp;↓ | 8.00 | 27.03&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 64.86&nbsp;🟢&nbsp;↓ | 3.24&nbsp;🟢&nbsp;↓ | 5.31&nbsp;🟢&nbsp;↓ | 76.77&nbsp;🟢&nbsp;↓ | 8.00 | 18.97&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 49.66&nbsp;🟢&nbsp;↓ | 2.48&nbsp;🟢&nbsp;↓ | 1.52&nbsp;🟢&nbsp;↓ | 60.05&nbsp;🟢&nbsp;↓ | 8.00 | 18.17&nbsp;🟢&nbsp;↓ |

Base BF16, seed 42 and guidance 1, without model-specific optimizations,
including single-frame VAE decode. The Leaf row is one standalone run after a
manual standby cache purge, with no purge requested by the runner; the other
rows use no standby purge. VMM Fast uses a native
encoder and BF16 native VAE; the other rows use Leaf for both. See the
[Qwen runtime results](qwen-image-21.md#windows-10241024-bf16-runtime-control)
and [phase measurements](qwen-image-21-phases.md#windows-10241024-bf16-runtime-control).

In the separate VMM Fast attention comparison, cuDNN SDPA took 57.56 s
versus 60.05 s with default SDPA, a 4.1% reduction in one run per case.
The PNG pixels differed. See the [Qwen attention results](qwen-image-21.md#transformer-joint-attention-on-vmm-fast).

These configured pipeline results do not isolate the effect of VMM, establish
an advantage over a matched `torch.compile` setup, or certify other models.
Native Linux, WSL, lower-RAM, VAE-route and quantization comparisons remain in
their separate groups. [Hardware and environment limits](hardware.md).

The [Linux Qwen 1024-square Leaf control](qwen-image-21.md#linux-10241024-bf16-runtime-control)
took 70.85 s with PyTorch expandable segments enabled, reaching 7.46 GiB
absolute GPU usage and 33.66 GiB process RSS.
With the same allocator configuration, Linux Qwen `one_shot_fast` took
67.51 s and reached 21.51 GiB peak RSS. Low RAM Safe took 71.60 s and
reached 22.41 GiB peak RSS. All three PNGs were byte-identical.
Linux Qwen VMM Fast with native encoder and VAE took 53.98 s, with
18.27 GiB peak RSS and the same PNG as those three controls.
Its cuDNN SDPA variant took 55.06 s, compared with 53.98 s for default
SDPA, and produced different RGB pixels.

## Choose a route or optimization

Use the model's recommendations rather than choosing a preset by name alone.
Transformer speed, encoder memory and decoder capacity can favor different
component strategies. A shorter denoise does not necessarily mean a shorter
pipeline. Do not combine the fastest phases from different runs into an
unmeasured pipeline result.

Runtime tables keep default SDPA and disable model-specific optimizations.
Attention comparisons vary the transformer attention on VMM Fast; model
features and saved precision formats remain separate comparisons. Operator
tracing runs separately from timed measurements. Each model's measurement
details describe its workload, component routes and comparison limits.

### What the measurements support

| Model / priority | What the selected measurements support | Tables |
| --- | --- | --- |
| LTX image / one-shot speed | VMM Fast had the shortest tracked total on Windows and Linux. On Linux RC5, `vmm_fast` took 31.44 s, `one_shot_fast` 38.77 s, official Leaf 47.55 s and `low_ram_safe` 56.34 s. The separate cuDNN and FlashAttention 2 comparisons took 30.15 s and 29.99 s overall. Their displayed denoise intervals were 0.73 s and 0.70 s longer than default SDPA; the lower totals came from other phases. | [LTX image](ltx23-image.md) |
| LTX video / one-shot speed | VMM Fast had the shortest tracked total in the Windows, native Linux and WSL 8-second runtime sets. On Linux, VMM Fast took 325.78 s, Leaf 342.97 s, One Shot Fast 352.56 s and Low RAM Safe 344.17 s. VMM Fast also had the lowest peak RSS at 33.07 GiB. The three refreshed Leaf/linear MP4s were byte-identical. | [LTX video](ltx23-video.md) |
| Qwen / runtime choice | On the Windows 1024-square recipe, `vmm_fast` with a native encoder and native VAE had the shortest tracked total and lowest peak process RSS. | [Qwen](qwen-image-21.md) |
| Lower process RAM for Windows video | `vmm_fast` has the lowest peak process RSS in the 8-second runtime comparison at 33.08 GiB. | [Video Windows](ltx23-video.md#windows-bf16-8-second-pressure-workload) |
| Decoder capacity | Prefer the measured component route for the workload. A resident VAE can avoid weight transfers when it fits; Leaf offload and supported tiling are alternatives under memory pressure. Weight offloading does not eliminate activation memory. Video defaults to exact-streaming decode; tiling is an explicit fallback. | [Image phases](ltx23-image-phases.md), [Video decoder controls](../user/ltx2-video-parameters.md), [Qwen routes](qwen-image-21.md) |
| FP8 weight-only | In the recorded precision pairs, Windows took 424.02 s versus its original BF16 reference of 390.85 s (+8.5%); Linux took 357.04 s versus 325.78 s (+9.6%). Peak RSS fell by 12.7% and 17.5%, respectively; absolute GPU peaks were effectively unchanged. The Windows percentage does not use the newer 384.23 s runtime control. Windows visual quality was user-accepted; Linux visual review is pending. | [Video FP8 measurements](ltx23-video.md#saved-fp8) |
| WSL | LTX Image VMM Fast and `wsl_compat` took 286.57 s and 330.23 s versus 464.88 s for official Leaf, with identical PNGs and more peak GPU memory. LTX Image cuDNN and FlashAttention 2 denoised in 76.45 s and 79.86 s versus 83.87 s for SDPA; sequential WSL variation limits this comparison. The current LTX Video VMM Fast SDPA control took 700.53 s versus 883.64 s for Leaf and 1960.70 s for `wsl_compat`, across separate configurations/sessions. In the matched VMM Fast attention pair, Sol took 600.55 s overall and 359.22 s to denoise versus 700.53 s and 461.95 s for SDPA, with user-accepted visual quality. Qwen VMM Fast and `wsl_compat` took 220.29 s and 226.38 s versus 420.09 s for Leaf, with the same absolute GPU peak. Qwen cuDNN SDPA took 8.34 s longer than its SDPA control. | [LTX Image WSL](ltx23-image.md#wsl-bf16), [video WSL](ltx23-video.md#wsl-bf16), [Qwen WSL](qwen-image-21.md#wsl-10241024-bf16-runtime-control) |

These are single workload observations, not causal allocator-only or low-RAM
hardware guarantees. Text encoding can dominate global RAM even when denoise RAM
falls. Compare phase peaks and total separately; scenario links inside each table
provide the relevant limitations.

For Linux video attention on VMM Fast, the separate Sol-Attn measurement took
214.54 s to denoise and 261.78 s overall, versus 274.26 s and 325.78 s for
default SDPA. Media validation passed; visual quality remains unconfirmed.
See the [Linux attention comparison](ltx23-video.md#linux-video-attention).
The WSL Sol pair is also complete, with user-accepted visual quality; its
new SDPA control is distinct from the earlier dense-attention comparison.

See [metric definitions](glossary.md), [presets](../user/presets.md) and
[reproduction](reproduction.md).

`low_ram_safe` did not have the lowest global RSS in the Windows tables.
Its name does not certify operation on low-RAM hardware. The separate LTX Image
[pre-VAE purge pair](ltx23-image.md#low_ram_safe-pre-vae-purge) did not improve
decode or total latency, and peak RSS was nearly unchanged. Keep purge disabled
for this recipe.
Purge benefits are workload-dependent and should not be generalized to every
runtime. Windows standby purge is not used on native Linux or WSL.

`nvme_baseline` remains an experimental alternative with manual recipes.
It did not beat VMM Fast in these Windows image measurements and is not part
of the primary video runtime series. Further hardware-specific evaluation
is needed before recommending it over the primary runtimes. This does not
remove the protected route.

Reclaim is not enabled in these reference recipes; the selected segmented-backing
measurements do not establish a repeatable benefit. Heavier workloads require
separate evaluation.

## Reproduce and interpret the results

Use the [benchmark controller](reproduction.md) to preview commands, calibrate
missing compatible profiles and run the selected model/family. Completed,
verified cases can be resumed from a session manifest; calibration time is
separate from inference time. Platform environments and budgets are distinct.
See [hardware and platform limits](hardware.md) before copying a recipe.

The eight-second video workload deliberately creates pressure on this 8 GiB
GPU. Weight residency and transfer behavior matter more under pressure; the
same duration on a larger GPU need not show the same differences. These
comparisons measure configured pipelines, not an isolated allocator effect,
and do not establish an advantage over a matched `torch.compile` setup.

- [Runtimes and presets](../user/presets.md)
- [Metric definitions](glossary.md)
- [Reproduction guide](reproduction.md)
