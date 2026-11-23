# Qwen Image 2.1 benchmarks

[Test hardware and platform limits](hardware.md).

[Model-specific VAE feature recipes](model-features.md#qwen-image) keep
decoder experiments separate from runtime comparisons. Linux feature results
are available; the matched Windows feature series remains pending.

Times are seconds. GPU peaks are sampled absolute driver usage, and RAM
peaks are process RSS, both in GiB. See the [measurement glossary](glossary.md).
In the runtime and attention tables, 🟢 ↓ marks a lower value and 🔴 ↑ marks a
higher value than the control for that table. Equal displayed values have no arrow.


**Full run** includes all tracked phases: component loading and setup, text
encoding, denoising, VAE decoding, cleanup, conversion and PNG saving. It excludes
calibration, command startup and downloads. Peaks cover the full tracked run.

**Denoising** reports the recorded sampling interval, including weight transfers
and scheduler work, rather than an isolated transformer-forward kernel timer.
Modular Qwen latency includes CPU latent readback. The Windows official Leaf
interval covers the sampling loop, including the scheduler and callbacks.
Both use their own monitored phase peaks, without added CUDA synchronization.

**VAE decode** excludes VAE loading, attachment, cleanup and PNG saving. Conversion
performed inside the recorded decode call remains included. Where only a
decode-call timer exists, latency is shown and its separate memory peaks remain
N/A. Windows official Leaf decode includes latent preparation and image
postprocessing, but excludes pipeline cleanup and saving. Linux and WSL Leaf runs record denoise and decode separately. The WSL
recipe also performs a synchronized cache cleanup between those phases. No phase is estimated by subtraction and
no aggregate or full-run peak is substituted for a narrower phase peak.

Average divides denoise wall latency by the recipe steps, retaining the same
warmup/readback scope. Detailed setup and output phases are available in the
linked phase measurements; nested intervals must not be added to their aggregates.

## Windows 1024×1024 — BF16 runtime control

Qwen Image 2.1 Base BF16, 20 steps, seed 42, default SDPA, and NVIDIA driver
617.42. The Leaf control is one standalone run after a manual Windows standby
cache purge; the runner's purge controls were disabled. The other rows use no
standby purge. This difference in initial conditions limits causal comparisons.
Every value for Leaf comes from that same run. Its generation aggregate includes
text encoding, denoising and VAE decode; the latter two are also measured separately.

#### Full run

| Preset | Transformer runtime | Text encoder route | VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | official leaf hooks | Leaf | Leaf | 107.66 | 8.00 | 35.50 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | linear_runtime | Leaf | Leaf | 66.08&nbsp;🟢&nbsp;↓ | 8.00 | 21.85&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | linear_runtime | Leaf | Leaf | 85.58&nbsp;🟢&nbsp;↓ | 8.00 | 27.03&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | vmm_runtime | Leaf | Leaf | 76.77&nbsp;🟢&nbsp;↓ | 8.00 | 18.97&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | vmm_fast_runtime | Native VMM | Native VMM | 60.05&nbsp;🟢&nbsp;↓ | 8.00 | 18.17&nbsp;🟢&nbsp;↓ |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | vmm_fast_runtime | Native VMM | Leaf | 62.70&nbsp;🟢&nbsp;↓ | 8.00 | 18.17&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 87.62 | 4.38 | 3.29 | 32.94 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 53.82&nbsp;🟢&nbsp;↓ | 2.69&nbsp;🟢&nbsp;↓ | 6.99&nbsp;🔴&nbsp;↑ | 13.80&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 65.53&nbsp;🟢&nbsp;↓ | 3.28&nbsp;🟢&nbsp;↓ | 6.57&nbsp;🔴&nbsp;↑ | 27.03&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 64.86&nbsp;🟢&nbsp;↓ | 3.24&nbsp;🟢&nbsp;↓ | 3.32&nbsp;🔴&nbsp;↑ | 6.00&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 49.66&nbsp;🟢&nbsp;↓ | 2.48&nbsp;🟢&nbsp;↓ | 2.45&nbsp;🟢&nbsp;↓ | 15.55&nbsp;🟢&nbsp;↓ |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 50.35&nbsp;🟢&nbsp;↓ | 2.52&nbsp;🟢&nbsp;↓ | 2.45&nbsp;🟢&nbsp;↓ | 15.55&nbsp;🟢&nbsp;↓ |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 5.59 | 8.00 | 35.47 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 2.61&nbsp;🟢&nbsp;↓ | 8.00 | 15.20&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 5.46&nbsp;🟢&nbsp;↓ | 8.00 | 15.21&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 5.31&nbsp;🟢&nbsp;↓ | 8.00 | 7.68&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 1.52&nbsp;🟢&nbsp;↓ | 8.00 | 3.19&nbsp;🟢&nbsp;↓ |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 1.95&nbsp;🟢&nbsp;↓ | 8.00 | 2.57&nbsp;🟢&nbsp;↓ |

### Transformer joint attention on VMM Fast

Both cases use the same 0.625 GiB transformer profile, native VMM encoder and
VAE, and no standby purge. Only masked joint attention in the transformer
changes. The default SDPA row is the VMM Fast runtime control above.

#### Full run

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 60.05 | 8.00 | 18.17 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 57.56&nbsp;🟢&nbsp;↓ | 8.00 | 18.17 |

#### Denoising

| Joint attention method | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 49.66 | 2.48 | 2.45 | 15.55 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 48.01&nbsp;🟢&nbsp;↓ | 2.40&nbsp;🟢&nbsp;↓ | 2.39&nbsp;🟢&nbsp;↓ | 15.57&nbsp;🔴&nbsp;↑ |

#### VAE decode

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 1.52 | 8.00 | 3.19 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 1.38&nbsp;🟢&nbsp;↓ | 8.00 | 3.16&nbsp;🟢&nbsp;↓ |

In one run per case, cuDNN reduced denoise/readback latency by 3.3% and tracked
total by 4.1%. The PNGs differed in RGB pixel values.

See [phase measurements](qwen-image-21-phases.md#windows-bf16) for this workload.

## Linux 1024×1024 — BF16 runtime control

The RC5 runs use NVIDIA driver 615.71.09, Torch 2.12.1+cu132,
Diffusers 0.41.0, default SDPA, and no standby purge. The Leaf control uses
streamed official hooks for encoder and transformer. Leaf and linear runs
use Leaf for the untiled VAE and PyTorch `expandable_segments:True`.
VMM Fast uses the native DDO allocator and native VMM encoder. Its two
rows compare native VMM and Leaf VAE routes with the same transformer profile.
The output PNGs were byte-identical.

#### Full run

| Preset | Transformer runtime | Text encoder route | VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | official leaf hooks | Leaf | Leaf | 70.85 | 7.46 | 33.66 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | linear_runtime | Leaf | Leaf | 67.51&nbsp;🟢&nbsp;↓ | 7.52&nbsp;🔴&nbsp;↑ | 21.51&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | linear_runtime | Leaf | Leaf | 71.60&nbsp;🔴&nbsp;↑ | 7.48&nbsp;🔴&nbsp;↑ | 22.41&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | vmm_fast_runtime | Native VMM | Native VMM | 53.98&nbsp;🟢&nbsp;↓ | 7.58&nbsp;🔴&nbsp;↑ | 18.27&nbsp;🟢&nbsp;↓ |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | vmm_fast_runtime | Native VMM | Leaf | 57.58&nbsp;🟢&nbsp;↓ | 7.53&nbsp;🔴&nbsp;↑ | 18.27&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 61.20 | 3.06 | 2.84 | 33.11 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 50.32&nbsp;🟢&nbsp;↓ | 2.52&nbsp;🟢&nbsp;↓ | 6.38&nbsp;🔴&nbsp;↑ | 13.97&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 56.22&nbsp;🟢&nbsp;↓ | 2.81&nbsp;🟢&nbsp;↓ | 6.34&nbsp;🔴&nbsp;↑ | 14.98&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 44.11&nbsp;🟢&nbsp;↓ | 2.21&nbsp;🟢&nbsp;↓ | 2.06&nbsp;🟢&nbsp;↓ | 15.70&nbsp;🟢&nbsp;↓ |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 47.19&nbsp;🟢&nbsp;↓ | 2.36&nbsp;🟢&nbsp;↓ | 2.14&nbsp;🟢&nbsp;↓ | 15.70&nbsp;🟢&nbsp;↓ |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 1.61 | 7.30 | 33.66 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 1.47&nbsp;🟢&nbsp;↓ | 7.52&nbsp;🔴&nbsp;↑ | 15.15&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 1.47&nbsp;🟢&nbsp;↓ | 7.48&nbsp;🔴&nbsp;↑ | 2.95&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 1.36&nbsp;🟢&nbsp;↓ | 7.58&nbsp;🔴&nbsp;↑ | 3.29&nbsp;🟢&nbsp;↓ |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 1.74&nbsp;🔴&nbsp;↑ | 7.53&nbsp;🔴&nbsp;↑ | 2.95&nbsp;🟢&nbsp;↓ |

The Leaf VAE alternative took 3.60 s longer overall and used 0.05 GiB less
peak GPU memory than the native VAE in these single runs. Both PNGs had
identical RGB pixels. The native VAE remains the selected VMM Fast route.

See [phase measurements](qwen-image-21-phases.md#linux-bf16) for this workload.


### Transformer joint attention on VMM Fast

Both cases use the same final 0.625 GiB transformer profile and native
encoder and VAE. The default SDPA row is the VMM Fast runtime measurement
above. A separate diagnostic confirmed the cuDNN operator. Timed runs
had tracing disabled.

#### Full run

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 53.98 | 7.58 | 18.27 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 55.06&nbsp;🔴&nbsp;↑ | 7.65&nbsp;🔴&nbsp;↑ | 18.27 |

#### Denoising

| Joint attention method | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 44.11 | 2.21 | 2.06 | 15.70 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 45.14&nbsp;🔴&nbsp;↑ | 2.26&nbsp;🔴&nbsp;↑ | 2.00&nbsp;🟢&nbsp;↓ | 15.74&nbsp;🔴&nbsp;↑ |

#### VAE decode

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 1.36 | 7.58 | 3.29 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 1.34&nbsp;🟢&nbsp;↓ | 7.65&nbsp;🔴&nbsp;↑ | 3.30&nbsp;🔴&nbsp;↑ |

In one measurement per case, cuDNN took 1.08 s longer overall and 1.02 s
longer through denoise/readback. The output RGB pixels differed. Default
SDPA was faster in this Linux comparison.

## WSL 1024×1024 — BF16 runtime control

All three runs use default SDPA, non-streamed Leaf for the encoder and VAE,
and no standby purge. Each row is one complete run on an RTX 3060 Ti with
NVIDIA driver `617.42`. The output PNGs were byte-identical. The official
Leaf recipe uses `PYTORCH_ALLOC_CONF=expandable_segments:False` and a separate
pre-decode cleanup; this recipe-local setting does not change other defaults.

#### Full run

| Preset | Transformer runtime | Text encoder route | VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | official leaf hooks | Leaf | Leaf | 420.09 | 8.00 | 37.82 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | linear_runtime | Leaf | Leaf | 226.38&nbsp;🟢&nbsp;↓ | 8.00 | 22.38&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | vmm_fast_runtime | Leaf | Leaf | 220.29&nbsp;🟢&nbsp;↓ | 8.00 | 22.40&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 313.73 | 15.69 | 3.42 | 37.82 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 107.54&nbsp;🟢&nbsp;↓ | 5.38&nbsp;🟢&nbsp;↓ | 6.49&nbsp;🔴&nbsp;↑ | 15.36&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 110.31&nbsp;🟢&nbsp;↓ | 5.52&nbsp;🟢&nbsp;↓ | 7.61&nbsp;🔴&nbsp;↑ | 16.05&nbsp;🟢&nbsp;↓ |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 5.10 | 8.00 | 33.04 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 5.39&nbsp;🔴&nbsp;↑ | 8.00 | 3.18&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 2.81&nbsp;🟢&nbsp;↓ | 8.00 | 3.06&nbsp;🟢&nbsp;↓ |

See [phase measurements](qwen-image-21-phases.md#wsl-bf16) for this workload.


### Transformer joint attention on VMM Fast

Both cases reuse the same 4.34 GiB transformer profile and non-streamed Leaf
encoder and VAE. The cuDNN operator was confirmed in a separate diagnostic run.
The table uses the unprofiled measurement. Denoise latency includes CPU latent
readback.

#### Full run

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 220.29 | 8.00 | 22.40 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 228.63&nbsp;🔴&nbsp;↑ | 8.00 | 22.39&nbsp;🟢&nbsp;↓ |

#### Denoising

| Joint attention method | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 110.31 | 5.52 | 7.61 | 16.05 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 122.10&nbsp;🔴&nbsp;↑ | 6.11&nbsp;🔴&nbsp;↑ | 7.57&nbsp;🟢&nbsp;↓ | 16.08&nbsp;🔴&nbsp;↑ |

#### VAE decode

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 2.81 | 8.00 | 3.06 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 2.83&nbsp;🔴&nbsp;↑ | 8.00 | 3.11&nbsp;🔴&nbsp;↑ |

The cuDNN run took 8.34 s longer overall and produced different RGB pixels.

## Windows — saved SDNQ INT4

RC5 measurements use Base auxiliaries, 1024×1024, 20 steps, seed 42,
guidance 1, default KV cache and default SDPA. Both cases use BF16 Leaf
encoder and VAE routes, without standby purge or model optimizations.
DDO preserves the packed SDNQ forward and applies its own 6 GiB profile.

#### Full run

| Preset | Transformer runtime | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | official Leaf hooks | 86.03 | 8.00 | 22.28 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | sdnq_runtime | 69.80&nbsp;🟢&nbsp;↓ | 8.00 | 22.26&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 58.40 | 2.92 | 2.53 | 18.88 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 50.07&nbsp;🟢&nbsp;↓ | 2.50&nbsp;🟢&nbsp;↓ | 6.64&nbsp;🔴&nbsp;↑ | 15.06&nbsp;🟢&nbsp;↓ |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 5.20 | 8.00 | 16.40 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 5.10&nbsp;🟢&nbsp;↓ | 8.00 | 16.43&nbsp;🔴&nbsp;↑ |

The PNGs are identical pixel for pixel within this pair. Denoise/readback
was 14.3% shorter in the DDO observation. These are sequential single
measurements: the unchanged Leaf encoder phase fell from 19.98 s to
10.94 s in the later run. The 18.9% shorter tracked total cannot be
attributed entirely to transformer offloading. Calibration is excluded;
this pair does not assess INT4 quality against BF16. See the
[recipe](reproduction.md#qwen-saved-sdnq-int4-pair) and
[phase measurements](qwen-image-21-phases.md#windows-sdnq-int4).

## Recommendations

For this 1024-square Windows workload on the tested 8 GiB GPU, `vmm_fast`
with a native encoder and native VAE had the shortest tracked total and lowest
peak process RSS among the measured routes. Its GPU peak occurred during VAE
decode. Consider VAE tiling if decode activations exceed
available GPU memory. See [presets](../user/presets.md) and
[reproduction](reproduction.md).

On the tested WSL configuration, both profiled DDO runtimes finished in
roughly half the Leaf control's tracked time. VMM Fast had the shortest
tracked total. Of the two staged runtimes, `wsl_compat` used less peak GPU
memory during denoise. All three reached the same absolute GPU peak across
the pipeline. Default SDPA was faster in the VMM Fast attention comparison.
