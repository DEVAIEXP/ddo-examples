# Qwen Image 2.1 phase measurements

[Results and recommendations](qwen-image-21.md) · [Measurement details](qwen-image-21-details.md) · [Measurement glossary](glossary.md)

Each phase uses presets or attention methods as rows. Latency is wall time; GPU peaks are sampled absolute driver usage and RAM peaks are process RSS, in GiB. Values come from the same selected runs as the main results. Full-run totals remain on that page.

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
postprocessing, but excludes pipeline cleanup and saving. Linux and WSL Leaf
observations lack isolated intervals. No phase is estimated by subtraction and
no aggregate or full-run peak is substituted for a narrower phase peak.

Average divides denoise wall latency by the recipe steps, retaining the same
warmup/readback scope. Detailed setup and output phases are available in the
linked phase measurements; nested intervals must not be added to their aggregates.


Text encoder aggregates include loading, attachment where applicable, encoding and cleanup. Their latency sums disjoint recorded phases and their memory is the maximum recorded peak. Other setup/output phases are expandable below. N/A means unavailable, not zero.

<a id="windows-bf16"></a>

## Windows 1024×1024 — BF16 runtime control

The Leaf rows use the same staged run as the main comparison, with no standby
purge. Collect phases report DDO metrics and do not exist for Leaf.

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 12.29 | 2.28 | 18.97 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 4.83 | 2.28 | 18.97 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 11.02 | 3.41 | 22.32 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 4.06 | 2.28 | 18.97 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 5.74 | 3.44 | 18.17 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 6.17 | 3.44 | 18.17 |

### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 87.81 | 2.13 | 16.99 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 53.82 | 6.99 | 13.80 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 65.53 | 6.57 | 27.03 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 64.86 | 3.32 | 6.00 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 49.66 | 2.45 | 15.55 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 50.35 | 2.45 | 15.55 |

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 2.16 | 8.00 | 4.74 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 2.61 | 8.00 | 15.20 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 5.46 | 8.00 | 15.21 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 5.31 | 8.00 | 7.68 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 1.52 | 8.00 | 3.19 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 1.95 | 8.00 | 2.57 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 0.10 | 1.10 | 4.67 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 0.10 | 1.10 | 14.69 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 0.10 | 2.23 | 14.37 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 0.09 | 1.10 | 6.62 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.09 | 7.28 | 1.90 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 0.13 | 7.25 | 2.57 |

**Load transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 0.08 | 1.10 | 3.66 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 0.05 | 1.10 | 3.66 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 0.07 | 2.22 | 13.73 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 0.09 | 1.10 | 3.66 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.08 | 1.11 | 1.53 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 0.08 | 1.11 | 1.53 |

**Attach transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 0.03 | 1.10 | 3.67 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 3.99 | 6.00 | 21.85 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 1.70 | 5.57 | 18.16 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 1.23 | 1.50 | 5.91 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 1.04 | 1.11 | 2.33 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 1.25 | 1.11 | 2.21 |

**Prepare latents**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 0.00 | 1.10 | 3.67 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 0.01 | 6.00 | 13.74 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 0.01 | 5.60 | 18.22 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 0.01 | 1.35 | 5.93 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.03 | 1.11 | 2.57 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 0.02 | 1.11 | 2.55 |

**Collect transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | N/A | N/A | N/A |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 0.00 | 6.94 | 13.80 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 0.00 | 6.55 | 27.03 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 0.02 | 3.32 | 6.00 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.00 | 2.45 | 15.54 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 0.00 | 2.45 | 15.54 |

**Cleanup transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 0.70 | 2.13 | 16.98 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 0.15 | 6.94 | 13.80 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 0.74 | 6.55 | 27.03 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 0.13 | 3.32 | 6.00 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 1.00 | 2.45 | 15.54 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 1.25 | 2.45 | 15.54 |

**Load VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 0.85 | 1.10 | 5.52 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 0.27 | 1.10 | 15.02 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 0.25 | 2.23 | 14.94 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 0.29 | 1.10 | 6.86 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.06 | 1.85 | 1.61 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 0.94 | 1.85 | 3.25 |

**Collect VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | N/A | N/A | N/A |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 0.00 | 8.00 | 15.27 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 0.00 | 8.00 | 15.24 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 0.00 | 8.00 | 7.70 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.00 | 7.58 | 3.21 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 0.00 | 7.22 | 2.59 |

**Cleanup VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 0.18 | 8.00 | 4.82 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | 0.23 | 8.00 | 15.27 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | 0.23 | 8.00 | 15.24 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | 0.23 | 8.00 | 7.70 |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.24 | 7.58 | 3.21 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | 0.12 | 7.22 | 2.59 |

**Attach VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#windows-1024-leaf-control) | 0.02 | 1.10 | 4.36 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-1024-one-shot-fast) | N/A | N/A | N/A |
| [`low_ram_safe`](qwen-image-21-details.md#windows-1024-low-ram-safe) | N/A | N/A | N/A |
| [`nvme_baseline`](qwen-image-21-details.md#windows-1024-nvme-baseline) | N/A | N/A | N/A |
| [`vmm_fast`](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.12 | 1.85 | 2.23 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#windows-1024-vmm-fast-leaf) | N/A | N/A | N/A |

</details>

## Transformer joint attention on VMM Fast

### Text encoder

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 5.74 | 3.44 | 18.17 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 5.30 | 3.44 | 18.17 |

### Denoising

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 49.66 | 2.45 | 15.55 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 48.01 | 2.39 | 15.57 |

### VAE decode

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 1.52 | 8.00 | 3.19 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 1.38 | 8.00 | 3.16 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Load transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.08 | 1.11 | 1.53 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.06 | 1.11 | 1.53 |

**Attach transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 1.04 | 1.11 | 2.33 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.93 | 1.11 | 2.17 |

**Prepare latents**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.03 | 1.11 | 2.57 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.02 | 1.11 | 2.59 |

**Collect transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.00 | 2.45 | 15.54 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.00 | 2.39 | 15.56 |

**Cleanup transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 1.00 | 2.45 | 15.54 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.95 | 2.39 | 15.56 |

**Load VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.06 | 1.85 | 1.61 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.03 | 1.80 | 1.63 |

**Attach VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.12 | 1.85 | 2.23 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.13 | 1.80 | 2.25 |

**Collect VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.00 | 7.58 | 3.21 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.00 | 7.58 | 3.19 |

**Cleanup VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.24 | 7.58 | 3.21 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.21 | 7.58 | 3.19 |

**Save image**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#windows-1024-vmm-fast-native-vae) | 0.09 | 7.28 | 1.90 |
| [cuDNN SDPA](qwen-image-21-details.md#windows-1024-joint-attention) | 0.09 | 7.26 | 1.87 |

</details>

<a id="linux-bf16"></a>

## Linux 1024×1024 — BF16 runtime control

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 6.10 | 1.71 | 19.08 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 6.42 | 1.87 | 19.09 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 11.31 | 1.84 | 22.41 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 5.10 | 3.04 | 18.27 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 5.37 | 3.10 | 18.27 |

### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 60.44 | 1.44 | 17.14 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 50.32 | 6.38 | 13.97 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 56.22 | 6.34 | 14.98 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 44.11 | 2.06 | 15.70 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 47.19 | 2.14 | 15.70 |

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 1.54 | 7.36 | 5.07 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 1.47 | 7.52 | 15.15 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 1.47 | 7.48 | 2.95 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 1.36 | 7.58 | 3.29 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 1.74 | 7.53 | 2.95 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 0.41 | 0.52 | 4.13 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 0.55 | 0.70 | 14.21 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 0.41 | 0.66 | 1.96 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.41 | 6.85 | 1.99 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 0.41 | 6.43 | 2.02 |

**Load transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 0.11 | 0.52 | 3.78 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 0.04 | 0.68 | 3.76 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 0.03 | 0.65 | 1.59 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.04 | 0.69 | 1.62 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 0.04 | 0.75 | 1.62 |

**Attach transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 0.01 | 0.52 | 3.78 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 7.25 | 5.60 | 21.51 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 0.59 | 4.83 | 5.78 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.75 | 0.69 | 3.94 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 0.84 | 0.75 | 4.81 |

**Prepare latents**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 0.02 | 0.52 | 3.78 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 0.01 | 5.60 | 13.84 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 0.01 | 5.56 | 6.52 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.02 | 0.69 | 4.45 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 0.03 | 0.75 | 5.12 |

**Collect transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | N/A | N/A | N/A |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 0.00 | 6.38 | 13.97 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 0.00 | 6.34 | 14.98 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.00 | 2.06 | 15.70 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 0.00 | 2.09 | 15.70 |

**Cleanup transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 0.48 | 1.44 | 17.14 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 0.26 | 6.38 | 13.97 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 0.66 | 6.34 | 14.98 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 1.16 | 2.06 | 15.70 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 1.17 | 2.09 | 15.70 |

**Load VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 0.36 | 0.52 | 5.60 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 0.39 | 0.70 | 15.58 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 0.15 | 0.65 | 3.51 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.05 | 1.43 | 1.76 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 0.15 | 1.45 | 3.44 |

**Collect VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | N/A | N/A | N/A |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 0.00 | 7.46 | 15.15 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 0.00 | 7.42 | 2.96 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.00 | 7.18 | 3.31 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 0.00 | 6.43 | 2.95 |

**Cleanup VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 0.34 | 7.28 | 5.07 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | 0.38 | 7.46 | 15.15 |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | 0.35 | 7.42 | 2.96 |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.24 | 7.18 | 3.31 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | 0.19 | 6.43 | 2.95 |

**Attach VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#linux-1024-leaf-control) | 0.01 | 0.52 | 4.52 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-1024-one-shot-fast) | N/A | N/A | N/A |
| [`low_ram_safe`](qwen-image-21-details.md#linux-1024-low-ram-safe) | N/A | N/A | N/A |
| [`vmm_fast`](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.33 | 1.43 | 2.75 |
| [`vmm_fast` / Leaf VAE](qwen-image-21-details.md#linux-1024-vmm-fast-leaf) | N/A | N/A | N/A |

</details>

## Transformer joint attention on VMM Fast

### Text encoder

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 5.10 | 3.04 | 18.27 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 5.15 | 3.04 | 18.27 |

### Denoising

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 44.11 | 2.06 | 15.70 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 45.14 | 2.00 | 15.74 |

### VAE decode

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 1.36 | 7.58 | 3.29 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 1.34 | 7.65 | 3.30 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Load transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.04 | 0.69 | 1.62 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 0.04 | 0.69 | 1.62 |

**Attach transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.75 | 0.69 | 3.94 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 0.81 | 0.70 | 4.95 |

**Prepare latents**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.02 | 0.69 | 4.45 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 0.02 | 0.70 | 5.01 |

**Collect transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.00 | 2.06 | 15.70 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 0.00 | 2.00 | 15.74 |

**Cleanup transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 1.16 | 2.06 | 15.70 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 1.16 | 2.00 | 15.74 |

**Load VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.05 | 1.43 | 1.76 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 0.02 | 1.38 | 1.80 |

**Attach VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.33 | 1.43 | 2.75 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 0.34 | 1.38 | 3.02 |

**Collect VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.00 | 7.18 | 3.31 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 0.00 | 7.18 | 3.31 |

**Cleanup VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.24 | 7.18 | 3.31 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 0.23 | 7.18 | 3.31 |

**Save image**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#linux-1024-vmm-fast) | 0.41 | 6.85 | 1.99 |
| [cuDNN SDPA](qwen-image-21-details.md#linux-1024-joint-attention) | 0.41 | 6.86 | 1.99 |

</details>

<a id="wsl-bf16"></a>


## WSL 1024×1024 — BF16 runtime control

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 84.88 | 3.42 | 22.39 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 89.53 | 3.42 | 22.38 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 98.74 | 3.40 | 22.40 |

### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 312.09 | 2.26 | 20.62 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 107.54 | 6.49 | 15.36 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 110.31 | 7.61 | 16.05 |

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 5.33 | 8.00 | 3.09 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 5.39 | 8.00 | 3.18 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 2.81 | 8.00 | 3.06 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 0.45 | 2.24 | 3.09 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 0.46 | 2.24 | 3.18 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.45 | 7.07 | 3.06 |

**Load transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 0.20 | 2.24 | 2.09 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 0.17 | 2.24 | 2.09 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.20 | 3.40 | 2.10 |

**Attach transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 0.01 | 2.24 | 2.10 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 18.09 | 5.61 | 6.59 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 2.59 | 3.41 | 3.01 |

**Prepare latents**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 0.01 | 2.24 | 2.10 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 0.02 | 5.61 | 6.60 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.01 | 3.41 | 3.04 |

**Collect transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | N/A | N/A | N/A |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 0.00 | 6.49 | 15.36 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.00 | 7.59 | 16.05 |

**Cleanup transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 0.93 | 2.26 | 15.35 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 1.51 | 6.49 | 15.36 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 1.36 | 7.59 | 16.05 |

**Load VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 3.03 | 2.24 | 3.82 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 2.95 | 2.24 | 3.90 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 3.17 | 3.41 | 3.83 |

**Collect VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | N/A | N/A | N/A |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 0.00 | 8.00 | 3.18 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.00 | 7.07 | 3.06 |

**Cleanup VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Diffusers Leaf](qwen-image-21-details.md#wsl-1024-leaf-control) | 0.39 | 8.00 | 3.09 |
| [`wsl_compat`](qwen-image-21-details.md#wsl-1024-wsl-compat) | 0.20 | 8.00 | 3.18 |
| [`vmm_fast`](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.12 | 7.07 | 3.06 |

</details>


## Transformer joint attention on VMM Fast

### Text encoder

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 98.74 | 3.40 | 22.40 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 95.10 | 3.40 | 22.39 |

### Denoising

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 110.31 | 7.61 | 16.05 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 122.10 | 7.57 | 16.08 |

### VAE decode

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 2.81 | 8.00 | 3.06 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 2.83 | 8.00 | 3.11 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Load transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.20 | 3.40 | 2.10 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 0.22 | 3.40 | 2.10 |

**Attach transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 2.59 | 3.41 | 3.01 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 2.75 | 3.41 | 3.05 |

**Prepare latents**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.01 | 3.41 | 3.04 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 0.01 | 3.41 | 3.05 |

**Collect transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.00 | 7.59 | 16.05 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 0.00 | 7.57 | 16.08 |

**Cleanup transformer**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 1.36 | 7.59 | 16.05 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 1.35 | 7.57 | 16.08 |

**Load VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 3.17 | 3.41 | 3.83 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 3.16 | 3.41 | 3.86 |

**Collect VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.00 | 7.07 | 3.06 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 0.00 | 7.00 | 3.11 |

**Cleanup VAE**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.12 | 7.07 | 3.06 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 0.12 | 7.00 | 3.11 |

**Save image**

| Joint attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](qwen-image-21-details.md#wsl-1024-vmm-fast) | 0.45 | 7.07 | 3.06 |
| [cuDNN SDPA](qwen-image-21-details.md#wsl-1024-joint-attention) | 0.46 | 7.00 | 3.11 |

</details>

<a id="windows-sdnq-int4"></a>

## Windows — saved SDNQ INT4

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 19.98 | 3.41 | 22.28 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 10.94 | 3.41 | 22.26 |

### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 58.40 | 2.53 | 18.88 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 50.07 | 6.64 | 15.06 |

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 5.20 | 8.00 | 16.40 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 5.10 | 8.00 | 16.43 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Load transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 0.44 | 2.22 | 14.98 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 0.44 | 2.22 | 14.98 |

**Attach transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 0.02 | 2.22 | 14.95 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 2.02 | 4.91 | 18.75 |

**Prepare latents**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 0.00 | 2.24 | 14.95 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 0.02 | 4.91 | 14.98 |

**Collect transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 0.00 | 2.53 | 18.88 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 0.00 | 6.64 | 15.05 |

**Cleanup transformer**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 0.32 | 2.53 | 18.88 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 0.17 | 6.64 | 15.05 |

**Load VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 0.86 | 2.24 | 16.54 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 0.23 | 2.23 | 16.25 |

**Collect VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 0.00 | 8.00 | 16.43 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 0.00 | 8.00 | 16.46 |

**Cleanup VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 0.23 | 8.00 | 16.43 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 0.23 | 8.00 | 16.46 |

**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](qwen-image-21-details.md#windows-sdnq-int4) (reference) | 0.09 | 2.24 | 15.60 |
| [SDNQ INT4 DDO](qwen-image-21-details.md#windows-sdnq-int4) | 0.09 | 2.23 | 15.63 |

</details>

<a id="linux-qwen-vae-features"></a>

## Linux Qwen VAE features

The [Qwen feature observations](model-features.md#linux-qwen-vae-observations)
share one final 0.625 GiB transformer profile and native encoder/VAE routes.
Every row uses its own completed observation. GPU peaks are absolute and
RAM peaks are process RSS in GiB. Calibration is excluded.

### Text encoder

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 2.10 | 3.12 | 18.24 |
| `single-frame` | 2.12 | 3.11 | 18.22 |
| `residual-alias` | 2.06 | 3.11 | 18.22 |
| `direct-padding` | 2.08 | 3.05 | 18.22 |
| `bands-256` | 2.10 | 3.05 | 18.22 |
| `bands-128` | 2.12 | 3.05 | 18.22 |
| `bands-256-no-direct-padding` | 2.08 | 3.02 | 18.22 |
| `bands-128-no-direct-padding` | 2.08 | 3.03 | 18.22 |
| `tiling-only` | 2.09 | 3.02 | 18.22 |

### Denoising

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 44.30 | 2.13 | 15.71 |
| `single-frame` | 44.53 | 2.13 | 15.69 |
| `residual-alias` | 44.68 | 2.13 | 15.69 |
| `direct-padding` | 44.68 | 2.07 | 15.69 |
| `bands-256` | 44.69 | 2.07 | 15.69 |
| `bands-128` | 44.77 | 2.06 | 15.69 |
| `bands-256-no-direct-padding` | 44.58 | 2.05 | 15.69 |
| `bands-128-no-direct-padding` | 44.60 | 2.05 | 15.69 |
| `tiling-only` | 44.61 | 2.05 | 15.69 |

### VAE decode

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.41 | 7.66 | 3.34 |
| `single-frame` | 1.21 | 5.49 | 3.34 |
| `residual-alias` | 1.18 | 4.76 | 3.27 |
| `direct-padding` | 1.13 | 4.75 | 3.28 |
| `bands-256` | 1.21 | 2.93 | 3.31 |
| `bands-128` | 1.31 | 2.91 | 3.29 |
| `bands-256-no-direct-padding` | 1.26 | 2.77 | 3.27 |
| `bands-128-no-direct-padding` | 1.37 | 2.77 | 3.27 |
| `tiling-only` | 2.98 | 1.73 | 3.28 |

<details markdown="1">
<summary>Other setup and output phases</summary>

**Load text encoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.33 | 0.67 | 0.98 |
| `single-frame` | 0.33 | 0.67 | 0.96 |
| `residual-alias` | 0.34 | 0.67 | 0.96 |
| `direct-padding` | 0.34 | 0.61 | 0.96 |
| `bands-256` | 0.34 | 0.61 | 0.96 |
| `bands-128` | 0.34 | 0.61 | 0.96 |
| `bands-256-no-direct-padding` | 0.33 | 0.59 | 0.96 |
| `bands-128-no-direct-padding` | 0.33 | 0.59 | 0.96 |
| `tiling-only` | 0.33 | 0.59 | 0.96 |

**Attach text encoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.14 | 0.70 | 5.47 |
| `single-frame` | 1.21 | 0.70 | 5.25 |
| `residual-alias` | 1.25 | 0.69 | 6.47 |
| `direct-padding` | 1.23 | 0.63 | 6.63 |
| `bands-256` | 1.25 | 0.63 | 6.52 |
| `bands-128` | 1.24 | 0.63 | 6.26 |
| `bands-256-no-direct-padding` | 1.29 | 0.61 | 6.31 |
| `bands-128-no-direct-padding` | 1.28 | 0.61 | 6.33 |
| `tiling-only` | 1.25 | 0.61 | 6.35 |

**Cleanup text encoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.45 | 3.12 | 18.28 |
| `single-frame` | 1.43 | 3.12 | 18.26 |
| `residual-alias` | 1.43 | 3.11 | 18.26 |
| `direct-padding` | 1.43 | 3.06 | 18.26 |
| `bands-256` | 1.44 | 3.06 | 18.26 |
| `bands-128` | 1.44 | 3.06 | 18.26 |
| `bands-256-no-direct-padding` | 1.44 | 3.03 | 18.26 |
| `bands-128-no-direct-padding` | 1.43 | 3.03 | 18.26 |
| `tiling-only` | 1.44 | 3.03 | 18.26 |

**Load transformer**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.04 | 0.77 | 1.63 |
| `single-frame` | 0.04 | 0.77 | 1.61 |
| `residual-alias` | 0.04 | 0.76 | 1.61 |
| `direct-padding` | 0.04 | 0.71 | 1.61 |
| `bands-256` | 0.04 | 0.71 | 1.61 |
| `bands-128` | 0.04 | 0.71 | 1.61 |
| `bands-256-no-direct-padding` | 0.04 | 0.68 | 1.61 |
| `bands-128-no-direct-padding` | 0.04 | 0.68 | 1.61 |
| `tiling-only` | 0.04 | 0.68 | 1.61 |

**Attach transformer**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.77 | 0.77 | 3.96 |
| `single-frame` | 0.83 | 0.77 | 5.08 |
| `residual-alias` | 0.79 | 0.77 | 3.96 |
| `direct-padding` | 0.85 | 0.71 | 4.86 |
| `bands-256` | 0.77 | 0.71 | 3.94 |
| `bands-128` | 0.85 | 0.71 | 4.99 |
| `bands-256-no-direct-padding` | 0.85 | 0.68 | 4.96 |
| `bands-128-no-direct-padding` | 0.76 | 0.68 | 3.96 |
| `tiling-only` | 0.84 | 0.68 | 5.07 |

**Prepare latents**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.03 | 0.77 | 4.62 |
| `single-frame` | 0.03 | 0.77 | 5.34 |
| `residual-alias` | 0.03 | 0.77 | 4.85 |
| `direct-padding` | 0.03 | 0.71 | 5.24 |
| `bands-256` | 0.03 | 0.71 | 4.65 |
| `bands-128` | 0.03 | 0.71 | 5.48 |
| `bands-256-no-direct-padding` | 0.03 | 0.68 | 5.49 |
| `bands-128-no-direct-padding` | 0.03 | 0.68 | 4.57 |
| `tiling-only` | 0.03 | 0.68 | 5.45 |

**Collect transformer**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.00 | 2.13 | 15.71 |
| `single-frame` | 0.00 | 2.13 | 15.69 |
| `residual-alias` | 0.00 | 2.07 | 15.69 |
| `direct-padding` | 0.00 | 2.07 | 15.69 |
| `bands-256` | 0.00 | 2.07 | 15.69 |
| `bands-128` | 0.00 | 2.05 | 15.69 |
| `bands-256-no-direct-padding` | 0.00 | 2.05 | 15.69 |
| `bands-128-no-direct-padding` | 0.00 | 2.05 | 15.69 |
| `tiling-only` | 0.00 | 2.02 | 15.69 |

**Cleanup transformer**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.16 | 2.13 | 15.71 |
| `single-frame` | 1.16 | 2.13 | 15.69 |
| `residual-alias` | 1.17 | 2.07 | 15.69 |
| `direct-padding` | 1.16 | 2.07 | 15.69 |
| `bands-256` | 1.17 | 2.07 | 15.69 |
| `bands-128` | 1.17 | 2.05 | 15.69 |
| `bands-256-no-direct-padding` | 1.16 | 2.05 | 15.69 |
| `bands-128-no-direct-padding` | 1.16 | 2.05 | 15.69 |
| `tiling-only` | 1.16 | 2.02 | 15.69 |

**Load VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.02 | 1.51 | 1.77 |
| `single-frame` | 0.02 | 1.51 | 1.75 |
| `residual-alias` | 0.02 | 1.45 | 1.75 |
| `direct-padding` | 0.03 | 1.45 | 1.75 |
| `bands-256` | 0.03 | 1.45 | 1.75 |
| `bands-128` | 0.03 | 1.42 | 1.75 |
| `bands-256-no-direct-padding` | 0.02 | 1.42 | 1.75 |
| `bands-128-no-direct-padding` | 0.02 | 1.42 | 1.75 |
| `tiling-only` | 0.02 | 1.40 | 1.75 |

**Attach VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.33 | 1.51 | 2.74 |
| `single-frame` | 0.34 | 1.51 | 2.93 |
| `residual-alias` | 0.34 | 1.45 | 2.89 |
| `direct-padding` | 0.34 | 1.45 | 2.71 |
| `bands-256` | 0.34 | 1.45 | 2.69 |
| `bands-128` | 0.34 | 1.42 | 2.73 |
| `bands-256-no-direct-padding` | 0.33 | 1.42 | 2.74 |
| `bands-128-no-direct-padding` | 0.34 | 1.42 | 2.69 |
| `tiling-only` | 0.34 | 1.40 | 2.69 |

**Collect VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.00 | 7.26 | 3.31 |
| `single-frame` | 0.00 | 5.24 | 3.29 |
| `residual-alias` | 0.00 | 4.76 | 3.28 |
| `direct-padding` | 0.00 | 4.75 | 3.28 |
| `bands-256` | 0.00 | 2.93 | 3.28 |
| `bands-128` | 0.00 | 2.91 | 3.27 |
| `bands-256-no-direct-padding` | 0.00 | 2.77 | 3.28 |
| `bands-128-no-direct-padding` | 0.00 | 2.77 | 3.28 |
| `tiling-only` | 0.00 | 1.73 | 3.29 |

**Cleanup VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.23 | 7.26 | 3.31 |
| `single-frame` | 0.22 | 5.24 | 3.29 |
| `residual-alias` | 0.23 | 4.76 | 3.28 |
| `direct-padding` | 0.22 | 4.75 | 3.28 |
| `bands-256` | 0.23 | 2.93 | 3.28 |
| `bands-128` | 0.23 | 2.91 | 3.27 |
| `bands-256-no-direct-padding` | 0.22 | 2.77 | 3.28 |
| `bands-128-no-direct-padding` | 0.22 | 2.77 | 3.28 |
| `tiling-only` | 0.23 | 1.73 | 3.29 |

**Save image**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.41 | 6.93 | 1.99 |
| `single-frame` | 0.41 | 4.92 | 1.97 |
| `residual-alias` | 0.41 | 4.43 | 1.96 |
| `direct-padding` | 0.40 | 4.43 | 1.97 |
| `bands-256` | 0.40 | 2.60 | 1.96 |
| `bands-128` | 0.41 | 2.58 | 1.96 |
| `bands-256-no-direct-padding` | 0.41 | 2.44 | 1.96 |
| `bands-128-no-direct-padding` | 0.41 | 2.44 | 1.97 |
| `tiling-only` | 0.41 | 1.40 | 1.97 |

</details>

## Windows Qwen VAE features

The [Windows Qwen feature series](model-features.md#windows-qwen-vae-observations)
uses one shared final 0.625 GiB transformer profile and native encoder/VAE.
Each row is from its own completed run. GPU peaks are absolute and RAM peaks
are process RSS in GiB. Calibration is excluded; nested phases are not additive.

### Text encoder

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 2.24 | 3.44 | 18.13 |
| `single-frame` | 2.34 | 3.44 | 18.16 |
| `residual-alias` | 2.29 | 3.44 | 18.13 |
| `direct-padding` | 2.27 | 3.44 | 18.13 |
| `bands-256` | 2.24 | 3.44 | 18.14 |
| `bands-128` | 2.23 | 3.44 | 18.14 |
| `bands-256-no-direct-padding` | 2.23 | 3.44 | 18.13 |
| `bands-128-no-direct-padding` | 2.23 | 3.44 | 18.13 |
| `tiling-only` | 2.23 | 3.44 | 18.13 |

### Denoising

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 50.10 | 2.45 | 15.55 |
| `single-frame` | 50.12 | 2.45 | 15.54 |
| `residual-alias` | 50.36 | 2.45 | 15.55 |
| `direct-padding` | 50.42 | 2.45 | 15.55 |
| `bands-256` | 50.44 | 2.45 | 15.55 |
| `bands-128` | 50.45 | 2.45 | 15.55 |
| `bands-256-no-direct-padding` | 50.39 | 2.45 | 15.55 |
| `bands-128-no-direct-padding` | 50.41 | 2.45 | 15.55 |
| `tiling-only` | 50.41 | 2.45 | 15.55 |

### VAE decode

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.47 | 8.00 | 3.12 |
| `single-frame` | 1.22 | 5.84 | 3.09 |
| `residual-alias` | 1.21 | 5.16 | 3.11 |
| `direct-padding` | 1.15 | 5.16 | 3.08 |
| `bands-256` | 1.21 | 3.34 | 3.11 |
| `bands-128` | 1.34 | 3.34 | 3.08 |
| `bands-256-no-direct-padding` | 1.28 | 3.20 | 3.08 |
| `bands-128-no-direct-padding` | 1.39 | 3.20 | 3.08 |
| `tiling-only` | 3.36 | 2.18 | 3.09 |

<details markdown="1">
<summary>Other setup and output phases</summary>

**Load text encoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.41 | 1.03 | 0.79 |
| `single-frame` | 0.43 | 1.03 | 0.79 |
| `residual-alias` | 0.40 | 1.03 | 0.79 |
| `direct-padding` | 0.41 | 1.03 | 0.79 |
| `bands-256` | 0.40 | 1.03 | 0.79 |
| `bands-128` | 0.41 | 1.03 | 0.79 |
| `bands-256-no-direct-padding` | 0.39 | 1.03 | 0.79 |
| `bands-128-no-direct-padding` | 0.40 | 1.03 | 0.79 |
| `tiling-only` | 0.41 | 1.03 | 0.79 |

**Attach text encoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.55 | 1.04 | 1.52 |
| `single-frame` | 1.53 | 1.04 | 1.52 |
| `residual-alias` | 1.51 | 1.04 | 1.52 |
| `direct-padding` | 1.46 | 1.04 | 1.87 |
| `bands-256` | 1.57 | 1.04 | 1.96 |
| `bands-128` | 1.52 | 1.04 | 1.52 |
| `bands-256-no-direct-padding` | 1.51 | 1.04 | 1.52 |
| `bands-128-no-direct-padding` | 1.44 | 1.04 | 1.52 |
| `tiling-only` | 1.65 | 1.04 | 1.52 |

**Cleanup text encoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.96 | 3.44 | 18.17 |
| `single-frame` | 1.24 | 3.44 | 18.17 |
| `residual-alias` | 1.00 | 3.44 | 18.17 |
| `direct-padding` | 1.16 | 3.44 | 18.17 |
| `bands-256` | 1.01 | 3.44 | 18.17 |
| `bands-128` | 1.02 | 3.44 | 18.17 |
| `bands-256-no-direct-padding` | 1.21 | 3.44 | 18.17 |
| `bands-128-no-direct-padding` | 1.04 | 3.44 | 18.17 |
| `tiling-only` | 1.20 | 3.44 | 18.17 |

**Purge before transformer**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.00 | 1.11 | 1.53 |
| `single-frame` | 0.00 | 1.11 | 1.53 |
| `residual-alias` | 0.00 | 1.11 | 1.53 |
| `direct-padding` | 0.00 | 1.11 | 1.53 |
| `bands-256` | 0.00 | 1.11 | 1.53 |
| `bands-128` | 0.00 | 1.11 | 1.53 |
| `bands-256-no-direct-padding` | 0.02 | 1.11 | 1.53 |
| `bands-128-no-direct-padding` | 0.00 | 1.11 | 1.53 |
| `tiling-only` | 0.02 | 1.11 | 1.53 |

**Load transformer**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.07 | 1.11 | 1.53 |
| `single-frame` | 0.09 | 1.11 | 1.53 |
| `residual-alias` | 0.05 | 1.11 | 1.53 |
| `direct-padding` | 0.08 | 1.11 | 1.53 |
| `bands-256` | 0.05 | 1.11 | 1.53 |
| `bands-128` | 0.06 | 1.11 | 1.53 |
| `bands-256-no-direct-padding` | 0.05 | 1.11 | 1.53 |
| `bands-128-no-direct-padding` | 0.11 | 1.11 | 1.53 |
| `tiling-only` | 0.05 | 1.11 | 1.53 |

**Attach transformer**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.23 | 1.11 | 2.17 |
| `single-frame` | 1.04 | 1.11 | 2.17 |
| `residual-alias` | 1.25 | 1.11 | 2.17 |
| `direct-padding` | 1.09 | 1.11 | 2.17 |
| `bands-256` | 1.20 | 1.11 | 2.17 |
| `bands-128` | 1.21 | 1.11 | 2.17 |
| `bands-256-no-direct-padding` | 1.05 | 1.11 | 2.17 |
| `bands-128-no-direct-padding` | 1.10 | 1.11 | 2.17 |
| `tiling-only` | 1.12 | 1.11 | 2.17 |

**Prepare**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.02 | 1.11 | 2.61 |
| `single-frame` | 0.02 | 1.11 | 2.62 |
| `residual-alias` | 0.02 | 1.11 | 2.56 |
| `direct-padding` | 0.02 | 1.11 | 2.63 |
| `bands-256` | 0.02 | 1.11 | 2.68 |
| `bands-128` | 0.02 | 1.11 | 2.64 |
| `bands-256-no-direct-padding` | 0.02 | 1.11 | 2.66 |
| `bands-128-no-direct-padding` | 0.02 | 1.11 | 2.62 |
| `tiling-only` | 0.03 | 1.11 | 2.68 |

**Collect transformer**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.00 | 2.45 | 15.54 |
| `single-frame` | 0.00 | 2.45 | 15.54 |
| `residual-alias` | 0.02 | 2.45 | 15.54 |
| `direct-padding` | 0.00 | 2.45 | 15.55 |
| `bands-256` | 0.00 | 2.45 | 15.54 |
| `bands-128` | 0.00 | 2.45 | 15.54 |
| `bands-256-no-direct-padding` | 0.00 | 2.45 | 15.54 |
| `bands-128-no-direct-padding` | 0.00 | 2.45 | 15.55 |
| `tiling-only` | 0.00 | 2.45 | 15.55 |

**Cleanup transformer**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.91 | 2.45 | 15.54 |
| `single-frame` | 0.92 | 2.45 | 15.54 |
| `residual-alias` | 0.79 | 2.45 | 15.54 |
| `direct-padding` | 0.77 | 2.45 | 15.55 |
| `bands-256` | 0.76 | 2.45 | 15.54 |
| `bands-128` | 0.94 | 2.45 | 15.54 |
| `bands-256-no-direct-padding` | 0.78 | 2.45 | 15.54 |
| `bands-128-no-direct-padding` | 0.78 | 2.45 | 15.55 |
| `tiling-only` | 0.78 | 2.45 | 15.55 |

**Load vae**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.03 | 1.85 | 1.61 |
| `single-frame` | 0.03 | 1.85 | 1.61 |
| `residual-alias` | 0.03 | 1.85 | 1.61 |
| `direct-padding` | 0.03 | 1.85 | 1.61 |
| `bands-256` | 0.05 | 1.85 | 1.61 |
| `bands-128` | 0.03 | 1.85 | 1.61 |
| `bands-256-no-direct-padding` | 0.03 | 1.85 | 1.61 |
| `bands-128-no-direct-padding` | 0.03 | 1.85 | 1.61 |
| `tiling-only` | 0.03 | 1.85 | 1.61 |

**Attach vae**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.14 | 1.85 | 2.23 |
| `single-frame` | 0.14 | 1.85 | 2.23 |
| `residual-alias` | 0.14 | 1.85 | 2.23 |
| `direct-padding` | 0.14 | 1.85 | 2.23 |
| `bands-256` | 0.13 | 1.85 | 2.23 |
| `bands-128` | 0.13 | 1.85 | 2.28 |
| `bands-256-no-direct-padding` | 0.14 | 1.85 | 2.23 |
| `bands-128-no-direct-padding` | 0.14 | 1.85 | 2.23 |
| `tiling-only` | 0.12 | 1.85 | 2.23 |

**Collect vae**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.00 | 7.58 | 3.14 |
| `single-frame` | 0.00 | 5.59 | 3.10 |
| `residual-alias` | 0.00 | 5.16 | 3.10 |
| `direct-padding` | 0.00 | 5.16 | 3.10 |
| `bands-256` | 0.00 | 3.34 | 3.10 |
| `bands-128` | 0.00 | 3.34 | 3.10 |
| `bands-256-no-direct-padding` | 0.00 | 3.20 | 3.10 |
| `bands-128-no-direct-padding` | 0.00 | 3.20 | 3.10 |
| `tiling-only` | 0.00 | 2.18 | 3.11 |

**Cleanup vae**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.26 | 7.58 | 3.14 |
| `single-frame` | 0.20 | 5.59 | 3.10 |
| `residual-alias` | 0.20 | 5.16 | 3.10 |
| `direct-padding` | 0.22 | 5.16 | 3.10 |
| `bands-256` | 0.21 | 3.34 | 3.10 |
| `bands-128` | 0.25 | 3.34 | 3.10 |
| `bands-256-no-direct-padding` | 0.19 | 3.20 | 3.10 |
| `bands-128-no-direct-padding` | 0.26 | 3.20 | 3.10 |
| `tiling-only` | 0.20 | 2.18 | 3.11 |

**Save image**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.11 | 7.26 | 1.83 |
| `single-frame` | 0.09 | 5.26 | 1.79 |
| `residual-alias` | 0.09 | 4.84 | 1.79 |
| `direct-padding` | 0.11 | 4.83 | 1.79 |
| `bands-256` | 0.11 | 3.01 | 1.78 |
| `bands-128` | 0.11 | 3.01 | 1.78 |
| `bands-256-no-direct-padding` | 0.09 | 2.87 | 1.78 |
| `bands-128-no-direct-padding` | 0.09 | 2.87 | 1.79 |
| `tiling-only` | 0.10 | 1.85 | 1.80 |

</details>
