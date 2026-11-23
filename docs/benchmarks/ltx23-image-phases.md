# LTX 2.3 image phase measurements

[Results and recommendations](ltx23-image.md) · [Measurement details](ltx23-image-details.md) · [Measurement glossary](glossary.md)

Each phase uses presets or attention methods as rows. Latency is wall time; GPU peaks are sampled absolute driver usage and RAM peaks are process RSS, in GiB. Values come from the same selected runs as the main results. Full-run totals remain on that page.

**Full run** includes all tracked phases: component loading and setup, text
encoding, denoising, VAE decoding, cleanup, conversion and PNG saving. It excludes
calibration, command startup and downloads. Peaks cover the full tracked run.

**Denoising** reports the recorded sampling interval, including weight transfers
and scheduler work, rather than an isolated transformer-forward kernel timer.
The Windows Leaf, linear and VMM Fast runtime rows monitor denoising independently inside
the generation aggregate, excluding connector setup and transformer cleanup.
Other observations with only a call timer retain N/A for separate memory peaks.

**VAE decode** excludes VAE loading, attachment, cleanup and PNG saving. Conversion
performed inside the recorded decode call remains included. Where only a
decode-call timer exists, latency is shown and its separate memory peaks remain
N/A. No phase is estimated by subtraction and
no aggregate or full-run peak is substituted for a narrower phase peak.

Average divides denoise wall latency by the recipe steps, retaining the same
warmup/readback scope. Detailed setup and output phases are available in the
linked phase measurements; nested intervals must not be added to their aggregates.


Text encoder aggregates include loading, attachment where applicable, encoding and cleanup. Their latency sums disjoint recorded phases and their memory is the maximum recorded peak. Other setup/output phases are expandable below. N/A means unavailable, not zero.

<a id="windows-bf16"></a>

## Windows — BF16

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 21.78 | 4.79 | 25.90 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 19.76 | 4.79 | 25.90 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 33.11 | 4.85 | 45.82 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 10.39 | 6.66 | 25.89 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 22.32 | 4.79 | 25.90 |

### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 47.47 | 1.60 | 28.76 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 13.64 | 7.41 | 22.13 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 39.38 | 5.83 | 47.50 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 12.30 | 7.69 | 27.38 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 33.91 | 2.69 | 6.20 |

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 1.09 | 1.67 | 5.88 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 1.17 | 1.67 | 23.66 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 0.96 | 2.94 | 24.04 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 0.68 | 5.83 | 4.52 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 1.02 | 1.67 | 7.33 |

<details markdown="1">

<summary>Other setup and output phases</summary>
**Generate aggregate: connectors/setup/denoise/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 53.21 | 7.12 | 28.76 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 27.34 | 7.41 | 27.20 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 48.23 | 7.08 | 47.50 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 23.28 | 7.69 | 27.38 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 43.71 | 7.10 | 8.39 |

**VAE aggregate: load/decode/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 1.33 | N/A | N/A |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 1.37 | N/A | N/A |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 1.15 | N/A | N/A |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 1.28 | 5.83 | 4.55 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 1.23 | N/A | N/A |

**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 0.08 | 1.10 | 5.12 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 0.08 | 1.10 | 22.94 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 0.08 | 2.94 | 23.26 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 0.08 | 5.08 | 1.67 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 0.08 | 1.10 | 6.59 |

**Load VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 0.06 | 1.10 | 4.32 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 0.05 | 1.10 | 22.13 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 0.03 | 2.94 | 23.06 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 0.29 | 5.08 | 3.12 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 0.05 | 1.10 | 5.76 |

**Cleanup VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 0.18 | 1.67 | 5.88 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 0.16 | 1.67 | 23.70 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 0.16 | 2.94 | 24.03 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 0.31 | 5.83 | 4.55 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 0.15 | 1.67 | 7.35 |

</details>
## Transformer self-attention on VMM Fast

### Text encoder

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 10.39 | 6.66 | 25.89 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 10.24 | 6.66 | 25.89 |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 10.16 | 6.66 | 25.89 |

### Denoising

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 12.30 | 7.69 | 27.38 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 12.36 | 7.74 | 27.40 |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 12.35 | 7.69 | 27.39 |

### VAE decode

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 0.68 | 5.83 | 4.52 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 0.66 | 5.83 | 4.54 |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 0.67 | 5.83 | 4.57 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Generate aggregate: connectors/setup/denoise/cleanup**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 23.28 | 7.69 | 27.38 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 23.54 | 7.74 | 27.40 |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 23.27 | 7.69 | 27.39 |

**Load VAE**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 0.29 | 5.08 | 3.12 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 0.30 | 5.08 | 3.14 |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 0.29 | 5.08 | 3.13 |

**Cleanup VAE**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 0.31 | 5.83 | 4.55 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 0.29 | 5.83 | 4.56 |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 0.27 | 5.83 | 4.56 |

**Save image**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 0.08 | 5.08 | 1.67 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 0.06 | 5.08 | 1.68 |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 0.08 | 5.08 | 1.69 |

</details>

## `low_ram_safe` — pre-VAE purge

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`low_ram_safe`, no purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 33.11 | 4.85 | 45.82 |
| [`low_ram_safe`, pre-VAE purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 34.63 | 4.85 | 45.74 |

### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`low_ram_safe`, no purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 39.38 | 5.83 | 47.50 |
| [`low_ram_safe`, pre-VAE purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 39.37 | 5.83 | 47.51 |

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`low_ram_safe`, no purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 0.96 | 2.94 | 24.04 |
| [`low_ram_safe`, pre-VAE purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 1.60 | 2.94 | 24.04 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Generate aggregate: connectors/setup/denoise/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`low_ram_safe`, no purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 48.23 | 7.08 | 47.50 |
| [`low_ram_safe`, pre-VAE purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 48.35 | 7.11 | 47.51 |

**VAE aggregate: load/decode/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`low_ram_safe`, no purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 1.15 | 2.94 | 24.04 |
| [`low_ram_safe`, pre-VAE purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 1.79 | 2.94 | 24.06 |

**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`low_ram_safe`, no purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 0.08 | 2.94 | 23.26 |
| [`low_ram_safe`, pre-VAE purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 0.09 | 2.94 | 23.29 |

</details>

<a id="linux-bf16"></a>

## Linux — BF16

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | 11.56 | 4.22 | 26.19 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | 9.90 | 4.22 | 26.18 |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | 27.33 | 4.44 | 45.42 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | 10.61 | 6.20 | 26.09 |

### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | 30.85 | 1.19 | 28.92 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | 13.00 | 6.91 | 22.32 |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | 21.35 | 6.85 | 26.24 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | 11.86 | 5.76 | 27.57 |

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | 1.01 | 1.17 | 6.11 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | 0.96 | 1.21 | 23.88 |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | 1.04 | 1.13 | 3.38 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | 0.52 | 5.44 | 4.75 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Generate aggregate: connectors/setup/denoise/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | 34.55 | 6.60 | 28.93 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | 27.48 | 6.91 | 26.86 |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | 27.48 | 6.85 | 26.24 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | 19.06 | 6.58 | 27.57 |

**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | 0.20 | 0.74 | 4.56 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | 0.20 | 0.78 | 22.40 |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | 0.20 | 0.68 | 1.87 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | 0.20 | 4.69 | 1.90 |

**Load VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | 0.04 | 0.74 | 4.47 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | 0.04 | 0.78 | 22.31 |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | 0.09 | 0.68 | 1.78 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | 0.66 | 4.69 | 3.50 |

**Cleanup VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | 0.19 | 1.17 | 6.11 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | 0.19 | 1.21 | 23.95 |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | 0.19 | 1.13 | 3.43 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | 0.39 | 5.44 | 4.78 |

</details>

## Transformer self-attention on VMM Fast

### Text encoder

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 10.61 | 6.20 | 26.09 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 10.50 | 6.20 | 26.10 |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 10.39 | 6.19 | 26.10 |

### Denoising

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 11.86 | 5.76 | 27.57 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 12.59 | 5.76 | 27.62 |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 12.56 | 5.75 | 27.58 |

### VAE decode

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 0.52 | 5.44 | 4.75 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 0.44 | 5.44 | 4.76 |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 0.46 | 5.43 | 4.76 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Generate aggregate: connectors/setup/denoise/cleanup**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 19.06 | 6.58 | 27.57 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 17.99 | 6.58 | 27.62 |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 17.92 | 6.57 | 27.58 |

**Load VAE**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 0.66 | 4.69 | 3.50 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 0.63 | 4.69 | 3.43 |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 0.63 | 4.68 | 3.33 |

**Cleanup VAE**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 0.39 | 5.44 | 4.78 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 0.39 | 5.44 | 4.79 |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 0.38 | 5.43 | 4.79 |

**Save image**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 0.20 | 4.69 | 1.90 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 0.21 | 4.69 | 1.91 |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 0.21 | 4.68 | 1.91 |

</details>

<a id="wsl-bf16"></a>


## WSL — BF16

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 149.88 | 4.85 | 38.28 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 150.53 | 4.85 | 38.29 |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) | 160.23 | 4.84 | 37.57 |

### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 285.19 | 1.41 | 32.22 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 119.13 | 7.42 | 30.19 |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) | 83.87 | 7.73 | 30.47 |

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 4.43 | 1.55 | 6.51 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 4.40 | 1.55 | 6.54 |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) | 5.24 | 5.09 | 6.97 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Generate aggregate: connectors/setup/denoise/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 310.03 | 7.09 | 32.22 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 174.76 | 7.42 | 30.19 |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) | 120.53 | 7.73 | 30.47 |

**VAE aggregate: load/decode/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 4.74 | 1.55 | 6.51 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 4.72 | 1.55 | 6.55 |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) | 5.57 | 5.09 | 6.97 |

**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 0.23 | 1.10 | 5.75 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 0.22 | 1.10 | 5.79 |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) | 0.23 | 5.09 | 5.89 |

**Load VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 0.10 | 1.10 | 5.71 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 0.11 | 1.10 | 5.75 |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) | 0.10 | 5.09 | 5.85 |

**Cleanup VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 0.20 | 1.55 | 6.51 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 0.21 | 1.55 | 6.55 |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) | 0.22 | 5.09 | 6.97 |

</details>

## Transformer self-attention on VMM Fast

### Text encoder

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 160.23 | 4.84 | 37.57 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 150.88 | 4.84 | 37.78 |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 150.90 | 4.84 | 37.99 |

### Denoising

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 83.87 | 7.73 | 30.47 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 76.45 | 7.74 | 30.51 |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 79.86 | 7.76 | 30.47 |

### VAE decode

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 5.24 | 5.09 | 6.97 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 4.56 | 5.09 | 6.98 |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 5.34 | 5.09 | 6.97 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Generate aggregate: connectors/setup/denoise/cleanup**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 120.53 | 7.73 | 30.47 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 110.38 | 7.74 | 30.51 |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 113.47 | 7.76 | 30.47 |

**Load VAE**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 0.10 | 5.09 | 5.85 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 0.10 | 5.09 | 5.89 |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 0.10 | 5.09 | 5.85 |

**Cleanup VAE**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 0.22 | 5.09 | 6.97 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 0.27 | 5.09 | 6.98 |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 0.23 | 5.09 | 6.97 |

**Save image**

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 0.23 | 5.09 | 5.89 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 0.25 | 5.09 | 5.90 |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 0.24 | 5.09 | 5.89 |

</details>



<a id="windows-warm-process"></a>

## Windows `warm_process` — two generations in one process

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`warm_process`, two preparations and two generations](ltx23-image-details.md#windows-warm-process) | 22.39 | 4.79 | 25.99 |

### Denoising

| Generation | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| 1 | 13.41 | 7.45 | 22.13 |
| 2 | 13.30 | 7.41 | 22.13 |

Two generations share one process and only the final latent is decoded. Each
denoise interval has its own sampled memory peaks. Preparation and cache reuse
details remain on the results page.

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`warm_process`, two preparations and two generations](ltx23-image-details.md#windows-warm-process) | 0.31 | 2.80 | 22.17 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Generate aggregate: connectors/setup/denoise/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`warm_process`, two preparations and two generations](ltx23-image-details.md#windows-warm-process) | 45.29 | 7.45 | 27.98 |

**VAE aggregate: load/decode/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`warm_process`, two preparations and two generations](ltx23-image-details.md#windows-warm-process) | 1.61 | 2.80 | 23.45 |

**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`warm_process`, two preparations and two generations](ltx23-image-details.md#windows-warm-process) | 0.08 | 1.10 | 22.17 |

</details>

<a id="windows-sdnq-int4"></a>

## Windows — saved SDNQ INT4

### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](ltx23-image-details.md#windows-sdnq-int4) (reference) | 21.86 | 4.79 | 25.99 |
| [SDNQ INT4 DDO](ltx23-image-details.md#windows-sdnq-int4) | 8.10 | 4.79 | 25.95 |

### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](ltx23-image-details.md#windows-sdnq-int4) (reference) | 23.05 | 2.03 | 11.86 |
| [SDNQ INT4 DDO](ltx23-image-details.md#windows-sdnq-int4) | 10.79 | 7.07 | 6.59 |

### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](ltx23-image-details.md#windows-sdnq-int4) (reference) | 0.23 | 2.80 | 4.44 |
| [SDNQ INT4 DDO](ltx23-image-details.md#windows-sdnq-int4) | 0.22 | 2.80 | 6.63 |

<details markdown="1">

<summary>Other setup and output phases</summary>



**Generate aggregate: connectors/setup/denoise/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](ltx23-image-details.md#windows-sdnq-int4) (reference) | 29.14 | 7.10 | 11.86 |
| [SDNQ INT4 DDO](ltx23-image-details.md#windows-sdnq-int4) | 17.28 | 7.08 | 13.52 |

**VAE aggregate: load/decode/cleanup**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](ltx23-image-details.md#windows-sdnq-int4) (reference) | 1.05 | 2.80 | 5.43 |
| [SDNQ INT4 DDO](ltx23-image-details.md#windows-sdnq-int4) | 0.85 | 2.80 | 7.90 |

**Save image**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](ltx23-image-details.md#windows-sdnq-int4) (reference) | 0.08 | 1.10 | 4.46 |
| [SDNQ INT4 DDO](ltx23-image-details.md#windows-sdnq-int4) | 0.06 | 1.10 | 6.66 |

</details>
