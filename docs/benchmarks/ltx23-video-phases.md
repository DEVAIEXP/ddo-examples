# LTX 2.3 video phase measurements

[Results and recommendations](ltx23-video.md) · [Scenario details](ltx23-video-details.md) · [Measurement glossary](glossary.md)

Each table compares cases within one recorded phase. Cases are always rows;
columns are latency, sampled absolute GPU peak and process RSS peak, in GiB.
There are no comparison arrows in phase tables. Runtime, transformer attention,
precision and cumulative features are separate comparisons. All values refer
to the same selected observations as the results page; diagnostics and profile
calibration are excluded. Phase peaks must not be summed.

Denoising is sampling-loop latency, including prediction, weight transfers and
scheduler/latent updates. Legacy linear/Leaf records also include transformer
retirement; their published observations retain that scope. New instrumentation
records retirement separately. These values are not isolated transformer-forward
kernel timings. Windows Leaf/linear phase peaks come from the same selected
runtime reports as their published totals. The current Windows Leaf, One Shot Fast and Low RAM Safe observations
record decode and transformer retirement separately.

Video VAE decode covers only `decode_video`, excluding VAE loading/cleanup,
audio decoding and saving. Conversion performed inside that phase remains
included. N/A means that interval or metric was not recorded. Combined
decode-and-save values are retained only
under the explicitly labelled legacy aggregate. Other setup, decoder and output
phases are expandable below each comparison.

<a id="windows-bf16"></a>

## Windows — BF16, 8-second pressure workload

Official Leaf uses a post-reboot session; the other presets each use a separate observation from the VMM Fast SDPA control. Purge and identity differences are documented on the results page.

### Runtime comparison

#### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 20.47 | 3.28 | 25.75 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 19.67 | 3.28 | 25.75 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 19.96 | 3.28 | 25.80 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 10.95 | 5.02 | 26.05 |

#### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 363.33 | 8.00 | 41.34 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 335.02 | 8.00 | 50.35 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 419.30 | 8.00 | 42.36 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 321.35 | 8.00 | 33.08 |

#### Video VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 19.12 | 6.53 | 15.07 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 19.23 | 6.57 | 42.42 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 18.87 | 6.55 | 15.58 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 18.79 | 7.62 | 15.58 |

<details markdown="1">
<summary>Other setup phases</summary>

**Connectors**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 6.77 | 2.73 | 11.33 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 7.37 | 2.73 | 25.99 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 7.81 | 2.73 | 25.99 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 7.31 | 2.70 | 11.43 |

**Image conditioning encoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 2.11 | 1.44 | 5.89 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 2.21 | 1.44 | 13.96 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 2.31 | 1.44 | 13.96 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 2.20 | 2.70 | 8.09 |

**Audio conditioning preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.59 | 1.22 | 5.41 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 0.45 | 1.22 | 8.17 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.44 | 1.22 | 8.17 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.48 | 2.71 | 6.85 |

**Transformer load and preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 1.31 | 1.18 | 5.40 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 1.12 | 1.18 | 6.84 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 1.18 | 1.18 | 6.84 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 1.03 | 2.71 | 6.83 |

**Transformer attachment**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.11 | 1.11 | 5.40 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 9.97 | 1.13 | 41.53 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.51 | 1.53 | 7.22 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 6.96 | 2.71 | 8.49 |

**Transformer cleanup**

Leaf, One Shot Fast and Low RAM Safe record retirement separately from denoise.

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) | 0.19 | 8.00 | 41.34 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 0.20 | 8.00 | 50.26 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.20 | 8.00 | 42.36 |

</details>


<details markdown="1">
<summary>Other decoder and output phases</summary>

**Load video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.79 | 2.49 | 7.32 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 1.13 | 2.50 | 34.66 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 1.14 | 2.49 | 8.15 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 1.20 | 5.73 | 8.18 |

**Cleanup video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.31 | 6.53 | 10.13 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 0.29 | 6.58 | 37.54 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.25 | 6.55 | 10.97 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.36 | 6.43 | 9.10 |

**Load audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.14 | 1.19 | 10.21 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 0.15 | 1.17 | 37.59 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.15 | 1.17 | 11.03 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.16 | 5.15 | 9.17 |

**Audio VAE decode**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.17 | 1.39 | 10.13 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 0.18 | 1.39 | 37.54 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.16 | 1.39 | 10.97 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.17 | 5.19 | 9.11 |

**Cleanup audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.20 | 1.40 | 10.13 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 0.21 | 1.40 | 37.54 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.22 | 1.40 | 10.97 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.23 | 5.19 | 9.11 |

**Load vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.75 | 1.35 | 10.13 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 0.71 | 1.34 | 37.54 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.74 | 1.34 | 10.97 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.64 | 5.18 | 9.11 |

**Vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.77 | 1.58 | 6.35 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 0.67 | 1.58 | 33.76 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.72 | 1.58 | 7.19 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.84 | 5.19 | 7.26 |

**Cleanup vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 0.21 | 1.58 | 6.35 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 0.21 | 1.58 | 33.76 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 0.22 | 1.58 | 7.20 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.24 | 5.19 | 7.26 |

**Save video**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 2.03 | 1.11 | 7.53 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 2.09 | 1.11 | 34.94 |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 2.08 | 1.11 | 8.38 |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 1.98 | 5.07 | 8.44 |

**Decode and save aggregate**

This includes loading, decoding, cleanup and saving; it is not pure VAE decode.
Leaf and both linear presets record this aggregate without its own memory monitor;
their individual decoder and save peaks are reported above.

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 24.52 | N/A | N/A |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 24.89 | N/A | N/A |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 24.55 | N/A | N/A |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | N/A | N/A | N/A |

</details>


<a id="windows-sol-attention"></a>

Attention changes only the eligible transformer path; the same decoder scope is reported for all four mechanisms. Refer to the results page for matched versus contextual session comparisons. Operator diagnostics are outside these measurements.

### Transformer self-attention on VMM Fast

#### Text encoder

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 10.95 | 5.02 | 26.05 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 10.78 | 5.02 | 26.05 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 11.04 | 5.02 | 26.05 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 11.06 | 5.02 | 26.06 |

#### Denoising

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 321.35 | 8.00 | 33.08 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 273.46 | 8.00 | 33.03 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 274.68 | 8.00 | 33.13 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 211.65 | 8.00 | 33.06 |

#### Video VAE decode

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 18.79 | 7.62 | 15.58 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 18.94 | 7.62 | 15.84 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 18.95 | 7.62 | 15.88 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 18.66 | 7.62 | 16.03 |

<details markdown="1">
<summary>Other setup phases</summary>

**Connectors**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 7.31 | 2.70 | 11.43 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 7.12 | 2.70 | 11.34 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 7.43 | 2.70 | 11.38 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 7.32 | 2.70 | 11.35 |

**Image conditioning encoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 2.20 | 2.70 | 8.09 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 1.36 | 2.70 | 8.09 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 1.31 | 2.70 | 8.09 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 1.38 | 2.70 | 8.09 |

**Audio conditioning preparation**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.48 | 2.71 | 6.85 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 0.36 | 2.71 | 6.85 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 0.36 | 2.71 | 6.85 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 0.34 | 2.71 | 6.85 |

**Transformer load and preparation**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 1.03 | 2.71 | 6.83 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 0.98 | 2.71 | 6.84 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 0.97 | 2.71 | 6.83 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 0.97 | 2.71 | 6.83 |

**Transformer attachment**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 6.96 | 2.71 | 8.49 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 7.37 | 2.71 | 8.84 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 7.23 | 2.71 | 8.44 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 6.81 | 2.71 | 8.74 |

</details>

<details markdown="1">
<summary>Other decoder and output phases</summary>

**Load video VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 1.20 | 5.73 | 8.18 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 1.19 | 5.75 | 8.19 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 1.16 | 5.77 | 8.24 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 1.24 | 5.71 | 8.17 |

**Cleanup video VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.36 | 6.43 | 9.10 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 0.36 | 6.43 | 9.09 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 0.40 | 6.43 | 9.13 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 0.34 | 6.43 | 11.04 |

**Load audio VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.16 | 5.15 | 9.17 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 0.15 | 5.15 | 9.15 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 0.20 | 5.14 | 9.17 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 0.16 | 5.15 | 11.10 |

**Audio VAE decode**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.17 | 5.19 | 9.11 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 0.16 | 5.19 | 9.09 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 0.19 | 5.19 | 9.13 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 0.17 | 5.19 | 11.04 |

**Cleanup audio VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.23 | 5.19 | 9.11 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 0.23 | 5.19 | 9.09 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 0.23 | 5.19 | 9.13 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 0.23 | 5.19 | 11.04 |

**Load vocoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.64 | 5.18 | 9.11 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 0.62 | 5.19 | 9.09 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 0.63 | 5.19 | 9.13 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 0.73 | 5.19 | 11.04 |

**Vocoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.84 | 5.19 | 7.26 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 0.77 | 5.19 | 7.24 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 0.83 | 5.19 | 7.28 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 0.77 | 5.19 | 7.25 |

**Cleanup vocoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 0.24 | 5.19 | 7.26 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 0.24 | 5.19 | 7.24 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 0.29 | 5.19 | 7.28 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 0.29 | 5.19 | 7.25 |

**Save video**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 1.98 | 5.07 | 8.44 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 1.97 | 5.07 | 8.42 |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 1.97 | 5.07 | 8.46 |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 2.05 | 5.07 | 8.44 |

</details>

<a id="linux-bf16"></a>

## Linux — BF16, 8-second pressure workload

The runtime rows preserve their recorded allocator settings and host-cache release events. Those events are included in Full run, outside pure decode.

### Runtime comparison

#### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 10.86 | 2.87 | 26.03 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 9.81 | 2.86 | 26.03 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 9.91 | 2.86 | 26.08 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 10.55 | 4.63 | 26.26 |

#### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 290.26 | 7.62 | 37.90 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 274.65 | 7.41 | 29.28 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 296.78 | 7.63 | 42.03 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 274.26 | 7.69 | 33.07 |

#### Video VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 26.32 | 4.89 | 10.68 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 19.07 | 4.88 | 8.07 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 21.86 | 4.87 | 9.77 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 21.40 | 5.29 | 14.44 |

<details markdown="1">
<summary>Other setup phases</summary>

**Connectors**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 4.48 | 2.33 | 11.44 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 4.76 | 2.32 | 26.25 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 4.87 | 2.32 | 26.23 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 5.12 | 2.27 | 11.23 |

**Image conditioning encoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 1.20 | 0.86 | 5.80 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 1.97 | 0.85 | 13.86 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 1.72 | 0.85 | 13.85 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 1.76 | 2.28 | 7.99 |

**Audio conditioning preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.38 | 0.81 | 5.35 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 0.43 | 0.80 | 8.04 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.35 | 0.80 | 7.99 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.39 | 2.29 | 6.77 |

**Transformer load and preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 1.19 | 0.79 | 5.46 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 1.31 | 0.76 | 6.88 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 1.07 | 0.76 | 6.87 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 1.03 | 2.29 | 6.88 |

**Transformer attachment**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.09 | 0.71 | 5.46 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 26.22 | 0.79 | 43.40 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.29 | 1.11 | 7.23 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 2.94 | 2.29 | 14.89 |

</details>

<details markdown="1">
<summary>Other decoder and output phases</summary>

**Load video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.58 | 1.99 | 4.07 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 0.82 | 2.02 | 1.82 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.75 | 2.03 | 3.36 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.85 | 4.76 | 8.04 |

**Cleanup video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.35 | 4.89 | 3.40 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 0.35 | 4.88 | 1.21 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.35 | 4.87 | 2.65 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.37 | 5.29 | 7.46 |

**Load audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.07 | 0.70 | 3.40 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 0.11 | 0.76 | 1.30 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.11 | 0.75 | 2.74 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.13 | 4.73 | 7.52 |

**Audio VAE decode**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.07 | 0.80 | 3.40 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 0.05 | 0.78 | 1.21 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.03 | 0.77 | 2.65 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.05 | 4.77 | 7.46 |

**Cleanup audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.23 | 0.94 | 3.41 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 0.23 | 0.92 | 1.22 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.22 | 0.91 | 2.65 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.30 | 4.76 | 7.46 |

**Load vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.29 | 0.84 | 3.54 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 0.30 | 0.80 | 1.34 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.30 | 0.79 | 2.78 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.33 | 4.66 | 7.70 |

**Vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.53 | 1.14 | 3.52 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 0.54 | 1.13 | 1.34 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.50 | 1.11 | 2.77 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.59 | 4.82 | 7.56 |

**Cleanup vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 0.23 | 1.14 | 3.52 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 0.23 | 1.13 | 1.34 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 0.23 | 1.11 | 2.77 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.29 | 4.82 | 7.56 |

**Save video**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 1.83 | 0.70 | 4.71 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 1.87 | 0.68 | 2.53 |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 1.79 | 0.67 | 3.95 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 1.90 | 4.66 | 8.74 |

</details>

<a id="linux-cudnn-attention"></a>
<a id="linux-flash-attention"></a>
<a id="linux-sol-attention"></a>

Attention changes only the eligible transformer path; the same decoder scope is reported for all four mechanisms. Refer to the results page for matched versus contextual session comparisons. Operator diagnostics are outside these measurements.


### Transformer self-attention on VMM Fast

#### Text encoder

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 10.55 | 4.63 | 26.26 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 10.59 | 4.66 | 26.27 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 10.72 | 4.63 | 26.27 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 10.65 | 4.64 | 26.27 |

#### Denoising

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 274.26 | 7.69 | 33.07 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 275.15 | 7.70 | 33.08 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 268.28 | 7.69 | 33.08 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 214.54 | 7.69 | 33.09 |

#### Video VAE decode

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 21.40 | 5.29 | 14.44 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 22.74 | 5.02 | 14.44 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 19.05 | 4.97 | 14.24 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 19.40 | 5.68 | 13.96 |

<details markdown="1">
<summary>Other setup phases</summary>

**Connectors**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 5.12 | 2.27 | 11.23 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 4.67 | 2.32 | 11.32 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 4.60 | 2.25 | 11.45 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 4.63 | 2.28 | 11.42 |

**Image conditioning encoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 1.76 | 2.28 | 7.99 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 1.65 | 2.32 | 7.99 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 1.61 | 2.26 | 7.99 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 1.63 | 2.28 | 8.00 |

**Audio conditioning preparation**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 0.39 | 2.29 | 6.77 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.29 | 2.32 | 6.78 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.28 | 2.27 | 6.78 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.28 | 2.31 | 6.78 |

**Transformer load and preparation**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 1.03 | 2.29 | 6.88 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.87 | 2.35 | 6.89 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.75 | 2.27 | 6.89 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.81 | 2.31 | 6.90 |

**Transformer attachment**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 2.94 | 2.29 | 14.89 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 3.18 | 2.36 | 15.82 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 2.57 | 2.27 | 13.81 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 2.60 | 2.32 | 14.24 |

</details>

<details markdown="1">
<summary>Other decoder and output phases</summary>

**Load video VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 0.85 | 4.76 | 8.04 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.26 | 5.02 | 8.18 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.25 | 4.97 | 7.41 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.32 | 4.89 | 7.92 |

**Cleanup video VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 0.37 | 5.29 | 7.46 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.24 | 5.02 | 7.47 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.24 | 4.97 | 7.47 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.27 | 5.68 | 7.48 |

**Load audio VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 0.13 | 4.73 | 7.52 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.03 | 4.68 | 7.47 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.03 | 4.63 | 7.47 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.03 | 4.68 | 7.48 |

**Audio VAE decode**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 0.05 | 4.77 | 7.46 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.03 | 4.68 | 7.47 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.03 | 4.63 | 7.47 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.04 | 4.78 | 7.48 |

**Cleanup audio VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 0.30 | 4.76 | 7.46 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.22 | 4.68 | 7.47 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.22 | 4.63 | 7.47 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.24 | 4.77 | 7.48 |

**Load vocoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 0.33 | 4.66 | 7.70 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.19 | 4.68 | 7.48 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.18 | 4.63 | 7.48 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.18 | 4.68 | 7.49 |

**Vocoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 0.59 | 4.82 | 7.56 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.58 | 4.84 | 7.58 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.55 | 4.81 | 7.57 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.56 | 4.83 | 7.59 |

**Cleanup vocoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 0.29 | 4.82 | 7.56 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 0.31 | 4.84 | 7.58 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 0.25 | 4.81 | 7.57 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 0.26 | 4.83 | 7.59 |

**Save video**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 1.90 | 4.66 | 8.74 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 1.79 | 4.65 | 8.76 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 1.74 | 4.65 | 8.75 |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 1.79 | 4.68 | 8.77 |

</details>

<a id="wsl-bf16"></a>

## WSL — BF16, 8-second pressure workload

Each runtime row uses one complete execution, including its independently
measured decoder phases. VMM Fast SDPA and Sol share a matched pair; the
other attention measurements have separate session identities.

### Runtime comparison

#### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 150.04 | 2.98 | 38.16 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 143.87 | 2.98 | 38.39 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 137.94 | 2.96 | 38.27 |

#### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 641.02 | 7.75 | 38.42 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 1725.82 | 8.00 | 38.38 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 461.95 | 8.00 | 30.92 |

#### Video VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 19.52 | 5.31 | 21.15 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 19.49 | 5.31 | 13.17 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 20.95 | 7.77 | 13.44 |

<details markdown="1">
<summary>Other setup phases</summary>

**Connectors**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 42.04 | 2.73 | 14.66 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 39.39 | 2.73 | 38.20 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 42.19 | 2.96 | 38.14 |

**Image conditioning encoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 6.49 | 1.26 | 6.38 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 6.99 | 1.26 | 16.67 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 5.68 | 2.96 | 6.41 |

**Audio conditioning preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 0.92 | 1.19 | 5.90 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 1.07 | 1.19 | 6.66 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 0.84 | 2.97 | 5.92 |

**Transformer load and preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 2.58 | 1.19 | 5.98 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 2.46 | 1.19 | 6.02 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 2.47 | 2.97 | 5.99 |

**Transformer attachment**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 0.07 | 1.11 | 5.97 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 2.51 | 1.53 | 6.36 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 10.96 | 2.98 | 8.70 |

</details>

<details markdown="1">
<summary>Other decoder and output phases</summary>

**Load video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 7.10 | 2.45 | 14.84 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 7.63 | 2.45 | 6.84 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 7.68 | 5.90 | 7.23 |

**Cleanup video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 0.40 | 5.31 | 13.51 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 0.32 | 5.31 | 5.51 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 0.42 | 6.49 | 5.89 |

**Load audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 0.62 | 1.21 | 13.61 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 0.63 | 1.21 | 5.60 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 0.71 | 5.19 | 5.98 |

**Audio VAE decode**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 0.20 | 1.29 | 13.51 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 0.24 | 1.35 | 5.52 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 0.32 | 5.21 | 5.89 |

**Cleanup audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 0.36 | 1.35 | 13.52 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 0.29 | 1.35 | 5.52 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 0.31 | 5.21 | 5.89 |

**Load vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 1.64 | 1.36 | 13.76 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 1.61 | 1.33 | 5.75 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 1.58 | 5.20 | 6.13 |

**Vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 0.82 | 1.55 | 13.54 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 0.79 | 1.55 | 5.54 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 0.82 | 5.21 | 5.90 |

**Cleanup vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 0.24 | 1.55 | 13.54 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 0.25 | 1.55 | 5.54 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 0.27 | 5.21 | 5.90 |

**Save video**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 2.54 | 1.11 | 13.57 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 2.65 | 1.11 | 6.05 |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 2.52 | 5.08 | 6.41 |

</details>


Attention changes only the eligible transformer path; the same decoder scope is reported for all four mechanisms. Refer to the results page for matched versus contextual session comparisons. Operator diagnostics are outside these measurements.


### Transformer self-attention on VMM Fast

#### Text encoder

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 137.94 | 2.96 | 38.27 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 139.34 | 2.96 | 38.30 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 137.31 | 2.96 | 38.25 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 141.57 | 2.96 | 37.53 |

#### Denoising

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 461.95 | 8.00 | 30.92 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 405.87 | 8.00 | 30.95 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 400.50 | 8.00 | 30.94 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 359.22 | 8.00 | 31.01 |

#### Video VAE decode

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 20.95 | 7.77 | 13.44 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 19.81 | 7.70 | 13.66 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 20.23 | 7.65 | 13.44 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 20.22 | 7.64 | 13.62 |

<details markdown="1">
<summary>Other setup phases</summary>

**Connectors**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 42.19 | 2.96 | 38.14 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 39.23 | 2.96 | 38.14 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 39.56 | 2.96 | 38.05 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 42.13 | 2.96 | 37.39 |

**Image conditioning encoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 5.68 | 2.96 | 6.41 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 5.04 | 2.96 | 6.43 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 5.36 | 2.96 | 6.43 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 4.82 | 2.96 | 6.50 |

**Audio conditioning preparation**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 0.84 | 2.97 | 5.92 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 0.85 | 2.97 | 5.94 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 0.79 | 2.97 | 5.94 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 1.00 | 2.97 | 6.01 |

**Transformer load and preparation**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 2.47 | 2.97 | 5.99 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 2.27 | 2.97 | 6.00 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 2.29 | 2.97 | 6.01 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 2.26 | 2.97 | 6.08 |

**Transformer attachment**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 10.96 | 2.98 | 8.70 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 10.58 | 2.98 | 8.77 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 10.93 | 2.98 | 8.69 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 12.09 | 2.98 | 10.00 |

</details>

<details markdown="1">
<summary>Other decoder and output phases</summary>

**Load video VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 7.68 | 5.90 | 7.23 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 6.88 | 5.90 | 7.27 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 7.88 | 5.90 | 7.25 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 7.52 | 5.90 | 7.34 |

**Cleanup video VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 0.42 | 6.49 | 5.89 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 0.38 | 6.52 | 5.92 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 0.39 | 6.47 | 5.91 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 0.44 | 6.45 | 5.99 |

**Load audio VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 0.71 | 5.19 | 5.98 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 0.60 | 5.19 | 6.01 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 0.63 | 5.20 | 6.01 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 0.61 | 5.20 | 6.07 |

**Audio VAE decode**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 0.32 | 5.21 | 5.89 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 0.21 | 5.23 | 5.92 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 0.21 | 5.21 | 5.91 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 0.25 | 5.24 | 5.99 |

**Cleanup audio VAE**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 0.31 | 5.21 | 5.89 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 0.23 | 5.18 | 5.92 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 0.23 | 5.18 | 5.91 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 0.33 | 5.18 | 5.99 |

**Load vocoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 1.58 | 5.20 | 6.13 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 1.43 | 5.10 | 6.16 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 1.55 | 5.09 | 6.15 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 1.62 | 5.10 | 6.22 |

**Vocoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 0.82 | 5.21 | 5.90 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 0.75 | 5.28 | 5.93 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 0.78 | 5.28 | 5.92 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 0.86 | 5.28 | 6.00 |

**Cleanup vocoder**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 0.27 | 5.21 | 5.90 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 0.27 | 5.28 | 5.93 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 0.32 | 5.28 | 5.92 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 0.27 | 5.28 | 6.00 |

**Save video**

| Attention backend | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 2.52 | 5.08 | 6.41 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 2.56 | 5.10 | 6.44 |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 2.47 | 5.10 | 6.43 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 2.49 | 5.10 | 6.50 |

</details>

<a id="saved-fp8"></a>

## FP8 weight-only — 8-second workload

These separately profiled configurations retain the precision and environment caveats on the results page.

### Windows

#### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 10.95 | 5.02 | 26.05 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 10.61 | 4.95 | 26.05 |

#### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 321.35 | 8.00 | 33.08 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 330.27 | 8.00 | 28.82 |

#### Video VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 18.79 | 7.62 | 15.58 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 18.80 | 7.08 | 16.29 |

<details markdown="1">
<summary>Other setup phases</summary>

**Connectors**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 7.31 | 2.70 | 11.43 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 7.07 | 2.70 | 11.42 |

**Image conditioning encoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 2.20 | 2.70 | 8.09 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 2.14 | 2.70 | 8.09 |

**Audio conditioning preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 0.48 | 2.71 | 6.85 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 0.39 | 2.71 | 6.85 |

**Transformer load and preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 1.03 | 2.71 | 6.83 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 30.12 | 2.71 | 26.85 |

**Transformer attachment**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 6.96 | 2.71 | 8.49 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 10.77 | 2.72 | 28.41 |

</details>

<details markdown="1">
<summary>Other decoder and output phases</summary>

**Load video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 1.20 | 5.73 | 8.18 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 1.12 | 6.64 | 10.58 |

**Cleanup video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 0.36 | 6.43 | 9.10 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 0.31 | 6.45 | 13.14 |

**Load audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 0.16 | 5.15 | 9.17 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 0.16 | 5.14 | 13.18 |

**Audio VAE decode**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 0.17 | 5.19 | 9.11 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 0.15 | 5.22 | 13.12 |

**Cleanup audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 0.23 | 5.19 | 9.11 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 0.22 | 5.19 | 13.12 |

**Load vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 0.64 | 5.18 | 9.11 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 0.53 | 5.10 | 13.33 |

**Vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 0.84 | 5.19 | 7.26 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 0.83 | 5.24 | 13.22 |

**Cleanup vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 0.24 | 5.19 | 7.26 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 0.24 | 5.24 | 13.22 |

**Save video**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) / BF16 | 1.98 | 5.07 | 8.44 |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 1.98 | 5.08 | 14.40 |

</details>

### Linux

#### Text encoder

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 10.55 | 4.63 | 26.26 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 10.68 | 4.70 | 26.27 |

#### Denoising

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 274.26 | 7.69 | 33.07 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 289.78 | 7.68 | 27.27 |

#### Video VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 21.40 | 5.29 | 14.44 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 18.29 | 6.49 | 15.56 |

<details markdown="1">
<summary>Other setup phases</summary>

**Connectors**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 5.12 | 2.27 | 11.23 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 4.58 | 2.34 | 11.48 |

**Image conditioning encoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 1.76 | 2.28 | 7.99 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 1.60 | 2.34 | 7.99 |

**Audio conditioning preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 0.39 | 2.29 | 6.77 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 0.29 | 2.35 | 6.78 |

**Transformer load and preparation**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 1.03 | 2.29 | 6.88 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 17.31 | 2.35 | 25.61 |

**Transformer attachment**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 2.94 | 2.29 | 14.89 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 7.80 | 2.36 | 27.16 |

</details>

<details markdown="1">
<summary>Other decoder and output phases</summary>

**Load video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 0.85 | 4.76 | 8.04 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 0.22 | 4.79 | 8.58 |

**Cleanup video VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 0.37 | 5.29 | 7.46 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 0.28 | 6.05 | 7.98 |

**Load audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 0.13 | 4.73 | 7.52 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 0.07 | 4.69 | 7.98 |

**Audio VAE decode**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 0.05 | 4.77 | 7.46 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 0.04 | 4.79 | 7.98 |

**Cleanup audio VAE**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 0.30 | 4.76 | 7.46 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 0.23 | 4.79 | 7.98 |

**Load vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 0.33 | 4.66 | 7.70 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 0.32 | 4.70 | 8.22 |

**Vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 0.59 | 4.82 | 7.56 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 0.56 | 4.78 | 7.99 |

**Cleanup vocoder**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 0.29 | 4.82 | 7.56 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 0.25 | 4.78 | 7.99 |

**Save video**

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`vmm_fast`](ltx23-video-details.md#linux-native) / BF16 | 1.90 | 4.66 | 8.74 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 1.63 | 4.62 | 8.78 |

</details>

<a id="windows-cumulative-model-features"></a>

## Windows — cumulative model features

Each case adds one feature to its predecessor. These observations have their own control and do not replace runtime or attention results. The configured-budget rounding limitation, identical observed weight residency and output-quality caveat remain documented in the [feature results and recommendations](model-features.md#windows-observations).

### Cumulative feature comparison

#### Text encoder

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 10.71 | 5.02 | 26.05 |
| `rope-reuse` | 11.25 | 5.02 | 26.05 |
| `compact-timesteps` | 11.55 | 4.99 | 26.05 |
| `inplace-residuals` | 11.41 | 4.95 | 26.05 |
| `fused-rms-modulation` | 12.35 | 4.95 | 26.05 |
| `fused-rotary-pair` | 12.85 | 5.02 | 26.05 |
| `phase-modulation` | 12.57 | 5.02 | 26.05 |
| `frame-modulation` | 12.49 | 5.02 | 26.06 |
| `feed-forward-1024` | 12.69 | 5.02 | 26.05 |
| `normalization-1024` | 12.55 | 5.02 | 26.05 |

#### Denoising

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 319.52 | 8.00 | 33.04 |
| `rope-reuse` | 333.47 | 8.00 | 33.29 |
| `compact-timesteps` | 315.94 | 6.46 | 33.19 |
| `inplace-residuals` | 318.21 | 5.62 | 33.19 |
| `fused-rms-modulation` | 321.15 | 5.62 | 33.19 |
| `fused-rotary-pair` | 315.65 | 5.62 | 33.19 |
| `phase-modulation` | 320.28 | 4.61 | 33.19 |
| `frame-modulation` | 313.21 | 4.10 | 33.18 |
| `feed-forward-1024` | 324.72 | 3.60 | 33.19 |
| `normalization-1024` | 320.47 | 3.60 | 33.18 |

#### Video VAE decode

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 18.93 | 7.62 | 16.18 |
| `rope-reuse` | 18.87 | 7.51 | 16.20 |
| `compact-timesteps` | 18.93 | 5.82 | 16.07 |
| `inplace-residuals` | 19.18 | 5.24 | 16.12 |
| `fused-rms-modulation` | 19.47 | 5.24 | 15.67 |
| `fused-rotary-pair` | 19.65 | 5.24 | 16.12 |
| `phase-modulation` | 19.74 | 5.24 | 16.02 |
| `frame-modulation` | 19.76 | 5.24 | 15.85 |
| `feed-forward-1024` | 19.76 | 5.24 | 15.71 |
| `normalization-1024` | 19.78 | 5.24 | 16.11 |

<details markdown="1">
<summary>Other setup phases</summary>

**Connectors**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 7.17 | 2.70 | 11.41 |
| `rope-reuse` | 7.26 | 2.70 | 11.35 |
| `compact-timesteps` | 7.24 | 2.70 | 11.44 |
| `inplace-residuals` | 7.21 | 2.70 | 11.41 |
| `fused-rms-modulation` | 7.36 | 2.70 | 11.35 |
| `fused-rotary-pair` | 7.66 | 2.70 | 11.38 |
| `phase-modulation` | 7.85 | 2.70 | 11.39 |
| `frame-modulation` | 7.62 | 2.70 | 11.40 |
| `feed-forward-1024` | 7.35 | 2.70 | 11.44 |
| `normalization-1024` | 7.42 | 2.70 | 11.40 |

**Image conditioning encoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 2.03 | 2.70 | 8.09 |
| `rope-reuse` | 2.12 | 2.70 | 8.09 |
| `compact-timesteps` | 2.09 | 2.70 | 8.09 |
| `inplace-residuals` | 2.16 | 2.70 | 8.09 |
| `fused-rms-modulation` | 2.10 | 2.70 | 8.09 |
| `fused-rotary-pair` | 2.18 | 2.70 | 8.09 |
| `phase-modulation` | 2.24 | 2.70 | 8.09 |
| `frame-modulation` | 2.20 | 2.70 | 8.09 |
| `feed-forward-1024` | 2.12 | 2.70 | 8.09 |
| `normalization-1024` | 2.19 | 2.70 | 8.09 |

**Audio conditioning preparation**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.41 | 2.71 | 6.85 |
| `rope-reuse` | 0.44 | 2.71 | 6.85 |
| `compact-timesteps` | 0.40 | 2.71 | 6.85 |
| `inplace-residuals` | 0.40 | 2.71 | 6.85 |
| `fused-rms-modulation` | 0.39 | 2.71 | 6.85 |
| `fused-rotary-pair` | 0.40 | 2.71 | 6.85 |
| `phase-modulation` | 0.45 | 2.71 | 6.85 |
| `frame-modulation` | 0.44 | 2.71 | 6.85 |
| `feed-forward-1024` | 0.43 | 2.71 | 6.85 |
| `normalization-1024` | 0.46 | 2.71 | 6.85 |

**Transformer load and preparation**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.06 | 2.71 | 6.83 |
| `rope-reuse` | 1.26 | 2.71 | 7.01 |
| `compact-timesteps` | 1.28 | 2.71 | 7.01 |
| `inplace-residuals` | 1.17 | 2.71 | 7.01 |
| `fused-rms-modulation` | 1.22 | 2.71 | 7.01 |
| `fused-rotary-pair` | 1.27 | 2.71 | 7.01 |
| `phase-modulation` | 1.38 | 2.71 | 7.01 |
| `frame-modulation` | 1.39 | 2.71 | 7.01 |
| `feed-forward-1024` | 1.34 | 2.71 | 7.01 |
| `normalization-1024` | 1.46 | 2.71 | 7.01 |

**Transformer attachment**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 7.46 | 2.71 | 8.78 |
| `rope-reuse` | 7.60 | 2.71 | 8.73 |
| `compact-timesteps` | 7.47 | 2.71 | 9.17 |
| `inplace-residuals` | 7.07 | 2.71 | 8.89 |
| `fused-rms-modulation` | 8.11 | 2.71 | 9.05 |
| `fused-rotary-pair` | 7.22 | 2.71 | 8.88 |
| `phase-modulation` | 7.71 | 2.71 | 8.74 |
| `frame-modulation` | 7.29 | 2.71 | 9.08 |
| `feed-forward-1024` | 7.80 | 2.71 | 9.19 |
| `normalization-1024` | 7.36 | 2.71 | 9.04 |

</details>

<details markdown="1">
<summary>Other decoder and output phases</summary>

**Load video VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.19 | 5.74 | 8.18 |
| `rope-reuse` | 1.27 | 5.75 | 8.46 |
| `compact-timesteps` | 1.24 | 5.42 | 8.32 |
| `inplace-residuals` | 1.16 | 5.02 | 8.32 |
| `fused-rms-modulation` | 1.22 | 5.02 | 8.29 |
| `fused-rotary-pair` | 1.19 | 5.02 | 8.28 |
| `phase-modulation` | 1.18 | 4.00 | 8.31 |
| `frame-modulation` | 1.15 | 3.50 | 8.33 |
| `feed-forward-1024` | 1.23 | 2.99 | 8.37 |
| `normalization-1024` | 1.17 | 2.99 | 8.32 |

**Cleanup video VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.38 | 6.43 | 9.09 |
| `rope-reuse` | 0.31 | 6.42 | 11.10 |
| `compact-timesteps` | 0.31 | 5.82 | 9.08 |
| `inplace-residuals` | 0.31 | 5.24 | 9.07 |
| `fused-rms-modulation` | 0.30 | 5.24 | 9.07 |
| `fused-rotary-pair` | 0.30 | 5.24 | 9.07 |
| `phase-modulation` | 0.33 | 5.24 | 9.07 |
| `frame-modulation` | 0.30 | 5.24 | 9.07 |
| `feed-forward-1024` | 0.33 | 5.24 | 9.07 |
| `normalization-1024` | 0.31 | 5.24 | 9.06 |

**Load audio VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.19 | 5.15 | 9.15 |
| `rope-reuse` | 0.15 | 5.08 | 11.16 |
| `compact-timesteps` | 0.17 | 5.07 | 9.13 |
| `inplace-residuals` | 0.21 | 5.13 | 9.12 |
| `fused-rms-modulation` | 0.17 | 5.14 | 9.13 |
| `fused-rotary-pair` | 0.23 | 5.13 | 9.13 |
| `phase-modulation` | 0.17 | 5.13 | 9.12 |
| `frame-modulation` | 0.16 | 5.14 | 9.13 |
| `feed-forward-1024` | 0.18 | 5.14 | 9.13 |
| `normalization-1024` | 0.17 | 5.14 | 9.13 |

**Audio VAE decode**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.16 | 5.19 | 9.09 |
| `rope-reuse` | 0.15 | 5.08 | 11.10 |
| `compact-timesteps` | 0.18 | 5.07 | 9.08 |
| `inplace-residuals` | 0.15 | 5.21 | 9.07 |
| `fused-rms-modulation` | 0.18 | 5.21 | 9.07 |
| `fused-rotary-pair` | 0.19 | 5.21 | 9.07 |
| `phase-modulation` | 0.20 | 5.21 | 9.07 |
| `frame-modulation` | 0.18 | 5.21 | 9.07 |
| `feed-forward-1024` | 0.21 | 5.21 | 9.07 |
| `normalization-1024` | 0.18 | 5.21 | 9.07 |

**Cleanup audio VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.24 | 5.19 | 9.09 |
| `rope-reuse` | 0.27 | 5.08 | 11.10 |
| `compact-timesteps` | 0.23 | 5.07 | 9.08 |
| `inplace-residuals` | 0.27 | 5.19 | 9.07 |
| `fused-rms-modulation` | 0.28 | 5.19 | 9.07 |
| `fused-rotary-pair` | 0.37 | 5.19 | 9.07 |
| `phase-modulation` | 0.30 | 5.19 | 9.07 |
| `frame-modulation` | 0.33 | 5.19 | 9.07 |
| `feed-forward-1024` | 0.29 | 5.19 | 9.07 |
| `normalization-1024` | 0.28 | 5.19 | 9.07 |

**Load vocoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.66 | 5.19 | 9.09 |
| `rope-reuse` | 0.72 | 5.08 | 11.10 |
| `compact-timesteps` | 0.65 | 5.07 | 9.08 |
| `inplace-residuals` | 0.72 | 5.09 | 9.07 |
| `fused-rms-modulation` | 0.65 | 5.10 | 9.07 |
| `fused-rotary-pair` | 0.77 | 5.10 | 9.07 |
| `phase-modulation` | 0.65 | 5.10 | 9.07 |
| `frame-modulation` | 0.75 | 5.10 | 9.07 |
| `feed-forward-1024` | 0.66 | 5.10 | 9.07 |
| `normalization-1024` | 0.65 | 5.10 | 9.07 |

**Vocoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.78 | 5.19 | 7.25 |
| `rope-reuse` | 0.79 | 5.08 | 7.32 |
| `compact-timesteps` | 0.82 | 5.08 | 7.23 |
| `inplace-residuals` | 0.64 | 5.24 | 7.23 |
| `fused-rms-modulation` | 0.60 | 5.24 | 7.23 |
| `fused-rotary-pair` | 0.86 | 5.24 | 7.24 |
| `phase-modulation` | 0.63 | 5.24 | 7.24 |
| `frame-modulation` | 0.82 | 5.24 | 7.23 |
| `feed-forward-1024` | 0.66 | 5.24 | 7.23 |
| `normalization-1024` | 0.61 | 5.24 | 7.23 |

**Cleanup vocoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.29 | 5.19 | 7.25 |
| `rope-reuse` | 0.25 | 5.08 | 7.32 |
| `compact-timesteps` | 0.25 | 5.08 | 7.24 |
| `inplace-residuals` | 0.30 | 5.24 | 7.24 |
| `fused-rms-modulation` | 0.30 | 5.24 | 7.23 |
| `fused-rotary-pair` | 0.32 | 5.24 | 7.24 |
| `phase-modulation` | 0.30 | 5.24 | 7.24 |
| `frame-modulation` | 0.32 | 5.24 | 7.23 |
| `feed-forward-1024` | 0.30 | 5.24 | 7.23 |
| `normalization-1024` | 0.32 | 5.24 | 7.23 |

**Save video**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 2.38 | 5.07 | 8.43 |
| `rope-reuse` | 2.00 | 5.08 | 8.50 |
| `compact-timesteps` | 2.04 | 5.08 | 8.41 |
| `inplace-residuals` | 2.33 | 5.08 | 8.42 |
| `fused-rms-modulation` | 2.18 | 5.08 | 8.42 |
| `fused-rotary-pair` | 2.29 | 5.08 | 8.42 |
| `phase-modulation` | 2.31 | 5.08 | 8.42 |
| `frame-modulation` | 2.60 | 5.08 | 8.41 |
| `feed-forward-1024` | 2.37 | 5.08 | 8.41 |
| `normalization-1024` | 2.46 | 5.08 | 8.41 |

</details>


<a id="linux-cumulative-model-features"></a>

## Linux cumulative model features

The [Linux feature tables](model-features.md#linux-observations) use one
completed measurement per cumulative case at exactly 0.62890625 GiB configured
transformer residency. The RoPE row uses the completed retry, with the earlier
interruption preserved separately. Memory peaks are scoped to each named phase.

### Text encoder

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 10.86 | 4.66 | 26.25 |
| `rope-reuse` | 11.12 | 4.65 | 26.25 |
| `compact-timesteps` | 11.11 | 4.64 | 26.23 |
| `inplace-residuals` | 10.93 | 4.56 | 26.23 |
| `fused-rms-modulation` | 11.01 | 4.51 | 26.24 |
| `fused-rotary-pair` | 10.85 | 4.51 | 26.25 |
| `phase-modulation` | 10.84 | 4.55 | 26.25 |
| `frame-modulation` | 10.81 | 4.54 | 26.25 |
| `feed-forward-1024` | 10.88 | 4.52 | 26.24 |
| `normalization-1024` | 10.73 | 4.55 | 26.24 |

### Denoising

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 271.79 | 7.67 | 33.04 |
| `rope-reuse` | 295.18 | 7.69 | 33.54 |
| `compact-timesteps` | 282.02 | 6.14 | 33.53 |
| `inplace-residuals` | 266.85 | 5.15 | 33.53 |
| `fused-rms-modulation` | 265.38 | 5.13 | 33.53 |
| `fused-rotary-pair` | 259.77 | 5.13 | 33.54 |
| `phase-modulation` | 265.90 | 4.15 | 33.53 |
| `frame-modulation` | 262.79 | 3.64 | 33.54 |
| `feed-forward-1024` | 263.65 | 3.11 | 33.53 |
| `normalization-1024` | 264.41 | 3.14 | 33.53 |

### Video VAE decode

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 19.75 | 5.47 | 14.55 |
| `rope-reuse` | 20.85 | 5.33 | 14.28 |
| `compact-timesteps` | 19.76 | 5.34 | 14.35 |
| `inplace-residuals` | 19.34 | 4.72 | 14.52 |
| `fused-rms-modulation` | 19.59 | 4.70 | 14.61 |
| `fused-rotary-pair` | 19.07 | 4.70 | 14.17 |
| `phase-modulation` | 19.13 | 4.73 | 13.60 |
| `frame-modulation` | 19.73 | 4.72 | 14.54 |
| `feed-forward-1024` | 19.56 | 4.69 | 14.71 |
| `normalization-1024` | 19.22 | 4.73 | 14.43 |

<details markdown="1">
<summary>Other setup, decoder and output phases</summary>

**Connectors**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 4.81 | 2.30 | 11.40 |
| `rope-reuse` | 5.57 | 2.29 | 11.31 |
| `compact-timesteps` | 4.73 | 2.29 | 11.26 |
| `inplace-residuals` | 4.63 | 2.20 | 11.38 |
| `fused-rms-modulation` | 4.63 | 2.15 | 11.40 |
| `fused-rotary-pair` | 4.63 | 2.15 | 11.43 |
| `phase-modulation` | 4.64 | 2.19 | 11.39 |
| `frame-modulation` | 4.75 | 2.18 | 11.41 |
| `feed-forward-1024` | 4.62 | 2.16 | 11.46 |
| `normalization-1024` | 4.65 | 2.19 | 11.40 |

**Image conditioning encoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 2.02 | 2.31 | 7.97 |
| `rope-reuse` | 1.67 | 2.29 | 7.97 |
| `compact-timesteps` | 1.65 | 2.29 | 7.95 |
| `inplace-residuals` | 1.62 | 2.20 | 7.95 |
| `fused-rms-modulation` | 1.64 | 2.16 | 7.96 |
| `fused-rotary-pair` | 1.61 | 2.16 | 7.97 |
| `phase-modulation` | 1.62 | 2.19 | 7.97 |
| `frame-modulation` | 1.67 | 2.19 | 7.97 |
| `feed-forward-1024` | 1.62 | 2.16 | 7.96 |
| `normalization-1024` | 1.65 | 2.19 | 7.97 |

**Audio conditioning preparation**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.44 | 2.31 | 6.76 |
| `rope-reuse` | 0.35 | 2.30 | 6.76 |
| `compact-timesteps` | 0.30 | 2.30 | 6.74 |
| `inplace-residuals` | 0.28 | 2.21 | 6.74 |
| `fused-rms-modulation` | 0.28 | 2.17 | 6.75 |
| `fused-rotary-pair` | 0.29 | 2.17 | 6.76 |
| `phase-modulation` | 0.28 | 2.20 | 6.75 |
| `frame-modulation` | 0.28 | 2.19 | 6.76 |
| `feed-forward-1024` | 0.28 | 2.17 | 6.75 |
| `normalization-1024` | 0.28 | 2.20 | 6.75 |

**Transformer load and preparation**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.06 | 2.31 | 6.86 |
| `rope-reuse` | 1.38 | 2.30 | 7.37 |
| `compact-timesteps` | 1.22 | 2.30 | 7.36 |
| `inplace-residuals` | 1.17 | 2.21 | 7.36 |
| `fused-rms-modulation` | 1.17 | 2.17 | 7.37 |
| `fused-rotary-pair` | 1.16 | 2.17 | 7.37 |
| `phase-modulation` | 1.18 | 2.20 | 7.37 |
| `frame-modulation` | 1.16 | 2.19 | 7.37 |
| `feed-forward-1024` | 1.16 | 2.17 | 7.37 |
| `normalization-1024` | 1.16 | 2.20 | 7.37 |

**Transformer attachment**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 2.82 | 2.32 | 13.73 |
| `rope-reuse` | 3.18 | 2.31 | 15.78 |
| `compact-timesteps` | 2.81 | 2.30 | 15.08 |
| `inplace-residuals` | 2.72 | 2.22 | 14.35 |
| `fused-rms-modulation` | 2.69 | 2.17 | 14.40 |
| `fused-rotary-pair` | 2.71 | 2.17 | 14.79 |
| `phase-modulation` | 2.82 | 2.21 | 15.34 |
| `frame-modulation` | 3.22 | 2.20 | 16.18 |
| `feed-forward-1024` | 2.73 | 2.18 | 14.48 |
| `normalization-1024` | 2.77 | 2.21 | 15.61 |

**Load video VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.25 | 4.77 | 7.39 |
| `rope-reuse` | 0.28 | 5.33 | 8.01 |
| `compact-timesteps` | 0.30 | 4.59 | 7.92 |
| `inplace-residuals` | 0.25 | 4.50 | 7.38 |
| `fused-rms-modulation` | 0.25 | 4.48 | 7.39 |
| `fused-rotary-pair` | 0.25 | 4.48 | 7.39 |
| `phase-modulation` | 0.30 | 3.50 | 7.91 |
| `frame-modulation` | 0.25 | 2.98 | 7.39 |
| `feed-forward-1024` | 0.25 | 2.45 | 7.38 |
| `normalization-1024` | 0.30 | 2.48 | 7.90 |

**Cleanup video VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.26 | 5.47 | 7.44 |
| `rope-reuse` | 0.35 | 5.33 | 7.45 |
| `compact-timesteps` | 0.28 | 5.34 | 7.43 |
| `inplace-residuals` | 0.24 | 4.72 | 7.42 |
| `fused-rms-modulation` | 0.29 | 4.70 | 7.43 |
| `fused-rotary-pair` | 0.24 | 4.70 | 7.44 |
| `phase-modulation` | 0.24 | 4.73 | 7.43 |
| `frame-modulation` | 0.31 | 4.72 | 7.44 |
| `feed-forward-1024` | 0.31 | 4.69 | 7.43 |
| `normalization-1024` | 0.23 | 4.73 | 7.44 |

**Load audio VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.07 | 4.66 | 7.44 |
| `rope-reuse` | 0.09 | 4.64 | 7.45 |
| `compact-timesteps` | 0.03 | 4.59 | 7.43 |
| `inplace-residuals` | 0.03 | 4.56 | 7.42 |
| `fused-rms-modulation` | 0.03 | 4.54 | 7.43 |
| `fused-rotary-pair` | 0.03 | 4.54 | 7.44 |
| `phase-modulation` | 0.03 | 4.57 | 7.43 |
| `frame-modulation` | 0.04 | 4.56 | 7.44 |
| `feed-forward-1024` | 0.04 | 4.54 | 7.43 |
| `normalization-1024` | 0.03 | 4.57 | 7.44 |

**Audio VAE decode**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.04 | 4.76 | 7.44 |
| `rope-reuse` | 0.04 | 4.64 | 7.45 |
| `compact-timesteps` | 0.04 | 4.59 | 7.43 |
| `inplace-residuals` | 0.04 | 4.66 | 7.42 |
| `fused-rms-modulation` | 0.04 | 4.64 | 7.43 |
| `fused-rotary-pair` | 0.04 | 4.64 | 7.44 |
| `phase-modulation` | 0.04 | 4.67 | 7.43 |
| `frame-modulation` | 0.04 | 4.66 | 7.44 |
| `feed-forward-1024` | 0.04 | 4.64 | 7.43 |
| `normalization-1024` | 0.04 | 4.67 | 7.44 |

**Cleanup audio VAE**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.23 | 4.78 | 7.44 |
| `rope-reuse` | 0.31 | 4.65 | 7.45 |
| `compact-timesteps` | 0.24 | 4.59 | 7.43 |
| `inplace-residuals` | 0.23 | 4.67 | 7.43 |
| `fused-rms-modulation` | 0.23 | 4.65 | 7.43 |
| `fused-rotary-pair` | 0.24 | 4.65 | 7.44 |
| `phase-modulation` | 0.23 | 4.68 | 7.44 |
| `frame-modulation` | 0.30 | 4.67 | 7.44 |
| `feed-forward-1024` | 0.29 | 4.65 | 7.43 |
| `normalization-1024` | 0.23 | 4.68 | 7.44 |

**Load vocoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.33 | 4.68 | 7.68 |
| `rope-reuse` | 0.40 | 4.65 | 7.65 |
| `compact-timesteps` | 0.19 | 4.59 | 7.44 |
| `inplace-residuals` | 0.19 | 4.58 | 7.43 |
| `fused-rms-modulation` | 0.19 | 4.55 | 7.44 |
| `fused-rotary-pair` | 0.18 | 4.56 | 7.45 |
| `phase-modulation` | 0.18 | 4.59 | 7.45 |
| `frame-modulation` | 0.24 | 4.58 | 7.68 |
| `feed-forward-1024` | 0.18 | 4.55 | 7.45 |
| `normalization-1024` | 0.18 | 4.59 | 7.45 |

**Vocoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.58 | 4.83 | 7.54 |
| `rope-reuse` | 0.66 | 4.65 | 7.55 |
| `compact-timesteps` | 0.57 | 4.59 | 7.53 |
| `inplace-residuals` | 0.55 | 4.72 | 7.53 |
| `fused-rms-modulation` | 0.56 | 4.70 | 7.54 |
| `fused-rotary-pair` | 0.54 | 4.70 | 7.55 |
| `phase-modulation` | 0.55 | 4.73 | 7.54 |
| `frame-modulation` | 0.66 | 4.72 | 7.55 |
| `feed-forward-1024` | 0.55 | 4.70 | 7.54 |
| `normalization-1024` | 0.55 | 4.73 | 7.54 |

**Cleanup vocoder**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 0.25 | 4.83 | 7.54 |
| `rope-reuse` | 0.27 | 4.65 | 7.55 |
| `compact-timesteps` | 0.27 | 4.59 | 7.53 |
| `inplace-residuals` | 0.26 | 4.72 | 7.53 |
| `fused-rms-modulation` | 0.26 | 4.70 | 7.54 |
| `fused-rotary-pair` | 0.26 | 4.70 | 7.55 |
| `phase-modulation` | 0.26 | 4.73 | 7.54 |
| `frame-modulation` | 0.35 | 4.72 | 7.55 |
| `feed-forward-1024` | 0.25 | 4.70 | 7.54 |
| `normalization-1024` | 0.26 | 4.73 | 7.54 |

**Save video**

| Configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| `vmm-fast-control` | 1.80 | 4.67 | 8.72 |
| `rope-reuse` | 1.91 | 4.65 | 8.73 |
| `compact-timesteps` | 1.85 | 4.59 | 8.71 |
| `inplace-residuals` | 1.77 | 4.56 | 8.71 |
| `fused-rms-modulation` | 1.79 | 4.54 | 8.72 |
| `fused-rotary-pair` | 1.77 | 4.54 | 8.72 |
| `phase-modulation` | 1.79 | 4.58 | 8.72 |
| `frame-modulation` | 1.77 | 4.57 | 8.72 |
| `feed-forward-1024` | 1.75 | 4.54 | 8.72 |
| `normalization-1024` | 1.79 | 4.57 | 8.72 |

</details>
