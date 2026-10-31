# LTX 2.3 video benchmarks

[Test hardware and platform limits](hardware.md).

**Leaf control:** Windows and native Linux use Diffusers leaf offloading with
`use_stream=True`. WSL uses `use_stream=False` because streamed leaf offloading
was incompatible with the tested WSL environment; this is not a limitation
established for every WSL setup. This setting concerns weight offloading, not
video VAE decode streaming.

Distilled BF16, 1280×704, 5 seconds at 24 fps (121 frames), 8 steps, seed 43; real guitar image/prompt, no input audio, generated stereo audio. Shared official leaf encoder/auxiliaries and resident temporal GPU VAE decode.

Presets resolve to the executed [transformer runtimes](runtimes.md); the VAE route
is independent. Scenario names link directly to [measurement details](ltx23-video-details.md).
Times are seconds; GPU is sampled **absolute driver usage**, RAM is process **RSS**,
and memory units are **GiB**. Average is denoise wall divided by steps, including
warmup, not completed warm GPU timing. Tracked total excludes command startup and
downloads. **Before reading the tables, see the [labels and memory glossary](glossary.md).**

In comparable overview tables, official Diffusers leaf is the first row and the
reference for that table: 🟢 ↓ is lower, 🔴 ↑ is higher. Equal displayed values
have no arrow. Arrows describe measurements, not an overall preset ranking;
higher memory can accompany faster execution. Phase tables and separate workloads
without a leaf control have no arrows. Rows with different driver/cache sessions
or a reused non-side-by-side observation also have no arrows.

## Windows — BF16

Phase-table rows identify the complete pipeline by its transformer route.
Encoder, conditioning and VAE strategies are shared as stated above; a row
labeled `vmm_fast` does not imply native VMM execution in every phase.

A separate [two-step residual CPU source ownership validation](ltx23-video-details.md#residual-cpu-source-detachment)
recovered checkpoint-related commit headroom with exact output equivalence.
Its larger conditioned workload and experimental backing policy are not rows in
the matched performance table below.

| Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) (reference) | official leaf hooks | resident, temporal streaming | 392.91 | 49.11 | 452.33 | 4.72 | 41.34 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | linear_runtime | resident, temporal streaming | 262.22&nbsp;🟢&nbsp;↓ | 32.78&nbsp;🟢&nbsp;↓ | 336.44&nbsp;🟢&nbsp;↓ | 4.72 | 47.87&nbsp;🔴&nbsp;↑ |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | linear_runtime | resident, temporal streaming | 328.62&nbsp;🟢&nbsp;↓ | 41.08&nbsp;🟢&nbsp;↓ | 394.89&nbsp;🟢&nbsp;↓ | 6.81&nbsp;🔴&nbsp;↑ | 42.22&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | vmm_fast_runtime | resident, temporal streaming | 207.40&nbsp;🟢&nbsp;↓ | 25.92&nbsp;🟢&nbsp;↓ | 277.09&nbsp;🟢&nbsp;↓ | 5.21&nbsp;🔴&nbsp;↑ | 32.65&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | vmm_runtime | resident, temporal streaming | 218.46&nbsp;🟢&nbsp;↓ | 27.31&nbsp;🟢&nbsp;↓ | 281.67&nbsp;🟢&nbsp;↓ | 5.24&nbsp;🔴&nbsp;↑ | 25.99&nbsp;🟢&nbsp;↓ |

### Phase measurements

These are selected separately monitored phases, not an additive decomposition
of the tracked total. Cleanup, saving and other unlisted setup remain in the total.

#### Text encoder

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 16.67 | 3.28 | 25.75 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 17.15 | 3.28 | 25.75 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 16.79 | 3.28 | 25.75 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 17.14 | 2.96 | 25.80 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 16.86 | 3.28 | 25.75 |

#### Connectors

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 7.00 | 2.73 | 25.99 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 7.59 | 2.73 | 25.99 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 7.12 | 2.73 | 25.99 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 7.64 | 2.96 | 25.98 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 7.33 | 2.73 | 25.99 |

#### Image conditioning encoder

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 1.60 | 1.44 | 5.90 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 2.11 | 1.44 | 8.04 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 1.83 | 1.44 | 8.04 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 2.07 | 2.96 | 8.05 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 1.59 | 1.44 | 5.90 |

#### Audio conditioning preparation

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 0.14 | 1.10 | 5.30 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 0.16 | 1.10 | 6.69 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 0.14 | 1.10 | 6.69 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 0.16 | 2.96 | 6.69 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 0.14 | 1.10 | 5.30 |

#### Transformer load/prepare

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 1.55 | 1.59 | 5.45 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 1.73 | 1.59 | 6.84 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 1.53 | 1.59 | 6.84 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 1.82 | 2.96 | 6.84 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 1.64 | 1.59 | 5.45 |

#### Transformer attachment

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 0.15 | 1.10 | 5.45 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 11.29 | 1.12 | 41.65 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 2.60 | 4.44 | 10.02 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 4.65 | 2.96 | 8.36 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 2.92 | 1.89 | 7.25 |

#### Denoise

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 392.91 | 3.32 | 41.34 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 262.22 | 3.19 | 47.87 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 328.62 | 6.81 | 42.22 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 207.40 | 5.08 | 32.65 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 218.46 | 5.24 | 7.30 |

#### Video VAE decode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 22.52 | 4.72 | 11.57 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 23.47 | 4.72 | 33.02 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 22.84 | 4.72 | 12.49 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 24.00 | 5.21 | 12.36 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 22.72 | 4.72 | 11.77 |

#### Audio VAE decode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 0.06 | 1.20 | 9.59 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 0.08 | 1.20 | 31.31 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 0.06 | 1.20 | 10.47 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 0.08 | 3.40 | 10.50 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 0.06 | 1.20 | 10.09 |

#### Vocoder

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) | 0.42 | 1.46 | 9.72 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | 0.44 | 1.50 | 31.45 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | 0.44 | 1.50 | 10.61 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | 0.52 | 3.40 | 10.63 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | 0.42 | 1.47 | 10.23 |

### Pipeline total

Measured tracked totals and global peaks for the same complete runs above.
These are not sums of the phase tables. The encoder and VAE routes remain
as stated for this workload; the runtime column identifies the transformer.

| Method / scenario (details) | Transformer runtime | VAE route | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#windows-leaf) (reference) | official leaf hooks | resident, temporal streaming | 452.33 | 4.72 | 41.34 |
| [`one_shot_fast`](ltx23-video-details.md#windows-linear) | linear_runtime | resident, temporal streaming | 336.44 | 4.72 | 47.87 |
| [`low_ram_safe`](ltx23-video-details.md#windows-low-ram) | linear_runtime | resident, temporal streaming | 394.89 | 6.81 | 42.22 |
| [`vmm_fast`](ltx23-video-details.md#windows-native) | vmm_fast_runtime | resident, temporal streaming | 277.09 | 5.21 | 32.65 |
| [`nvme_baseline`](ltx23-video-details.md#windows-nvme) | vmm_runtime | resident, temporal streaming | 281.67 | 5.24 | 25.99 |

## Linux — BF16

The NVMe observation uses a different driver/cache session from the route trio;
it is not a controlled allocator-only A/B.

| Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) (reference) | official leaf hooks | resident, temporal streaming | 338.01 | 42.25 | 386.81 | 4.09 | 41.21 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | linear_runtime | resident, temporal streaming | 216.16&nbsp;🟢&nbsp;↓ | 27.02&nbsp;🟢&nbsp;↓ | 296.27&nbsp;🟢&nbsp;↓ | 4.05&nbsp;🟢&nbsp;↓ | 43.30&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | vmm_fast_runtime | resident, temporal streaming | 188.83&nbsp;🟢&nbsp;↓ | 23.60&nbsp;🟢&nbsp;↓ | 239.87&nbsp;🟢&nbsp;↓ | 5.63&nbsp;🔴&nbsp;↑ | 32.85&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | vmm_runtime | resident, temporal streaming | 318.65 | 39.83 | 379.54 | 5.27 | 41.90 |

### Phase measurements

These are selected separately monitored phases, not an additive decomposition
of the tracked total. Cleanup, saving and other unlisted setup remain in the total.

#### Text encoder

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 9.47 | 3.04 | 25.85 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 9.63 | 3.16 | 25.81 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 9.52 | 2.90 | 25.82 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 10.30 | 3.18 | 25.85 |

#### Connectors

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 5.04 | 2.51 | 26.02 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 5.03 | 2.63 | 26.02 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 4.93 | 2.90 | 26.00 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 5.24 | 2.64 | 26.06 |

#### Image conditioning encoder

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 1.33 | 1.04 | 5.56 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 1.69 | 1.17 | 7.71 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 1.92 | 2.91 | 7.69 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 1.23 | 1.16 | 5.59 |

#### Audio conditioning preparation

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 0.15 | 0.87 | 4.96 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 0.16 | 1.01 | 6.36 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.15 | 2.91 | 6.34 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 0.15 | 0.99 | 5.00 |

#### Transformer load/prepare

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 1.66 | 1.42 | 5.43 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 1.68 | 1.57 | 6.83 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 1.64 | 2.91 | 6.81 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 1.70 | 1.55 | 5.47 |

#### Transformer attachment

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 0.12 | 0.87 | 5.43 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 24.67 | 1.10 | 43.30 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 1.65 | 2.92 | 8.31 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 2.69 | 1.80 | 7.27 |

#### Denoise

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 338.01 | 3.33 | 41.21 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 216.16 | 3.01 | 32.56 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 188.83 | 5.06 | 32.85 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 318.65 | 5.27 | 41.90 |

#### Video VAE decode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 25.61 | 4.09 | 10.25 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 24.50 | 4.05 | 4.94 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 24.84 | 5.63 | 10.74 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 33.06 | 4.13 | 10.74 |

#### Audio VAE decode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 0.05 | 1.14 | 5.92 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 0.07 | 1.10 | 1.27 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.06 | 3.25 | 6.69 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 0.06 | 1.16 | 6.26 |

#### Vocoder

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) | 0.40 | 1.38 | 6.05 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | 0.44 | 1.36 | 1.40 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 0.50 | 3.25 | 6.81 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | 0.43 | 1.44 | 6.39 |

### Pipeline total

Measured tracked totals and global peaks for the same complete runs above.
These are not sums of the phase tables. The encoder and VAE routes remain
as stated for this workload; the runtime column identifies the transformer.

| Method / scenario (details) | Transformer runtime | VAE route | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| [Official leaf](ltx23-video-details.md#linux-leaf) (reference) | official leaf hooks | resident, temporal streaming | 386.81 | 4.09 | 41.21 |
| [`one_shot_fast` + host-cache release](ltx23-video-details.md#linux-linear) | linear_runtime | resident, temporal streaming | 296.27 | 4.05 | 43.30 |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | vmm_fast_runtime | resident, temporal streaming | 239.87 | 5.63 | 32.85 |
| [`nvme_baseline`](ltx23-video-details.md#linux-nvme) | vmm_runtime | resident, temporal streaming | 379.54 | 5.27 | 41.90 |

## WSL — BF16

Leaf controls are non-streamed; linear uses an explicit 3 GiB override and
native uses bounded registration. These are not Windows/Linux parity claims.

| Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | official leaf hooks | resident, temporal streaming | 1419.53 | 177.44 | 1644.53 | 4.14 | 38.62 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | linear_runtime | resident, temporal streaming | 489.98&nbsp;🟢&nbsp;↓ | 61.25&nbsp;🟢&nbsp;↓ | 750.73&nbsp;🟢&nbsp;↓ | 6.67&nbsp;🔴&nbsp;↑ | 38.51&nbsp;🟢&nbsp;↓ |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | vmm_fast_runtime | resident, temporal streaming | 372.65&nbsp;🟢&nbsp;↓ | 46.58&nbsp;🟢&nbsp;↓ | 604.45&nbsp;🟢&nbsp;↓ | 5.31&nbsp;🔴&nbsp;↑ | 38.54&nbsp;🟢&nbsp;↓ |

### Phase measurements

These are selected separately monitored phases, not an additive decomposition
of the tracked total. Cleanup, saving and other unlisted setup remain in the total.

#### Text encoder

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 132.55 | 2.98 | 38.54 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 144.30 | 2.98 | 38.51 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 131.31 | 2.94 | 38.54 |

#### Connectors

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 38.25 | 2.73 | 38.40 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 42.82 | 2.73 | 38.29 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 39.46 | 2.91 | 38.36 |

#### Image conditioning encoder

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 5.36 | 1.26 | 6.32 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 6.16 | 1.26 | 6.33 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 5.95 | 1.42 | 6.33 |

#### Audio conditioning preparation

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 0.30 | 1.10 | 5.73 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 0.23 | 1.10 | 5.74 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 0.23 | 1.42 | 5.73 |

#### Transformer load/prepare

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 2.68 | 1.63 | 5.84 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 2.84 | 1.63 | 5.84 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 2.77 | 1.74 | 5.84 |

#### Transformer attachment

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 0.09 | 1.10 | 5.73 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 18.20 | 4.41 | 9.13 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 7.65 | 1.56 | 6.11 |

#### Denoise

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 1419.53 | 3.03 | 38.62 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 489.98 | 6.67 | 38.47 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 372.65 | 4.54 | 30.77 |

#### Video VAE decode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 24.38 | 4.14 | 17.34 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 23.67 | 4.14 | 10.31 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 25.07 | 5.31 | 10.41 |

#### Audio VAE decode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 0.29 | 1.24 | 12.55 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 0.07 | 1.20 | 5.73 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 0.07 | 1.54 | 5.74 |

#### Vocoder

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) | 0.56 | 1.47 | 12.57 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | 0.59 | 1.47 | 5.74 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | 0.53 | 1.59 | 5.75 |

### Pipeline total

Measured tracked totals and global peaks for the same complete runs above.
These are not sums of the phase tables. The encoder and VAE routes remain
as stated for this workload; the runtime column identifies the transformer.

| Method / scenario (details) | Transformer runtime | VAE route | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | official leaf hooks | resident, temporal streaming | 1644.53 | 4.14 | 38.62 |
| [`wsl_compat` (3 GiB override)](ltx23-video-details.md#wsl-linear) | linear_runtime | resident, temporal streaming | 750.73 | 6.67 | 38.51 |
| [`vmm_fast` (bounded)](ltx23-video-details.md#wsl-native) | vmm_fast_runtime | resident, temporal streaming | 604.45 | 5.31 | 38.54 |

## VAE strategies — separate native-encoder workload

Five seconds, 1280×704, 24 fps, 8 steps, seed 43. Native text encoder; temporal GPU streaming in both decoder routes.

| Platform / Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Windows / `vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | vmm_fast_runtime | resident, temporal streaming | 198.58 | 24.82 | 261.67 | 5.14 | 28.42 |
| [Windows / `vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | vmm_fast_runtime | leaf, temporal streaming | 199.30 | 24.91 | 271.83 | 4.96 | 28.42 |
| [Linux / `vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | vmm_fast_runtime | resident, temporal streaming | 209.75 | 26.22 | 266.51 | 4.87 | 28.42 |
| [Linux / `vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | vmm_fast_runtime | leaf, temporal streaming | 209.50 | 26.19 | 272.25 | 4.79 | 28.42 |

### Phase measurements

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Text encoder | 12.08 | 5.14 | 25.29 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Connectors | 6.50 | 2.82 | 11.31 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Image conditioning encoder | 3.85 | 1.45 | 8.09 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Audio conditioning preparation | 0.15 | 1.45 | 6.73 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Transformer load/prepare | 1.81 | 1.71 | 6.88 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Transformer attachment | 6.23 | 1.72 | 3.75 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Denoise | 198.58 | 4.57 | 28.42 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Video VAE decode | 22.66 | 4.81 | 7.16 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Audio VAE decode | 0.07 | 1.54 | 5.89 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-windows-resident) | Vocoder | 0.48 | 1.54 | 6.02 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Text encoder | 12.15 | 4.96 | 25.29 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Connectors | 6.47 | 2.82 | 11.36 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Image conditioning encoder | 3.21 | 1.45 | 8.09 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Audio conditioning preparation | 0.14 | 1.45 | 6.73 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Transformer load/prepare | 1.99 | 1.71 | 6.88 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Transformer attachment | 6.66 | 1.72 | 3.75 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Denoise | 199.30 | 4.57 | 28.42 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Video VAE decode | 32.88 | 3.02 | 9.27 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Audio VAE decode | 0.06 | 1.57 | 7.06 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-windows-leaf) | Vocoder | 0.48 | 1.57 | 7.18 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Text encoder | 14.71 | 4.78 | 25.29 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Connectors | 5.41 | 2.43 | 11.06 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Image conditioning encoder | 1.83 | 1.04 | 7.78 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Audio conditioning preparation | 0.16 | 1.04 | 6.43 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Transformer load/prepare | 2.54 | 1.47 | 6.90 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Transformer attachment | 1.55 | 1.22 | 3.76 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Denoise | 209.75 | 4.14 | 28.42 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Video VAE decode | 24.39 | 4.87 | 6.23 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Audio VAE decode | 0.05 | 1.11 | 2.26 |
| [`vmm_fast` / resident VAE](ltx23-video-details.md#vae-linux-resident) | Vocoder | 0.45 | 1.16 | 2.38 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Text encoder | 14.65 | 4.79 | 25.29 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Connectors | 4.65 | 2.43 | 11.15 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Image conditioning encoder | 1.61 | 1.04 | 7.77 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Audio conditioning preparation | 0.13 | 1.04 | 6.42 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Transformer load/prepare | 2.01 | 1.47 | 6.89 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Transformer attachment | 1.56 | 1.22 | 3.75 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Denoise | 209.50 | 4.14 | 28.42 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Video VAE decode | 32.50 | 2.59 | 7.51 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Audio VAE decode | 0.03 | 1.13 | 2.53 |
| [`vmm_fast` / leaf VAE](ltx23-video-details.md#vae-linux-leaf) | Vocoder | 0.39 | 1.13 | 2.58 |

## Saved FP8 weight-only — separate precision pairs

Five-second video; auxiliary components remain unchanged within each platform pair. Lower transformer storage does not imply faster denoising or lower peak GPU usage.

| Platform / Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Windows / `vmm_fast` / BF16 control](ltx23-video-details.md#fp8-windows-bf16) | vmm_fast_runtime | resident, temporal streaming | 199.36 | 24.92 | 262.47 | 5.14 | 28.42 |
| [Windows / `vmm_fast` / saved FP8](ltx23-video-details.md#fp8-windows) | vmm_fast_runtime | resident, temporal streaming | 218.59 | 27.32 | 312.98 | 5.92 | 25.38 |
| [Linux / `vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | vmm_fast_runtime | resident, temporal streaming | 188.83 | 23.60 | 239.87 | 5.63 | 32.85 |
| [Linux / `vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | vmm_fast_runtime | resident, temporal streaming | 199.38 | 24.92 | 275.66 | 7.32 | 26.04 |

### Phase measurements

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-windows-bf16) | Text encoder | 12.22 | 5.14 | 25.29 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-windows-bf16) | Connectors | 6.67 | 2.82 | 11.44 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-windows-bf16) | Image conditioning encoder | 3.30 | 1.45 | 5.94 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-windows-bf16) | Audio conditioning preparation | 0.13 | 1.45 | 5.34 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-windows-bf16) | Transformer load/prepare | 1.40 | 1.78 | 5.49 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-windows-bf16) | Transformer attachment | 6.52 | 1.72 | 3.75 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-windows-bf16) | Denoise | 199.36 | 4.57 | 28.42 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-windows-bf16) | Decode/save aggregate | 26.87 | 4.81 | 7.46 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-windows) | Text encoder | 9.77 | 5.14 | 25.29 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-windows) | Connectors | 3.59 | 2.82 | 11.43 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-windows) | Image conditioning encoder | 3.13 | 1.45 | 5.94 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-windows) | Audio conditioning preparation | 0.12 | 1.45 | 5.34 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-windows) | Transformer load/prepare | 32.43 | 1.78 | 25.38 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-windows) | Transformer attachment | 10.46 | 1.72 | 23.70 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-windows) | Denoise | 218.59 | 5.92 | 23.85 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-windows) | Decode/save aggregate | 26.94 | 4.81 | 9.17 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Text encoder | 9.52 | 2.90 | 25.82 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Connectors | 4.93 | 2.90 | 26.00 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Image conditioning encoder | 1.92 | 2.91 | 7.69 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Audio conditioning preparation | 0.15 | 2.91 | 6.34 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Transformer load/prepare | 1.64 | 2.91 | 6.81 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Transformer attachment | 1.65 | 2.92 | 8.31 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Denoise | 188.83 | 5.06 | 32.85 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Video VAE decode | 24.84 | 5.63 | 10.74 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Audio VAE decode | 0.06 | 3.25 | 6.69 |
| [`vmm_fast` / BF16 control](ltx23-video-details.md#fp8-linux-bf16) | Vocoder | 0.50 | 3.25 | 6.81 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Text encoder | 9.53 | 2.82 | 25.83 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Connectors | 5.03 | 2.82 | 26.04 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Image conditioning encoder | 1.94 | 2.82 | 7.73 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Audio conditioning preparation | 0.15 | 2.82 | 6.38 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Transformer load/prepare | 23.70 | 2.82 | 24.30 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Transformer attachment | 5.93 | 2.83 | 25.70 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Denoise | 199.38 | 7.32 | 21.36 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Video VAE decode | 23.40 | 5.21 | 6.06 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Audio VAE decode | 0.06 | 3.05 | 2.00 |
| [`vmm_fast` / saved FP8](ltx23-video-details.md#fp8-linux) | Vocoder | 0.46 | 3.05 | 2.02 |

## WSL video with conditioning audio

Separate five-second workload with guitar image and input audio; not matched to the no-input-audio WSL comparison above.

| Platform / Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [WSL / `vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | vmm_fast_runtime | resident, temporal streaming | 367.62 | 45.95 | 512.35 | 5.32 | 27.63 |

### Phase measurements

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Text encoder | 41.74 | 5.19 | 24.21 |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Connectors | 38.45 | 2.87 | 14.01 |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Image conditioning encoder | 7.02 | 1.47 | 3.24 |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Audio conditioning preparation | 3.02 | 1.48 | 2.37 |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Transformer load/prepare | 2.88 | 1.82 | 2.60 |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Transformer attachment | 6.90 | 1.56 | 2.97 |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Denoise | 367.62 | 4.54 | 27.63 |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Video VAE decode | 25.20 | 5.32 | 6.89 |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Audio VAE decode | 0.04 | 1.56 | 2.50 |
| [`vmm_fast` / image + audio](ltx23-video-details.md#wsl-conditioned) | Vocoder | 0.61 | 1.64 | 2.58 |

## Recommendations

| Hardware / priority | Recommendation from these measurements |
| --- | --- |
| 8 GiB GPU, ample system RAM, speed | `vmm_fast` has the shortest tracked total in each primary platform set, with higher GPU peaks than the corresponding official leaf control. |
| Lower transformer-phase process RAM on Windows | `nvme_baseline` has much lower denoise RSS than native/linear and a close tracked total; the common encoder still needs about 26 GiB RSS. OS file cache is not included in process RSS. |
| Lower decoder GPU pressure | Leaf VAE with temporal streaming reduces decode GPU peak, at extra time and RAM. Prefer resident temporal decode when it fits. |
| Neither RAM nor GPU headroom is sufficient for decode | Consider supported VAE tiling; validate quality and account for assembled output memory. |
| WSL | Use documented bounded registration or the explicit 3 GiB `wsl_compat` override, not untested preset capacities. |
| Lower transformer storage / process RSS | Saved FP8 is compatible in the measured pairs, but slower than BF16 and not lower in peak GPU usage. |

`low_ram_safe` was not the lowest-memory Windows route. These five-second tables
do not establish capacity or performance for longer videos.

Single observations, not statistical or arbitrary-model guarantees. Missing
measurements are unavailable, not zero; phase peaks must not be summed.
See [presets](../user/presets.md), [reproduction](reproduction.md) and
[measurement details](ltx23-video-details.md) before comparing different recipes.
