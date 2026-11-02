# Qwen Image 2.1 benchmarks

[Test hardware and platform limits](hardware.md).

**Leaf control:** Windows and native Linux use Diffusers leaf offloading with
`use_stream=True`. WSL uses `use_stream=False` because streamed leaf offloading
was incompatible with the tested WSL environment; this is not a limitation
established for every WSL setup. This setting concerns weight offloading, not
video VAE decode streaming.

Base BF16, 512×512, 20 steps, seed 42, capybara prompt, CFG 1 and KV cache. Shared official leaf text encoder; resident untiled VAE on Windows/Linux, untiled non-streamed leaf VAE on WSL.

Presets resolve to the executed [transformer runtimes](runtimes.md); the VAE route
is independent. Scenario names link directly to [measurement details](qwen-image-21-details.md).
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
Encoder and VAE strategies remain those stated for each platform; a row labeled
`vmm_fast` does not imply native VMM execution in every phase.

The native row is a reused official-package observation, not a repeat alongside
the other routes.

| Scenario (details) | Transformer runtime | VAE route | Denoise/readback (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) (reference) | official leaf hooks | resident | 41.28 | 2.06 | 57.50 | 3.68 | 18.97 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | linear_runtime | resident | 17.94&nbsp;🟢&nbsp;↓ | 0.90&nbsp;🟢&nbsp;↓ | 41.36&nbsp;🟢&nbsp;↓ | 6.69&nbsp;🔴&nbsp;↑ | 21.58&nbsp;🔴&nbsp;↑ |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | linear_runtime | resident | 36.86&nbsp;🟢&nbsp;↓ | 1.84&nbsp;🟢&nbsp;↓ | 55.24&nbsp;🟢&nbsp;↓ | 4.36&nbsp;🔴&nbsp;↑ | 18.97 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | vmm_fast_runtime | resident | 17.67 | 0.88 | 36.64 | 6.12 | 18.97 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | vmm_runtime | resident | 38.08&nbsp;🟢&nbsp;↓ | 1.90&nbsp;🟢&nbsp;↓ | 56.32&nbsp;🟢&nbsp;↓ | 5.23&nbsp;🔴&nbsp;↑ | 18.97 |

### Phase measurements

These are selected separately monitored phases, not an additive decomposition
of the tracked total. Cleanup, saving and other unlisted setup remain in the total.

#### Text encoder load

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) | 0.96 | 1.03 | 0.84 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | 0.97 | 1.03 | 0.88 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | 0.93 | 1.03 | 0.88 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | 0.89 | 1.03 | 0.87 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | 0.95 | 1.03 | 0.88 |

#### Text encoder attachment

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) | 0.04 | 1.03 | 0.88 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | 0.04 | 1.03 | 0.88 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | 0.04 | 1.03 | 0.88 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | 0.03 | 1.03 | 0.87 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | 0.04 | 1.03 | 0.88 |

#### Text encode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) | 9.59 | 2.28 | 18.93 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | 9.91 | 2.28 | 18.93 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | 9.58 | 2.28 | 18.91 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | 9.58 | 3.40 | 18.93 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | 9.69 | 2.28 | 18.93 |

#### Text encoder cleanup

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) | 0.63 | 2.28 | 18.97 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | 0.61 | 2.28 | 18.97 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | 0.59 | 2.28 | 18.97 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | 0.60 | 2.24 | 18.97 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | 0.85 | 2.28 | 18.97 |

#### Transformer load

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) | 0.13 | 1.10 | 3.67 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | 0.10 | 1.10 | 3.66 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | 0.11 | 1.10 | 3.66 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | 0.17 | 2.24 | 3.66 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | 0.17 | 1.10 | 3.67 |

#### Transformer attachment

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) | 0.03 | 1.10 | 3.67 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | 6.66 | 6.00 | 21.58 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | 2.38 | 3.94 | 6.43 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | 1.42 | 2.24 | 4.30 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | 1.30 | 1.50 | 5.89 |

#### Denoise/readback

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) | 41.28 | 1.58 | 16.99 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | 17.94 | 6.69 | 13.80 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | 36.86 | 4.36 | 16.99 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | 17.67 | 6.12 | 17.69 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | 38.08 | 5.23 | 6.00 |

#### VAE load

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) | 0.98 | 1.10 | 5.55 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | 0.96 | 1.17 | 15.62 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | 0.95 | 1.34 | 5.61 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | 1.17 | 2.24 | 5.56 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | 0.97 | 1.15 | 7.54 |

#### VAE decode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) | 0.47 | 3.68 | 4.47 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | 0.49 | 3.68 | 14.54 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | 0.44 | 3.68 | 4.47 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | 0.47 | 3.90 | 4.48 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | 0.47 | 3.68 | 6.48 |

### Pipeline total

Measured tracked totals and global peaks for the same complete runs above.
These are not sums of the phase tables. The encoder and VAE routes remain
as stated for this workload; the runtime column identifies the transformer.

| Method / scenario (details) | Transformer runtime | VAE route | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#windows-leaf) (reference) | official leaf hooks | resident | 57.50 | 3.68 | 18.97 |
| [`one_shot_fast`](qwen-image-21-details.md#windows-linear) | linear_runtime | resident | 41.36 | 6.69 | 21.58 |
| [`low_ram_safe`](qwen-image-21-details.md#windows-low-ram) | linear_runtime | resident | 55.24 | 4.36 | 18.97 |
| [`vmm_fast` (reused observation)](qwen-image-21-details.md#windows-native) | vmm_fast_runtime | resident | 36.64 | 6.12 | 18.97 |
| [`nvme_baseline`](qwen-image-21-details.md#windows-nvme) | vmm_runtime | resident | 56.32 | 5.23 | 18.97 |

## Linux — BF16

The NVMe observation uses a different driver/cache session from the route trio;
it is not a controlled allocator-only A/B.

| Scenario (details) | Transformer runtime | VAE route | Denoise/readback (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) (reference) | official leaf hooks | resident | 35.61 | 1.78 | 43.91 | 3.68 | 18.88 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | linear_runtime | resident | 18.11&nbsp;🟢&nbsp;↓ | 0.91&nbsp;🟢&nbsp;↓ | 31.24&nbsp;🟢&nbsp;↓ | 6.09&nbsp;🔴&nbsp;↑ | 21.38&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | vmm_fast_runtime | resident | 13.03&nbsp;🟢&nbsp;↓ | 0.65&nbsp;🟢&nbsp;↓ | 20.77&nbsp;🟢&nbsp;↓ | 6.12&nbsp;🔴&nbsp;↑ | 18.88 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | vmm_runtime | resident | 179.00 | 8.95 | 188.66 | 5.33 | 18.96 |

### Phase measurements

These are selected separately monitored phases, not an additive decomposition
of the tracked total. Cleanup, saving and other unlisted setup remain in the total.

#### Text encoder load

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) | 0.90 | 1.09 | 0.95 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | 0.74 | 1.01 | 0.95 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | 0.65 | 0.99 | 0.95 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | 0.91 | 0.73 | 1.04 |

#### Text encoder attachment

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) | 0.02 | 1.09 | 0.98 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | 0.02 | 1.01 | 0.98 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | 0.02 | 0.99 | 0.98 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | 0.02 | 0.73 | 1.07 |

#### Text encode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) | 5.06 | 2.34 | 18.84 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | 3.42 | 2.27 | 18.87 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | 3.44 | 3.37 | 18.84 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | 5.01 | 1.98 | 18.93 |

#### Text encoder cleanup

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) | 0.60 | 2.33 | 18.88 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | 0.59 | 2.27 | 18.88 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | 0.56 | 2.21 | 18.88 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | 0.56 | 1.99 | 18.96 |

#### Transformer load

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) | 0.10 | 1.15 | 3.57 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | 0.04 | 1.08 | 3.55 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | 0.05 | 2.21 | 3.54 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | 0.14 | 0.80 | 3.65 |

#### Transformer attachment

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) | 0.01 | 1.15 | 3.57 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | 7.15 | 5.68 | 21.38 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | 0.60 | 2.21 | 4.20 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | 1.25 | 1.22 | 4.86 |

#### Denoise/readback

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) | 35.61 | 1.57 | 16.92 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | 18.11 | 6.09 | 14.24 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | 13.03 | 6.12 | 17.63 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | 179.00 | 5.33 | 18.03 |

#### VAE load

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) | 0.43 | 1.47 | 5.40 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | 0.28 | 1.15 | 15.53 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | 0.27 | 2.21 | 4.95 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | 0.43 | 1.49 | 6.45 |

#### VAE decode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) | 0.35 | 3.68 | 4.34 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | 0.36 | 3.74 | 14.91 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | 0.38 | 4.24 | 4.22 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | 0.45 | 3.73 | 5.44 |

### Pipeline total

Measured tracked totals and global peaks for the same complete runs above.
These are not sums of the phase tables. The encoder and VAE routes remain
as stated for this workload; the runtime column identifies the transformer.

| Method / scenario (details) | Transformer runtime | VAE route | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| [Official leaf](qwen-image-21-details.md#linux-leaf) (reference) | official leaf hooks | resident | 43.91 | 3.68 | 18.88 |
| [`one_shot_fast`](qwen-image-21-details.md#linux-linear) | linear_runtime | resident | 31.24 | 6.09 | 21.38 |
| [`vmm_fast`](qwen-image-21-details.md#linux-native) | vmm_fast_runtime | resident | 20.77 | 6.12 | 18.88 |
| [`nvme_baseline`](qwen-image-21-details.md#linux-nvme) | vmm_runtime | resident | 188.66 | 5.33 | 18.96 |

## WSL — BF16

Leaf controls are non-streamed; linear uses an explicit 3 GiB override and
native uses bounded registration. These are not Windows/Linux parity claims.

| Scenario (details) | Transformer runtime | VAE route | Denoise/readback (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) (reference) | official leaf hooks | leaf | 265.68 | 13.28 | 364.02 | 3.42 | 22.34 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | linear_runtime | leaf | 83.66&nbsp;🟢&nbsp;↓ | 4.18&nbsp;🟢&nbsp;↓ | 193.51&nbsp;🟢&nbsp;↓ | 4.28&nbsp;🔴&nbsp;↑ | 22.33&nbsp;🟢&nbsp;↓ |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | vmm_fast_runtime | leaf | 88.42&nbsp;🟢&nbsp;↓ | 4.42&nbsp;🟢&nbsp;↓ | 187.38&nbsp;🟢&nbsp;↓ | 6.72&nbsp;🔴&nbsp;↑ | 22.33&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | vmm_runtime | leaf | 244.37&nbsp;🟢&nbsp;↓ | 12.22&nbsp;🟢&nbsp;↓ | 349.09&nbsp;🟢&nbsp;↓ | 5.93&nbsp;🔴&nbsp;↑ | 22.34 |

### Phase measurements

These are selected separately monitored phases, not an additive decomposition
of the tracked total. Cleanup, saving and other unlisted setup remain in the total.

#### Text encoder load

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) | 1.44 | 1.04 | 0.94 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | 1.45 | 1.04 | 0.94 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | 1.67 | 1.04 | 0.94 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | 1.52 | 1.04 | 0.94 |

#### Text encoder attachment

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) | 0.02 | 1.04 | 0.94 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | 0.01 | 1.04 | 0.94 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | 0.02 | 1.04 | 0.94 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | 0.01 | 1.04 | 0.94 |

#### Text encode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) | 89.40 | 3.42 | 22.34 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | 86.47 | 3.42 | 22.33 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | 87.06 | 3.40 | 22.33 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | 92.05 | 3.42 | 22.34 |

#### Text encoder cleanup

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) | 1.44 | 3.42 | 20.35 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | 1.43 | 3.42 | 20.35 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | 1.38 | 2.24 | 20.35 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | 1.44 | 3.42 | 20.35 |

#### Transformer load

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) | 0.24 | 2.24 | 2.04 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | 0.25 | 2.24 | 2.04 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | 0.33 | 2.24 | 2.04 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | 0.37 | 2.24 | 2.05 |

#### Transformer attachment

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) | 0.01 | 2.24 | 2.05 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | 14.25 | 3.99 | 4.90 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | 2.31 | 2.25 | 2.42 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | 3.34 | 2.66 | 3.29 |

#### Denoise/readback

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) | 265.68 | 2.26 | 20.58 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | 83.66 | 4.28 | 15.30 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | 88.42 | 6.72 | 15.74 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | 244.37 | 5.93 | 16.31 |

#### VAE load

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) | 3.47 | 2.24 | 3.78 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | 3.05 | 2.24 | 3.78 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | 3.38 | 2.25 | 3.84 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | 2.99 | 2.25 | 4.81 |

#### VAE decode

| Method / scenario (details) | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) | 1.15 | 2.95 | 3.09 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | 1.13 | 2.95 | 3.03 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | 1.14 | 3.17 | 3.09 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | 1.15 | 2.96 | 4.08 |

### Pipeline total

Measured tracked totals and global peaks for the same complete runs above.
These are not sums of the phase tables. The encoder and VAE routes remain
as stated for this workload; the runtime column identifies the transformer.

| Method / scenario (details) | Transformer runtime | VAE route | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| [Official leaf (no stream)](qwen-image-21-details.md#wsl-leaf) (reference) | official leaf hooks | leaf | 364.02 | 3.42 | 22.34 |
| [`wsl_compat` (3 GiB override)](qwen-image-21-details.md#wsl-linear) | linear_runtime | leaf | 193.51 | 4.28 | 22.33 |
| [`vmm_fast` (bounded)](qwen-image-21-details.md#wsl-native) | vmm_fast_runtime | leaf | 187.38 | 6.72 | 22.33 |
| [`nvme_baseline`](qwen-image-21-details.md#wsl-nvme) | vmm_runtime | leaf | 349.09 | 5.93 | 22.34 |

## Windows 1024×1024 — separate VAE settings

Base BF16, 20 steps, seed 42; untiled leaf VAE. Native uses explicit cache trimming; this is not an allocator-only A/B.

| Platform / Scenario (details) | Transformer runtime | VAE route | Denoise/readback (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Windows / `one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | linear_runtime | leaf | 54.30 | 2.71 | 78.52 | 7.07 | 21.19 |
| [Windows / `vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | vmm_fast_runtime | leaf + trim | 52.82 | 2.64 | 72.54 | 6.63 | 18.97 |

### Phase measurements

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [`one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | Text encoder load | 0.97 | 1.03 | 0.85 |
| [`one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | Text encoder attachment | 0.04 | 1.03 | 0.87 |
| [`one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | Text encode | 9.80 | 2.28 | 18.96 |
| [`one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | Text encoder cleanup | 0.65 | 2.28 | 18.97 |
| [`one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | Transformer load | 0.11 | 1.10 | 3.66 |
| [`one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | Transformer attachment | 6.51 | 6.00 | 21.19 |
| [`one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | Denoise/readback | 54.30 | 7.07 | 13.80 |
| [`one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | VAE load | 0.87 | 1.10 | 15.66 |
| [`one_shot_fast` / leaf VAE](qwen-image-21-details.md#windows-1024-linear) | VAE decode | 1.58 | 6.20 | 14.58 |
| [`vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | Text encoder load | 0.90 | 1.03 | 0.85 |
| [`vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | Text encoder attachment | 0.04 | 1.03 | 0.87 |
| [`vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | Text encode | 9.79 | 2.24 | 18.93 |
| [`vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | Text encoder cleanup | 0.83 | 2.24 | 18.97 |
| [`vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | Transformer load | 0.14 | 2.24 | 3.67 |
| [`vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | Transformer attachment | 1.13 | 2.24 | 4.30 |
| [`vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | Denoise/readback | 52.82 | 6.63 | 17.69 |
| [`vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | VAE load | 0.85 | 2.24 | 5.61 |
| [`vmm_fast` / leaf VAE + trim](qwen-image-21-details.md#windows-1024-native) | VAE decode | 1.89 | 3.93 | 4.55 |

## Windows SDNQ INT4

Base 512×512, 20 steps, seed 42; BF16 auxiliaries, resident stock VAE. Output parity is within the INT4 pair, not a BF16 quality-equivalence claim.

| Platform / Scenario (details) | Transformer runtime | VAE route | Denoise/readback (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Windows / SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) (reference) | official leaf hooks | resident | 24.26 | 1.21 | 39.61 | 3.68 | 19.02 |
| [Windows / SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | sdnq_runtime | resident | 18.81&nbsp;🟢&nbsp;↓ | 0.94&nbsp;🟢&nbsp;↓ | 36.62&nbsp;🟢&nbsp;↓ | 3.68 | 19.02 |

### Phase measurements

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) | Text encoder load | 0.92 | 1.03 | 0.92 |
| [SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) | Text encoder attachment | 0.04 | 1.03 | 0.92 |
| [SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) | Text encode | 9.44 | 2.28 | 18.98 |
| [SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) | Text encoder cleanup | 0.68 | 2.28 | 19.02 |
| [SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) | Transformer load | 0.56 | 1.10 | 3.71 |
| [SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) | Transformer attachment | 0.02 | 1.10 | 3.71 |
| [SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) | Denoise/readback | 24.26 | 2.00 | 7.60 |
| [SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) | VAE load | 0.93 | 1.10 | 5.43 |
| [SDNQ / official leaf](qwen-image-21-details.md#int4-leaf) | VAE decode | 0.44 | 3.54 | 4.46 |
| [SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | Text encoder load | 0.92 | 1.03 | 0.92 |
| [SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | Text encoder attachment | 0.03 | 1.03 | 0.92 |
| [SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | Text encode | 9.55 | 2.28 | 18.98 |
| [SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | Text encoder cleanup | 0.67 | 2.28 | 19.02 |
| [SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | Transformer load | 0.54 | 1.10 | 3.71 |
| [SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | Transformer attachment | 2.64 | 2.15 | 10.85 |
| [SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | Denoise/readback | 18.81 | 2.87 | 7.19 |
| [SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | VAE load | 0.97 | 1.18 | 9.00 |
| [SDNQ / `one_shot_fast`](qwen-image-21-details.md#int4-ddo) | VAE decode | 0.44 | 3.68 | 7.93 |

## Recommendations

| Hardware / priority | Recommendation from these measurements |
| --- | --- |
| 8 GiB GPU, ample RAM, speed | `vmm_fast` or `one_shot_fast`: both shorten 512-square denoise versus official leaf, at higher GPU residency. Native has the shortest tracked total in each platform set. |
| Lower GPU usage, slower generation acceptable | Official leaf; Windows `low_ram_safe` is an intermediate speed/GPU tradeoff, not the lowest measured global RSS. |
| Decode approaches GPU capacity | Use leaf VAE; explicit trim is a separate choice. The 1024-square cases do not isolate allocator effects. |
| Neither RAM nor GPU headroom is sufficient for decode | Enable supported VAE tiling and check quality; changing the transformer preset does not remove VAE activations. |
| Saved INT4 acceptable | The measured SDNQ DDO pair is faster than its INT4 leaf control; it is not evidence of BF16-equivalent quality or native VMM quantized support. |
| WSL | Explicit-cap linear uses less GPU and has slightly shorter denoise than native; native has the shorter tracked total. |

The common text encoder dominates global RSS in several cases. Around 19–22 GiB
process RSS does not establish fit on a 16 GiB RAM computer.

Single observations, not statistical or arbitrary-model guarantees. Missing
measurements are unavailable, not zero; phase peaks must not be summed.
See [presets](../user/presets.md), [reproduction](reproduction.md) and
[measurement details](qwen-image-21-details.md) before comparing different recipes.
