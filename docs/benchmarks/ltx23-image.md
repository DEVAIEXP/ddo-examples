# LTX 2.3 image benchmarks

[Test hardware and platform limits](hardware.md).

**Leaf control:** Windows and native Linux use Diffusers leaf offloading with
`use_stream=True`. WSL uses `use_stream=False` because streamed leaf offloading
was incompatible with the tested WSL environment; this is not a limitation
established for every WSL setup. This setting concerns weight offloading, not
video VAE decode streaming.

Distilled BF16, 1280×704, 8 steps, seed 43, real prompt, CFG 1/rescale 0.7, no PAG. Shared official leaf text encoder and resident untiled VAE in the primary comparison.

Presets resolve to the executed [transformer runtimes](runtimes.md); the VAE route
is independent. Scenario names link directly to [measurement details](ltx23-image-details.md).
Times are seconds; GPU is sampled **absolute driver usage**, RAM is process **RSS**,
and memory units are **GiB**. Average is denoise wall divided by steps, including
warmup, not completed warm GPU timing. Tracked total excludes command startup and
downloads. [Metric definitions](glossary.md).

In comparable overview tables, official Diffusers leaf is the first row and the
reference for that table: 🟢 ↓ is lower, 🔴 ↑ is higher. Equal displayed values
have no arrow. Arrows describe measurements, not an overall preset ranking;
higher memory can accompany faster execution. Phase tables and separate workloads
without a leaf control have no arrows. Rows with different driver/cache sessions
or a reused non-side-by-side observation also have no arrows.

## Windows — BF16

| Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) (reference) | official leaf hooks | resident | 39.03 | 4.88 | 69.97 | 7.15 | 28.77 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | linear_runtime | resident | 11.99&nbsp;🟢&nbsp;↓ | 1.50&nbsp;🟢&nbsp;↓ | 54.33&nbsp;🟢&nbsp;↓ | 7.43&nbsp;🔴&nbsp;↑ | 27.12&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | linear_runtime | resident | 37.46&nbsp;🟢&nbsp;↓ | 4.68&nbsp;🟢&nbsp;↓ | 69.98&nbsp;🔴&nbsp;↑ | 7.18&nbsp;🔴&nbsp;↑ | 28.20&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | vmm_fast_runtime | resident | 15.15&nbsp;🟢&nbsp;↓ | 1.89&nbsp;🟢&nbsp;↓ | 48.73&nbsp;🟢&nbsp;↓ | 7.27&nbsp;🔴&nbsp;↑ | 29.53&nbsp;🔴&nbsp;↑ |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | vmm_runtime | resident | 30.28&nbsp;🟢&nbsp;↓ | 3.79&nbsp;🟢&nbsp;↓ | 60.50&nbsp;🟢&nbsp;↓ | 7.24&nbsp;🔴&nbsp;↑ | 26.02&nbsp;🟢&nbsp;↓ |

### Phase measurements

The encoder phase includes load/encode/cleanup; connectors are inside the
**generate aggregate**, which also includes transformer setup and denoise.
Do not add this aggregate to the denoise column above.

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | Text encoder phase (load/encode/cleanup) | 21.15 | 4.79 | 25.90 |
| [Official leaf](ltx23-image-details.md#windows-leaf) | Generate aggregate (connectors/setup/denoise) | 45.06 | 7.15 | 28.77 |
| [Official leaf](ltx23-image-details.md#windows-leaf) | VAE load | 0.98 | 2.39 | 5.59 |
| [Official leaf](ltx23-image-details.md#windows-leaf) | VAE decode | 0.32 | 2.80 | 4.39 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | Text encoder phase (load/encode/cleanup) | 21.00 | 4.79 | 25.90 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | Generate aggregate (connectors/setup/denoise) | 30.53 | 7.43 | 27.12 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | VAE load | 1.16 | 2.48 | 23.47 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | VAE decode | 0.30 | 2.80 | 22.17 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | Text encoder phase (load/encode/cleanup) | 21.33 | 4.79 | 25.90 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | Generate aggregate (connectors/setup/denoise) | 45.04 | 7.18 | 28.20 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | VAE load | 0.98 | 2.49 | 5.02 |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | VAE decode | 0.31 | 2.80 | 3.80 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | Text encoder phase (load/encode/cleanup) | 21.70 | 4.85 | 25.91 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | Generate aggregate (connectors/setup/denoise) | 23.54 | 7.27 | 29.53 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | VAE load | 1.14 | 3.44 | 5.12 |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | VAE decode | 0.39 | 3.44 | 3.81 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | Text encoder phase (load/encode/cleanup) | 21.44 | 4.79 | 26.02 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | Generate aggregate (connectors/setup/denoise) | 37.33 | 7.24 | 8.28 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | VAE load | 1.03 | 2.48 | 7.11 |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | VAE decode | 0.31 | 2.80 | 5.81 |

## Linux — BF16

The NVMe observation uses a different driver/cache session from the route trio;
it is not a controlled allocator-only A/B.

| Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | official leaf hooks | resident | 28.60 | 3.57 | 45.55 | 6.93 | 29.79 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | linear_runtime | resident | 12.79&nbsp;🟢&nbsp;↓ | 1.60&nbsp;🟢&nbsp;↓ | 43.29&nbsp;🟢&nbsp;↓ | 7.06&nbsp;🔴&nbsp;↑ | 26.97&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | vmm_fast_runtime | resident | 11.31&nbsp;🟢&nbsp;↓ | 1.41&nbsp;🟢&nbsp;↓ | 28.00&nbsp;🟢&nbsp;↓ | 6.90&nbsp;🟢&nbsp;↓ | 29.48&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](ltx23-image-details.md#linux-nvme) | vmm_runtime | resident | 134.89 | 16.86 | 154.75 | 6.89 | 30.31 |

### Phase measurements

The encoder phase includes load/encode/cleanup; connectors are inside the
**generate aggregate**, which also includes transformer setup and denoise.
Do not add this aggregate to the denoise column above.

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) | Text encoder phase (load/encode/cleanup) | 11.37 | 4.55 | 25.96 |
| [Official leaf](ltx23-image-details.md#linux-leaf) | Generate aggregate (connectors/setup/denoise) | 32.58 | 6.93 | 28.61 |
| [Official leaf](ltx23-image-details.md#linux-leaf) | VAE load | 0.83 | 2.38 | 29.79 |
| [Official leaf](ltx23-image-details.md#linux-leaf) | VAE decode | 0.34 | 2.64 | 28.68 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | Text encoder phase (load/encode/cleanup) | 11.40 | 4.38 | 26.03 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | Generate aggregate (connectors/setup/denoise) | 30.54 | 7.06 | 26.97 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | VAE load | 0.67 | 2.19 | 23.34 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | VAE decode | 0.33 | 2.51 | 22.13 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | Text encoder phase (load/encode/cleanup) | 10.95 | 4.63 | 25.94 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | Generate aggregate (connectors/setup/denoise) | 16.20 | 6.90 | 29.48 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | VAE load | 0.26 | 3.21 | 4.94 |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | VAE decode | 0.26 | 3.21 | 3.80 |
| [`nvme_baseline`](ltx23-image-details.md#linux-nvme) | Text encoder phase (load/encode/cleanup) | 11.64 | 4.51 | 26.04 |
| [`nvme_baseline`](ltx23-image-details.md#linux-nvme) | Generate aggregate (connectors/setup/denoise) | 141.70 | 6.89 | 30.31 |
| [`nvme_baseline`](ltx23-image-details.md#linux-nvme) | VAE load | 0.70 | 2.24 | 7.06 |
| [`nvme_baseline`](ltx23-image-details.md#linux-nvme) | VAE decode | 0.33 | 2.61 | 5.90 |

## WSL — BF16

Leaf controls are non-streamed; linear uses an explicit 3 GiB override and
native uses bounded registration. These are not Windows/Linux parity claims.

| Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | official leaf hooks | resident | 260.56 | 32.57 | 435.10 | 7.29 | 38.53 |
| [`wsl_compat` (3 GiB override)](ltx23-image-details.md#wsl-linear) | linear_runtime | resident | 149.73&nbsp;🟢&nbsp;↓ | 18.72&nbsp;🟢&nbsp;↓ | 335.39&nbsp;🟢&nbsp;↓ | 7.23&nbsp;🟢&nbsp;↓ | 38.53 |
| [`vmm_fast` (bounded)](ltx23-image-details.md#wsl-native) | vmm_fast_runtime | resident | 95.36&nbsp;🟢&nbsp;↓ | 11.92&nbsp;🟢&nbsp;↓ | 268.93&nbsp;🟢&nbsp;↓ | 7.17&nbsp;🟢&nbsp;↓ | 38.53 |

### Phase measurements

The encoder phase includes load/encode/cleanup; connectors are inside the
**generate aggregate**, which also includes transformer setup and denoise.
Do not add this aggregate to the denoise column above.

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) | Text encoder phase (load/encode/cleanup) | 139.13 | 4.85 | 38.53 |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) | Generate aggregate (connectors/setup/denoise) | 287.29 | 7.29 | 32.11 |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) | VAE load | 7.36 | 2.44 | 7.00 |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) | VAE decode | 0.93 | 2.70 | 5.69 |
| [`wsl_compat` (3 GiB override)](ltx23-image-details.md#wsl-linear) | Text encoder phase (load/encode/cleanup) | 134.60 | 4.85 | 38.53 |
| [`wsl_compat` (3 GiB override)](ltx23-image-details.md#wsl-linear) | Generate aggregate (connectors/setup/denoise) | 192.92 | 7.23 | 30.10 |
| [`wsl_compat` (3 GiB override)](ltx23-image-details.md#wsl-linear) | VAE load | 6.85 | 2.44 | 6.99 |
| [`wsl_compat` (3 GiB override)](ltx23-image-details.md#wsl-linear) | VAE decode | 0.63 | 2.70 | 5.69 |
| [`vmm_fast` (bounded)](ltx23-image-details.md#wsl-native) | Text encoder phase (load/encode/cleanup) | 135.09 | 4.70 | 38.53 |
| [`vmm_fast` (bounded)](ltx23-image-details.md#wsl-native) | Generate aggregate (connectors/setup/denoise) | 125.32 | 7.17 | 30.29 |
| [`vmm_fast` (bounded)](ltx23-image-details.md#wsl-native) | VAE load | 7.48 | 2.46 | 7.00 |
| [`vmm_fast` (bounded)](ltx23-image-details.md#wsl-native) | VAE decode | 0.62 | 2.78 | 5.70 |

## VAE strategies — separate native-encoder workload

1280×704, 8 steps, seed 43. These pairs use a native text encoder, unlike the shared-leaf-encoder comparisons above.

| Platform / Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Windows / `vmm_fast` / resident VAE](ltx23-image-details.md#vae-windows-resident) | vmm_fast_runtime | resident | 13.99 | 1.75 | 39.18 | 7.15 | 27.37 |
| [Windows / `vmm_fast` / leaf VAE](ltx23-image-details.md#vae-windows-leaf) | vmm_fast_runtime | leaf | 14.37 | 1.80 | 40.34 | 7.17 | 27.37 |
| [Linux / `vmm_fast` / resident VAE](ltx23-image-details.md#vae-linux-resident) | vmm_fast_runtime | resident | 17.12 | 2.14 | 40.30 | 6.56 | 27.33 |
| [Linux / `vmm_fast` / leaf VAE](ltx23-image-details.md#vae-linux-leaf) | vmm_fast_runtime | leaf | 17.03 | 2.13 | 37.65 | 6.53 | 27.36 |

### Phase measurements

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [`vmm_fast` / resident VAE](ltx23-image-details.md#vae-windows-resident) | Text encoder phase (load/encode/cleanup) | 14.29 | 6.79 | 25.12 |
| [`vmm_fast` / resident VAE](ltx23-image-details.md#vae-windows-resident) | Generate aggregate (connectors/setup/denoise) | 21.72 | 7.15 | 27.37 |
| [`vmm_fast` / resident VAE](ltx23-image-details.md#vae-windows-resident) | VAE load | 1.03 | 2.88 | 2.86 |
| [`vmm_fast` / resident VAE](ltx23-image-details.md#vae-windows-resident) | VAE decode | 0.33 | 3.19 | 1.65 |
| [`vmm_fast` / leaf VAE](ltx23-image-details.md#vae-windows-leaf) | Text encoder phase (load/encode/cleanup) | 14.65 | 6.81 | 25.12 |
| [`vmm_fast` / leaf VAE](ltx23-image-details.md#vae-windows-leaf) | Generate aggregate (connectors/setup/denoise) | 22.28 | 7.17 | 27.37 |
| [`vmm_fast` / leaf VAE](ltx23-image-details.md#vae-windows-leaf) | VAE load | 0.06 | 1.68 | 1.61 |
| [`vmm_fast` / leaf VAE](ltx23-image-details.md#vae-windows-leaf) | VAE decode | 1.12 | 2.23 | 3.17 |
| [`vmm_fast` / resident VAE](ltx23-image-details.md#vae-linux-resident) | Text encoder phase (load/encode/cleanup) | 15.63 | 6.26 | 25.14 |
| [`vmm_fast` / resident VAE](ltx23-image-details.md#vae-linux-resident) | Generate aggregate (connectors/setup/denoise) | 23.29 | 6.56 | 27.33 |
| [`vmm_fast` / resident VAE](ltx23-image-details.md#vae-linux-resident) | VAE load | 0.66 | 2.44 | 2.90 |
| [`vmm_fast` / resident VAE](ltx23-image-details.md#vae-linux-resident) | VAE decode | 0.34 | 2.76 | 1.65 |
| [`vmm_fast` / leaf VAE](ltx23-image-details.md#vae-linux-leaf) | Text encoder phase (load/encode/cleanup) | 14.97 | 6.23 | 25.13 |
| [`vmm_fast` / leaf VAE](ltx23-image-details.md#vae-linux-leaf) | Generate aggregate (connectors/setup/denoise) | 21.51 | 6.53 | 27.36 |
| [`vmm_fast` / leaf VAE](ltx23-image-details.md#vae-linux-leaf) | VAE load | 0.02 | 1.27 | 1.60 |
| [`vmm_fast` / leaf VAE](ltx23-image-details.md#vae-linux-leaf) | VAE decode | 0.75 | 1.82 | 3.19 |

### Completed CUDA timing — native-encoder VAE pairs only

These completed event durations are not interchangeable with host wall averages.

| Scenario (details) | CUDA phase (s) | CUDA warm, steps 2–8 (s/step) |
| --- | ---: | ---: |
| [Windows / resident VAE](ltx23-image-details.md#vae-windows-resident) | 14.64 | 1.37 |
| [Windows / leaf VAE](ltx23-image-details.md#vae-windows-leaf) | 15.01 | 1.37 |
| [Linux / resident VAE](ltx23-image-details.md#vae-linux-resident) | 17.40 | 1.06 |
| [Linux / leaf VAE](ltx23-image-details.md#vae-linux-leaf) | 17.30 | 1.06 |

## Windows SDNQ INT4

Saved INT4 transformer, 1280×704, 8 steps, seed 43; resident VAE. Compare within this pair, not as an isolated BF16-versus-INT4 speedup.

| Platform / Scenario (details) | Transformer runtime | VAE route | Denoise (s) | Average (s/step) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Windows / SDNQ / official leaf](ltx23-image-details.md#int4-leaf) (reference) | official leaf hooks | resident | 21.54 | 2.69 | 50.05 | 7.10 | 25.94 |
| [Windows / SDNQ / `one_shot_fast`](ltx23-image-details.md#int4-ddo) | sdnq_runtime | resident | 11.74&nbsp;🟢&nbsp;↓ | 1.47&nbsp;🟢&nbsp;↓ | 46.56&nbsp;🟢&nbsp;↓ | 7.10 | 25.99&nbsp;🔴&nbsp;↑ |

### Phase measurements

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [SDNQ / official leaf](ltx23-image-details.md#int4-leaf) | Text encoder phase (load/encode/cleanup) | 18.20 | 4.79 | 25.94 |
| [SDNQ / official leaf](ltx23-image-details.md#int4-leaf) | Generate aggregate (connectors/setup/denoise) | 30.38 | 7.10 | 11.85 |
| [SDNQ / official leaf](ltx23-image-details.md#int4-leaf) | VAE aggregate (load/decode/cleanup) | 1.40 | 2.80 | 13.20 |
| [SDNQ / `one_shot_fast`](ltx23-image-details.md#int4-ddo) | Text encoder phase (load/encode/cleanup) | 18.50 | 4.79 | 25.99 |
| [SDNQ / `one_shot_fast`](ltx23-image-details.md#int4-ddo) | Generate aggregate (connectors/setup/denoise) | 26.59 | 7.10 | 17.52 |
| [SDNQ / `one_shot_fast`](ltx23-image-details.md#int4-ddo) | VAE aggregate (load/decode/cleanup) | 1.40 | 4.97 | 11.44 |

## Windows `warm_process` — two generations in one process

1280×704 BF16, 8 steps per generation, initial seed 43; shared leaf encoder,
resident VAE. This is a repeated workload, not a single-image comparison.

| Measurement | First (s) | Second (s) |
| --- | ---: | ---: |
| [Transformer plan/attachment event](ltx23-image-details.md#windows-warm-process) | 11.30 | 4.69 |
| [Denoise-call wall](ltx23-image-details.md#windows-warm-process) | 12.18 | 11.94 |
| [Prepare through completed CPU latent readback](ltx23-image-details.md#windows-warm-process) | 13.66 | 13.39 |

| Scenario (details) | Tracked total for both generations (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`warm_process`](ltx23-image-details.md#windows-warm-process) | linear_runtime | 75.11 | 7.52 | 28.01 |

| Scenario (details) | Phase | Wall (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [`warm_process`](ltx23-image-details.md#windows-warm-process) | Text encoder phase | 21.76 | 4.79 | 25.94 |
| [`warm_process`](ltx23-image-details.md#windows-warm-process) | Generate aggregate (both preparations/generations) | 50.78 | 7.52 | 28.01 |
| [`warm_process`](ltx23-image-details.md#windows-warm-process) | VAE decode (final generation) | 0.30 | 2.80 | 22.17 |

## Recommendations

| Hardware / priority | Recommendation from these measurements |
| --- | --- |
| 8 GiB GPU, ample system RAM, one image | Compare `one_shot_fast` and `vmm_fast`. On Windows, `one_shot_fast` has shorter denoise time, while `vmm_fast` has the shortest tracked total in the shared-encoder comparison. On Linux, `vmm_fast` has both the shortest denoise time and tracked total. |
| Lower transformer-phase RAM priority | Inspect `nvme_baseline` phase peaks; its file-backed route can trade memory retention for transfers. Text encoding still reaches about 26 GiB RSS on Windows. |
| Lower decode GPU pressure | Leaf VAE lowers decode GPU peaks in the separate native-encoder pair, at extra decode time/RAM. |
| Neither RAM nor GPU headroom is sufficient for decode | Use supported VAE tiling and verify output quality; weight offloading alone does not eliminate activations. |
| WSL | Use the bounded native recipe or explicit-cap `wsl_compat` recipe. Do not copy Windows source-registration ceilings. |
| Repeated preparations in one process | Consider the separately measured `warm_process` case; its two-generation total is not a single-image total. |

`low_ram_safe` did not have the lowest global RSS in the Windows comparison.
No selected row certifies a low-system-RAM machine.

Single observations, not statistical or arbitrary-model guarantees. Missing
measurements are unavailable, not zero; phase peaks must not be summed.
See [presets](../user/presets.md), [reproduction](reproduction.md) and
[measurement details](ltx23-image-details.md) before comparing different recipes.
