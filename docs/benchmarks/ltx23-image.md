# LTX 2.3 image benchmarks

[Test hardware and platform limits](hardware.md).

**Leaf control:** Windows and native Linux use Diffusers leaf offloading with
`use_stream=True`. WSL uses `use_stream=False` because streamed leaf offloading
was incompatible with the tested WSL environment; this is not a limitation
established for every WSL setup. This setting concerns weight offloading, not
video VAE decode streaming.

Distilled BF16, 1280×704, 8 steps, seed 43, real prompt, CFG 1/rescale 0.7, no PAG. Each platform table lists the measured VAE route.

Presets resolve to the executed [transformer runtimes](../user/presets.md#transformer-runtimes); the VAE route
is independent. Preset labels link directly to [measurement details](ltx23-image-details.md).
Times are seconds; GPU is sampled **absolute driver usage**, RAM is process **RSS**,
and memory units are **GiB**. Denoise is wall latency, including warmup,
not completed warm GPU timing. Tracked total excludes command startup and
downloads. **Before reading the tables, see the [labels and memory glossary](glossary.md).**

Overview arrows compare each row with the first row of its table. 🟢 ↓ marks
a lower value and 🔴 ↑ marks a higher value. Equal displayed values have no
arrow. The arrows show observed differences. Phase tables and separate
workloads without a control omit arrows.


**Full run** includes all tracked phases: component loading and setup, text
encoding, denoising, VAE decoding, cleanup, conversion and PNG saving. It excludes
calibration, command startup and downloads. Peaks cover the full tracked run.

**Denoising** reports the recorded sampling interval, including weight transfers
and scheduler work, rather than an isolated transformer-forward kernel timer.
The Leaf, linear and VMM Fast runtime rows monitor denoising independently inside
the generation aggregate, excluding connector setup and transformer cleanup.

**VAE decode** excludes VAE loading, attachment, cleanup and PNG saving. Conversion
performed inside the recorded decode call remains included. No phase is
estimated by subtraction and no aggregate or full-run peak is substituted for a
narrower phase peak.

Average divides denoise wall latency by the recipe steps, retaining the same
warmup/readback scope. Detailed setup and output phases are available in the
linked phase measurements; nested intervals must not be added to their aggregates.

## Windows — BF16

The Windows RC5 runs use default SDPA and NVIDIA driver 617.42. The
`vmm_fast` and NVMe observations use standby purges at startup and before the
transformer. The other rows use no standby purge. The selected Leaf and linear PNGs are byte-identical. The rows come from separate sessions and
include the component routes shown below.

#### Full run

| Preset | Transformer runtime | Text encoder route | VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | official leaf hooks | Leaf | Leaf | 76.40 | 7.12 | 28.76 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | linear_runtime | Leaf | Leaf | 48.55&nbsp;🟢&nbsp;↓ | 7.41&nbsp;🔴&nbsp;↑ | 27.20&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | linear_runtime | Leaf | Leaf | 82.56&nbsp;🔴&nbsp;↑ | 7.08&nbsp;🟢&nbsp;↓ | 47.50&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | vmm_fast_runtime | Native VMM | Native VMM | 37.02&nbsp;🟢&nbsp;↓ | 7.69&nbsp;🔴&nbsp;↑ | 27.38&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | vmm_runtime | Leaf | Leaf | 69.54&nbsp;🟢&nbsp;↓ | 7.10&nbsp;🟢&nbsp;↓ | 25.90&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 47.47 | 5.93 | 1.60 | 28.76 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 13.64&nbsp;🟢&nbsp;↓ | 1.70&nbsp;🟢&nbsp;↓ | 7.41&nbsp;🔴&nbsp;↑ | 22.13&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 39.38&nbsp;🟢&nbsp;↓ | 4.92&nbsp;🟢&nbsp;↓ | 5.83&nbsp;🔴&nbsp;↑ | 47.50&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 12.30&nbsp;🟢&nbsp;↓ | 1.54&nbsp;🟢&nbsp;↓ | 7.69&nbsp;🔴&nbsp;↑ | 27.38&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 33.91&nbsp;🟢&nbsp;↓ | 4.24&nbsp;🟢&nbsp;↓ | 2.69&nbsp;🔴&nbsp;↑ | 6.20&nbsp;🟢&nbsp;↓ |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#windows-leaf) | 1.09 | 1.67 | 5.88 |
| [`one_shot_fast`](ltx23-image-details.md#windows-linear) | 1.17&nbsp;🔴&nbsp;↑ | 1.67 | 23.66&nbsp;🔴&nbsp;↑ |
| [`low_ram_safe`](ltx23-image-details.md#windows-low-ram) | 0.96&nbsp;🟢&nbsp;↓ | 2.94&nbsp;🔴&nbsp;↑ | 24.04&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-image-details.md#windows-native) | 0.68&nbsp;🟢&nbsp;↓ | 5.83&nbsp;🔴&nbsp;↑ | 4.52&nbsp;🟢&nbsp;↓ |
| [`nvme_baseline`](ltx23-image-details.md#windows-nvme) | 1.02&nbsp;🟢&nbsp;↓ | 1.67 | 7.33&nbsp;🔴&nbsp;↑ |

### Transformer self-attention on VMM Fast

The SDPA control is the runtime measurement above, using the final 2.1921 GiB
transformer residency profile, native encoder/VAE and both standby purges.
Only transformer self-attention changes; masked cross-attention retains default
SDPA. Separate diagnostics confirmed both operators; timed measurements have
the profiler off.

#### Full run

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 37.02 | 7.69 | 27.38 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 37.11&nbsp;🔴&nbsp;↑ | 7.74&nbsp;🔴&nbsp;↑ | 27.40&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 36.80&nbsp;🟢&nbsp;↓ | 7.69 | 27.39&nbsp;🔴&nbsp;↑ |

#### Denoising

| Self-attention method | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 12.30 | 1.54 | 7.69 | 27.38 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 12.36&nbsp;🔴&nbsp;↑ | 1.55&nbsp;🔴&nbsp;↑ | 7.74&nbsp;🔴&nbsp;↑ | 27.40&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 12.35&nbsp;🔴&nbsp;↑ | 1.54 | 7.69 | 27.39&nbsp;🔴&nbsp;↑ |

#### VAE decode

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#windows-native) (reference) | 0.68 | 5.83 | 4.52 |
| [cuDNN SDPA](ltx23-image-details.md#windows-self-attention) | 0.66&nbsp;🟢&nbsp;↓ | 5.83 | 4.54&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-image-details.md#windows-self-attention) | 0.67&nbsp;🟢&nbsp;↓ | 5.83 | 4.57&nbsp;🔴&nbsp;↑ |


Denoise times differed by at most 0.06 s. FlashAttention 2 had the shortest
tracked total in these single observations, by 0.22 s against SDPA; this does
not establish a repeatable speedup. Both alternate backends changed RGB pixels.
Visual quality remains unassessed.

### `low_ram_safe` — pre-VAE purge

This Windows pair uses RC5, NVIDIA driver 617.42, default SDPA, and Leaf
offload for the encoder and VAE. Both runs apply the same 4.04 GiB transformer
residency profile. See the [paired recipe](ltx23-image-details.md#windows-low-ram-pre-vae-purge).

#### Full run

| Preset | Transformer runtime | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [`low_ram_safe`, no purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | linear_runtime | 82.56 | 7.08 | 47.50 |
| [`low_ram_safe`, pre-VAE purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | linear_runtime | 87.36&nbsp;🔴&nbsp;↑ | 7.11&nbsp;🔴&nbsp;↑ | 47.51&nbsp;🔴&nbsp;↑ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [`low_ram_safe`, no purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 39.38 | 4.92 | 5.83 | 47.50 |
| [`low_ram_safe`, pre-VAE purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 39.37&nbsp;🟢&nbsp;↓ | 4.92 | 5.83 | 47.51&nbsp;🔴&nbsp;↑ |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`low_ram_safe`, no purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 0.96 | 2.94 | 24.04 |
| [`low_ram_safe`, pre-VAE purge](ltx23-image-details.md#windows-low-ram-pre-vae-purge) | 1.60&nbsp;🔴&nbsp;↑ | 2.94 | 24.04 |

The pre-VAE purge took 2.51 s and is included in the tracked total. It did
not improve this observation: decode took 1.60 s with purge versus 0.96 s
without it, and the full run took 87.36 s versus 82.56 s. Denoise and memory
peaks were nearly unchanged. Both PNGs were byte-identical. These single
observations support leaving purge disabled for this recipe; they do not
establish a general effect under other host-memory conditions.

See [phase measurements](ltx23-image-phases.md#windows-bf16) for this workload.

## Linux — BF16

The RC5 runs use NVIDIA driver 615.71.09, Torch 2.12.1+cu132, Diffusers
0.41.0 and default SDPA. Leaf and linear cases use Leaf for the encoder
and VAE. VMM Fast uses native VMM for both. Standby purge is disabled.
The output PNGs were byte-identical.

#### Full run

| Preset | Transformer runtime | Text encoder route | VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | official leaf hooks | Leaf | Leaf | 47.55 | 6.60 | 28.93 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | linear_runtime | Leaf | Leaf | 38.77&nbsp;🟢&nbsp;↓ | 6.91&nbsp;🔴&nbsp;↑ | 26.86&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | linear_runtime | Leaf | Leaf | 56.34&nbsp;🔴&nbsp;↑ | 6.85&nbsp;🔴&nbsp;↑ | 45.42&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | vmm_fast_runtime | Native VMM | Native VMM | 31.44&nbsp;🟢&nbsp;↓ | 6.58&nbsp;🟢&nbsp;↓ | 27.57&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | 30.85 | 3.86 | 1.19 | 28.92 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | 13.00&nbsp;🟢&nbsp;↓ | 1.62&nbsp;🟢&nbsp;↓ | 6.91&nbsp;🔴&nbsp;↑ | 22.32&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | 21.35&nbsp;🟢&nbsp;↓ | 2.67&nbsp;🟢&nbsp;↓ | 6.85&nbsp;🔴&nbsp;↑ | 26.24&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | 11.86&nbsp;🟢&nbsp;↓ | 1.48&nbsp;🟢&nbsp;↓ | 5.76&nbsp;🔴&nbsp;↑ | 27.57&nbsp;🟢&nbsp;↓ |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf](ltx23-image-details.md#linux-leaf) (reference) | 1.01 | 1.17 | 6.11 |
| [`one_shot_fast`](ltx23-image-details.md#linux-linear) | 0.96&nbsp;🟢&nbsp;↓ | 1.21&nbsp;🔴&nbsp;↑ | 23.88&nbsp;🔴&nbsp;↑ |
| [`low_ram_safe`](ltx23-image-details.md#linux-low-ram) | 1.04&nbsp;🔴&nbsp;↑ | 1.13&nbsp;🟢&nbsp;↓ | 3.38&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](ltx23-image-details.md#linux-native) | 0.52&nbsp;🟢&nbsp;↓ | 5.44&nbsp;🔴&nbsp;↑ | 4.75&nbsp;🟢&nbsp;↓ |

See [phase measurements](ltx23-image-phases.md#linux-bf16) for this workload.

### Transformer self-attention on VMM Fast

All three runs use the same final 0.625 GiB transformer profile, native VMM
encoder and VAE, and NVIDIA driver 615.71.09. Only transformer
self-attention changes. Masked cross-attention retains default SDPA.
The cuDNN and FlashAttention 2 operators were verified in separate diagnostic runs.

#### Full run

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 31.44 | 6.58 | 27.57 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 30.15&nbsp;🟢&nbsp;↓ | 6.58 | 27.62&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 29.99&nbsp;🟢&nbsp;↓ | 6.57&nbsp;🟢&nbsp;↓ | 27.58&nbsp;🔴&nbsp;↑ |

#### Denoising

| Self-attention method | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 11.86 | 1.48 | 5.76 | 27.57 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 12.59&nbsp;🔴&nbsp;↑ | 1.57&nbsp;🔴&nbsp;↑ | 5.76 | 27.62&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 12.56&nbsp;🔴&nbsp;↑ | 1.57&nbsp;🔴&nbsp;↑ | 5.75&nbsp;🟢&nbsp;↓ | 27.58&nbsp;🔴&nbsp;↑ |

#### VAE decode

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#linux-native) (reference) | 0.52 | 5.44 | 4.75 |
| [cuDNN SDPA](ltx23-image-details.md#linux-native-attention) | 0.44&nbsp;🟢&nbsp;↓ | 5.44 | 4.76&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-image-details.md#linux-native-attention) | 0.46&nbsp;🟢&nbsp;↓ | 5.43&nbsp;🟢&nbsp;↓ | 4.76&nbsp;🔴&nbsp;↑ |

In these single measurements, cuDNN and FlashAttention 2 took 0.74 s and
0.70 s longer to denoise than default SDPA, despite lower full-run totals
by 1.29 s and 1.45 s. The total advantage occurred outside denoising; these
observations do not establish a transformer speedup. Both outputs differed
from the SDPA control. Visual quality remains unassessed.

## WSL — BF16

The runtime measurements use DDO and allocator RC5, Torch `2.12.1+cu132`,
Diffusers `0.41.0`, NVIDIA driver `617.42`, default SDPA, and no standby purge.
The encoder and VAE use Leaf in all three runs. The official Leaf transformer
uses no stream, the DDO runtimes use their applied profiles, and VMM Fast
limits source registration to 0.5 GiB. All three runs produced identical PNGs.

#### Full run

| Preset | Transformer runtime | Text encoder route | VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | official leaf hooks | Leaf | Leaf | 464.88 | 7.09 | 38.28 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | linear_runtime | Leaf | Leaf | 330.23&nbsp;🟢&nbsp;↓ | 7.42&nbsp;🔴&nbsp;↑ | 38.29&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) \* | vmm_fast_runtime | Leaf | Leaf | 286.57&nbsp;🟢&nbsp;↓ | 7.73&nbsp;🔴&nbsp;↑ | 37.57&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 285.19 | 35.65 | 1.41 | 32.22 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 119.13&nbsp;🟢&nbsp;↓ | 14.89&nbsp;🟢&nbsp;↓ | 7.42&nbsp;🔴&nbsp;↑ | 30.19&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) \* | 83.87&nbsp;🟢&nbsp;↓ | 10.48&nbsp;🟢&nbsp;↓ | 7.73&nbsp;🔴&nbsp;↑ | 30.47&nbsp;🟢&nbsp;↓ |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official leaf (no stream)](ltx23-image-details.md#wsl-leaf) (reference) | 4.43 | 1.55 | 6.51 |
| [`wsl_compat`](ltx23-image-details.md#wsl-linear) | 4.40&nbsp;🟢&nbsp;↓ | 1.55 | 6.54&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-image-details.md#wsl-native) \* | 5.24&nbsp;🔴&nbsp;↑ | 5.09&nbsp;🔴&nbsp;↑ | 6.97&nbsp;🔴&nbsp;↑ |

### Transformer self-attention on VMM Fast

All methods use the accepted default-SDPA VMM Fast control, its applied
2.1699 GiB profile and the same component routes. Only transformer
self-attention changes; masked cross-attention retains default SDPA.
Separate diagnostic runs confirmed the cuDNN and FlashAttention 2
operators. Timed measurements have the profiler disabled.

#### Full run

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 286.57 | 7.73 | 37.57 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 266.45&nbsp;🟢&nbsp;↓ | 7.74&nbsp;🔴&nbsp;↑ | 37.78&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 270.27&nbsp;🟢&nbsp;↓ | 7.76&nbsp;🔴&nbsp;↑ | 37.99&nbsp;🔴&nbsp;↑ |

#### Denoising

| Self-attention method | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 83.87 | 10.48 | 7.73 | 30.47 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 76.45&nbsp;🟢&nbsp;↓ | 9.56&nbsp;🟢&nbsp;↓ | 7.74&nbsp;🔴&nbsp;↑ | 30.51&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 79.86&nbsp;🟢&nbsp;↓ | 9.98&nbsp;🟢&nbsp;↓ | 7.76&nbsp;🔴&nbsp;↑ | 30.47 |

#### VAE decode

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-image-details.md#wsl-native) (reference) | 5.24 | 5.09 | 6.97 |
| [cuDNN SDPA](ltx23-image-details.md#wsl-native-attention) | 4.56&nbsp;🟢&nbsp;↓ | 5.09 | 6.98&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-image-details.md#wsl-native-attention) | 5.34&nbsp;🔴&nbsp;↑ | 5.09 | 6.97 |

cuDNN and FlashAttention 2 denoised in 76.45 s and 79.86 s versus
83.87 s for SDPA. These single sequential observations are sensitive
to WSL and Windows host activity; they do not establish a repeatable
speedup. Both variants changed output pixels; visual quality remains
unassessed.

See [phase measurements](ltx23-image-phases.md#wsl-bf16) for this workload.

\* `vmm_fast` uses explicit component budgets in this WSL recipe: the applied
profile selects 2.1699 GiB for resident transformer weights, staging totals
384 MiB, the CPU source cache is capped at 24.625 GiB, and CPU memory
registration for CUDA transfers is capped at 0.5 GiB. These component
budgets do not cap total VRAM usage: activations and other components also
consume memory. This is the same `vmm_fast` preset, not a separate preset.



<a id="windows-warm-process"></a>

## Windows `warm_process` — two generations in one process

The Windows RC5 `warm-process` case is measured separately from the single-image
runtime series. It performs two transformer preparations and two generations
in one process, then decodes only the final latent. See the
[reuse recipe](reproduction.md#windows-ltx-image-process-reuse).

| Preparation / generation | Transformer load (s) | Offload attachment (s) | Denoise (s) | Average (s/step) |
| --- | ---: | ---: | ---: | ---: |
| First | 0.25 | 10.77 | 13.41 | 1.68 |
| Second | 0.24 | 3.21 | 13.30 | 1.66 |

#### Full run

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`warm_process`, two preparations and two generations](ltx23-image-details.md#windows-warm-process) | 69.36 | 7.45 | 27.98 |

#### Denoising

| Generation | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| First | 13.41 | 1.68 | 7.45 | 22.13 |
| Second | 13.30 | 1.66 | 7.41 | 22.13 |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [`warm_process`, two preparations and two generations](ltx23-image-details.md#windows-warm-process) | 0.31 | 2.80 | 22.17 |

The second preparation recorded plan and pinned-weight cache hits. Its attachment
latency was 70.23% lower. The shared encoder runs once, and only the second latent
is decoded and saved. The total is for this combined workload, not one image or
two independent complete pipeline runs. Both generations use the final prepared
transformer. See [phase measurements](ltx23-image-phases.md#windows-warm-process).

<a id="windows-sdnq-int4"></a>

## Windows — saved SDNQ INT4

RC5 measurements use the same saved INT4 transformer with default SDPA,
1280×704, eight steps and seed 43. Both cases use the original BF16 Leaf
encoder and a resident BF16 VAE, with standby purges disabled. DDO preserves
the SDNQ forward and applies its own 4.8027 GiB capacity profile.

#### Full run

| Preset | Transformer runtime | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](ltx23-image-details.md#windows-sdnq-int4) (reference) | official Leaf hooks | 52.13 | 7.10 | 25.99 |
| [SDNQ INT4 DDO](ltx23-image-details.md#windows-sdnq-int4) | sdnq_runtime | 26.31&nbsp;🟢&nbsp;↓ | 7.08&nbsp;🟢&nbsp;↓ | 25.95&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](ltx23-image-details.md#windows-sdnq-int4) (reference) | 23.05 | 2.88 | 2.03 | 11.86 |
| [SDNQ INT4 DDO](ltx23-image-details.md#windows-sdnq-int4) | 10.79&nbsp;🟢&nbsp;↓ | 1.35&nbsp;🟢&nbsp;↓ | 7.07&nbsp;🔴&nbsp;↑ | 6.59&nbsp;🟢&nbsp;↓ |

#### VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [SDNQ INT4 Leaf](ltx23-image-details.md#windows-sdnq-int4) (reference) | 0.23 | 2.80 | 4.44 |
| [SDNQ INT4 DDO](ltx23-image-details.md#windows-sdnq-int4) | 0.22&nbsp;🟢&nbsp;↓ | 2.80 | 6.63&nbsp;🔴&nbsp;↑ |

The verified PNGs are byte-identical and pixel-identical within this pair.
DDO denoise took 10.79 s versus 23.05 s for Leaf. These are sequential
single observations: the unchanged Leaf prompt phase took 21.86 s in
the control and 8.10 s in the later DDO run. The tracked-total difference
therefore includes warm-state effects and cannot be attributed entirely
to transformer offloading. Calibration is excluded from inference.
This comparison does not assess INT4 quality against BF16. See the
[recipe](reproduction.md#windows-ltx-image-sdnq-int4) and
[phase measurements](ltx23-image-phases.md#windows-sdnq-int4).

## Recommendations

| Hardware / priority | Recommendation from these measurements |
| --- | --- |
| 8 GiB GPU, ample system RAM, one BF16 image | VMM Fast with native encoder and VAE has the shortest tracked total on Windows and Linux. On Linux RC5, its 31.44 s total is 33.9% below official Leaf and 18.9% below One Shot Fast. One Shot Fast has the lowest peak process RSS at 26.86 GiB. |
| Lower transformer-phase RAM priority | Inspect `nvme_baseline` phase peaks; its file-backed route can trade memory retention for transfers. Text encoding still reaches about 26 GiB RSS on Windows. |
| VAE route | Use the route recorded for each current runtime; this series does not establish an isolated resident-versus-Leaf VAE ranking. |
| Neither RAM nor GPU headroom is sufficient for decode | Use supported VAE tiling and verify output quality; weight offloading alone does not eliminate activations. |
| WSL | Use the VMM Fast recipe with the component budgets described above or the explicit-cap `wsl_compat` recipe. Do not copy Windows source-registration ceilings. |
| Repeated preparations in one process | `warm_process` reused the plan and pinned weights, reducing attachment from 9.97 s to 3.25 s. Its 68.33 s total covers two preparations, two generations and one final decode/save. |

`low_ram_safe` did not have the lowest global RSS in the Windows comparison.
No selected row certifies a low-system-RAM machine.

Single observations, not statistical or arbitrary-model guarantees. Phase peaks
must not be summed.
See [presets](../user/presets.md), [reproduction](reproduction.md) and
[measurement details](ltx23-image-details.md) before comparing different recipes.
