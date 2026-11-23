# LTX 2.3 video benchmarks

[Test hardware and platform limits](hardware.md).

**Leaf control:** Windows and native Linux use Diffusers leaf offloading with
`use_stream=True`. WSL uses `use_stream=False` because streamed leaf offloading
was incompatible with the tested WSL environment; this is not a limitation
established for every WSL setup. This setting concerns weight offloading, not
video VAE decode streaming.

Presets resolve to the executed [transformer runtimes](../user/presets.md#transformer-runtimes); the VAE route
is independent. Scenario names link directly to [measurement details](ltx23-video-details.md).
Times are seconds; GPU is sampled **absolute driver usage**, RAM is process **RSS**,
and memory units are **GiB**. Average is denoise wall divided by steps, including
warmup, not completed warm GPU timing. Tracked total excludes command startup and
downloads. **Before reading the tables, see the [labels and memory glossary](glossary.md).**

In comparable overview tables, 🟢 ↓ marks a lower value and 🔴 ↑ marks a higher
value than the control for that table. Equal displayed values
have no arrow. Arrows describe measurements, not an overall preset ranking;
higher memory can accompany faster execution. The comparison tables below use
the same markers in each measurement scope; detailed phase tables have no arrows.
Session differences are noted with each comparison.

**Full run** includes all tracked phases: loading, text encoding, conditioning,
denoising, video/audio decoding, output conversion, encoding and saving. GPU
and RSS peaks cover that same tracked interval. Profile calibration, process
startup and downloads are excluded; this is not end-to-end command latency.

**Denoising** reports the sampling loop, including transformer prediction,
weight transfers and scheduler/latent updates, rather than an isolated forward
kernel timer. Legacy linear/Leaf measurements also include transformer retirement
inside that phase; newer instrumentation records retirement separately. Historical
values are preserved and are not relabelled as the narrower interval.
**Video VAE decode**
uses only the recorded `decode_video` interval, excluding VAE loading/cleanup,
audio decoding and file saving. Output conversion performed inside that interval
remains included: these records are phase latency, not a pure VAE-kernel timer.
Each phase's memory columns cover its own interval. N/A means that an unambiguous
phase record was unavailable in the reviewed artifacts; global peaks and combined
decode-and-save timings are never substituted for missing phase measurements.

## Windows — BF16, 8-second pressure workload

For the separate cumulative optimization series on VMM Fast BF16/default SDPA,
see [model-feature observations](model-features.md#windows-observations).
Those measurements use their own no-feature control and do not replace this
runtime comparison.

LTX 2.3 Distilled BF16, 1280×704, 193 frames at 24 fps, 8 denoising steps,
seed 43, and guitar image and audio conditioning. The RTX 3060 Ti 8 GiB runs
used NVIDIA driver 617.42 and default PyTorch SDPA. The operator trace was off.
Chunking, RoPE reuse and fused RoPE, phase modulation, compact timesteps,
in-place residuals and compilation were disabled. All four runs used
`exact-streaming` video decode on CUDA.

#### Full run

| Preset | Transformer runtime | Text encoder route | Video VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | Official Leaf hooks | Leaf | Resident | 421.12 | 8.00 | 41.34 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | `linear_runtime` | Leaf | Leaf | 403.49&nbsp;🟢&nbsp;↓ | 8.00 | 50.35&nbsp;🔴&nbsp;↑ |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | `linear_runtime` | Leaf | Leaf | 482.05&nbsp;🔴&nbsp;↑ | 8.00 | 42.36&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | `vmm_fast_runtime` | Native VMM | Resident | 384.23&nbsp;🟢&nbsp;↓ | 8.00 | 33.08&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 363.33 | 45.42 | 8.00 | 41.34 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 335.02&nbsp;🟢&nbsp;↓ | 41.88&nbsp;🟢&nbsp;↓ | 8.00 | 50.35&nbsp;🔴&nbsp;↑ |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 419.30&nbsp;🔴&nbsp;↑ | 52.41&nbsp;🔴&nbsp;↑ | 8.00 | 42.36&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 321.35&nbsp;🟢&nbsp;↓ | 40.17&nbsp;🟢&nbsp;↓ | 8.00 | 33.08&nbsp;🟢&nbsp;↓ |

#### Video VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#windows-8-second-leaf) (reference) | 19.12 | 6.53 | 15.07 |
| [`one_shot_fast`](ltx23-video-details.md#windows-8-second-one-shot-fast) | 19.23&nbsp;🔴&nbsp;↑ | 6.57&nbsp;🔴&nbsp;↑ | 42.42&nbsp;🔴&nbsp;↑ |
| [`low_ram_safe`](ltx23-video-details.md#windows-8-second-low-ram-safe) | 18.87&nbsp;🟢&nbsp;↓ | 6.55&nbsp;🔴&nbsp;↑ | 15.58&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-video-details.md#windows-8-second-vmm-fast) | 18.79&nbsp;🟢&nbsp;↓ | 7.62&nbsp;🔴&nbsp;↑ | 15.58&nbsp;🔴&nbsp;↑ |

Leaf, One Shot Fast and Low RAM Safe exclude transformer cleanup from
denoise (0.19 s, 0.20 s and 0.20 s respectively). All three record video
decode separately from loading, cleanup, audio decode and saving.

Official Leaf ran without a standby purge. The DDO runs used a purge before the
transformer. The Leaf reference was measured in a separate post-reboot session;
One Shot Fast and Low RAM Safe each use a separate observation.
The separately measured `vmm_fast` row is the default SDPA control for the
Windows VMM Fast attention comparison. Output hashes and comparison scope are
recorded in the measurement details.

See [phase measurements](ltx23-video-phases.md#windows-bf16) for this workload.

<a id="windows-video-attention"></a>

### Transformer self-attention on VMM Fast

All four runs use the same 8-second workload. Default SDPA uses the VMM Fast
row from the runtime table. Sol shares its freshly calibrated profile and
environment with that control. The cuDNN and FlashAttention 2 measurements
use the same nominal budget from earlier sessions with different recorded
code/environment identities; their comparison is contextual. Separate
diagnostics verified backend dispatch outside the timed measurements.

#### Full run

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 384.23 | 8.00 | 33.08 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 334.92&nbsp;🟢&nbsp;↓ | 8.00 | 33.03&nbsp;🟢&nbsp;↓ |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 337.02&nbsp;🟢&nbsp;↓ | 8.00 | 33.13&nbsp;🔴&nbsp;↑ |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 273.37&nbsp;🟢&nbsp;↓ | 8.00 | 33.06&nbsp;🟢&nbsp;↓ |

#### Denoising

| Self-attention method | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 321.35 | 40.17 | 8.00 | 33.08 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 273.46&nbsp;🟢&nbsp;↓ | 34.18&nbsp;🟢&nbsp;↓ | 8.00 | 33.03&nbsp;🟢&nbsp;↓ |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 274.68&nbsp;🟢&nbsp;↓ | 34.34&nbsp;🟢&nbsp;↓ | 8.00 | 33.13&nbsp;🔴&nbsp;↑ |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 211.65&nbsp;🟢&nbsp;↓ | 26.46&nbsp;🟢&nbsp;↓ | 8.00 | 33.06&nbsp;🟢&nbsp;↓ |

#### Video VAE decode

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#windows-8-second-vmm-fast) | 18.79 | 7.62 | 15.58 |
| [cuDNN SDPA](ltx23-video-details.md#windows-native-attention) | 18.94&nbsp;🔴&nbsp;↑ | 7.62 | 15.84&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-video-details.md#windows-native-attention) | 18.95&nbsp;🔴&nbsp;↑ | 7.62 | 15.88&nbsp;🔴&nbsp;↑ |
| [Sol-Attn (sparse)](ltx23-video-details.md#windows-sol-attention) | 18.66&nbsp;🟢&nbsp;↓ | 7.62 | 16.03&nbsp;🔴&nbsp;↑ |

In one timed run per case, Sol reduced denoise time by 34.1% and tracked total
by 28.9% against its matched SDPA control. Sol is approximate, and its MP4
differs from SDPA; the user accepted its visual quality. The cuDNN and
FlashAttention 2 totals differed by 2.10 s, with different MP4 hashes from
their SDPA reference. See the [Sol phase measurements](ltx23-video-phases.md#windows-sol-attention).

<a id="linux-bf16"></a>

## Linux — BF16, 8-second pressure workload

LTX 2.3 Distilled BF16, 1280×704, 193 frames at 24 fps, 8 denoising steps,
seed 43, and guitar image and audio conditioning. The RTX 3060 Ti 8 GiB ran
with NVIDIA driver 615.71.09, Torch 2.12.1+cu132 and Diffusers 0.41.0.
Default SDPA and CUDA `exact-streaming` decode were used. Chunking, RoPE
reuse and fusion, phase modulation, compact timesteps, inplace residuals,
compilation and standby purge are disabled. See the
[measurement details](ltx23-video-details.md#linux-leaf).

#### Full run

| Preset | Transformer runtime | Text encoder route | Video VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | Official Leaf hooks | Leaf | Resident | 342.96 | 7.62 | 38.13 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | `linear_runtime` | Leaf | Leaf | 352.56&nbsp;🔴&nbsp;↑ | 7.41&nbsp;🟢&nbsp;↓ | 43.40&nbsp;🔴&nbsp;↑ |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | `linear_runtime` | Leaf | Leaf | 344.17&nbsp;🔴&nbsp;↑ | 7.63&nbsp;🔴&nbsp;↑ | 42.04&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | `vmm_fast_runtime` | Native VMM | Resident | 325.78&nbsp;🟢&nbsp;↓ | 7.69&nbsp;🔴&nbsp;↑ | 33.07&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 290.26 | 36.28 | 7.62 | 37.90 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 274.65&nbsp;🟢&nbsp;↓ | 34.33&nbsp;🟢&nbsp;↓ | 7.41&nbsp;🟢&nbsp;↓ | 29.28&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 296.78&nbsp;🔴&nbsp;↑ | 37.10&nbsp;🔴&nbsp;↑ | 7.63&nbsp;🔴&nbsp;↑ | 42.03&nbsp;🔴&nbsp;↑ |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 274.26&nbsp;🟢&nbsp;↓ | 34.28&nbsp;🟢&nbsp;↓ | 7.69&nbsp;🔴&nbsp;↑ | 33.07&nbsp;🟢&nbsp;↓ |

#### Video VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf](ltx23-video-details.md#linux-leaf) (reference) | 26.32 | 4.89 | 10.68 |
| [`one_shot_fast`](ltx23-video-details.md#linux-linear) | 19.07&nbsp;🟢&nbsp;↓ | 4.88&nbsp;🟢&nbsp;↓ | 8.07&nbsp;🟢&nbsp;↓ |
| [`low_ram_safe`](ltx23-video-details.md#linux-low-ram-safe) | 21.86&nbsp;🟢&nbsp;↓ | 4.87&nbsp;🟢&nbsp;↓ | 9.77&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](ltx23-video-details.md#linux-native) | 21.40&nbsp;🟢&nbsp;↓ | 5.29&nbsp;🔴&nbsp;↑ | 14.44&nbsp;🔴&nbsp;↑ |

The One Shot Fast runner released 39.68 GiB of unused PyTorch pinned host cache
before VAE loading in 7.49 s, included in its tracked total. Its full run
took 9.60 s longer than Leaf despite a 15.61 s shorter denoise and 7.25 s
shorter video decode. Transformer attachment took 26.22 s. Its peak process
RSS was 43.40 GiB, compared with 38.13 GiB for Leaf.

VMM Fast had the shortest tracked total and lowest peak RSS in this series,
taking 17.19 s less overall than Leaf (5.0%), with a higher absolute GPU peak.
The three refreshed Leaf/linear MP4s were byte-identical.

See [phase measurements](ltx23-video-phases.md#linux-bf16) for this workload.

All three refreshed runs record `PYTORCH_ALLOC_CONF=expandable_segments:True`.
The Low RAM Safe command requests this setting explicitly; Leaf and One Shot
Fast use the shared runner default. Both linear cases applied their verified
0 GiB profiles. Low RAM Safe released 4.65 GiB of unused host cache in 0.88 s
before VAE loading. Cases were measured in separate processes; small latency
differences do not establish a repeatable speedup or regression.

<a id="linux-video-attention"></a>

### Transformer self-attention on VMM Fast

The same Linux 8-second recipe, component routes and saved VMM profile are
used below. Only video self-attention changes. The cuDNN operator check ran
separately; its profiler overhead is excluded from the timed measurement.

#### Full run

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 325.78 | 7.69 | 33.07 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 325.94&nbsp;🔴&nbsp;↑ | 7.70&nbsp;🔴&nbsp;↑ | 33.08&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 314.43&nbsp;🟢&nbsp;↓ | 7.69 | 33.08&nbsp;🔴&nbsp;↑ |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 261.78&nbsp;🟢&nbsp;↓ | 7.69 | 33.09&nbsp;🔴&nbsp;↑ |

#### Denoising

| Self-attention method | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 274.26 | 34.28 | 7.69 | 33.07 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 275.15&nbsp;🔴&nbsp;↑ | 34.39&nbsp;🔴&nbsp;↑ | 7.70&nbsp;🔴&nbsp;↑ | 33.08&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 268.28&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🟢&nbsp;↓ | 7.69 | 33.08&nbsp;🔴&nbsp;↑ |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 214.54&nbsp;🟢&nbsp;↓ | 26.82&nbsp;🟢&nbsp;↓ | 7.69 | 33.09&nbsp;🔴&nbsp;↑ |

#### Video VAE decode

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#linux-native) | 21.40 | 5.29 | 14.44 |
| [cuDNN SDPA](ltx23-video-details.md#linux-native-attention) | 22.74&nbsp;🔴&nbsp;↑ | 5.02&nbsp;🟢&nbsp;↓ | 14.44 |
| [FlashAttention 2](ltx23-video-details.md#linux-flash-attention) | 19.05&nbsp;🟢&nbsp;↓ | 4.97&nbsp;🟢&nbsp;↓ | 14.24&nbsp;🟢&nbsp;↓ |
| [Sol-Attn (sparse)](ltx23-video-details.md#linux-sol-attention) | 19.40&nbsp;🟢&nbsp;↓ | 5.68&nbsp;🔴&nbsp;↑ | 13.96&nbsp;🟢&nbsp;↓ |

These single measurements show no speed gain from cuDNN: the tracked totals
differed by 0.17 s. Both cuDNN runs completed with matching MP4 hashes; their
output differs from default SDPA. Media validation passed, while visual quality
has not yet been confirmed. See the
[cuDNN phase measurements](ltx23-video-phases.md#linux-cudnn-attention).

FlashAttention 2 reduced denoise time by 2.2% and tracked total by 3.5% versus
default SDPA in one timed run. The diagnostic confirmed the external FlashAttention
operator separately. Both FlashAttention MP4s match each other and differ from
SDPA and cuDNN; visual quality remains unconfirmed. See the
[FlashAttention phase measurements](ltx23-video-phases.md#linux-flash-attention).

Sol-Attn reduced denoise time by 21.8% and tracked total by 19.6% versus SDPA
in this single measurement. It uses approximate sparse attention in video
blocks 1 through 47 and FlashAttention 2 in block 0. Media validation passed;
its output differs from the dense variants and visual quality remains unconfirmed.
See the [Sol phase measurements](ltx23-video-phases.md#linux-sol-attention).

<a id="wsl-bf16"></a>

## WSL — BF16, 8-second pressure workload

LTX 2.3 Distilled BF16, 1280×704, 193 frames at 24 fps, 8 denoising steps,
seed 43, and guitar image and audio conditioning. The RTX 3060 Ti 8 GiB ran
with NVIDIA driver 617.42. The runtime series used default PyTorch SDPA and
CUDA `exact-streaming` decode. See the [WSL measurement settings](ltx23-video-details.md#wsl-8-second-workload).

#### Full run

| Preset | Transformer runtime | Text encoder route | Video VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | Official Leaf hooks | Leaf | Resident | 883.64 | 7.75 | 38.42 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | `linear_runtime` | Leaf | Leaf (no stream) | 1960.70&nbsp;🔴&nbsp;↑ | 8.00&nbsp;🔴&nbsp;↑ | 38.39&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | `vmm_fast_runtime` | Leaf | Resident | 700.53&nbsp;🟢&nbsp;↓ | 8.00&nbsp;🔴&nbsp;↑ | 38.27&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 641.02 | 80.13 | 7.75 | 38.42 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 1725.82&nbsp;🔴&nbsp;↑ | 215.73&nbsp;🔴&nbsp;↑ | 8.00&nbsp;🔴&nbsp;↑ | 38.38&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 461.95&nbsp;🟢&nbsp;↓ | 57.74&nbsp;🟢&nbsp;↓ | 8.00&nbsp;🔴&nbsp;↑ | 30.92&nbsp;🟢&nbsp;↓ |

#### Video VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Official Leaf (no stream)](ltx23-video-details.md#wsl-leaf) (reference) | 19.52 | 5.31 | 21.15 |
| [`wsl_compat`](ltx23-video-details.md#wsl-linear) | 19.49&nbsp;🟢&nbsp;↓ | 5.31 | 13.17&nbsp;🟢&nbsp;↓ |
| [`vmm_fast`](ltx23-video-details.md#wsl-native) | 20.95&nbsp;🔴&nbsp;↑ | 7.77&nbsp;🔴&nbsp;↑ | 13.44&nbsp;🟢&nbsp;↓ |

The runtime cases ran in separate sessions. VMM Fast finished 20.7% sooner
than Leaf in these observations. Leaf and WSL Compat produced byte-identical
MP4s, each validated at 1280×704 with 193 frames and stereo 48 kHz audio.
The VMM Fast output differs. WSL shares host resources with Windows; host
state and code/dependency identities differ between sessions, so these
measurements compare recorded configurations rather than isolate a runtime
effect. Single observations do not establish repeatability.

### Transformer self-attention on VMM Fast

The SDPA reference is the VMM Fast row in the runtime table above. SDPA
and Sol use the same final 0.6289 GiB transformer profile,
non-native text encoder, resident CUDA exact-streaming VAE, 0.375 GiB staging,
24.625 GiB source cache and 0.5 GiB source registration. Driver 617.42 and
the recorded package, source, native-library and fixture identities match
within this pair. No purge or other model optimization is enabled.
cuDNN and FlashAttention 2 retain measurements from an earlier session with
its own profile and source/dependency identities. Their arrows compare
recorded values against the current SDPA reference; those differences do
not isolate the attention backend from session and configuration effects.

#### Full run

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 700.53 | 8.00 | 38.27 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 639.05&nbsp;🟢&nbsp;↓ | 8.00 | 38.30&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 634.33&nbsp;🟢&nbsp;↓ | 8.00 | 38.25&nbsp;🟢&nbsp;↓ |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 600.55&nbsp;🟢&nbsp;↓ | 8.00 | 37.53&nbsp;🟢&nbsp;↓ |

#### Denoising

| Self-attention method | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 461.95 | 57.74 | 8.00 | 30.92 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 405.87&nbsp;🟢&nbsp;↓ | 50.73&nbsp;🟢&nbsp;↓ | 8.00 | 30.95&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 400.50&nbsp;🟢&nbsp;↓ | 50.06&nbsp;🟢&nbsp;↓ | 8.00 | 30.94&nbsp;🔴&nbsp;↑ |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 359.22&nbsp;🟢&nbsp;↓ | 44.90&nbsp;🟢&nbsp;↓ | 8.00 | 31.01&nbsp;🔴&nbsp;↑ |

#### Video VAE decode

| Self-attention method | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Default SDPA](ltx23-video-details.md#wsl-native) (reference) | 20.95 | 7.77 | 13.44 |
| [cuDNN SDPA](ltx23-video-details.md#wsl-native-attention) | 19.81&nbsp;🟢&nbsp;↓ | 7.70&nbsp;🟢&nbsp;↓ | 13.66&nbsp;🔴&nbsp;↑ |
| [FlashAttention 2](ltx23-video-details.md#wsl-native-attention) | 20.23&nbsp;🟢&nbsp;↓ | 7.65&nbsp;🟢&nbsp;↓ | 13.44 |
| [Sol-Attn (sparse)](ltx23-video-details.md#wsl-sol-attention) | 20.22&nbsp;🟢&nbsp;↓ | 7.64&nbsp;🟢&nbsp;↓ | 13.62&nbsp;🔴&nbsp;↑ |

Sol-Attn reduced denoise time by 22.2% and tracked total by 14.3% in this
single measured pair. The separate operator diagnostic confirmed eight
external FlashAttention 2 calls in video block 0 and 376 Sol annotations
in blocks 1–47, with zero video fallback calls. Both outputs contain 193
frames and finite audio; their MP4 hashes differ. The user accepted Sol's
visual quality.

The session profile was saved and applied successfully. Export to the global
profile directory was refused because a different profile already existed;
that file was preserved. This did not prevent the verified session profile
from being used by either measurement.

Separate diagnostics confirmed the cuDNN and external FlashAttention 2
operators and are excluded from timing. Both dense variants produced
different MP4 hashes from their original SDPA control. Their visual quality
and timing repeatability remain unassessed.

See [phase measurements](ltx23-video-phases.md#wsl-bf16) for this workload.

<a id="saved-fp8-weight-only-separate-precision-pairs"></a>
<a id="saved-fp8"></a>

## FP8 weight-only — 8-second workload

Use the same eight-second SDPA recipe and component routes as the platform's
VMM Fast BF16 runtime row above. Only the transformer uses a supported saved
TorchAO FP8 weight-only payload; auxiliary components remain BF16. Model
optimizations and compilation are disabled. Each precision calibrates its own
capacity profile; this compares complete profiled configurations, not a fixed
resident-budget precision microbenchmark.

Arrows compare against each platform's VMM Fast BF16 runtime row above.
The Windows percentage changes below retain the original precision-pair BF16
reference: 328.53 s denoise, 390.85 s tracked total and 33.02 GiB peak RSS.
The newer Windows runtime control is 321.35 s, 384.23 s and 33.08 GiB;
it is not the reference used to calculate those percentages. Linux retains
the same BF16 control for both comparisons. Windows FP8 uses DDO
0.1.0rc5+fp8floor.win1 and Linux uses 0.1.0rc5+fp8floor.2; the BF16 rows use
base RC5. Native allocator binaries are unchanged on each platform. These are
separately recorded configurations, not a matched package-only comparison.
See the [Windows scope](ltx23-video-details.md#fp8-windows) and
[Linux scope](ltx23-video-details.md#fp8-linux).

#### Full run

| Preset | Transformer runtime | Text encoder route | Video VAE route | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | `vmm_fast_runtime` | Native VMM | Resident | 424.02&nbsp;🔴&nbsp;↑ | 8.00 | 28.82&nbsp;🟢&nbsp;↓ |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | `vmm_fast_runtime` | Native VMM | Resident | 357.04&nbsp;🔴&nbsp;↑ | 7.68&nbsp;🟢&nbsp;↓ | 27.27&nbsp;🟢&nbsp;↓ |

#### Denoising

| Preset | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 330.27&nbsp;🔴&nbsp;↑ | 41.28&nbsp;🔴&nbsp;↑ | 8.00 | 28.82 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 289.78&nbsp;🔴&nbsp;↑ | 36.22&nbsp;🔴&nbsp;↑ | 7.68&nbsp;🟢&nbsp;↓ | 27.27&nbsp;🟢&nbsp;↓ |

#### Video VAE decode

| Preset | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Windows / FP8 weight-only](ltx23-video-details.md#fp8-windows) | 18.80 | 7.08 | 16.29 |
| [Linux / FP8 weight-only](ltx23-video-details.md#fp8-linux) | 18.29&nbsp;🟢&nbsp;↓ | 6.49&nbsp;🟢&nbsp;↓ | 15.56&nbsp;🟢&nbsp;↓ |

On Windows, FP8 took 8.5% longer overall, with denoise 0.5% longer and peak
process RSS 12.7% lower. The sampled absolute GPU peak was unchanged. The
transformer load/preparation and attachment took 40.89 s combined, versus
7.87 s for the BF16 reference. Media validation passed and the user accepted
the visual quality of this video.

On Linux, FP8 took 9.6% longer overall and 5.7% longer to denoise, while peak
process RSS was 17.5% lower. The absolute GPU peaks differed by only 0.01 GiB.
Linux video/audio validation passed; its visual quality remains unconfirmed.
These single observations establish no material VRAM or speed advantage. See the
[FP8 phase measurements](ltx23-video-phases.md#saved-fp8).

## Recommendations

| Hardware / priority | Recommendation from these measurements |
| --- | --- |
| 8 GiB GPU, ample system RAM, speed | `vmm_fast` has the shortest tracked total in the completed Windows, native Linux and WSL 8-second runtime sets. In the Windows 8-second workload its displayed peak GPU usage matches Official Leaf. |
| Lower process RAM on Windows | `vmm_fast` has the lowest peak process RSS in the 8-second runtime comparison at 33.08 GiB. |
| Lower decoder GPU pressure | Leaf VAE with temporal streaming reduces decode GPU peak, at extra time and RAM. Prefer resident temporal decode when it fits. |
| Neither RAM nor GPU headroom is sufficient for decode | Consider supported VAE tiling; validate quality and account for assembled output memory. |
| WSL | Prefer profiled `vmm_fast` with bounded source registration for this 8-second workload. `wsl_compat` completed but took longer than official Leaf. |
| Lower transformer storage / process RSS | FP8 weight-only reduced peak RSS by 12.7% on Windows and 17.5% on Linux, with totals 8.5% and 9.6% longer respectively. GPU peaks were effectively unchanged; prefer BF16 for speed in these configurations. |

`low_ram_safe` peaked at 42.42 GiB RSS in the Windows 8-second workload.
Capacity and performance depend on the workload and available memory.

Single observations, not statistical or arbitrary-model guarantees. Missing
measurements are unavailable, not zero; phase peaks must not be summed.
See [presets](../user/presets.md), [reproduction](reproduction.md) and
[measurement details](ltx23-video-details.md) before comparing different recipes.
