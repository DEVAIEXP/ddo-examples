# Cumulative model features

The `model-features` family measures model optimizations on VMM Fast separately
from runtime, attention and precision comparisons. The Windows LTX Video series
completed with BF16 and default SDPA; observations and comparison limits appear
below. Native Linux observations from the same ten-case queue appear in their
own section. Qwen VAE observations are available for Linux and Windows. LTX Image
has no additional cases in this series.

LTX Video rows add one feature to the preceding row. Qwen also includes
explicit alternative branches; their reference parent is identified below. The controller compares latency,
absolute GPU peak and peak RSS with both the preceding row and the control.
These are cumulative effects, including interactions with previously enabled
features; they do not establish the isolated effect of every feature.

## LTX 2.3 Video

The workload uses 1280×704, eight seconds, 193 frames at 24 fps, eight denoising
steps and seed 43, with the same guitar image, audio and prompt as the runtime
control. Encoder, resident CUDA exact-streaming VAE and source dispatcher remain
fixed. Windows requests standby purge before the transformer; Linux requests
no purge. Reclaim, compilation and alternative attention backends are disabled.

### Cases and comparison parents

| Case | What it does |
| --- | --- |
| `vmm-fast-control` | No model features; chunk sizes are zero |
| `rope-reuse` | Reuses position information instead of rebuilding it at each step |
| `compact-timesteps` | Keeps a smaller representation of each step's timing information |
| `inplace-residuals` | Reuses intermediate buffers to avoid extra copies |
| `fused-rms-modulation` | Combines normalization and scaling operations |
| `fused-rotary-pair` | Combines the position updates used by attention |
| `phase-modulation` | Reduces temporary buffers used for step-dependent adjustments |
| `frame-modulation` | Reduces temporary buffers used for frame-dependent adjustments |
| `feed-forward-1024` | Processes the feed-forward calculation in smaller pieces, size 1024 |
| `normalization-1024` | Processes normalization in smaller pieces, size 1024 |

### Profiles and output checks

Every case builds its own shape profile, using up to two three-step calibrations
without decode. Rotary reuse and compact timesteps change the transformer's
input signature, so the no-feature profile cannot validate all cases. The
control measurement applies its final profile recommendation. Subsequent
measurements explicitly use that same resident budget while checking their own
profile's input shapes. A different recommendation from a feature's calibration
does not increase its measurement budget. A failure stops the series.

Calibration and inference have separate outputs and timings. The full queue
contains at most 30 runner processes and 140 nominal denoising steps, excluding
bootstrap relaunches. The second calibration is skipped when the first produces
a final profile. Component budgets are not a total-VRAM limit.

### Preview the recipes

Preview the complete queue without running a model:

```powershell
python -B benchmarks/run_benchmark.py --platform windows --recipe ltx23-video-8s --family model-features
```

The controller uses the explicit `scripts/ltx2_video/run_model_features_distilled.py`
entry for this experiment. Ordinary VMM runner defaults remain unchanged.

<a id="windows-observations"></a>

### Windows — BF16

One observation per case on an RTX 3060 Ti 8 GiB, NVIDIA driver 617.42,
installed RC5, Torch 2.12.1+cu132 and Diffusers 0.41.0. Operator profiling was off.
Calibration time is excluded. These results belong to the cumulative feature
series; they do not replace the separate runtime or attention controls.

**Comparison limit:** the control applied the saved profile value of 0.6289 GiB;
the additions used its byte-derived recommendation, 0.62890625 GiB. Integer
budgets differ by 6,711 bytes. The strict equal-configured-budget criterion was
therefore not met. All measurements reported identical weight residency:
645,922,816 bytes total, comprising 608,174,080 execution-buffer bytes and
37,748,736 arena bytes. This does not authorize silently treating the configured
budgets as identical or rerunning the series automatically.

Times are seconds. Average is denoise latency divided by eight steps. Full run
includes all tracked setup, conditioning, denoising, video/audio decoding, output
conversion, encoding and saving; profile calibration and process startup are
excluded. Video VAE decode covers only the recorded `decode_video` interval,
including output conversion inside that phase but excluding VAE loading/cleanup,
audio decoding and saving. Memory peaks cover the same interval as each latency.
GPU peaks are sampled absolute usage and RAM peaks are process RSS, in GiB.
The global peak includes all tracked phases; it is distinct from the denoise
peak. See the [measurement glossary](glossary.md) and
[phase measurements](ltx23-video-phases.md#windows-cumulative-model-features).
Markers compare each observation with the first row: 🟢 ↓ is lower, 🔴 ↑ higher;
equal displayed values and unavailable values have no marker.

#### Full run

| Cumulative case | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Control: no features](ltx23-video-details.md#windows-cumulative-model-features) | 382.57 | 8.00 | 33.04 |
| [+ RoPE reuse](ltx23-video-details.md#windows-cumulative-model-features) | 397.53&nbsp;🔴&nbsp;↑ | 8.00 | 33.29&nbsp;🔴&nbsp;↑ |
| [+ Compact timesteps](ltx23-video-details.md#windows-cumulative-model-features) | 380.68&nbsp;🟢&nbsp;↓ | 6.46&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ In-place residuals](ltx23-video-details.md#windows-cumulative-model-features) | 383.24&nbsp;🔴&nbsp;↑ | 5.62&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Fused RMS modulation](ltx23-video-details.md#windows-cumulative-model-features) | 388.63&nbsp;🔴&nbsp;↑ | 5.62&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Fused rotary pair](ltx23-video-details.md#windows-cumulative-model-features) | 384.14&nbsp;🔴&nbsp;↑ | 5.62&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Phase modulation](ltx23-video-details.md#windows-cumulative-model-features) | 389.72&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Frame modulation](ltx23-video-details.md#windows-cumulative-model-features) | 381.97&nbsp;🟢&nbsp;↓ | 5.24&nbsp;🟢&nbsp;↓ | 33.18&nbsp;🔴&nbsp;↑ |
| [+ Feed-forward chunk 1024](ltx23-video-details.md#windows-cumulative-model-features) | 393.70&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Normalization chunk 1024](ltx23-video-details.md#windows-cumulative-model-features) | 388.66&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 33.18&nbsp;🔴&nbsp;↑ |

#### Denoising

| Cumulative case | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Control: no features](ltx23-video-details.md#windows-cumulative-model-features) | 319.52 | 39.94 | 8.00 | 33.04 |
| [+ RoPE reuse](ltx23-video-details.md#windows-cumulative-model-features) | 333.47&nbsp;🔴&nbsp;↑ | 41.68&nbsp;🔴&nbsp;↑ | 8.00 | 33.29&nbsp;🔴&nbsp;↑ |
| [+ Compact timesteps](ltx23-video-details.md#windows-cumulative-model-features) | 315.94&nbsp;🟢&nbsp;↓ | 39.49&nbsp;🟢&nbsp;↓ | 6.46&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ In-place residuals](ltx23-video-details.md#windows-cumulative-model-features) | 318.21&nbsp;🟢&nbsp;↓ | 39.78&nbsp;🟢&nbsp;↓ | 5.62&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Fused RMS modulation](ltx23-video-details.md#windows-cumulative-model-features) | 321.15&nbsp;🔴&nbsp;↑ | 40.14&nbsp;🔴&nbsp;↑ | 5.62&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Fused rotary pair](ltx23-video-details.md#windows-cumulative-model-features) | 315.65&nbsp;🟢&nbsp;↓ | 39.46&nbsp;🟢&nbsp;↓ | 5.62&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Phase modulation](ltx23-video-details.md#windows-cumulative-model-features) | 320.28&nbsp;🔴&nbsp;↑ | 40.04&nbsp;🔴&nbsp;↑ | 4.61&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Frame modulation](ltx23-video-details.md#windows-cumulative-model-features) | 313.21&nbsp;🟢&nbsp;↓ | 39.15&nbsp;🟢&nbsp;↓ | 4.10&nbsp;🟢&nbsp;↓ | 33.18&nbsp;🔴&nbsp;↑ |
| [+ Feed-forward chunk 1024](ltx23-video-details.md#windows-cumulative-model-features) | 324.72&nbsp;🔴&nbsp;↑ | 40.59&nbsp;🔴&nbsp;↑ | 3.60&nbsp;🟢&nbsp;↓ | 33.19&nbsp;🔴&nbsp;↑ |
| [+ Normalization chunk 1024](ltx23-video-details.md#windows-cumulative-model-features) | 320.47&nbsp;🔴&nbsp;↑ | 40.06&nbsp;🔴&nbsp;↑ | 3.60&nbsp;🟢&nbsp;↓ | 33.18&nbsp;🔴&nbsp;↑ |

#### VAE decode

| Cumulative case | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Control: no features](ltx23-video-details.md#windows-cumulative-model-features) | 18.93 | 7.62 | 16.18 |
| [+ RoPE reuse](ltx23-video-details.md#windows-cumulative-model-features) | 18.87&nbsp;🟢&nbsp;↓ | 7.51&nbsp;🟢&nbsp;↓ | 16.20&nbsp;🔴&nbsp;↑ |
| [+ Compact timesteps](ltx23-video-details.md#windows-cumulative-model-features) | 18.93 | 5.82&nbsp;🟢&nbsp;↓ | 16.07&nbsp;🟢&nbsp;↓ |
| [+ In-place residuals](ltx23-video-details.md#windows-cumulative-model-features) | 19.18&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 16.12&nbsp;🟢&nbsp;↓ |
| [+ Fused RMS modulation](ltx23-video-details.md#windows-cumulative-model-features) | 19.47&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 15.67&nbsp;🟢&nbsp;↓ |
| [+ Fused rotary pair](ltx23-video-details.md#windows-cumulative-model-features) | 19.65&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 16.12&nbsp;🟢&nbsp;↓ |
| [+ Phase modulation](ltx23-video-details.md#windows-cumulative-model-features) | 19.74&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 16.02&nbsp;🟢&nbsp;↓ |
| [+ Frame modulation](ltx23-video-details.md#windows-cumulative-model-features) | 19.76&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 15.85&nbsp;🟢&nbsp;↓ |
| [+ Feed-forward chunk 1024](ltx23-video-details.md#windows-cumulative-model-features) | 19.76&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 15.71&nbsp;🟢&nbsp;↓ |
| [+ Normalization chunk 1024](ltx23-video-details.md#windows-cumulative-model-features) | 19.78&nbsp;🔴&nbsp;↑ | 5.24&nbsp;🟢&nbsp;↓ | 16.11&nbsp;🟢&nbsp;↓ |

#### Output checks and recommendations

These recommendations apply to this cumulative workload, subject to the budget
limitation above. Each addition was measured once; small latency differences
are not evidence of a repeatable speedup or regression.

Compact timesteps, buffer reuse and modulation provided most of the memory
savings. The fused operations did not establish a repeatable speed gain.
Use chunking when extra memory headroom matters; it did not make this run faster.

The first eight MP4s, through frame modulation, are byte-identical. Adding
feed-forward chunking changes the MP4 hash; normalization chunking produces
the same hash as that preceding chunked case. The user assessed the final
chunked video as good overall, but reported a subtle dark spot in one frame.
Because the two chunked files are identical, that assessment concerns their
shared output. The corresponding control frame has not been compared, so
neither the spot's origin nor its attribution to chunking is established.
Decoded-pixel equivalence with the control remains unassessed.

The clearest observed benefit is lower GPU memory. Denoise peak fell from
8.00 GiB to 4.10 GiB through frame modulation, then to 3.60 GiB with chunking.
Global peak fell to 5.24 GiB because other phases also consume GPU memory.
The fully enabled case denoised in 320.47 s versus 319.52 s for the control;
these single observations do not establish a speed improvement. Calibration
was interleaved with measurements, and execution-order variation is not isolated.
The non-chunked chain preserves the exact output in this selected workload;
chunking offers additional denoise headroom with the visual caveat above.

<a id="linux-observations"></a>

### Linux — BF16

One completed observation per case on RTX 3060 Ti 8 GiB, NVIDIA driver
615.71.09, installed DDO/allocator RC5, Torch 2.12.1+cu132 and Diffusers
0.41.0. BF16, default SDPA and the same 1280×704/193-frame/eight-step/seed-43
guitar-conditioned workload are fixed. Encoder uses native VMM with a
4 GiB resident budget; video VAE uses resident CUDA exact-streaming.
Staging is 1.5 GiB and source cache/registration are 24.625 GiB each.
Source dispatcher is active. Purge, reclaim, compilation and operator
profiling are disabled. Calibration is excluded from the full-run latency.

Every measurement uses exactly 675,282,944 configured resident-weight bytes
(0.62890625 GiB), including the control. Each case has its own verified
final shape profile; other recommendations cannot increase the measurement
budget. Component budgets are not a total-VRAM ceiling.

**Comparison limit:** the first RoPE measurement was externally terminated
by Linux memory-pressure management and produced no accepted result. Its
calibration was preserved, and a new measurement completed on continuation.
Only that completed attempt appears below. The control precedes the
interruption; RoPE and the remaining measurements follow it. Host/cache
state and execution order are not isolated, so small timing differences
do not establish repeatable gains or regressions. No OS memory-policy
change or budget growth was used to resume.

The scope, units and markers match the Windows tables: each memory peak
covers the corresponding full-run, denoise or decode interval. Markers
compare with the no-feature control, not the preceding case. See
[phase measurements](ltx23-video-phases.md#linux-cumulative-model-features).

#### Full run

| Cumulative case | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Control: no features](ltx23-video-details.md#linux-cumulative-model-features) | 320.57 | 7.67 | 33.04 |
| [+ RoPE reuse](ltx23-video-details.md#linux-cumulative-model-features) | 347.32&nbsp;🔴&nbsp;↑ | 7.69&nbsp;🔴&nbsp;↑ | 33.54&nbsp;🔴&nbsp;↑ |
| [+ Compact timesteps](ltx23-video-details.md#linux-cumulative-model-features) | 331.07&nbsp;🔴&nbsp;↑ | 6.14&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ In-place residuals](ltx23-video-details.md#linux-cumulative-model-features) | 314.25&nbsp;🟢&nbsp;↓ | 5.15&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ Fused RMS modulation](ltx23-video-details.md#linux-cumulative-model-features) | 313.17&nbsp;🟢&nbsp;↓ | 5.13&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ Fused rotary pair](ltx23-video-details.md#linux-cumulative-model-features) | 306.76&nbsp;🟢&nbsp;↓ | 5.13&nbsp;🟢&nbsp;↓ | 33.54&nbsp;🔴&nbsp;↑ |
| [+ Phase modulation](ltx23-video-details.md#linux-cumulative-model-features) | 313.58&nbsp;🟢&nbsp;↓ | 4.73&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ Frame modulation](ltx23-video-details.md#linux-cumulative-model-features) | 311.60&nbsp;🟢&nbsp;↓ | 4.72&nbsp;🟢&nbsp;↓ | 33.54&nbsp;🔴&nbsp;↑ |
| [+ Feed-forward chunk 1024](ltx23-video-details.md#linux-cumulative-model-features) | 311.29&nbsp;🟢&nbsp;↓ | 4.70&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ Normalization chunk 1024](ltx23-video-details.md#linux-cumulative-model-features) | 311.94&nbsp;🟢&nbsp;↓ | 4.73&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |

#### Denoising

| Cumulative case | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Control: no features](ltx23-video-details.md#linux-cumulative-model-features) | 271.79 | 33.97 | 7.67 | 33.04 |
| [+ RoPE reuse](ltx23-video-details.md#linux-cumulative-model-features) | 295.18&nbsp;🔴&nbsp;↑ | 36.90&nbsp;🔴&nbsp;↑ | 7.69&nbsp;🔴&nbsp;↑ | 33.54&nbsp;🔴&nbsp;↑ |
| [+ Compact timesteps](ltx23-video-details.md#linux-cumulative-model-features) | 282.02&nbsp;🔴&nbsp;↑ | 35.25&nbsp;🔴&nbsp;↑ | 6.14&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ In-place residuals](ltx23-video-details.md#linux-cumulative-model-features) | 266.85&nbsp;🟢&nbsp;↓ | 33.36&nbsp;🟢&nbsp;↓ | 5.15&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ Fused RMS modulation](ltx23-video-details.md#linux-cumulative-model-features) | 265.38&nbsp;🟢&nbsp;↓ | 33.17&nbsp;🟢&nbsp;↓ | 5.13&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ Fused rotary pair](ltx23-video-details.md#linux-cumulative-model-features) | 259.77&nbsp;🟢&nbsp;↓ | 32.47&nbsp;🟢&nbsp;↓ | 5.13&nbsp;🟢&nbsp;↓ | 33.54&nbsp;🔴&nbsp;↑ |
| [+ Phase modulation](ltx23-video-details.md#linux-cumulative-model-features) | 265.90&nbsp;🟢&nbsp;↓ | 33.24&nbsp;🟢&nbsp;↓ | 4.15&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ Frame modulation](ltx23-video-details.md#linux-cumulative-model-features) | 262.79&nbsp;🟢&nbsp;↓ | 32.85&nbsp;🟢&nbsp;↓ | 3.64&nbsp;🟢&nbsp;↓ | 33.54&nbsp;🔴&nbsp;↑ |
| [+ Feed-forward chunk 1024](ltx23-video-details.md#linux-cumulative-model-features) | 263.65&nbsp;🟢&nbsp;↓ | 32.96&nbsp;🟢&nbsp;↓ | 3.11&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |
| [+ Normalization chunk 1024](ltx23-video-details.md#linux-cumulative-model-features) | 264.41&nbsp;🟢&nbsp;↓ | 33.05&nbsp;🟢&nbsp;↓ | 3.14&nbsp;🟢&nbsp;↓ | 33.53&nbsp;🔴&nbsp;↑ |

#### VAE decode

| Cumulative case | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Control: no features](ltx23-video-details.md#linux-cumulative-model-features) | 19.75 | 5.47 | 14.55 |
| [+ RoPE reuse](ltx23-video-details.md#linux-cumulative-model-features) | 20.85&nbsp;🔴&nbsp;↑ | 5.33&nbsp;🟢&nbsp;↓ | 14.28&nbsp;🟢&nbsp;↓ |
| [+ Compact timesteps](ltx23-video-details.md#linux-cumulative-model-features) | 19.76&nbsp;🔴&nbsp;↑ | 5.34&nbsp;🟢&nbsp;↓ | 14.35&nbsp;🟢&nbsp;↓ |
| [+ In-place residuals](ltx23-video-details.md#linux-cumulative-model-features) | 19.34&nbsp;🟢&nbsp;↓ | 4.72&nbsp;🟢&nbsp;↓ | 14.52&nbsp;🟢&nbsp;↓ |
| [+ Fused RMS modulation](ltx23-video-details.md#linux-cumulative-model-features) | 19.59&nbsp;🟢&nbsp;↓ | 4.70&nbsp;🟢&nbsp;↓ | 14.61&nbsp;🔴&nbsp;↑ |
| [+ Fused rotary pair](ltx23-video-details.md#linux-cumulative-model-features) | 19.07&nbsp;🟢&nbsp;↓ | 4.70&nbsp;🟢&nbsp;↓ | 14.17&nbsp;🟢&nbsp;↓ |
| [+ Phase modulation](ltx23-video-details.md#linux-cumulative-model-features) | 19.13&nbsp;🟢&nbsp;↓ | 4.73&nbsp;🟢&nbsp;↓ | 13.60&nbsp;🟢&nbsp;↓ |
| [+ Frame modulation](ltx23-video-details.md#linux-cumulative-model-features) | 19.73&nbsp;🟢&nbsp;↓ | 4.72&nbsp;🟢&nbsp;↓ | 14.54&nbsp;🟢&nbsp;↓ |
| [+ Feed-forward chunk 1024](ltx23-video-details.md#linux-cumulative-model-features) | 19.56&nbsp;🟢&nbsp;↓ | 4.69&nbsp;🟢&nbsp;↓ | 14.71&nbsp;🔴&nbsp;↑ |
| [+ Normalization chunk 1024](ltx23-video-details.md#linux-cumulative-model-features) | 19.22&nbsp;🟢&nbsp;↓ | 4.73&nbsp;🟢&nbsp;↓ | 14.43&nbsp;🟢&nbsp;↓ |

#### Output checks and recommendations

These are cumulative observations, including interactions with previously
enabled features. They do not measure each feature in isolation.

The lowest denoise time was observed with fused rotary pair (259.77 s),
but one run does not establish a repeatable speed gain. Modulation and
chunking mainly helped GPU memory. Normalization chunking added no further
sampled memory saving.

The first eight MP4s are byte-identical to the no-feature control. The two
chunked MP4s are identical to each other and differ from the control. The user
assessed their shared output as good overall, with mild artifacts similar to
the Windows observation. The affected decoded frames have not been compared
with the control; the artifacts' origin and attribution to chunking are not
established. Inspect output quality when selecting the chunked chain.

The final case reduced denoise peak from 7.67 to 3.14 GiB, while full-run
peak fell from 7.67 to 4.73 GiB because other phases still consume GPU memory.
Final denoise/full-run latency was 264.41/311.94 s versus 271.79/320.57 s
for the control. Treat these timing differences in light of the interrupted,
calibration-interleaved series. The separate runtime and attention controls
remain unchanged.

#### Preview the recipes

Preview the same queue without loading weights or initializing CUDA:

```bash
python -B benchmarks/run_benchmark.py --platform linux --recipe ltx23-video-8s --family model-features
```

Use the reviewed runtime checkpoint and conditioning files. Supply
`--ffmpeg-bin` when needed and `--expected-library-sha256` for preflight or
capture. `--session-action continue` preserves verified stages; it never
turns the interrupted attempt into an accepted measurement.

<a id="qwen-image"></a>

## Qwen Image 2.1

This is the single Qwen VAE-feature study. It is separate from runtime,
attention and precision comparisons. Linux and Windows observations appear below; visual acceptance is separate
from execution and pixel checks.

Fixed workload: Qwen Image 2.1 Base BF16, 1024×1024, 20 steps, seed 42,
guidance 1, default KV cache and the capybara prompt used by the runtime
control. Transformer joint attention stays on default SDPA. Native VMM
encoder and VAE use the same prepared BF16 VAE source as the runtime study.
Encoder residency is 4 GiB, VAE residency 0.375 GiB, total staging 0.625 GiB,
and source cache/registration ceilings 16 GiB each. The source dispatcher is
enabled. No standby purge, reclaim, compilation or quantization is requested.
Component budgets are not total-VRAM ceilings.

### Cases and comparison parents

The VAE turns the denoised result into the final image. These options change
that decoding stage, not the denoising calculation:

- **Single-frame** skips the temporal cache needed for sequences of frames.
- **Residual alias** reuses a decoder buffer instead of keeping another copy.
- **Direct padding** changes how borders are prepared for convolution; it can
  change image pixels slightly.
- **Bands 256 or 128** process parts of the decoder in horizontal strips to
  reduce temporary GPU memory. Smaller strips may take longer.
- **Original tiling** decodes overlapping image tiles. It can save more decoder
  memory, but adds work and can change pixels.

| Case | Decoder configuration | Comparison parent |
| --- | --- | --- |
| `vmm-fast-control` | Original untiled decoder, no VAE features | Control |
| `single-frame` | Single-frame without temporal cache | Control |
| `residual-alias` | Single-frame + residual alias | `single-frame` |
| `direct-padding` | Single-frame + alias + direct padding | `residual-alias` |
| `bands-256` | Single-frame + alias + direct padding + bands 256 | `direct-padding` |
| `bands-128` | Single-frame + alias + direct padding + bands 128 | `direct-padding` |
| `bands-256-no-direct-padding` | Single-frame + alias + bands 256 | `residual-alias` |
| `bands-128-no-direct-padding` | Single-frame + alias + bands 128 | `residual-alias` |
| `tiling-only` | Tiling on the original decoder; other features off | Control |

Both band sizes use two final decoder blocks. Replacing 256 rows with 128,
omitting direct padding, and selecting tiling are alternative branches, not
additions to the preceding row. The controller compares each result with its
named parent and the common no-feature control. This covers the original
single-frame/alias/band/tiling combinations as well as direct padding, without
claiming that an earlier resident-VAE timing is a new native-VAE result.

### Profiles and output checks

Only the control calibrates the transformer, with at most two three-step
builds and no VAE decode. The second is skipped if the profile is already
final. All nine measurements reuse that intact final profile and exactly its
integer-byte resident budget, converted to GiB without decimal rounding.
VAE features do not change transformer input shapes. The controller verifies
the explicit budget, profile shape, decoder class, feature application,
separate denoise/decode metrics, PNG dimensions, hashes and decoded pixels.

At most 11 runner processes and 186 nominal denoising steps produce nine
1024-square PNGs, excluding bootstrap relaunches. Tiling can change pixels;
different output requires separate visual assessment. No speed, exact-output
or memory-fit guarantee is inferred from a recipe preview. Runtime and
attention reference tables remain unchanged.

<a id="windows-qwen-vae-observations"></a>

### Windows — BF16

One observation per case on RTX 3060 Ti 8 GiB, NVIDIA driver 617.42,
installed DDO/allocator RC5, Torch 2.12.1+cu132 and Diffusers 0.41.0.
Only the control calibrated. All nine measurements reused its intact final
profile, matching shapes and exactly 671,088,640 resident-weight bytes
(0.625 GiB). Native encoder/VAE routes, the prepared BF16 VAE source,
component budgets, default SDPA and active source dispatcher were fixed.
No standby purge, reclaim, compilation or quantization was enabled.

Calibration is excluded. Full run includes all tracked loading, encoding,
denoising/readback, decoding, cleanup and PNG saving. Decode excludes VAE
loading/attachment, cleanup and saving. GPU peaks are absolute usage and
RAM peaks are process RSS, in GiB, each within its own measured interval.
Markers compare with the original untiled control: 🟢 ↓ lower, 🔴 ↑ higher;
ties at displayed precision are unmarked. Cases ran sequentially once;
small timing differences do not establish repeatability. VAE changes occur
after the transformer, so denoise variation is not a VAE speed effect.
See [phase measurements](qwen-image-21-phases.md#windows-qwen-vae-features).

#### Full run

| VAE configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Original untiled control](qwen-image-21-details.md#windows-vae-feature-study) | 59.94 | 8.00 | 18.17 |
| [Single-frame](qwen-image-21-details.md#windows-vae-feature-study) | 59.90&nbsp;🟢&nbsp;↓ | 5.84&nbsp;🟢&nbsp;↓ | 18.17 |
| [+ Residual alias](qwen-image-21-details.md#windows-vae-feature-study) | 59.80&nbsp;🟢&nbsp;↓ | 5.16&nbsp;🟢&nbsp;↓ | 18.17 |
| [+ Direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 59.79&nbsp;🟢&nbsp;↓ | 5.16&nbsp;🟢&nbsp;↓ | 18.17 |
| [Bands 256 / direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 59.84&nbsp;🟢&nbsp;↓ | 3.44&nbsp;🟢&nbsp;↓ | 18.17 |
| [Bands 128 / direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 60.13&nbsp;🔴&nbsp;↑ | 3.44&nbsp;🟢&nbsp;↓ | 18.17 |
| [Bands 256 / no direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 59.84&nbsp;🟢&nbsp;↓ | 3.44&nbsp;🟢&nbsp;↓ | 18.17 |
| [Bands 128 / no direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 59.87&nbsp;🟢&nbsp;↓ | 3.44&nbsp;🟢&nbsp;↓ | 18.17 |
| [Tiling only](qwen-image-21-details.md#windows-vae-feature-study) | 62.17&nbsp;🔴&nbsp;↑ | 3.44&nbsp;🟢&nbsp;↓ | 18.17 |

#### Denoising

| VAE configuration | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Original untiled control](qwen-image-21-details.md#windows-vae-feature-study) | 50.10 | 2.50 | 2.45 | 15.55 |
| [Single-frame](qwen-image-21-details.md#windows-vae-feature-study) | 50.12&nbsp;🔴&nbsp;↑ | 2.51&nbsp;🔴&nbsp;↑ | 2.45 | 15.54&nbsp;🟢&nbsp;↓ |
| [+ Residual alias](qwen-image-21-details.md#windows-vae-feature-study) | 50.36&nbsp;🔴&nbsp;↑ | 2.52&nbsp;🔴&nbsp;↑ | 2.45 | 15.55 |
| [+ Direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 50.42&nbsp;🔴&nbsp;↑ | 2.52&nbsp;🔴&nbsp;↑ | 2.45 | 15.55 |
| [Bands 256 / direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 50.44&nbsp;🔴&nbsp;↑ | 2.52&nbsp;🔴&nbsp;↑ | 2.45 | 15.55 |
| [Bands 128 / direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 50.45&nbsp;🔴&nbsp;↑ | 2.52&nbsp;🔴&nbsp;↑ | 2.45 | 15.55 |
| [Bands 256 / no direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 50.39&nbsp;🔴&nbsp;↑ | 2.52&nbsp;🔴&nbsp;↑ | 2.45 | 15.55 |
| [Bands 128 / no direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 50.41&nbsp;🔴&nbsp;↑ | 2.52&nbsp;🔴&nbsp;↑ | 2.45 | 15.55 |
| [Tiling only](qwen-image-21-details.md#windows-vae-feature-study) | 50.41&nbsp;🔴&nbsp;↑ | 2.52&nbsp;🔴&nbsp;↑ | 2.45 | 15.55 |

#### VAE decode

| VAE configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Original untiled control](qwen-image-21-details.md#windows-vae-feature-study) | 1.47 | 8.00 | 3.12 |
| [Single-frame](qwen-image-21-details.md#windows-vae-feature-study) | 1.22&nbsp;🟢&nbsp;↓ | 5.84&nbsp;🟢&nbsp;↓ | 3.09&nbsp;🟢&nbsp;↓ |
| [+ Residual alias](qwen-image-21-details.md#windows-vae-feature-study) | 1.21&nbsp;🟢&nbsp;↓ | 5.16&nbsp;🟢&nbsp;↓ | 3.11&nbsp;🟢&nbsp;↓ |
| [+ Direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 1.15&nbsp;🟢&nbsp;↓ | 5.16&nbsp;🟢&nbsp;↓ | 3.08&nbsp;🟢&nbsp;↓ |
| [Bands 256 / direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 1.21&nbsp;🟢&nbsp;↓ | 3.34&nbsp;🟢&nbsp;↓ | 3.11&nbsp;🟢&nbsp;↓ |
| [Bands 128 / direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 1.34&nbsp;🟢&nbsp;↓ | 3.34&nbsp;🟢&nbsp;↓ | 3.08&nbsp;🟢&nbsp;↓ |
| [Bands 256 / no direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 1.28&nbsp;🟢&nbsp;↓ | 3.20&nbsp;🟢&nbsp;↓ | 3.08&nbsp;🟢&nbsp;↓ |
| [Bands 128 / no direct padding](qwen-image-21-details.md#windows-vae-feature-study) | 1.39&nbsp;🟢&nbsp;↓ | 3.20&nbsp;🟢&nbsp;↓ | 3.08&nbsp;🟢&nbsp;↓ |
| [Tiling only](qwen-image-21-details.md#windows-vae-feature-study) | 3.36&nbsp;🔴&nbsp;↑ | 2.18&nbsp;🟢&nbsp;↓ | 3.09&nbsp;🟢&nbsp;↓ |

#### Output checks and recommendations

The original control, single-frame, residual alias and both band sizes without
direct padding produced byte-identical PNGs and identical RGB pixels. Direct
padding and its two band variants share a second PNG. Against the control,
mean absolute RGB differences were 0.11, 0.09 and 0.07 levels on the 8-bit
scale, with maxima 5, 6 and 3. Original tiling produced a third PNG, with
means 2.67, 2.14 and 1.37 and maxima 92, 77 and 79. These pixel comparisons
were also checked against each named branch parent. Windows visual assessment
of the changed outputs remains pending; Linux visual acceptance does not
automatically validate these Windows images.

Single-frame and alias reduced decoder memory while preserving the image.
Bands 256 without direct padding reduced it further to 3.20 GiB at 1.28 s;
128 rows used the same memory and took longer. Direct padding did not reduce
memory below those exact-image bands. Tiling used the least decoder memory
(2.18 GiB), but took 3.36 s and changed pixels. The full-run peak remained
3.44 GiB for both tiling and the band options.

Untiled full runs stayed near 59.80–60.13 s; tiling took 62.17 s. Most of the
benefit is decoder memory, not total latency. Existing runtime and attention
reference tables and runner defaults remain unchanged.

<a id="linux-qwen-vae-observations"></a>

### Linux — BF16

One observation per case on RTX 3060 Ti 8 GiB, NVIDIA driver 615.71.09,
installed DDO/allocator RC5, Torch 2.12.1+cu132 and Diffusers 0.41.0. All
nine measurements applied the same final transformer profile and exactly
671,088,640 configured resident-weight bytes (0.625 GiB). The native encoder,
BF16 native VAE source, component budgets, SDPA and source dispatcher were
fixed. No standby purge, reclaim, compilation or quantization was enabled.

Only the control calibrated, using two three-step runs without decode.
Calibration is outside the measured full-run latency. All other cases reused
that intact final profile. The transformer ran before the modified VAE;
small denoise differences are not evidence of a VAE speedup. Cases ran
sequentially once, so timing differences do not establish repeatability.

Times are seconds. Full run includes all tracked loading/setup, encoding,
denoising/readback, VAE decode, cleanup and saving. Decode excludes VAE
loading/attachment, cleanup and PNG saving. GPU peaks are sampled absolute
usage and RAM peaks are process RSS, in GiB; each peak covers the same scope
as its latency. 🟢 ↓ is lower and 🔴 ↑ higher than the original untiled
control; equal displayed values have no marker. Branch-parent comparisons
remain distinct from the common-control comparisons in these tables. See
[phase measurements](qwen-image-21-phases.md#linux-qwen-vae-features).

#### Full run

| VAE configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Original untiled control](qwen-image-21-details.md#linux-vae-feature-study) | 54.12 | 7.66 | 18.28 |
| [Single-frame](qwen-image-21-details.md#linux-vae-feature-study) | 54.29&nbsp;🔴&nbsp;↑ | 5.49&nbsp;🟢&nbsp;↓ | 18.26&nbsp;🟢&nbsp;↓ |
| [+ Residual alias](qwen-image-21-details.md#linux-vae-feature-study) | 54.37&nbsp;🔴&nbsp;↑ | 4.76&nbsp;🟢&nbsp;↓ | 18.26&nbsp;🟢&nbsp;↓ |
| [+ Direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 54.36&nbsp;🔴&nbsp;↑ | 4.75&nbsp;🟢&nbsp;↓ | 18.26&nbsp;🟢&nbsp;↓ |
| [Bands 256 / direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 54.43&nbsp;🔴&nbsp;↑ | 3.06&nbsp;🟢&nbsp;↓ | 18.26&nbsp;🟢&nbsp;↓ |
| [Bands 128 / direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 54.71&nbsp;🔴&nbsp;↑ | 3.06&nbsp;🟢&nbsp;↓ | 18.26&nbsp;🟢&nbsp;↓ |
| [Bands 256 / no direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 54.44&nbsp;🔴&nbsp;↑ | 3.03&nbsp;🟢&nbsp;↓ | 18.26&nbsp;🟢&nbsp;↓ |
| [Bands 128 / no direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 54.46&nbsp;🔴&nbsp;↑ | 3.03&nbsp;🟢&nbsp;↓ | 18.26&nbsp;🟢&nbsp;↓ |
| [Tiling only](qwen-image-21-details.md#linux-vae-feature-study) | 56.16&nbsp;🔴&nbsp;↑ | 3.03&nbsp;🟢&nbsp;↓ | 18.26&nbsp;🟢&nbsp;↓ |

#### Denoising

| VAE configuration | Latency (s) | Average (s/step) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| [Original untiled control](qwen-image-21-details.md#linux-vae-feature-study) | 44.30 | 2.21 | 2.13 | 15.71 |
| [Single-frame](qwen-image-21-details.md#linux-vae-feature-study) | 44.53&nbsp;🔴&nbsp;↑ | 2.23&nbsp;🔴&nbsp;↑ | 2.13 | 15.69&nbsp;🟢&nbsp;↓ |
| [+ Residual alias](qwen-image-21-details.md#linux-vae-feature-study) | 44.68&nbsp;🔴&nbsp;↑ | 2.23&nbsp;🔴&nbsp;↑ | 2.13 | 15.69&nbsp;🟢&nbsp;↓ |
| [+ Direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 44.68&nbsp;🔴&nbsp;↑ | 2.23&nbsp;🔴&nbsp;↑ | 2.07&nbsp;🟢&nbsp;↓ | 15.69&nbsp;🟢&nbsp;↓ |
| [Bands 256 / direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 44.69&nbsp;🔴&nbsp;↑ | 2.23&nbsp;🔴&nbsp;↑ | 2.07&nbsp;🟢&nbsp;↓ | 15.69&nbsp;🟢&nbsp;↓ |
| [Bands 128 / direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 44.77&nbsp;🔴&nbsp;↑ | 2.24&nbsp;🔴&nbsp;↑ | 2.06&nbsp;🟢&nbsp;↓ | 15.69&nbsp;🟢&nbsp;↓ |
| [Bands 256 / no direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 44.58&nbsp;🔴&nbsp;↑ | 2.23&nbsp;🔴&nbsp;↑ | 2.05&nbsp;🟢&nbsp;↓ | 15.69&nbsp;🟢&nbsp;↓ |
| [Bands 128 / no direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 44.60&nbsp;🔴&nbsp;↑ | 2.23&nbsp;🔴&nbsp;↑ | 2.05&nbsp;🟢&nbsp;↓ | 15.69&nbsp;🟢&nbsp;↓ |
| [Tiling only](qwen-image-21-details.md#linux-vae-feature-study) | 44.61&nbsp;🔴&nbsp;↑ | 2.23&nbsp;🔴&nbsp;↑ | 2.05&nbsp;🟢&nbsp;↓ | 15.69&nbsp;🟢&nbsp;↓ |

#### VAE decode

| VAE configuration | Latency (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: |
| [Original untiled control](qwen-image-21-details.md#linux-vae-feature-study) | 1.41 | 7.66 | 3.34 |
| [Single-frame](qwen-image-21-details.md#linux-vae-feature-study) | 1.21&nbsp;🟢&nbsp;↓ | 5.49&nbsp;🟢&nbsp;↓ | 3.34 |
| [+ Residual alias](qwen-image-21-details.md#linux-vae-feature-study) | 1.18&nbsp;🟢&nbsp;↓ | 4.76&nbsp;🟢&nbsp;↓ | 3.27&nbsp;🟢&nbsp;↓ |
| [+ Direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 1.13&nbsp;🟢&nbsp;↓ | 4.75&nbsp;🟢&nbsp;↓ | 3.28&nbsp;🟢&nbsp;↓ |
| [Bands 256 / direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 1.21&nbsp;🟢&nbsp;↓ | 2.93&nbsp;🟢&nbsp;↓ | 3.31&nbsp;🟢&nbsp;↓ |
| [Bands 128 / direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 1.31&nbsp;🟢&nbsp;↓ | 2.91&nbsp;🟢&nbsp;↓ | 3.29&nbsp;🟢&nbsp;↓ |
| [Bands 256 / no direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 1.26&nbsp;🟢&nbsp;↓ | 2.77&nbsp;🟢&nbsp;↓ | 3.27&nbsp;🟢&nbsp;↓ |
| [Bands 128 / no direct padding](qwen-image-21-details.md#linux-vae-feature-study) | 1.37&nbsp;🟢&nbsp;↓ | 2.77&nbsp;🟢&nbsp;↓ | 3.27&nbsp;🟢&nbsp;↓ |
| [Tiling only](qwen-image-21-details.md#linux-vae-feature-study) | 2.98&nbsp;🔴&nbsp;↑ | 1.73&nbsp;🟢&nbsp;↓ | 3.28&nbsp;🟢&nbsp;↓ |

#### Output checks and recommendations

The control, single-frame, residual alias, and both band sizes without direct
padding have byte-identical PNGs and identical RGB pixels. Direct padding and
both band sizes with it share a second PNG: mean absolute RGB differences
against the control are 0.11, 0.09 and 0.07 levels on the 8-bit scale, with
maximum differences 8, 8 and 5. Tiling produces a third image, with mean
differences 2.66, 2.14 and 1.37 and maxima 99, 93 and 75. Numerical differences
do not establish perceptual quality. The reviewer assessed both changed
outputs, direct padding and tiling, as visually good. This selected visual
acceptance does not imply pixel equality or quality across other recipes.

Single-frame and alias offer a simple reduction in decoder memory. Bands 256
without direct padding give more headroom while preserving the image:
2.77 GiB at 1.26 s. Bands 128 used the same memory and took longer. Direct
padding did not improve memory over the matching bands without it. Tiling
used the least decoder memory (1.73 GiB), but was slower (2.98 s).

Full-run totals stayed near 54.12–54.71 s for the untiled variants; tiling
took 56.16 s. The benefit is mainly decoder memory, with modest absolute
decode savings. These observations do not establish an end-to-end speedup.
No runner defaults or separate runtime/attention references were changed.

### Preview the recipes

Windows, after selecting the corresponding installed environment and source:

```powershell
python -B benchmarks/run_benchmark.py --platform windows --recipe qwen-image21 --family model-features --vae-model-path 'elismasilva/qwen-image-2.1-bf16-vae'
```

Linux, using the same prepared BF16 VAE source as the runtime benchmark:

```bash
python -B benchmarks/run_benchmark.py --platform linux --recipe qwen-image21 --family model-features --vae-model-path elismasilva/qwen-image-2.1-bf16-vae
```

Use `--checkpoint` to pin the reviewed Base snapshot and `--custom-blocks` to
select its clean reviewed code. Preflight/capture require the matching native
library SHA256 (`--expected-library-sha256` on Linux, `--expected-dll-sha256`
on Windows). The BF16 VAE config and weight files are hashed before capture
and become part of the session identity. No conversion is performed.
Do not combine tiling with bands, alias, direct padding or single-frame in
this study. Other combinations are not silently added to the queue.

## Evidence and continuation

Use `--list-cases` to list the queue. `--case` keeps the preceding cases and
stops after the selected addition. Preflight requires the reviewed installed
environment, checkpoint, custom blocks and native library hash. Execution must
be approved separately; append `--execute` only after reviewing its commands
and limits. Windows video execution needs elevation for its requested purge.

Sessions under `outputs/benchmarks/<platform>` preserve commands, profiles, metrics,
output hashes and failed attempts. Use `--session-action continue` to skip
verified completed cases, or `--session-action restart` to create a new session.
Source or environment changes require a compatible new session. Active attempts
are protected by the shared session lock and child-process checks.

For video, every addition records its MP4 SHA256 and the reference SHA256, with
equality flags against both the no-feature SDPA BF16 control and the preceding
case. An equal hash proves byte-identical files. A different container hash does
not by itself establish different decoded pixels or poor quality; visual
acceptance remains separate. Qwen additionally compares decoded PNG pixels.
Missing output or feature-application evidence cannot complete a case.
