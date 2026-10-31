# Comparison summary

Use each model's tables for measured values; no universal ranking is implied.

**Before reading the tables:** see the [labels and memory glossary](glossary.md)
for runtime labels, denoise/decode scope, tracked totals and GPU/RAM peaks.

## At a glance — Windows BF16

Existing single-run observations on an **RTX 3060 Ti 8 GiB / 64 GiB RAM**.
Method names below identify the transformer; all rows in these primary Windows
sets use a shared official leaf text encoder and resident VAE (temporal streaming
for video). Component choices can be combined; a native transformer does not
mean every pipeline component uses native VMM.

Times are seconds. GPU peaks are sampled **absolute driver usage**, RAM peaks
are process **RSS**, both in **GiB**, over the tracked run. Tracked total excludes
command startup and downloads. Denoise and other phase intervals can be nested;
do not sum them to reconstruct total. See [metric definitions](glossary.md).

VAE decode is the recorded decoder execution phase, excluding VAE loading.
For video it means the video decoder only; audio decode and vocoder are separate
phases in the detailed tables. Tracked total already includes decoding and other
pipeline work. Memory columns remain global run peaks, not decoder-only peaks.

### LTX video — 1280×704, 5 seconds, 121 frames, 8 steps

| Transformer method | Denoise (s) | Video VAE decode (s) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Diffusers streamed leaf](ltx23-video-details.md#windows-leaf) | 392.91 | 22.52 | 452.33 | 4.72 | 41.34 |
| [DDO linear](ltx23-video-details.md#windows-linear) | 262.22 | 23.47 | 336.44 | 4.72 | 47.87 |
| [DDO native VMM](ltx23-video-details.md#windows-native) | 207.40 | 24.00 | 277.09 | 5.21 | 32.65 |

See [each video phase and final total](ltx23-video.md#windows-bf16), including
conditioning and decoder costs. This workload generates audio without input
audio; later audio-conditioned recording recipes are different comparisons.

### LTX image — 1280×704, 8 steps

| Transformer method | Denoise (s) | VAE decode (s) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Diffusers streamed leaf](ltx23-image-details.md#windows-leaf) | 39.03 | 0.32 | 69.97 | 7.15 | 28.77 |
| [DDO linear](ltx23-image-details.md#windows-linear) | 11.99 | 0.30 | 54.33 | 7.43 | 27.12 |
| [DDO native VMM](ltx23-image-details.md#windows-native) | 15.15 | 0.39 | 48.73 | 7.27 | 29.53 |

Linear has shorter denoise; native has shorter total. Diffusers leaf has the
lowest global GPU peak of these three rows. See [image phases and final total](ltx23-image.md#windows-bf16).

### Qwen Image 2.1 — 512×512, 20 steps

| Transformer method | Denoise/readback (s) | VAE decode (s) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Diffusers streamed leaf](qwen-image-21-details.md#windows-leaf) | 41.28 | 0.47 | 57.50 | 3.68 | 18.97 |
| [DDO linear](qwen-image-21-details.md#windows-linear) | 17.94 | 0.49 | 41.36 | 6.69 | 21.58 |
| [DDO native VMM — reused observation](qwen-image-21-details.md#windows-native) | 17.67 | 0.47 | 36.64 | 6.12 | 18.97 |

The native Qwen row comes from a separate session, not a side-by-side control;
its small denoise difference from linear does not establish a reliable advantage.
Diffusers leaf uses less GPU memory. See [Qwen phases and final total](qwen-image-21.md#windows-bf16)
and the separate 1024-square results there; resolutions must not be mixed.

These configured pipeline results do not isolate the effect of VMM, establish
an advantage over a matched `torch.compile` setup, or certify other models.
Native Linux, WSL, lower-RAM, VAE-route and quantization comparisons remain in
their separate groups. [Hardware and environment limits](hardware.md).

## Choose what to investigate

| Model / priority | What the selected measurements support | Tables |
| --- | --- | --- |
| LTX image / one-shot speed | Windows linear has shorter denoise but native has shorter tracked total; Linux native is shortest in both primary measures. | [LTX image](ltx23-image.md) |
| LTX video / one-shot speed | Native fast VMM has the shortest tracked total in each primary five-second platform set. | [LTX video](ltx23-video.md) |
| Qwen / one-shot speed | Linear/native shorten 512-square denoise versus official leaf; native has the shortest tracked total in the selected sets. | [Qwen](qwen-image-21.md) |
| Lower transformer-phase process RAM | Windows LTX video protected NVMe uses much less denoise RSS with a close total to native; OS file cache and the encoder still matter. | [Video Windows](ltx23-video.md#windows-bf16) |
| Lower decoder GPU pressure | Leaf VAE can lower decode GPU peaks at extra time/RAM; use tiling if activations still do not fit. | [Image VAE](ltx23-image.md#vae-strategies-separate-native-encoder-workload), [video VAE](ltx23-video.md#vae-strategies-separate-native-encoder-workload) |
| Saved INT4 | Selected Windows SDNQ DDO image cases outperform their INT4 leaf controls; this is not BF16 quality equivalence or native VMM support. | [LTX INT4](ltx23-image.md#windows-sdnq-int4), [Qwen INT4](qwen-image-21.md#windows-sdnq-int4) |
| Saved FP8 | Selected LTX video pairs reduce process RSS but take longer and have higher global GPU peaks than BF16. | [Video precision pairs](ltx23-video.md#saved-fp8-weight-only-separate-precision-pairs) |
| WSL | Bounded native or explicit-cap linear can trade time for GPU usage; Windows/Linux registration limits and streamed leaf performance do not transfer automatically. | [Image WSL](ltx23-image.md#wsl-bf16), [video WSL](ltx23-video.md#wsl-bf16), [Qwen WSL](qwen-image-21.md#wsl-bf16) |

These are single workload observations, not causal allocator-only or low-RAM
hardware guarantees. Text encoding can dominate global RAM even when denoise RAM
falls. Compare phase peaks and total separately; scenario links inside each table
provide the relevant limitations.

See [metric definitions](glossary.md), [presets](../user/presets.md) and
[reproduction](reproduction.md).
