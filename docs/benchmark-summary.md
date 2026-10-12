# DDO benchmark summary

## Selected Windows SDNQ INT4 image comparison

Official Diffusers 0.41.0; purge confirmed, RSS monitoring, original BF16
auxiliaries and resident untiled VAE. Single-run observations:

| Model / workload | DDO denoise s | Streamed leaf denoise s | DDO total s | Leaf total s |
| --- | ---: | ---: | ---: | ---: |
| Qwen 2.1 base, 512x512 / 20 steps | 18.8056 | 24.2629 | 36.6166 | 39.6060 |
| LTX 2.3 image, 1280x704 / 8 steps | 11.7399 | 21.5413 | 46.5646 | 50.0533 |

DDO reduces selected denoise time by 22.5% and 45.5%, respectively; PNGs
are byte-identical within each pair. DDO can use more phase-local GPU/RAM
for residency, even when global peaks look similar. These are packed SDNQ
runtime comparisons, not native VMM or quantized-versus-BF16 quality claims.
Current LTX INT4 DDO denoise is shorter than its recent BF16 VMM reference,
but the full INT4 pipeline remains slower. See the SDNQ closure section in
[detailed results](../experiments/dynamic_offload_results.md) for memory
scopes, artifacts, limitations and the separate BF16 references.

Selected measurements on an 8 GiB GPU. Updated 2026-10-11.
These are single-run observations, not repeated-run statistics or claims for
all Diffusers models. DDO manages weights and offload; model-specific compute
optimizations belong to the independent Modular Diffusers custom blocks.

Start with the [closed Windows/native Linux LTX matrix](benchmark-matrix.md)
for denoise time, labeled iteration timing, full totals and VAE/global memory.
WSL is explicitly unavailable; full phase tables remain in the evidence log.

## Where DDO helped

The selected Windows [official Diffusers 0.41.0 transition](official-diffusers.md)
passed native BF16 image, video and Qwen generation with byte-identical media.
LTX image/video denoise wall was 13.7679/197.3890 s; Qwen 512-square was
17.6734 s. This closes package compatibility for those cases, not a matched
speedup claim or quantized/Linux/WSL acceptance.

Both comparisons below use LTX 2.3 distilled video, 1280x704, 5 seconds at
24 fps (121 frames), 8 steps, seed 43, real image/prompt, no conditioning
audio and generated output audio. Compare rows within a table, not across
precision backends or output scopes.

### Saved TorchAO FP8: native DDO versus official leaf offload

| Measurement | Official leaf control | DDO native VMM | Observed change |
| --- | ---: | ---: | ---: |
| Denoise | 411.22 s | 235.24 s | 42.8% less time (1.75x) |
| Tracked total, latent output only | 443.44 s | 316.29 s | 28.7% less time |
| Global sampled RSS peak | 27.63 GiB | 25.98 GiB | Lower |
| Global sampled absolute GPU peak | 3.31 GiB | 6.55 GiB | Higher |

This is the same saved FP8 checkpoint and local transformer, without optional
FP8 compilation or converted-weight reuse. The working official control uses
streamless leaf hooks and no_grad; native uses inference_mode and installs its
allocator before loading. Streamed official FP8 attempts failed on the selected
installed stack. This is not a win against a functioning streamed FP8 control,
nor an isolated allocator-only A/B. Both outputs are latents: totals exclude
VAE decode and MP4 encoding. Recorded latent differences were zero for this
selected pair, not evidence of BF16/FP8 quality equivalence.

Evidence: the dated FP8 five-second official group/native A/B in the
[detailed record](../experiments/dynamic_offload_results.md), with artifacts
under the historical host's outputs/fp8_5s_group_no_grad_control and
outputs/fp8_5s_native_control. This pair predates the primary-host migration.

### SDNQ INT4: DDO SDNQ runtime versus official streamed leaf

| Measurement | Official leaf control | DDO SDNQ runtime | Observed change |
| --- | ---: | ---: | ---: |
| Denoise | 649.70 s | 319.64 s | 50.8% less time (2.03x) |
| Tracked full-video total | 713.20 s | 389.37 s | 45.4% less time |

Both completed the selected INT4 video with GPU video decoding. DDO uses
normal PyTorch allocation, not native VMM; SDNQ math remains backend-owned.
The DDO result follows packed-parameter accounting and bounded CPU-pinning
corrections. These sequential results do not isolate each correction; standby
purge was not confirmed and system/cache state was not controlled as a matched
repeated experiment. Do not substitute interrupted INT8/INT4 attempts or the
instrumented 262.5-second leaf diagnostic for this completed control.

Evidence: dated SDNQ INT4 control/correction and completed DDO entries in the
[detailed record](../experiments/dynamic_offload_results.md).

## Current primary-host results: what to choose

### Full video: BF16 versus saved FP8 with native VMM

Same 5-second workload above; native encoder, 1.5 GiB transformer weight and
staging ceilings, 24.625 GiB source/registration ceilings, 1024-token FF and
normalization chunks, phase/compact modulation and in-place residuals.
PyTorch host-cache release and CUDA streamed decode were enabled. Windows
standby purge before the transformer succeeded in both administrative runs.
FP8 compute compilation and converted-weight reuse were disabled.

| Measurement | BF16 VMM | Saved FP8 VMM |
| --- | ---: | ---: |
| Transformer load/prepare + attach | 7.93 s | 42.88 s |
| Denoise | 199.36 s | 218.59 s |
| Decode/save | 26.87 s | 26.94 s |
| Tracked total, including standby purge | 262.47 s | 312.98 s |
| Global sampled RSS peak | 28.42 GiB | 25.38 GiB |
| Global sampled absolute GPU peak | 5.14 GiB | 5.92 GiB |

BF16 is faster here. FP8 reduces sampled process RAM by about 3.04 GiB, but
denoise is 9.6% longer and total is 19.2% longer; most additional setup time is
in loading/preparation. The report does not isolate its cause. Saved FP8
checkpoint size is about half the source BF16 size, but that is not a promise
of half the runtime VRAM. Both completed without selected native faults,
temporary fallbacks, admission failures, quarantine or outstanding tickets.
Video structure was verified; perceptual quality is not scored by these metrics.

Evidence: outputs/migration_video_bf16_standby_purge and
outputs/migration_video_fp8_standby_purge, each with metrics/modular_run.json.
The FP8 runner resolved the model root F:/models/ltx23-distilled-fp8-weight-only.
These are separate sequential runs, not statistical performance estimates.

### BF16 image: current capability, not a Diffusers speedup claim

LTX 2.3 image distilled, BF16 encoder/transformer, real prompt, 1280x704,
seed 43, 8 steps, no PAG: 1.388 s/step CUDA mean for steps 2-8,
15.40 s transformer CUDA phase with dependencies and 35.26 s tracked total.
Sampled absolute GPU peak was 6.88 GiB and RSS peak 27.37 GiB.
The post-reboot primary-host run recovered historical performance and saved
the same PNG as the earlier migration run. Standby purge was skipped for lack
of privilege in this image case. There is no matched official Diffusers control
for this final native encoder workload, so no image speedup ratio is claimed.
Evidence: outputs/migration_image_gate_after_reboot.

## Reading these numbers

- Denoise phase time is not necessarily steady-state time per step. Only the
  image row above explicitly uses completed CUDA event intervals for steps 2-8.
- Tracked totals are runner measurements, not process startup-to-exit latency;
  imports/bootstrap are outside the tracked scope. Latent-only totals must not
  be compared with full-video totals.
- GPU peaks are sampled absolute driver usage, not model residency or a direct
  measurement of Task Manager shared/WDDM occupancy. RSS is process RAM, not
  total system commitment. Requested settings are not executed-route evidence.
- Windows benchmarks going forward enable standby purge before the transformer,
  verify the successful event and include its time in the total. Keep previous
  no-purge results labeled. Purge requires the appropriate Windows privilege,
  remains optional and is not a Linux requirement.
- Reboot/purge repeats show sensitivity to system state. The video purge run
  reduced denoise from 206.50 to 199.36 s while costing 3.31 s; one comparison
  cannot establish causality or variance. Image recovered after reboot without
  any source or dependency change.
- These runs do not establish arbitrary-model/recipe support, Linux parity,
  long-run stability or perceptual equivalence. Model compute improvements and
  memory settings must not be marketed as allocator-only gains.

## Reproduce or inspect

Later merged-source BF16 gates using the local fork7d047b180 and the accepted
workload-specific DLLs recorded image1.370488s CUDA warm mean and38.4748s
tracked total; video200.9234s denoise and264.2061s tracked total. Confirmed
standby purge ran only before the transformer. Both output hashes match earlier
accepted artifacts and selected native failure counters are zero. Video denoise
is0.78% longer than the preceding BF16 purge run; this is compatibility closure,
not a new speedup claim or dependency-pin promotion. See the detailed record
for the exact scope and the non-comparable initial image attempt.

Use the root uv environment and the [image](ltx2-image.md) or
[video](ltx2-video.md) runner guide. The [recording guide](benchmarks.md) defines
required metadata; the [historical manifest](benchmark-reproduction.md)
preserves image workload and binary identities. The
[detailed record](../experiments/dynamic_offload_results.md) retains raw
measurements, limitations, failed candidates and interrupted runs. It is an
engineering evidence log, not the recommended performance leaderboard.

## Windows LTX VAE capacity comparison (2026-10-11)

Matched resident versus no-stream leaf decode, with native DDO transformer
and unchanged model math. Image: BF16, 1280x704, eight steps, real prompt,
seed 43, no PAG. Video: the same resolution/precision/seed/steps, five seconds
at 24 fps, real image/prompt, no input audio and generated output audio.
Both pairs have successful transformer standby purge and byte-identical
saved media, including agreement with their historical accepted outputs.

| Workload / VAE route | Decode (s) | VAE load+decode+cleanup (s) | Decode GPU peak (GiB) | Decode RSS peak (GiB) | Tracked pipeline total (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image / resident | 0.33 | 1.50 | 3.19 | 1.65 | 39.18 |
| Image / leaf | 1.12 | 1.35 | 2.23 | 3.17 | 40.34 |
| Video / resident | 22.66 | 23.87 | 4.81 | 7.16 | 261.67 |
| Video / leaf | 32.88 | 33.23 | 3.02 | 9.27 | 271.83 |

The GPU numbers are sampled absolute driver usage during VAE decode, not
global pipeline peaks. Image load shifts into decode under leaf; compare
combined VAE time as well as decode-only time. Video leaf saves approximately
1.79 GiB of decode GPU usage but adds 10.22 s of decode time. Keep existing
resident defaults; leaf is an explicit capacity alternative, not a speedup.
Global process RSS stays about 27.37 GiB (image) and 28.42 GiB (video), dominated
by earlier phases. These cases do not establish operation on low-RAM machines.

Evidence: outputs/ltx_image_vae_{resident,leaf}_gate and
outputs/ltx_video_vae_{resident,leaf}_gate. Complete per-phase tables, hashes,
workload settings and scope limits are in the dated
[detailed evidence log](../experiments/dynamic_offload_results.md).

## VAE decode capacity recommendations

These recommendations apply to the VAE phase only, not the transformer preset.
Do not extrapolate measured process RSS into an exact system-RAM requirement.

| VAE constraint | Suggested route | Tradeoff / boundary |
| --- | --- | --- |
| Resident decode fits the available VRAM | Keep resident decode | Avoid repeated weight uploads; selected video case is faster. |
| VRAM is tight and enough CPU RAM remains | Try leaf VAE offload | Reduce GPU weight residency at the cost of CPU RAM and uploads. |
| Resident decode does not fit and leaf is not viable with available RAM | Try supported spatial tiling; preserve/use temporal streaming for video | Reduce activation/output-window pressure; measure time and inspect quality. |

Tiling is the next capacity fallback for VAE decode, not a guarantee that any
RAM/VRAM combination will fit. It does not remove model-weight storage or the
CPU memory needed for assembled output frames. Smaller tiles/overlaps can cost
more time and change seams or temporal behavior; retain validated settings or
run a quality gate before recommending a change. If weights or assembled output
still cannot fit, reducing resolution/duration or using a different supported
execution route may be necessary. This recommendation authorizes no new test.

## Native Linux LTX image VAE pair (2026-10-11)

BF16, real prompt, 1280x704, seed 43, eight steps, no PAG, native encoder and
transformer. Same local checkpoint/source setup and candidate `.so` within the
pair; resident versus no-stream leaf VAE only. Both delivered PNGs match exactly.
No standby purge is required/executed on Linux. This is native Linux, not WSL.

| Metric | Resident | Leaf |
| --- | ---: | ---: |
| CUDA steady state, steps 2-8 (s/step) | 1.056121 | 1.056260 |
| Denoise modular-call wall time (s) | 17.1203 | 17.0253 |
| VAE decode (s) | 0.3418 | 0.7482 |
| VAE load+decode+cleanup (s) | 1.1702 | 0.9746 |
| VAE decode sampled absolute GPU peak (GiB) | 2.758850 | 1.821838 |
| VAE decode RSS peak (GiB) | 1.6525 | 3.1866 |
| Tracked pipeline total (s) | 40.2959 | 37.6539 |

Leaf saves about 0.94 GiB of decode GPU usage and uses 1.53 GiB more decode RSS.
The shorter pipeline total also includes faster preceding phases; it is not a
VAE-only speedup. Resident defaults are preserved; leaf is a capacity alternative.
Global GPU/RSS peaks remain approximately 6.55/27.35 GiB, dominated by earlier
components. CPU/GPU memory definitions and single-observation limitations above
still apply. Do not require cross-platform PNG equality or use these rows as
matched evidence for a general Windows/Linux speed ratio.

Evidence: outputs/ltx_image_vae_{resident,leaf}_linux_gate. Exact binary identity,
full phase tables and numerical/output acceptance are in the dated
[evidence log](../experiments/dynamic_offload_results.md). The subsequently closed
Linux video pair is included in the matrix; WSL remains pending, not zero or
implicitly supported.
