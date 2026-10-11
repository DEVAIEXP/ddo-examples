# DDO benchmark summary

Selected measurements on an 8 GiB GPU. Updated 2026-10-10.
These are single-run observations, not repeated-run statistics or claims for
all Diffusers models. DDO manages weights and offload; model-specific compute
optimizations belong to the independent Modular Diffusers custom blocks.

## Where DDO helped

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

Use the root uv environment and the [image](ltx2-image.md) or
[video](ltx2-video.md) runner guide. The [recording guide](benchmarks.md) defines
required metadata; the [historical manifest](benchmark-reproduction.md)
preserves image workload and binary identities. The
[detailed record](../experiments/dynamic_offload_results.md) retains raw
measurements, limitations, failed candidates and interrupted runs. It is an
engineering evidence log, not the recommended performance leaderboard.
