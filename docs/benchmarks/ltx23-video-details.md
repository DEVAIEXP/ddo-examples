# LTX 2.3 video — measurement details

[Back to comparison tables](ltx23-video.md).
Scenario links in those tables open the sections below; no local report download
is required to read the results.

**Labels and units:** read the [labels and memory glossary](glossary.md) before
comparing runtime names, phase timings, tracked totals or memory peaks.

## Windows 8-second workload

The [Windows pressure comparison](ltx23-video.md#windows-bf16-8-second-pressure-workload)
uses `OzzyGT/LTX-2.3-Distilled-1.1` snapshot
`f070fb11c2af41e90ad9012d1f495b5131753a4d`, the guitar image and vocals in the
benchmark inputs, and the prompt `the man plays the guitar and sings on an old western town`.
Decode and save covers CUDA `exact-streaming` video decode, audio decode and
saving the output.

### Windows cumulative model features

The [feature observations](model-features.md#windows-observations) use the same
guitar-conditioned 1280×704/193-frame/eight-step/seed-43 workload on installed
RC5, RTX 3060 Ti 8 GiB and driver 617.42. All ten measurements use BF16 and
default SDPA, native encoder, resident CUDA exact-streaming VAE, active source
dispatcher and purge only before the transformer. Reclaim, compilation and
operator tracing are off. Features accumulate in the published order.

Each video case calibrated its own final input-shape profile before measurement.
The control applied 0.6289 GiB from its profile; additions used the byte-derived
0.62890625 GiB recommendation. This 6,711-byte integer-budget difference means
the intended strict equal-configured-budget criterion was not met. Reported
weight residency was identical across measurements. Profiles matched actual
input shapes, requested features had application evidence, and all ten MP4s
passed dimensions/frame-count/audio-stream validation. No positive final native
failure or quarantine counter was observed in the recorded transformer reports.

MP4s through frame modulation are byte-identical to the control. Feed-forward
chunking changes the hash; normalization chunking preserves that chunked output.
Visual acceptance of the changed output remains pending. These single,
calibration-interleaved observations establish neither repeatability nor an
isolated speed gain for each feature. See the matching
[phase measurements](ltx23-video-phases.md#windows-cumulative-model-features).

### Windows 8-second Leaf

The dedicated Diffusers runner uses official streamed Leaf offload for the
transformer and a resident temporally streamed video VAE. No DDO calibration
profile or standby purge is used.

The selected RC5 observation uses driver 617.42 and records video decode,
audio decode, saving and transformer cleanup separately. Video decode covers
only the measured `decode_video` phase; transformer retirement is outside
denoise. The generated 193-frame MP4 matches the other reviewed Windows
Leaf output by SHA-256.

### Windows 8-second One Shot Fast

DDO RC5 resolves `linear/one_shot_fast` to `linear_runtime`. The applied
named profile selects a 0 GiB resident weight budget. The video VAE uses Leaf
with temporal streaming. The only standby purge is before the transformer.
The selected observation records transformer cleanup and video decode separately.
Its MP4 is byte-identical to the selected Leaf reference. Calibration is excluded
from the tracked inference total.

### Windows 8-second Low RAM Safe

DDO RC5 resolves `linear/low_ram_safe` to `linear_runtime`. The applied
named profile selects a 0 GiB resident weight budget. The video VAE uses Leaf
with temporal streaming. The only standby purge is before the transformer.

The selected measurement records transformer cleanup and video decode separately.
Its MP4 is byte-identical to the selected Leaf and One Shot Fast outputs.
Calibration is excluded from the tracked inference total.

### Windows 8-second VMM Fast

DDO RC5 resolves `vmm/auto` to `vmm_fast_runtime`. The applied transformer
profile selects a 0.63 GiB resident budget with 1.50 GiB of staging.
The run used the native VMM text encoder, a resident video VAE and the
native source dispatcher. The only standby purge was before the transformer.
The measurement applied a newly calibrated final profile with matching input shapes. This
default SDPA run is also the Windows VMM Fast attention control.

<a id="windows-native-attention"></a>

### Windows 8-second VMM Fast attention

Select `--platform windows --recipe ltx23-video-8s --family attention
--attention-runtime vmm-fast` with `--case cudnn-sdpa` or `--case flash-attn2`.
Pass the completed VMM Fast runtime manifest with `--reference-manifest`. See
the [benchmark reproduction guide](reproduction.md#windows-ltx-video-runtime-series).

The controller retains the native VMM encoder, the resident video VAE with
CUDA exact-streaming decode, and the standby purge before the transformer.
The transformer uses the same saved 0.63 GiB resident profile and 1.50 GiB
staging budget. All runs use NVIDIA driver 617.42. Only eligible video
self-attention changes. Separate diagnostics confirmed 384 calls to
`aten::_scaled_dot_product_cudnn_attention` and 384 calls to
`flash_attn::_flash_attn_forward` with FlashAttention 2.9.1. The timed runs
had tracing disabled.

The cuDNN and FlashAttention 2 sessions used completed default SDPA references
with matching recorded execution identities and profiles. Their code/environment
identities differ from the SDPA/Sol pair now displayed, so their comparison
with that pair is contextual. Each variant contributed one timed measurement.
Both variant MP4s passed the frame and audio checks and had different hashes
from their controls. Visual equivalence remains unverified.

<a id="windows-sol-attention"></a>

### Windows 8-second VMM Fast / Sol-Attn

Select `--platform windows --recipe ltx23-video-8s --family attention
--attention-runtime vmm-fast --case sol-attn`. The controller verifies an existing
compatible SDPA control or calibrates and measures one before the Sol diagnostic
and measurement. Calibration and the operator diagnostic are outside timed inference.

This Windows RC5 pair used driver 617.42, Torch 2.12.1+cu132, Diffusers 0.41.0,
FlashAttention 2.9.1 and triton-windows 3.7.1.post27. Both runs applied the same
final 0.63 GiB transformer profile, native encoder, source dispatcher, 1.50 GiB
staging, resident VAE and CUDA exact-streaming decode. Standby purge ran only
before the transformer. Other model features remained disabled.

Sol source revision `781e8341fb3b29d3aea5d50265772e9619340b66` uses external
FlashAttention 2 in video layer 0 and Triton Sol in layers 1–47, with `diag`
and `tau=1.0`. Audio, cross-attention and protected paths retain their existing
implementations. The separate diagnostic recorded eight external FlashAttention
calls and 376 Sol dispatch annotations, with zero video fallback. The Sol
annotation identifies dispatch, rather than a separately named device kernel.
Tracing was disabled in the measured runs.

SDPA took 321.35 s to denoise and 384.23 s overall; Sol took 211.65 s and
273.37 s. Both peaked at 8.00 GiB absolute GPU usage. Peak RSS was 33.08 GiB
and 33.06 GiB respectively. The Sol diagnostic and measurement MP4s were
byte-identical and differed from SDPA. All passed 1280×704/193-frame and
finite-audio validation. The user accepted Sol visual quality. This approximate
attention result is one measured workload, not bitwise or general quality equivalence.

## Workload and measurement scope

The primary BF16 runtime comparisons use Distilled BF16, 1280×704,
8 seconds at 24 fps (193 frames), 8 steps, seed 43 and guitar image, audio
and prompt conditioning. The video VAE uses CUDA exact temporal streaming.
Component routes and standby purge policies are specified per scenario.

Checkpoint: `OzzyGT/LTX-2.3-Distilled-1.1`, BF16 snapshot
`f070fb11c2af41e90ad9012d1f495b5131753a4d`. Official Diffusers 0.41.0
in these comparisons. Feed-forward and normalization chunking,
phase modulation, compact timesteps, inplace residuals, RoPE reuse and fusion,
and compilation are disabled. No LoRAs are used. Input audio conditioning
is enabled. The separate FP8 workload has its own measurement settings.

Text encoder, connectors, conditioning preparation, transformer load/prepare,
attachment and denoise are separately monitored `steps`. The text-encoder
phase contains its staged loading/encoding/cleanup, not only the model forward.
The denoise row is `steps[name=denoise]`; it is not a sum of completed
per-step GPU kernels.

Video VAE, audio VAE and vocoder are separate decode stages where the report
provides them. Older FP8 precision-pair reports expose `decode_and_save`
only: that aggregate includes more than video VAE execution. Temporal GPU
streaming uses CPU latent storage/stitching; it is not spatial VAE tiling.

All displayed phase rows are existing observations, not new runs or inferred
phase estimates. Memory comes from each recorded phase's
`peak_vram_absolute_gb` and `peak_ram_gb` with `ram_metric=rss`.
Times use `elapsed_sec`; global peaks and tracked totals come from the same
report as that row. Values are rounded to two decimals for display only.
Missing fields remain unavailable. Phase tables show selected boundaries, not
an additive decomposition; do not add aggregates to their child events or sum
memory peaks. Host callbacks do not establish completed warm GPU timing.
See [metric definitions](glossary.md).

## Platform conditions

The Windows 8-second DDO cases record a standby purge before the transformer.
Official Leaf runs without that purge. WSL auxiliary leaf hooks are non-streamed and native
registration is bounded; those routes are not equivalent to Windows/Linux
streamed controls. Do not treat different platform observations as causal A/Bs.

The Linux control uses NVIDIA driver 615.71.09; Windows and WSL use 617.42.
Primary cases are single observations, not repeated statistical estimates or
low-RAM certification. Matching workload settings do not establish identical
outputs across platforms.

## Linux BF16 scenario

### Linux cumulative model features

The [Linux feature observations](model-features.md#linux-observations) use
the same BF16, guitar-conditioned 1280×704/193-frame/eight-step/seed-43
workload, driver 615.71.09, installed DDO/allocator RC5, Torch 2.12.1+cu132
and Diffusers 0.41.0. Reviewed custom blocks are
`ae42dd1a04346f2f1e4ab11c209ea4af5357dd6e`. Default SDPA, native VMM encoder,
resident CUDA exact-streaming VAE and source dispatcher are fixed. Purge,
reclaim, FP8, compilation and operator profiling are disabled.

Each case has a verified final shape profile. All ten measurements use the
same explicit 675,282,944-byte transformer budget, including the control;
the Windows series' configured-budget rounding difference does not apply.
Encoder residency is 4 GiB; total staging is 1.5 GiB and source cache and
registration ceilings are 24.625 GiB each. Calibration is outside inference.

Linux memory-pressure management terminated the first RoPE measurement after
its progress bar reached eight steps, before its final report. That attempt
is excluded. Continuation reused its calibration and reran the measurement,
which completed. The control was measured before this interruption and all
subsequent rows afterward. The series is not a repeated or order-isolated
performance comparison; no OS memory-policy change or budget increase was
used for continuation. Each row uses one complete successful execution.

All ten MP4s passed 1280×704/193-frame/audio checks. Reported final native
failure and quarantine counters had no positive values. The first eight
files are byte-identical to the control. The final two chunked files are
identical to each other and differ from the control. Their shared output
received good overall visual feedback with mild artifacts; decoded-frame
comparison has not isolated their cause. Output equivalence is scoped to
this workload, not arbitrary shapes or configurations.

See [phase measurements](ltx23-video-phases.md#linux-cumulative-model-features)
for the exact setup, denoise, video/audio decode, cleanup and save intervals.

<a id="linux-leaf"></a>

### Linux / Official Leaf

Select `--platform linux --recipe ltx23-video-8s --family runtime
--case official-leaf` in the benchmark controller. The dedicated
`run_diffusers_leaf.py` entry uses official streamed Leaf hooks for the
text encoder and transformer, with `low_cpu_mem_usage=True`. It uses a
resident video VAE and CUDA `exact-streaming` decode for the video/audio
pipeline. No DDO calibration profile, native allocator installation,
standby purge or model optimization is enabled in the inference child.

The environment uses installed RC5 packages, Python 3.12.12, Torch
2.12.1+cu132, Diffusers 0.41.0 and NVIDIA driver 615.71.09 on an RTX 3060 Ti
8 GiB. Custom blocks are pinned at
`ae42dd1a04346f2f1e4ab11c209ea4af5357dd6e`. The allocator package identity is
verified by the controller; the dedicated Leaf child uses PyTorch allocation.
The checkpoint and conditioning inputs are those specified above.

The controller recorded completion, valid metrics and an MP4 containing
193 frames at 1280×704 and an audio stream. Artifact hashes were verified.
The selected run took 290.26 s to denoise, 26.32 s for video VAE decode,
and 342.97 s overall. Full-run peaks were 7.62 GiB absolute GPU and
38.13 GiB process RSS; video decode peaked at 4.89 GiB GPU and 10.68 GiB RSS.
Allocator startup records `expandable_segments:True`.

<a id="linux-linear"></a>

### Linux / `one_shot_fast`

Select `--platform linux --recipe ltx23-video-8s --family runtime
--case one-shot-fast`. The normal linear runner resolves
`linear/one_shot_fast` to `linear_runtime`. It uses streamed Leaf for the text
encoder and Leaf for the video VAE with CUDA exact-streaming decode. It uses
the same checkpoint, blocks, conditioning, model controls and environment as
the Linux Leaf control. No standby purge is enabled.

The measurement reused a verified successful capacity calibration and applied
its named 0 GiB resident-module profile; it did not repeat calibration. The
transformer used full pinned CPU weights without additional resident modules.
The normal native Linux runner released unused PyTorch pinned host cache
only after transformer retirement and before VAE loading.

The public cleanup report recorded 42,602,807,651 bytes (39.68 GiB) released
in 7.49 s. Live tensors and native source storage are preserved. This time
is included in the 352.56 s full run. Denoise took 274.65 s and video VAE
decode 19.07 s; transformer attachment took 26.22 s. Peak process RSS was
43.40 GiB and peak absolute GPU usage 7.41 GiB. The MP4 matched the Leaf
control by SHA-256. The selected total exceeded Leaf by 9.60 s despite
shorter denoise and decode, with the difference occurring in other phases.
Allocator startup records `expandable_segments:True`. These single
observations do not establish a general speed or low-RAM guarantee.

<a id="linux-low-ram-safe"></a>

### Linux / `low_ram_safe`

Select `--platform linux --recipe ltx23-video-8s --family runtime
--case low-ram-safe`. The completed measurement used the controller's
`--expandable-segments` diagnostic option, supplying
`PYTORCH_ALLOC_CONF=expandable_segments:True` before Python started. Its startup
JSON records that value and no legacy allocator-variable value. This is also
the shared runner default when neither allocator variable is explicitly set.
The selected Leaf and One Shot Fast records also report expandable segments enabled.

The measurement uses `linear_runtime`, no full CPU pinning, a 0 GiB
resident-module profile, default SDPA, exact-streaming Leaf VAE decode and no
standby purge. Recipe and model controls match the Linux control. It reused
the verified successful eight-step capacity calibration; calibration timing
is outside the measured runtime. Pre-VAE host-cache release is enabled.

The runtime completed in 344.17 s, including 296.78 s denoise and 21.86 s
video VAE decode. It released 4,992,876,899 bytes (4.65 GiB) of host cache
in 0.88 s, included in the full run. The host allocator active-byte counter
is unavailable. Peak absolute GPU usage was 7.63 GiB and peak RSS 42.04 GiB.
The verified 193-frame MP4 with audio matched the selected Linux Leaf and
One Shot Fast outputs by SHA-256.

A separately preserved measurement without verified pre-import allocator setup
failed in feed-forward GELU before finishing the first denoise step, requesting
688 MiB with 656.50 MiB driver memory free and 1.02 GiB reserved but unallocated
by PyTorch. This successful measurement does not by itself establish the cause
of that failure. No chunking, budget increase, OS policy change or automatic
retry was used. These single observations are not a general support guarantee.

<a id="linux-native"></a>

### Linux / `vmm_fast`

Select `--platform linux --recipe ltx23-video-8s --family runtime
--case vmm-fast`. The transformer uses `vmm/auto` (`vmm_fast_runtime`) with
source dispatcher enabled, 1.5 GiB total staging and 24.625 GiB source-cache
and registration ceilings. The encoder uses native VMM with its separate
4 GiB resident ceiling. Video VAE decode is resident CUDA exact-streaming;
audio VAE and vocoder remain the existing resident decode stages. Reclaim,
standby purge and model optimizations are disabled.

The first three-step calibration finalized the profile at its minimum
675,282,944-byte resident-weight budget (0.63 GiB displayed), with reason
`no_headroom`. No higher ceiling trial was needed, so the controller skipped
the second planned calibration and ran the eight-step measurement. The raw
capacity classification was `does_not_fit`, reflecting the absence of the
configured headroom margin at the floor; it is not a claim that this completed
run failed. The measurement applied the final profile with matching input shapes.

The installed RC5 artifact, checkpoint, custom blocks and environment match
the Linux controls. Dispatcher state was active, decode completed, audio was
finite and the MP4 contained 193 frames at 1280×704 with stereo audio. Its hash
matched all three Linux controls. The run took 274.26 s denoise, 26.24 s
decode/save and 325.78 s overall, with 7.69 GiB absolute GPU peak and
33.07 GiB peak RSS. This is one completed workload observation, without a
local-VRAM placement or general hardware-support guarantee.

<a id="linux-native-attention"></a>

### Linux / VMM Fast video self-attention

Select `--platform linux --recipe ltx23-video-8s --family attention
--case cudnn-sdpa --attention-runtime vmm-fast` with a completed default SDPA
reference manifest. The comparison reuses the runtime control's final profile,
native encoder, resident video VAE and CUDA exact-streaming decode. The recipe,
checkpoint, blocks, driver 615.71.09, Torch 2.12.1+cu132, Diffusers 0.41.0 and
installed RC5 native library remain unchanged. Purge, reclaim, chunking and
model-specific optimizations remain disabled; no calibration is repeated.

Only eligible video self-attention uses `cudnn-only`; other attention paths
retain their existing implementation. A separate diagnostic confirmed
`aten::_scaled_dot_product_cudnn_attention`. All 48 video layers recorded eight
calls each, with no fallback. The untraced measurement completed in 325.94 s,
including 275.15 s denoise and 26.38 s decode/save. The profiler run is excluded
from those timings.

The diagnostic log contains 838 Driver refusals and 838 successful allocation
recovery notices; the timed log contains 119 of each. Both runs completed all
eight steps without a traceback. These are recovered memory-pressure events,
not evidence of spare capacity or a bound on recovery overhead.

The controller validated complete metrics and an MP4 with 193 frames at
1280×704 and an audio stream. Decoded audio was finite, stereo and 48 kHz.
The diagnostic and timed MP4 hashes match each other and differ from the
default SDPA control. This does not establish numerical or visual equivalence;
visual quality remains unconfirmed.

<a id="linux-flash-attention"></a>

### Linux / VMM Fast / FlashAttention 2

Select `--platform linux --recipe ltx23-video-8s --family attention
--case flash-attn2 --attention-runtime vmm-fast` with the same completed SDPA
reference. This uses FlashAttention 2.8.3 for eligible video self-attention only;
all other recipe, runtime, profile, component and environment settings are those
of the Linux attention comparison above. No calibration is repeated.

The separate operator diagnostic recorded 384 calls to
`flash_attn::_flash_attn_forward`. All 48 video layers recorded eight calls with
no fallback. Other attention paths retained their existing operators. The
untraced measurement took 268.28 s to denoise, 22.56 s to decode/save and
314.43 s overall. The profiler run contributes none of these timings.

The diagnostic log contains 107 Driver refusals and successful recovery notices;
the timed log contains 36 of each. Both runs completed without a traceback.
Recovery counts alone do not establish the cause of a timing difference.
The controller validated metrics and the 193-frame 1280×704 MP4 with audio;
decoded audio was finite, stereo and 48 kHz. Both FlashAttention MP4 hashes
match each other and differ from SDPA and cuDNN. Visual quality remains
unconfirmed; byte differences do not measure perceptual quality.

<a id="linux-sol-attention"></a>

### Linux / VMM Fast / Sol-Attn

Select `--platform linux --recipe ltx23-video-8s --family attention
--case sol-attn --attention-runtime vmm-fast`. The comparison uses the same
Linux SDPA control, final profile, recipe and component routes. The optional
Sol source is Sana revision 781e8341fb3b29d3aea5d50265772e9619340b66, with
Triton 3.7.1 and FlashAttention 2.8.3. Video block 0 uses FlashAttention;
blocks 1 through 47 use Triton Sol, `diag`, `tau=1.0`. Other attention paths
are unchanged. No fused rotary or other model optimization is enabled.

The diagnostic recorded eight external FlashAttention calls and 376 Sol
dispatch annotations, with zero video fallback. The Sol annotation surrounds
the reviewed Triton interface; it is not a device-kernel trace. The separate
untraced measurement took 214.54 s to denoise, 23.11 s to decode/save and
261.78 s overall. Diagnostic JIT/autotuning and profiler timing are excluded
from that measurement.

The diagnostic and timed logs contain 20 and 44 Driver refusals, respectively,
each matched by a successful recovery notice; neither has a traceback. The
controller verified source identity, complete metrics, 193 video frames at
1280×704, an audio stream, and finite stereo decoded audio at 48 kHz.
Both Sol MP4 hashes match each other and differ from the dense variants.
Sol is approximate: media validation does not establish visual quality or
numerical equivalence. User visual review remains pending.

## WSL 8-second workload

The [WSL pressure comparison](ltx23-video.md#wsl-bf16) uses the same checkpoint,
prompt, guitar image and input audio as the Windows 8-second workload and
custom blocks at revision `ae42dd1a04346f2f1e4ab11c209ea4af5357dd6e`.
The WSL environment used DDO and allocator 0.1.0rc5, Diffusers 0.41.0, and
PyTorch 2.12.1+cu132. Windows standby purges were absent. Chunking, phase
modulation, compact timesteps, in-place residuals and RoPE reuse were disabled.
All cases used CUDA `exact-streaming` video decode.

<a id="wsl-leaf"></a>

### WSL 8-second Leaf

The dedicated Diffusers runner used official non-streamed Leaf weight offload
for the transformer and a resident video VAE. No DDO calibration profile applied.
The accepted run took 883.64 s overall, with 7.75 GiB absolute GPU peak and
38.42 GiB peak process RSS. Denoise took 641.02 s (80.13 s/step); video VAE
decode took 19.52 s, with 5.31 GiB GPU and 21.15 GiB RSS peaks.

<a id="wsl-linear"></a>

### WSL 8-second WSL Compat

`--ddo-backend linear --ddo-preset wsl_compat` executed `linear_runtime`.
The applied transformer profile selected a 0 GiB resident module budget.
The video VAE used non-streamed Diffusers Leaf weight offload.
The accepted run took 1960.70 s overall, with 8.00 GiB absolute GPU peak
and 38.39 GiB peak process RSS. Denoise took 1725.82 s (215.73 s/step);
video VAE decode took 19.49 s, with 5.31 GiB GPU and 13.17 GiB RSS peaks.
Calibration is excluded from inference latency. Both this run and the Leaf
control use default SDPA, CUDA exact-streaming decode and no standby purge
on NVIDIA driver 617.42. Their MP4s are byte-identical.
WSL host-state variation limits conclusions from these single observations.

<a id="wsl-native"></a>

### WSL 8-second VMM Fast

`--ddo-backend vmm --ddo-preset auto` resolved to `vmm_fast_runtime`.
The final transformer profile selected a 0.63 GiB resident weight budget at
the calibrated floor. Total staging was 384 MiB. Source cache and CUDA host
registration ceilings were 24.625 GiB and 0.5 GiB, respectively. The text
encoder used Diffusers group offload and the video VAE was resident.
Calibration ran separately from inference. The current runtime row uses the
SDPA control prepared with the bundled Sol source and profile applied at
0.6289 GiB: denoise 461.95 s, tracked total 700.53 s and peak RSS 38.27 GiB.
This is also the reference for the Sol pair. It differs in source/dependency
identity and output hash from the earlier dense-attention series below.
The global profile export collision did not overwrite the existing profile;
both current measurements used the verified session-local profile.

<a id="wsl-sol-attention"></a>

### WSL 8-second VMM Fast / Sol-Attn

The SDPA control and Sol measurement share Python 3.12, Torch 2.12.1+cu132,
DDO/allocator RC5, driver 617.42 and the recorded source, fixture and native
binary identities. Sol uses the reviewed source bundled with the examples.
Only eligible video self-attention changes: block 0 uses external
FlashAttention 2 and blocks 1–47 use Sol Triton with `tau=1` and `diag`.
Audio and cross-attention keep their default backends. There is no purge,
chunking, RoPE reuse/fusion, phase modulation, compilation, FP8 or reclaim.

The separately profiled diagnostic recorded eight external FlashAttention
calls and 376 Sol annotations, with eight calls per layer and zero video
fallbacks. The timed measurement had profiling off. The profile was final,
applied with matching shapes and the same resident budget in both runs;
the source dispatcher was active. Final reported fault/admission/backing/
source/pool failures and quarantine bytes were zero.

SDPA took 461.95 s to denoise, 35.60 s to decode/save and 700.53 s overall;
Sol took 359.22 s, 34.62 s and 600.55 s. Peak absolute GPU usage was 8.00 GiB
for both and peak process RSS was 38.27/37.53 GiB. Sol reduced denoise by
22.2% and tracked total by 14.3% in this single pair. Both MP4s passed
1280×704/193-frame/audio checks; decoded audio was finite, stereo and 48 kHz.
Their hashes differ, and the user accepted Sol's visual quality. These
observations do not establish repeatability or a physical-VRAM placement claim.

<a id="wsl-native-attention"></a>

### WSL 8-second VMM Fast attention

The cuDNN and FlashAttention 2 measurements originally used an SDPA control
of 402.30 s denoise, 641.76 s total and 38.13 GiB RSS. The main attention
table now uses the same newer SDPA reference as the runtime table, while
preserving these two original measurements. The session/profile and
source/dependency differences prevent interpreting their displayed gains
as isolated backend effects. Within their original series, the two variants
kept the profile and component routes, changing
only eligible video
self-attention via `--video-self-attention cudnn-only` or
`--video-self-attention flash-attn2`. Separate diagnostics observed 384
`aten::_scaled_dot_product_cudnn_attention` calls for cuDNN and 384
`flash_attn::_flash_attn_forward` calls for external FlashAttention 2
(2.8.3.post2). The FlashAttention 2 trace recorded zero fallback calls for
the targeted attention. Other attention paths retained the default backend.
All MP4s passed the resolution, frame-count and audio checks.

## FP8 weight-only — 8-second workload

<a id="fp8-windows"></a>

### Windows / VMM Fast / FP8 weight-only

This uses the same original BF16 auxiliary checkpoint, guitar image, conditioned
audio and prompt as the [Windows BF16 VMM Fast row](#windows-8-second-vmm-fast):
1280x704, eight seconds, 193 frames, 24 fps, eight steps and seed 43. The saved
transformer is TorchAO per-row E4M3 weight-only with logical BF16 backend math,
not native FP8 GEMM. SDPA, native encoder and resident CUDA exact-streaming VAE
are preserved. Model features, compilation, conversion reuse and reclaim are
disabled. Only the standby purge before transformer is enabled.

Use the benchmark controller with `--platform windows --recipe ltx23-video-8s
--family precision --case fp8-weight-only --checkpoint <original-bf16-snapshot>
--fp8-model-path elismasilva/ltx2.3_video_fp8_distilled_transformer`. The root must contain the saved `transformer`
config, index and shards; this recipe performs no conversion. The controller
hashes the payload before timing and calibrates a separate FP8 profile.

The installed Windows DDO candidate was 0.1.0rc5+fp8floor.win1, preserving the
existing automatic native-library discovery fix and adding only the Python
component-floor correction. Allocator 0.1.0rc5, Torch 2.12.1+cu132, TorchAO
0.18.0, Diffusers 0.41.0 and NVIDIA driver 617.42 were used on the RTX 3060 Ti.
The native DLL was unchanged. BF16 uses the earlier base RC5 measurement;
package/source identities differ, so this is a comparison of recorded complete
configurations rather than an isolated causal precision experiment.
After acceptance, a local RC5 wheel retained all 24 tested Python files byte
for byte and changed only distribution metadata. The temporary candidate name
identifies the measured run; it is not an additional runtime requirement.

One three-step calibration finalized at 806 MiB: two 370 MiB execution slots,
2 MiB persistent root capacity and 64 MiB arena headroom. `decision=floor`,
`reason=no_headroom` and `fit_class=does_not_fit` describe desired capacity
headroom, not a failed inference. The second calibration was skipped. The
eight-step measurement applied the final profile with matching shapes.
Staging was 1.5 GiB, source cache/registration 24.625 GiB each, and the native
encoder ceiling was 4 GiB. These are component limits, not a total VRAM cap.

Denoise was 330.27 s, decode/save 24.34 s and tracked total 424.02 s. Peak
sampled absolute GPU usage was 8.00 GiB and peak RSS 28.82 GiB. Relative to the
original precision-pair BF16 reference (328.53 s denoise, 390.85 s total and
33.02 GiB peak RSS), total was 8.5% longer, denoise 0.5% longer and RSS 12.7%
lower. These percentages do not use the newer Windows runtime control
(321.35 s denoise, 384.23 s total and 33.08 GiB peak RSS).
Transformer load/preparation plus attachment were 40.89 s versus 7.87 s;
no isolated conversion or kernel cost was measured. Final reported native
fault/admission/backing/source and pool failures and quarantine bytes were
zero; the dispatcher was active. Successful cleanup does not prove zero final
driver usage after process exit. Video metadata and finite audio checks passed,
and the user accepted the visual quality. This is selected workload acceptance.

<a id="fp8-linux"></a>

### Linux / VMM Fast / FP8 weight-only

The RC5 recipe uses the same 1280x704, eight-second, 193-frame, 24 fps,
eight-step, seed 43 guitar image and conditioned audio workload as the
[Linux VMM Fast BF16 row](ltx23-video.md#linux-bf16).
Default video SDPA, native encoder and resident CUDA exact-streaming VAE
routes are preserved. Chunking, phase modulation, compact timesteps, inplace
residuals, rotary reuse/fusion, compilation and reclaim are disabled.
There is no Linux standby purge.

Select `--platform linux --recipe ltx23-video-8s --family precision
--case fp8-weight-only --checkpoint /path/to/original-bf16-snapshot
--fp8-model-path elismasilva/ltx2.3_video_fp8_distilled_transformer` in the benchmark controller.
The converted root must contain `transformer/config.json`, its safetensors
index and shards. The auxiliary BF16 source stays in `--checkpoint`.
The controller hashes the saved shards before timing and records TorchAO,
environment and native-library identity. No conversion is performed.

The FP8 transformer receives its own named VMM capacity profile: up to two
three-step calibrations followed by one eight-step measurement. A final first
calibration skips the second. It does not reuse the BF16 residency budget.
Calibration is excluded from inference time. Staging remains 1.5 GiB, native
encoder ceiling 4 GiB and source cache/registration ceilings 24.625 GiB each;
these component budgets are not a total VRAM limit. The source dispatcher stays
enabled. No FP8 Linear compilation, conversion reuse or cache-trim experiment
is added.

Saved TorchAO per-row E4M3 weight-only retains logical BF16 backend math;
it does not establish native FP8 GEMM. Media validity and visual quality must
be checked independently. FP8 output need not be byte-identical to BF16.
The measured environment used DDO 0.1.0rc5+fp8floor.2, allocator 0.1.0rc5,
Torch 2.12.1+cu132, TorchAO 0.18.0, Diffusers 0.41.0 and NVIDIA driver
615.71.09 on the RTX 3060 Ti. The DDO candidate corrects Python component-layout
floor sizing; the native arena/backing implementation and binary are unchanged.
Automatic FP8 calibration requires that correction; do not assume the unpatched
RC5 package contains it. The final package integration remains pending.

One three-step calibration produced a final floor profile of 806 MiB
(0.7871 GiB): 740 MiB of execution slots, 2 MiB persistent root capacity and
64 MiB arena headroom. Its decision was `floor`, reason `no_headroom`, with
`fit_class=does_not_fit`; this denotes insufficient desired headroom, not a
failed inference. The second calibration was skipped. The full measurement
applied that profile with a matching input shape and completed all eight steps.
The BF16 comparison uses its separately calibrated approximately 0.629 GiB budget.

Measured denoise was 289.78 s, decode/save 21.89 s and tracked total 357.04 s.
Transformer load/prepare took 17.31 s and attachment 7.80 s, versus 1.03 s and
2.94 s in the main BF16 control. These phase timings identify extra setup
cost; they do not isolate conversion or kernel costs. Peak process RSS was
27.27 GiB versus 33.07 GiB for BF16. The 7.68 GiB absolute GPU peak was close
to BF16's 7.69 GiB.

The calibration and measurement logs contain 2057 and 7207 Driver refusals,
respectively, each paired with a successful allocation recovery notice and
no traceback. No final native fault, admission, execution-backing or source-read
failure was reported; quarantined pool bytes were zero. Recovery counts alone
do not explain the timing difference.

The controller verified artifact/profile hashes, actual FP8 configuration,
dispatcher activation and a 193-frame 1280×704 MP4 with audio. Decoded audio
was finite, stereo and 48 kHz. This is selected workload acceptance of the
floor correction and FP8 path, not general allocator or quantization support.
Visual quality remains unconfirmed.

## Residual CPU source detachment

Selected Windows validation on 2026-10-28 used distilled BF16 LTX 2.3, a
1920x1056 image with conditioning audio, 10.33 seconds at 24 fps, seed 2925624786
and the first two scheduler steps of an eight-step schedule. Both complete
transformer forwards contained 48 blocks; no third step or decode executed.
This is separate from the 1280x704 overview benchmark. The transformer executed
`vmm_fast_runtime` with an experimental execution-backing pressure DLL; this does
not promote that artifact or change an installed preset. VAE execution was absent.

The old weight-loading route and 1.5 GiB weight/staging budgets, 24.625 GiB
source/registration ceilings and 4 MiB allocator cache were retained. The single
loading intervention explicitly called the existing guarded
`detach_auxiliary_cpu_sources(enabled=True, max_copy_bytes=16 * 1024**2)` after
attachment and before the first forward. The model was exclusively owned and
in evaluation mode. Native readiness, CPU source ownership, alias/layout and
source provenance guards remained active. There was no compression or activation
eviction. The runner's existing option is
`--detach-transformer-auxiliary-cpu-sources`; this result does not enable it by
default or authorize arbitrary experimental runner controls.

The operation copied 7.88 MiB across 192 residual CPU groups and released their
large checkpoint mapping owners. Within that operation, process private commit
fell by 35.41 GiB. Full video/audio latents after scheduler step index one were
finite and bitwise identical to the preserved accepted reference, with zero
mismatched elements. Native/compute closure and scheduler restoration passed.

Windows process private commit and system commit below are **sampled peaks**,
in GiB, at 500 ms intervals. They are not RSS or local VRAM usage. System totals
include unrelated processes. Headroom is the sampled minimum of system commit
limit minus system commit. See [metric definitions](glossary.md).

| Observation scope | Peak process private commit (GiB) | Peak system commit (GiB) | Minimum system commit headroom (GiB) |
| --- | ---: | ---: | ---: |
| Full observed capture including loading | 48.08 | 71.96 | 23.96 |
| Two-step denoising interval | 42.07 | 59.18 | 36.74 |

The transient loading peak precedes detachment and remains. Correctness-capture
denoising time was 349.46 s, including final output readback, versus 352.25 s for
the earlier reference. This is not a controlled speed comparison or performance
acceptance. The demonstrated benefit is commit relief with selected exact-output
equivalence; no full eight-step/decode, generic CUDA stability, other model,
platform or installed-wheel acceptance is established.

Original evidence remains archived locally under
`storage-detach-two-step-enabled-4a110a9835f74117a631e804434657fa`, manifest
`06e66a9c4a8f87d6287d392c9d94877d9b6229d2e57652295a2d48469266c472`.
Reference: `execution-backing-two-step-numerical-enabled-35ad11099d7546e194864e11916f2dd9`.
These identities are archival evidence, not files required by public users.

## Reproduce this workload

Use the [runner guide](../user/ltx2-video.md) and
[comparison controls](reproduction.md), retaining the model, schedule, inputs,
component routes and the scenario-specific capacities above. Runner defaults
or a different custom-block revision do not guarantee equivalent conditions.
Review trusted custom-block code and keep platform environments separate.
Package/build validation and raw engineering investigations are developer concerns,
not additional timing rows or prerequisites for interpreting these tables.
