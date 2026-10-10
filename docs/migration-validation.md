# Host migration validation

## Selected SDNQ closure (2026-10-10)

The corrected generic DDO SDNQ preparation passed 85 CPU tests and two
subtests. Its capacity estimate now accounts for packed parameters/scales,
and CPU pin selection/accounting respects resolved capacity limits.
The approved INT4 image gate completed eight steps and saved a 1280x704 PNG;
this is functional acceptance, not matched before/after performance parity.
The approved INT4 video calibration saved a zero-extra-resident profile.
Reusing that profile completed five seconds at 24 fps/121 frames, 1280x704,
seed 43 and eight steps with real prompt/image, no source audio and generated
audio. GPU streaming decode and frame/audio checks passed; exit code was zero.
Denoise: 319.6422 s; decode/save: 28.0043 s; tracked total: 389.3677 s.
The recorded official streamed leaf control took 649.6985 s denoise.
These selected observations do not establish arbitrary-model/duration support
or numerical/visual parity. Native BF16 VMM and saved FP8 video migration
remain separate unexecuted gates. The dated records below retain their original
checkpoint status; see [current results](../experiments/dynamic_offload_results.md)
and the [benchmark guide](benchmarks.md) for evidence ownership and limitations.

## Didactic runner ownership (2026-10-10)

Supersedes the shared phase-dispatcher organization below. Every public video
runner now defines its own parser and phase orchestration. The complete minimal
example intentionally repeats loading, modular execution, DDO attachment and
retirement, and staged decoding. The internal video_workflow dispatcher was
removed. Telemetry and small compute adapters remain shared.

The consolidated CPU suite passed 22 tests in 55.842 seconds, including nine
guarded runner-help subtests. New contracts require public runner ownership,
matching basic generation defaults, and explicit linear workload profiling.
The existing telemetry, native bootstrap, route separation and trusted-code
contracts also passed. No CUDA generation or performance acceptance is implied.

## Shared helpers and scoped video entries (2026-10-10)

The CPU suite passed 19 tests, including nine guarded runner-help subtests,
in 64.116 seconds after consolidating telemetry and separating video entries.
Checks cover shared report fields, absolute versus baseline-relative VRAM,
explicit monitor failures, unchanged native defaults, CLI separation and native
bootstrap ordering. Existing trusted-code/no-weight loading checks also passed.
The installed TorchAO import warning remains; this is not FP8 acceptance.
No CUDA generation, allocator rebuild or performance benchmark was performed
for this reorganization. Previous generation evidence is historical evidence,
not automatic acceptance of the reorganized host.

Date: 2026-10-10. Platform: Windows. Environment: root uv-managed .venv,
Python 3.12.11, Torch 2.12.1+cu132, torchvision 0.27.1+cu132,
transformers 5.15.1, TorchAO 0.18.0, triton-windows 3.7.1.post27,
cuda-core 1.2.0 and cuda-bindings 13.4.1.

## Selected source revisions

- Diffusers: e71ebc5858961e2a37d6a7b161d00ae0063c3ab7 (Git dependency).
- Image custom blocks: a6e6f865dc6f8decd7ff681e91969bf4b07e9f21.
  Compared with the original image host's 5e3ff89 revision, only README changed.
- Video custom blocks: f5c712e93c52283cb101da26d4a4544c808666bf.
- DDO and allocator: editable sibling dependencies, resolved in uv.lock.

The independent clones are ignored, not submodules. No historical host is a
dependency. Original runners were copied rather than moved or deleted.

## CPU acceptance

With CUDA_VISIBLE_DEVICES=-1, PYTHONDONTWRITEBYTECODE=1,
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and HF_MODULES_CACHE=.hf_modules_cpu:

    .venv/Scripts/python.exe -B -m pytest tests/test_migration_cpu.py -q -p no:cacheprovider

Result: 11 passed, 5 subtests passed, 1 third-party TorchAO SyntaxWarning,
86.84 seconds. Contracts cover manifest/layout, clone separation, all five runner
help commands, video defaults/dry plan, explicit audio, FP8 opt-in boundaries,
same-namespace trusted code loading, no-weight pipeline construction and
absence of historical host imports.

    .venv/Scripts/python.exe -B tests/probe_hub_code_cpu.py
    .venv/Scripts/python.exe -B tests/probe_hub_code_cpu.py --local-files-only

Both real Hub code-only probes passed at the selected video revision:
weights_loaded=false and cuda_initialized=false. Each checked transformer/helper
class identity and construction without checkpoint components.
SSLKEYLOGFILE was unset only in the diagnostic process; TLS verification was
not disabled. Existing warning messages about TorchAO tensor imports and guider
APIs are not proof of an FP8 checkpoint failure or acceptance.

The direct remote loader initially failed because this Diffusers version treats
the dot in the Hub repository name as a Python package separator. The host now
resolves an immutable code-only snapshot with huggingface_hub, then calls
Diffusers' standard trust_remote_code local loader. Offline snapshot reuse
passed. No Diffusers source or inference forward was modified for this fix.

## Preserved boundaries and next gate

Image runner math/settings and video helpers are preserved. Changes are limited
to host paths, installed allocator imports, help-mode bootstrap suppression,
code-source resolution and explicit conditioning audio instead of an old local
asset. The new video default has no audio conditioning until requested.

No model weights, media, CUDA inference, timing benchmark, native rebuild or
DLL installation was performed. Do not claim output/performance parity from
these CPU checks. Automatic DLL discovery is not verification that the discovered
binary matches an accepted workload checkpoint.

Before generation, present one consolidated command using the previously accepted
binary, capacities, model/input files and optimization settings for approval.
Stop on the first ownership, memory or output anomaly. Native Linux remains a
separate environment and platform gate; consult DDO's Linux handoff.

## Approved Windows BF16 image execution — 2026-10-10

After the CPU-only sequence above, the user separately approved one image gate
from the new root .venv. The exact command used the copied image VMM runner,
native encoder and transformer, real default cat prompt, 1280x704, seed43,
eight steps, one generation, no PAG, local files only, RSS monitoring, CUDA
timing and saved metrics. Active budget4GiB, total staging1.5GiB (historical
image host argument0.5GiB times three), source/registration24.625GiB.
No residency calibration, native rebuild or DLL promotion.

The selected historical DLL was explicitly fixed rather than using automatic
discovery for this comparison. SHA256 before and after:
755fcdd8dd1a72b371402750c071ca4b2146c0ac6bfe6b94c9b78f50f1ebb729.

Functional gate PASS: exit0 and plausible saved 1280x704 PNG, zero selected
fault/admission/read/plan-mismatch/pool-quarantine counters and temporary
fallbacks, final live tickets/transfer-owned events zero. Encoder collection31
to zero, transformer736 to zero; encoder cache process committed/registered
bytes zero before denoise. All phase memory monitors report null error_type.
Standby purge safely skipped for the same missing Windows privilege as the
historical gate. This is not pixel-equality validation.

| Measurement | New host | Historical staged Windows gate |
| --- | ---: | ---: |
| CUDA transformer warm steps2-8 | 1.524931 s/it | 1.395602 s/it |
| CUDA transformer phase | 18.657049 s | 16.498256 s |
| Encoder phase | 20.6153 s | 12.6624 s |
| RunTracker total | 50.7519 s | 38.5135 s |
| Sampled peak RSS | 27.4590 GiB | 27.3479 GiB |

GPU absolute sampled peak7.716034GiB, baseline1.026855GiB and peak delta6.6892GiB
are distinct quantities, not shared-WDDM accounting. The historical approximate
absolute peak was6.97GiB. Total excludes interpreter/import/bootstrap startup.
Warm timing increased approximately9.27% and total approximately31.78%; do not
accept performance parity or attribute the difference to migration without a
matched contemporaneous control. No extra CUDA run followed.

Ignored artifacts: outputs/migration_image_gate/console.log, its PNG and metrics/.
PNG SHA256: 877876a4a8e1143a59e51a737ef3eb7f05a27980a6a44d20fed80d2ab581cced.
JSON SHA256: c422681a8a9d4a355cc0298b28762e58a2ad28e9d5729a03dc2e0d4a6cb30dcf.
Video and FP8 generation in this new environment remain separate unexecuted gates.

## SDNQ video integration checkpoint

The standalone SDNQ video runner now selects the cached dynamic INT4/INT8
checkpoint recipes and forces the existing `sdnq_runtime` for the transformer.
Backend weight loading and forwards remain intact; auxiliary components remain
BF16 with group offload. This addition changes neither DDO nor the native binary.

CPU validation: 27 tests passed in 57.066 seconds, including five new SDNQ
contracts and help checks for the new runner. `uv pip check` passed for 68
packages. SDNQ 0.2.7 imports and its Diffusers quantizer is registered with CUDA
initialization explicitly blocked. The CPU backend import reports unavailable
MSVC/OpenVINO acceleration and falls back to eager; this is not a CUDA kernel
or generation acceptance. No model weights or CUDA generation were executed.
The 5-second video performance comparison remains pending user approval.

The subsequently approved INT8 5-second/8-step run loaded all staged components
and attached `sdnq_runtime`, but was interrupted at the user's request before
the first denoise progress callback. No complete latency or video acceptance
was obtained. The historical SDNQ video comparison used INT4; this INT8 run is
not its matched rerun. The ignored partial diagnostic is preserved at
outputs/ddo_modular_sdnq_video/int8-interrupted-diagnostic.md. The process ended
and GPU allocation returned to idle. INT4 execution requires separate approval.

The separately approved INT4 run was also interrupted by the user after more
than eight minutes in denoise. No complete generation or step latency was
accepted. Preserve outputs/ddo_modular_sdnq_video/int4-console.log and
int4-interrupted-diagnostic.md locally. Redirected progress may be buffered;
the missing console steps do not establish first-step duration. Read-only
review identified uncalibrated activation headroom and an early CPU-parameter
return before SDNQ pinning as investigation candidates, not proven causes.
Neither runtime nor allocator code was changed. New CUDA comparisons remain
subject to explicit approval.

The approved INT4 official-leaf control was interrupted at the user's request
after three host callback intervals of45.4080,50.9061 and308.5511seconds.
These are unsynchronized host observations, not resolved CUDA step latency.
No video or complete benchmark was accepted. Preserve the ignored
outputs/ddo_modular_sdnq_video/int4-leaf-console.log and
int4-leaf-interrupted-diagnostic.md. This control exhibited slowdown without
DDO SDNQ persistent residency; its cause remains unresolved. DDO/native/image
runtime implementations were unchanged. The seven selected CPU contracts for
the explicit leaf route and step observer passed before execution.

## SDNQ INT4 leaf block diagnostic — 2026-10-10

An approved Nsight 2025.6.3 capture stopped after37.97seconds, before denoise.
Its diagnostics warn that the installed CUDA13.4 driver is unsupported by that
profiler build; this warning alone does not establish the cause of termination.
The subsequent direct runner failed while loading the input image with
`OPENSSL_Uplink: no OPENSSL_Applink`. A separately approved retry removed
SSLKEYLOGFILE only from the child environment, preserving TLS validation.

The retry used INT4, official streamed leaf,1280x704,5seconds,24fps,121frames,
8steps,seed43, the default real prompt/image, no input audio and opt-in host
block boundary hooks. No DDO, native allocator or image runner was modified.
Three new tracing CPU tests and seven existing SDNQ runner tests passed.
The hooks retain no input/output tensors, replace no forward and add no CUDA
synchronization. Hook/console overhead means this is diagnostic evidence,
not an uninstrumented performance acceptance.

All384 block invocations completed. The former third-step slowdown did not
reproduce. Current-stream CUDA event intervals, resolved after the existing
phase-end synchronization, were42.1181,30.8898,29.2081,29.9113,27.0139,
30.9433,32.7114 and39.4161seconds. RunTracker denoise was262.5seconds;
sampled peak VRAM delta6.97GiB and peak RSS17.30GiB. These are not absolute
WDDM/shared-memory measurements. At step8, block42 took5.8436seconds on the
host, versus0.5613seconds at step3; adjacent blocks also slowed. This does not
isolate transfer, dequantization or driver paging as the cause.

The historical INT4 leaf denoise213.9seconds is not a matched contemporaneous
control. Do not attribute the current difference or disappearance of the large
stall to tracing or SSL configuration. At this checkpoint, CPU decoding is
still active: no completed video or full-pipeline acceptance is claimed.
Console evidence is preserved locally in
outputs/ddo_modular_sdnq_video/int4-leaf-block-trace-retry.log.
