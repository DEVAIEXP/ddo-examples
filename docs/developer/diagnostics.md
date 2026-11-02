# Diagnostics and reporting

## Explicit image diagnostic entries

Use `scripts/ltx2_image/run_dynamic_modular_vmm_diagnostics_distilled.py` or
`scripts/qwen_image_21/run_dynamic_modular_vmm_diagnostics.py` for native adapter,
transport, workload-profile or GPU-event investigations. These expose experiment
controls separately from the normal image runners.
The LTX video diagnostic entry remains
`scripts/ltx2_video/run_dynamic_modular_diagnostics_distilled.py`.

Normal image and video recipes use only the canonical shared native route/capacity names.
Diagnostic flags are not required to enable DDO. An available diagnostic control
is not GPU validation or permission to rerun archived experiments.

## Shared reports

`scripts/helpers/inference_utils.py` owns RunTracker and memory collection.
Runners consume DDO's public `DynamicOffloadMetrics`, `as_dict()` and
`format_profile()`; they must not read hook/planner internals.
Reports combine workload identity, phase durations, step observations and
component reports without a second model-specific telemetry schema.

Record the monitor type: RSS and USS are not interchangeable. Absolute sampled
driver GPU usage differs from baseline-relative GPU deltas and PyTorch
allocated/reserved bytes. Missing data is unavailable, never zero.
Sampling can miss short peaks; preserve reported costs, failures and scope.
Incomplete monitoring must not silently become a valid peak.
See [metric definitions](../benchmarks/glossary.md) and
[recording rules](../benchmarks/recording.md).

## Host versus device time

A forward hook or callback measures host boundaries. CUDA work is asynchronous;
those intervals cannot separately attribute upload, dequantization or GPU completion.
Completed CUDA-event intervals are valid only after their completion is resolved.
Do not introduce per-step global synchronization simply to make callbacks look
like GPU timings. Do not sum nested aggregate phases.

Public SDNQ video commands exclude block tracing and workload-profile calibration.
Host hooks and console tracing add overhead and cannot establish synchronized
GPU timing. Use an explicitly scoped diagnostic capture for investigations.

## Compute options

Compilation, fused RMS/rotary kernels, compact timesteps, modulation, residual
reuse and activation chunking change computation or execution. They are not
passive telemetry and are not requirements for DDO compatibility.
Compilation and experimental fused RMS/rotary kernels use the explicit diagnostics
entry. Supported public chunking, timestep, modulation and residual options are
listed in [video parameters](../user/ltx2-video-parameters.md#model-computation).
Record each choice. External benchmark kernels are not bundled or implied by
the optional dependencies.

## Phase boundaries and cleanup

Unload a component, release its intermediates and retire its hooks before
starting the next memory-intensive phase. This does not prove that all cached
backing or cuBLAS workspaces returned to the driver.

Windows standby purge, PyTorch pinned-host cache release, native idle-pool trim
and cuBLAS workspace clearing are different operations. Each must keep live
ownership safe and report what actually occurred. No operation authorizes force
free after unproven CUDA completion or a device reset.

The public Qwen VMM runner exposes `--release-native-cache-before-vae` as an
explicit idle-pool trim request after transformer cleanup and before VAE loading.
`--clear-cublas-workspaces-before-vae` additionally requests experimental
single-GPU workspace cleanup and requires the trim switch. Both default off;
they are lifecycle changes, not passive measurement controls or automatic
consequences of component unload. The LTX image VMM runner exposes neither.
See the [Qwen parameter constraints](../user/qwen-image-21-parameters.md#vae-strategy-and-memory).

For native lifetime and transport rules, see
[DDO architecture](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/architecture.md).
