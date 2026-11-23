# Diagnostics and reporting

<a id="explicit-image-diagnostic-entries"></a>

## Diagnostic entries

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

RunTracker queries and prints the NVIDIA driver version before its tracked
timer starts. The saved `nvidia_driver_versions` field identifies the result;
an unavailable query stays unavailable. Do not move driver discovery into a
measured phase.

Record the monitor type: RSS and USS are not interchangeable. Absolute sampled
driver GPU usage differs from baseline-relative GPU deltas and PyTorch
allocated/reserved bytes. Missing data is unavailable, never zero.
Sampling can miss short peaks; preserve reported costs, failures and scope.
Incomplete monitoring must not silently become a valid peak.
See [metric definitions](../benchmarks/glossary.md) and
[reproduction guide](../benchmarks/reproduction.md#publish-benchmark-tables).

### Isolated inference phases

Use the runner's isolated denoising and VAE decode records for phase latency and
memory. Full-run peaks cover all tracked phases and can occur outside denoising.
Never substitute them for a missing phase peak.

`RunTracker.subphase()` shares the enclosing phase's sampler and adds no CUDA
synchronization or extra polling thread. It records a host-call interval with
sampled absolute GPU usage and the selected process-memory metric. The parent
aggregate includes that interval; do not sum the two.

The official Qwen pipeline's sampling loop and decode are observed through
`scripts/helpers/qwen_pipeline_phases.py`, without replacing its forwards.
Its decode interval includes latent preparation and image postprocessing,
excluding later hook cleanup and file saving. Model-specific staged decode
boundaries are recorded by the other runners. Preserve that scope rather than
describing every decode interval as a pure VAE forward.

## Host versus device time

A forward hook or callback measures host boundaries. CUDA work is asynchronous;
those intervals cannot separately attribute upload, dequantization or GPU completion.
Completed CUDA-event intervals are valid only after their completion is resolved.
Do not introduce per-step global synchronization simply to make callbacks look
like GPU timings. Do not sum nested aggregate phases.

Public SDNQ video commands exclude block tracing. They support their own SDNQ
capacity-profile calibration, separate from the timed generation.
Host hooks and console tracing add overhead and cannot establish synchronized
GPU timing. Use an explicitly scoped diagnostic capture for investigations.

## Compute options

Compilation, fused RMS/rotary kernels, compact timesteps, modulation, residual
reuse and activation chunking change computation or execution. They are not
passive telemetry and are not requirements for DDO compatibility.
Compilation and experimental fused RMS/rotary kernels use the explicit diagnostics
entry. Supported public chunking, timestep, modulation and residual options are
listed in [video parameters](../user/ltx2-video-parameters.md#model-computation).
Record each choice. Reviewed Sol-Attn source is bundled under `kernels/sol_attn`
in the examples checkout, with provenance and license files. FlashAttention 2
and Triton remain separate platform dependencies; the DDO wheel does not bundle
them or the model attention policy. See [optional kernels](../user/kernels.md).

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
