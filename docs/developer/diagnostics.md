# Diagnostics and reporting

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

SDNQ `--trace-transformer-blocks` installs removable entry/exit hooks, emits
flushed lines and records host durations. It does not replace forwards, retain
input/output tensors or synchronize CUDA. Console and hook costs make this a
diagnostic, not a performance-acceptance setting. Use unbuffered console capture
when investigating interrupted runs.

## Compute options

Compilation, fused RMS/rotary kernels, compact timesteps, modulation, residual
reuse and activation chunking change computation or execution. They are not
passive telemetry and are not requirements for DDO compatibility.
Use the explicit diagnostics entry and record each choice. External benchmark
kernels are not bundled or implied by the optional dependencies.

## Phase boundaries and cleanup

Unload a component, release its intermediates and retire its hooks before
starting the next memory-intensive phase. This does not prove that all cached
backing or cuBLAS workspaces returned to the driver.

Windows standby purge, PyTorch pinned-host cache release, native idle-pool trim
and cuBLAS workspace clearing are different operations. Each must keep live
ownership safe and report what actually occurred. No operation authorizes force
free after unproven CUDA completion or a device reset.

For native lifetime and transport rules, see
[DDO architecture](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/architecture.md).
