# Runner validation

## What to validate

| Change | Relevant checks |
| --- | --- |
| Arguments and preset selection | Parser contracts, precedence, unsupported-option errors and report identity |
| Component lifecycle | Intermediate cleanup, hook retirement, decode routing and absence of retained model owners |
| Model/custom-block integration | State/device contracts, checkpoint loading, schedule and output correctness |
| Reporting | Phase boundaries, memory metric identity, missing-field handling and saved JSON |
| Performance or native execution | Matching workload, selected native artifact and end-to-end output acceptance |

Tests live in `tests/`. CPU contracts use the repository root and
`modular_diffusers` on PYTHONPATH, Python `-B` and
`CUDA_VISIBLE_DEVICES=-1`; they must not initialize CUDA.
Use `-p no:cacheprovider` to avoid writing a pytest cache.
CPU mocks do not establish CUDA execution or installed-wheel compatibility.

## Execution checks

Define the command, workload, fixed memory limits and acceptance criteria before
a GPU comparison. Keep model revisions, precision, inputs, schedule, encoder/VAE
routes and monitoring consistent. Stop on errors or unexplained anomalies;
preserve the report before changing the configuration.

Validate output correctness as well as timing. Compare full-run, denoising and
isolated VAE decode latency and peaks separately. Full run includes all tracked
phases; isolated decode excludes saving. Do not sum nested phases, substitute
aggregate peaks for missing phase data or treat unavailable metrics as zero.
Changes to shared helpers need regression coverage for their affected runners,
not only the model that motivated the change.

Completed selected workloads and their limits are summarized in
[benchmarks by model](../benchmarks/README.md).
See the [reproduction guide](../benchmarks/reproduction.md#publish-benchmark-tables) for measurement requirements
and [DDO validation](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/validation.md)
for native runtime/package contracts.

## Benchmark controller checks

Use the [reproduction guide](../benchmarks/reproduction.md) for the public queue,
preflight and continuation workflow. CPU contracts cover command generation,
profile compatibility, effective routes, dispatcher requirements, output
verification and session locks. Preflight validates identities without CUDA;
neither it nor CPU mocks establishes model execution acceptance.

A completed case requires a successful child exit, valid metrics and verified
media. Preserve rejected attempts. Resume only a compatible session after
resolving the failure; do not edit its manifest to bypass identity checks.
Attention operator diagnostics are separate from performance measurements.
Feature-series acceptance also compares outputs against the control and the
parent of each branch. Pixel equality and visual approval are separate findings.

## WSL staging constraints

Protected NVMe staging uses PyTorch pinned CPU buffers. Native host registration
uses `cuMemHostRegister`; success with one path does not establish capacity for
the other. Storage, registration and GPU ceilings are separate limits.

A protected NVMe slot must fit a complete execution group. The tested LTX image
checkpoint has a maximum dense group payload of 512.50 MiB; two slots require at
least 1025 MiB before overhead. Its protected NVMe route has no accepted WSL
benchmark. Selected Qwen NVMe results do not establish that LTX capacity.

WSL leaf controls use `use_stream=False` because streamed leaf was incompatible
with the tested environment. Native Linux and Windows are separate validation
targets; neither inherits WSL's observed limits.
