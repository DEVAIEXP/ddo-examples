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

Validate output correctness as well as timing. Compare denoise and tracked total
separately; do not sum nested phases or treat missing metrics as zero.
Changes to shared helpers need regression coverage for their affected runners,
not only the model that motivated the change.

Completed selected workloads and their limits are summarized in
[benchmarks by model](../benchmarks/README.md).
See [recording rules](../benchmarks/recording.md) for measurement requirements
and [DDO validation](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/validation.md)
for native runtime/package contracts.

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
