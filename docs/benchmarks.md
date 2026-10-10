# Pipeline benchmarks

This repository owns model-specific runners, reproduction conditions and results.
DDO owns generic runtime implementation, API documentation and technical gates.

- [Recorded results](../experiments/dynamic_offload_results.md): dated measurements and limitations.
- [Historical native reproduction manifest](benchmark-reproduction.md): original image fixtures and binary identities.
- [Image runners](ltx2-image.md) and [video runners](ltx2-video.md): current invocation and options.
- [Host migration acceptance](migration-validation.md): environment-specific acceptance, not inherited performance.

## Recording a case

Add new model runners under `scripts/<model>/` and keep their model math in
independent custom blocks. Use the shared telemetry helper and root uv environment.
Record the exact command, checkpoint/revision, precision/backend, prompt/input
conditioning, resolution, duration/frame count/frame rate, seed and steps.
Include host/DDO/allocator/custom-block revisions, installed dependencies,
OS/GPU/driver and selected binary hash where applicable.

Report setup, transformer preparation, denoise, per-step timing when available,
decode/save and full tracked total separately. State whether imports/bootstrap
are excluded. Identify warmup, cache state and instrumentation; do not compare
host callback intervals as if they were synchronized CUDA event timings.
Distinguish baseline-relative GPU deltas from absolute driver usage, and RSS
from USS. Preserve missing historical fields as unavailable, not zero.

Keep outputs, checkpoints and raw exploratory logs ignored. Record artifact
paths/hashes and completion/errors alongside the summary. Preserve original
measurements and dated acceptance, including failed or interrupted cases.
Historical host runs are reference evidence, not validation of migrated runners.
Only label comparisons matched when their recorded conditions justify it.

Benchmark commands, hypotheses and acceptance criteria require user confirmation
before CUDA execution or parameter changes. Stop at the approved failure or
memory anomaly boundary. Moving documentation authorizes no execution.

Generic defects found through model testing are fixed in DDO with appropriate
technical contracts; model-specific optimizations remain in the host/custom
blocks. Document Windows and Linux acceptance separately.
