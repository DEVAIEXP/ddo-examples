# Pipeline benchmarks

This repository owns model-specific runners, reproduction conditions and results.
DDO owns generic runtime implementation, API documentation and technical gates.

- [Benchmark summary](benchmark-summary.md): start here for selected gains, losses and comparison limits.
- [Benchmarks by model](benchmarks/README.md): shared comparison rules, closed LTX cases and Qwen/platform coverage gaps.
- [Detailed evidence log](../experiments/dynamic_offload_results.md): dated measurements and limitations, including historical and rejected cases.
- [Historical native reproduction manifest](benchmark-reproduction.md): original image fixtures and binary identities.
- [Image runners](ltx2-image.md) and [video runners](ltx2-video.md): current invocation and options.
- [Host migration acceptance](migration-validation.md): environment-specific acceptance, not inherited performance.
- [Linux/WSL continuation](platform-validation-handoff.md): finite post-Windows gates, environment/binary checks and version-cut boundaries.

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

Keep the reader-facing summary separate from the engineering log. Use one
workload and output scope per table, identify the actual official control
(including stream/context differences), and show memory trade-offs as well
as speedups. Failed, interrupted or parity-rejected runs are not leaderboard
entries. Do not attribute custom-model compute improvements to DDO alone.
For Windows performance comparisons going forward, enable standby purge before
the transformer, confirm a successful event and include its cost in total time.
Missing privilege or a skipped purge must be labeled, not counted as executed.
This remains an optional Windows helper, not a Linux benchmark requirement.

Benchmark commands, hypotheses and acceptance criteria require user confirmation
before CUDA execution or parameter changes. Stop at the approved failure or
memory anomaly boundary. Moving documentation authorizes no execution.

Generic defects found through model testing are fixed in DDO with appropriate
technical contracts; model-specific optimizations remain in the host/custom
blocks. Document Windows and Linux acceptance separately.

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
