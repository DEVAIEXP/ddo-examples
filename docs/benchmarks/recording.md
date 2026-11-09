# Pipeline benchmarks

This repository owns model-specific runners, reproduction conditions and results.
DDO owns generic runtime implementation, API documentation and technical gates.

- [Benchmark summary](summary.md): start here for selected gains, losses and comparison limits.
- [Benchmarks by model](README.md): shared comparison rules, closed LTX cases and Qwen/platform coverage gaps.
- [Reproduction guide](reproduction.md): current host workloads and recorded binary identities.
- [Image runners](../user/ltx2-image.md) and [video runners](../user/ltx2-video.md): current invocation and options.
- [Host migration acceptance](../developer/validation.md): environment-specific acceptance, not inherited performance.
- [Linux/WSL continuation](../developer/validation.md): finite post-Windows gates, environment/binary checks and version-cut boundaries.

## Capturing the LTX 2.3 SDPA baseline

Run `python -B benchmarks/run_benchmark.py --platform windows --vram-gib 8`
to preview the recipe. The tool uses the selected Python environment and accepts
`--checkpoint`, `--fixtures-dir` and `--custom-blocks` for local inputs. The
fixture directory contains `man_guitar.png` and `diff_vocal_song.mp3`.
Configure the video runner prerequisites from the [setup guide](../user/setup.md).

The preview reports the exact runner arguments, excluding the unique run ID.
Run the reviewed command again with `--execute` to capture one observation.
Use the same platform and elevation for a comparison. Elevated Windows
requests standby purge before the run and before the transformer. Linux and WSL
request neither. Capacities other than 8 GiB use unvalidated resident budgets.

Each execution saves the resolved manifest, stdout and stderr, and the runner's
original JSON under `outputs/benchmarks/<platform>/ltx23-video-5s/`. A completed
result also produces a one-row CSV. Compare this series separately from older
recipes with different denoising controls.

## Windows LTX 2.3 regression control

Use the Windows `ltx23-video-5s` / `sdpa-baseline` capture as the reference
for changes that can affect this path. On the RTX 3060 Ti with NVIDIA driver
617.42, the benchmark script recorded
173.25 s for denoising, 244.13 s total, 7.45 GiB absolute peak VRAM, and
32.98 GiB peak RSS. The run completed all eight denoising steps and saved
121 frames.

Before accepting a feature, run this same control with optional features and
profiling disabled. Use the packaged Windows allocator for this control. Match
the hardware, driver, Python dependencies, resolved
allocator binary, model and inputs, purge policy, budgets and runner flags.
Record the exact binary identities. When the allocator is the component under
test, record both binaries and keep the other conditions matched.

The control must complete, preserve output correctness, and show no confirmed
increase in denoise time, total time, absolute peak VRAM or peak RSS. Compare
the output digest for this deterministic recipe. If a timing or peak moves,
use approved matched control and candidate repetitions to distinguish a
regression from run-to-run variation before accepting the change. A confirmed
regression blocks acceptance. Record opt-in feature measurements separately.

## Table readability

Place a short column explanation before each model's tables and link the
[glossary](glossary.md). Use explicit headers such as **Peak GPU (absolute, GiB)**
and **Peak RAM (RSS, GiB)**, with global versus phase scope stated. RSS alone
names a memory metric, not its peak; do not assume console Peak VRAM is absolute.
Label tracked totals and wall averages separately from completed CUDA timings.
Do not rename historical raw fields or change recorded values to match a new label.

## Matched comparisons, not an exhaustive matrix

Use official Diffusers offload as the control, then the selected DDO runtimes
with the same platform/model recipe. Keep transformer and VAE phase comparisons
separate, with identical timing/memory scopes. Full-pipeline controls without
isolated phases cannot supply missing phase values.

Freeze actual code/environment revisions, model/input identities, precision,
schedule/shape/steps, compute options, encoder/connector lifecycle, VAE route,
streaming/tiling, monitor and Windows purge policy before changing the offload
runtime. Runtime-specific storage/residency settings remain explicit; do not
force WSL and Linux to share an unvalidated registration ceiling.

BF16 and saved FP8 are separate offload comparisons. Same-precision official
versus DDO assesses offload; BF16 versus FP8 under one runtime assesses the
selected quantization recipe, not the allocator alone. SDNQ does not fill FP8.
Do not duplicate interrupted/intermediate candidates in headline tables.

Inventory existing JSON artifacts first. Reuse only genuinely matching records;
propose the minimal missing controls/repeats for approval. Compatibility acceptance
is not matched performance coverage. No automatic 27-case platform matrix or
all-preset expansion is authorized. If a small gain does not justify setup or
integration complexity, recommend the official control rather than oversell DDO.
Hardware recommendations must follow measured GPU/RAM phase tradeoffs, not
assume `low_ram_safe` or FP8 always minimizes memory.

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

## Local benchmark reports

Run the script with Python from the environment you intend to measure.
List the cases, then preview a plan with the local checkpoint and conditioning files.

```text
python -B benchmarks/run_benchmark.py --list-cases
python -B benchmarks/run_benchmark.py --platform windows --family runtime --case sdpa-baseline --vram-gib 16 --checkpoint PATH_TO_SNAPSHOT --fixtures-dir PATH_TO_FIXTURES
```

Place the conditioning image and audio together in
`benchmarks/inputs/ltx23-video-5s/`, or pass their directory with
`--fixtures-dir`. The repository does not distribute these files.
To reproduce the reference inputs, obtain copies with the following hashes.

| File | SHA-256 |
| --- | --- |
| `man_guitar.png` | `213964e961f05ee39a9a21ac07ec93f146e73ab135b79cda57f773928a81db90` |
| `diff_vocal_song.mp3` | `b4fe4ce9572aa43c893c39708d19ad94e37daddad69d9254541da661abe04436` |

The video runner references a [guitar image in the Diffusers examples dataset](https://huggingface.co/datasets/OzzyGT/diffusers-examples/tree/main/ltx2).
Verify the hash of any copy before using it as the reference image.
The reference audio currently requires a separately supplied copy.
For a local comparison, use the same files for every case. The manifest records
both hashes and whether they match the reference pair.
Pass `--custom-blocks` if the local video blocks are outside the default path.
Set the platform to `windows`, `linux`, or `wsl` to match the execution host.
The nominal VRAM setting scales the resident budget from 1.5 GiB at 8 GiB.
The 16 GiB example proposes 3 GiB of resident weights and is unvalidated.
Elevated Windows requests standby purge at startup and before the transformer.

After reviewing the plan, append `--execute` to capture one case.
The console shows runner output as it arrives and prints the summary on
success. The run directory also keeps stdout and stderr separately.
Share `summary.csv`,
`manifest.json`, and `metrics/modular_run.json` from the run directory for
the headline numbers, execution conditions, and per-phase detail.
The default run directory is under `outputs/benchmarks/<platform>/<recipe>/`.
Failed runs retain their manifest and logs.

| Family | Comparison | Where to publish |
| --- | --- | --- |
| `runtime` | Diffusers versus DDO runtimes, or a DDO option such as execution backing reclaim on versus off | Model runtime tables |
| `model-features` | One model-specific option against the matching control | Separate model features page |
| `attention` | Supported attention backends on the same workload, with fused rotary paired within each backend when supported | Model attention tables grouped by platform and workload |

Share a control only when the platform, GPU and driver, runtime, workload,
budgets, and settings match the comparison. Give Qwen Image and LTX Image
their own controls and feature lists as their recipes are added. The current
script runs the LTX video SDPA control, normalization case, and Windows
Official Leaf case. Use `--list-cases --family runtime` to see the Windows
runtime series. One Shot Fast, Low RAM Safe, the NVMe baseline, and VMM Fast
with reclaim are available as command previews pending review. The linear cases
use their preset memory policy. Compare reclaim against the same VMM Fast
control with reclaim disabled. Additional attention and model feature cases
require their own reviewed plans.

The Windows Official Leaf case uses the staged video entry point with the
Diffusers group-offloading API directly. The image model also has a staged
`run_diffusers_leaf.py` entry point, with resident VAE decode.
Both entry points collect phase metrics independently of DDO and omit
standby purge. Match the purge policy across the series before using these
captures as new performance controls.

## Workload selection for VRAM pressure

Keep a fixed workload for comparisons across GPUs. For capacity tests, also
select a workload that brings the official Diffusers Leaf control close to
the device's dedicated memory limit. Use the same workload for every runtime
on that GPU and report it separately from the fixed workload. Compare
completion, denoising time, total time, and absolute peak VRAM. Record the
phase of any failure.

In the Windows LTX 2.3 video pilot on an 8 GiB GPU, the official Leaf control
peaked at 5.88 GiB for 5 seconds and 8.00 GiB for 8 seconds. The latter reached
its peak during denoising and completed all 193 frames. An 8-second run on a
16 GiB GPU may leave more headroom. Measure pressure on that GPU before
selecting a longer workload. The capacity comparison tests whether DDO's
weight residency management leaves room for larger activations while
preserving completion and acceptable runtime.

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
