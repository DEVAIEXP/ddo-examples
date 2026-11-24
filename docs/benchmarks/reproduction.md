# Reproduction

The benchmark controller runs the same public runners used in the
[model guides](../user/README.md). It adds a queue, profile checks, output
validation and resumable sessions. You can use the controller for a series or
run an individual recipe directly.

## Prepare the environment

Follow the [installation guide](../user/setup.md) for your platform.
Run the controller from the repository root with that environment's Python.
Windows, native Linux and WSL use separate environments and native libraries.

Prepare the reviewed custom blocks before measuring. Each recipe uses a pinned
checkpoint revision from the standard Hugging Face cache (`HF_HUB_CACHE`, or
`HF_HOME/hub`, or the user default). `--preflight` and `--execute` download that
revision when it is missing; the measured run itself stays offline. To use a
checkpoint stored elsewhere, pass its folder with `--checkpoint`.
Video recipes also need FFmpeg, FFprobe and the conditioning files distributed
under `benchmarks/inputs/`. The eight-second LTX Video recipe uses
`benchmarks/inputs/ltx23-video-5s/man_guitar.png` and
`benchmarks/inputs/ltx23-video-5s/diff_vocal_song.mp3`; the directory name is
retained for the shared inputs, not the measured video duration.
See the [hardware notes](hardware.md) and
[runtimes and presets](../user/presets.md) for the scope of the published results.

| Recipe | Workload |
| --- | --- |
| `ltx23-image` | BF16, 1280×704, eight steps, seed 43 |
| `qwen-image21` | Base BF16, 1024×1024, 20 steps, seed 42 |
| `ltx23-video-8s` | BF16, 1280×704, eight seconds, 193 frames at 24 fps, eight steps, seed 43, image and audio conditioning |

These are the current measured workloads. The controller's nominal VRAM setting
is not a limit on total GPU memory, and a published result does not guarantee
that a recipe fits another machine.

## Choose a series and preview its commands

List the cases available for the selected platform, model and family:

```shell
python -B benchmarks/run_benchmark.py --platform windows --recipe ltx23-image --family runtime --list-cases
```

Each line shows a case name and what it runs:

```text
official-leaf               Diffusers Leaf reference (not a DDO preset), VAE diffusers_leaf
one-shot-fast               linear / one_shot_fast, transformer linear_runtime, VAE diffusers_leaf
low-ram-safe                linear / low_ram_safe, transformer linear_runtime, VAE diffusers_leaf
low-ram-safe-pre-vae-purge  linear / low_ram_safe, transformer linear_runtime, VAE diffusers_leaf, purge before VAE
nvme-baseline               vmm / nvme_baseline, transformer vmm_runtime, VAE diffusers_leaf
vmm-fast                    vmm / auto, transformer vmm_fast_runtime, VAE native_vmm
```

In the `runtime` family, cases are DDO backend/preset pairs, except
`official-leaf`, which is the official Diffusers Leaf reference. Other families
vary the attention method, component route, weight format or model feature;
their lines name that variation.

Omit `--execute` to preview the planned commands, with every runner argument,
without starting a model:

```shell
python -B benchmarks/run_benchmark.py --platform windows --recipe ltx23-image --family runtime
```

Add `--case one-shot-fast`, for example, to preview or run a single case.
Case names use hyphens; the DDO preset passed to the runner is
`one_shot_fast`.

| Family | Purpose |
| --- | --- |
| `runtime` | Compare the selected DDO presets with official Diffusers Leaf |
| `attention` | Compare transformer attention methods on VMM Fast |
| `precision` | Run the available saved FP8 or SDNQ INT4 comparisons |
| `components` | Compare supported encoder or VAE routes |
| `reuse` | Measure two LTX Image generations in one process |
| `model-features` | Measure the model's cumulative feature series |

Availability depends on the model and platform. Use `--list-cases` or
`--help` for the selected family rather than assuming every combination exists.
The primary video runtime series excludes the experimental NVMe baseline.

## Validate and execute

Preflight checks the environment, selected library, checkpoint and custom-block
identities before a model starts. It does not initialize CUDA. When explicitly
selecting a standalone VAE or FP8 repository, preparation can download its files
to the Hugging Face cache.

```shell
python -B benchmarks/run_benchmark.py --platform windows --recipe ltx23-image --family runtime --preflight
```

After reviewing the preview and a successful preflight, execute the same series:

```shell
python -B benchmarks/run_benchmark.py --platform windows --recipe ltx23-image --family runtime --session-action restart --execute
```

For Linux or WSL, select `--platform linux` or `--platform wsl` and run from
the corresponding environment. Video commands accept `--ffmpeg-bin` with the
directory containing FFmpeg and FFprobe.

The controller checks that the installed native library matches the hash
recorded in its own allocator wheel and stores that hash in the session
identity, so a session cannot continue with a different library. To require a
specific build, also pass its SHA256 with `--expected-dll-sha256` on Windows or
`--expected-library-sha256` on Linux and WSL.

The controller follows the selected recipe's purge policy. Purge is not a
universal prerequisite. Windows recipes that request standby purge need the
required Windows permissions; native Linux and WSL do not use Windows standby purge.

## What happens during a run

1. The controller records the execution identity and checks compatibility.
2. It reuses a compatible final profile or runs the required calibration stages.
   VMM Fast can need two three-step calibrations; the second is skipped if the
   first produces a final profile. Official Leaf needs no DDO profile.
3. It applies the final profile and runs the measured recipe. Calibration time
   is kept separate from inference time.
4. It validates the metrics, effective configuration and generated image or
   video before marking a case complete.

Profiles are specific to their workload and execution route. A different input
shape, precision or feature configuration can require another profile.
Accepted profiles are also exported to the normal DDO profile store for later
runner use; conflicting profiles are preserved rather than silently replaced.

Progress shows completed and total cases, the current case and stage, and the
output directory. A calibration that is unnecessary is reported as skipped.

## Results and continuation

By default, sessions are written under:

```text
outputs/benchmarks/<platform>/<recipe>/sessions/<session-id>/
```

Use `--output-root` to choose another base directory. A session contains its
manifest, individual attempts, console logs, metrics, profiles and generated
media. Keep the manifest together with the referenced files.

At the first failure, the controller preserves the attempt, reports how many
cases remain and stops. Correct the cause before continuing; it does not
automatically retry the model.

To resume a compatible session, repeat the selection arguments, and any
optional path or hash arguments you used, with `--session-action continue`:

```shell
python -B benchmarks/run_benchmark.py --platform windows --recipe ltx23-image --family runtime --session-action continue --execute
```

Completed, verified work is skipped. The first incomplete stage gets a new
attempt. `--session-action restart` creates another session and preserves the
previous one. An interactive invocation can offer this choice; non-interactive
execution must make it explicit. Locks and active-child checks prevent
simultaneous continuation. An incompatible session cannot be resumed by changing
only its manifest.

## Attention and model features

Attention comparisons use VMM Fast with SDPA as the control. For example:

```shell
python -B benchmarks/run_benchmark.py --platform windows --recipe ltx23-image --family attention --attention-runtime vmm-fast
```

The controller looks for a compatible verified SDPA control, or prepares one
before the requested attention cases. `--reference-manifest` selects a
particular control explicitly; an incompatible explicit reference is rejected.
Operator confirmation runs separately from the timed measurement.
Install the selected method's dependencies as described in the
[installation guide](../user/setup.md).

The `model-features` family measures the feature sequence documented in
[cumulative model features](model-features.md). It preserves the control's
resident budget. Video cases use profiles matching their feature shapes;
Qwen shares the control profile for its VAE feature cases. Output comparisons
include the control and the parent of each branch.

## Optional series

### Windows LTX Image process reuse

The `reuse` family measures two generations in one process, with a dedicated
`warm_process` profile:

```shell
python -B benchmarks/run_benchmark.py --platform windows --recipe ltx23-image --family reuse --case warm-process
```

The reported combined total covers both generations. The random generator
advances, so the final PNG is not expected to equal a single seed-43 generation.

<a id="windows-ltx-video-runtime-series"></a>

### LTX Video runtime series

```shell
python -B benchmarks/run_benchmark.py --platform windows --recipe ltx23-video-8s --family runtime --ffmpeg-bin "C:/ffmpeg/bin"
```

The preview shows the selected platform's encoder, transformer and VAE routes,
purge policy and exact-streaming decode settings. See the
[video results](ltx23-video.md) and [manual video recipes](../user/ltx2-video.md).

<a id="windows-ltx-image-sdnq-int4"></a>
<a id="qwen-saved-sdnq-int4-pair"></a>
<a id="ltx-video-saved-sdnq-int4-pair"></a>

### Saved SDNQ INT4 comparisons

For a supported Windows model, select `--family precision`. The saved INT4
snapshot also uses a pinned revision from the Hugging Face cache and is
downloaded by `--preflight` or `--execute` when missing; for the image models
only its packed transformer is fetched. Use `--sdnq-model-path` for a snapshot
stored elsewhere:

```shell
python -B benchmarks/run_benchmark.py --platform windows --recipe qwen-image21 --family precision
```

The pair uses official Leaf and DDO's `sdnq_runtime` with the same saved
quantized transformer and auxiliary weights. DDO uses its own SDNQ profile:
a two-step baseline and an optional two-step validation, without decode.
A dense BF16 or VMM residency profile cannot substitute for it.

Equivalent model-specific commands are in the optional weight-format sections
of the [LTX Image](../user/ltx2-image.md),
[Qwen](../user/qwen-image-21.md) and [LTX Video](../user/ltx2-video.md) guides.
The video INT4 investigation remains unresolved; its availability is not a
performance recommendation.

Saved video FP8 is a separate precision case, `fp8-weight-only`, selected
with `--fp8-model-path`. It uses a separate profile and does not convert weights
during the measured run. See the [video guide](../user/ltx2-video.md).

These examples are previews. Add `--preflight`, then the explicit session
action and `--execute` as above when ready to run.

## Run a recipe manually

The controller is optional. Each model guide contains direct commands with
arguments passed to the public runner:

| Model | Manual recipes |
| --- | --- |
| LTX 2.3 Image | [LTX Image runners and presets](../user/ltx2-image.md) |
| Qwen Image 2.1 | [Qwen runners and presets](../user/qwen-image-21.md) |
| LTX 2.3 Video | [LTX Video runners and presets](../user/ltx2-video.md) |

Choose the same preset, component routes, attention method, inputs, precision,
seed, steps, purge policy and profile used by the benchmark line.
The controller's preview is the exact command reference when reproducing a
particular case. Run any required profile-build commands first, then the
measurement with that final profile.

The Qwen official Leaf recipe on WSL explicitly uses
`PYTORCH_ALLOC_CONF=expandable_segments:False`. Apply this assignment to that
manual recipe only; it does not change other recipes or the shared runner default.

## Publish benchmark tables

Keep the original metrics, manifest and output identity with every accepted
measurement. Compare matching workloads and report the driver, environment,
effective routes, attention method and profile. Verify the generated media
before accepting a result.

Use the current model pages' Full run, Denoising and VAE decode tables.
Full run includes all tracked phases; denoising and VAE decode use their own
measured intervals and peaks. Saving time is excluded from isolated VAE decode.
Never fill a missing phase peak with a global peak, treat missing values as zero,
or combine phases from different runs. See the [measurement glossary](glossary.md).
