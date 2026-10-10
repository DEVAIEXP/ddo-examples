# LTX 2.3 video

The runner loads elismasilva/ltx2.3_video_custom_blocks with trust_remote_code.
Reviewed revision: f5c712e93c52283cb101da26d4a4544c808666bf.
Use --custom-blocks modular_diffusers/custom_blocks/ltx2_video for the ignored clone.
For the default Hub source, the runner downloads only Python files and the block
configuration at the reviewed revision, then uses Diffusers' trusted local-code
loader. This avoids its dotted-repository-ID package-name bug and supports offline
snapshot reuse. Model weights are loaded separately in their pipeline phases.

## Inspect before running

The public entry points follow the image naming convention:

| Script in scripts/ltx2_video/ | Responsibility |
| --- | --- |
| run_dynamic_minimal.py | Small staged BF16 CLI, linear DDO, no compute experiments |
| run_dynamic_modular_distilled.py | Linear DDO and compatibility/control presets |
| run_dynamic_modular_sdnq_distilled.py | Experimental SDNQ packed-module runtime; INT4 or INT8 transformer |
| run_dynamic_modular_vmm_distilled.py | VMM auto/native or protected nvme_baseline; explicit saved FP8 |
| run_dynamic_modular_diagnostics_distilled.py | Compilation/kernel experiments, instrumentation and FP8 group control |

Each public runner contains its own argument parsing and generation phases.
The minimal file is a complete, readable example: load trusted blocks, load
components per phase, attach DDO, call the modular pipelines, release weights,
decode and save. It deliberately repeats essential integration rather than
calling a hidden workflow with a mode parameter. It is staged, not an example
that loads the whole pipeline at once. Only telemetry and small compute adapters
are shared; model math remains in the custom blocks.
The historical ltx23_modular_vmm_video.py entry remains an unrestricted
compatibility alias for old commands; new examples use the scoped entry points.
Linear settings apply to the transformer; auxiliary phases retain their existing
group-offload route unless the explicit native encoder option is selected in VMM.
Availability of a linear CLI is not a new generation/performance acceptance.
The linear runner also exposes a transformer override, memory-monitor selection,
sequence length, RoPE reuse and explicit named workload profiling. The minimal
keeps only basic generation, preset and reporting arguments.

    uv run --no-sync python scripts/ltx2_video/run_dynamic_minimal.py --help
    uv run --no-sync python scripts/ltx2_video/run_dynamic_modular_distilled.py --dry-run --ddo-preset auto
    uv run --no-sync python scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py --dry-run

Diagnostics flags are rejected by the normal entries, not silently ignored.
Use the diagnostics entry to reproduce existing compilation/control experiments.

    uv run --no-sync python scripts/ltx2_video/ltx23_modular_vmm_video.py --help
    uv run --no-sync python scripts/ltx2_video/ltx23_modular_vmm_video.py --dry-run --ddo-backend vmm --ddo-preset auto --no-use-audio-condition

These commands do not load weights or initialize CUDA.
Defaults remain 1280x704, 5 seconds, 24 fps, seed 43 and 8 steps, with CPU decode.
No input audio is bundled: enable --use-audio-condition and supply --audio.
Audio loading through TorchCodec also needs compatible shared FFmpeg libraries.
Set DDO_RUNNER_FFMPEG_BIN to their directory on Windows when they are not on PATH.

## Native execution

This is a usage example, not an automatically approved benchmark:

    uv run --no-sync python scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py --ddo-backend vmm --ddo-preset auto --custom-blocks-revision f5c712e93c52283cb101da26d4a4544c808666bf --no-use-audio-condition --show-profile --show-metrics --save-metrics

Use --help for explicit --ddo-vmm-* residency, staging, storage and registration
ceilings. Migration does not enlarge budgets or promote experimental defaults.
--local-files-only requires cached code and weights (or the prepared clone).
--output-dir selects media and JSON reports.

The pipeline stages text, connectors, image VAE, audio VAE, transformer and decode.
--text-encoder-native-vmm requires an explicit positive encoder budget.
Streaming GPU decode is opt-in and retains its existing constraints.

## SDNQ execution

Install the optional backend with `uv sync --extra fp8 --extra kernels --extra sdnq`.
The standalone SDNQ runner retains backend loading, forward and dequantization.
It uses PyTorch allocation and DDO `sdnq_runtime`, not either VMM route. No DLL
path or native bootstrap is needed. Auxiliary components remain BF16 and use
Diffusers group offload; no quantized text encoder is implicitly enabled.
`--bits 8` (default) selects OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int8;
`--bits 4` selects the corresponding dynamic-int4 repository. Dynamic checkpoints
can contain mixed quantization formats; the option labels the checkpoint recipe.
`--model-path` overrides only the transformer source. CUDA loading stays staged.

    uv run --no-sync python scripts/ltx2_video/run_dynamic_modular_sdnq_distilled.py --dry-run --bits 8

The intended comparison is 1280x704, 5 seconds, 24 fps (121 frames), 8 steps,
seed 43, real prompt and image conditioning, without input audio or LoRAs.
Generation needs explicit approval; CPU contracts are not performance acceptance.
Historical image gains do not establish video gains: an older video INT4
residency probe was slower than official leaf offload. Preserve that evidence
and measure this modular runner separately. Reports use the shared RunTracker
schema and bit-specific output names under outputs/ddo_modular_sdnq_video.

For an explicit matched modular control, use
`--ddo-preset diffusers_leaf_offload_compat` in the same SDNQ runner. This routes
the transformer through official Diffusers streamed leaf group offload instead
of resident-module sdnq_runtime. Other components and generation inputs remain
unchanged. DDO/image/native runtime implementations are not modified by this
control. Use `python -u` when collecting the console through PowerShell Tee.
Newline step observations report host callback intervals (not synchronized GPU
completion). CUDA current-stream event intervals are resolved after the existing
phase-end synchronization, without introducing per-step global synchronization.
Output and JSON names distinguish leaf from DDO. A subsequent generic SDNQ fix
requires an image regression comparison before acceptance.

`--trace-transformer-blocks` is a disabled-by-default SDNQ diagnostic. It adds
removable forward hooks to the transformer blocks, emits flushed entry/exit
lines and records host boundary durations through RunTracker. It does not
replace forwards, retain input/output tensors, or synchronize CUDA. Host times
do not establish GPU completion or separately attribute upload/dequantization;
console and hook overhead makes this unsuitable for performance acceptance.
Use `python -u` and preserve the console if an interrupted run cannot save JSON.

## Model-specific options and FP8

Compact timesteps, phase modulation, residual reuse, activation chunking and
RMS/rotary compute variants belong to LTX, not generic DDO; all remain explicit.

    uv run --no-sync python scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py --dry-run --ddo-backend vmm --ddo-preset auto --transformer-fp8-weight-only --model-path F:/models/your-converted-model --local-files-only --no-use-audio-condition


## Consistent decode controls

The minimal, linear, SDNQ, native VMM and diagnostic video runners accept the
same `--decode-device` and `--stream-video-decode` controls. Transformer dtype
or offload route does not select a different VAE algorithm. Existing defaults
remain CPU decode without explicit streaming; tiling/framewise CPU decode is
not GPU streaming and can take substantially longer.

Select `--stream-video-decode --decode-device cuda:0` for the existing temporal
window decoder, with CPU latent storage/stitching and GPU VAE execution. Audio
decoding is staged as audio VAE then vocoder, unloading each before the next.
The minimal/linear/SDNQ runners now expose this same route; that code alignment
is not CUDA acceptance or permission to change validated defaults. No model
math, native runtime, image runner or custom-block clone was modified.
Their JSON `decode_route` records device, streaming selection, planned stages
and completed stages separately. A generation using the new runner route
requires approval.

## Shared telemetry

Image and video import the same scripts/helpers/inference_utils.py. JSON reports
distinguish baseline-relative VRAM from sampled absolute driver usage, identify
the RAM metric, and record sampling costs/failures. Incomplete monitoring raises
an error rather than silently reporting a valid peak. RunTracker consumes the
public DDO report protocol. Pinned-cache observation/release helpers are shared,
but release stays explicit and is not enabled by this reorganization.
Historical reports are not rewritten to add unavailable fields.

Install the fp8 extra. DDO owns physical payload loading/transport; TorchAO and
Diffusers retain backend math. FP8 Linear compilation and feed-forward reuse
are separate experiments, not compatibility prerequisites.
See DDO's docs/fp8-weight-only.md. --fp8-group-offload-control is a linear/off
compatibility comparison, not native VMM. External benchmark kernels are not bundled.
Historical performance evidence remains in DDO's experiments/dynamic_offload_results.md.
