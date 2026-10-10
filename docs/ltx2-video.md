# LTX 2.3 video

The runner loads elismasilva/ltx2.3_video_custom_blocks with trust_remote_code.
Reviewed revision: f5c712e93c52283cb101da26d4a4544c808666bf.
Use --custom-blocks modular_diffusers/custom_blocks/ltx2_video for the ignored clone.
For the default Hub source, the runner downloads only Python files and the block
configuration at the reviewed revision, then uses Diffusers' trusted local-code
loader. This avoids its dotted-repository-ID package-name bug and supports offline
snapshot reuse. Model weights are loaded separately in their pipeline phases.

## Inspect before running

    uv run --no-sync python scripts/ltx2_video/ltx23_modular_vmm_video.py --help
    uv run --no-sync python scripts/ltx2_video/ltx23_modular_vmm_video.py --dry-run --ddo-backend vmm --ddo-preset auto --no-use-audio-condition

These commands do not load weights or initialize CUDA.
Defaults remain 1280x704, 5 seconds, 24 fps, seed 43 and 8 steps, with CPU decode.
No input audio is bundled: enable --use-audio-condition and supply --audio.
Audio loading through TorchCodec also needs compatible shared FFmpeg libraries.
Set DDO_RUNNER_FFMPEG_BIN to their directory on Windows when they are not on PATH.

## Native execution

This is a usage example, not an automatically approved benchmark:

    uv run --no-sync python scripts/ltx2_video/ltx23_modular_vmm_video.py --ddo-backend vmm --ddo-preset auto --custom-blocks-revision f5c712e93c52283cb101da26d4a4544c808666bf --no-use-audio-condition --show-profile --show-metrics --save-metrics

Use --help for explicit --ddo-vmm-* residency, staging, storage and registration
ceilings. Migration does not enlarge budgets or promote experimental defaults.
--local-files-only requires cached code and weights (or the prepared clone).
--output-dir selects media and JSON reports.

The pipeline stages text, connectors, image VAE, audio VAE, transformer and decode.
--text-encoder-native-vmm requires an explicit positive encoder budget.
Streaming GPU decode is opt-in and retains its existing constraints.

## Model-specific options and FP8

Compact timesteps, phase modulation, residual reuse, activation chunking and
RMS/rotary compute variants belong to LTX, not generic DDO; all remain explicit.

    uv run --no-sync python scripts/ltx2_video/ltx23_modular_vmm_video.py --dry-run --ddo-backend vmm --ddo-preset auto --transformer-fp8-weight-only --model-path F:/models/your-converted-model --local-files-only --no-use-audio-condition

Install the fp8 extra. DDO owns physical payload loading/transport; TorchAO and
Diffusers retain backend math. FP8 Linear compilation and feed-forward reuse
are separate experiments, not compatibility prerequisites.
See DDO's docs/fp8-weight-only.md. --fp8-group-offload-control is a linear/off
compatibility comparison, not native VMM. External benchmark kernels are not bundled.
Historical performance evidence remains in DDO's experiments/dynamic_offload_results.md.
