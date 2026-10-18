# Reproduce the comparisons

Use the [installation guide](../user/setup.md) and the selected model's runner.
Download weights before timing; `--local-files-only` requires the relevant
model and custom code to be cached.

## Match the workload

Check the [reference hardware and platform limits](hardware.md) and record your
own configuration before comparing timings or memory usage.

Keep the model, precision, prompt, input media, seed, resolution, schedule and
steps unchanged. For video, also match duration, frame rate and input-audio
conditioning. Keep encoder and VAE strategies, compute options and runtime
capacities consistent with the model's table.

| Model | Workload | Conditions and runner instructions |
| --- | --- | --- |
| LTX image | 1280×704, 8 steps, seed 43 | [Image comparison](ltx23-image.md) and [runner guide](../user/ltx2-image.md) |
| LTX video | 1280×704, 5 seconds at 24 fps, 8 steps, seed 43 | [Video comparison](ltx23-video.md) and [runner guide](../user/ltx2-video.md); distinguish image-only from image-plus-audio conditioning |
| Qwen Image 2.1 base | 512×512, 20 steps, seed 42; separate 1024×1024 VAE cases | [Qwen comparison](qwen-image-21.md) and [runner guide](../user/qwen-image-21.md) |

Do not mix Windows, native Linux and WSL observations as one controlled
comparison. Their registration limits, driver behavior and offload settings differ.

## Collect comparable metrics

Use `--memory-monitor rss --show-metrics --save-metrics` where supported.
On Windows, request `--purge-before-transformer` from an elevated terminal
and verify that it succeeded. Purge is a no-op outside Windows.

Compare denoise and tracked total separately. Tracked totals exclude command
startup and downloads. Callback intervals are not completed GPU timings.
GPU absolute peaks, baseline-relative deltas, RSS and USS have different meanings;
see [metric definitions](glossary.md).

Measurements are single observations, not guaranteed performance on other
hardware. Contributor collection procedures belong in
[recording rules](recording.md).
