# Benchmarks by model

Measured model comparisons, grouped by platform. Each model's main page contains
tables and hardware recommendations; scenario names link to their configuration,
measurement scope and comparison limits.

For a short method / latency / peak-memory comparison, start with the
[comparison summary](summary.md#at-a-glance-windows-bf16). Primary model pages
then show a separate table for each phase and finish each platform set with the
measured pipeline total. Choose a component strategy from its own phase table;
do not combine the fastest rows into an unmeasured pipeline result.

| Model | Main comparison | Measurement details |
| --- | --- | --- |
| LTX 2.3 image | [Results and recommendations](ltx23-image.md) | [Scenario details](ltx23-image-details.md) |
| LTX 2.3 video | [Results and recommendations](ltx23-video.md) | [Scenario details](ltx23-video-details.md) |
| Qwen Image 2.1 | [Results and recommendations](qwen-image-21.md) | [Scenario details](qwen-image-21-details.md) |

## Read the tables

See [test hardware and platform limits](hardware.md) for the reference machine
and the separate Windows, native-Linux and WSL environments.

Main tables show denoise wall, seconds per step, tracked total and global sampled
absolute GPU/process RSS peaks. Phase tables include text encoding, transformer
and VAE measurements where recorded. Aggregated phases are labeled explicitly:
their times must not be added to nested events, and phase memory peaks are never
summed. Values use at most two decimals; missing observations are not zero.

The primary rows compare official Diffusers leaf with DDO linear, native fast
VMM and protected NVMe where measured. Separate groups cover different VAE,
precision, resolution or conditioning choices. Only selected successful results
are shown; single observations are not statistical or all-model certification.
Different drivers, caches, component routes and platform limits prevent causal
cross-platform rankings. Low process RSS does not establish fit on low-RAM hardware.

## Choose a route

Use the model's recommendations rather than choosing a preset by name alone.
Transformer speed, encoder memory and decoder capacity can favor different
component strategies. Tiling is a VAE activation fallback when decode does not fit,
not a guarantee that model weights or assembled outputs fit.

- [Comparison summary](summary.md)
- [Preset/runtime mapping](runtimes.md)
- [Metric definitions](glossary.md)
- [Reproduce the comparisons](reproduction.md)
- [Contributor recording rules](recording.md)
