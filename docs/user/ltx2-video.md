# LTX 2.3 video

## Model and requirements

The runners load
[LTX video custom blocks](https://huggingface.co/elismasilva/ltx2.3_video_custom_blocks).
Complete [installation](setup.md), including
[FFmpeg configuration on Windows](setup.md#configure-ffmpeg-on-windows) when
loading audio.

The default checkpoint is `OzzyGT/LTX-2.3-Distilled-1.1`. Missing files download to the
standard Hugging Face cache; later runs reuse cached files. Set `HF_HOME` or
`HF_HUB_CACHE` before running to use your preferred cache drive. Pass
`--distilled-model /path/to/model` to read an existing local folder directly.
Use `--local-files-only` only for offline runs with all required files cached.

## Available runners

| Script in `scripts/ltx2_video/` | Purpose |
| --- | --- |
| `run_dynamic_minimal.py` | Complete staged BF16 example with native VMM transformer |
| `run_dynamic_modular_distilled.py` | Linear DDO and compatibility presets |
| `run_dynamic_modular_vmm_distilled.py` | Native VMM, protected NVMe or saved FP8 |
| `run_dynamic_modular_sdnq_distilled.py` | Saved SDNQ INT4/INT8 transformer |
| `run_dynamic_modular_diagnostics_distilled.py` | Advanced compute and tracing experiments |

Inspect an entry without generating video:

```powershell
python -B scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py --help
```

## Run the model

The Windows BF16 workload uses 1280×704, 193 frames at 24 fps, eight steps,
seed 43, and guitar image and audio conditioning. The default checkpoint is `OzzyGT/LTX-2.3-Distilled-1.1`. Configure FFmpeg before running these
commands. See the [8-second benchmark](../benchmarks/ltx23-video.md#windows-bf16-8-second-pressure-workload)
for the measured routes and component settings.

### Official Diffusers Leaf

```powershell
python -B scripts/ltx2_video/run_diffusers_leaf.py `
  --no-phase-modulation --no-compact-timesteps --no-inplace-residuals `
  --custom-blocks modular_diffusers/custom_blocks/ltx2_video `
  --image benchmarks/inputs/ltx23-video-5s/man_guitar.png --audio benchmarks/inputs/ltx23-video-5s/diff_vocal_song.mp3 `
  --use-audio-condition --prompt 'the man plays the guitar and sings on an old western town' `
  --width 1280 --height 704 --seconds 8 --frame-rate 24 --steps 8 --seed 43 `
  --video-self-attention default --feed-forward-chunk-size 0 --normalization-chunk-size 0 --no-reuse-rotary-embeddings `
  --decode-device cuda:0 --stream-video-decode --video-decoder exact-streaming `
  --memory-monitor rss --save-metrics --output-dir outputs/ltx23-video/leaf
```

### Native VMM: `vmm/auto`

Run the calibration command in elevated PowerShell for the transformer purge.
If it reports `floor_measured`, run that same command once more to test the
ceiling. If it reports `final`, use the generation command.

```powershell
python -B scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py `
  --ddo-backend vmm --ddo-preset auto --ddo-profile auto `
  --no-ddo-vmm-execution-backing-reclaim `
  --ddo-vmm-total-staging-gb 1.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 `
  --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --purge-before-transformer `
  --build-ddo-profile `
  --custom-blocks modular_diffusers/custom_blocks/ltx2_video `
  --image benchmarks/inputs/ltx23-video-5s/man_guitar.png --audio benchmarks/inputs/ltx23-video-5s/diff_vocal_song.mp3 `
  --use-audio-condition --prompt 'the man plays the guitar and sings on an old western town' `
  --width 1280 --height 704 --seconds 8 --frame-rate 24 --steps 3 --seed 43 `
  --video-self-attention default --feed-forward-chunk-size 0 --normalization-chunk-size 0 --no-reuse-rotary-embeddings `
  --decode-device cuda:0 --stream-video-decode --video-decoder exact-streaming `
  --memory-monitor rss --save-metrics --output-dir outputs/ltx23-video/vmm-profile
```

```powershell
python -B scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py `
  --ddo-backend vmm --ddo-preset auto --ddo-profile auto `
  --no-ddo-vmm-execution-backing-reclaim `
  --ddo-vmm-total-staging-gb 1.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 `
  --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --purge-before-transformer `
  --custom-blocks modular_diffusers/custom_blocks/ltx2_video `
  --image benchmarks/inputs/ltx23-video-5s/man_guitar.png --audio benchmarks/inputs/ltx23-video-5s/diff_vocal_song.mp3 `
  --use-audio-condition --prompt 'the man plays the guitar and sings on an old western town' `
  --width 1280 --height 704 --seconds 8 --frame-rate 24 --steps 8 --seed 43 `
  --video-self-attention default --feed-forward-chunk-size 0 --normalization-chunk-size 0 --no-reuse-rotary-embeddings `
  --decode-device cuda:0 --stream-video-decode --video-decoder exact-streaming `
  --memory-monitor rss --save-metrics --output-dir outputs/ltx23-video/vmm-fast
```

`--ddo-profile auto` identifies the transformer workload from the resolution,
frame count and activation settings. Recalibrate when those change. The
three-step calibration stops before decode. Use `--video-decoder streaming-tiles`
when the exact streaming decoder exceeds the available VRAM.

## Runtime and preset selection

### Linear DDO

Calibrate each preset using the command in
[Linear capacity profile](#linear-capacity-profile). Use elevated PowerShell on
Windows for the transformer purge.

**`one_shot_fast`**

```powershell
python -B scripts/ltx2_video/run_dynamic_modular_distilled.py `
  --ddo-backend linear --ddo-preset one_shot_fast --ddo-profile ltx23-video-1280x704-8s-8steps-one-shot-fast `
  --no-phase-modulation --no-compact-timesteps --no-inplace-residuals --purge-before-transformer `
  --custom-blocks modular_diffusers/custom_blocks/ltx2_video `
  --image benchmarks/inputs/ltx23-video-5s/man_guitar.png --audio benchmarks/inputs/ltx23-video-5s/diff_vocal_song.mp3 `
  --use-audio-condition --prompt 'the man plays the guitar and sings on an old western town' `
  --width 1280 --height 704 --seconds 8 --frame-rate 24 --steps 8 --seed 43 `
  --video-self-attention default --feed-forward-chunk-size 0 --normalization-chunk-size 0 --no-reuse-rotary-embeddings `
  --decode-device cuda:0 --stream-video-decode --video-decoder exact-streaming `
  --memory-monitor rss --save-metrics --output-dir outputs/ltx23-video/one-shot-fast
```

**`low_ram_safe`**

```powershell
python -B scripts/ltx2_video/run_dynamic_modular_distilled.py `
  --ddo-backend linear --ddo-preset low_ram_safe --ddo-profile ltx23-video-1280x704-8s-8steps-low-ram-safe `
  --no-phase-modulation --no-compact-timesteps --no-inplace-residuals --purge-before-transformer `
  --custom-blocks modular_diffusers/custom_blocks/ltx2_video `
  --image benchmarks/inputs/ltx23-video-5s/man_guitar.png --audio benchmarks/inputs/ltx23-video-5s/diff_vocal_song.mp3 `
  --use-audio-condition --prompt 'the man plays the guitar and sings on an old western town' `
  --width 1280 --height 704 --seconds 8 --frame-rate 24 --steps 8 --seed 43 `
  --video-self-attention default --feed-forward-chunk-size 0 --normalization-chunk-size 0 --no-reuse-rotary-embeddings `
  --decode-device cuda:0 --stream-video-decode --video-decoder exact-streaming `
  --memory-monitor rss --save-metrics --output-dir outputs/ltx23-video/low-ram-safe
```

The named profiles apply the calibrated resident-module budget. The Windows
8-second comparison used 0 GiB for both linear presets. See [presets](presets.md)
for memory trade-offs and [shared controls](runner-controls.md) for purge options.

On native Linux, the linear runner releases unused PyTorch pinned host cache
after retiring the transformer and before loading the VAE. This default applies
to DDO linear presets; Windows, WSL, compatibility routes and the dedicated
official Leaf runner keep their existing defaults. Use
`--no-release-pytorch-host-cache-before-vae` to disable or
`--release-pytorch-host-cache-before-vae` to enable explicitly. The release time
is included in the tracked total, with counters recorded in the JSON. It is
separate from standby purge. The selected Linux 8-second One Shot Fast run
completed with this cleanup and produced the same MP4 as the Leaf control.
This is one workload observation; other presets and hardware need validation.

## Optional LoRA

Add these arguments to a BF16 recipe to load Soft Enhance at strength 0.8:

```text
--lora vrgamedevgirl84/LTX_2.3_Soft_Enhance_Style_LoRa --lora-weight-name LTX2.3_Soft_Enhance.safetensors --lora-scale 0.8
```

`--lora` also accepts a local safetensors file or directory. Use
`--lora-revision` to pin a Hub revision. No adapter is loaded by default.
For multiple adapters, repeat `--lora` and `--lora-scale` in the same order.
When specifying filenames or revisions, provide one per adapter; use `auto`
for an automatically selected filename or default revision. For example:

```text
--lora vrgamedevgirl84/LTX_2.3_Soft_Enhance_Style_LoRa --lora-weight-name LTX2.3_Soft_Enhance.safetensors --lora-scale 0.8
--lora Lightricks/LTX-2-19b-LoRA-Camera-Control-Static --lora-weight-name ltx-2-19b-lora-camera-control-static.safetensors --lora-scale 0.5
```

Both adapters stay active. Additional adapters can increase memory and execution
cost; loading compatibility does not establish visual quality or CUDA acceptance.
The adapter stays unfused and is loaded before offload; the base checkpoint is
not rewritten. Only transformer adapters are accepted. Saved FP8 plus LoRA is
not enabled by this recipe.

The console confirms the active adapter and strength; saved metrics include
`lora_adapters`. Loading checks do not establish native VMM execution support:
that combination still needs a CUDA comparison. Visual changes alone are not
a reliable activation check.

## Optional weight formats

### SDNQ INT4

Install only the SDNQ optional backend; FP8 is not a prerequisite.
Use `--bits 4` or `--bits 8` in the SDNQ runner to choose the corresponding
saved OzzyGT transformer. Auxiliary components remain BF16.
`--ddo-preset diffusers_leaf_offload_compat` selects its Diffusers leaf control.
Image measurements do not establish video performance.

The following command matches the eight-second INT4 benchmark workload:
Leaf encoder and conditioning, resident VAE with CUDA exact-streaming, default
SDPA, no purge and no model optimizations. Configure FFmpeg as described above.
The runner downloads the selected transformer from
`OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int4` and uses the original BF16
checkpoint for auxiliary components. Cached Hub files are reused;
`--model-path` accepts a local SDNQ transformer directory.

```powershell
python -B scripts/ltx2_video/run_dynamic_modular_sdnq_distilled.py `
  --bits 4 --ddo-backend linear --ddo-preset one_shot_fast `
  --custom-blocks modular_diffusers/custom_blocks/ltx2_video `
  --image benchmarks/inputs/ltx23-video-5s/man_guitar.png `
  --audio benchmarks/inputs/ltx23-video-5s/diff_vocal_song.mp3 --use-audio-condition `
  --prompt 'the man plays the guitar and sings on an old western town' `
  --width 1280 --height 704 --seconds 8 --frame-rate 24 --steps 8 --seed 43 `
  --video-self-attention default --no-reuse-rotary-embeddings `
  --stream-video-decode --video-decoder exact-streaming --decode-device cuda:0 `
  --ddo-profile ltx23-video-1280x704-8s-8steps-sdnq-int4 `
  --memory-monitor rss --save-metrics --output-dir outputs/ltx23-video/sdnq-int4
```

Before generation, run this command with `--build-ddo-profile` and a separate
output directory. It measures the first two steps of the eight-step schedule
without decode. If the report says `probe_pending`, replace that flag with
`--validate-ddo-profile` for a two-step validation. Use the generation command
only after the profile is final. BF16 linear and VMM profiles are separate.

For the Leaf transformer control, replace the preset with
`--ddo-preset diffusers_leaf_offload_compat` and omit `--ddo-profile`.
See the [benchmark recipe](../benchmarks/reproduction.md#ltx-video-saved-sdnq-int4-pair).
This video comparison remains under investigation; the recipe is not a
recommendation for faster generation than BF16.

### FP8 weight-only

Install the optional `fp8` dependencies. The saved TorchAO FP8 weight-only
transformer is published at
[`elismasilva/ltx2.3_video_fp8_distilled_transformer`](https://huggingface.co/elismasilva/ltx2.3_video_fp8_distilled_transformer).
The runner accepts this Hub ID or an existing local directory and resolves it
before measured execution. Hub downloads use the standard Hugging Face cache
(`HF_HOME`/`HF_HUB_CACHE` when configured).

For the eight-second Linux recipe without model optimizations, pass the
parameters directly and configure FFmpeg as described above. Both the original
BF16 checkpoint and the FP8 transformer use their default Hub IDs.

```bash
python -B scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py \
  --ddo-backend vmm --ddo-preset auto \
  --transformer-fp8-weight-only \
  --custom-blocks modular_diffusers/custom_blocks/ltx2_video \
  --image benchmarks/inputs/ltx23-video-5s/man_guitar.png \
  --audio benchmarks/inputs/ltx23-video-5s/diff_vocal_song.mp3 --use-audio-condition \
  --prompt "the man plays the guitar and sings on an old western town" \
  --width 1280 --height 704 --seconds 8 --frame-rate 24 --steps 8 --seed 43 \
  --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 \
  --ddo-vmm-total-staging-gb 1.5 \
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 \
  --no-ddo-vmm-execution-backing-reclaim \
  --video-self-attention default \
  --feed-forward-chunk-size 0 --normalization-chunk-size 0 \
  --no-reuse-rotary-embeddings \
  --stream-video-decode --video-decoder exact-streaming --decode-device cuda:0 \
  --ddo-profile ltx23-video-1280x704-8s-8steps-vmm-fast-fp8-weight-only \
  --memory-monitor rss --save-metrics --output-dir outputs/ltx23-video-fp8-8s
```

The auxiliary pipeline source remains BF16. Model optimizations, compilation,
reclaim and Linux purge are off. Create the FP8-specific capacity profile
before this direct inference command; the benchmark controller handles that
calibration automatically. A BF16 profile is not an FP8 profile.
The flag does not convert BF16 weights or accept arbitrary FP8 checkpoints.
It reduces weight storage; it does not promise FP8 compute or faster generation.
Compilation is not required. See DDO's
[FP8 guide](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/user/fp8-weight-only.md).

## Metrics and profile

Add `--show-profile --show-metrics --save-metrics`.
`--output-dir` selects media and JSON output; the runner prints saved paths.
Use `--memory-monitor rss` for comparisons with the RSS benchmark tables.

## Runner-specific parameters

### Linear capacity profile

Use the complete eight-second command for `one_shot_fast` or
`low_ram_safe` above and add `--build-ddo-profile`. Keep its own profile
name, all eight steps, the guitar image/audio and prompt, default SDPA,
disabled model optimizations, exact-streaming decoder and pre-transformer
purge. Calibration stops before decode/export. Then remove
`--build-ddo-profile` and select another output directory for generation.
Do not use a five-second or feature-enabled profile for this workload.

The Windows comparison applied a zero resident-module budget for these
linear presets. Use a clean shell without inherited DDO budget overrides
so the saved profile remains authoritative. The native source dispatcher
stays enabled by its public DDO default in VMM Fast.
See [profile storage and scope](runner-controls.md#persistent-capacity-profiles-for-linear-runners).

See [video parameters](ltx2-video-parameters.md) and
[shared controls](runner-controls.md). Compute experiments are optional, not
requirements for DDO offloading.
