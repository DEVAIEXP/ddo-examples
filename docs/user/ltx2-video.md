# LTX 2.3 video

## Model and requirements

The runners load
[LTX video custom blocks](https://huggingface.co/elismasilva/ltx2.3_video_custom_blocks).
Complete [installation](setup.md), including
[FFmpeg configuration on Windows](setup.md#configure-ffmpeg-on-windows) when
loading audio.

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

Choose the platform-specific native recipe in
[video benchmarks](../benchmarks/ltx23-video.md). The comparison workload is
1280×704, 5 seconds at 24 fps, 8 steps and seed 43, with guitar-image conditioning.
Input-audio conditioning is a separate workload.

For that Windows native workload, the public command uses the documented
capacities and computation options below. Run in an elevated terminal for the
explicit Windows purge phases. This command generates video; it is not a dry run.

The transformer residency budget comes from a VMM residency profile instead of
a fixed value. Build it once per workload and machine by running the same
command twice with `--build-ddo-profile --steps 3` (about 2 minutes each; each
build stops before decode), then generate normally:

```powershell
$recipe = @(
  "--ddo-backend", "vmm", "--ddo-preset", "auto",
  "--width", "1280", "--height", "704", "--seconds", "5", "--frame-rate", "24", "--seed", "43",
  "--ddo-vmm-total-staging-gb", "1.5",
  "--ddo-vmm-source-cache-gb", "24.625", "--ddo-vmm-source-registered-gb", "24.625",
  "--ddo-vmm-execution-backing-reclaim",
  "--feed-forward-chunk-size", "1024", "--normalization-chunk-size", "1024",
  "--phase-modulation", "--compact-timesteps", "--inplace-residuals",
  "--decode-device", "cuda:0", "--stream-video-decode",
  "--purge-before-run", "--purge-after-text-encoder", "--purge-before-transformer", "--no-purge-after-run",
  "--memory-monitor", "rss", "--metrics-level", "2", "--show-profile", "--show-metrics", "--save-metrics",
  "--ddo-profile", "auto")
# Build 1/2 (structural floor) and 2/2 (hardware ceiling), once per workload and machine.
python -B scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py @recipe --build-ddo-profile --steps 3
python -B scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py @recipe --build-ddo-profile --steps 3
# Generation with the saved budget.
python -B scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py @recipe --steps 8
```

`--ddo-profile auto` names the profile from the resolution, frames and the
activation options, so changing them selects another profile. Without the two
builds the generation still runs, at DDO's structural floor. On the reference
Windows 8 GB machine the builds chose that floor (0.629 GiB; the measured
speedup of a 4.77 GiB ceiling was 0.11 % per GiB, below the 0.5 % threshold).
An explicit `--ddo-vmm-resident-weight-budget-gb` still overrides the profile.

This usage recipe explicitly enables [execution backing reclaim](runner-controls.md#execution-backing-reclaim).
That opt-in does not change the recorded conditions or results of historical benchmark rows.

To condition on input audio, add `--use-audio-condition --audio C:/media/song.mp3`
with your actual file. Keep the same image and audio sources when comparing
durations; `--seconds 10` changes the duration without replacing either source.
Conditioned results are separate from the image-only benchmark above.

For GPU temporal decoding, include these options in the selected recipe:

```text
--decode-device cuda:0 --stream-video-decode
```

They are explicit because the current parser still defaults to CPU decode without
streaming. Temporal streaming reduces the VAE's working set; it does not change
the transformer route. Audio VAE and vocoder execution are staged separately.

## Runtime and preset selection

### Linear recipe

`run_dynamic_modular_distilled.py` defaults to the recorded linear computation
settings: feed-forward and normalization chunks of 1024, phase modulation,
compact timesteps, inplace residuals and RoPE reuse. These are model computation
options, separate from the DDO memory plan. Minimal and SDNQ entries do not
inherit these defaults.

```powershell
python -B scripts/ltx2_video/run_dynamic_modular_distilled.py `
  --ddo-backend linear --ddo-preset one_shot_fast `
  --width 1280 --height 704 --seconds 5 --frame-rate 24 --steps 8 --seed 43 `
  --decode-device cuda:0 --stream-video-decode `
  --purge-before-run --purge-after-text-encoder --purge-before-transformer --no-purge-after-run `
  --memory-monitor rss --show-profile --show-metrics --save-metrics
```

Input audio remains optional: add `--audio C:/media/song.mp3 --use-audio-condition`
to preserve your conditioning source. That changes the workload relative to the
historical image-conditioned benchmark without input audio.

The memory plan remains automatic. The recorded Windows linear run selected
full CPU weight pinning and a zero automatic resident-module budget. This is
not a fixed 6 GiB GPU recipe: available RAM determines whether the same plan is
admissible today. `pin=skip_insufficient_ram` indicates a different plan and can
affect latency. Computation defaults alone do not guarantee the historical time.
CPU decode remains the parser default; the recipe explicitly selects temporal
GPU decoding. Run elevated on Windows to preserve the requested purge phases.

`linear/auto` selects `one_shot_fast`; reducing a resident ceiling does not select
`low_ram_safe`. In particular, `DDO_MAX_RESIDENT_MODULE_BUDGET_GB=3` only caps the
automatic resident-module budget. Full CPU pinning can still select a zero budget.
To request the conservative linear policy, explicitly use
`--ddo-preset low_ram_safe`: its defaults disable automatic budgeting and CPU
pinning and set a 3 GiB resident-module budget. Neither value limits total GPU
memory. Explicit environment settings can override preset defaults; remove a
temporary override after its test. The startup log distinguishes the selected
preset, automatic ceilings and fixed budgets.

Use [presets](presets.md) for backend and memory tradeoffs.
`--distilled-model` selects the BF16 pipeline/auxiliary source; `--model-path` overrides
only the transformer. Native encoder offload is a separate option with its own
capacity. Windows standby purge uses the [shared controls](runner-controls.md).

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

### SDNQ

Install only the SDNQ optional backend; FP8 is not a prerequisite.
Use `--bits 4` or `--bits 8` in the SDNQ runner to choose the corresponding
saved OzzyGT transformer. Auxiliary components remain BF16.
`--ddo-preset diffusers_leaf_offload_compat` selects its Diffusers leaf control.
Image measurements do not establish video performance.

### FP8 weight-only

Install the optional `fp8` dependencies and supply a previously saved supported
TorchAO FP8 weight-only transformer:

```text
--model-path F:/models/your-converted-model --transformer-fp8-weight-only
```

Add these to the VMM recipe, keeping its auxiliary BF16 pipeline source.
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

Calibrate the conditioned five-second guitar workload below. Use your actual
image/audio paths and an elevated terminal on Windows for purge. This runs the
eight denoise steps with calibration enabled and stops before decode/export.

```powershell
python -B scripts/ltx2_video/run_dynamic_modular_distilled.py `
  --ddo-backend linear --ddo-preset one_shot_fast `
  --width 1280 --height 704 --seconds 5 --frame-rate 24 --steps 8 --seed 43 `
  --image C:/media/man_guitar.png --audio C:/media/song.mp3 --use-audio-condition `
  --ddo-profile ltx-video-bf16-1280x704-5s-conditioned --build-ddo-profile `
  --decode-device cuda:0 --stream-video-decode `
  --purge-before-transformer --memory-monitor rss --save-metrics `
  --output-dir outputs/ltx_video_linear_profile
```

The decode options are included for subsequent generation; calibration stops
before decode and does not measure decoder time or memory. To generate using the
profile, remove `--build-ddo-profile` and choose a new output directory. Keep
`--decode-device cuda:0 --stream-video-decode` for the GPU streaming decoder.
The linear computation defaults remain chunks of 1024,
phase modulation, compact timesteps, inplace residuals and RoPE reuse. Recalibrate
under a different name when changing duration, resolution, conditioning or math
options. See [profile storage and scope](runner-controls.md#persistent-capacity-profiles-for-linear-runners).

See [video parameters](ltx2-video-parameters.md) and
[shared controls](runner-controls.md). Compute experiments are optional, not
requirements for DDO offloading.
