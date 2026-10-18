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
| `run_dynamic_minimal.py` | Complete staged BF16 example |
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

For GPU temporal decoding, include these options in the selected recipe:

```text
--decode-device cuda:0 --stream-video-decode
```

They are explicit because the current parser still defaults to CPU decode without
streaming. Temporal streaming reduces the VAE's working set; it does not change
the transformer route. Audio VAE and vocoder execution are staged separately.

## Runtime and preset selection

Use [presets](presets.md) for backend and memory tradeoffs.
`--pipeline-path` selects auxiliary weights; `--transformer-path` overrides
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
--transformer-path F:/models/your-converted-model --transformer-fp8-weight-only
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

See [video parameters](ltx2-video-parameters.md) and
[shared controls](runner-controls.md). Compute experiments are optional, not
requirements for DDO offloading.
