# LTX 2.3 image

## Model and requirements

The runners use
[LTX image custom blocks](https://huggingface.co/elismasilva/ltx2.3_image_custom_blocks).
Complete [installation](setup.md) before running.

## Available runners

| Script in `scripts/ltx2_image/` | Purpose |
| --- | --- |
| `run_dynamic_minimal.py` | Small, complete staged DDO example |
| `run_dynamic_modular_distilled.py` | Linear DDO and Diffusers compatibility presets |
| `run_dynamic_modular_vmm_distilled.py` | Native VMM or protected NVMe transformer |
| `run_dynamic_modular_sdnq_distilled.py` | Saved SDNQ transformer |

Inspect options without generating an image:

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_distilled.py --help
```

## Run the model

This linear recipe fixes the published image dimensions, steps and seed:

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_distilled.py --ddo-preset one_shot_fast --width 1280 --height 704 --steps 8 --seed 43 --purge-before-transformer --memory-monitor rss --show-metrics --save-metrics
```

Run from an elevated Windows terminal for standby purge. The real default cat
prompt is used; downloads and a different environment can change elapsed time.
For controlled performance comparisons, use the conditions in
[image benchmarks](../benchmarks/ltx23-image.md), including encoder and VAE strategy.

## Optional LoRA

The image runners accept the same `--lora`, `--lora-weight-name`,
`--lora-revision` and `--lora-scale` controls as the
[video runners](ltx2-video.md#optional-lora). Use a compatible transformer-only
adapter; native VMM execution with LoRA requires separate CUDA validation.
No adapter is loaded by default.

## Runtime and preset selection

See [presets](presets.md). Native VMM requires a compatible installed allocator.
Do not treat a short command with implicit capacities as the published native
benchmark: active weight, staging and retained/registered source limits matter.
`--text-encoder-native-vmm` independently selects the native encoder path.

### Native VMM on Windows

This runner defaults to `vmm/auto` using the public native route, BF16,
1280×704, eight steps, seed 43, CFG 1/rescale 0.7, the real default prompt and
no PAG. The text encoder uses leaf offload and the VAE remains resident.
The following recipe keeps the measured memory limits explicit rather than
making hardware-specific limits universal defaults:

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py `
  --vmm-resident-weight-budget-gb 4 `
  --vmm-staging-buffer-gb 0.5 `
  --ddo-vmm-source-cache-gb 24.625 `
  --ddo-vmm-source-registered-gb 24.625 `
  --purge-before-transformer `
  --memory-monitor rss --metrics-level 2 `
  --show-profile --show-metrics --save-metrics
```

Run from an elevated Windows terminal for purge. Staging totals 1.5 GiB in this
native recipe. Purge and report controls remain optional. These capacities are
not a WSL recipe; see platform-specific benchmark conditions. Use your selected
environment's explicit Python if it is not activated.
Selecting `--ddo-preset nvme_baseline` still selects the protected NVMe route;
it does not enable the public native-fast path. Diagnostic adapter selection
also remains explicit.

## Optional weight formats

For saved SDNQ weights, install the optional SDNQ backend following
[setup](setup.md) and use `run_dynamic_modular_sdnq_distilled.py`.
It does not require the FP8 extra or a native DLL. The image
[benchmark](../benchmarks/ltx23-image.md) compares the selected INT4 DDO and leaf cases.

## Metrics and profile

Add `--show-profile --show-metrics --save-metrics` where supported.
`--output-dir` selects the output location; the runner prints the JSON path.
Use RSS monitoring for comparison with the published RSS tables.

## Runner-specific parameters

See [image parameters](ltx2-image-parameters.md) and
[shared controls](runner-controls.md).
