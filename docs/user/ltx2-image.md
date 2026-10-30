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

### Linear DDO: `one_shot_fast`

This linear recipe fixes the published image dimensions, steps and seed:

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_distilled.py --ddo-preset one_shot_fast --width 1280 --height 704 --steps 8 --seed 43 --purge-before-transformer --memory-monitor rss --show-metrics --save-metrics
```

Run from an elevated Windows terminal for standby purge. The real default cat
prompt is used; downloads and a different environment can change elapsed time.
For controlled performance comparisons, use the conditions in
[image benchmarks](../benchmarks/ltx23-image.md), including encoder and VAE strategy.

### Native VMM: `vmm/auto` on Windows

Install the matching native allocator wheel from the [setup guide](setup.md).
Run this command from the examples checkout in your activated environment:

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py `
  --ddo-backend vmm --ddo-preset auto --preset one_shot_fast `
  --ddo-public-native-runtime `
  --width 1280 --height 704 --steps 8 --seed 43 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --vmm-resident-weight-budget-gb 4 `
  --ddo-vmm-total-staging-gb 1.5 `
  --ddo-vmm-source-cache-gb 24.625 `
  --ddo-vmm-source-registered-gb 24.625 `
  --ddo-vmm-page-mb 32 `
  --purge-before-transformer `
  --memory-monitor rss --metrics-level 2 `
  --show-profile --show-metrics --save-metrics
```

This selects `vmm_fast_runtime` for the BF16 transformer. `--preset one_shot_fast`
provides base settings; it does not switch this command to the linear runtime.
The real default cat prompt is used, the text encoder uses official leaf/group
offload, and the VAE remains resident. No native build is performed at inference.

The weight arena is capped at 4 GiB; total staging is 1.5 GiB. Retained CPU
source storage and CUDA registration each have a separate 24.625 GiB ceiling.
These are the measured Windows recipe's capacities, not a total GPU-memory cap
or universal model defaults. Activations and other components also need memory.
For purge, use an elevated Windows terminal. For Linux/WSL capacities and staging
semantics, see [runtime and preset selection](#runtime-and-preset-selection).

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

Purge and report controls remain optional. Use your selected environment's
explicit Python if it is not activated.
Selecting `--ddo-preset nvme_baseline` still selects the protected NVMe route;
it does not enable the public native-fast path. Diagnostic adapter selection
also remains explicit.

For this image harness, prefer `--ddo-vmm-total-staging-gb` to state total
capacity. The legacy `--vmm-staging-buffer-gb` describes each slot: the public
native route multiplies it by three. Thus legacy `0.5` means 1.5 GiB total,
while legacy `0.375` means 1.125 GiB total. Do not pass both controls.

Selected installed rc3 image validation used **0.375 GiB total** staging on WSL,
with weights 4 GiB, retained source cache 24.625 GiB and registration 0.5 GiB.
Native Linux used **1.5 GiB total** staging and 24.625 GiB registration with
the same weight/cache ceilings. These are separate fixed workloads, not portable
defaults. See [installed-wheel validation conditions](../benchmarks/ltx23-image-details.md#installed-rc3-functional-validation)
before reproducing or comparing them.

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
