# LTX 2.3 image

## Model and requirements

The runners use
[LTX image custom blocks](https://huggingface.co/elismasilva/ltx2.3_image_custom_blocks).
Complete [installation](setup.md) before running.

## Available runners

| Script in `scripts/ltx2_image/` | Purpose |
| --- | --- |
| `run_dynamic_minimal.py` | Small staged BF16 example with native VMM transformer |
| `run_dynamic_modular_distilled.py` | Linear DDO and Diffusers compatibility presets |
| `run_dynamic_modular_vmm_distilled.py` | Native VMM or protected NVMe transformer |
| `run_dynamic_modular_vmm_diagnostics_distilled.py` | Native diagnostics and execution experiments |
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
  --ddo-backend vmm --ddo-preset auto `
  --width 1280 --height 704 --steps 8 --seed 43 `
  --ddo-vmm-resident-weight-budget-gb 4 `
  --ddo-vmm-total-staging-gb 1.5 `
  --ddo-vmm-source-cache-gb 24.625 `
  --ddo-vmm-source-registered-gb 24.625 `
  --ddo-vmm-execution-backing-reclaim `
  --purge-before-transformer `
  --memory-monitor rss --metrics-level 2 `
  --show-profile --show-metrics --save-metrics
```

This usage recipe explicitly enables [execution backing reclaim](runner-controls.md#execution-backing-reclaim).
That opt-in does not change the recorded conditions or results of historical benchmark rows.

The runner defaults to `--ddo-backend vmm --ddo-preset auto`, selecting
`vmm_fast_runtime` through the public DDO API. Guidance 1, rescale 0.7,
disabled PAG and 32 MiB pages are already defaults; no API-selection flag is needed.
The real default cat prompt is used, the text encoder uses official leaf/group
offload, and the VAE remains resident. No native build is performed at inference.

All four public LTX image entries accept `--custom-blocks SOURCE` and
`--custom-blocks-revision REVISION`. Omitted source preserves the existing local
`modular_diffusers/custom_blocks/ltx2_image` clone. To select Hub code, use
`--custom-blocks elismasilva/ltx2.3_image_custom_blocks` and optionally a reviewed
commit revision. These select executable code, not weights; use trusted sources.
A revision applies only to Hub code, not a local clone. See the
[parameter inventory](ltx2-image-parameters.md) for loading and report semantics.

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

The public image VMM runners share one [DDO parameter vocabulary and defaults table](runner-controls.md#public-native-image-interface).
Use the separate diagnostic entry for native execution investigations.

Purge and report controls remain optional. Use your selected environment's
explicit Python if it is not activated.
Selecting `--ddo-preset nvme_baseline` still selects the protected NVMe route;
it does not enable the public native-fast path. Diagnostic adapter selection
belongs to the separate diagnostic entry.

Use only `--ddo-vmm-total-staging-gb` to state total host staging capacity.
You do not need to know the internal slot count.

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

### Linear capacity profile

Create a profile for the BF16 linear runner (real GPU calibration; run elevated
on Windows for purge). The existing short calibration executes up to two
denoise steps and stops before decode; it does not save an image.

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_distilled.py `
  --ddo-backend linear --ddo-preset one_shot_fast `
  --width 1280 --height 704 --steps 8 --seed 43 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --ddo-profile ltx-image-bf16-1280x704 --build-ddo-profile `
  --purge-before-transformer --memory-monitor rss --save-metrics `
  --output-dir outputs/ltx_image_linear_profile
```

For normal generation with that profile, repeat the command without
`--build-ddo-profile`, using a separate output directory. Preserve the checkpoint,
custom blocks and calculation options. See [profile storage and scope](runner-controls.md#persistent-capacity-profiles-for-linear-runners).

See [image parameters](ltx2-image-parameters.md) and
[shared controls](runner-controls.md).
