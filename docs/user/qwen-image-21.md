# Qwen Image 2.1

## Model and requirements

The runners use
[Qwen Image 2.1 custom blocks](https://huggingface.co/elismasilva/qwen_image_2.1_custom_blocks)
with official Diffusers. Complete [installation](setup.md) first.

## Available runners

| Script in `scripts/qwen_image_21/` | Purpose |
| --- | --- |
| `run_dynamic_minimal.py` | Staged base/Turbo example with native VMM transformer |
| `run_dynamic_modular.py` | Linear DDO and compatibility presets |
| `run_dynamic_modular_vmm.py` | Native VMM or protected NVMe |
| `run_dynamic_modular_vmm_diagnostics.py` | Native execution and transport experiments |
| `run_diffusers_leaf.py` | Conventional Diffusers leaf control |
| `run_dynamic_modular_sdnq.py` | Saved SDNQ transformer |

Inspect available options:

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular.py --help
```

## Run the model

### Linear DDO: one_shot_fast

This base recipe fixes the 512-square comparison workload:

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular.py --variant base --width 512 --height 512 --steps 20 --seed 42 --ddo-preset one_shot_fast --text-encoder-preset diffusers_leaf_offload_compat --vae-resident --purge-before-transformer --memory-monitor rss --show-metrics --save-metrics
```

It uses the default capybara prompt. Run elevated on Windows for standby purge;
outside Windows purge is a no-op. Exact platform comparisons and native recipes
are in [Qwen benchmarks](../benchmarks/qwen-image-21.md).

### Native VMM: vmm/auto on Windows

After installing the native allocator through the [setup guide](setup.md), run:

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular_vmm.py `
  --width 512 --height 512 --steps 20 --seed 42 `
  --ddo-vmm-resident-weight-budget-gb 4 `
  --ddo-vmm-total-staging-gb 0.625 `
  --ddo-vmm-source-cache-gb 16 `
  --ddo-vmm-source-registered-gb 16 `
  --ddo-vmm-execution-backing-reclaim `
  --vae-resident --purge-before-transformer `
  --memory-monitor rss --show-profile --show-metrics --save-metrics
```

This usage recipe explicitly enables [execution backing reclaim](runner-controls.md#execution-backing-reclaim).
That opt-in does not change the recorded conditions or results of historical benchmark rows.

The runner defaults to `--ddo-backend vmm --ddo-preset auto --variant base`.
This selects `vmm_fast_runtime`, using the same default capybara prompt, BF16,
CFG 1 and enabled KV cache. The runner stages the text encoder with official
Diffusers leaf offload; the command explicitly selects a resident, untiled VAE.
The capacities match the recorded Windows native workload: 4 GiB of active
weights, 640 MiB of total host staging and separate 16 GiB source-storage and
CUDA-registration ceilings. These are separate budgets, not a total VRAM limit.

Platform-specific conditions and the protected `vmm/nvme_baseline` comparison
remain documented in [measurement details](../benchmarks/qwen-image-21-details.md).
The SDNQ runner is a separate route and does not use this native recipe.

The public image VMM runners share one [DDO parameter vocabulary and defaults table](runner-controls.md#public-native-image-interface).
Specify total staging capacity in GiB. Native execution investigations use
the separate diagnostic entry.

Turbo uses its saved sampling schedule, not an arbitrary base step count.
Do not substitute Turbo in a base benchmark.

## Runtime and preset selection

See [presets](presets.md). The encoder and VAE can use different strategies from
the transformer. `--vae-resident` matches a resident-VAE control;
`--vae-leaf-offload` favors reduced weight residency.
If VAE activations still exceed memory, try `--vae-tiling`.

## Optional weight formats

Use the separate [SDNQ INT4 guide](qwen-sdnq.md) for saved quantized weights.
This is not a native VMM or general FP8 recipe.

## Metrics and profile

Use `--show-profile --show-metrics --save-metrics` where supported.
Reports and media are saved under `--output-dir`; the runner prints their paths.
Select `--memory-monitor rss` to match the RSS benchmark tables.

## Runner-specific parameters

### Linear capacity profile

Create a profile for the BF16 base workload below. This performs 20 real denoise
steps and stops before VAE loading, decode and image export. Run elevated on
Windows for the requested purge.

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular.py `
  --ddo-backend linear --ddo-preset one_shot_fast `
  --variant base --width 1024 --height 1024 --steps 20 --seed 42 `
  --ddo-profile qwen21-base-bf16-1024 --build-ddo-profile `
  --purge-before-transformer --memory-monitor rss --save-metrics `
  --output-dir outputs/qwen_linear_profile
```

For normal generation, repeat the same recipe without `--build-ddo-profile` and
use a separate output directory. Turbo needs its own profile name and retains
its saved checkpoint schedule; `--steps` does not replace Turbo sigmas.
See [profile storage and scope](runner-controls.md#persistent-capacity-profiles-for-linear-runners).

See [Qwen parameters](qwen-image-21-parameters.md) and
[shared controls](runner-controls.md).
