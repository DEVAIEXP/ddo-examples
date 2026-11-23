# Qwen Image 2.1

Use `--image-joint-attention cudnn` to request cuDNN SDPA for the joint
text/image attention. The default is SDPA. The external FlashAttention 2
backend cannot consume the joint padding mask used by this recipe.

## Model and requirements

The runners use
[Qwen Image 2.1 custom blocks](https://huggingface.co/elismasilva/qwen_image_2.1_custom_blocks)
with official Diffusers. Complete [installation](setup.md) first.

The default checkpoint is `Qwen/Qwen-Image-2.1`. Missing files download to the
standard Hugging Face cache; later runs reuse cached files. Set `HF_HOME` or
`HF_HUB_CACHE` before running to use your preferred cache drive. Pass
`--model-path /path/to/model` to read an existing local folder directly.
Use `--local-files-only` only for offline runs with all required files cached.

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

The Windows BF16 examples below reproduce the current 1024×1024 runtime
workloads through the public runners, without the benchmark controller.
Use the Base checkpoint snapshot `d26bb61231c349cf6b7896fa83353113880e1ba3`
and reviewed custom blocks from [measurement details](../benchmarks/qwen-image-21-details.md).
Replace the local checkpoint and standalone BF16 VAE paths with your own.
Run from an activated, installed RC5 environment without inherited DDO budget,
purge or attention overrides. Use the same environment for calibration and inference.
All primary recipes use default SDPA, the standard KV cache, no purge,
and no single-frame, alias, band, padding, tiling or compile optimizations.
The feature recipe below is separate from the runtime comparison.

### Official Diffusers Leaf

No DDO capacity profile is required. Encoder, transformer and VAE use Leaf.

```powershell
python -B scripts/qwen_image_21/run_diffusers_leaf.py `
  --prompt 'A capybara wearing a wizard hat, reading by candlelight, oil painting' `
  --width 1024 `
  --height 1024 `
  --steps 20 `
  --seed 42 `
  --memory-monitor rss `
  --metrics-level 2 `
  --save-metrics `
  --vae-leaf-offload `
  --image-joint-attention default `
  --output-dir outputs/qwen-image21/official-leaf
```

### Linear DDO: `one_shot_fast`

Before generation, run this same command with `--build-ddo-profile`. Keep all 20 steps for the linear calibration. Calibration stops before decode; keep the same profile name and use a separate output directory.

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular.py `
  --prompt 'A capybara wearing a wizard hat, reading by candlelight, oil painting' `
  --width 1024 `
  --height 1024 `
  --steps 20 `
  --seed 42 `
  --memory-monitor rss `
  --metrics-level 2 `
  --save-metrics `
  --vae-leaf-offload `
  --custom-blocks modular_diffusers/custom_blocks/qwen_image_21 `
  --ddo-backend linear `
  --ddo-preset one_shot_fast `
  --ddo-profile qwen-image21-1024x1024-20steps-one-shot-fast `
  --output-dir outputs/qwen-image21/one-shot-fast
```

### Linear DDO: `low_ram_safe`

Before generation, run this same command with `--build-ddo-profile`. Keep all 20 steps for the linear calibration. Calibration stops before decode; keep the same profile name and use a separate output directory.

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular.py `
  --prompt 'A capybara wearing a wizard hat, reading by candlelight, oil painting' `
  --width 1024 `
  --height 1024 `
  --steps 20 `
  --seed 42 `
  --memory-monitor rss `
  --metrics-level 2 `
  --save-metrics `
  --vae-leaf-offload `
  --custom-blocks modular_diffusers/custom_blocks/qwen_image_21 `
  --ddo-backend linear `
  --ddo-preset low_ram_safe `
  --ddo-profile qwen-image21-1024x1024-20steps-low-ram-safe `
  --output-dir outputs/qwen-image21/low-ram-safe
```

### VMM Fast with native BF16 VAE

Before generation, run this same command with `--build-ddo-profile`. Use `--steps 3` for calibration and repeat once only if the profile reports `floor_measured`; restore `--steps 20` for generation. Calibration stops before decode; keep the same profile name and use a separate output directory.

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular_vmm.py `
  --prompt 'A capybara wearing a wizard hat, reading by candlelight, oil painting' `
  --width 1024 `
  --height 1024 `
  --steps 20 `
  --seed 42 `
  --memory-monitor rss `
  --metrics-level 2 `
  --save-metrics `
  --vae-vmm-offload `
  --vae-vmm-resident-weight-budget-gb 0.375 `
  --custom-blocks modular_diffusers/custom_blocks/qwen_image_21 `
  --ddo-backend vmm `
  --ddo-preset auto `
  --no-ddo-vmm-execution-backing-reclaim `
  --text-encoder-native-vmm `
  --no-vae-no-temporal-cache `
  --image-joint-attention default `
  --ddo-vmm-total-staging-gb 0.625 `
  --ddo-vmm-source-cache-gb 16 `
  --ddo-vmm-source-registered-gb 16 `
  --ddo-profile qwen-image21-1024x1024-20steps-vmm-fast `
  --output-dir outputs/qwen-image21/vmm-fast
```

### Experimental NVMe alternative

This protected route uses a fixed budget and needs no capacity profile.

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular_vmm.py `
  --prompt 'A capybara wearing a wizard hat, reading by candlelight, oil painting' `
  --width 1024 `
  --height 1024 `
  --steps 20 `
  --seed 42 `
  --memory-monitor rss `
  --metrics-level 2 `
  --save-metrics `
  --vae-leaf-offload `
  --custom-blocks modular_diffusers/custom_blocks/qwen_image_21 `
  --ddo-backend vmm `
  --ddo-preset nvme_baseline `
  --no-ddo-vmm-execution-backing-reclaim `
  --no-text-encoder-native-vmm `
  --no-vae-no-temporal-cache `
  --ddo-vmm-resident-weight-budget-gb 1.5 `
  --ddo-vmm-total-staging-gb 1.5 `
  --ddo-vmm-source-cache-gb 0 `
  --ddo-vmm-source-registered-gb 0 `
  --output-dir outputs/qwen-image21/nvme-baseline
```

The native VAE recipe requires a standalone BF16 VAE, not the original
FP32 `vae` directory. Its 0.375 GiB weight budget is independent of the
transformer profile and staging. For the measured Leaf VAE alternative,
replace `--vae-vmm-offload` and
`--vae-vmm-resident-weight-budget-gb` with `--vae-leaf-offload`.
The source dispatcher remains enabled by the public DDO default.
Profile calibration adapts the transformer budget to the current hardware;
it does not promise the same timing or budget as the published GPU.

### Single-frame 1024-square recipe

The VMM runner defaults to `--vae-no-temporal-cache` for untiled single-image
decode. Use `--no-vae-no-temporal-cache` to select the original decoder.
The [Qwen VAE feature study](../benchmarks/model-features.md#qwen-image)
defines nine configurations and their comparison parents, with Windows and Linux
measurements. The following single-frame example uses the
native BF16 VAE route from that study. The published BF16 VAE is available at
[`elismasilva/qwen-image-2.1-bf16-vae`](https://huggingface.co/elismasilva/qwen-image-2.1-bf16-vae).
The runner downloads it to the standard Hugging Face cache before measured
execution. An existing local VAE directory is also accepted.

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular_vmm.py `
  --variant base --width 1024 --height 1024 --steps 20 --seed 42 `
  --custom-blocks modular_diffusers/custom_blocks/qwen_image_21 `
  --ddo-backend vmm --ddo-preset auto --ddo-profile auto `
  --text-encoder-native-vmm `
  --ddo-vmm-total-staging-gb 0.625 `
  --ddo-vmm-source-cache-gb 16 --ddo-vmm-source-registered-gb 16 `
  --vae-vmm-offload `
  --vae-vmm-resident-weight-budget-gb 0.375 `
  --vae-no-temporal-cache --image-joint-attention default `
  --no-ddo-vmm-execution-backing-reclaim `
  --memory-monitor rss --metrics-level 2 --show-metrics --save-metrics `
  --output-dir outputs/qwen_single_frame_1024
```

Build a compatible transformer profile before measuring inference; keep its
resident budget fixed when comparing VAE options. For the original untiled
control, replace `--vae-no-temporal-cache` with its negative form.
Add `--vae-alias-residuals`, `--vae-direct-padding` and `--vae-band-rows 256`
or `128` only according to the selected configuration. Band decode uses the
reviewed local custom blocks and two final decoder blocks. Tiling is a
separate original-decoder case and may change the image.

## Runtime and preset selection

See [presets](presets.md). The encoder and VAE can use different strategies from
the transformer. `--vae-resident` matches a resident-VAE control;
`--vae-leaf-offload` favors reduced weight residency.
If VAE activations still exceed memory, try `--vae-tiling`.

## Optional weight formats

### SDNQ INT4

Install the [SDNQ optional dependencies](setup.md). No FP8 extra or native
allocator is required. The runner downloads the saved transformer from
[`OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit`](https://huggingface.co/OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit)
and loads BF16 auxiliary components from `Qwen/Qwen-Image-2.1`. Cached Hub files
are reused. `--model-path` accepts a local packed transformer directory;
`--auxiliary-model` selects a different auxiliary source.

This Windows benchmark recipe uses Base, default SDPA and KV cache, Leaf
encoder and VAE, with no purge or model optimizations:

```powershell
python -B scripts/qwen_image_21/run_dynamic_modular_sdnq.py `
  --variant base --width 1024 --height 1024 --steps 20 --seed 42 `
  --prompt 'A capybara wearing a wizard hat, reading by candlelight, oil painting' `
  --custom-blocks modular_diffusers/custom_blocks/qwen_image_21 `
  --ddo-preset one_shot_fast `
  --text-encoder-leaf-offload --vae-leaf-offload --image-joint-attention default `
  --ddo-profile qwen-image21-1024x1024-20steps-sdnq-int4 `
  --memory-monitor rss --metrics-level 2 --save-metrics `
  --output-dir outputs/qwen-image21/sdnq-int4
```

Before generation, run this command with `--build-ddo-profile` and a separate
output directory. It measures a two-step baseline without decode. If the
report says `probe_pending`, replace that flag with `--validate-ddo-profile`
for the two-step validation. Once the profile is final, run the command above.
BF16 linear and VMM profiles cannot replace this packed-runtime profile.

For the Leaf transformer control, replace the preset with
`--ddo-preset diffusers_leaf_offload_compat` and omit `--ddo-profile`.
Both routes preserve the packed SDNQ forward.

The [resumable benchmark](../benchmarks/reproduction.md#qwen-saved-sdnq-int4-pair)
calibrates automatically, verifies metrics and PNGs, and records pixel
differences. The Windows RC5 pair produced pixel-identical PNGs; see the
[INT4 results](../benchmarks/qwen-image-21.md#windows-saved-sdnq-int4) and
[phase measurements](../benchmarks/qwen-image-21-phases.md#windows-sdnq-int4).
Turbo, INT8 and Linux/WSL INT4 are outside this recipe.

## Metrics and profile

Use `--show-profile --show-metrics --save-metrics` where supported.
Reports and media are saved under `--output-dir`; the runner prints their paths.
Select `--memory-monitor rss` to match the RSS benchmark tables.

## Runner-specific parameters

### Linear capacity profile

Use the complete command for the selected preset above, adding
`--build-ddo-profile` with the same 1024×1024 workload, prompt, routes,
attention and purge policy. Remove that flag for inference. Each preset
needs its own saved profile. Turbo needs a separate profile and preserves
its checkpoint sampling schedule; it is not the Base benchmark recipe.
See [profile storage and scope](runner-controls.md#persistent-capacity-profiles-for-linear-runners).

See [Qwen parameters](qwen-image-21-parameters.md) and
[shared controls](runner-controls.md).
