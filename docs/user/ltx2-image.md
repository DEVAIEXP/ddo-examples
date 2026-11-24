# LTX 2.3 image

Use `--image-self-attention flash-attn2` or `--image-self-attention cudnn`
to select the transformer self-attention backend. The default is SDPA.
Cross-attention keeps SDPA and the text mask.

## Model and requirements

The runners use
[LTX image custom blocks](https://huggingface.co/elismasilva/ltx2.3_image_custom_blocks).
Complete [installation](setup.md) before running.

The default checkpoint is `elismasilva/ltx2.3-image-distilled-1.1`. Missing files download to the
standard Hugging Face cache; later runs reuse cached files. Set `HF_HOME` or
`HF_HUB_CACHE` before running to use your preferred cache drive. Pass
`--model-path /path/to/model` to read an existing local folder directly.
Use `--local-files-only` only for offline runs with all required files cached.

## Available runners

| Script in `scripts/ltx2_image/` | Purpose |
| --- | --- |
| `run_diffusers_leaf.py` | Official Diffusers leaf offload reference |
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
The Windows examples below use the benchmark cat prompt:
"Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy."
Use checkpoint snapshot `5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9`
and the reviewed blocks from [measurement details](../benchmarks/ltx23-image-details.md).
The commands use the public runners directly, without the benchmark controller.
Default SDPA, disabled model optimizations and the recorded component routes
are part of the recipe; calibration selects a budget for your hardware. Run them from the
examples checkout with your Python environment activated. Each command includes
its own options.

Use an installed RC5 environment without inherited DDO budget, purge or
attention overrides. Calibration and inference must use that same environment.
The checkpoint and prompt defaults match this workload; when selecting local
weights explicitly, add `--model-path` pointing to the snapshot above.

These recipes require DDO's opt-in purge defaults: omitting a purge flag leaves
that phase disabled. The earlier RC5 wheel enables some purges through its
presets; check the [installed-package caveat](runner-controls.md#windows-standby-purge)
before using that wheel to reproduce the no-purge rows.

### Official Diffusers Leaf

```powershell
python -B scripts/ltx2_image/run_diffusers_leaf.py `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --vae-leaf-offload `
  --memory-monitor rss --metrics-level 2 --save-metrics --output-dir outputs/ltx23-image/leaf
```

**Resident VAE option.** Omit `--vae-leaf-offload` to use a resident VAE
with this runner. Prefer trying this route when the full VAE and its decode
activations fit comfortably in GPU memory after the transformer is released.
Leaf transfers weights between CPU and GPU during decode, so its overhead
depends on the hardware and host-to-GPU bandwidth. Compare load, decode and
cleanup together. In the Windows 1280×704 reference comparison, the full VAE
phase took 1.34 s with Leaf and 1.65 s with the resident VAE.

### Linear DDO: `one_shot_fast`

If you need to create the profile, run the calibration command in
[Linear capacity profiles](#linear-capacity-profile) first.

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_distilled.py `
  --ddo-backend linear --ddo-preset one_shot_fast `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --ddo-profile ltx23-image-1280x704-8steps-one-shot-fast `
  --vae-leaf-offload `
  --memory-monitor rss --metrics-level 2 --save-metrics --output-dir outputs/ltx23-image/one-shot-fast
```

### Linear DDO: `low_ram_safe`

Create the matching profile in [Linear capacity profiles](#linear-capacity-profile)
before using this command. The saved profile supplies the resident budget.

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_distilled.py `
  --ddo-backend linear --ddo-preset low_ram_safe `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --ddo-profile ltx23-image-1280x704-8steps-low-ram-safe `
  --vae-leaf-offload `
  --memory-monitor rss --metrics-level 2 --save-metrics --output-dir outputs/ltx23-image/low-ram-safe
```

On Windows, you can add `--purge-before-vae` to this generation command in an
elevated PowerShell. See the [paired measurement](../benchmarks/ltx23-image.md)
for the effect on this workload.

### Native VMM: `vmm/auto`

Use elevated PowerShell for startup and pre-transformer purges. Install the matching allocator wheel from the [setup guide](setup.md). The default checkpoint also supplies its `vae` subdirectory.
If you need a profile, follow [VMM residency calibration](#vmm-residency-calibration).

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py `
  --ddo-backend vmm --ddo-preset auto `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --ddo-profile ltx23-image-1280x704-8steps-vmm-fast `
  --ddo-vmm-total-staging-gb 1.5 `
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 `
  --text-encoder-native-vmm --no-ddo-vmm-execution-backing-reclaim `
  --vae-vmm-offload `
  --vae-vmm-resident-weight-budget-gb 1.5 `
  --purge-before-run --purge-before-transformer `
  --memory-monitor rss --metrics-level 2 --save-metrics --output-dir outputs/ltx23-image/vmm-fast
```

The profile selects the transformer residency budget. The VAE's 1.5 GiB weight
budget is separate from the transformer's budget, staging and activation memory.
To use the Leaf VAE, replace `--vae-vmm-offload` and its weight budget option
with `--vae-leaf-offload`.

### NVMe control on Windows

This experimental alternative may be discontinued in future releases; see
[preset status](presets.md#nvme-baseline-status).

Use an elevated PowerShell for the two purges. This recipe uses Leaf for the
encoder and VAE and a fixed transformer budget. Use a DDO version with automatic
NVMe library discovery and install the matching allocator wheel.

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py `
  --ddo-backend vmm --ddo-preset nvme_baseline `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --no-text-encoder-native-vmm --vae-leaf-offload `
  --ddo-vmm-resident-weight-budget-gb 1.5 --ddo-vmm-total-staging-gb 1.5 `
  --ddo-vmm-source-cache-gb 0 --ddo-vmm-source-registered-gb 0 `
  --no-ddo-vmm-execution-backing-reclaim `
  --purge-before-run --purge-before-transformer `
  --memory-monitor rss --metrics-level 2 --save-metrics --output-dir outputs/ltx23-image/nvme
```

The published DDO RC5 wheel predates automatic NVMe discovery. With that wheel,
set `DDO_VMM_LIBRARY_PATH` to the DLL from the installed allocator wheel.

### Linear capacity profile

Run the calibration for the preset you want to use. Once the profile is saved,
use the corresponding generation command above.

**`one_shot_fast`**

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_distilled.py `
  --ddo-backend linear --ddo-preset one_shot_fast `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --ddo-profile ltx23-image-1280x704-8steps-one-shot-fast --build-ddo-profile `
  --vae-leaf-offload `
  --memory-monitor rss --metrics-level 2 --save-metrics --output-dir outputs/ltx23-image/one-shot-fast-profile
```

**`low_ram_safe`**

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_distilled.py `
  --ddo-backend linear --ddo-preset low_ram_safe `
  --width 1280 --height 704 --seed 43 --steps 8 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --ddo-profile ltx23-image-1280x704-8steps-low-ram-safe --build-ddo-profile `
  --vae-leaf-offload `
  --memory-monitor rss --metrics-level 2 --save-metrics --output-dir outputs/ltx23-image/low-ram-safe-profile
```

### VMM residency calibration

In an elevated PowerShell, run this command twice
with the same profile name. The first run measures the floor and the second
tests the ceiling if needed. If the first run already reports `final`, you can
start generating images.

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py `
  --ddo-backend vmm --ddo-preset auto `
  --width 1280 --height 704 --seed 43 --steps 3 `
  --guidance-scale 1 --guidance-rescale 0.7 --no-pag `
  --ddo-profile ltx23-image-1280x704-8steps-vmm-fast --build-ddo-profile `
  --ddo-vmm-total-staging-gb 1.5 `
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 `
  --text-encoder-native-vmm --no-ddo-vmm-execution-backing-reclaim `
  --vae-vmm-offload `
  --vae-vmm-resident-weight-budget-gb 1.5 `
  --purge-before-run --purge-before-transformer `
  --memory-monitor rss --metrics-level 2 --save-metrics --output-dir outputs/ltx23-image/vmm-profile
```

The LTX image runners accept `--custom-blocks SOURCE` and
`--custom-blocks-revision REVISION`. Omitted source loads the trusted Hub code
from `elismasilva/ltx2.3_image_custom_blocks`; add a reviewed commit revision to
pin it. To use a local checkout, such as the
`modular_diffusers/custom_blocks/ltx2_image` clone, pass its folder instead.
These select executable code, not weights; use trusted sources.
A revision applies only to Hub code, not a local clone. See the
[parameter inventory](ltx2-image-parameters.md) for loading and report semantics.


## Optional LoRA

The image runners accept the same `--lora`, `--lora-weight-name`,
`--lora-revision` and `--lora-scale` controls as the
[video runners](ltx2-video.md#optional-lora). Use a compatible transformer-only
adapter. A selected Windows RC5 VMM Fast run completed with the Soft Enhance
adapter at scale 0.8, BF16, 1280×704, eight steps and seed 43. The adapter remained
unfused, and its base storage was preserved. Calibrate a separate profile with
the same adapter and scale before generating; this result does not validate
every adapter or platform.
No adapter is loaded by default.

Add these options to the VMM calibration and generation recipes above:

```powershell
--lora vrgamedevgirl84/LTX_2.3_Soft_Enhance_Style_LoRa `
--lora-weight-name LTX2.3_Soft_Enhance.safetensors `
--lora-revision 53c45045b8a229b49904f74d2640f30a6ae8a9bb `
--lora-scale 0.8
```

A LoRA changes the generated image even with the same prompt and seed. Matching
the image without an adapter is not an acceptance criterion for this recipe.

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

For the current WSL benchmark recipe, keep the same image workload but use
Leaf encoder and VAE, `--no-text-encoder-native-vmm`, 0.375 GiB total staging,
24.625 GiB source cache and 0.5 GiB source registration, without Windows
purge flags. Calibrate a separate WSL profile with those same settings.
See [current platform conditions](../benchmarks/ltx23-image-details.md#wsl-native).
The public DDO source dispatcher default remains enabled for native source caching.


## Optional weight formats

### SDNQ INT4

Install the optional SDNQ backend following [setup](setup.md). No FP8 extra or
native allocator is required. The runner downloads the saved INT4 transformer
from `elismasilva/ltx2.3-image-distilled-1.1-sdnq-int4` and uses the original
BF16 checkpoint for auxiliary components. Hub files are reused from the cache;
`--sdnq-model-path` accepts an existing local transformer directory.

This Windows benchmark recipe uses Leaf text encoding, a resident VAE, default
SDPA and no purge or model optimizations. Run from the examples checkout:

```powershell
python -B scripts/ltx2_image/run_dynamic_modular_sdnq_distilled.py `
  --bits 4 --ddo-preset one_shot_fast `
  --custom-blocks modular_diffusers/custom_blocks/ltx2_image `
  --width 1280 --height 704 --steps 8 --seed 43 `
  --guidance-scale 1 --guidance-rescale 0.7 `
  --text-encoder-group-offload --image-self-attention default `
  --ddo-profile ltx23-image-1280x704-8steps-sdnq-int4 `
  --memory-monitor rss --metrics-level 2 --save-metrics `
  --output-dir outputs/ltx23-image/sdnq-int4
```

The default prompt is the benchmark cat prompt above. Before generation, run
this command with `--build-ddo-profile` and a separate output directory. It
measures two steps without decode. If the report says `probe_pending`, replace
that flag with `--validate-ddo-profile` for the two-step validation. Once the
profile is final, run the command above. Keep its nominal eight-step schedule;
BF16 linear and VMM profiles cannot replace this SDNQ profile.

For the Leaf transformer control, replace the preset with
`--ddo-preset diffusers_leaf_offload_compat` and omit `--ddo-profile`.
Both routes preserve the packed SDNQ forward. The image
[benchmark controller](../benchmarks/reproduction.md#windows-ltx-image-sdnq-int4)
prepares a matched INT4 DDO/Leaf pair; the selected Windows RC5 results are
in the [image benchmark](../benchmarks/ltx23-image.md#windows-saved-sdnq-int4).

## Metrics and profile

Add `--show-profile --show-metrics --save-metrics` where supported.
`--output-dir` selects the output location; the runner prints the JSON path.
Use RSS monitoring for comparison with the published RSS tables.

See [image parameters](ltx2-image-parameters.md) and
[shared controls](runner-controls.md).
