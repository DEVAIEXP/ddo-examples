# LTX image parameters

This page lists the public LTX image controls, with native VMM defaults in the
detailed tables and differences for the other entries below.
See [shared controls](runner-controls.md#public-native-image-interface) for the
DDO route/capacity and reclaim options, reporting defaults and Windows purge switches.
`--help` / `-h` prints the installed checkout's interface without generating an image.
Each entry exposes only implemented controls; shared controls keep one spelling.

## Interface scope by runner

| Entry | Code source selection | CLI scope |
| --- | --- | --- |
| `run_dynamic_modular_vmm_distilled.py` | `--custom-blocks` / `--custom-blocks-revision` | Native route selection, capacities, component loading and reporting. |
| `run_dynamic_minimal.py` | `--custom-blocks` / `--custom-blocks-revision` | Minimal native VMM transformer integration; canonical names, basic sampling and reporting. |
| `run_dynamic_modular_distilled.py` | `--custom-blocks` / `--custom-blocks-revision` | Linear/compatibility entry; canonical names, component loading and PAG. |
| `run_dynamic_modular_sdnq_distilled.py` | `--custom-blocks` / `--custom-blocks-revision` | Packed SDNQ/leaf entry; canonical names and quantization-specific options. |
| `run_dynamic_modular_vmm_diagnostics_distilled.py` | Existing fixed local clone | Separate full native experiment interface, intentionally broader than the public entry. |

All four public entries use the shared parameter vocabulary. Select the runtime with
`--ddo-backend` and `--ddo-preset`; the three non-VMM entries support backend
`linear` only and default to preset `auto`. SDNQ preserves its packed forward
and supports `auto`, `one_shot_fast`, `low_ram_safe`, `wsl_compat`, `warm_process`
and `diffusers_leaf_offload_compat`. The latter selects the official leaf control.
This naming cleanup does not replace quantized math with a dense linear forward.

| Controls | Minimal | Linear | SDNQ | VMM |
| --- | --- | --- | --- | --- |
| Checkpoint, custom blocks, device, prompt, dimensions, steps, seed, guidance, output | Yes | Yes | Yes | Yes |
| Unified `--lora` options | Yes | Yes | Yes | Yes |
| Purge phases, memory monitor, metrics display/save, denoise progress | Yes | Yes | Yes | Yes |
| `--metrics-level`, decode timestep/noise, PAG, loading controls | No | Yes | Yes | Yes |
| `--text-encoder-dynamic-offload` | No | Yes | No | Yes |
| VMM capacities and execution backing reclaim | Yes | No | No | Yes |
| `--local-files-only`, `--show-profile`, native encoder/VAE overrides | No | No | No | Yes |
| `--bits`, `--text-encoder-bits` | No | No | Yes | No |
| `--ddo-profile NAME`, `--build-ddo-profile` | No | Yes (linear capacity profile) | No | Yes (VMM residency profile, `NAME` or `auto`) |

The minimal entry uses its existing component defaults. Linear and SDNQ retain
their preset-driven profile display; they do not expose a separate profile
display switch. The BF16 modular linear entry separately exposes persistent
capacity calibration as shown below. Their output directories default to `outputs/ltx_image_modular` and
`outputs/ltx_image_modular_sdnq_distilled`; minimal also uses `outputs/ltx_image_modular`.
SDNQ accepts `--bits 4` (default) or `8` for transformer checkpoint selection;
`--text-encoder-bits` is unset by default (original encoder) and accepts `8` only.
In SDNQ, `--model-path` selects the
auxiliary pipeline weights; the packed transformer and text encoder sources
remain selected by their bit-depth settings. The default text encoder group
offload is enabled. No unimplemented dynamic encoder option is accepted.

## Execution

| Control | Purpose and default |
| --- | --- |
| `--model-path PATH_OR_ID` | Pipeline checkpoint. Default `elismasilva/ltx2.3-image-distilled-1.1`, overridable by the `MODEL_PATH` environment variable. |
| `--custom-blocks SOURCE` | Trusted executable LTX image code: a local directory containing `modular_config.json` or a Hub repository. Defaults to the existing `modular_diffusers/custom_blocks/ltx2_image` clone under the examples root. |
| `--custom-blocks-revision REVISION` | Optional Hub code revision; prefer a commit SHA. Rejected for local directories; does not switch a local Git branch. |
| `--local-files-only` | VMM only, default off; require cached weights/code rather than downloading. |
| `--device` | Execution device; default `cuda:0`. Does not establish support for other devices. |
| `--prompt TEXT` | Text conditioning; default is the calico-cat/fisheye/flower-crown prompt in the runner. |
| `--negative-prompt TEXT` | Negative conditioning; default empty. |
| `--width`, `--height` | Dimensions; defaults 1280 and 704. |
| `--steps`, `--seed` | Defaults 8 and 43. |
| `--guidance-scale` | Guidance strength; default 1.0. |
| `--guidance-rescale` | Guidance rescaling; default 0.7. |
| `--decode-timestep` | VAE decode timestep; default 0.0. |
| `--decode-noise-scale` | Optional decode noise scale; default unset (`None`). |
| `--pag` | Enable perturbed-attention guidance; disabled by default. |
| `--pag-scale` | PAG strength when enabled; default 0.2. |
| `--pag-layers` | Comma-separated transformer layer indices for PAG; default `28`. |
| `--output-dir DIRECTORY` | Default `outputs/ltx_image_modular_vmm`, relative to the working directory; output images and enabled reports. |

The dense entries use BF16; SDNQ keeps its packed backend. None exposes Qwen's
`--variant` selector.
Custom blocks are Python code, not weights: loading them executes trusted code.
All LTX image phases, transformer and connectors come from the same resolved
code namespace. Missing required exports fail instead of falling back to a
different checkout. Reports record the requested source/revision and resolved
location/snapshot. In VMM, `--local-files-only` also applies to Hub code resolution.
Other entries retain their existing download behavior.
Source selection has CPU validation; GPU equivalence for a different revision
still requires separate validation.

## Components and loading

| Control | Purpose and default |
| --- | --- |
| `--low-cpu-mem-usage` | Default on; use the lower-CPU-memory transformer loading path. |
| `--text-encoder-low-cpu-mem-usage` | Default on; lower-CPU-memory text encoder loading. |
| `--text-encoder-group-offload` | Explicit encoder group/leaf compatibility override. Omitted inherits environment/preset resolution; ordinary default enabled. |
| `--text-encoder-dynamic-offload` | Explicit encoder DDO override. Omitted inherits environment/preset resolution; ordinary default disabled. |
| `--text-encoder-native-vmm` | Default on for `vmm/auto` (off on other routes or with an explicit group-on/dynamic-off encoder override); `--no-text-encoder-native-vmm` selects leaf group offload. Native encoder ownership is independent of the transformer and cannot be combined with explicit group-on or dynamic-off compatibility overrides. A measured 1280x704 Windows pair (standby purge before run) encoded in 10.65 s versus 17.27 s with leaf, peak RSS 3.55 GB versus 25.9 GB, identical image; this is a recipe default, not a general claim. |
| `--text-encoder-vmm-resident-weight-budget-gb` | 4 GiB with the native encoder; encoder-only ceiling, never the transformer's residency budget. |
| `--vae-leaf-offload` | Default off; explicitly select no-stream leaf VAE offload instead of the resident decoder. |

Boolean loading/group/dynamic controls also accept `--no-...`. The VAE leaf
override accepts `--no-vae-leaf-offload`. This entry does not expose native idle
cache trim, cuBLAS clearing, VAE tiling or the Qwen single-frame VAE controls.

## LoRA

No adapter is loaded by default. Adapters are loaded on the CPU transformer
before offload hooks, without fusing or replacing the backend forward.

| Control | Purpose |
| --- | --- |
| `--lora SOURCE` | Transformer-only LoRA file, directory or Hub source; repeat for multiple adapters. |
| `--lora-weight-name NAME` | Filename per source; `auto` selects automatically. Not accepted with a direct local file. |
| `--lora-revision REVISION` | Hub revision per source; `auto` uses the default revision. Not accepted with a direct local file. |
| `--lora-scale VALUE` | Finite strength per adapter; default 1.0 for each. |

These four LoRA controls are shared by all public LTX image entries; SDNQ does
not expose separate named soft/crisp adapter switches.
When providing filenames, revisions or scales, supply each option once per
`--lora`, in the same order. These options require `--lora`. Native VMM with
LoRA still requires separate CUDA validation; see [the model guide](ltx2-image.md#optional-lora).

Keep encoder, VAE and transformer choices fixed when reproducing a benchmark.

## Reporting and diagnostic entries

The public entry accepts metrics levels 0, 1 and 2. Level 2 includes public DDO
component reports; lower levels omit those detailed snapshots. Display/save
switches are independent; level 0 does not mean all monitoring is disabled.

Among the LTX image entries, `--fake-prompt` is available only in
`run_dynamic_modular_vmm_diagnostics_distilled.py`; it skips real prompt
encoding and is not a full-pipeline benchmark. Native adapter diagnostics
also belong to that diagnostic entry.

The native VMM entry `run_dynamic_modular_vmm_distilled.py` accepts
`--ddo-profile NAME|auto` and `--build-ddo-profile` for the VMM residency
profile: run the build twice with `--steps 3` (floor, then hardware ceiling;
each build stops before decode), then generate with the same arguments.
`auto` names the profile `ltx23-image-<W>x<H>-<digest>` from the resolution and
the options that change activations (PAG, guidance, sequence length). Without
an explicit `--ddo-vmm-resident-weight-budget-gb`, `vmm/auto` uses the saved
profile or DDO's structural floor; the NVMe baseline keeps 2 GiB. See
[VMM residency profile](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/user/native-vmm.md#vmm-residency-profile).
The BF16 modular linear runner separately supports `--ddo-profile NAME` and
`--build-ddo-profile`; see the [linear calibration recipe](ltx2-image.md#linear-capacity-profile).
Profiles and metrics use the [shared switches](runner-controls.md#diagnostics).
See [developer diagnostics](../developer/diagnostics.md) before interpreting timings.
