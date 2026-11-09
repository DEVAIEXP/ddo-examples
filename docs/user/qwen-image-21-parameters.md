# Qwen Image 2.1 parameters

This page lists the public controls of `run_dynamic_modular_vmm.py`.
See [shared controls](runner-controls.md#public-native-image-interface) for the
DDO route/capacity and reclaim options, reporting defaults and Windows purge switches.
`--help` / `-h` prints the installed checkout's interface without generating an image.
Options in other Qwen entries are identified separately below.

## Execution

The BF16 linear `run_dynamic_modular.py` also supports `--ddo-profile NAME`
and `--build-ddo-profile`, independently of display-only `--show-profile`.
See the [linear calibration recipe](qwen-image-21.md#linear-capacity-profile).

This VMM entry accepts the same two options for the VMM residency profile
(`NAME` or `auto`): run the build twice with `--steps 3` (Turbo keeps its
saved sigmas; floor, then hardware ceiling; each build stops before decode),
then generate with the same arguments. `auto` names the profile
`qwen-image21-<W>x<H>-<digest>` from the resolution and variant. Without an
explicit `--ddo-vmm-resident-weight-budget-gb`, `vmm/auto` uses the saved
profile or DDO's structural floor (previously the generic 4 GiB default). See
[VMM residency profile](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/user/native-vmm.md#vmm-residency-profile).
The capacity controls are not exposed by the public minimal, SDNQ or Diffusers
leaf entries.

| Control | Purpose |
| --- | --- |
| `--text-encoder-native-vmm` | Default on for `vmm/auto` (off on other routes); `--no-text-encoder-native-vmm` selects Diffusers leaf offload. Loads the Qwen3-VL encoder through DDO's weight-aware loader with native component ownership. A measured 512² Windows pair (standby purge before run, benchmark transport) encoded in 3.2 s versus 13.7 s with leaf, identical image; this is a recipe default, not a general claim. |
| `--text-encoder-vmm-resident-weight-budget-gb` | 4 GiB with the native encoder; encoder-only ceiling, never the transformer's residency budget. |
| `--attention-backend default\|cudnn` | Default `default` (PyTorch's SDPA order). `cudnn` prefers flash > cuDNN > memory-efficient > math during the denoise only; ineligible inputs fall back. Kernels change, model math does not, but outputs can differ numerically: compare images before adopting it. Part of the `auto` profile name. Not yet measured for Qwen. |

| Control | Purpose |
| --- | --- |
| `--variant base` / `turbo` | Default `base`; selects the checkpoint family and preparation schedule. |
| `--model-path PATH_OR_ID` | Pipeline checkpoint containing the model components; omitted selects `Qwen/Qwen-Image-2.1` or its Turbo counterpart from `--variant`. |
| `--custom-blocks SOURCE` | Executable modular Python code, not weights. Default `elismasilva/qwen_image_2.1_custom_blocks`; accepts a trusted Hub repository or local checkout containing `modular_config.json`. |
| `--custom-blocks-revision REVISION` | Optional Hub code revision; prefer a commit SHA. Rejected for a local checkout; it does not switch local Git branches. |
| `--local-files-only` | Default off. Require cached model and custom-code files instead of downloading them. |
| `--prompt TEXT` | Text conditioning; default is the capybara/wizard/candlelight prompt in the runner. |
| `--width`, `--height` | Output dimensions; both default to 1024. |
| `--seed` | Random seed; default 42. |
| `--steps` | Default 25 for base. Turbo uses checkpoint sigmas; this is not a Turbo schedule replacement. The published base recipes use 20 explicitly. |
| `--output-dir DIRECTORY` | Default `outputs/qwen_image_21_vmm` under the examples root; saves `image.png` and enabled reports. |

Only use custom code you trust: loading it executes Python. Both native image
entries use BF16; this Qwen entry computes on `cuda:0`, with CFG 1 and KV cache
enabled. These are fixed execution choices, not additional CLI options.

## VAE strategy and memory

All switches below are off by default. With no VAE route override,
`vmm/auto` selects no-stream leaf offload through DDO. The protected NVMe
route preserves its preset component policy.

| Control | Purpose and constraints |
| --- | --- |
| `--vae-leaf-offload` | Explicit Diffusers leaf route without streaming; overrides the preset VAE route. |
| `--vae-resident` | Explicit resident VAE decode. Cannot be combined with leaf or native VAE selection. |
| `--vae-vmm-offload` | Experimental native VAE component ownership. Requires `--ddo-backend vmm --ddo-preset auto`, a separate VAE path and a finite positive weight budget. Cannot be combined with leaf/resident selection. |
| `--vae-model-path DIRECTORY` | Existing standalone BF16 VAE directory, loaded locally; required only with native VAE selection. |
| `--vae-vmm-resident-weight-budget-gb GIB` | Independent VAE weight ceiling; required only with native VAE selection. Does not change the transformer budget. |
| `--vae-tiling` | Spatial tiling to reduce VAE activation demand; may change decode latency. |
| `--vae-no-temporal-cache` | Opt in to the custom VAE's single-frame, untiled decode path. Tiled decode takes precedence. |
| `--vae-alias-residuals` | Reuse residual inputs in that custom single-frame path. Requires `--vae-no-temporal-cache`; not active under tiling. |
| `--vae-direct-padding` | Use custom convolution padding in that single-frame path. Requires `--vae-no-temporal-cache`; not active under tiling. |
| `--vae-band-rows ROWS` | Opt-in band decode in that single-frame path: the last `--vae-band-blocks` decoder up blocks (default 2, the highest-resolution ones) and the output head run in horizontal bands of `ROWS` output rows, each with a halo covering their receptive field, so the activation peak drops without seams. The middle block (global attention) and earlier blocks run whole. Equal to the whole decode on CPU; GPU output needs a parity check. Requires `--vae-no-temporal-cache`, not with `--vae-tiling`, and custom blocks providing `enable_band_decode` (select a revision or local checkout with `--custom-blocks`). |
| `--vae-band-blocks N` | Number of final up blocks decoded in bands; requires `--vae-band-rows`. |
| `--release-native-cache-before-vae` | After transformer cleanup, explicitly request native idle-pool trim before loading the VAE. Requires a supporting allocator. Releases only eligible idle backing, not live tensors; independent of Windows standby purge. |
| `--clear-cublas-workspaces-before-vae` | Experimental single-GPU cuBLAS workspace cleanup at the same boundary. Requires `--release-native-cache-before-vae`. Does not imply all driver/context memory was freed. |

These optional controls change component execution or memory lifecycle; they are
not passive diagnostics. A supported switch is not general model/platform or
performance acceptance. Preserve the exact VAE choices when comparing results.

## Other Qwen runners

All public Qwen entries use the shared parameter vocabulary. Linear and SDNQ
capacity settings follow the selected preset.
Advanced native experiments use `run_dynamic_modular_vmm_diagnostics.py`.

| Entry | Interface and defaults |
| --- | --- |
| `run_dynamic_minimal.py` | Native VMM transformer (`--ddo-backend vmm` / `--ddo-preset auto`), native capacities and opt-in execution backing reclaim; base/Turbo, custom blocks, offline loading and opt-in VAE tiling. Defaults: 1024², base 25 steps, seed 42, full memory monitoring. No separate profile/metrics display switches. |
| `run_dynamic_modular.py` | Linear/compatibility route; same source/sampling vocabulary as VMM. Defaults: 1024², base 25 steps, seed 42, RSS/metrics level 2, display/save enabled, profile disabled. Supports the component controls described below. |
| `run_dynamic_modular_sdnq.py` | Packed base INT4 or its leaf control; backend `linear`, preset `auto`, 512², 20 steps, seed 42. Same reporting defaults as linear. `--model-path` overrides the packed transformer; `--model-revision` optionally pins it. `--auxiliary-model` defaults to `Qwen/Qwen-Image-2.1`. See the [SDNQ guide](qwen-sdnq.md). |
| `run_diffusers_leaf.py` | Conventional full-pipeline control, with no DDO runtime or custom blocks. No backend/preset selectors. Pipeline `--model-path`, prompt, dimensions, steps, seed, offline loading, VAE tiling and reporting only; defaults 512²/20 steps/seed 42 and RSS/level 2. Exposes purge before run/transformer and after run, with no separate encoder boundary. |

`--text-encoder-preset` and `--vae-linear-offload` belong to entries that expose
them, such as the linear runner; they are not options of this public VMM entry.
The linear encoder override accepts `diffusers_leaf_offload_compat` only;
omitting it preserves preset resolution. Linear and SDNQ share the explicit
VAE resident/leaf/linear overrides (mutually exclusive), tiling and single-frame
cache/residual/padding controls. Eager `--vae-linear-offload` requires a linear
preset; the residual and padding options require `--vae-no-temporal-cache`.
Native VAE ownership and native idle-cache/cuBLAS cleanup remain VMM-only.

The SDNQ guide covers only the selected base INT4 recipe, not Turbo.
The native image entry uses the same [DDO route/capacity and reclaim parameters](runner-controls.md#public-native-image-interface)
as LTX image. Native timing/transport experiments belong
to `run_dynamic_modular_vmm_diagnostics.py`.

## Reporting and diagnostic entries

Use the [shared metrics/profile controls](runner-controls.md#diagnostics).
The public entry accepts metrics levels 1 and 2; it does not accept level 0.
GPU-event timing and transport experiments belong to the
separate diagnostic entry. See [developer diagnostics](../developer/diagnostics.md)
for cleanup and measurement scope.
