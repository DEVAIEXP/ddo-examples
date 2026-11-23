# Shared runner controls

Run commands from the examples root in the installed environment. Each runner's
`--help` defines the options it supports; minimal and conventional controls
expose only a subset. Shared options use the same spelling and meaning.

## Execution

| Control | Purpose |
| --- | --- |
| `--ddo-backend linear` / `vmm` | Transformer route family |
| `--ddo-preset NAME` | Choose a [runtime and preset](presets.md) |
| `--model-path PATH_OR_ID` | Checkpoint source; in video entries, a transformer-only override |
| `--distilled-model PATH_OR_ID` | LTX video BF16 pipeline and auxiliary weights |
| `--custom-blocks SOURCE`, `--custom-blocks-revision REVISION` | Model code source and optional revision, where supported |
| `--local-files-only` | Require cached model/code files; input media URLs are separate |
| `--output-dir DIRECTORY` | Output media and reports |
| `--ddo-vmm-resident-weight-budget-gb VALUE` | Active transformer weight capacity in GiB |
| `--ddo-vmm-total-staging-gb VALUE` | Total reusable VMM staging capacity in GiB |
| `--ddo-vmm-source-cache-gb VALUE` | Retained source capacity in GiB |
| `--ddo-vmm-source-registered-gb VALUE` | CUDA-registered source capacity in GiB |
| `--ddo-vmm-execution-backing-reclaim` | Opt into pressure-triggered reclaim of idle execution backing in native fast |
| `--vae-resident`, `--vae-leaf-offload`, `--vae-linear-offload` | Explicit VAE strategy, where implemented |
| `--vae-tiling` / `--no-vae-tiling` | Spatial VAE tiling, where supported |

### Model downloads and local folders

The model runners select Hub repositories by default. Missing files download
to the standard Hugging Face cache; subsequent runs reuse the cached files.
Set `HF_HOME` or `HF_HUB_CACHE` before starting the runner to choose your cache
location. There is no required drive or folder named `hf_cache`.

| Source | Default repository |
| --- | --- |
| LTX Image BF16 | `elismasilva/ltx2.3-image-distilled-1.1` |
| LTX Video BF16 and auxiliary components | `OzzyGT/LTX-2.3-Distilled-1.1` |
| Qwen Base BF16 | `Qwen/Qwen-Image-2.1` |
| Qwen native BF16 VAE, when `--vae-vmm-offload` is selected | `elismasilva/qwen-image-2.1-bf16-vae` |
| LTX Video saved FP8, when `--transformer-fp8-weight-only` is selected | `elismasilva/ltx2.3_video_fp8_distilled_transformer` |

Omit the source option to use its default. Pass `--model-path PATH_OR_ID`
to override the checkpoint; LTX Video uses `--distilled-model PATH_OR_ID` for
the pipeline and `--model-path PATH_OR_ID` for a transformer override.
An existing local folder is read directly, including folders outside the cache.
`--local-files-only` is an explicit offline option: it requires all needed
model and code files to be available locally and does not download missing files.

Native Qwen VAE accepts `--vae-model-path PATH_OR_ID` as an override. Native
LTX Image VAE uses the selected pipeline's `vae` subfolder unless an explicit
VAE source is supplied. Component weight budgets remain separate and required
where documented. SDNQ and Qwen Turbo source selection follow their model guides.

Custom blocks execute trusted Python code. Use sources you trust; a Hub revision
does not change the HEAD of a local clone. Exact benchmark revisions belong in
the model's reproduction recipe.

All public LTX image/video and Qwen entries use the shared vocabulary, including
`--ddo-preset` and `--custom-blocks` where supported. The
linear and SDNQ entries support `--ddo-backend linear` only. All three minimal
entries use native `--ddo-backend vmm --ddo-preset auto`; their transformer uses
DDO's source-aware loading and bootstrap before CUDA-owning imports. Qwen's
conventional leaf control has no DDO backend/preset selectors. See the
[LTX image interface matrix](ltx2-image-parameters.md#interface-scope-by-runner)
for supported component and reporting controls. Diagnostic controls are kept
out of these public commands. See the [video interface matrix](ltx2-video-parameters.md#interface-scope-by-runner)
for video-specific sources, defaults and supported controls. Its native VMM entry
uses the same DDO route/capacity and reclaim controls below; video capacities and staging
defaults are listed in [video parameters](ltx2-video-parameters.md#native-capacity-and-components).

### PyTorch allocator startup default

All LTX Image, LTX Video and Qwen runner entries request
`PYTORCH_ALLOC_CONF=expandable_segments:True` before importing Torch, Diffusers
or DDO. The standard-library helper also runs before native bootstrap so the
fresh child inherits the setting. Bootstrap still installs the native allocator
before CUDA-owning imports.

If either `PYTORCH_ALLOC_CONF` or its legacy spelling
`PYTORCH_CUDA_ALLOC_CONF` is already present, the runner preserves the explicit
configuration, including an empty value. To disable the default explicitly,
set `PYTORCH_ALLOC_CONF=expandable_segments:False` before starting Python.
Use one spelling for explicit configuration.

This configures the ordinary PyTorch allocator. DDO's native pool retains its
own allocation policy, segmented backing, budgets and lifecycle. It does not
perform model chunking, change precision or set a total VRAM limit.

Reports record the allocator variables as `pytorch_allocator_environment`.
The linear LTX Video runner also records the values captured before its helper
imports as `pytorch_allocator_startup_environment`. Reading these variables
adds no CUDA query and happens outside inference timing. CPU tests cover the
18 entries and override precedence. The selected Linux Low RAM Safe 8-second
measurement completed with the explicit startup setting; the common setup
change has not been GPU-revalidated across every runner and platform.

### Public native image interface

All minimal entries select only `vmm/auto`. Their transformer capacity defaults
come from DDO: 4 GiB active weights, 0.625 GiB total staging, zero retained-source
cache and zero registration ceiling. These are configurable defaults, not the
model benchmark recipes. Reclaim defaults to false and requires its positive
switch. Encoders and decoders retain each minimal's existing component routes;
the VMM selector refers to the transformer. Use the dedicated linear entries
for `linear/auto` and other linear presets, and the full VMM entries for the
protected NVMe route or additional model controls.

`scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py` and
`scripts/qwen_image_21/run_dynamic_modular_vmm.py` expose exactly the same seven
DDO route/capacity and reclaim controls below. Model inputs,
sampling and supported component options remain model-specific.

| Control | LTX image default | Qwen image default |
| --- | --- | --- |
| `--ddo-backend` | `vmm` | `vmm` |
| `--ddo-preset` | `auto` | `auto` |
| `--ddo-vmm-resident-weight-budget-gb` | Saved profile or structural floor in native fast; 2 GiB fallback for other routes | 4 GiB without a selected profile; profile recommendation when selected |
| `--ddo-vmm-total-staging-gb` | Effective 1.875 GiB in native fast | Effective 0.625 GiB in native fast |
| `--ddo-vmm-source-cache-gb` | 0 (disabled) | 0 (disabled) |
| `--ddo-vmm-source-registered-gb` | 0 (inherits source-storage ceiling) | 0 (inherits source-storage ceiling) |
| `--ddo-vmm-execution-backing-reclaim` | False | False |

`vmm/auto` selects experimental native fast; `vmm/nvme_baseline` selects the
protected NVMe runtime. This Qwen VMM entry requires backend `vmm`; use its
linear entry for backend `linear`. The LTX entry also accepts backend `linear`,
although the dedicated linear runner is the documented starting point.
See [runtime selection](presets.md) for the route mapping.

These preserve the existing capacities, not the published benchmark overrides.
The model guides specify the measured capacities explicitly. Staging always
means total host staging in GiB; it is not activation VRAM or total GPU memory.
For the protected NVMe route, omitted staging preserves the existing two-slot
defaults (LTX total 1.25 GiB; Qwen total 1.25 GiB). Source-cache options apply to
the native fast route; do not treat them as NVMe controls.

The native source dispatcher follows DDO's public enabled default when the
source cache is used. It is not an extra model optimization switch. Component
reports expose `fast_source_cache_source_dispatcher_enabled`; routes without
that cache do not have an applicable dispatcher measurement.

LTX generation defaults are 1280×704, 8 steps, seed 43, guidance 1, guidance
rescale 0.7 and no PAG. Qwen defaults are base, 1024×1024, 25 steps and seed 42;
its 20-step benchmark recipe therefore keeps `--steps 20` explicit.
The public API selection is automatic. Public VMM runners support workload
capacity profiles as described below. Native adapter selection, GPU timing
and transport experiments belong in the
[diagnostic entries](../developer/diagnostics.md#explicit-image-diagnostic-entries).

Reporting defaults remain LTX: full memory monitoring, metrics level 1,
metrics display enabled and save disabled; Qwen: RSS, level 2, metrics display
and save enabled. Separate profile display is off by default in both.
Use the same explicit reporting settings for comparisons. Parameter availability
does not establish GPU or performance acceptance.

### Windows standby purge

Use the phase switches exposed by the selected runner:
`--purge-before-run`, `--purge-after-text-encoder`,
`--purge-before-transformer` and `--purge-after-run`.
The corresponding `--no-purge-...` switch disables a phase.
Some linear LTX entries also expose `--purge-before-vae`; check the selected
runner's help. This is separate from CUDA tensor cleanup before decode.

Run elevated when requesting purge and check the reported result.
Purge is off by default. Add a positive switch for each phase you want to
enable. Explicit switches take precedence over environment overrides where the
runner supports them. The published DDO RC5 wheel still enables the first
three phases through `one_shot_fast`. When using that wheel, pass the
corresponding `--no-purge-...` switches to disable those inherited phases.
Keep the phase switches explicit in benchmark recipes and inspect the saved
purge result rather than inferring success from a default. No administrator
terminal is required when all purge phases are disabled.
Outside Windows the helper is a safe no-op. Purge does not free live tensors or
guarantee that native backing has returned to the driver.

### Execution backing reclaim

The three public VMM entries accept `--ddo-vmm-execution-backing-reclaim` and
`--no-ddo-vmm-execution-backing-reclaim`. The default is **false**: only the positive
CLI switch enables it, even when `DDO_VMM_EXECUTION_BACKING_RECLAIM=1` is inherited.
Enabling requires `--ddo-backend vmm --ddo-preset auto`; linear, SDNQ and protected
NVMe execution do not support this mechanism. Do not export the environment
variable as shared setup for linear/SDNQ recipes.

Native execution buffers have stable virtual addresses and physical backing.
Under pressure, the allocator can release eligible idle backing after the last
consumer has completed. Buffers still leased or awaiting completion are protected.
When needed again, backing is recreated and its contents materialized before
the consumer can use it. This does not continuously release/recreate buffers:
without pressure they can remain available for reuse.

The potential benefit is more available GPU memory for activations and less
pressure from idle execution buffers. Recreating and materializing backing costs
time, so a speed improvement depends on the workload and hardware. This is not
tensor compression and does not increase the configured weight budget. An enabled
setting with zero releases can be valid when pressure or eligible idle buffers
are absent. Pool accounting does not prove local VRAM placement.

## Diagnostics

| Control | Purpose |
| --- | --- |
| `--show-profile` | Display the component profile |
| `--show-metrics` | Display collected measurements |
| `--save-metrics` | Save the JSON report |
| `--memory-monitor rss` / `full` | Select RSS / full USS monitoring |
| `--metrics-level 1` / `2` | Basic / detailed reporting, where implemented |
| `--ddo-gpu-timing` | Collect completed CUDA-event timing, where supported |
| `--ddo-profile NAME`, `--build-ddo-profile` | Select or build a persistent transformer capacity profile in supported linear, VMM and SDNQ entries; see the route-specific rules below |

GPU timing is available only where the selected runner exposes it. Capacity
profile selection is available in public modular runners; it does not require
enabling a diagnostic entry.

### Persistent capacity profiles for native VMM and SDNQ

Public modular VMM runners accept `--ddo-profile NAME` and
`--build-ddo-profile`. On `vmm/auto`, calibration runs three denoise steps
without decode. Run the same build again only when the report requests the
ceiling measurement; a final profile is ready for inference. The benchmark
controller handles this sequence. Calibration time is separate from generation.
Protected `vmm/nvme_baseline` uses its fixed budget instead of this profile.

Profiles must match the component, checkpoint, hardware and workload. Saved
FP8 uses its own profile, not a BF16 profile. SDNQ entries use their own packed
runtime calibration and optional validation; follow the model guide rather
than substituting a linear BF16 or VMM profile. Minimal runners do not expose
profile calibration.

### Persistent capacity profiles for linear runners

The BF16 modular linear entries for LTX image, LTX video and Qwen expose the same
two controls: `--ddo-profile NAME` selects a saved profile;
`--build-ddo-profile --ddo-profile NAME` creates one. Building requires a name and
a linear DDO preset (`auto`, `one_shot_fast`, `low_ram_safe`, `wsl_compat` or
`warm_process`). Native VMM and SDNQ use the separate rules above.
`--show-profile` only displays component measurements; it does not calibrate or
select a persistent profile.

Calibration is real GPU inference with a zero resident-module budget. DDO measures
the denoise memory peak and recommends a resident budget after subtracting its
headroom (1 GiB by default). This estimates capacity, not the fastest budget.
All three calibration modes stop before VAE decode and media export. LTX image
uses its existing short calibration of up to two denoise steps; LTX video uses
the requested distilled steps; Qwen base uses the requested steps, while Turbo
preserves its saved checkpoint schedule. These measurements are not complete
pipeline or decode peaks.

Profiles are JSON files under `~/.ddo/profiles` (on Windows,
`C:/Users/<user>/.ddo/profiles`), unless `DDO_PROFILE_DIR` is set. DDO owns the
profile identity, storage and admission of recommendations. Choose a distinct
name for each checkpoint, shape/duration, conditioning and computation recipe:
the profile name must distinguish workload changes that the model identity
alone does not capture. Reuse the same recipe and remove only
`--build-ddo-profile` to use the stored recommendation. Explicit fixed budgets
take precedence; automatic RAM admission and configured ceilings still apply.
A missing compatible recommendation is reported and retains the normal plan.

See executable creation/reuse examples for [LTX image](ltx2-image.md#linear-capacity-profile),
[LTX video](ltx2-video.md#linear-capacity-profile) and
[Qwen](qwen-image-21.md#linear-capacity-profile). Calibration adds GPU monitoring
and completion checks; CPU interface tests do not establish GPU acceptance.

### Public native reporting

For the two public native image entries, use the following reporting controls.

| Control | Behavior in the public native image entries |
| --- | --- |
| `--memory-monitor rss` / `full` | Process RSS or full USS monitoring. Full monitoring can add overhead; neither mode measures allocator ownership or proves local VRAM placement. |
| `--metrics-level` | Qwen accepts 1/2; LTX accepts 0/1/2. Level 2 additionally records public DDO component snapshots; phase monitoring and explicit display/save are separate. |
| `--show-profile` | Display the component's public DDO profile; default off. |
| `--show-metrics` | Display collected observations; default on. |
| `--save-metrics` | Persist the runner's JSON metrics; Qwen defaults on, LTX off. |
| `--show-denoise-steps` | Print per-step progress; default on. Callback times are host observations, not synchronized GPU compute durations. |

Supported display/save switches also accept their negative forms.
Both public image entries accept `--no-show-profile`, `--no-show-metrics`,
`--no-save-metrics`, `--no-show-denoise-steps` and `--no-local-files-only`.
The four purge switches above likewise accept negative forms. Other booleans
accept only the forms listed by their runner; do not infer a `--no-` form for
every opt-in experimental flag.
RSS and USS are different memory measurements; callback intervals are not
synchronized GPU timings. See [metric labels](../benchmarks/glossary.md).
Runners print the NVIDIA driver version and record it as
`nvidia_driver_versions`. Driver discovery happens before the tracked inference
interval, so its subprocess time is excluded from inference latency.
All listed LTX image/video and Qwen runners accept `--memory-monitor rss` and
`--memory-monitor full` and pass the choice to the shared monitor. Defaults
remain runner-specific to preserve existing recipes; select a mode explicitly
when comparing measurements.

Model guides list their additional controls. Contributors can consult
[diagnostics and reporting](../developer/diagnostics.md) for timing semantics
and the [runner interface contract](../developer/runner-interface.md).

## Model sources and investigations

Use `--model-path` for the checkpoint in all public LTX image entries and the
public Qwen entries. SDNQ source roles are specified in the model guides.
Native API selection is automatic. Use the dedicated diagnostic entries for
adapter, backing and transport investigations.

Complete model-specific inventories: [Qwen native image parameters](qwen-image-21-parameters.md)
and [LTX image parameters](ltx2-image-parameters.md).
