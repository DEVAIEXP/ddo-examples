# Shared runner controls

Run commands from the examples root in the installed environment. Each runner's
`--help` defines the options it supports; minimal and conventional controls
expose only a subset. Shared options use the same spelling and meaning.

## Execution

| Control | Purpose |
| --- | --- |
| `--ddo-backend linear` / `vmm` | Transformer route family |
| `--ddo-preset NAME` | Choose a [runtime and preset](presets.md) |
| `--pipeline-path PATH_OR_ID` | Pipeline and auxiliary weights |
| `--transformer-path PATH_OR_ID` | Separate transformer weights, where supported |
| `--custom-blocks SOURCE`, `--custom-blocks-revision REVISION` | Model code source and optional revision, where supported |
| `--local-files-only` | Require cached model/code files; input media URLs are separate |
| `--output-dir DIRECTORY` | Output media and reports |
| `--ddo-vmm-resident-weight-budget-gb VALUE` | Active transformer weight capacity in GiB |
| `--ddo-vmm-total-staging-gb VALUE` | Total reusable VMM staging capacity in GiB |
| `--ddo-vmm-source-cache-gb VALUE` | Retained source capacity in GiB |
| `--ddo-vmm-source-registered-gb VALUE` | CUDA-registered source capacity in GiB |
| `--vae-resident`, `--vae-leaf-offload`, `--vae-linear-offload` | Explicit VAE strategy, where implemented |
| `--vae-tiling` / `--no-vae-tiling` | Spatial VAE tiling, where supported |

Custom blocks execute trusted Python code. Use sources you trust; a Hub revision
does not change the HEAD of a local clone. Exact benchmark revisions belong in
the model's reproduction recipe.

### Windows standby purge

Use the phase switches exposed by the selected runner:
`--purge-before-run`, `--purge-after-text-encoder`,
`--purge-before-transformer` and `--purge-after-run`.
The corresponding `--no-purge-...` switch disables a phase.

Run elevated when requesting purge and check the reported result.
Explicit switches take precedence over environment settings, then preset defaults.
Outside Windows the helper is a safe no-op. Purge does not free live tensors or
guarantee that native backing has returned to the driver.

## Diagnostics

| Control | Purpose |
| --- | --- |
| `--show-profile` | Display the component profile |
| `--show-metrics` | Display collected measurements |
| `--save-metrics` | Save the JSON report |
| `--memory-monitor rss` / `full` | Select RSS / full USS monitoring |
| `--metrics-level 1` / `2` | Basic / detailed reporting, where implemented |
| `--ddo-gpu-timing` | Collect completed CUDA-event timing, where supported |
| `--ddo-profile NAME`, `--build-ddo-profile` | Select or build a workload profile, where supported |

Supported display/save switches also accept their negative forms.
RSS and USS are different memory measurements; callback intervals are not
synchronized GPU timings. See [metric labels](../benchmarks/glossary.md).
All listed LTX image/video and Qwen runners accept `--memory-monitor rss` and
`--memory-monitor full` and pass the choice to the shared monitor. Defaults
remain runner-specific to preserve existing recipes; select a mode explicitly
when comparing measurements.

Model guides list their additional controls. Contributors can consult
[diagnostics and reporting](../developer/diagnostics.md) for timing semantics
and [interface compatibility](../developer/runner-interface.md) for older commands.
