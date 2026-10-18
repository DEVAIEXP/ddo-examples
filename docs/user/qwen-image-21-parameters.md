# Qwen Image 2.1 parameters

Use the selected runner's `--help` for its full list.
See [shared controls](runner-controls.md) for source, preset and reporting options.

## Execution

| Control | Purpose |
| --- | --- |
| `--variant base` / `turbo` | Model and sampling schedule family |
| `--width`, `--height`, `--seed`, `--prompt` | Generation inputs |
| `--steps` | Base schedule length; not a Turbo schedule replacement |
| `--text-encoder-preset NAME` | Independent encoder policy, where supported |
| `--vae-resident`, `--vae-leaf-offload`, `--vae-linear-offload` | VAE strategy, where implemented |
| `--vae-tiling` | Spatial tiling when VAE activations do not fit |

The SDNQ guide covers only the selected base INT4 recipe, not Turbo.

## Diagnostics

Use the [shared metrics/profile controls](runner-controls.md#diagnostics).
Native cache release and cuBLAS workspace clearing are explicit advanced
boundary operations, not automatic consequences of transformer unload.
Their semantics and timing limits belong to
[developer diagnostics](../developer/diagnostics.md).
