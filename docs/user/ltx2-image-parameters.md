# LTX image parameters

Use the selected runner's `--help` for its full list.
See [shared controls](runner-controls.md) for source, preset and reporting options.

## Execution

- `--width`, `--height`, `--steps`, `--seed`, `--prompt`: generation inputs.
- `--text-encoder-native-vmm`: native encoder in the VMM runner, independently of the transformer.
- `--auxiliary-preset`: auxiliary component policy in the VMM harness.

Keep encoder, VAE and transformer choices fixed when reproducing a benchmark.

## Diagnostics

`--fake-prompt` skips real prompt encoding; it is not a full-pipeline benchmark.
Profiles and metrics use the [shared switches](runner-controls.md#diagnostics).
See [developer diagnostics](../developer/diagnostics.md) before interpreting timings.
