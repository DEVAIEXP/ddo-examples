# LTX video parameters

Use the selected runner's `--help` for its full list.
See [shared controls](runner-controls.md) for source, preset and reporting options.

## Execution

| Control | Purpose |
| --- | --- |
| `--width`, `--height` | Frame dimensions |
| `--seconds`, `--frame-rate` | Duration and frame rate |
| `--steps`, `--seed`, `--prompt` | Generation inputs |
| `--image` | Conditioning image path or URL |
| `--use-audio-condition --audio PATH` | Enable conditioning from input audio |
| `--no-use-audio-condition` | Generate without input-audio conditioning |
| `--decode-device cpu` / `cuda:0` | VAE execution device |
| `--stream-video-decode` | Enable temporal window decoding |
| `--transformer-fp8-weight-only` | Load supported saved FP8 weights from the transformer override |
| `--bits 4` / `8` | Saved SDNQ checkpoint choice in the SDNQ entry |

Input-audio conditioning and generated output audio are different.
Omitting streaming currently leaves it disabled; GPU streaming is an explicit
recipe choice, not an already changed parser default.

## Diagnostics

The diagnostics entry exposes compute experiments and tracing not accepted by
normal entries. SDNQ's `--trace-transformer-blocks` prints block boundaries.
Tracing adds overhead and should not be enabled for a performance comparison.

Compact timestep, modulation, residual, chunking, RMS/rotary and compilation
options are model-specific compute changes. Keep the benchmark's exact choices;
they are not generic DDO presets. See
[developer diagnostics](../developer/diagnostics.md).
