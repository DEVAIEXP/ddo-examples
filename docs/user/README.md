# Run the examples

## 1. Install

Follow [package installation](setup.md) to prepare an environment without DDO
source checkouts or a native compiler.

## 2. Choose a model

These pages contain runner commands and model-specific options:

- [LTX 2.3 image](ltx2-image.md)
- [LTX 2.3 video and audio](ltx2-video.md)
- [Qwen Image 2.1](qwen-image-21.md)

Run commands from the examples repository root. For options shared across models,
including metrics and Windows standby purge, use the
[runner controls reference](runner-controls.md).

The primary model recipes use the public runners directly and match the
workloads in the current benchmark tables. The benchmark controller is optional.
Follow the selected platform's component routes, attention, purge policy,
budgets and profile calibration instructions; feature experiments are separate
recipes. Timings depend on hardware, driver and host memory state.

See [Runtimes and presets](presets.md) to choose a route based on the model measurements and available memory.

For optional model attention code, dependencies and provenance, see
[optional kernels](kernels.md).
