# LTX 2.3 image

The image runners import the independent clone at
modular_diffusers/custom_blocks/ltx2_image.

## Stable linear route

    uv run --no-sync python scripts/ltx2_image/run_dynamic_modular_distilled.py --help
    uv run --no-sync python scripts/ltx2_image/run_dynamic_modular_distilled.py --preset auto --show-metrics --save-metrics

For the smallest existing staged integration example:

    uv run --no-sync python scripts/ltx2_image/run_dynamic_minimal.py --help

Use --model-path for weights. Profiles and staged prompt/denoise/decode behavior
are preserved. --fake-prompt is diagnostic, not full prompt-performance evidence.

## Native VMM

    uv run --no-sync python scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py --ddo-backend vmm --ddo-preset auto --ddo-public-native-runtime --help

Bootstrap precedes Torch import. Set explicitly approved storage and registration
ceilings with --ddo-vmm-source-cache-gb and --ddo-vmm-source-registered-gb.
--text-encoder-native-vmm opts into the separately staged encoder path.
The historical image profile flag is --show-ddo-profile. Use --show-metrics
and --save-metrics for reports.

## SDNQ compatibility

Install the sdnq extra, then inspect:

    uv run --no-sync python scripts/ltx2_image/run_dynamic_modular_sdnq_distilled.py --help

Preserve backend-specific forwards and use compatibility presets. SDNQ is not
the saved TorchAO FP8 native contract. No new image optimization or CUDA benchmark
is part of migration.
