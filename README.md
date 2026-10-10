# DDO examples

Primary host for model-specific Modular Diffusers runners using
[DDO](https://github.com/DEVAIEXP/diffusers-dynamic-offloader).
Generic allocation, transfers and FP8 transport stay in DDO; model math stays
in custom blocks.

## Setup

Python 3.12, Git and uv are required. Clone this repository next to
diffusers-dynamic-offloader, with its allocator submodule initialized.

    uv sync --extra fp8 --extra kernels
    uv run --no-sync scripts/setup_custom_blocks.py

The root .venv supports image and video. Add --extra sdnq for SDNQ.
Diffusers is pinned to the fork's LTX image-support commit; video transformer
improvements come from custom blocks, not a modified Diffusers installation.
CUDA 13.2 Torch and audio pins follow the existing host environments.

Custom-block clones live under ignored modular_diffusers/custom_blocks/.
They are independent repositories, not submodules. Review their code before
execution. Setup refuses to overwrite existing clones at different revisions.
Old image/video host repositories are optional and unchanged.

## Runners

| Model | Usage |
| --- | --- |
| LTX 2.3 image | [Image guide](docs/ltx2-image.md): linear, native VMM and SDNQ |
| LTX 2.3 video | [Video guide](docs/ltx2-video.md): staged video/audio, BF16 and saved FP8 |

Run commands from the repository root with uv run --no-sync.
Outputs, weights, virtual environments and custom-block clones are untracked.

## Native allocator

DDO discovers an installed native binary or a conventional checkout build.
Normally users do not supply a DLL path. Editable installation alone does not
compile or provide that binary: use a reviewed allocator wheel or build.
See the DDO repository's docs/native-wheel-delivery.md and docs/native-vmm.md.
DDO_VMM_LIBRARY_PATH remains an explicit diagnostic override.

CPU migration checks and one selected Windows BF16 image generation passed.
Performance parity, video generation and saved FP8 execution in this new
environment remain unaccepted; further CUDA gates require explicit approval.
Linux needs its own .so and environment; Windows evidence does not validate
Linux, WSL, macOS or arbitrary models. See DDO's docs/linux-handoff.md.
Copied runners retain historical diagnostics, not new recommended presets.
See [migration validation](docs/migration-validation.md) for the accepted CPU
checks and the remaining approved-generation gate.
