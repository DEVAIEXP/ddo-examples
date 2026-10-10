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
| LTX 2.3 video | [Video guide](docs/ltx2-video.md): staged video/audio, BF16, saved FP8 and experimental SDNQ |

Run commands from the repository root with uv run --no-sync.
Outputs, weights, virtual environments and custom-block clones are untracked.

Both model directories use scripts/helpers/inference_utils.py for the same
telemetry schema. Video follows the image names: run_dynamic_minimal.py,
run_dynamic_modular_distilled.py and run_dynamic_modular_vmm_distilled.py.
Video compilation/kernel experiments and comparison controls have a separate
run_dynamic_modular_diagnostics_distilled.py entry. The old video filename
remains a compatibility alias; no experimental options become enabled defaults.
Reusable compute utilities live under helpers/execution; LTX adapters live
under helpers/ltx2. Each public runner owns its visible parsing and phase
orchestration. Minimal examples intentionally repeat the essential integration
instead of delegating generation to a hidden dispatcher. Image optimization support
must be validated separately before enabling video-specific callbacks there.

## Benchmarks

Pipeline comparisons and measured results live here. See the
[benchmark guide](docs/benchmarks.md) and
[results](experiments/dynamic_offload_results.md). DDO retains technical
runtime/allocator documentation. Historical runs do not validate migrated runners.

## Native allocator

DDO discovers an installed native binary or a conventional checkout build.
Normally users do not supply a DLL path. Editable installation alone does not
compile or provide that binary: use a reviewed allocator wheel or build.
See the DDO repository's docs/native-wheel-delivery.md and docs/native-vmm.md.
DDO_VMM_LIBRARY_PATH remains an explicit diagnostic override.

CPU migration checks and one selected Windows BF16 image generation passed.
Selected SDNQ INT4 image and full five-second video execution also passed;
the corrected DDO video denoise took 319.6422 s versus the recorded leaf
control's 649.6985 s. See the benchmark results for exact scope and limitations.
BF16 image performance parity, native VMM video generation and saved FP8
execution in this new environment remain unaccepted; further CUDA gates
require explicit approval.
Linux needs its own .so and environment; Windows evidence does not validate
Linux, WSL, macOS or arbitrary models. See DDO's docs/linux-handoff.md.
Copied runners retain historical diagnostics, not new recommended presets.
See [migration validation](docs/migration-validation.md) for the accepted CPU
checks and the remaining approved-generation gate.
