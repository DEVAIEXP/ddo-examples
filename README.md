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
Diffusers is pinned to the official PyPI package `diffusers==0.41.0`.
Model-specific image/video improvements come from independent custom blocks,
not a modified Diffusers installation. CPU compatibility is checked separately
from generation/performance acceptance. Do not inject a fork through PYTHONPATH.
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
| Qwen-Image 2.1 | [Candidate guide](docs/qwen-image-21.md): local BF16 base/Turbo modular blocks; CUDA validation pending |

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
[benchmark summary](docs/benchmark-summary.md) for gains, trade-offs and
comparison limits; the [benchmark guide](docs/benchmarks.md) for recording;
and the [detailed evidence log](experiments/dynamic_offload_results.md).
Selected five-second video controls observed 1.75x FP8 native denoise speedup
versus streamless official leaf and 2.03x SDNQ INT4 denoise speedup versus the
recorded streamed leaf control. These are not allocator-only or universal gains.
BF16 native remains faster than saved FP8 native in the current full-video case.
DDO retains technical
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
The selected BF16 native VMM five-second/eight-step full-video gate passed:
203.7925 s denoise and 266.9297 s tracked total, with successful cleanup.
Historical timings are reference evidence, not a matched migration speedup.
The post-reboot BF16 image repeat recovered historical performance (1.388 s
CUDA warm step mean, 35.26 s tracked total). Administrative Windows video
runs with confirmed standby purge completed for BF16 and saved FP8: respectively
199.36 / 218.59 s denoise and 262.47 / 312.98 s tracked full-video total.
FP8 is functional for this selected checkpoint, not faster than BF16 and not
a general quantization-recipe acceptance. Further CUDA gates require approval.
Linux needs its own .so and environment; Windows evidence does not validate
Linux, WSL, macOS or arbitrary models. See DDO's docs/linux-handoff.md.
Copied runners retain historical diagnostics, not new recommended presets.
See [migration validation](docs/migration-validation.md) for the accepted CPU
checks and the remaining approved-generation gate.
