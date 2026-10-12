# Primary host Linux environment (2026-10-11)

## Current package transition (2026-10-12)

### Audio selection now locked

The reviewed host pyproject selects the explicit PyTorch CPU index for
TorchAudio only when `sys_platform == 'linux'` (including WSL). The lock records
`2.11.0+cpu` on that platform and preserves PyPI `2.11.0` for Windows. Torch,
TorchVision and TorchCodec retain their explicit CUDA 13.2 index. This closes
the manual audio-override reproducibility gap; historical instructions below
describe preparation before this correction.

`uv lock --check` passed. A locked/inexact dev-group sync dry-run against the
existing `.venv-linux` proposed only refreshing editable DDO metadata, not
changing Torch, Diffusers or audio/media packages. No actual sync or fresh-venv
install was performed; this is resolution/dry-run evidence, not a second
generation acceptance or a promise for all architectures/distributions.

For reviewed environment setup, select the platform environment explicitly:

```bash
UV_PROJECT_ENVIRONMENT=.venv-linux uv sync --locked --group dev --inexact --link-mode copy
```

Use `.venv-wsl-gpu` instead on the selected WSL host; do not overwrite Windows
`.venv`. Native builds and FFmpeg system dependencies remain separate.

Only Diffusers was replaced in the existing host `.venv-linux` using:

```bash
uv pip install --python .venv-linux/bin/python --no-deps --link-mode copy 'diffusers==0.41.0'
```

Actual imports now resolve official `0.41.0` inside `.venv-linux/site-packages`,
not the editable fork. Torch remains `2.12.1+cu132`, TorchAudio `2.11.0+cpu`,
TorchCodec `0.15.0+cu132` and PyAV `18.1.0`. Imports passed with CUDA hidden and
`torch.cuda.is_initialized()` false; `uv pip check` reported all 81 packages
compatible. No automatic sync, Windows environment change or model generation
was performed. Historical model results below retain their original fork.

The host manifest/lock now select official Diffusers 0.41.0. The remaining
installed override is the Linux/WSL audio-only CPU wheel, not Diffusers.
Until platform-specific audio selection is locked, reproduce it after a reviewed
sync with the command below; do not assume ordinary sync preserves the override:

```bash
uv pip install --python .venv-linux/bin/python --no-deps --link-mode copy \
  --index-url https://download.pytorch.org/whl/cpu 'torchaudio==2.11.0+cpu'
```

Use the direct interpreter or `uv run --no-sync` for the accepted environment.
This audio workaround is not a model CPU fallback or an instruction to change
Windows packages. CUDA candidate and targeted model acceptance remain separate.

## Historical preparation and acceptance (2026-10-11)

Current official-package CPU acceptance: the host suite plus four selected
DDO/allocator CPU files passed **152 tests and 10 subtests in 36.93 s** on
2026-10-12, with CUDA hidden. This is separate from the historical results below
and does not grant CUDA/model acceptance to the changed environment.

The separate root `.venv-linux` was created with uv 0.12.21 and Python 3.12.14.
The existing Windows `.venv` and DDO's older Linux environment were preserved.
Preparation and subsequent selected CUDA/model gates are recorded below;
imports alone are not CUDA acceptance. The finite Linux sequence is closed.
See [Windows continuation](windows-handoff.md) for remaining work and the
installed overrides that still need a reproducible project profile.

## Installed setup

The reviewed locked host dependencies and dev group were installed using
`UV_PROJECT_ENVIRONMENT=.venv-linux uv sync --locked --group dev --link-mode copy`.
DDO and the allocator are editable sibling checkouts. Two subsequent overrides
are intentional and must not be silently lost on another sync:

- Diffusers is editable at `/run/media/eliseu/IA/Projetos/diffusers`, revision
  `7d047b180`, package version `0.41.0.dev0`, matching the merged local fork.
  The host lock still points to the older `e71ebc5` revision.
- TorchAudio is `2.11.0+cpu` from the official PyTorch CPU index. The plain
  Linux 2.11.0 wheel uses CUDA 13.0 and refused to import with Torch CUDA 13.2.
  The CPU audio wheel imported successfully; PyTorch remains `2.12.1+cu132`.
  No model or allocator CPU fallback is implied. A matching
  `torchaudio==2.12.1+cu132` wheel was unavailable on the inspected official index.

Use the activated interpreter directly, or `uv run --no-sync`, until the reviewed
source/platform overrides are incorporated into a reproducible project profile.
A fresh automatic sync can restore the older Diffusers and incompatible audio
wheel. Do not change the Windows environment or common dependency pins blindly.

```bash
cd /run/media/eliseu/AI2/ProjetosIA/ddo-examples
source .venv-linux/bin/activate
python -V
```

Core imports passed: Torch, local Diffusers, local DDO/allocator, TorchAudio,
PyAV and SoundFile. `torch.cuda.is_initialized()` remained false. Package
consistency also passed after the audio replacement. Imports are not proof of
model generation or CUDA allocator safety.

## Media prerequisite: resolved

After the user installed the identified system package, TorchCodec 0.15.0+cu132
and PyAV 18.1.0 imported successfully. Torch remained 2.12.1+cu132, TorchAudio
2.11.0+cpu, and CUDA initialization remained false. The dependency diagnosis
and original required action are preserved below; the import blocker is resolved.

TorchCodec `0.15.0+cu132` cannot load its FFmpeg 8 backend because
`libavdevice.so.62` is absent. Other FFmpeg 8 shared libraries are installed.
APT reports `libavdevice62` candidate `7:8.0.1-3ubuntu2` on Ubuntu 26.04.1.
No system package or driver was installed during environment preparation.

Run in the user's terminal if approved, then recheck imports:

```bash
sudo apt install --no-install-recommends libavdevice62
```

This addressed the identified missing library; successful TorchCodec import
has now been verified. Any additional missing dependency must be reported.
Do not start the video benchmark merely because this package was installed.

## Python CPU gate: passed

The eight files selected in the platform handoff passed: **52 tests in 6.31 s**,
with no skips reported. This used the host `.venv-linux`, `CUDA_VISIBLE_DEVICES=-1`,
`PYTHONPATH` set to the host root and its `modular_diffusers` directory, Python
`-B` and pytest `-p no:cacheprovider`.

Scope: DDO VAE component policy, linear convolution ownership, device-cache
helper mocks, allocator optional-trim ABI mocks, LTX streaming device/flag
contracts, Qwen leaf/linear runner contracts and video decode route contracts.
No native C++ build/test executable, real checkpoint, CUDA gate or benchmark
was run. These passes do not validate the preserved `.so` against current sources.

## Native library boundary

The preserved `build-linux/libddo_vmm.so` still has SHA256
`a87979f8773a010ddc2c781a37f3ef55112ca558a9d93083b972f51246a2b245`.
Export inspection found no `ddo_vmm_torch_trim_idle_v1`. No rebuild, native load,
CUDA probe or benchmark was performed. Continue through
[platform handoff](platform-validation-handoff.md), with CPU contracts first
and reviewed commands/confirmation before any CUDA gate.

Subsequent approved compilation produced a separate candidate at
`ddo-vmm-allocator/build-linux-idle-trim-20261011/libddo_vmm.so`, SHA256
`db63138da2fdcca328a244ff885ca44ecd8b33683825b2ab0f6f39a95b3cba79`.
The original library remains unchanged. Static export inspection confirms
`ddo_vmm_torch_trim_idle_v1`; no CUDA probe or model has run with this candidate.
Build options, warnings and the proposed small ownership gate are recorded in
DDO `docs/native-linux-idle-trim-gate.md`. Compilation is not CUDA acceptance.

The user subsequently approved the isolated CUDA idle-trim probe and it passed
on that candidate: 32 MiB pending idle backing released, the 1024-byte active
payload preserved exactly, repeated trim and allocation-after-trim successful,
and final mapped/live/pending/quarantined bytes and failures all zero. This is
a small pool ownership gate, not a model benchmark. Its executed evidence is in
DDO `docs/native-linux-idle-trim-gate.md`; image/video generation remains pending.

The subsequently approved primary-host BF16 image and five-second video
resident/leaf pairs also completed, with identical delivered PNG/MP4 hashes
within each Linux pair and no reported native or monitor failures. No WSL,
FP8/SDNQ, alternate VAE runtime or heavier model was tested. Current measurements
and recommendations are in benchmark-matrix.md and the dated pipeline evidence
log. Reproducible source/audio overrides and the review/commit checkpoint remain
pending; do not claim that a fresh locked sync reproduces this tested environment.
