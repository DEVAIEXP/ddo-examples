# Primary host Linux environment (2026-10-11)

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
