# WSL developer setup

Audio reproducibility update (2026-10-12): host pyproject/lock now select
TorchAudio `2.11.0+cpu` from the explicit CPU index on Linux/WSL while preserving
Windows selection and CUDA Torch. The manual audio install in the historical
tested recipe below is no longer necessary for a sync using the updated lock.
The lock passed validation and a native-Linux sync dry-run; no automatic WSL
resync or generation repeat was executed. Preserve the existing WSL environment
and select its path explicitly when a sync is reviewed.

Selected BF16 validation closed on 2026-10-12: LTX image, conditioned five-second
video and Qwen base with leaf VAE. This is selected-workload compatibility,
not general WSL support or matched cross-platform performance. Quantized WSL
routes remain untested. See the dated results in experiments/dynamic_offload_results.md.

Use the GPU-enabled distribution where /dev/dxg exists and
/usr/lib/wsl/lib/nvidia-smi works. Never install a Linux NVIDIA kernel driver
inside WSL. Tested Ubuntu 22.04.3/glibc 2.35, WSL kernel 6.18.33.2, RTX 3060 Ti
8 GiB, about 39 GiB RAM and 32 GiB swap. A venv from another distribution/user
can have dangling Python links. Create a separate environment and preserve others.

## Python and media

Install uv if absent. Tested uv 0.9.11, Python 3.12.12, Torch 2.12.1+cu132,
official Diffusers 0.41.0 and TorchCodec 0.15.0+cu132:

```bash
uv venv .venv-wsl-gpu --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-wsl-gpu uv sync --locked --link-mode copy
uv pip install --python .venv-wsl-gpu/bin/python --no-deps --link-mode copy \
  --index-url https://download.pytorch.org/whl/cpu 'torchaudio==2.11.0+cpu'
sudo apt install ffmpeg
ffmpeg -version
.venv-wsl-gpu/bin/python -B -c "import torch, torchcodec; print(torchcodec.__version__); print(torch.cuda.is_initialized())"
```

FFmpeg 4.4.2 system shared libraries resolved media imports in this distribution.
Other distributions may require different libav packages. The audio-only CPU
wheel resolves the inspected version mismatch; inference stays CUDA. Fresh sync
can undo this override: use the direct interpreter or uv run --no-sync. Do not
apply it blindly to Windows. Reproducible audio selection remains roadmap work.
If --locked fails, review the lock change instead of silently dropping the guard.

## Explicit native developer build

Python editable installation does not compile the native library. The preserved
Linux binary required newer glibc/C++ ABI. Build locally without changing glibc
or overwriting accepted binaries. Install system g++ if absent; tested g++ 11.4.0.
Pip toolkit extras install compiler/headers, not a GPU kernel driver:

```bash
uv pip install --python .venv-wsl-gpu/bin/python --link-mode copy \
  'cuda-toolkit[nvcc,cccl]==13.2.1' 'cmake>=3.25,<5'
.venv-wsl-gpu/bin/cmake \
  -S ../diffusers-dynamic-offloader/ddo-vmm-allocator \
  -B ../diffusers-dynamic-offloader/ddo-vmm-allocator/build-wsl-idle-trim-20261012 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="$PWD/.venv-wsl-gpu/lib/python3.12/site-packages/nvidia/cu13" \
  -DCUDA_cuda_driver_LIBRARY=/usr/lib/wsl/lib/libcuda.so \
  -DDDO_VMM_BUILD_PROBES=OFF -DDDO_VMM_NATIVE_SPAN_POOL=ON \
  -DDDO_VMM_NATIVE_JOINED_SPANS=OFF -DDDO_VMM_NATIVE_SOURCE_DISPATCHER=OFF \
  -DDDO_VMM_SOURCE_TEST_HOOKS=OFF -DDDO_VMM_ALLOCATION_DIAGNOSTICS=OFF \
  -DDDO_VMM_BUILD_EXECUTION_OPS=OFF
.venv-wsl-gpu/bin/cmake --build \
  ../diffusers-dynamic-offloader/ddo-vmm-allocator/build-wsl-idle-trim-20261012 \
  --parallel 2
ldd ../diffusers-dynamic-offloader/ddo-vmm-allocator/build-wsl-idle-trim-20261012/libddo_vmm.so
sha256sum ../diffusers-dynamic-offloader/ddo-vmm-allocator/build-wsl-idle-trim-20261012/libddo_vmm.so
```

Observed nvcc 13.2.78/CMake 4.4.4; system CMake 3.22.1 was too old.
Corrected candidate SHA256:
5915cca767c89ab7606c04f2001c9eddffa762386c852cc9f517708a2d03584c.
Native cgroup-root identity was corrected without weakening admission guards.
Native CPU contracts, 53 selected Python CPU tests and three CUDA cache contracts
passed. Rebuilding later sources does not inherit this binary's acceptance.

## Cache and registration

Set HF_HUB_CACHE=/mnt/f/hf_cache/hub before Python to reuse Windows weights.
Offline full-snapshot lookup may reject missing non-runtime files; inspect
required configs/shards and use explicit snapshot paths where appropriate.
Local custom-block clones stay ignored.

Standalone registration passed 1152 MiB twice and failed at 1216 MiB, at 64 MiB
resolution. This is not a universal WSL limit. Models used 384 MiB staging plus
512 MiB source registration, separately from retained storage/GPU residency.
No OS limits changed. Registration churn remains costly. Do not assign the
observed total independently to staging and cache or silently raise ceilings.
The bounded probe is documented in sibling DDO docs/host-registration-probe.md.
All CUDA/model tests require reviewed commands and approval; do not rerun the
closed sequence automatically. Unified wheels/discovery/setup remain roadmap work.
