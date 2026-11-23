# Install packages for the examples

Follow the common steps first, then choose the Windows or Linux/WSL section.
Install optional packages only for the features you plan to use.
For source development, see [developer setup](../developer/setup.md).

## Common steps

Use Python 3.12, uv, Git and a compatible NVIDIA GPU/system driver. The packaged
native allocator needs no compiler or Docker installation.

Clone the examples:

```sh
git clone https://github.com/DEVAIEXP/ddo-examples.git
cd ddo-examples
```

DDO and the allocator are installed as packages; their source repositories are
not required. Run the examples directly from this checkout. Keep Windows,
native Linux and WSL environments separate.

The commands below install DDO and allocator `0.1.0rc5`, together with the
model dependencies in `pyproject.toml` and versions in `constraints.txt`.
They select an explicit Python environment, so activation is not needed during
installation. Matching packages already installed are reused. Stop if a command
fails. This guide uses `uv pip`.

### Choose optional features

Ordinary SDPA recipes need only the basic installation. Installing an extra
makes a feature available; select it separately in the model's runner.

| If you want to use… | Install… |
| --- | --- |
| FlashAttention 2 | The `flash-attn` extra (Python 3.12, Torch 2.12.1+cu132; see your platform section) |
| Sol-Attn for LTX video | FlashAttention 2 **and** Triton |
| Saved FP8 weight-only models | The `fp8` extra (TorchAO) |
| SDNQ quantized models | The `sdnq` extra |

<a id="optional-sol-attn-for-ltx-video"></a>

Sol-Attn source is included in `kernels/sol_attn` in this examples repository.
It uses FlashAttention 2 for the first video transformer block and Triton for
the remaining eligible blocks. After installation, select
`--video-self-attention sol-attn`. See [optional kernels](kernels.md) for
licenses, supported conditions and measured results.

## Windows PowerShell

### Create the environment

Run from the cloned examples root. If you already have a compatible package
environment, set `$python` to its interpreter instead of creating another one.

```powershell
$examples = (Get-Location).Path
uv venv --managed-python --python 3.12.12 .venv-packages
$python = (Resolve-Path (Join-Path $examples '.venv-packages/Scripts/python.exe')).Path
```

### Install the basic packages

```powershell
uv pip install --python $python --link-mode copy --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132' 'torchvision==0.27.1+cu132' 'torchcodec==0.15.0+cu132'
uv pip install --python $python --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml"
uv pip check --python $python
```

### Optional: FlashAttention 2

This extra installs the third-party Windows wheel for Python 3.12,
Torch 2.12.1 and CUDA 13.2:

```powershell
uv pip install --python $python --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml" --extra flash-attn
```

### Optional: Sol-Attn

Install both dependencies in one command. The `kernels` extra selects
`triton-windows` on Windows:

```powershell
uv pip install --python $python --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml" --extra flash-attn --extra kernels
```

### Optional: FP8 or SDNQ

For saved FP8 weight-only models:

```powershell
uv pip install --python $python --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml" --extra fp8
```

For SDNQ models:

```powershell
uv pip install --python $python --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml" --extra sdnq
```

### Configure FFmpeg on Windows

For video/audio, use an FFmpeg **shared** build containing DLLs such as
`avcodec-*.dll`, `avformat-*.dll` and `avutil-*.dll`. An `ffmpeg.exe` on PATH
alone is not sufficient for TorchCodec's shared-library loading.

All LTX video runners accept `DDO_RUNNER_FFMPEG_BIN`. Set it to the directory
containing those DLLs, not to the executable. Replace this example with your
installation path, and run it in the same PowerShell session as the runner:

```powershell
$env:DDO_RUNNER_FFMPEG_BIN = 'C:/tools/ffmpeg/bin'
```

This setting lasts for the current terminal session and is inherited by the
native bootstrap child. Set it again after opening another terminal. It is a
video-runner setting, not a DDO allocator option. Linux/WSL use their system
FFmpeg shared libraries instead.

Check audio-library loading before generating, using the selected `$python`:

```powershell
& $python -I -B -c "import os; dlls=os.add_dll_directory(os.environ['DDO_RUNNER_FFMPEG_BIN']); import torch, torchcodec; print('TorchCodec: PASS'); print('CUDA initialized:', torch.cuda.is_initialized())"
```

Expected: `TorchCodec: PASS` and `CUDA initialized: False`. This checks library
loading, not model execution or audio/video output.

After installing your chosen options, check the environment:

```powershell
uv pip check --python $python
```

## Native Linux or WSL Bash

WSL uses the Windows NVIDIA driver. Install Linux dependencies inside WSL,
but do not install a Linux NVIDIA GPU driver there.

### Create the environment

Run from the cloned examples root. If you already have a compatible package
environment, set `python` to its interpreter instead.

```bash
examples="$PWD"
uv venv --managed-python --python 3.12.12 .venv-packages
python="$examples/.venv-packages/bin/python"
```

### Install the basic packages

```bash
uv pip install --python "$python" --link-mode copy --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132' 'torchvision==0.27.1+cu132' 'torchcodec==0.15.0+cu132'
uv pip install --python "$python" --link-mode copy --index-url https://download.pytorch.org/whl/cpu 'torchaudio==2.11.0+cpu'
uv pip install --python "$python" --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml"
uv pip check --python "$python"
```

CPU TorchAudio handles audio decoding; model inference remains CUDA.

### Optional: FlashAttention 2

The `flash-attn` extra installs a FlashAttention 2.8.3 wheel built by this
project for Python 3.12, Torch 2.12.1 and CUDA 13.2 on x86-64. WSL2 and
native Linux receive separate wheels: the extra selects the WSL2 wheel when
the kernel release contains `microsoft`. Both wheels require glibc 2.34 or
newer (Ubuntu 22.04 or newer). They are published with their SHA256 hashes in
the
[FlashAttention wheels release](https://github.com/DEVAIEXP/ddo-examples/releases/tag/flash-attn-2.8.3-torch2.12.1-cu132-cp312):

```bash
uv pip install --python "$python" --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml" --extra flash-attn
```

These wheels match only the Torch build above. For another Torch, CUDA or
Python version, install your own compatible wheel instead:

```bash
uv pip install --python "$python" --no-deps /path/to/compatible-flash-attn.whl
```

An existing compatible installation can be reused. Check the installed versions:

```bash
"$python" -I -B -c "import torch, flash_attn; print('Torch:', torch.__version__); print('FlashAttention:', flash_attn.__version__)"
```

### Optional: Sol-Attn

First install FlashAttention 2 as above, then add Triton:

```bash
uv pip install --python "$python" --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml" --extra kernels
```

The `kernels` extra selects `triton` on Linux and WSL. Sol source is already
included with the examples; no separate source checkout is needed.

### Optional: FP8 or SDNQ

For saved FP8 weight-only models:

```bash
uv pip install --python "$python" --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml" --extra fp8
```

For SDNQ models:

```bash
uv pip install --python "$python" --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml" --extra sdnq
```

### Configure FFmpeg for video/audio

Install compatible system FFmpeg shared libraries. On Ubuntu 22.04:

```bash
sudo apt install ffmpeg
```

Package names differ on other distributions. Check library loading before
loading a model:

```bash
"$python" -I -B -c "import torch, torchcodec; print('TorchCodec: PASS'); print('CUDA initialized:', torch.cuda.is_initialized())"
```

Expected: `TorchCodec: PASS` and `CUDA initialized: False`. This checks library
loading, not generation. After installing your chosen options:

```bash
uv pip check --python "$python"
```

## Prepare model code and run

Prepare the LTX and Qwen code clones using the same Python:

```powershell
# Windows PowerShell
Set-Location -LiteralPath $examples
& $python -B scripts/setup_custom_blocks.py
```

```bash
# Native Linux or WSL
cd "$examples"
"$python" -B scripts/setup_custom_blocks.py
```

These are independent model-code repositories, not model weights. The helper
uses `main` for new clones and preserves existing checkouts without pulling or
overwriting them. A different origin is rejected. To prepare one model at an exact
revision, use `--model ltx2_image --revision <commit-sha>`; an existing checkout
must already match that revision. Custom blocks
execute Python: review their source and pinned revision before trusting them.
Other model guides explain their Hub/local code-loading options.

Activate the selected environment before using the model guides' `python -B`
commands. These paths also work when the environment is outside the checkout:

```powershell
# Windows PowerShell
. (Join-Path (Split-Path -Parent $python) 'Activate.ps1')
```

```bash
# Native Linux or WSL
source "$(dirname "$python")/activate"
```

Alternatively, replace `python` with the environment's explicit interpreter.
Choose a runner from the [model guides](README.md#2-choose-a-model) and obtain its
weights under their licenses. Do not use an unrelated active environment.

Selected platform/model evidence is not all-model compatibility. WSL staging and
registration ceilings may differ; use documented conditions rather than
automatically increasing budgets.

## Common setup errors

### `Could not load libtorchcodec` on Windows

First follow [FFmpeg configuration](#configure-ffmpeg-on-windows). Confirm the
directory contains the shared DLLs and that the variable is set in the terminal
launching the runner. A missing-module message can mean a DLL dependency is
missing, even when `libtorchcodec_core*.dll` itself exists.

If the import check still fails, preserve the complete traceback and verify
Torch/TorchCodec versions against the installation commands above. Do not
upgrade one independently or copy unrelated DLLs into the Python environment.
Resolve this check before repeating a model generation.
