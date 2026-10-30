# Install packages for the examples

This path uses installed packages, not editable DDO checkouts. For modifying
local sources, use [developer setup](../developer/setup.md) instead.

## Prerequisites

Use Python 3.12, uv, Git and a compatible NVIDIA GPU/system driver.
Clone the examples and install the packages below. No MSVC, CMake, nvcc or
Docker is needed to use the packaged native allocator.

Video/audio also needs compatible FFmpeg system shared libraries. On Ubuntu
22.04, install the distribution's `ffmpeg` package. Package names differ on other
distributions. WSL uses the Windows GPU driver, not a Linux GPU driver.

On Windows, use an FFmpeg **shared** build and put its `bin` directory on PATH
in the same terminal before starting an audio-conditioned runner. An executable-only
FFmpeg installation is insufficient for TorchCodec: it needs the shared DLLs
(including `avcodec`, `avformat` and `avutil`). For an existing shared installation:

```powershell
$env:PATH = 'C:\path\to\ffmpeg-shared\bin;' + $env:PATH
```

The runner and its bootstrap child inherit this PATH. Check audio decoding before
loading the model; a TorchCodec library-loading error is a dependency failure,
not a denoising result.

## 1. Clone the examples

Clone only `ddo-examples` to use the runners. DDO and the allocator are installed
as packages; their source repositories are not required. Choose a parent folder,
then run:

```sh
git clone https://github.com/DEVAIEXP/ddo-examples.git
cd ddo-examples
```

You do not install the examples as a Python package: run their scripts from this
checkout. One environment holds DDO, the allocator and all runner dependencies.

## 2. Create one environment

From the cloned examples root, create a uv-managed Python environment. This
avoids inheriting a Conda interpreter and its C++ runtime libraries. Keep
Windows, native Linux and WSL environments separate. If you already installed
DDO into a compatible environment, skip creation and select its Python below.

### Windows PowerShell

```powershell
$examples = (Get-Location).Path
uv venv --managed-python --python 3.12.12 .venv-packages
$python = (Resolve-Path (Join-Path $examples '.venv-packages/Scripts/python.exe')).Path
```

### Native Linux or WSL Bash

```bash
examples="$PWD"
uv venv --managed-python --python 3.12.12 .venv-packages
python="$examples/.venv-packages/bin/python"
```

Explicit interpreter paths select the environment without activation. Do not
set PYTHONPATH or DDO_VMM_LIBRARY_PATH to development checkouts or old builds.

## 3. Install packages and model dependencies

DDO and the allocator are installed from PyPI through `pyproject.toml`, pinned
to the validated `0.1.0rc2` pair. CUDA Torch packages come from the PyTorch index.
Stop if any command fails.

### Windows PowerShell

```powershell
uv pip install --python $python --link-mode copy --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132' 'torchvision==0.27.1+cu132' 'torchcodec==0.15.0+cu132'
uv pip install --python $python --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml"
uv pip check --python $python
```

### Native Linux or WSL Bash

```bash
uv pip install --python "$python" --link-mode copy --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132' 'torchvision==0.27.1+cu132' 'torchcodec==0.15.0+cu132'
uv pip install --python "$python" --link-mode copy --index-url https://download.pytorch.org/whl/cpu 'torchaudio==2.11.0+cpu'
uv pip install --python "$python" --no-sources --link-mode copy --index-url https://pypi.org/simple/ --constraint "$examples/constraints.txt" -r "$examples/pyproject.toml"
uv pip check --python "$python"
```

CPU TorchAudio handles audio decoding; model inference remains CUDA.
Dependencies come from `pyproject.toml`, with versions constrained by
`constraints.txt`. Already installed matching packages are reused.
No wheel folder or sibling source checkout is needed.

For optional recipes, add the required `--extra fp8`, `--extra kernels` or
`--extra sdnq` to the final metadata-install command. Install only what the runner
uses. This setup uses `uv pip`; do not substitute `uv sync` without a matching
package-based lock.

The optional `--extra flash-attn` installs a third-party Windows FlashAttention
2.9.1 wheel for Python 3.12, Torch 2.12.1 and CUDA 13.2. It does not install a
Linux/WSL build or change the runners' attention backend. Installation alone
does not establish kernel execution; backend and attention-mask compatibility
must be validated separately. Ordinary recipes do not require this extra.

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

## 4. Prepare model code and run

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
