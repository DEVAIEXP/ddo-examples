# Editable source and platform setup

Use Python 3.12, Git and uv. To change only runners, follow
[user setup](../user/setup.md) and edit the cloned examples; installed DDO packages
are sufficient. To develop DDO or its allocator as well, clone their sources
beside the examples checkout:

```sh
git clone --recurse-submodules https://github.com/DEVAIEXP/diffusers-dynamic-offloader.git
```

Run that command from the parent of `ddo-examples`. If DDO is already cloned,
initialize its pinned allocator with `git -C diffusers-dynamic-offloader submodule
update --init --recursive`. Do not clone another copy inside the examples.
DDO-only contributors can instead follow
[DDO source setup](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/installation.md)
without installing model runners.

## Separate platform environments

| Platform | Environment |
| --- | --- |
| Windows | `.venv` |
| Native Linux | `.venv-linux` |
| WSL | `.venv-wsl-gpu` |

Never replace another platform's environment. First install the runner
dependencies using [package setup](../user/setup.md), choosing the environment
above for your platform. Then explicitly replace only DDO and the allocator
with editable checkouts. Nothing in the main `pyproject.toml` selects local
repositories automatically.

### Native Linux

```bash
python="$PWD/.venv-linux/bin/python"
uv pip install --python "$python" --no-sources --no-deps --link-mode copy \
  -e ../diffusers-dynamic-offloader \
  -e ../diffusers-dynamic-offloader/ddo-vmm-allocator
uv pip install --python "$python" --no-sources --constraint constraints.txt --group dev
uv pip check --python "$python"
.venv-linux/bin/python -B scripts/setup_custom_blocks.py
```

For WSL substitute `.venv-wsl-gpu` in both commands.

### Windows PowerShell

```powershell
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
uv pip install --python $python --no-sources --no-deps --link-mode copy `
  -e ../diffusers-dynamic-offloader `
  -e ../diffusers-dynamic-offloader/ddo-vmm-allocator
uv pip install --python $python --no-sources --constraint constraints.txt --group dev
uv pip check --python $python
.venv/Scripts/python.exe -B scripts/setup_custom_blocks.py
```

Use explicit interpreter paths. Optional extras are `fp8`, `kernels` and `sdnq`;
install them through the package setup before switching to editable sources.
`--no-deps` above changes only the two source packages, preserving the model
dependencies. Editable installation is for development, not release validation.

## Dependencies and native builds

The dependency constraints retain official Diffusers 0.41.0 and Torch 2.12.1+cu132.
Do not inject an editable Diffusers fork or historical host via PYTHONPATH.
Linux/WSL TorchAudio 2.11.0+cpu is for audio decoding, not CPU model inference.
TorchCodec/PyAV need compatible system FFmpeg shared libraries: Ubuntu 22.04 uses
`ffmpeg`; FFmpeg package/library names vary by distribution.

An editable allocator install supplies Python code, not a compiled DLL/.so.
Use a compatible reviewed native artifact or DDO's
[maintainer release/build procedure](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/releases.md).
Native builds need the platform toolchain and compatible CUDA toolkit.
There is no import-time/inference-time build or download. Discovery does not scan
arbitrary experimental builds. WSL keeps the Windows-provided GPU driver.

## Independent model code

Clones under ignored `modular_diffusers/custom_blocks/` are independent Git
repositories, not submodules or vendored source. Review executable code and pin
revisions. The helper refuses mismatched existing checkouts. Read each clone's
instructions before editing and commit its changes in that repository.
Weights are obtained separately under their licenses.

## Validation boundaries

Follow the [validation contracts](validation.md) before changing execution.
Storage, registration and GPU ceilings are separate. Use the model guide's
validated capacities for the selected platform; successful setup does not prove
that a workload fits. WSL limits are described in the validation guide.

## Dependency constraints

`constraints.txt` retains third-party versions from the previous development
lock without binding DDO or the allocator to sibling paths. It is a version
constraint file, not a hash-verified lock. User installation uses package indexes
and `uv pip`; source installation explicitly replaces only DDO and the allocator
with editable checkouts. A source-based lock is not a user installation input.
