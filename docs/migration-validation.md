# Host migration validation

Date: 2026-10-10. Platform: Windows. Environment: root uv-managed .venv,
Python 3.12.11, Torch 2.12.1+cu132, torchvision 0.27.1+cu132,
transformers 5.15.1, TorchAO 0.18.0, triton-windows 3.7.1.post27,
cuda-core 1.2.0 and cuda-bindings 13.4.1.

## Selected source revisions

- Diffusers: e71ebc5858961e2a37d6a7b161d00ae0063c3ab7 (Git dependency).
- Image custom blocks: a6e6f865dc6f8decd7ff681e91969bf4b07e9f21.
  Compared with the original image host's 5e3ff89 revision, only README changed.
- Video custom blocks: f5c712e93c52283cb101da26d4a4544c808666bf.
- DDO and allocator: editable sibling dependencies, resolved in uv.lock.

The independent clones are ignored, not submodules. No historical host is a
dependency. Original runners were copied rather than moved or deleted.

## CPU acceptance

With CUDA_VISIBLE_DEVICES=-1, PYTHONDONTWRITEBYTECODE=1,
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and HF_MODULES_CACHE=.hf_modules_cpu:

    .venv/Scripts/python.exe -B -m pytest tests/test_migration_cpu.py -q -p no:cacheprovider

Result: 11 passed, 5 subtests passed, 1 third-party TorchAO SyntaxWarning,
86.84 seconds. Contracts cover manifest/layout, clone separation, all five runner
help commands, video defaults/dry plan, explicit audio, FP8 opt-in boundaries,
same-namespace trusted code loading, no-weight pipeline construction and
absence of historical host imports.

    .venv/Scripts/python.exe -B tests/probe_hub_code_cpu.py
    .venv/Scripts/python.exe -B tests/probe_hub_code_cpu.py --local-files-only

Both real Hub code-only probes passed at the selected video revision:
weights_loaded=false and cuda_initialized=false. Each checked transformer/helper
class identity and construction without checkpoint components.
SSLKEYLOGFILE was unset only in the diagnostic process; TLS verification was
not disabled. Existing warning messages about TorchAO tensor imports and guider
APIs are not proof of an FP8 checkpoint failure or acceptance.

The direct remote loader initially failed because this Diffusers version treats
the dot in the Hub repository name as a Python package separator. The host now
resolves an immutable code-only snapshot with huggingface_hub, then calls
Diffusers' standard trust_remote_code local loader. Offline snapshot reuse
passed. No Diffusers source or inference forward was modified for this fix.

## Preserved boundaries and next gate

Image runner math/settings and video helpers are preserved. Changes are limited
to host paths, installed allocator imports, help-mode bootstrap suppression,
code-source resolution and explicit conditioning audio instead of an old local
asset. The new video default has no audio conditioning until requested.

No model weights, media, CUDA inference, timing benchmark, native rebuild or
DLL installation was performed. Do not claim output/performance parity from
these CPU checks. Automatic DLL discovery is not verification that the discovered
binary matches an accepted workload checkpoint.

Before generation, present one consolidated command using the previously accepted
binary, capacities, model/input files and optimization settings for approval.
Stop on the first ownership, memory or output anomaly. Native Linux remains a
separate environment and platform gate; consult DDO's Linux handoff.
