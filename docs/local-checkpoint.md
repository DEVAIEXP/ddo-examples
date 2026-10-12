# Local source checkpoint — 2026-10-11

The user approved local commits after the selected Windows BF16 and INT4
image gates. These are source checkpoints, not remote publication or a general
release/platform freeze. The documentation commit containing this manifest
follows the host source revisions below; inspect actual HEAD/status before
continuing. No binary was rebuilt, replaced or promoted during this closure.

| Repository | Reviewed source revision | Responsibility |
| --- | --- | --- |
| DDO allocator | `bf3280600f0a8695547b1847459989e75b8eac9a` | Explicit ownership-safe idle trim, optional ABI and probes |
| DDO | `83694f8f022a307f0fa46113ef7c6c182dec0ec7` | Component VAE policies, opt-in linear convolution, public cache release, allocator pin and official development dependency |
| Examples core | `e601e362629514558700f6c44d3f09336bc79f68` | Staged Qwen runners, reporting, official Diffusers and explicit LTX image loader |
| Examples SDNQ | `33af1d41ec5e171d08679557daea913c8ed7d6ff` | Separate Qwen INT4 runner and matched image leaf controls |
| LTX video custom blocks | `9b788c02d15cfcc43d50548d75cffeb84ca5bc24` | Explicit streaming execution device and official guidance CPU compatibility |
| LTX image custom blocks | `1eba529b4a838a8590fbc362ad2249dd8f8b01ea` | Existing accepted image implementation; no new change in this closure |
| Qwen custom blocks | `e15661930050bf79c14a7f2c655bd65f65d13a40` | Independent base/Turbo Hub-compatible code checkout |

The independent custom-block clones are ignored by the host, not submodules.
Their local commits must be preserved separately. No push was performed;
remote code may not include these revisions. Continue with the local paths
used in the accepted gates until publication is explicitly authorized.
Weights, outputs, caches, `.internal/` evidence and the user's DDO
`processo.txt` were not staged. Preserve ignored artifacts separately from Git.

## Verification at commit closure

A consolidated CPU-only sequence passed 142 tests with
`CUDA_VISIBLE_DEVICES=-1`: allocator optional trim ABI, DDO cache release,
linear convolution and VAE policy, native component adapters, host loader,
conversion/reporting, Qwen contracts/VAE/SDNQ controls, LTX image SDNQ/VAE
controls and video guidance compatibility. Staged diffs passed whitespace
checks with Windows CRLF treated as line endings. CPU checks do not replace
the prior build/CUDA evidence associated with each preserved binary.

Latest user-executed Windows image comparisons passed on Diffusers 0.41.0:
Qwen INT4 DDO/leaf denoise 18.8056/24.2629 s and LTX image INT4 DDO/leaf
11.7399/21.5413 s, with byte-identical PNGs within each pair. Exact scopes,
phase memory, hashes and separate BF16 references are in
[detailed results](../experiments/dynamic_offload_results.md).

## Next platform boundary

Read [platform handoff](platform-validation-handoff.md),
[official Diffusers transition](official-diffusers.md) and the sibling DDO
`docs/linux-handoff.md`. Existing Linux acceptance uses a different, preserved
fork/audio environment. Inspect its direct interpreter, import paths, freeze,
binaries and local source revisions before updating it; use separate platform
environments and preserve its TorchAudio/FFmpeg workaround.

Official-package generation, INT4 and WSL remain unvalidated there. Start with
reviewed CPU/import checks; show exact commands, bounds, hypothesis and criteria
before a finite CUDA proposal. Do not repeat closed Linux-fork/Windows matrices,
overwrite accepted `.so` files or promote a candidate automatically. SDNQ image
closure itself changed no C++ code and requires no allocator rebuild.
