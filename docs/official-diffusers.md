# Official Diffusers package transition

The current Windows example host uses the official PyPI wheel
`diffusers==0.41.0`, rather than a Git dependency, editable fork or PYTHONPATH
override. Model-specific custom-block code stays in its independent repositories.
The local fork remains a source-reference checkout only.

## Scope and evidence (October 11, 2026)

- First validate the published wheel in isolation with the existing host interpreter.
- 95 host CPU tests and 10 subtests passed against the isolated official wheel.
  The same suite passed again against the installed Windows host package
  (95 tests and 10 subtests), without the fork on PYTHONPATH.
- Five LTX image/video custom-block compatibility tests passed. The video test
  now obtains the guidance class from either API layout and mocks the legacy
  layout instead of requiring the removed module to be installed.
- 34 selected DDO component-policy, linear-convolution and device-cache CPU
  contracts passed against the installed official wheel.
- The host manifest and lockfile select the PyPI registry package, not Git.
- Only Diffusers was replaced in Windows `.venv`; Torch remains
  `2.12.1+cu132`. The agent did not execute CUDA; the user subsequently ran
  the approved image, video and Qwen BF16 gates, all completed successfully.
- DDO's optional Windows/Linux development profiles use the same published
  version; its generic runtime dependency remains unpinned.
- Existing warnings about TorchAO imports are not FP8 validation. Saved FP8 and
  SDNQ generation need their own approved checks if affected.

Verify actual imports without setting PYTHONPATH to the fork:

```powershell
.\.venv\Scripts\python.exe -B -c "import diffusers; print(diffusers.__version__, diffusers.__file__)"
```

CPU compatibility is not image-quality, CUDA ownership or performance acceptance.
Preserve historical artifacts with their original Diffusers source and revision;
do not relabel previous benchmarks as package-release results. Before approving
other generation cases, propose the finite workload and wait for approval.
Do not change budgets, allocator binaries or compute
options as part of the dependency comparison.

## Image checkpoint loader correction

The first user-run Windows image generation stopped before transformer loading:
DDO's default AutoModel loader tried to resolve the checkpoint class
`LTX2ImageTransformer2DModel` in installed Diffusers. That class belongs to
the independent image custom blocks and is not exported by the official package.
Earlier import/graph CPU checks did not exercise this checkpoint dispatch.

The minimal, linear and VMM image runners now pass the custom transformer class
explicitly through DDO's public `model_loader` argument. DDO/allocator loading
and transport are unchanged; no global Diffusers class registration is added.
Ten selected CPU checks passed, including a real tiny safetensors checkpoint
round-trip while AutoModel and CUDA initialization are forbidden. The weights
loaded exactly. The successful user repeat produced the same PNG hash as the
previous Windows resident-VAE gate.

## Selected Windows generation acceptance

| Case | Denoise wall s | Tracked total s | Output |
| --- | ---: | ---: | --- |
| LTX image BF16, 1280x704 / 8 steps | 13.7679 | 35.2343 | Byte-identical PNG |
| LTX video BF16, 1280x704 / 5 s / 8 steps | 197.3890 | 259.0260 | Byte-identical MP4 |
| Qwen base BF16, 512x512 / 20 steps | 17.6734 | 36.639479 | Byte-identical PNG |

Windows standby purge executed before the transformer in all cases. Timings
include purge and are staged totals, not process startup wall time. Qwen's
older Windows control used a different monitor, so no matched speedup is
claimed. LTX single-run timing differences do not prove a package-release
speedup. See [raw evidence summary](../experiments/dynamic_offload_results.md)
for paths, hashes, memory scopes, binary identities and fixed controls.
Only these Windows BF16 cases are closed; quantized/platform tests remain
outside this gate.

## Other environments

A separate selected Windows INT4 image gate subsequently passed for Qwen 2.1
and LTX 2.3 image using DDO packed SDNQ and official streamed leaf. This does
not extend the BF16 native gate or validate FP8, quantized video, INT8 or
other platforms. See [detailed results](../experiments/dynamic_offload_results.md)
and [Qwen SDNQ integration](qwen-sdnq.md).

Linux's previous local-fork override is historical evidence, not a validation
of this wheel. Its separate audio-wheel workaround must not be overwritten by
a blind sync. Linux/WSL package generation checks remain separate.

The historical `ltx2.3-image-diffusers` host is not migrated: conventional apps
and runners still import fork-only `LTX2ImagePipeline` and
`LTX2ImageImg2ImgPipeline`. Modular custom blocks in the primary host passed
the selected CPU contracts, but that does not make the old conventional scripts
compatible. Preserve the old environment until those entry points are explicitly
migrated or separated from its modular environment.
