# Qwen-Image 2.1 integration candidate

The independent [Qwen custom blocks](https://huggingface.co/elismasilva/qwen_image_2.1_custom_blocks)
implement text-to-image only, using the native 2.1 transformer, Qwen3-VL
encoder/processor and native Diffusers VAE. They do not call the conventional
pipeline for inference. Reading its checkpoint configuration preserves
`sample_sigmas`, especially for Turbo.

## Entry points

Modular runners default to trusted code from that Hub repository, not an import
from the ignored local clone. Use `--custom-blocks-revision <published SHA>` to
pin code. `--local-files-only` requires both model and code snapshots in cache.
For unpublished changes or offline checkout development, pass
`--custom-blocks modular_diffusers/custom_blocks/qwen_image_21` instead.
The code snapshot contains only Python modules and `modular_config.json`; it
does not download model weights. Reports record requested and resolved code
revisions separately. Remote use requires publishing the local code commit.

- `scripts/qwen_image_21/run_dynamic_minimal.py`: visible, phase-staged BF16
  didactic example with DDO linear routing; kept separate from full reporting.
- `scripts/qwen_image_21/run_dynamic_modular.py`: full phase-staged
  linear/compatibility runner with shared reporting and lifecycle controls.
- `scripts/qwen_image_21/run_dynamic_modular_vmm.py`: separate experimental
  native transformer runner; text encoder stays on explicit leaf compatibility.
- These modular runners accept `--variant base|turbo` and an optional `--model-path`.
  Turbo requires the checkpoint's saved schedule; `--steps` does not replace it.
- `scripts/qwen_image_21/run_diffusers_leaf.py`: conventional base-model
  control using the official streamed leaf offload, not a DDO allocator/runtime.
- SDNQ runners are a subsequent task. BF16 runners do not load SDNQ checkpoints.
- No model-specific modification has been made to DDO or its allocator.
- Decoder tiling is opt-in native Diffusers behavior. Inspect for seams.

## Environment and validation boundary

### Local untiled single-frame VAE candidate

Full modular linear/VMM runners accept `--vae-no-temporal-cache` to load
`QwenImage21ImageVAE` from the local custom blocks. It inherits the native VAE
and preserves its checkpoint keys, operators, normalization, unpatchification,
clamp and output contract. Only explicitly enabled single-frame untiled decode
omits the temporal feature caches. Tiling, other input shapes and the disabled
option delegate to the original implementation. Native slicing still works.
No installed Diffusers class is patched, and no DDO/allocator/LTX behavior is
changed. The conventional leaf control and minimal runner stay unchanged.

The temporal-cache option passed a selected 1024x1024/two-step CUDA pair;
both untiled PNGs are byte-identical. Decode wall time was 5.9766 versus
1.3127 s and sampled absolute driver peak was 7.9995 versus 7.6955 GiB.
It remains opt-in; this is not arbitrary-model/shape acceptance.
Tiny CPU contracts establish
exact decode parity with/without patchification and batch slicing, repeated
execution, no retained temporal tensors, unchanged checkpoint keys/loading,
native delegation and exception cleanup. They do not establish real-checkpoint
GPU peak/time or arbitrary-shape support. GPU validation must compare the same
untiled workload with and without this option, not against a tiled image.

`--vae-alias-residuals` is a separate option requiring
`--vae-no-temporal-cache`. Local residual up blocks reuse their immutable
input instead of cloning it, and omit that retained reference where no
shortcut exists. They share the original child modules and parameter objects;
no second set of weights is allocated. Cached, training and grad-enabled
forwards delegate to native behavior. Instrumented/unknown blocks are refused.
Normalizations keep their original FP32 computation; upsampling is unchanged.
Ten CPU VAE contracts passed, including FP32/BF16 exact block/full-decode parity,
unchanged inputs, repeated execution and parameter identity. GPU memory/time
benefits are limited to the selected CUDA evidence below. Keep this change separate from temporal-cache removal
in the eventual independent custom-block repository history.

Residual aliasing subsequently passed the selected two-step 1024x1024 CUDA
case: decode was 1.2658 s with 6.9060 GiB sampled absolute driver peak, versus
1.3127 s / 7.6955 GiB without aliasing. Untiled PNG hashes are identical.
The timing difference is small; do not claim a repeatable speedup from it.

The combined cacheless/residual options also passed the selected base BF16
1024x1024, 20-step, seed-42 untiled closure. PNG bytes match the original
20-step untiled reference exactly. Decode was 1.2704 s with a sampled absolute
driver GPU peak of 6.9356 GiB; no monitor/native failures were reported.
Use --vae-no-temporal-cache --vae-alias-residuals explicitly for this selected
route. Do not add --vae-direct-padding. This does not validate Turbo or other
shapes; the native/tiled paths remain available and defaults are unchanged.

`--vae-direct-padding` is a rejected CUDA candidate requiring
`--vae-no-temporal-cache`. Local image convolutions pass symmetric zero padding
to conv2d rather than materializing F.pad output. They share original parameter
objects and preserve dtype, stride, dilation and groups. The route is active
only inside the untiled single-frame inference decode, with scope reset on
failure; normalizations and upsampling are unchanged. Native cached, training
and grad-enabled forwards remain available. CPU convolution and full-decoder
contracts pass exact FP32/BF16 parity, including noncontiguous input and stride
cases. CUDA algorithms can still differ: reject the candidate if the selected
GPU image is not identical to the protected reference. Keep it opt-in and
separate in history; CPU parity is not CUDA quality or performance acceptance.

The 2026-10-11 two-step 1024x1024 CUDA gate completed, but failed exact image
preservation. Decode fell from 1.2658 s / 6.9060 GiB to 1.1789 s / 6.2085 GiB,
yet 140,735 pixels differed (maximum channel difference 13 on the 8-bit PNG).
This does not establish perceptual degradation, but violates the agreed gate.
Leave direct padding disabled; retain the accepted cacheless/residual options.

The accepted host dependency remains pinned to the older LTX fork. It does not
export the 2.1 components. During candidate development use the newer local
Diffusers source through a process-scoped `PYTHONPATH`; do not update the lockfile
or accepted LTX environment merely to try this model.

The custom-block directory is ignored by the host, like the LTX clones, but
currently has no independent remote. Back it up separately. Packaging/publication
and remote-code runner loading follow only after local validation and a chosen
custom-block repository.

On October 10, 2026, eight CPU contracts passed (5.34 s after formatting):
native noise layout and per-prompt expansion, missing Turbo schedule refusal,
saved-sigma preservation, four CFG/KV-cache combinations including a fresh
second generation, and native VAE decode parity. Ruff checks passed.

A separate local offline probe compared the complete modular workflow with
QwenImage21Pipeline, using tiny real Qwen3-VL, native transformer and VAE
components and a real locally constructed tokenizer/processor. Base used two
steps; the Turbo probe used a synthetic three-sigma fixture shared by both
pipelines (not the production Turbo schedule). Both image comparisons passed
at atol=rtol=1e-6. The fixtures contain random weights and do not establish
published-checkpoint quality or numerical parity.

Source under test: local Diffusers fork revision
`7d047b18092beebdb6d2aebadf1ccc254787dbc4`. The environment and lockfile were
not upgraded. Transformers emitted existing processor-docstring diagnostics
and the environment emitted optional TorchAO import warnings; neither was
treated as quantized-backend acceptance.

Run the small contract suite from the host root in PowerShell:

```powershell
$env:PYTHONPATH = 'F:\Projetos\diffusers\src'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_qwen_image_21_contracts.py
```

Use a separate terminal for generation: do not retain CUDA_VISIBLE_DEVICES=-1.
These checks are not native residency acceptance or performance evidence.
CUDA generation requires a separately approved capacity gate and workload.
Selected base BF16 512x512 and 1024x1024 runs, including native VAE tiling,
are recorded in [benchmark results](../experiments/dynamic_offload_results.md).
They are selected-workload evidence, not arbitrary Qwen/quantization support.

Host callback intervals are asynchronous wall observations, not isolated CUDA
step timings. The end-to-end denoise metric includes CPU latent readback.
RunTracker reports and DDO public snapshots keep the existing host schema.

## Full-runner lifecycle and reporting revision

### Current VAE preset policy

Both full modular runners now delegate VAE routing to DDO's public
`enable_offload(..., component="vae")`. No host-built convolution config or
direct host group-offload call is required. `auto`/`one_shot_fast`,
`low_ram_safe` and native `vmm_fast` select official leaf without a stream.
`warm_process`, `wsl_compat` and the protected NVMe preset preserve the
historical resident VAE. These policies do not alter transformer budgets.

Keep `--vae-leaf-offload` as an explicit compatibility override; it is no longer
required for the selected presets. `--vae-resident` reproduces resident decode.
The linear host's `--vae-linear-offload` selects the experimental DDO
`linear_convolution` route; the VMM host's existing `--vae-vmm-offload` remains
an explicit saved-source/capacity option. Overrides are mutually exclusive.
Every route now supplies a public DDO VAE report, including leaf and resident.

Native pool release remains explicit with `--release-native-cache-before-vae`:
detach/drop the transformer first and use a supporting DLL. Selecting leaf does
not itself trim native memory. The following development records retain their
historical conditions; their statements about resident defaults or pending
real-VAE validation are superseded by this policy and the benchmark evidence.
The routing migration is CPU-tested, not a newly executed CUDA benchmark.

### Experimental native VAE component ownership

The native VMM runner additionally accepts `--vae-vmm-offload`, mutually
exclusive with `--vae-leaf-offload`. Default resident and official leaf routes
are unchanged. Require native `vmm/auto`, an existing standalone BF16 directory
via `--vae-model-path`, and a finite positive
`--vae-vmm-resident-weight-budget-gb`. This VAE ceiling is independent of the
transformer's ceiling. No default VAE budget or automatic capacity growth.

The original selected VAE checkpoint is FP32; casting it during load removes
exact BF16 file-source compatibility. Prepare a separate source-compatible
checkpoint on CPU, preserving the original:

```powershell
.\.venv\Scripts\python.exe -B scripts/helpers/convert_safetensors_component_bf16.py `
  --source "<original-model-snapshot>\vae" `
  --output "F:\models\qwen-image-21-vae-bf16"
```

The converter supports a standalone unsharded dense component, refuses an
existing destination, preserves config and verifies every saved value against
the original cast to BF16. It records source/output hashes in `conversion.json`.
Non-floating tensors retain their dtype. It performs no inference or CUDA
initialization. The selected saved copy contains 238 BF16 parameters, with all
238 immutable descriptors bound. Its component geometry has 24 units and two
execution slots totaling 345,122,816 payload bytes before native page alignment.
This is CPU geometry, not native capacity or image acceptance.

The runner uses DDO's existing generic component ownership; it does not replace
convolution forward math or add a Qwen-specific allocator. Loading, native
attachment, report collection and cleanup are visible phases. Configure local
single-frame/residual options before attach. Public reports cover VAE source
preflight and runtime; cleanup retires the VAE before dropping references.

Only a tiny BF16 Conv2d/GroupNorm CUDA component has passed exact resident/native
comparison, repetition and decode cancellation. Real-VAE validation remains
pending. ModuleList execution units differ from official leaf geometry; VMM
is not assumed to reduce memory or improve decode time. Keep tiling and rejected
direct padding disabled for comparison with the accepted PNG. Do not change
the transformer or text encoder settings when evaluating this option.

Native VMM additionally offers --release-native-cache-before-vae as an
unvalidated explicit lifecycle candidate. After transformer detach/reference
cleanup and flush, DDO synchronizes once and asks the installed DLL to release
whole-idle pool backing. Live/partially used/quarantined backing is not freed;
event completion and checked unmap are required. The report includes native
pool snapshots before/after. Unsupported DLLs refuse the requested route.
This is GPU pool trimming, not a CPU standby purge and not a promise of zero
process VRAM. Use the separately built candidate only after its small CUDA gate;
keep the previously accepted DLL and default behavior unchanged.

The selected same-candidate 20-step pair preserved exact PNG bytes. Trim alone
gave 3.9272 GiB / 1.8943 s decode, but retained a live 8.125 MiB workspace.
Adding --clear-cublas-workspaces-before-vae eliminated that cuBLAS workspace
and zeroed the native pool before VAE load. This guarded private Torch operation
requires a quiescent single-visible-GPU compute thread without graph capture.
It does not change workspace size or inference math. Selected full-cleanup
decode was 4.7456 GiB / 2.1736 s: complete boundary release was not the lowest
observed decode peak. Neither option becomes automatic; normal driver/context
allocations remain even when native pool gauges are zero.

Full staged runners additionally expose --vae-leaf-offload, an optional
memory-saving candidate using official Diffusers leaf group offload for the VAE
only. Weights load on CPU; local cacheless/residual options are configured before
hooks. The onload device is CUDA, offload device CPU, and use_stream is false:
no extra warm-up decode or lazy stream prefetch is requested. Without this flag,
the VAE remains resident as before. Reports identify the executed VAE route.
Keep tiling and direct padding disabled for the exact-image comparison; leaf
may reduce weight residency but cannot remove decoder activation memory.
The selected base BF16 1024x1024, 20-step CUDA comparison completed with an
identical PNG: decode peak fell from 6.9356 to 5.0542 GiB, while decode time
rose from 1.2704 to 1.8115 s. Keep this a memory-saving opt-in, not a faster
default or arbitrary-shape support claim. No standby purge before VAE was
performed. DDO/DLL/model math is unchanged.

The full linear and VMM runners visibly orchestrate metadata, encoder load and
attach, encode, encoder cleanup, optional standby purge, transformer load and
attach, prepare, denoise, report collection, transformer cleanup, VAE load,
decode, VAE cleanup and image save. Only CPU conditioning survives encoder
cleanup. After preparation its original dictionary is dropped; prepared state
survives through denoise only. The CPU latent survives transformer cleanup,
then is discarded after native decode. Component pipelines, state, callbacks
and module references are released before `flush()` at each successful boundary.
No custom model forward, allocator policy or residency ceiling is introduced.

The conventional leaf control intentionally retains the full pipeline during
generation. Turning it into a staged runner would change the control recipe.
Its components are released after generation. The minimal stays a self-contained
didactic example rather than a wrapper around the full runner.

Full runners compose the existing shared `RunTracker` through
`scripts/helpers/runner_reporting.py`. Controls are:

- `--memory-monitor rss|full`: RSS is the default, cheaper observation;
  `full` samples USS and can impose monitoring overhead. Do not compare the
  two RAM measures as if they were the same quantity.
- `--metrics-level 1|2`: phase/workload observations, or additionally public
  DDO component reports (default 2). Full modular hosts now also report VAE
  leaf/resident through DDO; the conventional leaf control is not a DDO runtime.
- `--show-metrics`, `--show-profile`, `--show-denoise-steps` and
  `--save-metrics`: display/persistence controls; Boolean options have `--no-*`
  forms except `--show-profile`.
- `--ddo-gpu-timing`: optional CUDA boundary events, resolved once at completion,
  never a per-step synchronize. Event scopes include stream dependencies and
  are not isolated kernel timings. For the conventional control the measured
  phase includes encoding and decoding.
- `--purge-before-transformer`: opt-in Windows-only DDO standby helper.
  Requested and actual results are recorded; privilege failures must not be
  treated as success. Outside Windows it is a safe no-op. The conventional
  control invokes it once at the first transformer forward, not before encoding.

`metrics.json` uses reporting revision 2, records requested controls, executed
workload, phase observations, environment and actual Diffusers source path.
Failures/interruption preserve partial observations without hiding the original
exception. This does not promise recovery from a poisoned CUDA context.
The conventional control now also uses asynchronous host callbacks; its old
per-callback synchronized results remain historical, not silently relabeled.

On October 11, 2026, 26 CPU contracts passed in 5.57 s across Qwen blocks,
shared helpers and reporting. New checks cover success/error persistence,
interruption, monitoring failures and report-detail controls. Ruff passed.
The revised lifecycle still requires separately approved CUDA generation and
performance verification. Historical artifacts remain unchanged; a passing
CPU suite is not proof that all memory/performance opportunities are exhausted.
