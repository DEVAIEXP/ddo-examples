# Linux and WSL validation handoff (2026-10-11)

This is the primary host continuation record after the selected Windows VAE
gates. It carries state without requiring this conversation. It authorizes
neither CUDA execution nor publication. Read host/DDO/allocator AGENTS.md first,
then DDO docs/linux-handoff.md and this document. Use actual Linux mount paths,
not Windows drive syntax.

## Native Linux checkpoint: selected sequence closed (2026-10-11)

The primary host `.venv-linux` was created and its media prerequisites resolved;
see linux-host-environment.md for actual editable sources and required overrides.
The selected eight-file CPU gate passed 52 tests. An isolated idle-trim `.so`
was compiled without overwriting the accepted library, and its approved small
CUDA ownership probe passed. Both approved image and five-second video
resident/leaf pairs completed, with identical delivered hashes within each pair
and no reported native/monitor failures. The selected results and limitations
are in benchmark-matrix.md and the dated evidence log.

Do not rerun the completed Linux preparation/CUDA sequence automatically. The
checklists below preserve the reviewed recipe, not a request to repeat it.
Remaining closure: reproducible Linux source/audio overrides, review and local
commit checkpoint with independent custom-block preservation, and the separately
approved WSL assessment. No complete version freeze, publication, general-model,
FP8/SDNQ Linux or WSL native support is implied by the selected BF16 gates.

The separately approved Qwen Image 2.1 base BF16 512-square trio subsequently
passed: conventional streamed leaf, linear auto and native VMM fast, 20 steps,
seed 42 and stock resident VAE. All three delivered PNG hashes match exactly.
Current results and lifecycle/memory scope limits live in
`docs/benchmarks/qwen-image-21.md`; full phase tables are in the evidence log.
Do not rerun or expand that closed trio automatically. Next work is the reviewed
matched Windows gaps, requested Qwen SDNQ Windows assessment, independent WSL
gate and reproducible environment/local-commit closure. This is not 1024 Linux,
quantized backend or arbitrary-model acceptance.

## Selected Windows checkpoint

### Subsequent official-package and INT4 closure

Reviewed source changes are now committed locally; see
[the checkpoint manifest](local-checkpoint.md). Preserve independent clones and
ignored artifacts separately. Earlier uncommitted-state wording below is
historical; inspect current status rather than repeating the commit checklist.

Windows now uses official PyPI Diffusers 0.41.0. The selected BF16 image/video
and Qwen 512-square gates passed, as did the separate SDNQ INT4 image DDO/leaf
pairs for Qwen and LTX image. See [package transition](official-diffusers.md)
and the [evidence log](../experiments/dynamic_offload_results.md).
The quantized pairs use normal allocation and preserve SDNQ forward math;
they imply no native SDNQ, Linux/WSL or video acceptance on this package.
Do not repeat accepted Windows cases or prior Linux-fork gates automatically.

For a Linux package transition, first preserve its freeze/import paths and
audio workaround, review the official-package installation separately, and run
CPU import/loader contracts with CUDA hidden. Then propose a finite generation
gate with actual paths/limits for confirmation. Quantized/platform coverage is
optional separate scope, not an automatic Cartesian model/preset matrix.

The LTX image/video resident versus no-stream leaf gates completed with identical
saved media to their controls and historical accepted outputs. Resident defaults
remain unchanged. Leaf is explicit: less decode GPU usage, more CPU RAM; video
decode is slower. See [summary](benchmark-summary.md), [VAE gate](ltx-vae-validation.md)
and the complete dated [phase evidence](../experiments/dynamic_offload_results.md).

- Image: BF16, real prompt, 1280x704, seed 43, eight steps, native text encoder,
  no PAG; transformer ceiling 4 GiB, total staging 1.5 GiB.
- Video: BF16, 1280x704, five seconds, 24 fps (121 frames), seed 43, eight steps,
  existing guitar image/prompt, no conditioning audio, generated output audio.
  Encoder ceiling 4 GiB; transformer/staging ceilings 1.5 GiB each; source and
  registration limits 24.625 GiB. Preserve the accepted 1024 chunk settings,
  phase modulation, compact timesteps, in-place residuals and streaming decode.
- Selected LTX DLL: build-joined-spans-cu128-20261006/Release/ddo_vmm.dll,
  SHA256 EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.
- Qwen explicit idle trim used a different candidate:
  build-idle-trim-cu128-20261011/Release/ddo_vmm.dll,
  SHA256 468A854EB550C6299C3DE785B65ECD4F4DF382C488B6F1C83AA6899983F12CFE.

Do not promote or overwrite either DLL. The handoff includes uncommitted DDO,
allocator and host changes, plus ignored independent custom-block clones.
Before transfer or version freeze, inventory each checkout and preserve those
clones separately. A clone's presence on Windows does not mean its new code is
on Hugging Face or available in another checkout. Do not assume remote main
includes these changes; no push is authorized.

## What changed and what needs a new .so

Python-only changes include component VAE preset routing, explicit linear dense
convolution support, host decode/reporting and the LTX streaming execution-device
fix. They do not themselves require a C++ rebuild. Linear convolution is an
eager opt-in using the normal PyTorch allocator; it must not be enabled in a
native VMM process.

Earlier idle-trim work changed these allocator files:
native/include/ddo_vmm.h, native/include/ddo_vmm_torch_pool.h,
native/src/ddo_vmm_torch_allocator.cpp and native/src/ddo_vmm_torch_pool.cpp,
plus the Python bindings. Therefore the current Linux binary must be inspected;
do not assume it exports ddo_vmm_torch_trim_idle_v1. Missing trim is not silently
equivalent to successful cache release. The LTX leaf gate uses no trim and can
be assessed independently, but a release including that native feature needs
its own Linux build and small ownership gate.

Preserve the accepted Linux library and its hash first. Old acceptance belongs
to its actual binary/workload, not the latest tree. The allocator build helper
currently fixes build-linux/build-linux-cpu and has no output-directory option:
inspect its library --dry-run, then agree on a separate candidate build path
before compilation. Do not overwrite build-linux/libddo_vmm.so blindly.
No kernel overcommit changes, automatic budget growth, driver replacement or
forced release of live allocations is allowed.

## First session: inspect and prepare, without CUDA execution

Record native Linux versus WSL identity, OS/kernel, mount/filesystem, driver,
compiler/CMake/toolkit, Python/Torch/CUDA versions, available RAM, commit limits
and cgroup constraints. Inspect NVIDIA driver information; do not start a CUDA
probe or call torch.cuda.is_available as an incidental environment check.
Check shared mount execution permissions and whether caches/builds need a
Linux-local filesystem. Do not replace an existing environment blindly.

From each repository, record git status, revisions and source diffs, including
the independent custom-block repositories. Resolve installed import paths:
DDO and allocator must be the current local checkout. Diffusers must be
identified by its actual import location and source. Windows
now uses official PyPI `diffusers==0.41.0`, not the merged fork or e71ebc5.
Previous Linux acceptance remains tied to its local fork and audio overrides.
Review the resolved environment and lock before changing either platform.

Use a separate host .venv-linux for native Linux and .venv-wsl for WSL, created
with uv/Python 3.12 only if absent. Never sync into the Windows .venv. Use
UV_PROJECT_ENVIRONMENT explicitly for uv operations; review --dry-run first.
Install local editable DDO/allocator and the reviewed official Diffusers package in that
platform environment, preserving dependency versions and recording uv pip freeze.
Do not automatically install quantization or Triton extras for the BF16 gate.
The host's current pins include torchaudio 2.11.0 with Torch 2.12.1 and an older
Diffusers Git revision: inspect the actual accepted Windows inventory and choose
a coherent platform setup rather than silently assuming a fresh locked install
matches the tested environment. Dependency changes are a separate reviewed step.

Review DDO docs/linux-environment.md as historical setup guidance, not instructions
to reuse its older host or overwrite today's environment. Use portable environment
paths and the actual model/media cache paths. Windows standby purge is a safe
no-op outside Windows and must not become a Linux/WSL requirement.

## CPU gate

After the actual mount/environment paths are known, let DDO_ROOT and EXAMPLES_ROOT
refer to those two checkouts. Select the appropriate platform interpreter:

```bash
# Native Linux; choose .venv-wsl instead for the independent WSL CPU gate.
PYTHON="$EXAMPLES_ROOT/.venv-linux/bin/python"
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" -B -m pytest -q -p no:cacheprovider \
  "$DDO_ROOT/tests/test_vae_component_policy_cpu.py" \
  "$DDO_ROOT/tests/test_linear_convolution_cpu.py" \
  "$DDO_ROOT/tests/test_device_cache.py" \
  "$DDO_ROOT/ddo-vmm-allocator/tests/test_native_trim_cpu.py" \
  "$EXAMPLES_ROOT/tests/test_ltx_vae_decode_cpu.py" \
  "$EXAMPLES_ROOT/tests/test_qwen_vae_leaf_runner_cpu.py" \
  "$EXAMPLES_ROOT/tests/test_qwen_vae_linear_runner_cpu.py" \
  "$EXAMPLES_ROOT/tests/test_video_decode_routes_cpu.py"
```

These tests initialize no CUDA or model generation. Run from the host root with
the reviewed editable imports. Do not carry CUDA_VISIBLE_DEVICES=-1 into model
commands. If compilation is needed, run the native CPU recipe and actual candidate
Linux build before proposing CUDA; compilation alone is not acceptance.

## Finite native Linux CUDA proposal

Present exact commands, library hash, hypothesis and criteria; await confirmation.
Do not automatically repeat the old seven-contract suite or all historical models.

1. If the native pool changed/rebuilt, propose the existing tiny allocator/idle
   trim ownership probe, with fixed limits and a fresh process. Require exact
   payload, no quarantine/failure, safe pending retirement and allocation after
   trim. A missing export or failed gate stops native model testing.
2. Propose the image resident/leaf pair first, at the Windows workload above,
   fresh bootstrap and fixed ceilings. Stop on the first failure or anomaly.
3. After acceptance, propose the video resident/leaf pair at five seconds only.
   No 8/10-second expansion, FP8, SDNQ, convolution VAE or heavier-model matrix
   is implied.

The revised streaming decoder is in the local video clone: pass its actual local
path through --custom-blocks, rather than loading the stale pinned Hub revision.
Keep audio VAE and vocoder routes unchanged. GPU ownership cleanup precedes VAE
loading. Do not automatically enable idle trim or clear cuBLAS in this LTX pair.

Within a platform pair, require completed phases, no CUDA/ownership error,
identical delivered image/video output where deterministic, and finite unchanged
audio. If hashes differ, compare decoded content and report differences instead
of declaring parity. Do not require byte identity across different platforms or
claim a cause from one timing observation. Use DDO public reports; missing monitor
fields/errors cannot count as successful zero-memory measurements.

## Independent WSL proposal

WSL is not native Linux and is not promised native VMM support. Start with its
separate CPU/environment check and record the host driver plus WSL kernel/CUDA
interface. Do not install a Linux NVIDIA kernel driver inside WSL. Treat source
registration, pinning, filesystem and admission observations as independent.

Before a native WSL model run, require an explicitly approved small native
capability/ownership gate with the WSL-compatible library and actual fixed
bounds. An unavailable capability is recorded as unsupported/blocked, not fixed
by raising ceilings or copying a Windows DLL. A native Linux .so does not inherit
acceptance in WSL merely because it loads.

If native WSL fails, stop that route. A separate user-approved linear/wsl_compat
control can establish a usable fallback; label its executed route clearly and
do not compare it as native VMM parity. Agree on the two VAE routes actually
supported before proposing image/video cases. No automatic fallback benchmark
or attempt to emulate WDDM is authorized.

## Benchmark table and version cut

For each accepted workload/platform pair record tracked total, denoise wall time,
executed step count, explicitly labeled phase-average s/step and CUDA warm timing
only when available. Record every phase's elapsed time, sampled absolute GPU peak
and RSS/USS identity, setup/source/cache settings, requested and executed routes,
environment/binary/code identity, completion, output parity and actual purge state.
Do not sum aggregate decode events with their constituent phases.

Keep workload-matched summary/per-phase tables and VAE-only hardware advice.
Resident when decode fits; leaf when GPU weight residency is the limit and RAM
allows it; supported spatial tiling/temporal streaming when activation capacity
still prevents decode. Tiling cannot guarantee capacity for weights or assembled
CPU outputs. Use N/A for unavailable historical fields and unsupported platforms;
a complete table is honest coverage, not an obligation to rerun every old case.

After the finite gates, record acceptance/limitations and inspect the final diffs.
Commit reviewed changes locally only when the user confirms that step: allocator
first, DDO submodule pin second, independent custom-block repos and host separately.
Preserve ignored local evidence/clones; no push, history rewrite or binary promotion.
Only then label a version frozen and start the heavier video model.

## Resume prompt

For the current Windows continuation, use [windows-handoff.md](windows-handoff.md).
The prompt below preserves the earlier platform proposal; do not interpret it
as a request to repeat the subsequently completed Linux sequence.

> Read AGENTS.md in the examples host, DDO and allocator. Then read DDO
> docs/linux-handoff.md and host docs/platform-validation-handoff.md. Continue the
> finite platform closure, not Windows performance tuning. Inspect mount paths,
> local dirty state, preserved Linux binary, exports and actual editable imports;
> create/reuse a separate platform uv environment. Run CPU contracts first.
> Before compilation that would overwrite an accepted artifact, agree on a
> separate candidate path. Before any CUDA gate or benchmark, show exact command,
> fixed bounds, hypothesis and acceptance criteria and await my confirmation.
> Native Linux and WSL are independent. Stop at the first failure, preserve
> evidence, and do not expand the model/parameter matrix or publish anything.
