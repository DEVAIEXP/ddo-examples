# Windows continuation after selected Linux validation

Updated 2026-10-11. This is the current continuation handoff for the primary
examples host. The approved Linux sequence is complete. Do not automatically
repeat tests, expand the matrix, commit, publish or promote binaries.

## Start here

### Subsequent Windows closure

Reviewed changes are now committed locally; see the
[source checkpoint manifest](local-checkpoint.md). Earlier dirty-state and
commit-pending statements below are historical. No remote publication occurred.

The primary host now uses official PyPI Diffusers 0.41.0. Selected BF16 LTX
image/video and Qwen 512-square package gates passed, followed by selected
SDNQ INT4 image DDO/streamed-leaf comparisons for Qwen and LTX image. PNGs
match within each quantized pair; these are normal-allocation SDNQ routes,
not native VMM. See [package transition](official-diffusers.md),
[Qwen SDNQ](qwen-sdnq.md) and the current evidence log. Historical pending
items below must not trigger repetition of those closed cases.

Linux's accepted fork/audio environment remains separate; official-package,
INT4 and WSL acceptance have not been established there. Preserve Linux
overrides and binaries until an explicitly reviewed transition. No source or
DLL rebuild was required for these SDNQ image comparisons. Local closure is
not remote publication or a promise that all platform/recipe tests have passed.

Read host, DDO and allocator AGENTS.md, then this document and
[benchmarks by model](benchmarks/README.md). The older DDO Windows handoff
preserves October 4 evidence; its old host/environment instructions are not the
current execution root. Use `E:\ProjetosIA\ddo-examples` on Windows.

| Checkout | Linux location | Windows location |
| --- | --- | --- |
| Examples | `/run/media/eliseu/AI2/ProjetosIA/ddo-examples` | `E:\ProjetosIA\ddo-examples` |
| DDO | `/run/media/eliseu/AI2/ProjetosIA/diffusers-dynamic-offloader` | `E:\ProjetosIA\diffusers-dynamic-offloader` |
| Allocator | DDO's `ddo-vmm-allocator` submodule | Same submodule inside DDO |
| Merged local Diffusers | `/run/media/eliseu/IA/Projetos/diffusers` | `F:\Projetos\diffusers` |

Independent custom-block clones remain ignored under
`modular_diffusers/custom_blocks/`. Historical LTX hosts are optional references,
not required execution roots. Preserve local edits, ignored clones, outputs and
logs; a remote clone or Git bundle does not necessarily contain them.

## What this Linux session completed

- Created host `.venv-linux` with uv 0.12.21 and Python 3.12.14, preserving
  Windows `.venv` and the older DDO Linux environment.
- Installed editable sibling DDO/allocator and the merged local Diffusers fork,
  revision `7d047b180`, package version `0.41.0.dev0`.
- Resolved the audio import mismatch with TorchAudio `2.11.0+cpu`; Torch remains
  CUDA-enabled `2.12.1+cu132`, torchvision `0.27.1+cu132`. This is not a model
  or allocator CPU fallback. The user installed system `libavdevice62` for
  TorchCodec/FFmpeg 8. Imports and package consistency passed (81 packages).
- Passed 52 selected CPU contracts in 6.31 s, then 27 Qwen/block/VAE CPU
  contracts in 4.05 s. CPU acceptance is not CUDA acceptance.
- Compiled an isolated Linux idle-trim candidate with GCC 15.2.0, CMake 4.2.3
  and CUDA toolkit 13.2.78 from DDO's older `.venv-linux`. No C++ source,
  Windows DLL, driver, kernel policy or admission ceiling was changed here.
- Passed the approved small native CUDA idle-trim probe: active allocation
  preserved, idle/pending mappings released, repeat trim and subsequent
  allocation passed; final live/mapped/pending/quarantined bytes were zero.
- Completed seven approved real generations: two LTX image VAE routes, two
  five-second LTX video VAE routes and three Qwen 512-square routes. No new
  model math, benchmark budget or compute optimization was introduced here.

Environment details and overrides: [Linux host environment](linux-host-environment.md).
Native build/probe evidence: sibling DDO `docs/native-linux-idle-trim-gate.md`.

## Selected results, not a universal ranking

| Linux case | Denoise wall (s) | Tracked total (s) | Global absolute GPU peak (GiB) | Global RSS peak (GiB) |
| --- | ---: | ---: | ---: | ---: |
| LTX image, resident VAE | 17.1203 | 40.2959 | 6.559631 | 27.3297 |
| LTX image, leaf VAE | 17.0253 | 37.6539 | 6.534973 | 27.3595 |
| LTX video, resident VAE | 209.7483 | 266.5132 | 4.871399 | 28.4240 |
| LTX video, leaf VAE | 209.5043 | 272.2485 | 4.787964 | 28.4177 |
| Qwen official streamed leaf, full pipeline | N/A | 45.3132 | 4.800171 | 32.598476 |
| Qwen DDO linear auto, staged | 18.1429 | 34.7060 | 5.580688 | 18.897018 |
| Qwen DDO VMM fast, staged | 15.6808 | 23.2622 | 5.719055 | 18.867943 |

LTX image: BF16, 1280x704, seed 43, 8 steps, real prompt, native encoder and
transformer. CUDA warm mean (steps 2–8) was 1.056121/1.056260 s/step.
LTX video: BF16, 1280x704, five seconds, 24 fps/121 frames, seed 43, 8 steps,
image conditioning, generated audio but no conditioning audio. Temporal GPU
streaming is not proof of active spatial tiling. Qwen: base BF16, 512x512,
seed 42, 20 steps, CFG 1, native KV cache, stock resident untiled VAE.
Qwen DDO encoding used the compatible leaf route, not a native encoder.

The Qwen control's generate phase includes encoding, denoise and decode;
separate denoise is unavailable, not zero. No CUDA warm timing was requested
for video/Qwen. Totals exclude imports/bootstrap. Sampled absolute GPU/RSS
peaks are not enforced instantaneous ceilings. Setup/lifecycle differences
prevent allocator-only attribution or a universal Windows/Linux speed ratio.

LTX leaf decode reduced phase GPU peak by 0.9370 GiB for image and 2.2821 GiB
for video, with higher decode time/RSS. Resident video remains the faster
selected route when it fits. Preserve explicit routes and existing defaults.

Within each LTX pair, delivered outputs were byte-identical; all three Qwen
PNGs were byte-identical. Video decoded to 121 frames with finite stereo audio.
No native ownership/fallback/quarantine/admission or monitor error occurred in
the accepted runs. This does not establish arbitrary model/recipe support.

Artifacts below are local/ignored; preserve them rather than overwriting:

- `outputs/ltx_image_vae_{resident,leaf}_linux_gate/`
- `outputs/ltx_video_vae_{resident,leaf}_linux_gate/`
- `outputs/qwen21_{leaf,linear,vmm}_linux_512/`
- `.internal/logs/` and `.internal/inputs/man_guitar.png`

Full per-phase tables, snapshots, settings and output hashes are in
[the evidence log](../experiments/dynamic_offload_results.md). Reader-facing
pages are [image](benchmarks/ltx23-image.md), [video](benchmarks/ltx23-video.md)
and [Qwen](benchmarks/qwen-image-21.md). Missing historical fields stay N/A.

## Binary identity and preservation

Paths below are relative to DDO's allocator checkout:

| Linux artifact | SHA256 |
| --- | --- |
| Preserved `build-linux/libddo_vmm.so` | `a87979f8773a010ddc2c781a37f3ef55112ca558a9d93083b972f51246a2b245` |
| Tested `build-linux-idle-trim-20261011/libddo_vmm.so` | `db63138da2fdcca328a244ff885ca44ecd8b33683825b2ab0f6f39a95b3cba79` |

The candidate was explicitly selected, not promoted over the preserved binary.
No DLL was rebuilt on Linux. On Windows verify the intended existing DLL and
clear any inherited Linux `DDO_VMM_LIBRARY_PATH`; do not use a .so or infer
acceptance from the default filename. Selected Windows records identify the
LTX joined-spans CUDA 12.8 DLL by SHA256
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`
and the Qwen idle-trim DLL by
`468A854EB550C6299C3DE785B65ECD4F4DF382C488B6F1C83AA6899983F12CFE`.
Confirm paths against the original case artifacts before execution.

## Remaining work, in order

1. **Inspect and preserve the current checkpoint.** No commits or push were
   made in this Linux session. Working trees contain earlier Windows code
   changes plus new documentation; shared NTFS CRLF/LF noise is not necessarily
   a logical change. Inspect scoped diffs with `--ignore-space-at-eol`; do not
   reset, normalize globally or overwrite user files such as `processo.txt`.
   Review local commits with the user: allocator first, DDO gitlink second,
   independent custom-block repositories and host separately. No history
   rewrite, remote update or publication is implied.
2. **Make environments reproducible.** Windows host pyproject/lock now selects
   the official PyPI `diffusers==0.41.0`; only that package was replaced in
   Windows `.venv`. See [official package transition](official-diffusers.md)
   for CPU evidence and pending generation acceptance. Linux's previous
   local-fork and audio-wheel overrides remain historical installed overrides;
   a fresh sync can undo them. Use Linux's direct interpreter or
   `uv run --no-sync` until its environment is reviewed. Do not apply the
   Linux CPU audio wheel blindly to Windows or inject the fork through PYTHONPATH.
3. **Fill only matched Windows Qwen gaps.** Inspect existing artifacts first.
   The current Linux trio is 512-square/stock resident VAE; recent selected
   Windows rows are 1024-square with different VAE/cleanup settings. The old
   512-square conventional result is historical, not a matched current control.
   If needed, propose the same finite current three-route BF16 workload with
   exact commands, fixed limits and acceptance criteria. Record successful
   Windows standby purge before transformer where used; the user may execute
   elevated commands. Do not rerun already closed LTX pairs automatically.
4. **Assess Qwen SDNQ on Windows separately.** Select checkpoint/revision and
   bit width, verify dependencies and preserve backend-specific forward math.
   No Qwen SDNQ runner or generation was validated in this Linux session. Start
   from the appropriate Diffusers group/leaf compatibility route; TorchAO FP8
   native acceptance does not imply SDNQ native compatibility. Require explicit
   approval before a finite test; do not regress frozen LTX routes.
5. **Validate WSL independently if requested.** Use a separate `.venv-wsl`;
   inspect environment, source transport, admission/cgroup/process bounds and
   binary capabilities first. A Linux .so loading is not WSL acceptance. Do
   not install a Linux NVIDIA kernel driver inside WSL or raise ceilings to
   pass. Propose a small native ownership/capacity gate before model tests;
   stop on failure. A separately approved linear/wsl_compat fallback is not a
   native VMM result. No Cartesian model/preset matrix is required.
6. **Review benchmark coverage and cut the selected version.** Preserve PASS,
   historical/unmatched, unavailable and untested labels. WSL and other
   presets/models remain limitations until measured. Finish reviewed local
   commits and the environment recipe before calling the release checkpoint
   frozen. Heavier video models are the next task, not part of this closure.

## Windows inspection only

These commands do not run inference. Confirm paths/interpreter exist first:

```powershell
Set-Location 'E:\ProjetosIA\ddo-examples'
git status --short
git -C ..\diffusers-dynamic-offloader status --short
git -C ..\diffusers-dynamic-offloader\ddo-vmm-allocator status --short
git -C 'F:\Projetos\diffusers' log -1 --oneline
.\.venv\Scripts\python.exe -B -c "import diffusers, diffusers_dynamic_offloader; print(diffusers.__file__); print(diffusers_dynamic_offloader.__file__)"
Get-Item Env:DDO_VMM_LIBRARY_PATH -ErrorAction SilentlyContinue
```

## Resume prompt

> Read AGENTS.md in ddo-examples, DDO and allocator, then host
> docs/windows-handoff.md and docs/benchmarks/README.md. The selected native
> Linux CPU/idle-trim/LTX image-video/Qwen512 sequence is closed. Preserve local
> changes, ignored custom-block clones, evidence and accepted binaries. Inspect
> Windows paths, actual editable imports, lockfile mismatch and intended DLL
> before acting. Continue the remaining finite work in order: reproducibility
> and reviewed local checkpoint, matched Windows Qwen gaps, Qwen SDNQ, then
> independent WSL validation. Before any CUDA/benchmark, show the exact command,
> hypothesis and approval criteria and wait for my confirmation. No automatic
> reruns, budget increases, environment replacement, push or history cleanup.
