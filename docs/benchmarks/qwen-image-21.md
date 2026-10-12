# Qwen Image 2.1 benchmarks

Updated 2026-10-11. Status: selected Windows BF16 evidence and the approved
Linux 512-square leaf/linear/VMM trio PASS. SDNQ and WSL remain pending.
Use [common rules](README.md), [runner integration](../qwen-image-21.md) and the
[dated evidence log](../../experiments/dynamic_offload_results.md).

## Current selected workload

Base BF16, 1024x1024, seed 42, 20 steps, default capybara prompt, CFG 1,
native KV cache, untiled decode. Accepted local VAE cacheless/residual options
are explicit; rejected direct padding stays disabled. Turbo's saved schedule,
other precisions, tiling and other shapes are separate scenarios.

## Recorded Windows cases

| Route/case | Denoise wall (s) | Wall average (s/step) | Total (s) | Global GPU (GiB) | Global RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear one_shot_fast, preset-selected leaf VAE | 54.2983 | 2.7149 | 78.5180 | 7.07 | 21.19 |
| Native VMM, leaf VAE and explicit idle trim | 52.8170 | 2.6409 | 72.5433 | 6.63 | 18.97 |
| Native Linux | N/A | N/A | N/A | N/A | N/A |
| WSL | N/A | N/A | N/A | N/A | N/A |

Artifacts: outputs/qwen21_vae_preset_auto/metrics.json and
outputs/qwen21_vae_leaf_trim_enabled/metrics.json. These rows have different
allocator/cleanup settings; do not claim an allocator-only matched speedup.
Their accepted PNG SHA256 is
1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45.
Wall averages include warmup/readback, not completed CUDA steady state.

## Phase and VAE evidence

The evidence log contains full Windows phase/memory artifacts and selected
leaf, native VMM and normal-allocator linear convolution VAE comparisons.
They are compatibility/capacity variants, not all matched rankings. Native
transformer plus explicit idle trim/leaf is a selected capacity option, not
full cache release or automatic VAE native opt-in. The latest preset routing
checks preserve actual route and report successful Windows purge. Historical
reports missing fields remain unavailable; do not fill them from another case.

## Linux gaps and bounded proposal

The VMM .so and LTX workloads passed on this Linux host, but this does not prove
Qwen model-specific execution. Qwen local weights are present. Before CUDA,
review one shared workload for official streamed Diffusers leaf, linear auto
and native VMM; agree on resolution and the same VAE/precision/output scope.
Do not compare a 512-square conventional control to 1024-square modular runs
as matched evidence. Standard and staged pipelines have different lifecycle
scope even at equal dimensions; state that rather than changing the control.

Use at most one approved base-model run per route, stop on the first error or
memory anomaly, and inspect output parity and per-phase telemetry. Do not add
low_ram_safe, Turbo, every VAE mode or long-run repeats automatically. Reuse
existing defaults/native ceilings only after verifying their actual settings.
The commands below describe the finite sequence subsequently approved and
executed; this page is not permission to repeat or expand it.

Preflight on the new Linux host: 27 Qwen/block/local-VAE CPU contracts passed
in 4.05 s, no skips. The local Qwen snapshot is
d26bb61231c349cf6b7896fa83353113880e1ba3; all three selected CUDA generations
then completed. Preserve their exact delivered PNG hashes and raw observations.
The inspected historical conventional Windows control was 512x512, not 1024x1024.
The finite shared Linux proposal is therefore base BF16, 512x512, seed 42,
20 steps, the default prompt, stock resident VAE in all three routes, no tiling
or model-specific VAE options. Linear uses auto/one_shot_fast; native VMM uses
the existing 4 GiB active / 0.625 GiB staging / 16 GiB source and registration
ceilings. No idle trim or cuBLAS clear is added to this comparison.
The conventional control retains the full pipeline; modular routes remain
phase-staged. Those lifecycle differences must remain visible, even with equal
dimensions and resident VAE. Later matched Windows rows may still be needed;
the recorded 1024-square cases are not replaced by these new 512-square cases.

## Coverage and later Windows/WSL work

| Combination | Windows | Native Linux | WSL |
| --- | --- | --- | --- |
| BF16 linear auto/one_shot_fast | Selected evidence | Shared 512-square gate PASS | Not tested |
| BF16 native VMM fast | Selected evidence | Shared 512-square gate PASS | Not tested |
| Conventional Diffusers leaf | Historical evidence; inspect workload/scope | Shared 512-square control PASS | Not tested |
| low_ram_safe | Selected Windows case recorded | Not in proposed minimal gate | Not tested |
| Other linear/compat presets and nvme_baseline | Not in current matched comparison | Not in proposed gate | Not tested |
| VAE resident/leaf/linear convolution/native | Selected component evidence, differing allocators | Not generalized from CPU tests | Not tested |
| SDNQ | Requested future Windows task | Not planned here | Not tested |
| Turbo/FP8 | Not in this BF16 closure | Not planned here | Not tested |

SDNQ needs its selected model/revision, bit width, dependency and backend-specific
forward preserved. Do not assume the saved TorchAO FP8 native payload contract
admits SDNQ. Propose its compatibility route and CUDA gate separately on Windows.
No direct-padding promotion or change to frozen LTX runtimes is authorized.

## Hardware guidance

Use the same phase-specific guidance as the other pages: faster weight transport
can increase peaks; offload does not remove activations. Leaf VAE can save VRAM
but use more RAM; tiling is an activation fallback with a quality check. Selected
process RSS around 19-21 GiB is not low-RAM system acceptance. Selected Linux
512-square measurements are below; WSL and other hardware capacities remain
unvalidated. Recommendations must stay within measured workload/phase scope.

## Closed Linux 512-square comparison

| Linux route | Denoise/readback (s) | Phase average (s/step) | Tracked total (s) | Global GPU (GiB) | Global RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official streamed leaf (full pipeline) | N/A | N/A | 45.3132 | 4.800171 | 32.598476 |
| DDO linear auto (staged) | 18.1429 | 0.9071 | 34.7060 | 5.580688 | 18.897018 |
| DDO VMM fast (staged) | 15.6808 | 0.7840 | 23.2622 | 5.719055 | 18.867943 |

All three saved PNGs are byte-identical, SHA256
f53ebe128845d491a98a02b56e693d1b993fecf6f2777239df84d920612eb11f.
The control's separate denoise time is N/A: its generate phase includes encoder,
denoise and VAE. DDO phases include denoise/readback; phase-average s/step includes
warmup, not synchronized CUDA warm timing. No CUDA per-step timing was requested.
VMM is faster in this selected system-level case but uses more peak GPU than
the official control; staged DDO uses less RSS. Do not attribute lifecycle/setup
differences to the allocator alone, generalize to 1024 or promise low-RAM fit.
Full per-phase tables and fixed ceilings are in the dated evidence log. Artifacts:
outputs/qwen21_{leaf,linear,vmm}_linux_512. No model code or budget was changed.
Selected Linux coverage is now closed; matched Windows gaps, SDNQ and WSL remain
separate next tasks. No automatic rerun or expansion is authorized.
