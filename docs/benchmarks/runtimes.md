# Transformer runtimes, presets and VAE routes

Backend, preset and component route are different dimensions. Read these labels
before interpreting memory or speed comparisons.

## DDO selector to executed transformer runtime

The dense BF16/FP16 mapping below does not override quantized backend detection
or explicit component configuration.

| Backend / preset selector | Resolved route | What executes transformer weights |
| --- | --- | --- |
| linear / auto | `one_shot_fast` → linear_runtime | DDO dense offload under normal PyTorch CUDA allocation |
| linear / `one_shot_fast` | linear_runtime | Same default linear policy, explicitly requested |
| linear / `low_ram_safe` | linear_runtime | Conservative linear policy; different capacity/copy tradeoff |
| linear / `wsl_compat` | linear_runtime | WSL-safe linear fallback, not native VMM |
| linear / `warm_process` | linear_runtime | Process-lifetime cache policy; compare repeated-process workloads separately |
| vmm / auto (or explicit `vmm_fast`) | vmm_fast_runtime | Target native C++ allocator, arena and retained source transport |
| vmm / `nvme_baseline` | vmm_runtime | Protected legacy NVMe-backed VMM route, distinct from native fast |
| `diffusers_offload_compat` | plan / Diffusers block group offload | Official Diffusers compatibility hooks, not DDO linear_runtime |
| `diffusers_leaf_offload_compat` | plan / Diffusers leaf group offload | Official Diffusers compatibility hooks, not DDO native fast |
| off | No DDO hooks | Caller-controlled placement, not a performance preset |
| `one_shot_fast` with the accepted SDNQ packed recipe | sdnq_runtime | DDO packed-module route preserving SDNQ forward math; normal CUDA allocator |

A standalone official Diffusers control does not have a DDO preset; label it
not used rather than inventing one. Requested selectors and recorded settings
can differ: retain actual report fields and label the executed runtime. A native
runner may preserve a `one_shot_fast` settings label while its explicit native
configuration executes vmm_fast_runtime; the label alone is not runtime proof.

## VAE is independent from the transformer runtime

| VAE route | Meaning |
| --- | --- |
| Resident | Decoder weights stay on the execution GPU during decode |
| Leaf | Official no-stream leaf offload uploads decoder modules as needed |
| DDO linear convolution | Explicit normal-allocator convolution adapter, not the ordinary resident route |
| Native VMM VAE | Explicit native decoder source/capacity integration; not implied by a VMM transformer |
| Tiling / temporal streaming | Decoder activation/output-window strategies, not transformer presets |

Resident VAE can accompany a DDO linear, native-fast or official-control
transformer. It does **not** mean the complete transformer is resident.
Likewise, leaf VAE does not identify which runtime denoised the latents.

Native bootstrap installs a process-wide CUDA allocator. Therefore two leaf-VAE
cases can still use different underlying process allocators and retained caches.
VAE route equality alone does not make a memory comparison fully matched.

## How to read coverage

Model pages identify the executed transformer runtime and requested selector,
then list VAE strategy separately. Coverage rows without measurements are not
leaderboard entries or failed tests. Distinguish:

- Selected measured PASS for the precise recorded workload.
- Historical/separate evidence, not matched to the current table.
- Not tested or not covered by the selected gate.
- Protected baseline retained but not rebenchmarked.

The legacy NVMe route is retained and is not renamed or merged into native fast.
Do not reconstruct an old 4.9 s/step observation as a matched modern result when
the prompt/encoder, source revision, monitor or timing scope is unavailable.
No complete model/preset/platform matrix or new benchmark is authorized merely
to make the documentation's coverage table exhaustive.
