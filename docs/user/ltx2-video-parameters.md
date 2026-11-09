# LTX video parameters

The four public entries use canonical names only. Use the selected runner's
`--help` for its supported subset and [shared controls](runner-controls.md)
for lifecycle and reporting semantics.

## Interface scope by runner

| Controls | Minimal | Linear | Native VMM | SDNQ |
| --- | --- | --- | --- | --- |
| Generation, conditioning, custom blocks, LoRA, purge and reports | Yes | Yes | Yes | Yes |
| `--ddo-backend` | `vmm` | `linear` | `vmm` | `linear` |
| `--ddo-preset` | `auto` | Linear/compatibility presets | `auto` or `nvme_baseline` | Linear or leaf compatibility presets |
| Separate transformer `--model-path` | No | Yes | Yes | Yes |
| `--max-sequence-length`, `--reuse-rotary-embeddings` | No | Yes | Yes | Yes |
| Native capacities and execution backing reclaim | Yes | No | Yes | No |
| Chunking, phase modulation, compact timesteps and inplace residuals | No | Yes | Yes | No |
| Native text encoder (`--[no-]text-encoder-native-vmm`, encoder budget) | Yes | No | Yes | No |
| Other native encoder/VAE controls | No | No | Yes | No |
| `--bits` | No | No | No | Yes |
| `--metrics-level` | No | No | Yes | No |
| `--ddo-profile NAME`, `--build-ddo-profile` | No | Yes | No | No |

Unsupported options are rejected, including abbreviated names. Backend and
preset selection do not change the input audio or decode device automatically.

## Sources and generation

| Control | Purpose |
| --- | --- |
| `--distilled-model` | Defaults to `OzzyGT/LTX-2.3-Distilled-1.1`; BF16 pipeline/auxiliary source. Also supplies the BF16 transformer unless overridden; SDNQ selects its quantized transformer separately. |
| `--model-path` | Optional transformer-only override; auxiliary components still use `--distilled-model`. |
| `--custom-blocks` | Defaults to `elismasilva/ltx2.3_video_custom_blocks`; trusted model code, or a local development directory. |
| `--custom-blocks-revision` | Optional Hub branch, tag or commit; omission uses `main`. Does not switch a local clone. With `--local-files-only`, only the cached snapshot is available. |
| `--local-files-only` | False; require cached model/code files when enabled. Input-media URLs are separate. |
| `--prompt` | Defaults to `the man plays the guitar and sings on an old western town`. |
| `--image` | Defaults to the guitar image URL from the OzzyGT Diffusers examples dataset; accepts a path or URL. |
| `--audio`, `--use-audio-condition` | No default audio; conditioning is off. Enabling requires an explicit audio file. Accepts `--no-use-audio-condition`. |
| `--width`, `--height` | 1280 × 704; positive multiples of 32. |
| `--seconds`, `--frame-rate` | 5 seconds, 24 fps; finite positive values. |
| `--steps`, `--seed` | 8 steps, seed 43; distilled entries accept 1–8 steps. |
| `--max-sequence-length` | 256; positive text token limit in linear, VMM and SDNQ. |
| `--bits` | 8 in SDNQ; choose 4 or 8 for the corresponding saved transformer. Auxiliary components remain BF16. |
| `--lora`, `--lora-weight-name`, `--lora-revision`, `--lora-scale` | Optional transformer adapters; see [LoRA usage](ltx2-video.md#optional-lora) for repeated adapters and defaults. |

Input-audio conditioning and generated output audio are different. The default
image does not supply an audio file. Custom blocks execute trusted Python code.

## Native capacity and components

The VMM entry exposes the same DDO route/capacity and reclaim controls as the image entries.
Capacities use GiB; they are independent limits, not total GPU usage.

| Control | Default and meaning |
| --- | --- |
| `--ddo-backend`, `--ddo-preset` | `vmm`, `auto`; experimental native fast. `nvme_baseline` selects the protected NVMe route. |
| `--ddo-vmm-resident-weight-budget-gb` | 4 GiB; active transformer weight capacity. |
| `--ddo-vmm-total-staging-gb` | Omitted: native fast uses 0.625 GiB total host staging; NVMe uses two 0.625 GiB slots. Explicit values specify total staging for either route. |
| `--ddo-vmm-source-cache-gb` | 0; native retained-source cache disabled. |
| `--ddo-vmm-source-registered-gb` | 0; native registered-source ceiling inherits the source-storage ceiling. |
| `--ddo-vmm-execution-backing-reclaim` | False; opt into reclaim of eligible idle execution backing under pressure. Requires `vmm/auto`; the environment variable does not enable it in public VMM entries. See [behavior and tradeoffs](runner-controls.md#execution-backing-reclaim). |
| `--text-encoder-native-vmm` | On for `vmm/auto` (Minimal and Native VMM entries), off otherwise: the Gemma text encoder is loaded through DDO's weight-aware loader with native component ownership. `--no-text-encoder-native-vmm` restores Diffusers leaf group offload. Requesting it on another route is rejected. |
| `--text-encoder-vmm-resident-weight-budget-gb` | 4 GiB with the native text encoder; encoder-only capacity, separate from the transformer ceiling. Two execution slots of the largest encoder component must fit (0.625 GiB is refused at attachment). Rejected when the native text encoder is off. |
| `--transformer-fp8-weight-only` | Off; load supported saved TorchAO FP8 weights. Requires native fast, a local `--model-path` and `--local-files-only`; does not convert BF16 weights. |
| `--release-pytorch-host-cache-before-transformer` | Off; request PyTorch host-cache release at the phase boundary. Separate from Windows standby purge and native source-cache capacity. |

The native text encoder default comes from one Windows LTX 2.3 video 8 s
1280x704 pair with a cold standby cache in both runs: encoder phase 11.2 s
native versus 20.8 s leaf, total 321.3 s versus 329.7 s, identical output; the
native encoder raised that phase's absolute GPU peak from about 3.0 to 5.0 GB,
below the run's decode peak. It is a recipe default, not general acceptance.

Native API and transport selection are automatic. Source-cache settings apply
to native fast, not the protected NVMe runtime. Benchmark capacities are explicit
overrides rather than these defaults.

## Model computation

| Control | Default and meaning |
| --- | --- |
| `--reuse-rotary-embeddings` | True in linear, VMM and SDNQ; prepare generation-local RoPE once. Accepts `--no-reuse-rotary-embeddings`. |
| `--feed-forward-chunk-size` | 1024 in linear, 0 in VMM. Nonnegative token chunk size; 0 disables chunking. |
| `--normalization-chunk-size` | 1024 in linear, 0 in VMM. Nonnegative token chunk size; 0 disables chunking. |
| `--phase-modulation` | On in linear, off in VMM; use phase-local modulation buffers. Linear accepts `--no-phase-modulation`. |
| `--compact-timesteps` | On in linear, off in VMM; project spatially uniform video timesteps without the dense repeated representation. Unsupported layouts remain dense. Linear accepts `--no-compact-timesteps`. |
| `--inplace-residuals` | On in linear, off in VMM; reuse model-owned residual states in inference mode. Linear accepts `--no-inplace-residuals`. |

These are custom-block computation options, not generic DDO presets. Preserve
the exact choices when comparing measurements. Compilation and experimental
fused RMS/rotary kernels belong in the [diagnostics entry](../developer/diagnostics.md).

## Decode and output

| Control | Default and meaning |
| --- | --- |
| `--decode-device` | `cpu`; choose `cuda:0` explicitly for GPU decoding. |
| `--stream-video-decode` | Off; temporal window decoding requires `--decode-device cuda:0`. |
| `--video-decoder` | `streaming-tiles`; used with `--stream-video-decode`. `streaming-tiles` decodes overlapping temporal windows and blends them. `exact-streaming` runs the VAE decoder over the whole sequence in temporal streaming, without window seams or overlap recomputation; it needs custom blocks that export `LTX2VideoExactStreamingDecoderStep` and a VAE without decoder timestep conditioning. |
| `--vae-leaf-offload` | Off, VMM only; requires GPU streaming. Audio VAE and vocoder routes stay separate. |
| `--output-type` | `video`, VMM only; `latent` skips output decoding and cannot be combined with streaming. |
| `--output-dir` | `outputs/ddo_modular_video`, or `outputs/ddo_modular_sdnq_video` for SDNQ. |

## Lifecycle and reports

All four entries expose `--purge-before-run`, `--purge-after-text-encoder`,
`--purge-before-transformer` and `--purge-after-run`, with negative forms.
Omission inherits DDO environment/preset policy; it does not force purge off.
See [Windows standby purge](runner-controls.md#windows-standby-purge).

| Control | Default and meaning |
| --- | --- |
| `--memory-monitor` | `rss`; `full` additionally collects USS and can add overhead. |
| `--metrics-level` | 2 in VMM; accepts 0/1/2. Level 2 records public DDO component snapshots. |
| `--show-profile` | Off; display the component profile. |
| `--show-metrics`, `--save-metrics` | Both on; display observations and save JSON. |
| `--dry-run` | Off; print a phase plan without weights, network or CUDA. |
| `--help` | List the selected entry's arguments without generating video. |

Display/save switches accept their `--no-...` forms. Phase observations do not
prove GPU completion or local VRAM placement. CPU parser checks do not establish
GPU or performance acceptance.
## Video self-attention backend

All LTX video runners accept `--video-self-attention` with `default`,
`flash-attn2`, `sol-attn`, or `cudnn`. The default preserves the model's
configured attention. `flash-attn2` uses the optional external FlashAttention 2
package. `sol-attn` uses Triton Sol-Attn (`diag`, `tau=1.0`) and uses
FlashAttention 2 for the first transformer block. `cudnn` keeps PyTorch scaled
dot-product attention but sets the backend priority to flash, cuDNN,
memory-efficient, math; it needs no extra package. PyTorch builds without
flash attention, such as the Windows wheels, then use cuDNN attention for
eligible inputs instead of the memory-efficient kernel, and fall back to the
remaining backends when cuDNN is not eligible. These choices apply only to
eligible video self-attention. Audio, cross-attention, and masked or perturbed calls retain
their original routes. Saved metrics include per-layer dispatch counts.

Install the optional FlashAttention 2 wheel as described in [Setup](setup.md).
For Sol-Attn, make the Sana `techniques/sparse_backends` directory importable
in the runner environment. The tested Sana revision is
[`781e834`](https://github.com/NVlabs/Sana/commit/781e8341fb3b29d3aea5d50265772e9619340b66).
The selected GPU validation used the VMM Distilled runner on an RTX 3060 Ti.
Sol-Attn accepts dynamic token counts and requires CUDA BF16 video attention
with head dimension 128. The validated recipe used 14,080 video tokens
(1280 × 704, 5 seconds at 24 fps). A 10-second run encountered an illegal
CUDA memory access during denoising. Its cause remains under investigation.
