# Qwen-Image 2.1 saved SDNQ INT4

`scripts/qwen_image_21/run_dynamic_modular_sdnq.py` is a separate,
phase-staged runner. It preserves SDNQ's quantized module forward and uses
DDO's packed `sdnq_runtime`, not dense-linear hooks or the native VMM allocator.
The existing BF16 runners and custom blocks are unchanged.

The initial checkpoint is
[OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit](https://huggingface.co/OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit),
pinned to `59d4c3610379d91fb023c4d0b6527c76755491c4`.
Its saved recipe uses dynamic quantization and Hadamard rotation. Load only
its transformer; text encoder, processor, scheduler and VAE come from the
cached `Qwen/Qwen-Image-2.1` BF16 checkpoint. Import SDNQ before loading weights.
A non-SDNQ recipe or missing SDNQ modules is an error, not a dense fallback.

The runner defaults to base, 512 x 512, 20 steps, seed 42 and the same prompt
as the BF16 Qwen runners. Turbo is not an accepted SDNQ route. Checkpoint
overrides are explicit and do not inherit another repository's pinned revision.

For a transformer-only offload comparison, change only `--ddo-preset` between
`one_shot_fast` and `diffusers_leaf_offload_compat`. The former explicitly
selects the packed runtime; the latter selects Diffusers streamed leaf offload.
Both keep auxiliary components on the same one-shot policy. Use
`--vae-resident` to match the accepted BF16 resident, untiled VAE workload.
Run Windows comparisons elevated with `--purge-before-transformer`; inspect
the reported purge result instead of assuming success.

Shared reporting records phase times, callback intervals, memory, the saved
quantization recipe and public DDO reports. Callback intervals are not
synchronized per-step GPU timings. Compare denoise separately from setup;
the first download is not an inference performance measurement.

Status: the user completed both selected Windows INT4 routes on official
Diffusers 0.41.0. DDO denoise took 18.8056 s versus leaf's 24.2629 s;
both produced the same PNG. See the selected SDNQ closure in
[detailed results](../experiments/dynamic_offload_results.md) for exact
timing/memory scopes and artifact hashes. This is not generic SDNQ model
support or acceptance of INT8, Turbo, Linux/WSL or quantized encoders.
Do not automatically repeat the full BF16 benchmark matrix.
