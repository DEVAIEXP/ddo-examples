# Optional kernels

The examples host selects model-specific kernels separately from the DDO
offload runtime. Installing the DDO wheel does not select an attention backend.
Optional dependencies must be installed in the same environment as the runner.

## Sol-Attn for LTX video

The repository includes the reviewed Sol source under `kernels/sol_attn`.
All LTX video entry points use the shared loader when selecting:

```sh
--video-self-attention sol-attn
```

The loader verifies source hashes and imports the bundled copy lazily. No
`.pth`, `PYTHONPATH`, separate Sana checkout or download during inference is
required. Windows, native Linux and WSL use this same loading path with separate
platform environments. Install Triton and external FlashAttention 2 using
[optional Sol setup](setup.md#optional-sol-attn-for-ltx-video).

Our LTX integration selects external FlashAttention 2 in the first video block
and Triton Sol in the remaining eligible blocks, with `diag` and `tau=1.0`.
It preserves protected audio and cross-attention routes. Sol is approximate;
see [attention controls](ltx2-video-parameters.md#video-self-attention-backend)
and the [measured results](../benchmarks/ltx23-video.md) for the selected GPU
validation and quality scope. Shared loading does not establish acceptance
for every runner, shape or GPU.

## Origin and licenses

The 57 upstream files were copied without modification from
[NVlabs/Sana revision 781e834](https://github.com/NVlabs/Sana/tree/781e8341fb3b29d3aea5d50265772e9619340b66/techniques/sparse_backends/sol_attn).
The upstream README declares Apache-2.0 for repository code. The examples
preserve that README, the Apache license text, the
[third-party notices](https://github.com/NVlabs/Sana/blob/781e8341fb3b29d3aea5d50265772e9619340b66/techniques/sparse_backends/sol_attn/THIRD_PARTY_NOTICES.md)
and the included FlashAttention BSD license. Original authorship remains upstream.

`kernels/sol_attn_source.json` records the revision and normalized-text hashes.
The custom loader and LTX routing are host integration code; they do not change
the upstream kernel calculation. See [architecture](../developer/architecture.md#optional-attention-kernel-ownership)
for the ownership boundary.
