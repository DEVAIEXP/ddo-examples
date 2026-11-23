# Optional runner kernels

These sources belong to the model runner host, not the generic DDO wheel.
LTX video runners load the bundled Sol-Attn source only when
`--video-self-attention sol-attn` is selected. No `.pth`, `PYTHONPATH`, separate
Sana checkout or runtime download is required. Use the same optional dependencies
in the runner's environment on Windows, native Linux and WSL; see
[setup](../docs/user/setup.md#optional-sol-attn-for-ltx-video).

## Sol-Attn provenance and licenses

`sol_attn/` is an unmodified copy of
[`techniques/sparse_backends/sol_attn` in NVlabs/Sana](https://github.com/NVlabs/Sana/tree/781e8341fb3b29d3aea5d50265772e9619340b66/techniques/sparse_backends/sol_attn),
revision `781e8341fb3b29d3aea5d50265772e9619340b66`. The upstream README states
that repository code is released under Apache-2.0. Its complete README is
preserved as [SOL_UPSTREAM_README.md](SOL_UPSTREAM_README.md), and the full
license text is included as [LICENSE.apache-2.0](LICENSE.apache-2.0).
Original authorship is retained; this copy is not an independent implementation.

Preserve [upstream third-party notices](sol_attn/THIRD_PARTY_NOTICES.md) and
[the FlashAttention license](sol_attn/sm100/LICENSE.flash-attention), including
the notices for vendored code. This host's LTX integration selects the Triton
backend; the copied architecture-specific code does not establish validation
on those other architectures.

`sol_attn_source.json` records file hashes after CRLF-to-LF normalization, so
Windows and Linux checkouts verify the same text. The loader validates those
hashes, keeps upstream's internal `sol_attn` imports intact, and refuses an
already-loaded external package. Benchmark reports also retain raw file hashes.
Updates require review, a new provenance manifest and separate validation;
do not update the manifest simply to silence a source-integrity failure.
