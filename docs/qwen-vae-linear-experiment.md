# Qwen VAE linear convolution experiment

The BF16 linear runner exposes `--vae-linear-offload` as an explicit experimental
alternative to `--vae-leaf-offload`. They are mutually exclusive. Transformer and
VAE run in the same normal PyTorch allocator process; this is not a native VMM
comparison and requires no DLL or converted checkpoint. Leaf is now selected
by the DDO VAE policy for auto/one_shot_fast/low_ram_safe; `--vae-resident`
reproduces the previous resident choice. Other selected presets retain resident
decode. `--vae-leaf-offload` remains a compatible explicit override.

DDO owns convolution parameter transfers, native-forward execution, cleanup and
public reporting. The local decode block accepts an optional `execution_device`
because CPU source placement must not determine where computation executes.
The original Diffusers classes are not modified.

Use the same base model, 1024 x 1024, seed 42, 20 steps, CFG 1, KV cache,
`--vae-no-temporal-cache --vae-alias-residuals`, no tiling or direct-padding,
and successful Windows standby purge for both cases. Keep the transformer preset
and all its budgets identical. Compare the `decode` phase's sampled absolute
driver GPU peak and completed duration, not the baseline-relative pipeline delta.
Check exact equality of both PNGs and compare against the preserved accepted PNG.
Report VAE metrics via the public DDO report and confirm complete phase cleanup.

The four small normal-allocator CUDA contracts passed, followed by the selected
20-step matched pair. Leaf decode measured 1.5958 s / 6.469936 GiB sampled
absolute driver peak; DDO linear measured 1.3642 s / 6.503990 GiB. PNG SHA256
was identical to the accepted reference:
`1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45`.
Functionality passed; no memory reduction was demonstrated. The new delegation
of configuration/routing to DDO has CPU contracts, without a new CUDA run.
Run the small
normal-allocator convolution gate before real-model benchmarks, with explicit
approval. Stop at the first failure, ownership anomaly or image mismatch. The
official leaf route remains the accepted capacity option until this candidate
demonstrates an advantage; do not automatically combine it with native VMM.
