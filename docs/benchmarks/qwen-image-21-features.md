# Qwen Image 2.1 feature measurements

These model-specific measurements are separate from the
[runtime comparisons](qwen-image-21.md). Read the
[memory and timing definitions](glossary.md) before comparing peaks.

## Windows: resident VAE decode at 1024 square

Base BF16, 1024x1024, 20 steps, seed 42, CFG 1 and KV cache, using the prompt
"A capybara wearing a wizard hat, reading by candlelight, oil painting".
Checkpoint snapshot: `d26bb61231c349cf6b7896fa83353113880e1ba3`.
Native VMM text encoder and transformer, resident VAE, default attention,
no compilation. Transformer residency uses the 0.625 GiB structural floor
without a saved profile; staging is 0.625 GiB and retained/registered source
budgets are 16/16 GiB. Execution-backing reclaim is disabled. Windows standby
purge succeeded after text encoding and before the transformer; before-run
and after-run purges were disabled.

Selected development-checkout runs on the reference Windows 8 GiB GPU / 64 GiB
RAM machine. Python 3.12.12, Torch 2.12.1+cu132, Diffusers 0.41.0,
Transformers 5.15.1. Examples revision `1ba82b1`, local Qwen custom blocks
revision `cd1efe4`. The rc5 environment interpreter used the DDO/allocator
source checkout; this is not wheel-only validation. The reported DLL path was
`ddo-vmm-allocator/build/Release/ddo_vmm.dll`; its post-run on-disk SHA256 was
`f81867613507ea4f44307a0e3089b403b536fd16df181cd8fc69e46ee3a359a7`.
A loaded-binary hash and NVIDIA driver version are unavailable in these reports.

All rows are individual observations, not repeated timing statistics.
Decode is the monitored VAE phase, excluding VAE load and cleanup. Both GPU
columns are sampled **absolute driver-used GiB**, including other GPU usage;
they are not allocator-owned bytes or baseline-relative console increments.
The full-run peak can occur outside decode. Sampling interval is 100 ms.

| VAE variant | Decode (s) | Decode peak GPU (absolute, GiB) | Full-run peak GPU (absolute, GiB) | Tracked total (s) | Output against control |
| --- | ---: | ---: | ---: | ---: | --- |
| Untiled control | 3.08 | 8.00 | 8.00 | 67.84 | Reference |
| Single-frame | 1.33 | 6.54 | 6.54 | 66.33 | Identical PNG |
| Single-frame + residual alias | 1.48 | 5.46 | 5.46 | 66.23 | Identical PNG |
| Single-frame + alias + bands 256 | 1.61 | 3.50 | 3.50 | 64.70 | Identical PNG |
| Single-frame + alias + bands 128 | 1.67 | 3.50 | 3.50 | 70.59 | Identical PNG |
| Tiling only | 2.95 | 2.19 | 3.44 | 67.07 | PSNR 36.45 dB; visual acceptance pending |

All six runs completed 20 steps with exit 0 and no observed native refusal or
cleanup failure. The first five PNGs share SHA256
`1ff1eca154725130c6205adf273ce069cd192ba8f8f385c07fea792ff7753b45`.
The tiling PNG SHA256 is
`22149aa6e2729026443605fa4d7ee4254f7492e1fe1f5fa5df317f3d5d07cd18`;
its RGB comparison has maximum absolute channel difference 92/255.
PSNR is computed over the saved RGB PNGs; it does not establish visual acceptance.

Bands 256 saved 4.50 GiB of sampled absolute peak against the control while
preserving this output exactly. Bands 128 did not lower the sampled peak further.
Their larger total was mainly in denoise (55.30 versus 50.17 s), before the
modified decode; it cannot be attributed to band size.
These results do not establish parity at other resolutions or for video.

### Choose a decode configuration

For the tested 1024-square recipe on an 8 GiB GPU, start with
`--vae-no-temporal-cache`: it had the shortest observed decode, preserved the
control PNG exactly and left about 1.46 GiB below the GPU's nominal capacity.
This is now the VMM runner's default decode setting. It is not a memory-fit
guarantee when other applications use the GPU. The original decoder remains
available with `--no-vae-no-temporal-cache`; linear and SDNQ defaults are unchanged.

If more memory headroom matters, add `--vae-alias-residuals`; add
`--vae-band-rows 256` as well for the measured 3.50 GiB full-run peak.
Both combinations preserved the selected output exactly. Band rows 128
offered no additional sampled peak benefit. Tiling reduced decode memory
further but changed the image, so it remains a separate quality tradeoff.
See the [1024-square usage recipe](../user/qwen-image-21.md#single-frame-1024-square-recipe).

### Reproduction controls

Use `scripts/qwen_image_21/run_dynamic_modular_vmm.py` with the workload and
budgets above, `--ddo-backend vmm --ddo-preset auto`, native text encoder,
`--vae-resident`, `--attention-backend default`, and a fresh profile directory
with `--ddo-profile auto` (no profile build). For band decode, select the local
Qwen custom-block checkout containing `cd1efe4` using `--custom-blocks`.
This development revision is not implied to be available from the Hub.

| Variant | Additional flags |
| --- | --- |
| Control | `--no-vae-no-temporal-cache` |
| Single-frame | `--vae-no-temporal-cache` |
| Alias | `--vae-no-temporal-cache --vae-alias-residuals` |
| Bands 256 | `--vae-no-temporal-cache --vae-alias-residuals --vae-band-rows 256` |
| Bands 128 | `--vae-no-temporal-cache --vae-alias-residuals --vae-band-rows 128` |
| Tiling only | `--vae-tiling --no-vae-no-temporal-cache` |

Band rows used two final decoder blocks and the module-derived halo of 11
input rows. Tiling did not enable single-frame, alias or band decode.
Source run IDs share prefix `f1-20261109-vae-` and suffixes `base`, `single`,
`alias`, `band256`, `band128`, `tiling`. Outputs were saved under
`outputs/benchmarks/windows/qwen-image21/rc5-profile-1024-use-native-<run-id>/`.
Original metrics, image hashes and logs are retained; public interpretation
does not require access to those local artifacts.
