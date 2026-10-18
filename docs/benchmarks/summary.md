# Comparison summary

Use each model's tables for measured values; no universal ranking is implied.

| Model / priority | What the selected measurements support | Tables |
| --- | --- | --- |
| LTX image / one-shot speed | Windows linear has shorter denoise but native has shorter tracked total; Linux native is shortest in both primary measures. | [LTX image](ltx23-image.md) |
| LTX video / one-shot speed | Native fast VMM has the shortest tracked total in each primary five-second platform set. | [LTX video](ltx23-video.md) |
| Qwen / one-shot speed | Linear/native shorten 512-square denoise versus official leaf; native has the shortest tracked total in the selected sets. | [Qwen](qwen-image-21.md) |
| Lower transformer-phase process RAM | Windows LTX video protected NVMe uses much less denoise RSS with a close total to native; OS file cache and the encoder still matter. | [Video Windows](ltx23-video.md#windows-bf16) |
| Lower decoder GPU pressure | Leaf VAE can lower decode GPU peaks at extra time/RAM; use tiling if activations still do not fit. | [Image VAE](ltx23-image.md#vae-strategies-separate-native-encoder-workload), [video VAE](ltx23-video.md#vae-strategies-separate-native-encoder-workload) |
| Saved INT4 | Selected Windows SDNQ DDO image cases outperform their INT4 leaf controls; this is not BF16 quality equivalence or native VMM support. | [LTX INT4](ltx23-image.md#windows-sdnq-int4), [Qwen INT4](qwen-image-21.md#windows-sdnq-int4) |
| Saved FP8 | Selected LTX video pairs reduce process RSS but take longer and have higher global GPU peaks than BF16. | [Video precision pairs](ltx23-video.md#saved-fp8-weight-only-separate-precision-pairs) |
| WSL | Bounded native or explicit-cap linear can trade time for GPU usage; Windows/Linux registration limits and streamed leaf performance do not transfer automatically. | [Image WSL](ltx23-image.md#wsl-bf16), [video WSL](ltx23-video.md#wsl-bf16), [Qwen WSL](qwen-image-21.md#wsl-bf16) |

These are single workload observations, not causal allocator-only or low-RAM
hardware guarantees. Text encoding can dominate global RAM even when denoise RAM
falls. Compare phase peaks and total separately; scenario links inside each table
provide the relevant limitations.

See [metric definitions](glossary.md), [presets](../user/presets.md) and
[reproduction](reproduction.md).
