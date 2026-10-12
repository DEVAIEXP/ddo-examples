# LTX 2.3 image benchmarks

Updated 2026-10-11. Status: **selected BF16 Windows/native Linux closure complete**.
This does not mean every preset, precision, resolution or platform was tested.
WSL remains pending. Use [common rules and coverage](README.md) and the
[engineering evidence log](../../experiments/dynamic_offload_results.md).

## Image: 1280x704, seed 43, eight steps

BF16 encoder/transformer, real prompt, no PAG, native staged encoder/transformer,
resident or no-stream leaf VAE. Wall denoise is the modular call only, not the
entire Generate/setup phase. CUDA warm mean uses completed intervals 2-8.

| Platform | VAE | Denoise wall (s) | CUDA phase (s) | CUDA warm (s/step) | Tracked total (s) | Global GPU (GiB) | Global RSS (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Windows | Resident | 13.9869 | 14.6394 | 1.368514 | 39.1770 | 7.15 | 27.37 |
| Windows | Leaf | 14.3668 | 15.0109 | 1.371384 | 40.3444 | 7.17 | 27.37 |
| Native Linux | Resident | 17.1203 | 17.3995 | 1.056121 | 40.2959 | 6.56 | 27.33 |
| Native Linux | Leaf | 17.0253 | 17.3034 | 1.056260 | 37.6539 | 6.53 | 27.36 |
| WSL | Not validated | N/A | N/A | N/A | N/A | N/A | N/A |

Linux has a longer first CUDA step in these runs despite a shorter warm mean;
tracked totals include component loading, setup and cleanup, not only compute.
Do not attribute the difference in total time to the VAE alone.


## VAE phase

| Workload | Platform | VAE | Decode (s) | Decode GPU (GiB) | Decode RSS (GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| Image | Windows | Resident | 0.3306 | 3.19 | 1.65 |
| Image | Windows | Leaf | 1.1209 | 2.23 | 3.17 |
| Image | Native Linux | Resident | 0.3418 | 2.76 | 1.65 |
| Image | Native Linux | Leaf | 0.7482 | 1.82 | 3.19 |


These are sampled absolute driver GPU and process RSS peaks during decode,
not global pipeline peaks. Full load/decode/cleanup/save and encoder/transformer
phase tables are in the dated evidence log. Do not sum aggregate events twice.
Within each platform pair the delivered media hashes are identical; no
cross-platform byte equality or statistically established speed ratio is claimed.

## Recommendations

Keep resident decode when it fits, particularly when RAM is the tighter limit.
Explicit leaf trades slower decode and more CPU RAM for lower GPU residency.
Use supported tiling/streaming for activation pressure only with a decoder/quality
gate; it cannot guarantee space for weights or assembled CPU output. These are
VAE recommendations, not automatic transformer-preset selection. Global process
peaks near 27-28 GiB do not establish low-RAM operation.

## Runtime/preset coverage

| Combination | Windows | Native Linux | WSL |
| --- | --- | --- | --- |
| vmm/auto (vmm_fast), resident VAE | Selected closure PASS | Selected closure PASS | Not tested |
| vmm/auto, leaf VAE | Selected closure PASS | Selected closure PASS | Not tested |
| linear/auto (one_shot_fast) | Not in this matched closure | Not in this matched closure | Not in this matched closure |
| low_ram_safe / wsl_compat / warm_process | Not in this matched closure | Not in this matched closure | Not in this matched closure |
| Diffusers block/leaf compatibility / off | Not in this matched closure | Not in this matched closure | Not in this matched closure |
| vmm/nvme_baseline | Protected; not rerun here | Protected; not rerun here | Not validated |

Historical results remain in the evidence log; lack of a current matched row is
not a failed test. Auto aliases are not duplicate benchmark cases. FP8/SDNQ and
warm-process repeated generations have different scopes and remain separate.

## Reproduction and evidence

Use the primary host uv environment and [image runner guide](../ltx2-image.md).
The [platform handoff](../platform-validation-handoff.md) preserves fixed limits,
checkpoint/binary identities and approval boundaries. Exact current artifact
folders are outputs/ltx_image_vae_{resident,leaf}_gate on Windows
and outputs/ltx_image_vae_{resident,leaf}_linux_gate on Linux.
Detailed JSON and dated phase tables remain authoritative. No repeat of closed
cases is required merely to populate an untested preset row.
