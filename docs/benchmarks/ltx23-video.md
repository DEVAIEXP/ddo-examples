# LTX 2.3 video benchmarks

Updated 2026-10-11. Status: **selected BF16 Windows/native Linux closure complete**.
This does not mean every preset, precision, resolution or platform was tested.
WSL remains pending. Use [common rules and coverage](README.md) and the
[engineering evidence log](../../experiments/dynamic_offload_results.md).

## Video: 1280x704, five seconds, 24 fps, eight steps

121 frames, seed 43, BF16, guitar image/prompt, no conditioning audio and generated
output audio. Native staged encoder/transformer, unchanged temporal streaming and
staged audio VAE/vocoder. Average wall time below is denoise wall/8, includes
warmup/readback and is not synchronized CUDA steady-state. Per-step CUDA timing
was unavailable in these normal video runs; N/A is not zero.

| Platform | VAE | Denoise wall (s) | Wall average (s/step) | CUDA warm | Tracked total (s) | Global GPU (GiB) | Global RSS (GiB) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| Windows | Resident | 198.5835 | 24.8229 | N/A | 261.6728 | 5.14 | 28.42 |
| Windows | Leaf | 199.3031 | 24.9129 | N/A | 271.8269 | 4.96 | 28.42 |
| Native Linux | Resident | 209.7483 | 26.2185 | N/A | 266.5132 | 4.87 | 28.42 |
| Native Linux | Leaf | 209.5043 | 26.1880 | N/A | 272.2485 | 4.79 | 28.42 |
| WSL | Not validated | N/A | N/A | N/A | N/A | N/A | N/A |


## VAE phase

| Workload | Platform | VAE | Decode (s) | Decode GPU (GiB) | Decode RSS (GiB) |
| --- | --- | --- | ---: | ---: | ---: |
| Video | Windows | Resident | 22.6573 | 4.81 | 7.16 |
| Video | Windows | Leaf | 32.8762 | 3.02 | 9.27 |
| Video | Native Linux | Resident | 24.3894 | 4.87 | 6.23 |
| Video | Native Linux | Leaf | 32.5026 | 2.59 | 7.51 |


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

The accepted video route already uses temporal streaming. Its helper disables
built-in VAE tiling/framewise flags per window and restores them afterward.
This pair does not validate new spatial tiling or a new window geometry.

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

Use the primary host uv environment and [video runner guide](../ltx2-video.md).
The [platform handoff](../platform-validation-handoff.md) preserves fixed limits,
checkpoint/binary identities and approval boundaries. Exact current artifact
folders are outputs/ltx_video_vae_{resident,leaf}_gate on Windows
and outputs/ltx_video_vae_{resident,leaf}_linux_gate on Linux.
Detailed JSON and dated phase tables remain authoritative. No repeat of closed
cases is required merely to populate an untested preset row.
