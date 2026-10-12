# Benchmarks by model

Updated 2026-10-11. Reader-facing results use the same structure and comparison
rules; the [detailed evidence log](../../experiments/dynamic_offload_results.md)
retains dated measurements and raw acceptance, not a leaderboard.

Continue on Windows using the [current handoff](../windows-handoff.md);
the selected Linux sequence is closed, not permission for automatic reruns.

| Model/workload | Selected Windows closure | Selected Linux closure | Remaining |
| --- | --- | --- | --- |
| [LTX 2.3 image](ltx23-image.md) | BF16 VMM, resident/leaf VAE PASS | Same selected pair PASS | WSL; other presets not in this closure |
| [LTX 2.3 video](ltx23-video.md) | Five-second BF16 VMM, resident/leaf VAE PASS | Same selected pair PASS | WSL; other recipes separate |
| [Qwen Image 2.1](qwen-image-21.md) | Selected BF16 routes/VAE cases recorded | Shared 512-square leaf/linear/VMM trio PASS | Matched Windows gaps; Windows SDNQ; WSL |

## Standard comparison rules

Each page fixes checkpoint/revision, base/turbo variant, precision, prompt/input,
resolution, seed, steps, video duration/fps/audio and output scope. Primary rows:
VMM fast, linear auto/one_shot_fast and official Diffusers leaf where applicable.
Controls/presets remain visible even if not tested; do not run the Cartesian
product of all models, presets, VAE strategies, precisions and platforms.

Report denoise wall, phase-average s/step (including warmup), completed CUDA warm
mean only when available, tracked total, global sampled absolute GPU/RSS and
per-phase elapsed/GPU/RSS. Use N/A, not zero, for unavailable fields. Separate
RSS/USS, global/phase peaks and wall/CUDA scopes. Imports/bootstrap are outside
tracked totals. Short-phase sampled peaks are not enforced instantaneous limits.
Compare like workloads within a platform; no universal cross-platform ratio.

Coverage distinguishes selected PASS, historical/unmatched, OOM/failure, not
tested, unsupported and not applicable. auto and one_shot_fast are equivalent
linear resolutions, not two separate speed observations. low_ram_safe is a
capacity/copy tradeoff, not proven lower RSS; warm_process needs a separate
same-process repeated-run workload. Native VMM remains explicitly experimental
outside its selected contracts. Protected nvme_baseline is not retuned here.

Hardware recommendations separate transformer and VAE. Resident VAE when it
fits; leaf when VRAM is tight and RAM allows; supported tiling/streaming when
activation capacity requires it. Tiling does not remove weights/assembled output
and can affect time/quality. Record executed routes rather than requested flags.
Windows standby purge is optional but executed state is required for these
performance comparisons; outside Windows it is a no-op, not a prerequisite.

## Remaining finite sequence

1. Preserve the closed Qwen Linux comparison. Do not reopen its three cases or
   native tuning automatically; details and full phase tables are in its page/log.
2. On Windows, fill only explicit matched-data gaps and assess Qwen SDNQ with
   the chosen checkpoint/backend. Preserve SDNQ forward math; start with its
   compatible group-offload route, not assumed native VMM admission.
3. Assess WSL independently, with its environment/capacity gate first. No WSL
   support or output parity is inferred from native Linux.
4. Review reproducible environments and local commits, then cut the selected
   version. No publication, history rewrite or automatic benchmark expansion.
