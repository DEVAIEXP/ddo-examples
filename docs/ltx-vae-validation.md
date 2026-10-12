# LTX VAE candidate validation

Status: 22 CPU contracts passed. The selected Windows image and five-second
video resident/leaf gates completed with identical saved output hashes on
2026-10-11. VAE leaf capacity compatibility is accepted for these workloads;
resident defaults remain unchanged. Native Linux image and five-second video
pairs subsequently passed with identical delivered media within each pair,
recorded in benchmark-matrix.md; WSL follow-up remains pending.
This is a bounded candidate gate, not a new default or a performance claim.

## Candidate scope

- Image/video VMM runners expose `--vae-leaf-offload` through DDO's public
  `enable_offload` API. Without the flag, resident decode is preserved.
- Leaf uses no stream and retains the checkpoint dtype and original VAE math.
  Only the video VAE decode changes route; conditioning encode, audio VAE and
  vocoder retain their existing behavior.
- Streaming tiles use the pipeline execution device, not the current parameter
  device. Temporal geometry, blending, CPU stitching and flag restoration stay
  unchanged. Use the local video custom-block clone for this candidate; the
  pinned Hub code does not acquire local edits automatically.
- Load, decode, cleanup and save have separate sampled memory/time phases.
  The historical combined decode name is an aggregate event, not an additional
  phase to sum. Historical JSON is not rewritten.
- No Qwen-specific cache removal, residual aliasing, dtype change, new spatial
  tiling policy, automatic native cache trim or allocator rebuild is included.

## Proposed finite Windows gate

Await user confirmation before executing. Use the root uv environment and the
same selected DLL for both members of each pair. An elevated terminal is useful
for standby purge; record whether purge actually succeeded.

Compare resident vs `--vae-leaf-offload`, changing only that flag and output
directory:

1. LTX image: BF16, 1280x704, seed 43, real prompt, eight steps, native encoder,
   existing 4 GiB transformer/encoder settings, no PAG, no new tiling.
2. LTX video: BF16, 1280x704, five seconds at 24 fps (121 frames), eight steps,
   seed 43, the existing reference image/prompt, no input audio. Retain native
   encoder (4 GiB), transformer ceiling 1.5 GiB, source/registration 24.625 GiB,
   streaming GPU decode and accepted transformer options. Audio is still generated.

Hypothesis: leaf reduces VAE weight residency without changing decoded output;
the benefit can be small because activation memory is unchanged. Do not assume
the Qwen reduction transfers to LTX.

Accept compatibility only if both routes complete, image pixels match, video
frame count/resolution and decoded frame pixels match, audio remains finite and
unchanged, and there is no CUDA or ownership failure. Compare decoded content,
not only MP4 file hashes. Reject a quality change. Promote a recommendation only
if the measured memory/time tradeoff is useful; do not alter the default merely
because the route is compatible. Stop at the first failure or memory anomaly.

## Release follow-up

After Windows acceptance, consolidate matched benchmark summary and per-phase
tables in this repository, with unavailable fields marked N/A. Distinguish
average phase seconds per step from measured CUDA steady-state. Recommendations
must be workload-specific; `low_ram_safe` does not prove lower process RSS.

Prepare a separate native Linux and WSL handoff. Existing Linux evidence remains
valid only for its recorded binary/workload. WSL is an independent experimental
gate, not a substitute for native Linux and not promised VMM support. Do not fill
platform rows before actual execution. Freeze the selected version after those
finite gates, before starting heavier video-model work.
