# DDO native VMM benchmark results

## Selected WSL BF16 closure - 2026-10-12

User-executed single runs on RTX 3060 Ti 8 GiB, Ubuntu 22.04.3 WSL2,
Python 3.12.12, Torch 2.12.1+cu132 and official Diffusers 0.41.0.
See [installation/build recipe](../docs/wsl-host-environment.md).
Native candidate: build-wsl-idle-trim-20261012/libddo_vmm.so, SHA256
5915cca767c89ab7606c04f2001c9eddffa762386c852cc9f517708a2d03584c.
All native routes used 384 MiB staging. RAM is process RSS; GPU peaks are
sampled absolute driver usage including baseline, not PyTorch allocated bytes.
Windows standby purge does not execute on WSL.

| Workload | Source registration MiB | Denoise wall s | Tracked total s | Absolute GPU peak GiB | RSS peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| LTX image, 1280x704, 8 steps, seed 43 | 128 | 104.6665 | 186.0851 | 7.2150 | 26.3132 |
| LTX image, same workload | 512 | 85.9101 | 167.4604 | 7.2863 | 26.3148 |
| LTX conditioned video, 1280x704, 5 s/24 fps, 8 steps, seed 43 | 512 | 367.6186 | 512.3510 | 5.3179 | 27.6250 |
| Qwen 2.1 base, 512x512, 20 steps, seed 42 | 512 | 88.4189 | 187.3762 | 6.7208 | 22.3319 |

Image: real native prompt encoding, BF16, no PAG, 4 GiB GPU weight ceiling,
24.625 GiB retained source capacity, local custom blocks and model snapshot
5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9. First-step times 35.0077/39.8262 s;
remaining-seven-step host means 9.9534/6.5847 s. The larger registration ceiling
reduced steady-state intervals by 33.8% and denoise wall by 17.9%, but first-step
time grew. Read bytes and hit/miss counts matched. Native register/unregister
accumulated durations fell from 70.7361 to 47.6829 s; these may overlap work
and must not be added to phase wall time.

Video: OzzyGT/LTX-2.3-Distilled-1.1, local blocks, real default prompt,
man_guitar.png plus diff_vocal_song.mp3 conditioning. Native encoder/transformer
ceilings 4/1.5 GiB; retained source 24.625 GiB. Feed-forward/normalization chunks
1024, phase modulation, compact timesteps, inplace residuals and RoPE reuse;
no experimental compute kernels/compilation. GPU streaming decode produced
121 frames and finite stereo 48 kHz audio. Decode-video wall 25.1970 s.
45.9523 s/step is the denoise phase average including startup, not steady-state.
This is not matched to earlier Windows/native-Linux runs without input audio.

Qwen: model snapshot d26bb61231c349cf6b7896fa83353113880e1ba3, local blocks,
default capybara prompt, BF16, CFG 1, native KV cache, 4 GiB GPU weight ceiling,
16 GiB source capacity. Twenty callbacks completed; later intervals about
3.65 s are asynchronous host observations, not GPU step timings. Denoise wall
includes CPU readback. Text encode 87.0648 s; VAE decode 1.1395 s. VAE executed
official no-stream leaf, untiled stock AutoencoderKLQwenImage21, with no local
cacheless/padding/residual optimizations. Decode GPU/RSS peaks 3.1733/3.0856 GiB.
Global RAM peak belongs to text encoding.

Media were saved and selected native faults, temporary tensors, pool failures,
quarantine and source admission denials were zero. Video encoder/transformer
and Qwen transformer snapshots also show zero live tickets/transfer-owned
events. Reports do not establish final driver-memory release or visual parity.
Local custom-block HEADs were not recorded in JSON; do not infer revision parity.

Evidence (ignored local outputs; phase tables remain in these JSON artifacts):

- outputs/ltx_image_modular_vmm/wsl_bounded_registration_20261012
- outputs/ltx_image_modular_vmm/wsl_registered512mib_20261012
- outputs/ltx_video_wsl_audio_5s_20261012
- outputs/qwen21_wsl_vmm_leaf_512_20261012
- outputs/wsl_host_registration_probe.json

The standalone probe passed 1152 MiB twice and rejected 1216 MiB at 64 MiB
resolution. The 384+512 MiB model settings leave margin below that observation;
registration churn remains significant. This closes the finite BF16 WSL sequence,
not SDNQ/FP8, arbitrary models, matched speedups over leaf, or installation delivery.

## Windows SDNQ INT4 image closure with official Diffusers 0.41.0 — 2026-10-11

The user completed the selected Qwen and LTX image comparisons in the primary
`.venv`: Python 3.12.11, Torch 2.12.1+cu132, Diffusers 0.41.0,
Transformers 5.15.1, SDNQ 0.2.7. These use normal PyTorch allocation, not
native VMM. DDO preserves SDNQ backend math and selects `sdnq_runtime`;
the controls select official streamed leaf offload in plan mode. No quantized
forward, allocator DLL or BF16 runner was changed for this comparison.

Common controls: real default prompts, no LoRA/PAG, original BF16 auxiliary
components, resident untiled stock VAE, RSS monitor sampled every 100 ms,
Windows standby purge confirmed before transformer loading. Qwen is base,
512x512, 20 steps, seed 42, CFG 1, native KV cache. Its transformer is
`OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit` at
`59d4c3610379d91fb023c4d0b6527c76755491c4`, with dynamic SDNQ/Hadamard.
LTX image is 1280x704, eight steps, seed 43, using
`elismasilva/ltx2.3-image-distilled-1.1-sdnq-int4` and the existing local
custom transformer. No revision field was recorded for that LTX checkpoint;
do not invent a pin or a matched source-revision guarantee.

| Selected workload / transformer route | Denoise wall s | Phase-average s/step | Tracked total s | Global absolute GPU peak GiB | Global RSS peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen INT4 / DDO packed runtime | 18.8056 | 0.9403 | 36.6166 | 3.6831 | 19.0167 |
| Qwen INT4 / official streamed leaf | 24.2629 | 1.2131 | 39.6060 | 3.6831 | 19.0174 |
| LTX image INT4 / DDO packed runtime | 11.7399 | 1.4675 | 46.5646 | 7.1031 | 25.9857 |
| LTX image INT4 / official streamed leaf | 21.5413 | 2.6927 | 50.0533 | 7.0979 | 25.9418 |

Denoise decreased by 22.5% in Qwen and 45.5% in LTX image for these
single-run DDO-versus-leaf pairs. This does not establish repeated-run
statistics or a universal backend speedup. Qwen wall timing includes final CPU
latent readback. LTX records its existing modular-call wall event; callback
intervals are asynchronous and are not synchronized steady-state timings.
Tracked totals include loading, attachment, purge, decode and save, but not
a separate first download or complete process startup.

Qwen's denoise-only sampled absolute GPU peak is 2.8745 GiB for DDO versus
1.9995 GiB for leaf; denoise RSS is 7.1870 versus 7.6030 GiB. The same
global GPU peak is set by later VAE work, and global RSS by encoder cleanup.
DDO trades higher transformer-phase GPU residency for speed here.
LTX's legacy generation phase aggregates connectors, loading, purge, attachment
and denoise: absolute GPU peaks are 7.1031/7.0979 GiB and RSS peaks
17.5173/11.8542 GiB (DDO/leaf). These are not isolated transformer compute
peaks. Its decode-phase GPU/RSS peaks are 4.9702/11.4389 versus
2.7964/13.1953 GiB, showing that retained normal-allocator memory can affect
the later phase even though both VAEs are resident. Do not interpret GPU usage
as quantized weight size or add per-phase peaks together.

All recorded memory-monitor errors are null; purge events took Qwen
1.6166/1.7912 s and LTX 2.3656/2.6401 s (DDO/leaf). Both Qwen JSONs
report `completed`, 20 executed steps and no error. LTX's older schema has no
status field; eight callback entries, completed save metrics and valid 1280x704
PNGs establish the selected completion, rather than treating a missing field
as an explicit success flag.

DDO and leaf produced byte-identical PNGs within each quantized model:

- Qwen: `a42313df0e4163f6c112b3ce24b0e7076e5780a9fe9b0330d8ec3651cce97eba`.
- LTX image: `d8ed344a244f150dc36d0c3cc6f2c8b7a84eb08d6c058cbf6f28d2a8bb34cff2`.

The images were visually inspected and are plausible. This proves route parity
for the selected SDNQ outputs, not quantized-versus-BF16 numerical or perceptual
equivalence. Raw JSONs and images remain unchanged.

Recent BF16 native-VMM references on the same official package remain separate:
Qwen denoise 17.6735 s / total 36.6395 s; LTX image denoise 13.7679 s /
total 35.2343 s. Thus current LTX INT4 DDO denoise is about 14.7% shorter
than that BF16 observation, unlike the older INT4 result of 16.4873 s, but
the full INT4 runner is slower. Encoder/offload and setup paths differ from
native BF16; this is not an isolated quantization comparison. Historical
results are preserved and must not replace these current package observations.

Evidence under primary-host `outputs/`:

- `qwen21_sdnq_int4_ddo_512/metrics.json` and `image.png`.
- `qwen21_sdnq_int4_leaf_512/metrics.json` and `image.png`.
- `ltx_image_sdnq_int4_ddo_041/metrics/ltx23_image_distilled_modular_sdnq_ddo_runtime_sdnq4_text_encoder_original_nopag_nolora_1280x704_steps8_seed43.json` and the same-stem PNG in its parent.
- `ltx_image_sdnq_int4_leaf_041/metrics/ltx23_image_distilled_modular_sdnq_ddo_runtime_sdnq4_text_encoder_original_nopag_nolora_1280x704_steps8_seed43.json` and the same-stem PNG in its parent.

This closes the selected Windows INT4 image DDO/leaf comparisons, not INT8,
quantized encoders, Turbo, all recipes/models, Linux/WSL or a new video gate.
No further CUDA run or budget tuning follows automatically.

## Qwen-Image 2.1 cuBLAS cleanup before leaf VAE — 2026-10-11

User completed outputs/qwen21_vae_leaf_blas_trim_enabled with the same
build-idle-trim-cu128-20261011 candidate and 1024x1024/base BF16/20-step/seed-42
workload as the trim pair below. Added --clear-cublas-workspaces-before-vae
to --release-native-cache-before-vae; official nonstreamed leaf VAE, cacheless
decode and residual reuse remained enabled. No tiling, direct padding or extra
standby purge; the before-transformer standby control was unchanged.

| Observation | Trim only | cuBLAS cleanup + trim |
| --- | ---: | ---: |
| VAE-load sampled absolute driver GPU peak, GiB | 2.2378 | 1.0776 |
| Decode sampled absolute driver GPU peak, GiB | 3.9272 | 4.7456 |
| Decode wall time, s | 1.8943 | 2.1736 |
| Cleanup phase wall time, s | 0.1100 | 0.1201 |

Before cuBLAS cleanup, one active 8,519,680-byte workspace retained
1,247,805,440 native mapped bytes. Clearing it reduced all native mapped/live/
pending/quarantined gauges to zero, without failures. The subsequent explicit
trim released nothing further: normal retirement already freed the backing.
This confirms the workspace source of retention in this selected real workload.
Zero native pool bytes do not imply zero driver/context VRAM or host source RAM.

PNG bytes match SHA256
1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45.
Twenty steps completed without monitor/native failures, mismatches or temporary
tensors. Denoise 51.5893 s, separate VAE load 0.8326 s, decode RSS peak 4.4638
GiB, total 72.1282692 s. Full cleanup worked but did not produce the lowest decode
peak/time. Keep these explicit options; do not promote an automatic default or
claim that a lower pre-VAE baseline guarantees lower activation-time peaks.
Single sequential observations; no general model/platform acceptance.

## Qwen-Image 2.1 native idle trim before leaf VAE — 2026-10-11

User completed qwen21_vae_leaf_trim_control and qwen21_vae_leaf_trim_enabled
under outputs, both with candidate build-idle-trim-cu128-20261011 DLL SHA256
468A854EB550C6299C3DE785B65ECD4F4DF382C488B6F1C83AA6899983F12CFE.
Same base BF16 1024x1024, 20 steps, seed 42, capybara prompt, CFG 1, native KV,
4 GiB weight / 0.625 GiB staging / 16 GiB source, RSS, cacheless/residual VAE,
official nonstreamed VAE leaf, no tiling/direct padding. Purge before transformer
only. The second case additionally enabled --release-native-cache-before-vae.

| Observation | No explicit trim | Explicit trim |
| --- | ---: | ---: |
| Decode wall time, s | 1.8069 | 1.8943 |
| Decode sampled absolute driver GPU peak, GiB | 5.0542 | 3.9272 |
| Decode sampled process RSS peak, GiB | 4.4630 | 4.5544 |
| VAE load wall time, s | 0.8490 | 0.8469 |
| Denoise wall time, s | 52.1545 | 52.8167 |
| Tracked total, s | 71.3454 | 72.5433 |

Trim phase took 0.1100 s. Both completed 20 native plan passes without
monitor/native failures, mismatches or temporary tensors. Both PNG bytes
match accepted SHA256
1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45.
Decode sampled peak fell 1.1270 GiB, but this is NOT the amount freed directly:
native trim snapshots show only 2,097,152 bytes released (1,247,805,440 to
1,245,708,288 mapped bytes). One live 8,519,680-byte lease pins the remaining
whole backing; it is correctly preserved. No quarantine/failures. The snapshot
does not identify this lease's owner; investigate it before declaring complete
transformer cleanup or causal peak savings. Single sequential measurements,
no automatic default promotion, no forced live release or VAE VMM acceptance.

## Qwen-Image 2.1 VAE official leaf offload comparison — 2026-10-11

User completed outputs/qwen21_vae_leaf_20steps, repeating the accepted base
BF16 1024x1024, 20-step, seed-42 workload below and adding --vae-leaf-offload.
Cacheless decode and residual aliasing remained enabled; tiling and direct
padding remained disabled. Official Diffusers VAE leaf offload used CPU storage,
CUDA onload and use_stream=false, without a warm-up decode. Same preserved DLL,
4 GiB weight / 0.625 GiB staging / 16 GiB source cache, RSS monitoring and
successful standby purge before transformer. No purge before VAE was requested.

| Decode observation | Resident VAE | VAE leaf |
| --- | ---: | ---: |
| Wall time, s | 1.2704 | 1.8115 |
| Sampled absolute driver GPU peak, GiB | 6.9356 | 5.0542 |
| Sampled process RSS peak, GiB | 4.5016 | 4.4934 |
| Separate VAE load, s | 0.9083 | 0.8647 |

Leaf reduced sampled decode GPU peak by 1.8814 GiB (27.13%) while adding
0.5411 s (42.59%) to decode in this pair. Denoise was 51.3993 s and tracked
total 70.6702058 s; total-time variation must not be attributed to VAE offload
alone. Image PNG bytes exactly match the accepted resident and original native
20-step references: SHA256
1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45.
Completed 20 native plan passes, with no monitor errors, native failures,
plan mismatches or temporary tensors; live tickets/transfer-owned events were
zero at collection. Selected workload only, not general shape/model support.
Keep leaf opt-in as a memory/time tradeoff, not an automatic faster default.

## Qwen-Image 2.1 accepted untiled VAE: 20-step closure — 2026-10-11

User completed outputs/qwen21_vae_accepted_20steps: base BF16, 1024x1024,
20 steps, seed 42, the capybara prompt below, CFG 1 and native KV cache.
Executed local cacheless decode and residual aliasing, with tiling and direct
padding disabled. Same preserved build-joined-spans-cu128-20261006 DLL,
4 GiB weight budget, 0.625 GiB staging, 16 GiB source cache and RSS monitoring;
Windows standby purge succeeded. Used the host .venv and local Diffusers fork.

- Decode: 1.2704 s, sampled absolute driver GPU peak 6.93563461304 GiB,
  sampled process RSS peak 4.5016 GiB; separate VAE load: 0.9083 s.
- Denoise through readback: 51.7984 s; tracked total: 70.5770291 s.
- Completed 20 native execution-plan passes; no monitor errors, native fault,
  admission, pool or read failures, plan mismatches or temporary tensors.
  Live source tickets and transfer-owned events were zero at collection.
- PNG is byte-identical to outputs/qwen21_base_vmm_1024_20steps/image.png:
  SHA256 1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45.

This closes the selected base BF16 untiled VAE gate without image changes.
Options remain explicit; Turbo, other resolutions and arbitrary models are not
validated by this case. Historical original decode timings include VAE loading
and use different monitoring/lifecycle controls: do not compare them as matched
isolated decode speedups or attribute the entire total difference to the VAE.
Direct padding remains rejected. No DDO, DLL or installed Diffusers changes.

## Qwen-Image 2.1 direct convolution padding: rejected — 2026-10-11

User ran outputs/qwen21_vae_padding_untiled_gate with the same base BF16
1024x1024, two-step, seed-42 untiled workload as the accepted residual-reuse
case below, adding only --vae-direct-padding. Executed controls confirm all
three local options, no tiling, the preserved DLL, and successful standby purge.
Run completed two steps. Decode was 1.1789 s, sampled absolute driver GPU peak
6.20849609375 GiB, and sampled process RSS peak 4.5002 GiB; tracked total
23.7383171 s. Prior accepted decode was 1.2658 s / 6.90600204468 GiB.

Candidate PNG SHA256:
6104F9E3F6C94424EFAB3F550952A319D628B0515CD68AD6BE254823AF08A1A0.
Unlike the accepted reference B012FF78DEA49C4AD85E26C0F8906833EF8F2A3F0B0673CC4FB3593E93BB4209,
decoded RGBA pixels differ: 140,735 changed pixels, 155,685 changed channels,
maximum absolute 8-bit channel difference 13, mean difference 0.03920841217
across all RGBA channels. Both images are 1024x1024. This is not merely PNG
metadata variation. Perceptual degradation is not established, but the exact
image-preservation gate failed: reject direct padding and leave it disabled.
Keep the accepted temporal-cache removal and residual reuse unchanged.
Single sequential timing/memory observations are not general performance claims.

## Qwen-Image 2.1 untiled VAE residual input reuse — 2026-10-11

User ran the same two-step 1024x1024 cacheless workload recorded below,
additionally enabling --vae-alias-residuals. All other requested controls and
executed DLL/weight/staging/source capacities matched. Standby purge succeeded;
tiling remained disabled. The local blocks reuse immutable residual input and
share child modules/parameters; FP32 normalizations and upsampling are unchanged.

| Decode phase observation | Cacheless | Cacheless + residual reuse |
| --- | ---: | ---: |
| Wall time, s | 1.3127 | 1.2658 |
| Sampled absolute driver GPU peak, GiB | 7.6955 | 6.9060 |
| Sampled process RSS peak, GiB | 4.4999 | 4.5004 |

Candidate completed, with no monitor/native failures, temporary fallback or
live source tickets at collection. PNG SHA256 matches the protected untiled
reference: B012FF78DEA49C4AD85E26C0F8906833EF8F2A3F0B0673CC4FB3593E93BB4209.
Artifacts: outputs/qwen21_vae_alias_untiled_gate. This is selected workload
acceptance, not general VAE support. The 0.0469 s timing difference is small
and not a proven repeatable speedup. Direct convolution padding was not enabled;
it is a separate pending CUDA candidate. No default, DDO or DLL was changed.

## Qwen-Image 2.1 untiled VAE temporal-cache removal — 2026-10-11

User ran base BF16 1024x1024, two steps, seed 42, the capybara prompt used
below, CFG 1 and native KV cache. Both runs used staged VMM with leaf encoder,
4 GiB weight budget, 0.625 GiB staging and 16 GiB source cache, RSS monitoring
and a successfully executed before-transformer standby purge. Both explicitly
disabled tiling. Diffusers source was the local fork; the recorded native DLL
was build-joined-spans-cu128-20261006/Release/ddo_vmm.dll (preserved hash below).
Only the second run enabled the local --vae-no-temporal-cache option.

| Recorded phase observation | Original native VAE | Local cacheless VAE |
| --- | ---: | ---: |
| Decode wall time, s | 5.9766 | 1.3127 |
| Decode sampled absolute driver GPU peak, GiB | 7.9995 | 7.6955 |
| Decode sampled process RSS peak, GiB | 6.7512 | 4.4999 |
| Tracked total, s | 29.6231 | 24.6114 |

Both completed with no monitor/runtime failures or temporary native fallback;
PNG files are byte-identical, SHA256
B012FF78DEA49C4AD85E26C0F8906833EF8F2A3F0B0673CC4FB3593E93BB4209.
Artifacts: outputs/qwen21_vae_native_untiled_gate and
outputs/qwen21_vae_cacheless_untiled_gate. Decode timing includes native
postprocessing/CPU image materialization, but not VAE load; sampled driver
usage is not a tensor-allocation census. Single sequential observations:
do not attribute every total-time difference to the cache change.
The local override omits single-frame temporal feature retention; weights,
normalization FP32, upsampling, clamp and postprocessing are unchanged.
Residual aliasing is a later separate candidate, not enabled in this pair.

## Qwen full-runner reporting/lifecycle revision — 2026-10-11

The full linear, native VMM and conventional leaf runners now share RunTracker
reporting through a small lifecycle helper. The staged runners explicitly drop
conditioning/preparation/component references at phase boundaries, retain only
the CPU latent for decode, and expose optional Windows standby purge. RSS is
the new full-runner monitoring default; optional CUDA boundary timing does not
synchronize every step. The conventional control remains full-pipeline, but
its new callback observations are asynchronous. These differences are recorded
as reporting revision 2; do not overwrite or reinterpret the older artifacts
below. The minimal runner remains a separate didactic example.

26 CPU contracts passed; CUDA revalidation of this runner revision is pending.
No DDO/DLL/model-math change or new performance claim is part of this revision.
See [Qwen runner guidance](../docs/qwen-image-21.md) for phase and metric scopes.

## Qwen-Image 2.1 base BF16: native VAE tiling at 1024x1024 — 2026-10-11

User repeated the native VMM 1024x1024/20-step/seed-42 workload below with
only `--vae-tiling` added and a separate output directory. Same prompt,
BF16 checkpoint, native forward, weight/staging/source ceilings and encoder
leaf route. The runner calls the native `vae.enable_tiling()`; no custom
decoder implementation or DDO/allocator change was made.

| Recorded measurement | Without tiling | Native tiling |
| --- | ---: | ---: |
| Decode phase including load, s | 7.7426 | 4.9448 |
| Sampled absolute driver GPU peak during decode, GiB | 7.9995 | 2.5815 |
| Baseline-relative sampled GPU peak during decode, GiB | 6.9727 | 1.5547 |
| Sampled process USS peak during decode, GiB | 6.7645 | 5.6412 |
| Tracked total, s | 85.1288 | 84.2664 |
| Denoise through CPU readback, s | 59.8853 | 58.0559 |

Both GPU baselines were 1.026855469 GiB. Decode absolute peak fell by
5.4180 GiB (67.73%); recorded decode phase was 36.14% shorter.
Single sequential observations include loading and monitoring; do not
attribute encoder/denoise variation or total-time changes to VAE tiling.
The tiled run completed 20 steps; all monitor errors are null.
Visual inspection found similar composition and no obvious tile seams at
the inspected display scale, not a quantitative/perceptual parity guarantee.
Tiled PNG differs as expected; SHA256:
`22149AA6E2729026443605FA4D7EE4254F7492E1FE1F5FA5DF317F3D5D07CD18`.
Artifacts: `outputs/qwen21_base_vmm_1024_20steps_tiling/image.png` and
`metrics.json`. Native tiling is a useful selected-workload low-VRAM
option; no runner defaults were changed.

## Qwen-Image 2.1 base BF16: linear versus VMM at 1024x1024 — 2026-10-11

User-run linear follow-up uses `run_dynamic_minimal.py`, `--ddo-preset auto`,
the same base checkpoint/prompt, 1024x1024, 20 steps and seed 42 as the
native VMM case below. Same root environment/process-scoped Diffusers source;
no VAE tiling or new DDO/allocator change. Linear selected 4.875 GiB resident
module weights and 8.3462 GiB pinned CPU weights. Text encoder remains leaf
compatibility in both routes.

| Recorded measurement | Linear modular | Native VMM modular |
| --- | ---: | ---: |
| Tracked total, s | 93.9298 | 85.1288 |
| Encoder phase including load/cleanup, s | 13.2328 | 13.9221 |
| Transformer phase including load/setup/cleanup, s | 69.5129 | 63.1961 |
| Decode phase including load, s | 10.7350 | 7.7426 |
| Sampled absolute driver GPU peak in transformer, GiB | 7.0983 | 6.3183 |
| Sampled absolute driver GPU peak in decode, GiB | 7.9995 | 7.9995 |
| Sampled process USS peak across phases, GiB | 20.8134 | 19.0192 |

Both completed and produced byte-identical PNG files, SHA256
`1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45`.
Linear artifact: `outputs/qwen21_base_linear_1024_20steps/image.png` and
`metrics.json`. VMM artifact is recorded below. All linear monitor errors
are null. Linear has no separate denoise timing; do not derive its s/it by
dividing the transformer phase by 20.

VMM total was 8.8011 s (9.37%) lower and its transformer phase was 6.3168 s
(9.09%) shorter. These single sequential observations combine runtime/setup,
monitoring and system/cache-state differences, not an isolated steady-state
speedup estimate. Full USS monitoring remained enabled. Both native decodes
reached approximately the device's entire 8 GiB in sampled absolute driver
usage; no increased-resolution or extra benchmark approval is implied.

## Qwen-Image 2.1 base BF16: native VMM 1024x1024 — 2026-10-11

User-run follow-up to the 512x512 comparison below. Same base checkpoint,
prompt, seed 42, BF16, 20 steps, native model forward/KV cache, staged modular
runner, source/environment and selected DLL. Only requested width/height and
output directory changed. Weight-residency ceiling remains 4 GiB, total
staging 0.625 GiB, native source cache 16 GiB. Encoder remains leaf compatibility.
No VAE tiling or explicit standby purge was requested.

| Recorded measurement | VMM 512x512 | VMM 1024x1024 |
| --- | ---: | ---: |
| Tracked total, s | 43.6096 | 85.1288 |
| Encoder phase including load/cleanup, s | 10.9224 | 13.9221 |
| Transformer phase including load/setup/cleanup, s | 30.2419 | 63.1961 |
| Denoise through CPU latent readback, s | 26.5746 | 59.8853 |
| Decode phase including load, s | 2.1508 | 7.7426 |
| Sampled absolute driver GPU peak during transformer, GiB | 6.3578 | 6.3183 |
| Sampled absolute driver GPU peak during decode, GiB | 3.7280 | 7.9995 |
| Sampled process USS peak across phases, GiB | 19.0198 | 19.0192 |

Generation completed with 20 native execution-plan passes. The collected
transformer snapshot reports zero fault/guard/CUDA/capacity admission failures,
execution-plan mismatches, temporary tensors, pool failures/quarantine,
source-read failures and cache admission denials. Source-cache hits/misses
remain 6,671/213; source-read bytes 14,230,284,408 and DMA bytes
222,076,338,176 match the 512x512 case. These counters do not imply identical
compute costs. Live tickets and transfer-owned events are zero at collection,
not a post-exit allocation census. All phase monitor errors are null.

Later asynchronous host callbacks were mostly 2.72-2.82 s (step 3: 2.3068 s);
these are not CUDA-event step timings. The 59.8853 s denoise metric includes
final CPU readback. No matched linear or conventional 1024x1024 run exists yet.

Visual inspection found a coherent 1024x1024 image matching the prompt.
Its SHA256 is
`1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45`.
Different resolution means no pixel-parity comparison to the 512x512 output.
Artifact: `outputs/qwen21_base_vmm_1024_20steps/image.png` and `metrics.json`.
Raw candidate labels remain unchanged.

Selected workload functionality passes. The native VAE decode reached
approximately the entire 8 GiB device in sampled absolute driver usage.
This is a capacity warning, not an OOM or proof of shared-WDDM spill.
Do not infer spare decode capacity or authorize higher resolution from this
result. Native tiling could be evaluated separately if a safer decode route
is needed; it is not enabled automatically. Full USS monitoring is retained;
its query time is not subtracted from generation time.

## Qwen-Image 2.1 base BF16: linear, official leaf and native VMM — 2026-10-11

User-run Windows observations from the primary host root `.venv`.
Workload: `Qwen/Qwen-Image-2.1`, BF16, text-to-image, 512x512, 20 steps,
seed 42, one image, CFG scale 1, native KV cache enabled. Prompt:
"A capybara wearing a wizard hat, reading by candlelight, oil painting".
Linear was completed on October 10; leaf and VMM on October 11 (local time).

The process-scoped source was `F:/Projetos/diffusers/src`; the conventional
report confirms that import path. The last inspected source revision was
`7d047b18092beebdb6d2aebadf1ccc254787dbc4`, not embedded in these reports.
Model cache snapshot: `d26bb61231c349cf6b7896fa83353113880e1ba3`.
The host dependency lock and accepted LTX paths were not changed.

| Recorded measurement | DDO linear modular | Conventional Diffusers leaf | DDO native VMM modular |
| --- | ---: | ---: | ---: |
| Tracked total, s | 50.6873 | 379.3020 | 43.6096 |
| Encoder phase including load/cleanup, s | 14.7564 | unavailable separately | 10.9224 |
| Transformer phase including load/setup/cleanup, s | 30.9378 | unavailable separately | 30.2419 |
| Decode phase including load, s | 4.6695 | unavailable separately | 2.1508 |
| Load/setup, s | included in staged phases | 1.5702 | included in staged phases |
| Conventional generation call, s | not applicable | 377.6377 | not applicable |
| Denoise through CPU latent readback, s | unavailable | unavailable separately | 26.5746 |
| Sampled absolute driver GPU peak, GiB | 6.5347 | 5.0874 | 6.3578 |
| Sampled process USS peak, GiB | 21.5977 | 33.0219 | 19.0198 |

Linear: `run_dynamic_minimal.py`, `auto/one_shot_fast`,
`linear_runtime` transformer; about 4.875 GiB resident module weights and
8.3462 GiB pinned CPU weights. Encoder used official leaf compatibility.

Conventional: `run_diffusers_leaf.py`, unmodified `QwenImage21Pipeline`;
official leaf on transformer and encoder, streams enabled, record_stream false,
low_cpu_mem_usage true, initial non_blocking false. Native VAE on CUDA.
No DDO runtime, custom blocks, allocator or forward patch. It retained the
full pipeline components instead of releasing them phase by phase.
Synchronized callback intervals for steps 2-20 averaged 18.2963 s;
the 28.2732 s first interval also includes prompt encoding.

VMM: `run_dynamic_modular_vmm.py`, `vmm/auto -> vmm_fast`,
native transformer transport, 4 GiB weight-residency ceiling, 0.625 GiB total
staging, 16 GiB source-cache ceiling; registration ceiling defaults to the
storage ceiling. Encoder stayed on official leaf compatibility, not native
Qwen encoder VMM. No VAE tiling or explicit standby purge was requested.
The native binary recorded in the report resolves to
`ddo-vmm-allocator/build-joined-spans-cu128-20261006/Release/ddo_vmm.dll`.
Inspected SHA256:
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.

VMM host callback intervals for steps 3-20 averaged 0.96994 s
(range 0.94583-0.99449); they are asynchronous wall observations, NOT isolated
CUDA step measurements. Step 1 was 8.11413 s and step 2 only 0.09342 s:
do not interpret those intervals as completed GPU step durations.
The native snapshot records 20 execution-plan passes, 6,671 source-cache hits,
213 misses, 14,230,284,408 source-read bytes, zero CPU-source bindings,
zero fault/guard/CUDA/capacity admission failures, execution-plan mismatches,
temporary fallback tensors, source-read failures, source-cache admission
denials and allocator-pool failures/quarantine. Live source tickets and
transfer-owned events are zero at collection; this is not a post-exit census.

All three PNG files have identical SHA256:
`550C28BB34648ABE520D59C5C5506CB55488F221E8CCFB6F22D8E13CCD1B6CF1`.
Visual inspection found a coherent image matching the prompt. This establishes
selected-case output equivalence, not arbitrary-prompt/model support.
Raw static candidate validation labels are preserved unchanged.

The VMM tracked total was 13.96% lower than linear, but its transformer phase
was only 2.25% shorter. Most of the total difference came from encoder/decode
phases; do not attribute the full total improvement to native offloading.
The conventional total was about 7.48x the linear total and 8.70x VMM, but these
single sequential, differently orchestrated runs are not an allocator-only
or instrumentation-neutral speedup benchmark. Full USS monitoring was used
in all cases; conventional generation accumulated 326.5688 s in RAM queries
in the monitor thread. Those queries may affect execution and cannot simply
be subtracted from wall time. Normalize monitoring and timing scope before
making a causal performance claim; no extra benchmark is automatically required.

Artifacts (each directory contains `image.png` and `metrics.json`):
- `outputs/qwen21_base_minimal_20steps`
- `outputs/qwen21_base_diffusers_leaf_20steps`
- `outputs/qwen21_base_vmm_20steps`

The first offline VMM attempt failed while resolving the processor, before
transformer setup. Runners now resolve the cached processor component directory
explicitly; offline CPU loading passed before the successful generation.
DDO and allocator code were not modified for Qwen.

## User-run merged-source LTX BF16 closure — 2026-10-10

Root host .venv, PYTHONPATH=F:/Projetos/diffusers/src, HF_HOME=F:/hf_cache.
Same-terminal import verification resolved the fork source; the inspected clean
checkout remains7d047b18092beebdb6d2aebadf1ccc254787dbc4. Raw reports do not
embed a source revision. Installed manifest/lockfile remain pinned at e71ebc5.
Local image/video custom-block commits:1eba529/a820889; no remote promotion.

The first image attempt, outputs/merge_gate_image, used build/Release/ddo_vmm.dll,
inspected SHA2564AC3D8B5396E0CD47A5DA0F08560EA0C6783697153BD1072B23DFB7F57CA7AA3.
It completed an identical PNG but recorded1.690625s CUDA warm mean,17.016760s
CUDA phase and44.1937s tracked total, with three purge events totaling4.9727s.
The proposed command had omitted explicit DLL selection. This is not a matched
historical binary/fork comparison: DLL and purge phases differ. An interrupted
video attempt likewise is not a completed gate; teardown retention messages
alone do not prove an inference failure or permanent process hang.

### Corrected fixed-DLL runs

The user executed the corrected commands presented in the conversation:
image script scripts/ltx2_image/run_dynamic_modular_vmm_distilled.py;
video script scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py.
BF16 encoder/transformer, real default prompts,1280x704,seed43,eight steps.
Image: one generation,no PAG,no fake prompt,public native runtime,CUDA event
timing,metrics level2,RSS monitoring. Video: real default image conditioning,
no input audio,generated output audio,5seconds at24fps/121frames,local video
custom-block clone,FF/norm chunks1024,phase/compact modulation,in-place
residuals,rotary reuse,host-cache release and CUDA streamed decode.
Optional compilation/kernels are off; no calibration,budget increase or rebuild.

Image encoder/transformer ceilings4GiB; staging CLI0.5 resolves to total1.5GiB.
Video encoder/transformer ceilings4/1.5GiB,total staging1.5GiB.
Both source/registration ceilings24.625GiB. SSLKEYLOGFILE removed from the
test terminal. DDO_PURGE_WINDOWS_STANDBY_*: BEFORE_RUN=0,
AFTER_TEXT_ENCODER=0,BEFORE_TRANSFORMER=1,AFTER_RUN=0. Only the requested
before-transformer purge executed, with successful privilege/event evidence.

Image explicitly selected DDO_VMM_LIBRARY_PATH:
E:/ProjetosIA/diffusers-dynamic-offloader/ddo-vmm-allocator/build-lifecycle-off-20261002/Release/ddo_vmm.dll.
Inspected SHA256:755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729.
Video explicitly selected both environment and --ddo-vmm-library-path:
E:/ProjetosIA/diffusers-dynamic-offloader/ddo-vmm-allocator/build-joined-spans-cu128-20261006/Release/ddo_vmm.dll.
Inspected SHA256:EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.

| Measurement | Previous selected reference | Merged-source gate |
| --- | ---: | ---: |
| Image CUDA warm steps2-8, s/it | 1.387940 | 1.370488 |
| Image CUDA phase with dependencies, s | 15.397565 | 14.871349 |
| Image tracked total, s | 35.2571 | 38.4748 |
| Video denoise, s | 199.3626 | 200.9234 |
| Video decode/save, s | 26.8720 | 27.1645 |
| Video tracked total, s | 262.4741 | 264.2061 |

Image warm mean is1.26% lower; total includes2.1682s successful purge, unlike
the prior post-reboot image reference with skipped purge. Video denoise is0.78%
longer and total0.66% longer,including2.4353s purge. These single sequential
observations are not statistical equivalence or causal fork/DLL/purge evidence.
Sampled global absolute GPU peaks: image6.887031555GiB,video5.142089844GiB;
sampled RSS peaks: image27.3697GiB,video28.418830872GiB. These are not
physical model residency or shared-WDDM measurements.

Both component snapshots report zero selected fault/guard/CUDA/capacity/read/
cache-admission/pool failures,plan mismatches,temporary fallbacks and quarantine;
final live tickets and transfer-owned events are zero. Image encoder collection
38->0 and transformer711->0; video encoder9->0 and transformer0->0.
Video records completed status,eight native execution-plan passes,121frames at
1280x704 and finite stereo audio[2,240480] at48000Hz. All phase monitor errors
are null. Snapshots are not a post-exit allocation census.

PNG SHA256:877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED.
MP4 SHA256:56BD4CF1D6F20CD2C978FDA6280902197A24FC179E303AC98F65EE02EB7FD891.
Each matches its earlier selected primary-host artifact, not arbitrary-model
quality acceptance. Artifacts: outputs/merge_gate_image_fixed_dll (PNG and
metrics/run-slug JSON); outputs/merge_gate_video_fixed_dll (MP4 and
metrics/modular_run.json). User report IDs:d4a4d3d6-9ddb-4c23-833a-83e0e58cc7a6
and4231dc4b-eaba-42d8-a3e2-5fab0d341bf5. Original wrong-DLL report ID:
845f0387-e0c9-427b-89eb-3c47b8dd28a4.

Selected merged-source BF16 image/video CUDA closure passes. This does not
promote dependencies,accept FP8/SDNQ on the new fork,optional keyframes,arbitrary
models or Linux. The video's raw static validation label still says CUDA
pending; preserve it unchanged as metadata, superseded for this workload by
completed media/phase evidence. No extra CUDA test follows automatically.

Start with the [benchmark summary](../docs/benchmark-summary.md) for selected
completed comparisons and current recommendations. This page is the detailed
engineering evidence log: historical pending statements, interrupted tests and
rejected candidates are preserved, not a ranked list of recommended settings.

## Evidence ownership — 2026-10-10

This repository owns pipeline benchmarks and results across models.
The records below were transferred from DDO without changing measurements,
raw evidence or historical acceptance boundaries. Historical host paths and
dated pending statements describe their original checkpoints, not current
acceptance in this repository. DDO retains runtime/allocator contracts and
technical gates. New records must identify model, precision, conditioning,
dimensions/frame rate, steps, environment/revisions, binary identity, timing
scope, memory definitions and artifact paths. Missing metadata is unavailable,
not zero. See the [benchmark guide](../docs/benchmarks.md).

Selected development acceptance, updated 2026-10-04. DDO is model-agnostic;
LTX is the benchmark workload, not a library dependency or architecture policy.
Intermediate validation runs are retained locally in the internal implementation
report, separately from this public summary. Recorded measurements and raw
benchmark artifacts are unchanged.

## Windows image host migration gate — 2026-10-10

One approved BF16 image generation completed in the independent `ddo-examples`
uv-managed `.venv`: real prompt, 1280x704, seed43, eight steps, no PAG,
native encoder then native transformer. Preserved active4GiB, total staging1.5GiB,
source/registration24.625GiB and candidate DLL SHA256
`755fcdd8dd1a72b371402750c071ca4b2146c0ac6bfe6b94c9b78f50f1ebb729`.
No native build, library promotion, budget or runtime-default change.

| Measurement | Historical staged Windows gate | New host, single run |
| --- | ---: | ---: |
| CUDA warm steps2–8 | 1.395602 s/it | 1.524931 s/it |
| CUDA transformer phase with dependencies | 16.498256 s | 18.657049 s |
| Encoder phase | 12.6624 s | 20.6153 s |
| RunTracker total | 38.5135 s | 50.7519 s |
| Sampled peak RSS | 27.3479 GiB | 27.4590 GiB |

Functional migration **PASS**: exit0, saved plausible 1280x704 image; selected
fault/admission/read/plan-mismatch/pool-quarantine counters zero, temporary
fallbacks zero, encoder tickets31→0 and transformer736→0, transfer-owned
events zero at final snapshots and all phase monitor errors null.
Before denoise, encoder source process committed/registered bytes are zero.
Current sampled driver GPU absolute peak7.716034GiB is distinct from
baseline-relative delta6.6892GiB (baseline1.026855GiB). This is not WDDM shared
memory accounting and is higher than the historical approximately6.97GiB peak.

Do not accept performance parity from this run: warm timing is approximately9.27%
higher, total approximately31.78% higher. The historical gate is a reference,
not a contemporaneous matched control; the timing difference's cause is not
established. No pixel-equality comparison or repeated benchmark was performed.
Artifacts remain ignored under `ddo-examples/outputs/migration_image_gate/`;
PNG SHA256 `877876a4a8e1143a59e51a737ef3eb7f05a27980a6a44d20fed80d2ab581cced`.
No further CUDA run is authorized by this result.

## Video compact timestep / residual CPU checkpoint — 2026-10-07

The sibling video host and editable local Diffusers fork now expose separate,
default-OFF compact timestep and inference-owned residual-reuse options.
Consolidated CPU regression sequence: **147 passed, 11 subtests passed**.
Compact modulation expansion and residual reuse pass exact small FP32/BF16
checks; changed projection dimensions have a separate tiny CPU arithmetic bound,
not approved real-model CUDA equivalence. Admission uses prepared CPU masks,
retaining dense execution for spatial variation and reference-layout mismatches.
No CUDA run, allocator build, budget/default change or performance acceptance
followed. Native normalization/modulation and paired RoPE fusion remain pending.
Host implementation and limitations: `E:\LTX-2\model_execution_optimizations.md`.

## Video feed-forward chunk A/B — 2026-10-07

Approved two-step diagnostic: BF16, 1280x704, 8 s at 24 fps (193 frames),
seed 43, no audio conditioning, latent output, without Nsight. Transformer
ceiling 1.5 GiB, encoder ceiling 4 GiB, source cache/registration 24.625 GiB,
staging 1.5 GiB, phase-local modulation enabled and normalization chunk 1024.
Only the feed-forward chunk changed; DLL and other requested controls matched.

| Observation | FF chunk 1024 | FF chunk 2048 |
| --- | ---: | ---: |
| CUDA step 1 | 52.8863 s | 50.2793 s |
| CUDA step 2 | 44.6857 s | 42.6468 s |
| CUDA denoise phase | 97.5960 s | 92.9493 s |
| Total, latent output | 136.0103 s | 130.9887 s |
| Sampled absolute GPU peak | 7.9995 GiB | 7.9995 GiB |

Second-step improvement was **4.5629%**, below the predeclared 5% threshold.
CPU comparison found finite but nonidentical video/audio latents; relative RMS
differences were **4.8191% / 2.5589%**. No numerical tolerance was approved;
do not claim equivalence or rendered-quality regression from these differences.
Both runs exited 0, with selected native failure/quarantine/fallback counters
zero and diagnostic events released. **Not accepted for promotion**: preserve
chunk 1024, existing binaries, defaults and frozen routes. This single ordered
pair does not establish stable performance or eight-step/video-output parity.
Artifact paths and numerical details: [auxiliary gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md).

## Video phase-local modulation diagnostic — 2026-10-06

Approved bounded Windows comparison: BF16, 1280x704, 8 s at 24 fps, seed 43,
two denoise steps, no audio conditioning, latent output. Same joined-span DLL,
2 GiB transformer/4 GiB encoder ceilings and 1024 FF/norm chunks; only the
local Diffusers `--phase-modulation` option changed. This is not the image
benchmark below or an eight-step/full-decode video benchmark.

| Observation | Original modulation | Phase-local modulation |
| --- | ---: | ---: |
| Denoise CUDA phase | 300.5835 s | 98.6875 s |
| CUDA step 1 / 2 | 104.2100 / 196.2883 s | 49.8097 / 48.8539 s |
| Total, latent output | 339.5551 s | 138.3953 s |
| Cumulative observed native activation-pool peak | 6.1523 GiB | 5.1445 GiB |

Completed with exit 0, no selected failures/quarantine/temporary fallbacks,
retired source tickets and released diagnostic events. Saved video/audio
latents are finite and bit-exact against the control, with identical artifact
SHA256. Sampled GPU baseline-relative peak remains 6.9727 GiB: pool reduction
is not a claim of reduced whole-GPU peak. Defaults and image/linear/NVMe paths
remain unchanged; option remains explicit. Two steps do not establish settled
steady-state or eight-step performance. Full hashes, paths, scope and evidence:
`docs/native-execution-auxiliary-gate.md`.

The separately approved **eight-step** continuation completed with exit 0:
denoise **454.5071 s**, latent-output total **494.8370 s**, cumulative native
pool peak **5.1445 GiB**. CUDA steps: **47.28 / 47.42 / 50.87 / 43.62 / 44.17 /
46.55 / 129.36 / 45.19 s**. Steps 2–8 mean **58.17 s**, median **46.55 s**;
the seventh-step outlier prevents approximately 49 s/step stability acceptance.
All saved latents are finite, final selected safety counters are zero and
retirement completed. This is not decoded-video quality, eight-step bit-exact
parity or a matched comparison against historical MP4 totals. No new test or
default promotion followed; evidence is in the same gate document.

A later separately approved video-output attempt was interrupted 120 s after
its eighth host callback, without an MP4 or final timing/cleanup report. Its
command inadvertently selected CPU/non-streaming decode instead of the prior
validated streaming CUDA decode; the finalization timeout is not a matched
regression. An independent observer recorded approximately 26.5–26.8 GiB of
worker shared GPU memory, sustained graphics/memory clocks and lower power
during slow intervals. This does not prove WDDM paging causality. No default
or allocator change was made; full-video acceptance remains open. Evidence
and the command-level correction are recorded in the same gate document.

The separately approved corrected streaming-CUDA MP4 repetition completed
with exit 0: **697.7858 s total**, **612.0763 s CUDA denoise**, **45.0749 s
decode/save**. CUDA steps: **56.35 / 46.45 / 65.25 / 122.35 / 64.32 / 49.93 /
91.32 / 116.02 s**, steps 2–8 mean **79.38 s**. The runner validated 193 frames
at 1280x704 and finite audio; selected final safety counters were zero,
tickets ended at zero and diagnostic events were released. Native pool peak
remained **5.1445 GiB**. Output/retirement completed, but denoise stability and
performance acceptance remain open; no default promotion or further execution.
Full command, artifact hash and limitations are in the same gate document.

A separately approved single comparison changed only the transformer resident
weight ceiling from **2 to 1.5 GiB**: CUDA denoise **436.0448 s**, decode/save
**42.7006 s**, total **519.4920 s**, exit 0. CUDA steps **48.28 / 44.54 / 57.53 /
112.73 / 46.47 / 41.95 / 42.16 / 42.33 s**; steps 2–8 mean **55.39 s** includes
the fourth-step outlier. Effective resident weights were approximately
**1.4766 GiB**. The MP4 hash matches the preceding 2 GiB run; selected safety
counters were zero and retirement completed. Observed denoise improved
approximately **28.8%**, but the slow step remains and one run does not prove
causality/stability. Native pool and sampled whole-run GPU peaks were unchanged.
No default promotion or further benchmark; details are in the same gate document.

User-run Nsight CUDA/WDDM diagnostic on 2026-10-07 at the same **1.5 GiB**
ceiling completed with CUDA denoise **346.2786 s**, total **430.2082 s** and
steps **41.63–45.16 s**, without reproducing the slow interval or recorded
standby purge. Saved MP4 hash matches the preceding two runs; selected safety
checks and retirement completed. This instrumented run is not a matched
benchmark or stability acceptance. CUDA/WDDM data is present, but the profiler
warns that driver 13.4 is unsupported and uses 13.2 tracing libraries.
Limitations and trace evidence are recorded in the same gate document.

User-run standby purge before the transformer was confirmed (2.9247 s) in
the subsequent instrumented run. Without/with purge: CUDA denoise
**346.2786 / 345.5177 s**, mean steps 2–8 **43.0134 / 43.0688 s**, full total
**430.2082 / 430.7291 s**. Saved MP4 hashes match, selected safety counters
were zero and cleanup completed. This pair shows no meaningful purge benefit
for this workload; neither run reproduced the slow window. Keep `vmm_fast`
standby purge disabled by default and the separate PyTorch host-cache release
intact. No further repeat or code/default change; full scope is in the gate document.

## Workload and scope

- LTX 2.3 distilled image model, BF16 transformer and text encoder.
- Real prompt, 1280×704, seed 43, eight steps, no PAG, one generation.
- Phase-staged native encoder → unload → native transformer → VAE → save.
- GPU: NVIDIA RTX 3060 Ti, 8 GiB. Active weight budget 4 GiB; native source
  storage and registration ceilings 24.625 GiB each; total staging 1.5 GiB;
  native PyTorch allocator cache 512 MiB. RAM must also fit loader/runtime
  demands and safety margins; this is not a low-system-RAM benchmark.
- Normal linear defaults and the protected NVMe route are unchanged. Native
  component ownership remains explicit opt-in, not automatic encoder support.

## Final staged observations

| Scope | Windows, 2026-10-03 | Native Linux, 2026-10-04 |
| --- | ---: | ---: |
| Encoder load / native attachment | 0.7951 / 2.5477 s | 0.6240 / 1.2323 s |
| Entire prompt phase | 12.6624 s | 11.7280 s |
| Transformer load / prepare | 0.3965 / 0.6261 s | 0.3953 / 0.7808 s |
| First transformer CUDA step | 6.729011 s | 5.788787 s |
| Steady state, CUDA mean of steps 2–8 | 1.395602 s/step | 1.050970 s/step |
| Transformer CUDA phase, including dependencies | 16.498256 s | 13.145578 s |
| Entire Generate phase, including connectors/setup | 24.2141 s | 19.0590 s |
| VAE phase / image save | 1.5570 / 0.0760 s | 1.2131 / 0.2129 s |
| RunTracker total | 38.5135 s | 32.2142 s |
| Sampled peak process RSS | 27.3479 GiB | 27.3482 GiB |
| Sampled absolute driver GPU usage | approximately 6.97 GiB¹ | 6.499573 GiB |

¹ Windows absolute use is reconstructed from delta 5.9403 GiB plus rounded
console baseline 1.03 GiB; an exact baseline was not persisted in that historical
JSON. Linux persisted baseline 0.722290 GiB and delta 5.7773 GiB separately.
Missing historical fields are unavailable, not zero.

RunTracker includes encoder, generation, VAE and saving, but excludes interpreter,
import and bootstrap startup. CUDA events measure the compute stream with
dependencies; host callbacks are not CUDA timing. First step and steady state
are separate. Nested setup timings must not be summed.

These are single staged runs on different OS/driver environments, not a statistical
cross-platform comparison or process-cold benchmark. File/driver cache state was
not equalized; no causal explanation of the timing difference is established.
Separate Windows transformer reuse validation observed 1.401990 / 1.397915 /
1.394595 s/step over three generations; it does not establish Linux repetition
or whole-pipeline repeated-generation support.

## Correctness and bounded safety

Both selected staged runs completed with real prompt encoding, native denoising,
a visually checked 1280×704 image and saved reports. No temporary tensor fallback,
source admission denial/bypass, execution-plan mismatch, allocator quarantine or
memory-monitor error was reported. Encoder source counters were zero before
transformer denoise; tickets were collected to zero at completion boundaries.

Linux transformer cache committed/registered 24.25 GiB, with 1.5 GiB staging
separate. Aggregate source reads 25,846,148,360 bytes versus 187,588,136,576 DMA
bytes show reuse, not a per-step read trace. Memory was sampled every 100 ms;
peaks are not instantaneous enforced caps or WDDM shared-memory measurements.
A plausible image does not prove cross-platform numerical/pixel equivalence.
The selected Windows encoder additionally passed its embedding/mask reference check.

Critical source, registration, event, ownership and cancellation contracts passed
in their recorded platform scopes. This is not arbitrary-model, multimodal,
multi-GPU, compiled-forward or driver-loss recovery acceptance. Quantized forwards
retain compatibility routes. No native VMM support is promised for WSL or macOS.

## Evaluation against Diffusers

No new matched official Diffusers block/leaf benchmark was run with this final
native staged encoder configuration. Earlier linear/compatibility measurements
are historical and must not be used to calculate a speedup for this version.
The table reports DDO performance, not superiority over official Diffusers.

For a matched evaluation, hold model revision, precision, prompt, seed, shape,
steps, guidance, phase staging, attention implementation and timing boundaries
constant. Record dependency/driver versions, executed component routes, memory
definitions and cache conditions. Report setup, first step, steady state,
complete denoising and full pipeline separately. Native allocator installation
requires a fresh process; label warm generations separately from cold invocations.

## Reproduction

See the [benchmark manifest](../docs/benchmark-reproduction.md) for environment,
external host fixture, model revision, commands and native binary identity.
[Runtime support](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-support-status.md) and the
[metric catalog](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/metrics.md) define integration and reporting semantics.

Python wheels do not bundle native binaries. Reproduction currently requires
the allocator submodule/platform build and sibling host fixture, not a
self-contained installed-wheel benchmark. No additional benchmark or publication
is implied by this report.

## Selected Windows video host gate (2026-10-05)

The sibling `E:\LTX-2` multimodal host completed a selected native transformer
case:1280x704, five seconds/121frames at24fps, seed43, BF16, real prompt/image/
audio conditioning and eight distilled steps. Native weight residency was2GiB;
total staging1.5GiB and retained source storage/registration24.625GiB remained
unchanged. The host explicitly enabled1024-token standard Diffusers feed-forward
and RMSNorm chunking to bound activation temporaries. These are optional host
experiments, not DDO residency heuristics, new DDO presets or image defaults.
The selected allocator binary and native implementation were unchanged.

The preceding approved four-step synchronous diagnostic completed generation,
unload, decode and save in202.3265s total (128.1s generation, diagnostic only).
The conditional eight-step run, without launch blocking, completed those same
phases in308.7771s total (236.4s generation; console timing rounded). Progress
averaged29.46s/step including startup. No speedup claim or matched image/Diffusers
comparison is made for this larger video workload.

The eight-step report recorded192materializations and zero fault failures,
temporary tensors and collected span-pool quarantine bytes. Tickets were
collected from2 to0; transformer hooks were removed before decode. The encoder
and auxiliary components retained their existing group-offload/staged routes.
Observed absolute driver usage after generation was5.18GiB (4.15GiB above the
baseline); tracker RSS observations reached29.10GiB. Generation monitoring was
disabled: these are not continuous denoise peaks or enforced caps.

MP4/JSON artifacts are in the host's
`outputs/ddo_native_video_20261005_115722_179/`; the MP4 has1280x704/121frames and
an audio stream. CPU decode of all frames passed; this does not establish visual
quality or full-model GPU numerical equivalence. Reproduction details and the
preceding failed candidates remain in the host's `ddo-native-video.md`. The
selected gate is not a promise for arbitrary video duration, shapes or models,
and the accepted image route was not modified or rebenchmarked.

### Video input-audio conditioning control (2026-10-05)

The approved eight-step control changed only `LTX_USE_AUDIO_CONDITION=0`;
the console confirmed `use_audio_condition=False`. All selected shape, seed,
native budgets, activation chunks and allocator binary settings above were
preserved. Audio-video transformer execution and generated audio remained
enabled. This removes input-audio conditioning, not the audio model branch.

| Case | Progress average | Generation phase (console) | Full run (JSON) |
| --- | ---: | ---: | ---: |
| Previous image + input-audio conditioning | 29.46 s/step | 236.4 s | 308.7771 s |
| Image conditioning, no input-audio conditioning | 27.86 s/step | 223.6 s | 297.2001 s |

The observed progress average decreased by approximately 5.4%; the generation
phase decreased by 12.8 s. These are single-run observations, not an isolated
or repeat-validated estimate of the conditioning cost. All eight steps,
hook removal, transformer unload, video/audio decode and save completed without
a reported failure. The console recorded 192 materializations, zero temporary
fallbacks, zero collected span-pool quarantine bytes and tickets collected from
2 to 0. Absolute driver usage after generation was 5.17 GiB; tracker RSS
observations reached 28.748 GiB. Generation monitoring remained disabled;
these observations do not establish continuous denoise peaks.

Artifacts: sibling host `outputs/ddo_native_video_20261005_201936_215/`, with
MP4 and JSON under `metrics/`. CPU decoding verified all 121 frames at
1280x704, a nonconstant first frame and the presence of an audio stream;
visual/audio quality and GPU numerical equivalence were not assessed.
The saved JSON does not expose the input-audio switch as a top-level field;
the executed command and console configuration establish this control's route.
No library, native allocator, preset or image runner was changed.

### Video native pool opportunistic retirement (2026-10-05)

The approved candidate changed only native activation-cache trimming: `free`
queries consumer completion and retires only fully completed backing, while
allocation-pressure retirement retains its event waits. Admission and cache
ceilings, host chunks, BF16, 1280x704, 121 frames, 24 fps, seed 43, eight steps
and the no-input-audio condition were preserved. The accepted binary and host
defaults were not replaced.

| Matched no-input-audio case | Progress average | Generation phase (console) | Full run (JSON) |
| --- | ---: | ---: | ---: |
| Accepted allocator control | 27.86 s/step | 223.6 s | 297.2001 s |
| Nonblocking opportunistic-retirement candidate | 26.09 s/step | 209.4 s | 282.7551 s |

Observed progress improved by approximately 6.4%, generation by 14.2 s and
total by 14.4450 s (4.9%). These are single-run observations, not repeat-validated
speedups. Pool eviction wait fell from 59.075447 s to 0.0001051 s cumulatively
through the report snapshot, while native batch-submit time rose from
27.2338389 s to 49.0033472 s. This changed scheduling/overlap; neither counter is
an independent denoise cost or GPU DMA elapsed time. Do not sum these counters
or claim the entire removed pool wait as elapsed-time savings.

Seven CPU contracts and six small CUDA stream/cache contracts passed before
the model run. Generation, cleanup, unload, decode and save completed with
zero reported pool failures, quarantine bytes, temporary tensors, execution-plan
mismatches and source-read failures. Materializations remained 192 and execution
buffer uploads remained 512. Source tickets were collected from 2 to 0.
Post-generation driver usage was 4.95 GiB absolute; tracker RSS observations
reached 29.0081 GiB. Generation monitoring was disabled; these are not continuous
denoise peaks. Pending pool ownership remains accounted until completion is
observed, even if the GPU already completed the work.

Candidate SHA256:
`E478BB872F593C0BCD272DFDBC9389DC430EA04951D4B95F2FF3197F08B218DE`.
Artifacts: sibling host `outputs/ddo_native_video_20261005_203826_052/`.
CPU inspection decoded 121 frames at 1280x704, one audio stream and a
nonconstant first frame. Quality and GPU numerical equivalence were not assessed.
Commands, ownership details and limitations are recorded in
`docs/native-pool-nonblocking-gate.md`. The candidate remains explicitly selected
for this gate, not promoted to host defaults; the image workload was not rerun.

### Video Diffusers activation-lifetime candidate (2026-10-05)

The source-only candidate releases consumed local activation references in
`LTX2VideoTransformerBlock.forward`, without arithmetic or attention changes.
The video environment used the editable local fork, branch `ltx-image`, HEAD
`0424acb0728f5421d20852728ae9817482a1ef20`, plus the recorded lifetime patch.
DDO dependency pins and the sibling image environment were not updated.
The transformer source before the patch matched the pinned fork; the intervening
commit changes a separate image pipeline, not this video transformer.

The approved tiny real-block CUDA gate passed exact FP32/BF16 output comparisons
with both cross-modal directions enabled, using the normal PyTorch allocator.
Its PyTorch allocated peak was 8596992 bytes. The conditional full video used
the same native pool candidate and all conditions/budgets from the preceding
no-input-audio case. This separates GPU arithmetic checks from native lifetime
integration; neither establishes general image/model non-regression.

| Case | Progress average | Generation phase (console) | Full run (JSON) |
| --- | ---: | ---: | ---: |
| Native pool candidate, retaining forward | 26.09 s/step | 209.4 s | 282.7551 s |
| Native pool candidate, early-reference-release forward | 29.44 s/step | 236.1 s | 310.9901 s |

The observed denoise average was 12.8% slower and total increased by 28.2350 s.
This candidate did not meet the selected performance criterion and is not
recommended for promotion on this evidence. It completed generation, cleanup,
decode and save with zero reported pool failures, quarantine bytes, temporary
tensors, source-read failures and execution-plan mismatches. Materializations
and execution-buffer uploads remained 192 and 512; source tickets reached zero.

Effective cache retention differed despite identical configured limits:
retained source bytes fell from 22604243016 to 16912135472, peak committed bytes
from 23085449216 to 18387828736, and source-cache pressure evictions rose from
371 to 480. Native bypass/source-read bytes rose from 102139290488 to
139096044040. These are bytes serviced through the file-backed bypass route,
not proof of physical-disk reads. Batch-submit wall time rose from 49.0033472 s
to 84.2293399 s; pool eviction wait remained negligible at 0.0001577 s.
The native cache respects dynamic RAM/commit headroom, not just its configured
ceiling. These observations do not isolate the source patch's causal effect or
identify the reason for the headroom difference. Do not compensate by raising
limits or disabling host-memory admission guards.

Driver usage after generation was 4.95 GiB absolute; tracker global RSS
observations reached 25.7256 GiB, including the text-encoder phase. Generation
monitoring remained disabled; these are not continuous denoise peaks.
CPU inspection confirmed 121 frames at 1280x704, one audio stream and a
nonconstant first frame. Quality/full-model numerical equivalence was not assessed.

Artifacts: sibling host `outputs/ddo_native_video_20261005_210813_488/`.
Source SHA256: `581703C58AF60D1D124C087265465E13C585D2B8CFB486C510A1A5B082F265E8`.
The local patch and editable installation remain diagnostic candidates, not
published dependency updates. No rerun, budget change or modular custom-block
implementation followed this gate. Preserve the prior 26.09 s/step reference;
resolve effective host-cache comparability before drawing a speedup conclusion.

#### Read-only source-cache pressure audit

The persistent transport leaves `host_headroom_bytes` at its native API default
of 1 GiB and resolves 64 MiB source slots for this model-window workload.
Before every cache request, `poll_pressure()` samples Windows
`GlobalMemoryStatusEx`: physical RAM available and available commit capacity
(`ullAvailPageFile`, not free space on the pagefile disk). It uses the smaller
observation. Below 1.0625 GiB it suspends growth; while suspended and below
1.125 GiB it discards completed tail storage in 64 MiB units. Growth resumes
once the upper threshold is reached. An unavailable observation also suspends
growth. Referenced/filling slots are not discarded to force room.

For the persistent-buffer route, explicit registration trimming preserves
storage; the pressure path explicitly requests storage discard. Thus the
371/480 pressure-eviction counts reflect this native headroom response, not a
reduced configured cache ceiling. Zero `admission_denials` does not mean no
pressure: misses while growth is suspended bypass before slot admission, and
are counted separately as 10702/14601 pressure-blocked requests. A final
`pressure_suspended=0` does not erase earlier pressure episodes.

Both reported process limits were 28051505152 bytes (26.125 GiB, including
1.5 GiB registered host staging), and neither cache approached the configured
24.625 GiB ceiling at its recorded commitment peak. The zero DXGI system
headroom option in the host JSON is a different control; it does not disable
the source cache's 1 GiB host reserve. Do not conflate GPU weight budget,
registered host staging, source commitment and driver shared-memory usage.

The existing snapshot cannot identify whether RAM, commit, failed observation,
or outside-process consumption initiated each pressure episode. Current machine
observations cannot reconstruct that history. A bounded future diagnostic should
capture native observations at admission/pressure transitions, their RAM/commit
values, thresholds, committed/registered bytes, reclaim outcome and reason, via
DDO's public report rather than runner-internal metric accesses. Preserve all
limits and ownership rules; do not add per-tensor logging or an automatic budget
increase. No code, binary, environment or CUDA execution changed in this audit.

### Native pressure observations candidate (2026-10-05)

Following approval, added fixed-size native pressure observations and an optional
snapshot export, propagated through DDO's public report and readable profile.
No admission threshold, cache ceiling, budget, ownership policy or additional OS
poll changed. Candidate SHA256:
`15D349814F4E6D19C3F055D91BB91F018125E2306ACE8FB60E484FCD00976323`.
Eight C++ CPU contracts and 59 Python tests/two subtests passed. The old selected
DLL remains usable: six Python pressure/report tests passed, optional export test
skipped. No CUDA execution or performance claim for this binary. The two small
CUDA contracts followed conditionally by one unchanged video case are proposed
in [the gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-pressure-observations-gate.md), pending confirmation.

### Approved pressure-observation video result (2026-10-05)

Both selected small CUDA contracts passed after repairing an inaccessible pytest
temporary directory with a fresh workspace base directory. The approved matched
BF16 video then completed: 1280x704, five seconds, 121 frames at 24 fps, seed 43,
eight steps, no input audio, unchanged 2 GiB GPU weight budget / 1.5 GiB staging /
24.625 GiB source and registration ceilings. Local lifetime-patched Diffusers
remained installed. Progress final average was 28.60 s/step, console generation
229.4 seconds and JSON total 303.0063 seconds: still slower than the 26.09 s/step
pool candidate, not a demonstrated performance improvement.

The new public report recorded 241 pressure entries, all commit-limited, and 241
recoveries; no RAM-limited entries or failed observations. Minimum sampled RAM
available was 17.1824 GiB versus only 0.7825 GiB available commit. Pressure evicted
476 slots and released 31943819264 bytes cumulatively. Final retained bytes were
16979244336 (15.8132 GiB), bypass route bytes 140737816120. These do not establish
physical disk traffic or identify the consuming allocation. Post-run read-only
CIM showed a fixed 32 GiB pagefile, not automatically managed; no OS setting changed.

Generation/cleanup/decode/save completed, tickets reached zero, with zero reported
pool failures/quarantine/temporary tensors/source-read failures/plan mismatches.
CPU media inspection decoded 121 frames at the requested shape/rate and one audio
stream. Quality and full-model numerical equivalence remain unassessed. Driver
after-generation usage was 4.95 GiB absolute; RAM observations reached 25.7253 GiB
including encoding, not continuously sampled denoise peaks. Encoder used group
offload, not the native ownership route. No default promotion or extra run.

Artifacts: sibling host `outputs/ddo_native_video_20261005_221330_003/`.
Detailed observations and conditions: [pressure gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-pressure-observations-gate.md).

### Pre-transformer PyTorch host-cache boundary candidate (CPU only)

Added an explicit opt-in DDO helper to release unused PyTorch pinned host blocks
once after prior phases complete, before native transformer setup. The video
runner's new flag is disabled by default and scoped to native VMM; image, linear,
NVMe, native DLL, Diffusers forward and memory ceilings remain unchanged.
The public report separates raw inconsistent active counters from validated
values and records allocator/CPU observations before and after release.
60 DDO CPU tests plus two subtests and 13 video-host CPU tests plus 12 subtests
passed. No CUDA execution, measured recovered bytes or performance claim for
this candidate. Proposed one small CUDA gate and one matched video are documented
in [the boundary gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/pytorch-host-cache-boundary.md), pending approval.

### Approved early host-cache release video result (2026-10-05)

The small CUDA live-pinned-block/copy gate passed, then the approved matched
video with `--release-pytorch-host-cache-before-transformer` completed. Same
BF16, 1280x704, five seconds, 121 frames/24 fps, seed 43, eight steps, no input
audio, DLL/local Diffusers forward, activation chunk sizes and budgets.

| Measurement | Without early release | With opt-in release |
| --- | ---: | ---: |
| Final progress average, s/step | 28.60 | 26.23 |
| Generation console seconds | 229.4 | 210.4 |
| Full run JSON seconds | 303.0063 | 283.1354 |
| Retained native cache bytes | 16979244336 | 20322541640 |
| Commit-pressure episodes | 241 | 206 |
| Native cache bypass-route bytes | 140737816120 | 117728656640 |

The boundary relinquished 3487499715 PyTorch allocator bytes (3.248 GiB) before
transformer setup; adjacent snapshots showed similar-scale private commitment
reduction and available-commit increase. New validated active-byte fields stayed
null because the raw Torch active counter exceeded allocated bytes. Generation
improved by 19.0 seconds and total by 19.8709 seconds in this single sequential
comparison. No repeat-validated causality or new overall record: the earlier pool
case remains 26.09 s/step / 209.4 s generation / 282.7551 s total.

Pressure remains: all 206 episodes were commit-limited. No guards/ceilings or
pagefile policy changed. Generation/cleanup/decode/save and CPU frame inspection
completed, with zero reported temporary tensors, mismatches, source-read failures,
pool failures or quarantine and tickets reaching zero. Full-model numerical
equivalence/quality unassessed; no continuous denoise memory peak measurement.
Artifacts: sibling host `outputs/ddo_native_video_20261005_223421_903/`.
Keep the flag opt-in, defaults/frozen image routes unchanged. Detailed report:
[boundary gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/pytorch-host-cache-boundary.md).

### Approved residual CPU source detachment video (2026-10-06)

The single approved video added only
`--detach-transformer-auxiliary-cpu-sources` to the preceding phase diagnostic:
BF16, 1280x704, five seconds, 121 frames/24 fps, seed 43, eight steps, no input
audio, unchanged pressure-observation DLL/local Diffusers forward, 2 GiB resident
weight ceiling, 1024-token feed-forward/normalization chunks and early host-cache
release. Artifact: sibling host
`outputs/ddo_native_video_20261006_091222_765/`.

| Measurement | Previous phase diagnostic | Residual detachment |
| --- | ---: | ---: |
| Final progress average, s/step | 26.35 | 24.87 |
| Generation console seconds | 211.4 | 199.7 |
| Full run JSON seconds | 283.9525 | 276.9484 |
| Attached process private commitment, GiB | 44.0165 | 8.5854 |
| Retained native cache bytes | 19999143216 | 26362339400 |
| Commit-pressure entries / recoveries | 213 / 213 | 0 / 0 |
| Native cache bypass-route bytes | 118439786536 | 92894603200 |
| Native cache batch-submit seconds | 59.3932947 | 20.4131390 |

DDO copied 8257536 bytes once across 192 CPU owners/extents from four shards,
with zero unbound CPU groups and no GPU placement change. The boundary took
0.0710587 seconds. Adjacent private-byte samples fell from 47266713600 to
9218457600: 38048256000 bytes (35.4352 GiB) recovered in this workload. Available
commit increased by 38049525760 bytes. This real-model observation supersedes
the unresolved recovery hypothesis for this selected case, not the preserved
synthetic CPU probe's different result. The generic operation's
`commit_recovery_proven=false` remains intentionally conservative; OS samples,
not copied payload size, establish the measured recovery.

Native cache retained 24.5518 GiB within the unchanged 24.625 GiB ceiling. All
30841 pressure observations were valid; minimum sampled available RAM/commit
were 10427731968/31595470848 bytes. Required fault/guard/CUDA/capacity/source-read/
pool failures, temporary fallbacks, plan mismatches, quarantine and failed
reclaims were zero. Tickets were 0 to 0; materializations/buffer uploads stayed
192/512. All five phase reports completed without inspection errors.

Generation improved by 11.7 seconds and full runtime by 7.0041 seconds in this
single sequential comparison; no repeat-tested causality or general model/image
regression claim. More useful cache residency raised observed global tracker
RSS to 29.1745 GiB, versus 25.7260 GiB previously; this is not a continuous denoise
peak. After-generation private bytes were 38267658240, below the prior
69840949248. Driver usage after pipeline return remained about 4.95 GiB absolute
/ 3.93 GiB over baseline. Encoder execution remained Diffusers group-offload,
not native encoder execution. Cleanup, decode and save passed. CPU media
inspection decoded all 121 frames, 1280x704/24 fps, with one generated audio
stream and first-frame pixel standard deviation 51.4805808; quality/full-model
equivalence unassessed. Keep the correction opt-in; no further CUDA run,
parameter expansion, default promotion or compatibility removal is authorized
by this completed sequence.

### Approved read-only video phase diagnostic (2026-10-05)

Same early-release workload/DLL/local forward/budgets, adding only
`--inspect-host-memory-phases`. Artifact:
sibling host `outputs/ddo_native_video_20261005_225817_983/`.
Final progress 26.35 s/step, generation 211.4 seconds, JSON total 283.9525 seconds;
near the prior 26.23 / 210.4 / 283.1354, not a new speed record.

All five public phase reports completed without inspection errors. Loading
increased private commitment from 8.1216 to 43.5684 GiB with little RSS growth.
After early cache release, private bytes were 42.4982 GiB; after native attach,
44.0165 GiB. Registered CPU tensor storage fell from 37977676288 to 8257536 bytes
(4186 to 192 extents), while private commitment did not fall similarly. Residual
scale/shift parameters resolve to shards; their small visible storage lengths
do not enumerate hidden parent mappings. A bounded CPU synthetic bulk-loader
check confirmed a 64 KiB tensor could retain its file mapping, but did not show
immediate private-commit recovery after cloning/dropping the mapped view.
This is an ownership-correction candidate, not proven recovered video memory.

Native cache retained 19999143216 bytes (18.6257 GiB); 213/213 pressure entries/
recoveries remained commit-limited. Generation, cleanup, save and decoding all
121 video frames passed. Required failure/fallback/quarantine/mismatch counters
were zero; tickets collected 2 to 0. No new continuous memory peak or quality/
equivalence claim, GPU-budget increase or default change. Detailed phase table,
CPU limitations and next correction scope are in the
[boundary gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/pytorch-host-cache-boundary.md).

### Approved modular video smoke gate (2026-10-06)

The sibling host's separate modular runner completed the approved BF16 video
case: 1280x704, five seconds, 121 frames/24 fps, seed 43, two distilled steps,
no input-audio lock and packed latent output (no decode). The pressure-observation
DLL, local Diffusers transformer forward, 2 GiB weight ceiling, 1024-token FF/
RMSNorm chunks, 24.625 GiB source/registration ceiling, early host-cache release
and bounded auxiliary CPU source detachment were unchanged.

Artifact: sibling host `outputs/ddo_modular_video_20261006_102749_566/`.
Denoise phase: 55.2658 seconds; final progress average: 27.54 s/step; full
latent-output run: 92.7974 seconds. This two-step startup-containing smoke gate
is not a matched comparison with the accepted eight-step 24.87 s/step control.
No performance promotion or decode/visual-quality claim.

CPU-only inspection confirmed entirely finite float32 packed video/audio tensors
of shapes [1, 14080, 128] and [1, 126, 128]. Native plan passes: two;
materializations/signature hits/execution-buffer uploads: 48/24/128. Required
fault/guard/CUDA/capacity/source-read/pool failures, temporary fallbacks, plan
mismatches, quarantined bytes and failed reclaims were zero. Cache pressure
entries/recoveries: 0/0; failed observations: zero of 7735 observations. Source
tickets were 0 to 0 at collection. The process retired the native component,
saved latents and exited successfully; the snapshot was taken before retirement,
not a post-retirement ownership census.

Sampled absolute driver peak: 7.6819 GiB, or 6.6551 GiB over the 1.0269 GiB
baseline. Sampled process RSS peak: 28.2050 GiB. Cache retained 26362339400 bytes
within its unchanged ceiling. Minimum available RAM/commit:
2728017920/20441145344 bytes; do not automatically expand memory ceilings.
The earlier failed tokenizer/declared-schedule handoffs and correction evidence
are preserved in the sibling host's `ddo-modular-video.md`.
Existing reference/image/frozen-runtime routes were not changed or rerun.
Real-model numerical parity, decode and an explicitly approved matched benchmark
remain pending; this completed smoke gate authorizes no additional CUDA run.

## Approved modular video eight-step gate (2026-10-06)

Sibling artifact: `outputs/ddo_modular_video_20261006_112207_195/`,
`metrics/modular_run.json`. Selected BF16 workload: 1280 x 704, 5 seconds,
24 fps/121 frames, seed 43, real prompt, image conditioning, eight steps,
no input-audio lock, packed latent output (no decode). Native budgets remain
2 GiB resident weights, 1.5 GiB staging, 24.625 GiB source/registration;
FF/RMSNorm chunks remain 1024. Pressure-observation DLL SHA256 remains
`15D349814F4E6D19C3F055D91BB91F018125E2306ACE8FB60E484FCD00976323`.

Candidate host changes: positive-only distilled connectors, inference-mode
phase execution and generation-local precomputed RoPE. Local Diffusers forward
SHA256: `F37835B2935C01ED1DC23C67BAF702A9B6CB0D399611512A2E627FC23A8BAC71`;
the default non-precomputed path is preserved. 24 CPU contracts previously
passed, including exact small-transformer RoPE parity.

Eight-step progress average: **24.39 s/step**, versus the recorded reference
runner's **24.87 s/step**. Denoise phase including final readback/synchronization:
**195.3200 seconds**. Total latent-output run: **234.9401 seconds**. The approximately
1.9% progress-average difference is one selected observation, not an isolated
optimization effect, repeatability claim or full-pipeline/decode comparison.
The earlier two-step 27.54 s/step result remains startup-containing smoke evidence.

CPU-only readback confirmed finite float32 packed video/audio tensors,
[1, 14080, 128] and [1, 126, 128]. Sampled absolute GPU peak: 7.6155 GiB;
baseline-relative delta: 6.5886 GiB over 1.02685546875 GiB. Sampled RSS peak:
28.4167 GiB. Historical reference memory monitoring was different; no matched
memory-reduction claim. Native minimum available RAM/commit:
3668602880/20231127040 bytes.

Native plan passes/materializations/signature hits/execution-buffer uploads:
8/192/168/512. Required fault/guard/CUDA/capacity/source-read/pool failures,
admission denials, temporary fallback tensors/bytes, plan mismatches,
quarantine and failed reclaims were zero. Cache pressure entries/recoveries:
0/0; failed observations: zero. Retained source bytes: 26362339400; total
resident weights: 2143289344 bytes, within the fixed ceiling. Source tickets:
0 to 0 at nonblocking collection. Snapshot precedes successful retirement,
not a post-retirement ownership census.

Status: selected eight-step safety/performance criterion met; output saved and
run completed. No decoded video, real-checkpoint numerical/visual parity,
native encoder integration or CUDA image-regression claim. No further CUDA
execution authorized by this completed gate.

## Approved staged native encoder/video gate (2026-10-06)

Sibling artifact: `outputs/ddo_modular_video_20261006_113526_724/`,
`metrics/modular_run.json`. Same eight-step BF16/1280 x 704/5-second/24-fps/
121-frame/seed-43 real-prompt image-conditioned workload as the preceding
modular gate; no input-audio lock or decode. Explicit encoder opt-in uses
4 GiB weight ceiling, component ownership, 2 MiB pages and 1 MiB bounded
buffers; transformer retains its approved 2 GiB ceiling. Staging/source/
registration, chunks, DLL and local forward hashes remain unchanged.

Encoder phase: **13.5180 s**, versus **17.7256 s** in the preceding group-offload
run. Denoise phase: **190.3552 s**; final progress average **23.77 s/step**,
versus **24.39 s/step**. Total latent-output run: **228.9680 s**, versus
234.9401 s. Transformer attach increased from 5.1255 to 7.9564 s; do not
attribute all total/denoise differences to the encoder or claim repeatability.

CPU-only comparison of the two saved packed float32 outputs found bit-exact
video [1, 14080, 128] and audio [1, 126, 128]: finite, torch.equal true,
maximum and mean absolute difference zero. This is selected final-latent parity
with the preceding modular group-offload run, not pixel/audio decode parity,
raw encoder-embedding equality or arbitrary-checkpoint support.

Preflight bound 1065/1065 source groups with zero missing/rejected/ineligible
groups or invalid/ambiguous headers; 82 execution units, six bounded buffers,
34,818 buffer bytes. Executed public reports confirm native encoder ownership
with 4 GiB, then native transformer with 2 GiB. Resident-weight snapshots:
4030726144 and 2143289344 bytes. Nonblocking encoder ticket collection: 7 to 0;
transformer: 0 to 0. Successful retirement and process exit 0, no post-retirement
allocation census.

Both reports have zero required fault/guard/CUDA/capacity/source-read/pool
failures, admission denials, temporary tensors/bytes, plan mismatches,
quarantine, failed observations/reclaims and cache pressure entries/recoveries.
Encoder retained/process-registered/committed: 23555211264 bytes; transformer
retained: 26362339400, process-registered/committed: 26440892416 bytes, within
the unchanged source ceiling. These observed snapshots do not constitute
a continuous no-overlap census.

Sampled absolute GPU peaks: encoder 5.0132 GiB, denoise 7.6969 GiB;
denoise baseline-relative 6.6700 GiB over 1.02685546875 GiB. Sampled RSS peak:
28.4537 GiB. Native minimum available RAM/commit: encoder
8268324864/25797361664, transformer 4287311872/19938123776 bytes.

Status: selected staged native encoder/video safety and no-regression gate PASS.
Encoder remains explicit opt-in; streaming decode, decoded-media quality and
CUDA image regression were not tested or promoted. No further CUDA case is
authorized by this completed sequence.

## Approved modular full-video gate (2026-10-06)

Sibling artifact: `E:/LTX-2/outputs/ddo_modular_video_20261006_115322_220/`,
report `metrics/modular_run.json`, media `ltx23_modular_1280x704_5.0s_seed43.mp4`.
The approved eight-step BF16 workload retains 1280 x 704, five seconds,
24 fps, 121 causal-grid frames, seed 43 and no input-audio conditioning.
Native encoder/transformer ceilings remain 4/2 GiB; source/registration
24.625 GiB, staging 1.5 GiB and FF/RMSNorm chunks 1024 are unchanged.
DLL and local Diffusers forward hashes are unchanged from the preceding gate.

Executed from `E:/LTX-2`:

```powershell
.\run_ddo_vmm_video_modular.ps1 -DisableTlsKeyLog -LibraryPath 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-pressure-observations-20261005\Release\ddo_vmm.dll' --local-files-only --width 1280 --height 704 --seconds 5 --steps 8 --seed 43 --no-use-audio-condition --output-type video --decode-device cuda:0 --stream-video-decode --ddo-vmm-resident-weight-budget-gb 2 --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --release-pytorch-host-cache-before-transformer --inspect-host-memory-phases --detach-transformer-auxiliary-cpu-sources
```

Hypothesis: bounded temporal GPU decode completes after transformer retirement
without keeping the entire decoded video on GPU. Acceptance: no native failure
or OOM, expected frame dimensions/count, finite audio and a readable MP4.

Phase seconds: encoder 14.1422, connectors 6.7997, video encoder 3.9754,
empty audio encoder 0.1470, transformer load/prepare 2.1805, attach 8.0256,
denoise 197.0318, decode/save 30.1365. Total: **266.5138 s**;
final denoise progress average: **24.60 s/step**. The preceding latent-only
23.77 s/step was not reproduced; decode follows denoise and is not an
explanation for this variation. The recorded original full-MP4 control total
was 276.9484 s. Neither difference establishes an isolated or repeatable gain.

Sampled absolute driver GPU peaks: encoder 5.1772 GiB, denoise 7.6247 GiB,
decode/save 5.2983 GiB; baseline-relative deltas 4.1504/6.5978/4.2715 GiB
over 1.02685546875 GiB. Peak sampled process RSS: 28.3814 GiB;
decode/save RSS peak 7.4905 GiB. Sampling is not an allocator census or
a matched memory comparison with the historical control.

Both native reports have zero required fault/guard/CUDA/capacity/source-read/
pool failures, admission denials, temporary fallbacks, plan mismatches,
quarantine, failed observations/reclaims and cache pressure entries/recoveries.
Encoder tickets collected 7 to 0; transformer 0 to 0. Resident-weight
snapshots: 4030726144/2143289344 bytes. Retirement succeeded; snapshots
are not post-retirement ownership measurements. Process exit: 0.

CPU-only full PyAV readback confirmed H.264, 121 frames at 1280 x 704,
24 fps, container duration 5.041667 s; AAC stereo at 48000 Hz, all decoded
audio finite. Generated audio has 240480 samples/channel; decoded AAC has
240640 due to codec padding. First-frame RGB standard deviation 51.6658
confirms nonblank variation, not visual quality. No packed-latent comparison
was performed for this MP4 run.

Status: selected full-pipeline execution/media-structure gate PASS. User
visual/audio review remains pending. No general model support, image CUDA
regression, input-audio conditioning or decoded-media parity claim; no
additional CUDA execution is authorized by this completed gate.

## Approved modular duration extension: eight seconds (2026-10-06)

Artifact: `E:/LTX-2/outputs/ddo_modular_video_20261006_120540_047/`;
report `metrics/modular_run.json`; media
`ltx23_modular_1280x704_8.0s_seed43.mp4`. User approved an eight-second
case followed conditionally by ten seconds after checking safety and media.
Exact command is the preceding full-video gate with only `--seconds 8`
substituted for `--seconds 5`. Resolution 1280 x 704, 24 fps, BF16, eight
steps, seed 43, no input-audio conditioning, all ceilings, chunks and
DLL/forward hashes remain unchanged. No implementation or preset change.

Hypothesis: native execution and bounded temporal decode permit longer
duration without raising memory ceilings. Criteria: no OOM/native failure
or memory anomaly, finite audio, complete readable MP4 and 193 frames.

Process exit 0, status completed. Total **593.8051 s**; denoise
**512.8294 s**, final progress average **64.05 s/step**; decode/save
**42.5483 s**. Console cumulative timestamps show first step 179.51 s,
second approximately 86 s, then approximately 41 s per step. The latter
are rounded console differences, not separately instrumented timings;
the full average must not be replaced by the faster late-step estimate.
The duration extension passed capacity/execution, not a performance target.

Sampled absolute GPU denoise peak **7.9995 GiB**, delta **6.9727 GiB**
over baseline 1.02685546875 GiB; sampled RSS peak **29.3297 GiB**.
Decode/save absolute GPU/RSS: 5.3198/11.3892 GiB. Dedicated memory is
near the physical ceiling; no shared-memory or paging measurement was
collected to explain the slow startup. No causal cache/driver claim.

Both native reports have zero fault/guard/CUDA/capacity/source-read/pool
failures, admission denials, temporary tensors/bytes, plan mismatches,
quarantine, failed observations/reclaims and pressure entries/recoveries.
Encoder tickets 7 to 0; transformer 0 to 0, nonblocking collection.
Native minimum available RAM/commit bytes: encoder 12280246272/28619956224,
transformer 6498349056/20888707072. Retirement succeeded.

CPU-only full PyAV readback confirmed 193 H.264 frames at 1280 x 704,
24 fps, duration 8.041667 s; AAC stereo at 48000 Hz, all audio finite.
Generated audio samples/channel: 384480; decoded AAC: 385024 (padding).
Selected eight-second gate PASS, allowing the already approved ten-second
case. Visual/audio quality and numerical parity remain user-review/unverified
items; no image route was modified or rebenchmarked.

### Conditional ten-second result (same approved sequence)

Artifact: `E:/LTX-2/outputs/ddo_modular_video_20261006_121621_526/`,
`metrics/modular_run.json`, `ltx23_modular_1280x704_10.0s_seed43.mp4`.
Executed the preceding full-video command with only `--seconds 10`
substituted, after the eight-second media/native gate passed. No budget,
implementation, preset, DLL or forward change. Expected frame count: 241.

Process exit 0, completed. Total **899.4344 s**; denoise **808.3324 s**,
final progress average **100.97 s/step**; decode/save **53.2628 s**.
Encoder/connectors/video encode/empty audio encode/load+prepare/attach:
13.5706/6.7985/3.6856/0.1415/2.0754/7.5912 s. Console first step:
272.92 s; rounded cumulative differences place second near 177 s and
steps three through eight near 59-60 s each. These are startup-containing
single runs, not a long-duration performance baseline or causal diagnosis.

Sampled absolute GPU denoise peak **7.9995 GiB**, delta **6.9727 GiB**
over the same baseline; peak RSS **31.4914 GiB**. Decode/save absolute
GPU/RSS peaks: 5.2964/16.1433 GiB. No automatic ceiling growth or parameter
calibration. Dedicated GPU memory remains near full, but shared-memory
paging and the cause of the first-two-step cost were not measured.

Both native reports again have zero required failures/denials, temporary
fallbacks, mismatches, quarantine, failed observations/reclaims and cache
pressure entries/recoveries. Tickets: encoder 7 to 0, transformer 0 to 0;
retirement succeeded. Native minimum available RAM/commit bytes: encoder
12664147968/28591906816, transformer 5264330752/19394445312.

CPU-only full media decode: H.264, 241 frames at 1280 x 704, 24 fps,
10.041667 s duration; AAC stereo 48000 Hz, all audio finite. Generated
samples/channel: 480480; decoded AAC: 481280 (padding).

Selected ten-second capacity/media gate PASS: this workload exceeds the
user-reported earlier eight-second duration limit without an OOM. This is
not a claim of faster generation, general maximum supported duration,
matched external-reference performance or image CUDA non-regression.
User visual/audio acceptance remains pending. The approved sequence is
complete; no further CUDA run, tuning or implementation change is authorized.

### Startup allocation diagnosis prepared (non-CUDA only)

Optional driver allocation/pool counters and existing GPU/native timeline
integration are ready in a separate diagnostic build. Normal execution and
the accepted DLL are preserved. Pure-CPU counter contract, 41 Python contracts
with candidate exports and 33 host CPU contracts passed; no CUDA run followed.
See [scope, hash and proposed four-step eight-second gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-allocation-diagnostics-gate.md).
This preparation does not establish the cause of the slow first two steps.

### Eight-second startup diagnostic collected (2026-10-06)

Approved four-step latent-only diagnostic completed, exit 0; packed video
and audio finite. Compute-stream intervals: 204.8341/89.8687/106.2195/93.0882 s;
phase 494.1370 s, total 532.6823 s. Native timeline complete (1187 samples,
zero drops), original safety counters zero and encoder/transformer tickets
retired. Diagnostic-only, not matched performance or image regression evidence.

Measured allocation driver-call host time totals 6.7117 s, insufficient to
explain denoise directly. Transformer-block windows dominate, but are not
kernel-only time. Two first-step access calls returned non-success; existing
access-retry context synchronization is outside the measured wrappers, so
allocation-path waiting remains unresolved. Do not remove that safety wait
or tune ceilings on this evidence. Accepted DLL/policy unchanged. Details and
artifact: [native allocation diagnosis](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-allocation-diagnostics-gate.md).

### Access-retry timing prepared (non-CUDA)

Separate diagnostic candidate adds existing access-retry synchronization and
enclosing fault timing plus access return categories, without changing policy.
CPU C++ contract and 44 Python contracts passed; previous binary hashes
unchanged. No CUDA execution. Follow-up command/hypothesis/criteria await
confirmation in [native allocation diagnosis](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-allocation-diagnostics-gate.md).

### Access-retry diagnostic collected (2026-10-06)

Approved same eight-second/four-step latent-only diagnostic completed, exit 0.
Native schema 2 and timeline complete (1187 samples, zero drops), packed
video/audio finite, original native safety counters zero and tickets retired.
Phase 406.3883 s, total 443.9171 s; event intervals
244.9366/61.4196/49.0461/50.8503 s. Not a matched benchmark or a claimed speedup.

Existing model-arena retry synchronization: zero calls/time; enclosing 98 faults:
0.0981724 s. One first-step access returned NOT_READY outside that measured retry;
final inference succeeded. Driver-call categories totaled 7.3684 host seconds.
This rules out the suspected retry wait for this run, not all driver/forward
waiting or WDDM paging. Preserve its safety wait and accepted memory policy.
Unresolved execution-window cost needs operator/host-versus-GPU attribution
before a correction. Artifact and details:
[native allocation diagnosis](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-allocation-diagnostics-gate.md).

### Sampled operator tracing prepared (non-CUDA)

Generic DDO selected-forward profiling/public reporting and optional host
consumption are prepared, default off. Two selected invocations only; profiling
may synchronize and is not benchmark evidence. 47 DDO CPU and 34 host CPU
contracts passed. Native binary/forward hashes preserved, no CUDA execution.
The next single command and acceptance criteria await user confirmation in
[sampled operator gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-operator-diagnostics-gate.md).

The approved operator diagnostic dispatch was interrupted: execution tool
returned `aborted`, without session/output/exit code. Observed directory
`E:/LTX-2/outputs/ddo_modular_video_20261006_150618_511/` contains no files;
subsequent process inspection showed no active high-memory inference process.
No diagnostic or safety result is available; interruption cause unknown.
No automatic repeat. Relaunch requires confirmation.

### Sampled operator diagnostic completed after approved relaunch

Artifact `E:/LTX-2/outputs/ddo_modular_video_20261006_150751_571/`, exit 0.
Exactly two valid CUDA operator captures; native timeline complete/no drops,
packed video/audio finite and final native safety counters zero; tickets retired.
One recovered access OUT_OF_MEMORY return is recorded separately. Phase
539.3473 s, total 579.3895 s; event intervals
160.6900/233.8860/97.3991/47.2788 s. Diagnostic-only, not matched performance.

Both sampled forwards contain six pageable CPU-to-GPU copies matching the
modulation tables: 172032 bytes/block, 8257536 bytes across 48 blocks. These
tables exceed the existing 16 KiB small-local threshold and remain CPU after
source detachment. Stream waits within `to/copy_` sum approximately 1.20/2.50 s
in the samples, whereas pageable copy device time is approximately 14/16 us.
Waits may drain prior GPU work; do not count them as guaranteed savings or
extrapolate one block to all step cost. Preserve attention/backend/budgets.
Propose accounted native auxiliary ownership rather than a Python residency
cache or shared-threshold change. Full trace interpretation and next gate scope:
[sampled operator diagnostic](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-operator-diagnostics-gate.md).

## 2026-10-06 — execution-unit auxiliary ownership preparation

Implemented a default-off, model-agnostic native auxiliary ownership candidate
after the bounded operator diagnostic. Auxiliary parameters now join existing
immutable source batches, arena views and event-fenced execution slots without
raising the GPU ceiling or the shared small-tensor threshold. Existing dense
prefetch is preserved; component-wide encoder ownership remains separate.

CPU evidence: DDO selected contracts **110 passed, 2 subtests passed**; host CPU
contracts using `E:/LTX-2/.venv-vmm` **53 passed, 31 subtests passed**. An initial
host collection attempt in DDO's environment lacked `torchaudio`; no CUDA model
failure or dependency replacement occurred. No CUDA gate or new performance
measurement was executed. Earlier accepted results remain unchanged.

The new flag is `--ddo-vmm-execution-auxiliary-ownership`. The unchanged diagnostic
DLL hash is `335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.
See [implementation and proposed small CUDA gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — auxiliary ownership small CUDA gate

The user approved the two-case native auxiliary lifecycle gate. Actual result:
**2 passed in 5.26 s**, exit 0, using DDO's `.venv-vmm` and the unchanged
diagnostic DLL SHA256
`335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.

Both ordinary-error and cooperative-interruption cases passed BF16 numerical
comparison, alias identity, immutable source reuse, zero retained CPU source
bindings, zero asserted fault/fallback/admission/bypass/pending-event counters
and final alias/source/hook cleanup. The small test used its explicit 0.07 GiB
weight ceiling; it did not alter real-model ceilings or the default route.

This is small CUDA contract evidence, not a video performance result, full
allocator-bootstrap gate or image regression acceptance. A real-video bounded
diagnostic with the new flag is proposed, not executed. Its full command,
hypothesis and acceptance criteria are in
[the auxiliary ownership gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md).

### Auxiliary candidate geometry audit (CPU/meta only)

Read-only local config/header inspection reported CUDA uninitialized. It found
864 added auxiliary groups: 672 previously-small groups (4.125 MiB) and 192
residual groups (7.875 MiB). First-block spans grew 68 → 86, but source windows
stayed 13 and **no unit touched additional windows**. Largest aligned payload
grew 773131264 → 773393408 bytes, while rounded two-slot capacity remained
1547698176 bytes. Increased slot capacity or new file windows therefore do not
explain this candidate's regression. Runtime pressure/cache causes remain
unproven without the missing final counters. Proposed next correction preserves
the existing small-parameter route and adds durable public per-step diagnostics;
no corrective patch or new CUDA execution was performed during this audit.

### Narrowed auxiliary candidate and durable partial metrics (CPU only)

The user approved implementation without CUDA execution. The revised selector
preserves the existing small-parameter threshold (inclusive boundary) and joins
only residual parameters to native batches. Expected first-block spans: 72
instead of the broad candidate's 86; aligned payload 773303296 bytes; unchanged
two-slot page capacity 1547698176 bytes. It includes 192 residual groups while
preserving 672 previously-small groups. CPU/meta inspection reported CUDA
uninitialized; these are not runtime performance measurements.

DDO selected contracts: **120 passed, 2 subtests passed**. Host CPU contracts:
**53 passed, 31 subtests passed**. A new public default-off metrics journal
persists before-denoise and per-step partial snapshots without completion
collection or CUDA waits, and the host enables it only for denoise diagnostics.
The snapshot explicitly does not prove GPU completion or replace final gates.

DLL unchanged; no CUDA or new model execution occurred. The earlier broad
candidate's regression and interrupted artifacts remain preserved. The revised
small CUDA fixture uses the production threshold and checks that a small
parameter retains its existing route; execution still requires confirmation.
See [the updated gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — real-video auxiliary candidate interrupted

Approved bounded 8-second/4-step latent diagnostic with auxiliary ownership:
`E:/LTX-2/outputs/ddo_modular_video_20261006_155602_695`.
The user requested interruption after a timing regression. Wrapper child exit
-1; enclosing command exit 1. First progress step 473.24 s versus the previous
diagnostic's 160.689953125 s CUDA-event step; second step approximately 198 s
from rounded console elapsed. No four-step average, final metrics, packed
latents or normal cleanup acceptance is available.

Only invocation 0 was exported: pageable H2D copies 6 → 0 and stream-sync
events 6 → 0, with 897 kernel events in both captures. Post-attach CPU source
storage observation was 0 bytes. However, guarded forward publication moves
the native upload preamble inside the profiler scope, so the new capture's
larger pinned byte total is not a matched whole-step transfer comparison.

The candidate is **not accepted for promotion**. Preserve defaults and previous
accepted results; no budgets or attention backend were changed. Static review
identifies a broader change to small-parameter ownership and native batch/view
geometry beyond the six residual table copies, but final native counters are
unavailable and the slowdown's cause is not yet established. Full partial
evidence and the next investigation boundary are in
[the auxiliary ownership gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — revised residual-only small CUDA gate

The user approved only the two revised small CUDA cases, not another video run.
Actual result: **2 passed in 5.20 s**, process exit 0, DDO `.venv-vmm`, unchanged
DLL SHA256 `335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.

With the production 16 KiB threshold, small Parameter identity and CUDA placement
were preserved; native residual aliases, BF16 parity, immutable source reuse,
asserted fault/fallback/admission/bypass/pending-event counters and unloading
checks passed for ordinary error and cooperative interruption. This is acceptance
of the revised small lifecycle contract only, not real-video speed, image
regression or journal publication under CUDA. Defaults and all earlier measured
results remain unchanged. No long or real-model execution occurred in this gate.

## 2026-10-06 — residual-only two-step video diagnostic, interrupted

Approved case: 8 s, 1280x704, seed 43, two steps, latent output, no audio
conditioning, 2 GiB transformer / 4 GiB encoder ceilings, residual-only native
ownership and partial metrics journal; no operator profiler. DLL SHA256 remains
`335F13918FC384B58D2A831C7D336522BF91D0504DDA5FB0B5D0D2704E5ECFF8`.

Transformer attach **8.3725 s**. Partial ownership report confirms 192 groups /
7.875 MiB with the existing 16 KiB threshold. First callback host elapsed
**177.9379351 s**; no second callback arrived before the user-requested stop,
approximately **768 s** later. These are host observations, not final GPU event
timings or a matched benchmark. Forced child exit -1 is not a CUDA exception.

Output `E:/LTX-2/outputs/ddo_modular_video_20261006_165752_982` preserves
`before_denoise.json` and `step_0000.json`; console is
`E:/LTX-2/outputs/residual_auxiliary_console_20261006_165752_896.log`.
No final metrics, packed latents, numerical validation or normal cleanup result
exists. First partial failure/quarantine counters are zero but completion is
unproven. Candidate **not accepted for promotion**; narrowing small-parameter
ownership did not resolve the slowdown. Preserve accepted defaults and budgets.
See [the auxiliary gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md) for conditions
and limitations. No follow-up CUDA run was started.

## 2026-10-06 — pending-span fragmentation CPU reproduction

### Subsequent joined-span bounded video completion

User-approved repeated 8 s / 1280x704 / seed 43 / two-step latent diagnostic,
same residual ownership and budgets, changing only the DLL to SHA256
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Both 240 s callback limits and the 120 s finalization limit passed; exit 0.
Attach **7.8578 s**, total **339.5551 s**, CUDA phase **300.58353125 s**,
steps **104.2100390625 / 196.28828125 s**, tail **0.08518723 s**.
Two steps do not establish settled steady state.

Output `E:/LTX-2/outputs/ddo_modular_video_20261006_175732_123` contains final
metrics, all three partial snapshots and finite float32 packed video
`[1,22000,128]` / audio `[1,201,128]`, verified independently on CPU. Selected
failure/quarantine/fallback/ownership counters passed; native retirement and
unload completed normally. Activation pool observed peak **6.15234375 GiB**;
GPU peak **6.97265625 GiB is baseline-relative**, RSS peak **29.54099274 GiB**.
Snapshot growth reduction, WDDM paging and sole causal attribution are not
proven. Accepted for this bounded selected-workload completion only, not
eight-step performance, image regression, parity or default promotion.
See [the gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md) for raw paths,
limitations and the static historical validation label in the JSON.

CPU-only MSVC x64 reproduction against the unchanged production ledger:
**61 checks passed**, exit 0. Four adjacent idle 1024-byte spans with different
pending fences cannot satisfy a 2048-byte request despite 4096 idle bytes.
Completion observations restore the fit; shared-fence pending reuse and
live/quarantine/rollback/byte-conservation controls pass. No CUDA library is
linked, no GPU operation is executed and no production allocator code or DLL
changes. This confirms a conservative ledger limitation, not the cause of the
interrupted video's slowdown. Reproduction and next implementation boundary:
[auxiliary gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — joined-span small CUDA gate passed

User-approved eight-case sequence: **8 passed in 6.09 s**, exit 0, DDO
`.venv-vmm`. Candidate SHA256 verified:
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
The two new same/cross-stream joined reservations reused the original 2 MiB
address without increased mapped bytes/backing count, retained both unfinished
predecessor events, produced correct consumer values and recorded zero
failure/quarantine counters. Six existing allocator ordering cases passed.
Acceptance is limited to this small native contract: no video performance,
stall-causality or image regression claim. Defaults and old binaries remain
unchanged; no model run followed. Details:
[auxiliary gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md).

## 2026-10-06 — opt-in joined-span implementation, CPU only

Separate dependency-preserving native multi-span operation implemented behind
the default-OFF `DDO_VMM_NATIVE_JOINED_SPANS` build option. CPU: **176 new checks
passed**, including seven injected metadata failures; **1292 original ledger
checks passed**; stream dependency contract passed. Candidate DLL compiled with
MSVC x64 / CUDA 12.8.93, SHA256
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Prior binaries/defaults preserved. No CUDA execution or video benchmark occurred;
the proposed eight-case small gate still requires confirmation. No performance
or real-model causal claim. Full build and gate boundary:
[auxiliary gate](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/native-execution-auxiliary-gate.md).
# Video Triton fusion candidate — 2026-10-07

Experimental opt-in RMSNorm/modulation and paired Q/K rotary callbacks are
implemented in the local editable Diffusers/video host, with generic Triton
kernels in the allocator adapter. Both flags remain off by default. Existing
image runners, allocator binaries, chunks and residency budgets are unchanged.
CPU routing/regression: **164 passed, 11 subtests passed in 10.40 s**.
Kernel CUDA/native-VMM numerical validation is pending user approval; no
performance gain or real-model parity is claimed. Prepared gate and constraints:
the sibling video host's `model_execution_optimizations.md` and
`probe_fused_execution_cuda.py`. Shared cross-modal norms remain unchanged.

The first approved CUDA attempt stopped in the split-RoPE reference because
its reshape used frequency batch 1 for activation batch 2. The gate did not
pass. A reference-only zero-stride batch expansion corrected the synthetic
oracle; candidate kernels, inputs and tolerances remain unchanged. Subsequent
CPU regression: **168 passed, 11 subtests passed in 8.37 s**.

Approved corrected CUDA/native-VMM retry passed with exit code 0: 36 numerical
comparisons, including non-default streams and tiny-model path restoration.
RMS FP32 max absolute error `1.9073486328125e-6`; BF16, rotary, and tiny-model
outputs were exact in the selected sample. 28 RMS/modulation and 16 paired rotary
submissions; allocator reported `pluggable` with PyTorch `2.12.1+cu132`.
The accepted DLL, image host, defaults and budgets remain unchanged. This is a
small-contract gate, not a real-video performance or quality acceptance.

## Video operation Inductor baseline — 2026-10-07

Approved BF16 eager/Inductor/manual-Triton microbenchmark stopped at the
Inductor RMS/modulation precision gate, before timed rounds. Windows CUDA
compilation/execution with the pluggable allocator succeeded, but 9378/4194304
elements exceeded the approved `rtol=atol=1/128` eager-parity bound; max absolute
difference 0.125. Generated code fused reduction/modulation while omitting
intermediate BF16 rounding. The default-policy artifact is preserved at
`E:\LTX-2\outputs\execution_backends_20261007_01`. There are no accepted
steady timings, manual-kernel comparison or offloading conclusions from this
attempt. A precision-cast-preserving repeat requires separate approval.

The approved repeat with `emulate_precision_casts=True` completed, preserving
the same workload, DLL and tolerances; all selected BF16 comparisons were exact.
Warm CUDA-event interval medians in milliseconds (eager / Inductor / manual
Triton): RMS/modulation `0.648862 / 0.158912 / 0.092768`; interleaved paired
rotary `0.976298 / 0.195134 / 0.166758`; split paired rotary
`0.597914 / 0.162181 / 0.137062`. First-call compile/cache time is separate.
Artifact: `E:\LTX-2\outputs\execution_backends_20261007_02\benchmark.json`.
One synthetic sequence, with some overlapping per-round ranges: not a
real-video speedup, general statistical conclusion or group-offloading result.
Original default-precision failure remains preserved; no default promotion.

## Real-video fusion A/B — 2026-10-07

Approved matched fresh-process pair: BF16, 1280x704, 8 seconds at 24 fps
(193 frames), seed 43, two steps, real prompt, image conditioning, no input
audio conditioning, joint audio generation, latent output. Compact timesteps
and in-place residuals off. Identical protected DLL, chunks and budgets;
only the candidate enabled RMS/modulation and paired-rotary fusion.

| Measurement | Baseline | Both fusions |
| --- | ---: | ---: |
| Total run, seconds | 135.6195 | 119.9721 |
| Denoise CUDA phase, seconds | 94.038078 | 81.589656 |
| Second step CUDA interval, seconds | 47.842949 | 40.142801 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.213604 |
| Sampled absolute GPU peak, GiB | 7.9995 | 7.2405 |
| Sampled process RSS peak, GiB | 28.701458 | 28.486149 |

Both completed with exit code 0, zero selected fault/admission/transport/cache
failures, quarantine and temporary fallbacks, and completed timing/cleanup.
Candidate submissions: 576 RMS/modulation and 384 paired rotary. Second-step
interval reduction approximately 16.1%; one ordered pair, not statistical
or final eight-step acceptance. First-step JIT cost was not isolated.

**Numerical acceptance failed:** CPU comparison found finite, nonidentical
latents. Video relative RMS difference `0.02875492`, max absolute difference
`0.87924790`; audio relative RMS difference `0.02205039`, max absolute
difference `0.10100091`. Recorded generation metadata and shapes matched.
The agreed no-divergence criterion was not met; do not promote or claim
quality parity. Small-contract exactness did not generalize to this real model.
Cause remains unestablished. Both flags stay off by default; no additional
CUDA test or relaxed tolerance was authorized from this result.

Artifacts: `E:\LTX-2\outputs\ddo_modular_video_20261007_121841_203` (baseline)
and `E:\LTX-2\outputs\ddo_modular_video_20261007_122144_968` (candidate).
Full conditions and comparison are in the sibling host's
`model_execution_optimizations.md`. Image routes and allocator defaults remain
unchanged; this is not an image regression gate or group-offloading comparison.

## Same-input real-video fusion diagnosis — 2026-10-07

Approved one-step diagnostic retained the preceding A/B's model, BF16,
1280x704, 8 seconds / 24 fps / 193 frames, seed 43, chunks, budgets and DLL.
It compared the first six RMS/modulation calls and four rotary pairs on the
same actual inputs, always returning the original eager results downstream.
Completed with exit code 0; all 14 comparisons finite, normal cleanup, zero
selected native failure/quarantine counters, ring fallback bytes and live source
tickets. Protected DLL/hash unchanged. This is not a performance benchmark.

Video RMS/modulation calls 1/3/5 differed in respectively 181/413/237 of
90112000 elements, with maximum absolute differences
`0.0078125 / 0.03125 / 0.00390625` and relative RMS differences
`0.000004788072 / 0.000011778770 / 0.000007793214`.
All three audio RMS/modulation calls and all eight sampled split-rotary outputs
were exact. The result localizes an operation-level numerical difference to
video RMS/modulation in this sample; it does not prove the arithmetic substage
responsible or explain all accumulated final-latent divergence. It is not
all-layer rotary acceptance. Both fusion defaults remain off.

Artifact:
`E:\LTX-2\outputs\fusion_same_input_20261007_130030_567\same_input_report.json`;
normal runner metrics and eager-return latents are preserved in its `run`
subdirectory. Host report: `model_execution_optimizations.md`. No additional
CUDA sequence or numerical-tolerance relaxation followed.

## Conservative modulation correction — 2026-10-07, CPU only

The opt-in callback now preserves the original norm forward and its installed
chunking, then fuses only modulation. Separate modulation dtype rounding is
retained; paired rotary and all protected runtime defaults/budgets/DLLs are
unchanged. Submission snapshots identify the original norm backend explicitly.
No historical artifact was rewritten; earlier fused-reduction timing and peak
results do not apply to this revision.

CPU-only validation: **72 passed in 6.24 s**, including eight new mocked-launch
handoff cases for FP32/BF16, affine/bias and chunked/unmodified norm forwards.
These are controller/regression checks, not kernel numerical acceptance.
The proposed small CUDA contracts followed by the same-input one-step gate
are consolidated in the sibling host's `validate_modulation_correction.ps1`.
No CUDA execution, real-model parity or speedup is claimed for this correction;
the sequence still requires user approval.

## Corrected modulation operation gates — 2026-10-07

User-approved consolidated sequence completed with exit code 0: all 36 small
CUDA comparisons were exact, followed by all 14 finite and bit-exact first-block
real-video same-input comparisons. The three video RMS/modulation calls that
previously differed now had zero differences. PyTorch `2.12.1+cu132` / CUDA
`13.2` / pluggable allocator; original norm forward plus Triton modulation,
with unchanged paired rotary, protected DLL/hash, budgets and defaults.

Encoder and transformer completed source-ticket collection with zero remaining
tickets, selected native failure/quarantine counters and ring fallback bytes.
Artifact:
`E:\LTX-2\outputs\fusion_same_input_20261007_133053_249\same_input_report.json`;
normal metrics and eager-return latent output remain in `run/`. CPU-only
comparison against the preceding diagnostic's saved video/audio latents was
bit-exact and finite, without CUDA initialization. Both diagnostics returned
original eager outputs downstream, so this is control consistency and sampled
operation validation, not full candidate-output parity or a speedup result.

No additional CUDA run or default promotion followed. Matched two-step final
latent/performance acceptance requires separate approval. The rejected earlier
fused-reduction candidate's 16.1% timing reduction does not apply to this revision.

## Corrected end-to-end modulation/rotary A/B — 2026-10-07

Approved matched fresh-process pair: BF16, 1280x704, eight seconds at 24 fps
(193 frames), two steps, seed 43, real prompt/image, no input audio conditioning,
joint audio generation and latent output. Same DLL, budgets and chunks as the
preceding A/B. Only two fusion flags and output directory differed in recorded
requested controls. Actual corrected candidate outputs fed the model.

| Measurement | Baseline | Corrected candidate |
| --- | ---: | ---: |
| Run total, seconds | 131.1571 | 124.7787 |
| Denoise CUDA phase, seconds | 91.723539 | 85.009664 |
| Second step CUDA interval, seconds | 44.555465 | 41.541945 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.504395 |
| Sampled absolute GPU peak, GiB | 7.9995 | 7.5312 |
| Sampled process RSS peak, GiB | 28.561218 | 28.452538 |

**Selected gate passed:** second-step reduction **6.7635%** (target at least 5%);
run total reduction **4.8632%**. CPU comparison found finite, bit-exact final
video/audio latents, with zero differing elements and zero maximum/RMS errors;
recorded generation metadata matched. No CUDA initialization in this comparison.
Both completed safely with zero selected native failure/quarantine/ring-fallback
counters and remaining source tickets, completed CUDA timing/timelines and
released events. Candidate submitted 576 composite norm/modulation and 384
paired rotary operations. DLL/hash, budgets and default routes unchanged.

One ordered two-step pair, not statistical proof, eight-step acceptance or
decoded-video quality/timing comparison. Flags remain off by default; no
further CUDA run or promotion followed. Artifact root:
`E:\LTX-2\outputs\corrected_modulation_ab_20261007_134003_382`.
Host details: `model_execution_optimizations.md`.

Memory interpretation: `global_peak_vram_gb` is a baseline-relative delta,
not absolute usage. Baseline here was `1.02685546875 GiB`; sampled absolute
phase peaks come from `peak_vram_absolute_gb`. The preceding October 7 fusion
table's label is corrected and its absolute row added; raw artifacts and
recorded delta values remain unchanged.

## Corrected eight-step video candidate closure — 2026-10-07

Approved single candidate: BF16, 1280x704, eight seconds at 24 fps (193 frames),
eight steps, seed 43, real prompt/image, no input audio conditioning, joint
audio generation and latent output without decode. Only step count and artifact
directory differ from the accepted corrected two-step candidate. Same protected
DLL, budgets, chunks and explicit modulation/rotary callbacks; defaults unchanged.

| Measurement | Corrected candidate |
| --- | ---: |
| Run total, seconds | 375.6927 |
| Denoise CUDA phase, seconds | 335.218125 |
| First step CUDA interval, seconds | 43.324703 |
| Mean steps 2-8 CUDA interval, seconds | 41.690544 |
| Steps 2-8 minimum / maximum, seconds | 40.442004 / 43.348531 |
| Sampled baseline-relative GPU peak, GiB | 6.655048 |
| Sampled absolute GPU peak, GiB | 7.6819 |
| Sampled process RSS peak, GiB | 28.466244 |

Selected closure passed with exit code 0 and finite saved video/audio latents
checked on CPU without CUDA initialization. Eight complete CUDA step intervals;
native timeline 2412 samples, zero dropped/failure/pending samples, released
events. Selected native failure/quarantine/ring-fallback counters and remaining
source tickets were zero. Transformer materializations 64, temporary fallbacks
zero; composite norm/modulation submissions 2304 and rotary-pair submissions
1536. Original norm forward and protected DLL/hash remained unchanged.

This is bounded safety and observed timing, not a matched eight-step speedup or
final numerical-parity comparison. Bit-exact final-output parity and measured
6.7635% second-step reduction belong to the accepted two-step A/B only. No
decoded-video timing/quality, general long-run stability, image-route validation
or default promotion is claimed. No additional CUDA run followed.

Artifact root:
`E:\LTX-2\outputs\corrected_modulation_8steps_20261007_140009_668\candidate`.
Host details: `model_execution_optimizations.md`. Memory deltas are relative
to the initial 1.02685546875 GiB baseline; absolute peaks are reported separately.

## Regional compilation integration gate — 2026-10-07

Approved fresh-process two-step control versus regional Inductor compilation,
using the same eight-second BF16 1280x704/24-fps workload, seed 43, prompt/image,
budgets, chunks, SDPA selection and protected allocator. Both optional execution
kernel callbacks were off. No external kernel was loaded.

Control completed and passed native safety/ticket collection and finite-latent
checks: total 128.0026 s, CUDA denoise phase 87.234930 s, step intervals
43.363313 / 43.832199 s. Sampled baseline-relative GPU peak 6.972656 GiB,
absolute GPU peak 7.9995 GiB, process RSS peak 28.673908 GiB.

Candidate **rejected at tracing** before completing a step: Dynamo attempted
to trace the native component forward's host materialization preamble and
encountered an unsupported ctypes `_FuncPtr.__call__` in the arena-info query.
48 repeated blocks were configured, but no completed compiled denoise or final
candidate output exists. Candidate timing/parity/completed cleanup counters are
unavailable; do not interpret configuration success as execution acceptance.
This is a native-hook/compilation boundary issue, not proof that Windows cannot
run Inductor. No performance comparison or speedup is claimed for compilation.

Artifacts: `E:\LTX-2\outputs\regional_compile_ab_20261007_152225_714`.
Stopped on failure without retry, tracing relaxation, hook removal or budget
change. Protected allocator hash and accepted independent candidate unchanged.

## Compute-only compilation/native small gate — 2026-10-07

After the module-call tracing rejection, the diagnostic host compiles only
declared compute forwards before native attachment. Existing native wrappers
remain outside that graph; no allocator code/binary or runtime defaults change.
This is an adapted regional experiment, not the unmodified Diffusers
`compile_repeated_blocks` module-call boundary.

Approved small gate passed: two generic BF16 units, width 128, batch 2, seed 43,
native auxiliary ownership, 0.07 GiB native ceiling, 2 MiB staging, 64 MiB source/
registration limits. Four finite, bit-exact outputs against eager CUDA (two
default-stream and two non-default-stream invocations), zero selected native
failure/quarantine/ring-fallback counters, zero remaining source tickets and
completed native retirement/compute-forward restoration. Source read bytes
remained 132776, with no reread after the first invocation.

PyTorch `2.12.1+cu132`, CUDA 13.2; Inductor fullgraph/static, precision-cast
emulation enabled, CUDA Graphs disabled. Protected DLL/hash unchanged.
Artifact: `E:\LTX-2\outputs\regional_vmm_gate_20261007_154748_190\report.json`.
This is bounded integration acceptance, not LTX parity/performance or general
model support. A corrected real-model A/B remains separately approval-gated.

## Compute-only regional LTX A/B — 2026-10-07, parity rejected

Approved corrected two-step pair with the same eight-second BF16 workload,
budgets, chunks and protected allocator. Only compilation flag/output directory
differed. Native materialization stayed outside the compiled compute forward;
the earlier ctypes tracing failure did not recur. No extra kernels were enabled.

| Measurement | Eager control | Compute-only Inductor |
| --- | ---: | ---: |
| Run total, seconds | 142.3055 | 160.0945 |
| CUDA denoise phase, seconds | 101.783156 | 120.570742 |
| First CUDA step interval, seconds | 54.160285 | 81.907914 |
| Second CUDA step interval, seconds | 47.598875 | 38.637332 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.515648 |
| Sampled absolute GPU peak, GiB | 7.9995 | 7.5425 |
| Sampled process RSS peak, GiB | 28.591766 | 28.966026 |

Observed second-step reduction 18.8272%, with larger total due to the initial
interval including compilation. One ordered pair with a slower control than
previous rounds; not statistical proof or a direct ranking against those rounds.
Both completed with finite outputs and passed selected native safety/ticket/
timing checks, but final **bit-exact parity failed** and the sequence exited 1.
CPU-only comparison: video max error 0.964721680, RMS 0.031353850, relative RMS
3.4264475%; audio max 0.093261838, RMS 0.013786504, relative RMS 2.2103660%.
No CUDA initialization during comparison. Numerical cause is not localized yet.

Artifact root: `E:\LTX-2\outputs\regional_compile_ab_20261007_160559_683`.
No tolerance relaxation, retry or promotion; protected DLL/budgets/defaults
unchanged. Accepted independent kernels keep their earlier bit-exact evidence.

## Isolated compiled RMSNorm diagnostic — 2026-10-07

Approved bounded synthetic comparison, no LTX checkpoint/native allocator or
external kernel: BF16 non-affine RMSNorm, width 4096, 1024 tokens, seed 43.
PyTorch 2.12.1+cu132 / CUDA 13.2, fullgraph/static Inductor, precision-cast
emulation enabled and CUDA Graphs disabled. Both outputs finite, but 34 of
4194304 elements differed; maximum absolute error 0.015625, RMS error
0.000019602589408010927. Exit 1 at the first mismatch; remaining cases not run.

This localizes a reproducible isolated numerical difference to compiled
RMSNorm, not the complete real-model divergence. No performance measurement,
parity relaxation, allocator change, rerun or default promotion is implied.
Artifact: `E:\LTX-2\outputs\compiled_rms_gate_01\report.json`.

## Isolated opaque eager RMSNorm CUDA gate — 2026-10-07

Approved correction diagnostic passed with exit 0: four BF16 synthetic cases,
4096/2048 channels, non-affine/affine, 1024 tokens, seed 43. All outputs finite
and bit-exact, zero differing elements and maximum/RMS errors. Fullgraph/static
Inductor calls the installed eager RMSNorm behind an opaque operator;
precision-cast emulation on and CUDA Graphs off, PyTorch 2.12.1+cu132 / CUDA 13.2.
No LTX checkpoint, native allocator or external kernel involved.

This accepts the isolated numerical boundary, not full-model parity,
native integration or performance. Defaults and protected allocator unchanged.
Artifact: `E:\LTX-2\outputs\compiled_rms_opaque_gate_01\report.json`.

## Regional compilation with opaque eager RMSNorm — 2026-10-07, parity rejected

Approved matched two-step/eight-second BF16 pair, original budgets and protected
allocator; candidate 48 compute blocks and 384 opaque installed eager RMSNorm
boundaries, chunk size 1024, extra kernels off. Both generations completed with
finite outputs and passed selected native safety/event/ticket checks, but the
sequence exited 1 at the zero-tolerance final video comparison.

| Measurement | Eager | Compiled with opaque RMSNorm |
| --- | ---: | ---: |
| Total runner, seconds | 220.9844 | 156.4098 |
| CUDA phase, seconds | 179.687781 | 115.338961 |
| First CUDA step, seconds | 130.394578 | 67.002844 |
| Second CUDA step, seconds | 49.266117 | 48.312063 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.811882 |
| Sampled process RSS peak, GiB | 28.824039 | 28.898205 |

Only approximately 1.94% second-step reduction in this ordered pair. The unusually
slow eager first interval prevents interpreting the total difference as a
reliable compilation speedup. CPU-only final comparison: video maximum error
0.906602740, RMS 0.029376684, relative RMS 3.2103767%; audio maximum error
0.093359470, RMS 0.014644632, relative RMS 2.3479481%. Both tensors finite.
The isolated RMSNorm fix is insufficient for full-model parity. No repeat,
threshold relaxation or default promotion; accepted independent kernels intact.

Artifact: `E:\LTX-2\outputs\regional_compile_eager_rms_ab_20261007_171446_372`.

## Bounded compiled-operation real-input diagnosis — 2026-10-07

Approved first selected block audit, original workload geometry/native budgets:
six RMS/modulation calls plus four rotary pairs, 14 compared outputs, all finite
and bit-exact on identical real inputs. Operation-level fullgraph/static
Inductor, opaque installed eager RMSNorm, 1024 chunks, precision-cast emulation,
CUDA Graphs off; independent Triton kernels not executed. Protected DLL unchanged.

Expected cutoff after selected coverage, no completed model generation.
Diagnostic reports `diagnosed`; partial runner reports `failed` with the expected
`AuditStop`. No full transformer safety/retirement/timing gate is implied by the
wrapper exit 0. Full-block numerical divergence remains unlocalized; selected
independent operations do not prove equivalent cross-operation graph lowering.
No rerun or default promotion.

Artifact: `E:\LTX-2\outputs\compiled_ops_first_block_20261007_173501_968\same_input_report.json`.

Coverage clarification after read-only source/generated-code review: candidate
C's opaque boundaries cover Diffusers RMSNorm only. Q/K modules are
`torch.nn.RMSNorm` and remained compiler-visible. The selected 14-output audit
starts rotary comparisons after eager Q/K normalization; it does not validate
that normalization. This uncovered path is not yet a proven numerical cause.
No additional CUDA test or accepted-performance/default change followed.

## Isolated PyTorch Q/K RMSNorm diagnosis — 2026-10-07, parity rejected

Approved bounded test of installed `torch.nn.RMSNorm`: BF16, non-affine,
4096 channels, 1024 tokens, seed 43, fullgraph/static Inductor with precision
casts preserved and CUDA Graphs off. Both outputs finite, but 34 of 4194304
elements differed; maximum absolute error 0.015625, RMS error
0.000019602589408010927. Exit 1 at first mismatch; remaining cases not run.
PyTorch 2.12.1+cu132 / CUDA 13.2, no LTX/native allocator/external kernel.

This demonstrates numerical drift in the normalization type left outside the
candidate's Diffusers-only boundary; it does not establish the whole model's
cause. Preserve each backend's original norm semantics in any future correction.
No retry, default promotion or accepted-runtime modification.
Artifact: `E:\LTX-2\outputs\compiled_torch_rms_gate_01\report.json`.

## Per-type opaque RMSNorm regional compile checkpoint — 2026-10-07, accepted two-step pair

Approved consolidated sequence passed: four isolated opaque PyTorch RMSNorm
cases finite/bit-exact, then the original eight-second/two-step BF16 real-model
eager-versus-compiled pair. Final video and audio latents finite and bit-exact;
selected native timing/event/failure/source-collection checks passed for both.
48 compiled compute blocks, 384 Diffusers plus 576 PyTorch opaque original
norms, no new Q/K chunking; optional fused kernels off. Fullgraph/static Inductor,
precision-cast emulation, no CUDA Graphs; unchanged budgets and protected DLL.

| Measurement | Eager control | Compiled with both RMSNorm types opaque |
| --- | ---: | ---: |
| Total runner, seconds | 146.5828 | 143.0399 |
| CUDA phase, seconds | 104.887734 | 101.953273 |
| First CUDA step, seconds | 55.217703 | 59.663430 |
| Second CUDA step, seconds | 49.631367 | 42.266355 |
| Sampled baseline-relative GPU peak, GiB | 6.972656 | 6.402809 |
| Sampled process RSS peak, GiB | 29.206753 | 28.867348 |

Approximately 14.84% second-step reduction and 3.5429 s lower total in this
ordered pair. First step includes lazy compilation, cache effects not isolated.
No statistical/general-support claim or direct ranking against independent
kernels from another pair. No eight-step compiled run, decode validation,
extra benchmark or default promotion. Earlier numerical rejections preserved.

Artifacts: `E:\LTX-2\outputs\compiled_torch_rms_opaque_20261007_183143_283\report.json`
and `E:\LTX-2\outputs\regional_compile_both_rms_ab_20261007_183149_787`.

## Single compiled eight-step candidate — 2026-10-07, completion/safety accepted

Approved unchanged eight-second BF16 workload, 1280x704/24 fps, seed 43,
real prompt/image, no input audio conditioning, latent output. 48 compiled
blocks, 384 Diffusers and 576 PyTorch original opaque norms; extra kernels off.
Original budgets/native DLL preserved, no retuning or automatic run extension.

| Measurement | Compiled candidate |
| --- | ---: |
| Total runner, seconds | 395.0373 |
| CUDA phase, seconds | 353.873750 |
| First CUDA step, seconds | 46.933320 |
| Steps 2–8 mean, seconds | 43.845735 |
| Steps 2–8 minimum / maximum, seconds | 42.366328 / 44.580965 |
| Sampled baseline-relative GPU peak, GiB | 6.413925 |
| Sampled process RSS peak, GiB | 28.492538 |

Eight complete CUDA intervals; 2412 timeline samples, zero dropped/failure/
pending samples, events released. Selected native fault/admission/pool/transport/
pressure checks passed, encoder/transformer completion collections zero tickets.
64 materializations, zero temporary fallbacks. Final video/audio finite in
CPU-only checks. No matched eight-step parity, decode or general-support claim.
First interval includes lazy compilation; cache/compile costs not isolated.

Historical independent-kernel eight-step observation was 41.690544 s/step and
375.6927 s total in a separate run. Do not interpret that nonmatched comparison
as proven regression or compiler advantage; no accepted default replacement.
Artifact: `E:\LTX-2\outputs\regional_compile_both_rms_8steps_20261007_191352_468\candidate`.

### External Q/K kernel — corrected small-operation oracle gate (October 7)

The historical small gate used Diffusers RMSNorm, but installed LTX attention
Q/K uses `torch.nn.RMSNorm`. A separately approved gate using that original
PyTorch operation passed all 12 BF16 comparisons bit-exactly: zero differing
elements, maximum and RMS error zero; finite outputs and no CUDA execution
error reported. Widths 2048/4096, token pairs 17/17 and 17/5, seed 43,
non-default stream and `rtol=atol=1/128` unchanged. Both selected binary hashes
stayed unchanged. No third-party CUDA math or DDO runtime/default changed.

Command: `& E:\LTX-2\run_external_qk_rope_gate.ps1 -TorchRmsNorm`.
Artifact: `E:\LTX-2\outputs\external_qk_rope_torch_rms_gate_20261007_204831_328\report.json`.
The earlier Diffusers-oracle failure remains preserved, not evidence of failure
against the actual Q/K operation. This is not a model performance result or
final-latent/decoded-media acceptance. No model benchmark followed.

### External Q/K compiled model A/B — final parity rejection (October 7)

The separately approved sequence compared regional compilation with both eager
norm boundaries, without/with the pinned attributed external Q/K operation.
Selected 1280x704, eight-second/24-fps, two-step, seed-43 BF16 native workload;
no input audio conditioning, joint audio generation, latent output/no decode.
Budgets/chunks and both pinned binaries unchanged. No eager/independent-kernel
control in this pair.

| Observation | Compiled control | Compiled + external Q/K |
| --- | ---: | ---: |
| Total runner seconds | 123.6231 | 130.7741 |
| CUDA phase seconds | 82.733921875 | 91.699000000 |
| First CUDA step seconds | 43.068718750 | 52.21701953125 |
| Second CUDA step seconds | 39.64747265625 | 39.46130859375 |
| Sampled baseline-relative GPU peak GiB | 6.322315216 | 6.038021088 |
| Sampled process RSS peak GiB | 28.502681732 | 28.856861115 |

Both runs passed selected native safety/completion/collection checks: 606
timeline samples each, zero drops/failures/pending, released events, zero final
encoder/transformer source tickets and selected failure counters. Each recorded
16 materializations, zero temporary fallbacks. External callbacks configured on
288 attention modules and retired; installed opaque-norm counts are not calls.
The A/B exited 1 on final parity, despite individual runner completion.

CPU-only comparison found finite but non-bit-exact final video/audio, relative
RMS differences 3.038259483% / 2.435441215%. The synthetic kernel gate passing
does not establish model parity or identify the sole cause of this divergence.
Second-step difference only 0.1861640625 s (about 0.47%); one ordered pair does
not demonstrate a reliable advantage. Candidate total 7.1510 s longer; lazy
compilation/cache costs not isolated. No default promotion or CUDA retry.
Command: `& E:\LTX-2\run_corrected_modulation_ab.ps1 -ExternalQkRopeAb`.
Artifact: `E:\LTX-2\outputs\regional_compile_external_qk_ab_20261007_211322_386`.

Comparison coverage is consolidated in the sibling host's
`E:\LTX-2\execution_comparison_summary.md`: measured pairs, numerical acceptance,
published external-reference results kept separate, and unimplemented/deferred
combinations. A new independent Triton fused Q/K norm/RoPE route without/with
regional compilation now has an optional prototype, not CUDA-validated or benchmarked. Its same-input
actual-model operation parity must precede any timing matrix. No CUDA run,
budget/default change or author message is authorized by this documentation update.

Independent prototype preparation: host CPU suite passed 105 tests in 11.33 s;
the two-stage launcher command preview passed without CUDA. Existing allocator,
kernels and defaults are unchanged. Pending separate CUDA approval, the gate
requires 12 bit-exact synthetic comparisons followed by bounded first-block
actual-input comparisons (at most 96 sampled token rows per component, four Q/K
calls across two candidates). Stop on the first failed stage or mismatching pair;
no timing matrix or full-generation acceptance is implied. Host diagnostic:
`E:\LTX-2\run_independent_qk_gate.ps1`.

The approved independent Q/K gate stopped at Triton compilation before any
comparison or model load (exit 1). An inferred reshape dimension needed an
explicit `tl.constexpr` annotation. Preserved report:
`E:\LTX-2\outputs\independent_qk_gate_20261007_223212_835\small\report.json`.
Both protected native/external binary hashes are unchanged. The annotation was
corrected and the CPU suite passed again (105 tests, 10.16 s); corrected-source
CUDA execution remains pending separate approval. No numerical or performance
acceptance is established by this attempt.

The separately approved corrected repeat passed all 12 synthetic CUDA Q/K
comparisons bit-exact. Actual-input strict parity then failed at the first video
self-attention pair: 96 sampled rows / 393,216 BF16 elements per output. External
Q/K differed in 1/2 elements (max absolute 0.0001220703125 / 0.0078125);
independent Triton Q/K differed in 1/3 elements (max absolute 0.0001220703125
for both). All outputs finite; no regional compilation in this diagnosis.
Sequence exited 1 and stopped before later attention calls or a completed step.
Both protected binary hashes remained unchanged. Artifact:
`E:\LTX-2\outputs\independent_qk_gate_20261007_223423_023`.
No timing benchmark or default promotion. These small operation differences do
not establish the cause of earlier final-latent drift or a platform limitation;
sample/full-input eager normalization and separate norm/RoPE arithmetic remain
unisolated. Full generation/retirement is not accepted by a diagnostic cutoff.

A one-pair arithmetic-isolation diagnostic is prepared in the host launcher
(`-ArithmeticIsolation`), without repeating synthetic gates or benchmarks.
It separates original sampled/full normalization and RoPE from candidate
identity-RoPE normalization and fused-rotation differences. Sixteen scalar
records; stop after the first pair; successful collection is not parity
acceptance. CPU suite passed 106 tests in 11.04 s; command preview passed.
CUDA diagnosis remains pending separate approval; protected defaults/binaries
are unchanged.

Approved arithmetic-isolation result (first actual video self-attention pair):
all 16 records finite; original sampled/full RMSNorm and RoPE comparisons exact.
Candidate normalization via identity RoPE differed in external Q/K 2/2 elements,
independent Triton Q/K 1/2 elements, per 393,216-element output. Independent norm
max absolute difference 0.0001220703125 for both; external 0.00048828125 /
0.00390625. Fused rotation versus original RoPE of the same candidate norm was
bit-exact for both candidates. This isolates observed first-pair differences to
normalization, not sampling or rotation in this case, without proving the cause
of prior full-generation drift or a platform defect. Sixteen records collected,
exit 0 means diagnosis complete, not parity accepted. Protected binaries unchanged;
no benchmark/default change. Artifact:
`E:\LTX-2\outputs\independent_qk_gate_20261007_224239_203\actual_inputs\audit.json`.

Independent normalization correction prepared: explicit four-member FP32 FMA
accumulation and fixed 128-logical-lane reduction order, four warps for both
widths. Final BF16 boundary, RoPE, allocator and defaults unchanged. CPU suite
108 passed in 11.53 s; subsequent launcher contracts 15 passed in 7.16 s.
Corrected-source CUDA compilation/parity remains unvalidated. Pending approval,
`run_independent_qk_gate.ps1 -ArithmeticIsolation -ValidateSynthetic` requires
12 bit-exact synthetic comparisons before the bounded first-pair arithmetic
diagnosis. No timing or final-generation acceptance is implied.

Approved grouped-reduction validation stopped during Triton compilation before
comparisons or model load: a loop-local constexpr could not be redefined by
static unrolling. Artifact:
`E:\LTX-2\outputs\independent_qk_gate_20261007_225458_012\small\report.json`.
Removed the constexpr declaration without changing the arithmetic schedule;
CPU suite 108 passed in 10.67 s. Protected binary hashes unchanged. Corrected
CUDA compilation/parity remains pending separate approval; no benchmark.

Approved corrected repeat:
`E:\LTX-2\outputs\independent_qk_gate_20261007_225729_339`.
All 12 synthetic comparisons bit-exact. On the first actual video self-attention
pair (96 sampled rows, width 4096), independent normalization, rotation and
combined Q/K outputs now all match exactly, with zero maximum/RMS error.
Original sampled/full controls exact; unchanged external candidate reproduced
its earlier discrepancies. Independent source SHA256:
`7a6a4ad43121b10ecd1896e99ea5ed2fc590dcec9a6129d0fc64fa40422c6f18`.
Both protected binaries unchanged. Sixteen finite diagnostic records collected;
exit 0 is bounded diagnosis completion, not model/performance acceptance.
Real audio/cross-attention, later blocks, full-shape and compiled-model parity
remain unvalidated. No benchmark, tolerance relaxation or default promotion.

The bounded four-call independent-only gate is prepared in the sibling host:
`run_independent_qk_gate.ps1 -IndependentOnly`. Eight finite bit-exact sampled
Q/K comparisons are required across first-block video/audio self-attention and
both audio/video cross directions; stop on the first mismatch or fourth call.
Text-cross attention without RoPE is outside this callback. Original outputs
continue through earlier calls; no full-step/model or performance acceptance.
Unchanged kernel, budgets, binaries/defaults; no repeated synthetic gate.
CPU suite 109 passed in 12.14 s; command preview passed. CUDA execution awaits
separate approval.

Approved independent-only first-block Q/K gate completed (exit 0,
`bounded_operations_passed`): four calls / eight finite bit-exact outputs.
Video self-attention full token counts 22000/22000, width 4096; audio self
201/201, width 2048; audio-to-video 22000/201 and video-to-audio 201/22000,
width 2048. Shared tables on self-attention; distinct tables on cross directions.
Each comparison samples 96 rows, zero maximum/RMS error. Unchanged independent
source hash and both protected binaries. Artifact:
`E:\LTX-2\outputs\independent_qk_gate_20261007_230632_822\actual_inputs\audit.json`.
Expected fourth-call cutoff; original outputs propagate in earlier calls.
No complete step, candidate propagation, later-block, compiled-model, retirement
or performance acceptance. No default promotion or benchmark.

Optional independent Q/K runner integration prepared in the sibling host;
default off, native route only, no allocator/residency/default changes.
CPU suite passed 116 tests in 13.37 s. Proposed matched two-step eager-control
versus fused-Q/K candidate command:
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -IndependentQkRopeAb`.
Preview passed; selected workload/budgets unchanged. Require selected native
safety/completion/retirement gates, preserved source/binary hashes, finite and
bit-exact final video/audio latents. No CUDA execution yet; compiled comparison
and eight-step cases are not automatically authorized by preparing this pair.

## Independent grouped fused Q/K: approved two-step model pair

Artifact: `E:\LTX-2\outputs\independent_fused_qk_ab_20261007_233301_602`.
Command: `& E:\LTX-2\run_corrected_modulation_ab.ps1 -IndependentQkRopeAb`.
Selected native staged BF16 1280x704, eight seconds/24fps, two steps, seed43,
no input audio, latent-only; existing budgets/chunks unchanged; no compilation.
Both completed with finite and bit-exact final video/audio; A/B exit 0.

| Metric | Control | Independent fused Q/K |
| --- | ---: | ---: |
| Total runner seconds | 152.3389 | 123.8883 |
| CUDA phase seconds | 113.2387421875 | 85.1459453125 |
| First CUDA step seconds | 46.3002734375 | 43.67172265625 |
| Second CUDA step seconds | 66.9156484375 | 41.4525546875 |
| Sampled baseline-relative GPU peak GiB | 6.97265625 | 6.621509552001953 |
| Sampled process RSS peak GiB | 28.825973510742188 | 28.496761322021484 |

Both passed selected native safety/retirement: 606 timeline samples each,
zero drops/failures/pending, events released; final component source tickets
zero and selected failure counters zero. Each 16 materializations, zero temporary
fallbacks. Candidate configured on 288 attentions, denoising complete, callback
retired; accepted source and allocator hashes unchanged. Control second interval
unusually slow (cause unisolated): no reliable speedup percentage or ranking
against earlier different pairs. Selected two-step numerical/integration acceptance
only; no decoded/eight-step/general-model/compiled-candidate acceptance or default
promotion. Further compiled comparison requires separate approval.

## Independent grouped fused Q/K plus regional compilation

Approved command:
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -IndependentQkRopeAb -RegionalCompileAb`.
Artifact: `E:\LTX-2\outputs\regional_compile_independent_fused_qk_ab_20261007_234009_127`.
Same selected two-step BF16 native workload/budgets; both compiled with original
norms opaque, Q/K candidate only difference. Both completed with finite,
bit-exact final video/audio; sequence exit 0. Selected native source/timeline/
admission/transport/retirement checks passed. Each 606 samples, no failed/dropped/
pending, events released; final source tickets zero; 16 materializations and no
temporary fallback. Candidate 288 callbacks configured, completed and retired;
accepted source/native hashes unchanged.

| Metric | Compiled control | Compiled + independent Q/K |
| --- | ---: | ---: |
| Total runner seconds | 122.1532 | 129.0745 |
| CUDA phase seconds | 82.5077109375 | 89.5672421875 |
| First CUDA step seconds | 43.276421875 | 49.12148828125 |
| Second CUDA step seconds | 39.21261328125 | 40.42636328125 |
| Sampled baseline-relative GPU peak GiB | 6.346477508544922 | 6.114017486572266 |
| Sampled process RSS peak GiB | 28.502220153808594 | 28.549884796142578 |

No incremental gain observed: second interval 1.21375 s / about 3.10% longer,
total 6.9213 s longer. One ordered pair is not a general regression claim;
compilation/cache/run variation not isolated. Bounded eager/compiled independent
comparison now recorded with accepted selected parity, without reliable added
kernel speedup. Existing defaults protected; no automatic retry, tuning matrix,
eight-step model/general-platform acceptance or author message.

## Previous Triton kernels with regional compilation: preparation only

An isolated optional compute route now wraps the existing modulation and paired
RoPE kernels as opaque operations. It keeps the original norm forward and the
previous kernels' arithmetic/launch geometry, without changing residency,
budgets, allocator source/binary, or existing runtime defaults. The independent
grouped Q/K normalization candidate is not part of this comparison.

131 targeted CPU capture/routing/launcher regression contracts passed in 16.04
seconds. The protected allocator hash remained unchanged. CUDA correctness and
performance are pending separate confirmation. The prepared host command is
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompiledTritonAb`: two small CUDA
compute cases first, then one matched compiled-control/compiled-plus-previous-
Triton pair at the selected 1280x704 / 8-second / 24-fps / two-step workload.
Stop at a failed gate; require finite bit-exact final video/audio and existing
native safety checks. No automatic eight-step run, default promotion or new
full-normalization fusion. Without a clear incremental benefit, close this
branch rather than extending a tuning matrix.

## Previous Triton kernels with regional compilation: approved bounded result

Approved command: `& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompiledTritonAb`.
Artifact: `E:\LTX-2\outputs\regional_compile_triton_execution_ab_20261008_000702_294`.
Two small compute CUDA contracts passed in 8.06 seconds: selected original,
opaque and compiled outputs finite and bit-exact. The subsequent native real-
model control/candidate completed with finite, bit-exact final video/audio.
Sequence exit 0; selected native gates passed. Each recorded 606 timeline
samples, zero failures/drops/pending, events released, source tickets zero,
16 materializations and no temporary fallback. Candidate 48 block / 288
attention callback bindings configured, completed and retired; these are not
invocation counts. Original norm boundaries, protected allocator and previous
kernel source preserved. Wrapper hash:
`39b6470bc0aa1340a8185f1ed31f2e13626d9e56945b65cd4923c33e2230098a`.

| Metric | Compiled control | Compiled + previous Triton |
| --- | ---: | ---: |
| Total runner seconds | 126.6132 | 132.9614 |
| CUDA phase seconds | 85.1692109375 | 94.8239921875 |
| First CUDA interval seconds | 44.9488203125 | 53.854515625 |
| Second CUDA interval seconds | 40.19797265625 | 40.9497734375 |
| Sampled baseline-relative GPU peak GiB | 6.332401275634766 | 6.185237884521484 |
| Sampled process RSS peak GiB | 28.494911193847656 | 28.857677459716797 |

No gain observed: second interval about 1.87% slower, total 6.3482 seconds
longer. One ordered pair is not a general slowdown claim, and first intervals
do not isolate compilation/cache costs. Numerical/native integration accepted
only for this selected two-step latent workload; no decoded/eight-step/general-
model acceptance or default promotion. Close the bounded combination branch
without repeat or further fusion tuning. Next agreed opportunities are compact
timesteps and in-place residuals; new CUDA execution requires confirmation.

## Compact timestep/residual combination: preparation only

Prepared host command (CUDA approval pending):
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualAb`.
Both sides use regional compilation and opaque original norm boundaries;
candidate alone enables the already-implemented compact timesteps and in-place
residual additions. No compute kernel, budget, allocator binary or default is
changed. One small BF16 CUDA contract must pass exact output/input/projection-row
checks before one matched native real-model pair at the selected 1280x704,
8-second, 24-fps, two-step workload. Retain native checks and bit-exact final
video/audio; require executed compact layout rather than only requested flags.

CPU preparation passed 137 contracts and 11 subtests in 20.23 seconds. These
are not CUDA numerical or performance acceptance. Combined benefit, if any,
must be assessed within this pair without claiming individual attribution.
No automatic repeated/eight-step runs, tolerance relaxation, budget changes,
or promotion; existing image, linear/NVMe and native accepted routes preserved.

## Compact timestep/residual combination: approved bounded result

Approved command: `& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualAb`.
Artifact: `E:\LTX-2\outputs\regional_compile_compact_residual_ab_20261008_001757_883`.
Small BF16 CUDA compute gate passed in 21.36 seconds with bit-exact outputs,
unchanged caller tensors/checkpoint keys, projection rows 16 -> 4. TF32 remained
disabled despite the compiler warning. Both native real-model runs completed;
finite bit-exact final video/audio, sequence exit 0. Selected native gates
passed: each 606 samples, zero failures/drops/pending, events released, final
source tickets zero, 16 materializations and no temporary fallback. Candidate
reports compact and residual controls executed with timestep repeat 880; control
requested both false (missing execution fields remain unavailable). Protected
allocator, budgets, original opaque norms and earlier routes unchanged.

| Metric | Compiled control | Compiled + compact/residual |
| --- | ---: | ---: |
| Total runner seconds | 121.1271 | 129.5201 |
| CUDA phase seconds | 83.921671875 | 91.6996796875 |
| First CUDA interval seconds | 43.69741796875 | 51.6938828125 |
| Second CUDA interval seconds | 40.2056171875 | 39.99374609375 |
| Sampled global baseline-relative GPU peak GiB | 5.560756683349609 | 3.935546875 |
| Sampled denoise baseline-relative GPU peak GiB | 5.5608 | 3.8184 |
| Sampled denoise absolute driver GPU peak GiB | 6.5876 | 4.8452 |
| Sampled process RSS peak GiB | 28.495643615722656 | 29.0654296875 |

Only about 0.53% second-interval reduction; total 8.3930 seconds longer. No
demonstrated speedup; compilation/cache/run variability not isolated. Lower
sampled denoise GPU peak (1.7424 GiB) is the useful memory observation; candidate
global peak occurred in encoder. Driver samples are not a tensor allocation
census or total/shared WDDM commitment, and this combined pair cannot attribute
the effect to either control individually. RSS about 0.57 GiB higher.
Retain optional memory-oriented candidate with selected two-step parity/native
acceptance, not a new default or larger-workload guarantee. No automatic repeat,
eight-step/ten-second/decode/general-model acceptance or budget/image-route change.

## Single compact/residual eight-step validation: preparation only

Prepared command (CUDA approval pending):
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualEightSteps`.
One candidate only, reusing the accepted two-step configuration, native DLL,
original norm boundaries and budgets; only denoise step count increases to
eight. Duration remains eight seconds, 1280x704, 24 fps, seed 43, no input audio,
joint audio generation, latent-only output. No repeated small gate, new kernel,
decoder, ten-second trial or automatic second run. Thirty CPU launcher contracts
passed in 12.54 seconds. Completion requires eight complete timing intervals,
finite saved video/audio and existing native cleanup/admission/transport checks.
This is not a matched eight-step numerical comparison or performance claim.

## Single compact/residual eight-step candidate: approved completion

Approved retry command:
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualEightSteps -DisableTlsKeyLog`.
Artifact: `E:\LTX-2\outputs\regional_compile_compact_residual_8steps_20261008_100346_240`.
The first attempt stopped in SSL preflight before CUDA/model execution; the
approved process-scoped TLS-key-log opt-in passed preflight and its environment
was restored. No TLS certificate validation/encryption change. Same selected
1280x704 / eight-second / 24-fps / 193-frame BF16 workload and budgets as the
accepted two-step pair, now eight steps. No input audio conditioning, joint
audio generation, latent-only output; no decoder or second run.

Completed, exit 0, saved video/audio finite. Both optimizations executed, repeat
880, original opaque norm/compilation configuration completed. Native gates
passed: eight timing intervals, 2412 samples, zero failures/drops/pending,
events released, final source tickets zero; 64 materializations, no temporary
fallback. Protected DLL unchanged; earlier image/linear/NVMe/native routes intact.

| Metric | Single candidate |
| --- | ---: |
| Total runner seconds | 370.9189 |
| CUDA denoise phase seconds | 329.50753125 |
| First CUDA interval seconds | 49.71496484375 |
| Mean CUDA steps 2-8 seconds | 39.968712611607145 |
| Minimum/maximum CUDA steps 2-8 seconds | 39.14959765625 / 40.755296875 |
| Sampled global baseline-relative GPU peak GiB | 4.115234375 |
| Sampled denoise baseline-relative GPU peak GiB | 3.8184 |
| Sampled denoise absolute driver GPU peak GiB | 4.8452 |
| Sampled process RSS peak GiB | 28.501537322998047 |

Lower sampled denoise GPU usage persisted; global peak occurred in encoder
(sampled absolute 5.1421 GiB). Driver samples are not total/shared WDDM commitment
or a tensor census. Prior separate eight-step observations: corrected eager
Triton mean 41.690544 s / total 375.6927 s; opaque-norm regional compilation
mean 43.845735 s / total 395.0373 s. This run has the lowest observed mean/total
of those selected records, not a matched speedup claim. No eight-step numerical
reference comparison, decoded-quality/general-model/ten-second acceptance,
default promotion or automatic next benchmark.

## Single ten-second capacity trial: historical preparation

Prepared command (separate CUDA confirmation pending):
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualTenSeconds -DisableTlsKeyLog`.
One native compact/residual candidate, ten seconds at 24 fps (241 frames),
1280x704, BF16, seed 43, eight steps, no input audio conditioning, joint audio
generation, latent-only output. All settings/budgets from the accepted single
eight-second candidate remain unchanged; only duration and artifact paths differ.
Thirty-nine CPU launcher contracts passed in 14.76 seconds. Require finite
outputs, completion and existing native ownership/admission/transport checks.
No decoder/repeated small gate/second run, budget increase, resolution reduction,
criteria relaxation or default change. CUDA and ten-second capacity are pending.

## Single ten-second capacity candidate: approved completion

Approved command:
`& E:\LTX-2\run_corrected_modulation_ab.ps1 -CompactResidualTenSeconds -DisableTlsKeyLog`.
Artifact: `E:\LTX-2\outputs\regional_compile_compact_residual_10s_8steps_20261008_102137_296`.
One 1280x704 BF16 native compact/residual candidate, ten seconds / 24 fps / 241
frames / eight steps / seed 43, no input audio conditioning, joint audio
generation, latent-only output. Same selected DLL, original opaque norm boundaries,
compilation and budgets as the accepted eight-second candidate; no code/budget/
resolution change or second attempt. Process-scoped TLS key-log opt-in retained
certificate validation/encryption.

Completed without OOM, exit 0; saved video/audio finite. Both optimizations
executed, repeat 880. Native gates passed: eight timing intervals, 2412 samples,
zero failures/drops/pending, events released, final tickets zero; 64
materializations, no temporary fallback. Protected allocator hash unchanged.

| Metric | Ten-second single candidate |
| --- | ---: |
| Total runner seconds | 525.1151 |
| CUDA denoise phase seconds | 485.89515625 |
| First CUDA interval seconds | 71.39725 |
| Mean CUDA steps 2-8 seconds | 59.21119754464286 |
| Minimum/maximum CUDA steps 2-8 seconds | 58.349015625 / 61.05320703125 |
| Sampled global baseline-relative GPU peak GiB | 4.228515625 |
| Sampled denoise baseline-relative GPU peak GiB | 4.2285 |
| Sampled denoise absolute driver GPU peak GiB | 5.2554 |
| Sampled process RSS peak GiB | 29.214431762695312 |

Selected ten-second denoising capacity accepted, not decoded media/quality,
numerical-reference parity, longer-duration or general-model support. This
no-input-audio latent-only workflow is not matched evidence against historical
audio-conditioned limits. Eight/ten seconds differ in workload; do not interpret
their times as a regression A/B. Driver samples are not total/shared WDDM
commitment or a tensor census. Defaults/earlier routes intact; no automatic
next generation or decode. Saved latents permit separately approved decoding
without repeating the transformer.

### Selected ten-second saved-latent decode completion

The separately approved host offline decoder completed with exit code 0, without
OOM or rerunning transformer/text encoder. Existing temporal streaming video
decode and staged audio VAE/vocoder used normal PyTorch allocation, not a new
native allocator gate. Process elapsed 60.6969448 s (loading, decoding, encoding
and source verification included); video stage 51.0414008 s, audio VAE 0.6233013 s,
vocoder 1.5186058 s. Ten host CPU admission tests and a CUDA-free real-artifact
dry-run passed before execution.

ffprobe independently confirmed H.264 1280 x 704, 24 fps, 241 frames, 10.041667 s;
AAC stereo 48000 Hz, 10.010000 s. Pre-encode audio was finite, `[2, 480480]`.
Saved latent hash was unchanged. Host artifact:
`E:/LTX-2/outputs/regional_compile_compact_residual_10s_8steps_20261008_102137_296/candidate/ltx23_modular_10s_decoded.mp4`,
with adjacent `.decode.json` report. Selected media completion accepted; visual/
audio quality and decode-reference parity remain unreviewed. Separate-process
decode is not a matched full-pipeline timing; prior denoise evidence, defaults,
protected runtimes and native binary remain unchanged.

### Ten-second original-audio-conditioned selected capacity gate

The approved host trial used BF16, 1280 x 704, 24 fps, 241 frames, seed 43 and
eight steps with the prior native limits and compiled compact/residual controls.
The original 11.064-second local audio was cropped to the first ten seconds for
conditioning. Artifact: `E:/LTX-2/outputs/regional_compile_compact_residual_audio_10s_8steps_20261008_111849_823/candidate`.

Runner total 496.0545 s (latent-only); CUDA phase 453.3800625 s; first interval
59.23809375 s; mean intervals 2-8 56.30334040178571 s/it. Sampled denoise absolute
driver GPU peak 5.3823 GiB; global baseline-relative peak 4.35546875 GiB; process
RSS peak 28.66124725341797 GiB. No OOM; saved video/audio latents finite. Native
timeline complete with 2412 samples, events released and zero failure/drop/pending;
source tickets collected, selected fault/admission/transport/quarantine counters
zero, 64 materializations, zero temporary fallbacks, protected DLL unchanged.

The model completed, but the new launcher workload assertion initially returned
exit 1 by comparing RunTracker's phase-list `steps` field with integer eight.
After correcting this metadata-only assertion, the same artifact passed the full
safety gate in CPU-only verification; no inference retry or artifact modification.
Actual eight-step completion is checked by complete CUDA intervals. Selected
conditioned denoising accepted; decode/media quality and matched reference parity
remain pending. Do not infer a conditioning speedup from separate ordered runs.

The separately approved offline conditioned decode subsequently completed without
OOM: 60.3013161 s including loading/encoding/hash checks, no transformer/encoder
reload. Adjacent output `ltx23_modular_10s_audio_conditioned.mp4` and `.decode.json`.
ffprobe verified 241 H.264 frames, 1280 x 704, 24 fps, 10.041667 s and AAC stereo
48000 Hz, 10.010000 s. Waveform finite, original latent hash unchanged. Selected
conditioned media completion accepted; user synchronization/quality review pending.
Normal PyTorch allocation was used for the existing decoders; no native/default
change. This separate decode does not replace the latent-only timing evidence.

### Ten-second conditioned dense-versus-compact timestep observation

One approved dense control completed, exit 0, compared with the saved compact
conditioned run. Only timestep compaction changed; residuals in-place, compilation,
BF16, audio/image, 1280 x 704, 24 fps, 241 frames, eight steps, seed 43, native DLL
and ceilings remained the same. Control artifact:
`E:/LTX-2/outputs/regional_compile_dense_timesteps_audio_10s_8steps_20261008_113918_836/candidate`.

| Measurement | Dense | Compact |
| --- | ---: | ---: |
| Mean CUDA steps 2-8 s/it | 64.03968694 | 56.30334040 |
| Runner seconds, latent-only | 712.7422 | 496.0545 |
| First CUDA interval seconds | 222.415859375 | 59.23809375 |
| Sampled denoise absolute driver GPU peak GiB | 7.9995 | 5.3823 |
| Sampled process RSS peak GiB | 29.29014206 | 28.66124725 |

Compact observed 7.7363 s/it lower (about 12.08%) and 2.6172 GiB less sampled
denoise GPU peak. Total difference includes an unusually long dense first
interval; compilation/paging causality was not measured. Single ordered runs
are not statistical or general-model evidence. CPU comparison found final
video/audio latents bit-exact and finite. Dense control passed all selected
native gates, eight complete intervals, 2412 timeline samples, released events,
zero selected failure counters, collected tickets, 64 materializations and no
temporary fallbacks. 45 CPU launcher contracts passed before execution. No OOM,
budget changes, candidate rerun, extra decode or default/legacy route changes.
Retain compact timesteps as an optional selected-workload memory feature with
observed performance benefit; no wider-model support claim.

### Optional FP8 weight-only small native correctness gate

On 2026-10-08, the explicitly approved eight-layer 512-wide Windows sequence
passed BF16 control followed by TorchAO 0.18.0 FP8 weight-only, exit 0. Evidence:
`E:/LTX-2/outputs/fp8_native_gate_05/{bf16_control,fp8}/report.json`.
Each checkpoint was saved and reloaded as safetensors; existing backend/model
math forward was preserved. Three repeated outputs matched each case's own
CUDA reference exactly; this does not assert FP8 equals BF16. FP8 data/scales
were verified exactly. Both used native resident and streamed units, recovered
from intentional RuntimeError/KeyboardInterrupt, reused cached file bytes and
finished with zero transfer events, batch events and tickets after batch close.
Both completed meta retirement and store cleanup, without fault failures or
temporary fallbacks. Preserved DLL and production defaults were unchanged.

Physical parameter bytes were 4,202,496 BF16 and 2,121,728 FP8. Native residency
was 6 MiB for both due to the diagnostic's page/slot geometry; cache was 6 MiB,
staging 4 MiB, combined host ceiling 10 MiB, allocator cache 4 MiB. This is not
an LTX performance/quality benchmark or proof of VRAM reduction. Real LTX
weights remain BF16. Failed preparatory fixtures and corrected gate contracts
are documented in `docs/fp8-weight-only-integration-plan.md`, not counted as
passing runs. Reusable conversion tooling and explicit real-model integration
remain pending.

The subsequent selected public API gate passed both cases, exit 0, using saved
CPU fixtures: `E:/LTX-2/outputs/fp8_public_gate_02/{bf16,fp8}_report.json`.
This validates DDO loading, attachment, metrics and cleanup, not only direct
adapter calls. Three repeated outputs matched each case's own CUDA reference;
physical FP8 data/scales matched exactly. Native streaming, unchanged source
reads, interruption recovery and meta/source-descriptor retirement passed.
Fault failures, temporary fallbacks and cache admission denials were zero.
Both native residency snapshots were 6 MiB, with declared 64 MiB host cache,
4 MiB staging and 68 MiB combined host ceiling; DLL remained unchanged. This is
correctness evidence, not LTX FP8 quality/performance evidence. The reusable
converter and explicit loader/runner option are implemented; full LTX conversion
remains pending, requiring bounded-memory preparation and a separate output
drive. Earlier preparatory failures remain historical evidence, not passing gates.

Bounded offline conversion preparation subsequently passed 129 CPU tests and
six subtests in 16.50 s (`fp8_bounded_integration_cpu_01`). The tensor-at-a-time
converter matched the existing converter's saved payload hashes and the public
loader, without full CPU model loading. Original source files and untouched
tensors were preserved; invalid index and source-change failures retained
diagnostics. Header-only inspection found 4,186 BF16 source tensors, largest
288 MiB. Real output is planned at
`F:/models/ltx23-distilled-fp8-weight-only/transformer`; conversion and real-model
FP8 quality/performance remain pending. This is not a CUDA benchmark result.

On 2026-10-09 the approved bounded CPU conversion of the real transformer
completed, exit 0. Output:
`F:/models/ltx23-distilled-fp8-weight-only/transformer`. The source's 35.3695 GiB
of tensor data became 17.7235 GiB across ten safetensors shards, approximately
49.89% smaller (output includes headers). 1,660 linear weights were converted;
all saved FP8 bytes, FP32 scales and untouched tensors reconstructed exactly.
Source hashes matched between passes; original BF16 files were not modified.
Observed process peak working set was approximately 15.11 GiB. Report:
`conversion_report.json`, `passed`, exact reload true, CUDA initialized false.
No real-model inference ran; quality, VRAM and speed improvements remain
unmeasured. The accepted BF16 baseline and protected runtimes are unchanged.

The first approved real FP8 inference gate (1280x704, 5 s, 24 fps, four steps,
seed 43, audio conditioning) failed before transformer attachment/denoising:
CPU immutable-source binding rejected a recycled temporary flatten-clone
address between shards. Encoder/conditioning completed; no CUDA/OOM error was
reported and no FP8 denoise time was measured. Artifacts:
`E:/LTX-2/outputs/fp8_real_5s_4steps_gate_01.console.log` and gate metrics.
After limiting temporary validation registries to each shard, focused CPU
contracts passed 29 tests. Real CPU loading then passed with 4,186 logical
tensors, 5,846 correctly bound physical sources and exact saved payload hashes;
CUDA remained uninitialized. No DLL, runtime budgets or BF16 route changed.
Corrected CUDA generation remains unvalidated pending another approved attempt.
The consolidated corrected CPU integration suite passed 130 tests and six
subtests in 17.39 s; this does not replace real-model CUDA acceptance.

The approved unchanged retry `fp8_real_5s_4steps_gate_02` passed real checkpoint
loading/source binding but failed component layout preflight before denoising:
root-owned parameters overlap descendant execution units. Meta-only inspection
found two BF16 root tables totaling 24 KiB. The failed-run elapsed was
103.4579 s, load/prepare 32.5 s; this is not FP8 denoising performance.
No CUDA/OOM error was reported. A bounded component-lifetime root-state
extension is proposed, not implemented or accepted. No DLL, weight budget,
overlap guard or protected runtime changed.

The approved FP8-only bounded root-state extension is now CPU-accepted:
138 tests and six subtests in 17.23 s (`fp8_root_final_cpu_01`). It reuses one
existing native lifetime-pinned buffer and immutable source upload, subtracts
its page-rounded capacity from the unchanged total ceiling, includes backing
in residency metrics, and fences retirement/destruction. The dense root guard
and native DLL are unchanged. Real CPU geometry passed with 24 KiB root state,
76 execution units and a largest slot payload of 387,323,904 bytes. This is
not CUDA generation/performance acceptance. A small lifecycle CUDA gate is
required before another LTX attempt; neither ran in this implementation step.

The subsequently approved small root-state CUDA gate passed BF16 and FP8,
exit 0, approximately 6.09 s. Evidence:
`E:/LTX-2/outputs/fp8_root_public_gate_01/{bf16,fp8}_report.json`.
Each case matched its own reference exactly, recovered after RuntimeError and
KeyboardInterrupt, and completed cleanup. FP8 root payload was 24 KiB, uploaded
once into a 2 MiB native backing charged against its 8 MiB total fixture ceiling;
arena ceiling remained 6 MiB. Observed total weight residency was 8 MiB including
root backing. Faults/fallbacks/cache denials were zero and cached read bytes
did not increase across repeats. This is lifecycle acceptance only; real LTX
FP8 quality/performance and four-step generation remain unvalidated.

The separately approved real FP8 retry `fp8_real_5s_4steps_gate_03` completed
on 2026-10-09, exit 0: 1280x704, 5 s, 24 fps/121 frames, four steps, seed 43,
image/prompt/audio conditioning, native BF16 encoder, no compilation or custom
execution kernels, latent output. Transformer/encoder weight ceilings remained
1.5/4 GiB; chunk sizes 1024, phase modulation, compact timesteps and in-place
residuals enabled. Load/prepare: 33.0865 s; attach: 85.4993 s; denoise:
109.2867 s (progress approximately 27.09 s/it); total: 259.2684 s, without decode.
Sampled peak GPU: 5.4516 GiB baseline-relative / 6.4784 GiB absolute; peak RSS:
27.2126 GiB. Fault failures, temporary tensors and cache admission denials: zero.
Root state uploaded once within the unchanged ceiling; completion tickets
retired to zero and process cleanup completed without a reported error.
CPU-only saved-latent validation passed: finite video and audio tensors.
Evidence: `E:/LTX-2/outputs/fp8_real_5s_4steps_gate_03/metrics/modular_run.json`.
This is functional acceptance, not decoded quality, BF16 parity or evidence of
a performance improvement. The compiled 10 s BF16 baseline is not a matched
control; do not compare its per-step result directly with this gate.

The separately approved saved-latent decode completed with exit 0 on 2026-10-09,
without reloading transformer/encoder or rerunning denoising. Existing staged
BF16 decoders used ordinary CUDA allocation, not the native allocator. Elapsed:
33.7275 s (video 25.4463 s, audio VAE 0.5412 s, vocoder 1.2046 s).
Output: `E:/LTX-2/outputs/fp8_real_5s_4steps_gate_03/fp8_decoded.mp4`.
Independent ffprobe confirmed H.264, 1280x704, 24 fps, 121 frames/5.041667 s;
AAC stereo 48000 Hz, 5.010000 s; file size 1,023,116 bytes. Audio was finite.
Source latent SHA-256 remained
`c5a36eab7c22c74d6d90c9e8f42b331f5fb311728b4a8f39bf10f87da15c1373`.
Decode evidence is `fp8_decoded.decode.json` beside the MP4. Visual/audio quality
is not reviewed; matched BF16 quality/performance comparison remains pending.

### Approved five-second BF16 quality control (2026-10-09)

`E:/LTX-2/outputs/bf16_5s_4steps_fp8_quality_control_01` completed with exit 0.
Workload controls match the FP8 gate: 1280x704, 5 s/121 frames at 24 fps, four
steps, seed 43, same prompt/image/audio, chunk sizes 1024, phase modulation,
compact timesteps and in-place residuals, no compilation. Weight ceilings and
preserved DLL are unchanged. BF16 uses the existing dense route, whereas FP8
requires composite component ownership; these timings are an integration-level
comparison, not an isolated dtype/kernel experiment. Cache warmth is uncontrolled.

| Phase/observation | BF16 control | FP8 gate |
| --- | ---: | ---: |
| Transformer load/prepare (s) | 1.6204 | 33.0865 |
| Transformer attach (s) | 6.5577 | 85.4993 |
| Denoise, four steps (s) | 118.1183 | 109.2867 |
| Total without decode (s) | 155.3928 | 259.2684 |
| Sampled global GPU peak, absolute (GiB) | 4.9624 | 6.4784 |
| Sampled process RSS peak (GiB) | 26.6593 | 27.2126 |

BF16 fault failures, temporary tensors and cache admission denials were zero.
The report records 107 commit-pressure entries and recoveries, not zero pressure.
Completion collection retired two tickets to zero; cleanup completed without a
reported error. CPU-only validation found finite video/audio latents with matching
shapes. Relative latent RMSE, `RMS(FP8-BF16)/RMS(BF16)`, was 0.194170 for video
and 0.003182 for audio. This measures final latent divergence, not perceptual
quality or temporal stability; no visual conclusion follows from it alone.
BF16 decode/user quality review remains pending and requires separate approval.

The subsequently approved standalone BF16 decode completed with exit 0 in
32.7062 s, without rerunning denoising. Existing staged decoders used ordinary
CUDA allocation; transformer/encoder were not loaded. Output:
`E:/LTX-2/outputs/bf16_5s_4steps_fp8_quality_control_01/bf16_decoded.mp4`.
Independent ffprobe confirmed H.264, 1280x704, 24 fps, 121 frames/5.041667 s;
AAC stereo 48000 Hz, 5.010000 s; file size 1,000,593 bytes. Audio was finite
and source latents unchanged (SHA-256
`840ca4569bf5cf38e399350ff0e983226ac664ac210d9dfa194ee5b5d38ea2ed`).
Evidence: adjacent `bf16_decoded.decode.json`. Both BF16 and FP8 media are now
available for user review; no visual/audio quality verdict has been recorded.

The user subsequently reviewed both five-second/four-step videos and reported
similar temporal artifacts, with no clear visual degradation attributable to
FP8. This is an initial subjective acceptance for that pair only, not numerical
parity or general-model/eight-step quality acceptance. No enhance LoRA was used.

### Ten-second FP8 comparison stopped at BF16 control (2026-10-09)

The approved sequence started with uncompiled BF16, 1280x704, 10 s/241 frames
at 24 fps, eight steps, seed 43, image/prompt/audio, compact timesteps,
in-place residuals, phase modulation and 1024 chunk sizes. Diagnostic timing
was enabled; weight ceilings stayed 1.5/4 GiB and the protected DLL hash matched.
`E:/LTX-2/outputs/bf16_10s_8steps_fp8_comparison_01` failed with an illegal
CUDA memory access before the first completed-step callback. Exit 1, recorded
elapsed 70.8338 s; no latent artifact or completed denoising timing exists.
Cleanup/event recording also failed after the CUDA context entered error.
The FP8 case was not started; do not report an A/B performance result.
Evidence: `metrics/modular_run.json` and
`metrics/denoise_snapshots/before_denoise.json` in the failed control directory.

Read-only comparison against the accepted compiled ten-second BF16 case found
more differences than compilation: execution auxiliary ownership, auxiliary
CPU-source detachment, pre-transformer host-cache release and host-memory phase
inspection were also disabled. The proposed control reused the five-second
recipe, not all accepted ten-second lifecycle controls. These differences are
established; their causal role in the illegal access is not established.
Do not change DLL/budgets or rerun automatically. A revised comparison must
preserve the accepted BF16 lifecycle controls and document the different FP8
composite-ownership requirements; further CUDA execution needs new approval.

The separately approved corrected BF16 control subsequently passed, exit 0:
`E:/LTX-2/outputs/bf16_10s_8steps_fp8_comparison_02`. Same 10 s/241-frame,
eight-step audio-conditioned workload and ceilings; compilation remains off.
Execution auxiliary ownership, host-cache release, auxiliary source detachment
and phase inspection were restored to the accepted BF16 settings. DLL unchanged.
Load/prepare: 2.1817 s; attach: 8.0553 s; denoise wall time: 462.5770 s;
CUDA phase: 462.48834375 s; first CUDA step: 58.65807422 s;
mean CUDA steps 2-8: 57.68680134 s/it. Total without decode: 502.5720 s.
Sampled GPU peak: 6.4530 GiB baseline-relative / 7.4798 GiB absolute;
process RSS peak: 28.6438 GiB. Fault failures, temporary tensors, cache
admission denials and cache pressure entries/recoveries were zero. No completion
tickets remained. CPU-only verification found finite float32 packed video
`(1, 27280, 128)` and audio `(1, 251, 128)` latents. Source storage observation
after attach and after generation was zero CPU bytes, not a process-RAM claim.
Evidence: completed `metrics/modular_run.json` and eight step snapshots.
This accepts the corrected configuration, not a causal isolation of the prior
illegal access. FP8 has not yet run at 10 s; the A/B result remains pending.

### Completed ten-second BF16/FP8 comparison (2026-10-09)

After separate approval, `E:/LTX-2/outputs/fp8_10s_8steps_comparison_01`
completed with exit 0. Same generation controls as corrected BF16: 1280x704,
10 s/241 frames, 24 fps, eight steps, seed 43, real prompt/image and source
audio, 1024 FF/norm chunks, phase modulation, compact timesteps, in-place
residuals, diagnostic timing and no compilation. Native weight ceilings remain
1.5 GiB transformer / 4 GiB BF16 encoder; staging 1.5 GiB and source/cache
ceilings 24.625 GiB. Protected DLL unchanged. Both cases release host cache
and inspect memory phases. BF16 uses validated execution auxiliary ownership
and source detachment; FP8 instead uses composite component ownership and
bounded root state, without incompatible auxiliary detachment. This is a
single integration-level comparison, not a statistically repeated or isolated
weight-dtype/kernel experiment; cache warmth is uncontrolled.

| Measurement | BF16 corrected control | FP8 weight-only |
| --- | ---: | ---: |
| Transformer load/prepare (s) | 2.1817 | 31.6734 |
| Transformer attach (s) | 8.0553 | 83.5672 |
| First CUDA step (s) | 58.6581 | 67.0129 |
| Mean CUDA steps 2-8 (s/it) | 57.6868 | 67.9183 |
| Denoise wall time (s) | 462.5770 | 542.5450 |
| CUDA phase including dependencies (s) | 462.4883 | 542.4588 |
| Total without decode (s) | 502.5720 | 687.5026 |
| Sampled absolute GPU peak (GiB) | 7.4798 | 7.4578 |
| Global process RSS peak (GiB) | 28.6438 | 25.5203 |
| Denoise process RSS peak (GiB) | 28.6438 | 24.0728 |
| Final source-cache committed/registered bytes | 26,440,892,416 | 19,193,135,104 |
| Cumulative source-cache DMA bytes | 210,610,591,808 | 146,452,988,032 |

The FP8 candidate has approximately 17.3% longer denoising, 17.7% longer
steady-state step time and 36.8% longer total elapsed, despite approximately
30.5% fewer cumulative source-cache DMA bytes. Global RSS is lower, but sampled
GPU peaks are effectively alike. Source-cache counters are scoped native
snapshots/cumulative traffic, not total RAM or shared WDDM occupancy. CUDA-event
time and host progress intervals have different boundaries; use the CUDA-event
values for first/steady-state comparisons. There is no measured speed advantage
for this selected FP8 implementation on this workload; do not infer that all
FP8 backends or architectures are slower or that DMA is the proven bottleneck.

Both routes reported zero fault failures, temporary tensors, cache admission
denials and cache pressure episodes. FP8 completion collection retired 132
tickets to zero. CPU-only verification found finite float32 video
`(1, 27280, 128)` and audio `(1, 251, 128)` latents, without CUDA initialization.
Evidence: both case directories' `metrics/modular_run.json`, eight step snapshots
and saved latent artifacts. No ten-second decode or quality review ran.
Keep accepted BF16 as the performance reference; FP8 remains optional for
checkpoint-size/host-memory savings. Further optimization requires evidence
separating setup, dequantization/math and component scheduling costs, not an
automatic budget increase, compilation opt-in or another benchmark sweep.

### FP8 CPU ownership-index candidate (2026-10-09)

The selected FP8 discovery previously traversed the entire component for each
composite weight and each ordinary parameter/buffer. The candidate captures
owner aliases and physical-storage identities once in a discovery-local index,
then reuses those groups. It still refuses distinct owners sharing composite
payload storage and preserves repeated-module names. No forward, native binary,
residency ceiling, transport, or protected runtime behavior was changed.

CPU validation: **34 passed in 7.39 s** across test_quantized_owners_cpu.py,
test_quantized_arena_cpu.py, and test_quantized_sources_cpu.py, using the
host .venv-vmm with TorchAO 0.18.0, CUDA_VISIBLE_DEVICES=-1, and pytest
external plugin autoload disabled. A new assertion verifies one ownership-index
traversal; tests cover wrapper/parameter/buffer storage aliases. The first DDO
environment attempt skipped three modules because TorchAO was absent. The first
host attempt stopped all 34 tests at their CPU guard because imports had
initialized CUDA before fixture entry; it is not passing CPU evidence.

No candidate CUDA/model performance measurement has run. Reduced attach work
does not establish faster denoising; selected FP8/BF16 comparison results above
remain unchanged. Compilation stays excluded pending a separate compatibility
contract, not automatically enabled as a consequence of this CPU change.

The separately approved existing public CUDA gate then passed, exit 0, using
the prepared root-state fixture in
.internal/test-artifacts/fp8-owner-index-preflight-20261009 and reports in
.internal/test-artifacts/fp8-owner-index-cuda-20261009. Both the unchanged BF16
control and FP8 candidate reported exact repeated outputs, completed cleanup,
and recovery from deliberate RuntimeError and KeyboardInterrupt. FP8 physical
payload verification passed, including the bounded 24 KiB BF16 root state.
The protected DLL SHA256 remained
EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.
Limits: BF16 weight 6 MiB, FP8 weight 8 MiB (existing root-state fixture charge),
host cache 64 MiB, staging 4 MiB, allocator cache 4 MiB. This validates the
selected small integration contract, not LTX performance, quality, or general
model support. No video benchmark ran after the ownership-index change.

### FP8 compile capture CPU contract (2026-10-09)

Two CPU fullgraph/static Dynamo+AOT eager cases passed: compiled backend math
uses updated qdata/scale storage after physical .data replacement and updated
wrappers after one owner publication. Changed scale intentionally changes the
reference output, detecting stale-weight capture. Repeated compiled results
equal current backend eager output exactly; the original Linear forward class
remains unchanged, and final retirement invalidates both payloads to meta.
The corrected consolidated ownership/source/arena/compile suite reports
**36 passed in 8.54 s**, with CUDA hidden and no CUDA initialization. An initial
fixture attempted a forbidden second publication and was corrected without
relaxing the production ownership state machine.

This is not CUDA Inductor, fused-kernel, native-hook compilation, or performance
acceptance. The existing public small CUDA probe now has an explicit opt-in
compile-fp8-linear candidate: only FP8 Linear compute forwards are compiled
before native attach, fullgraph/static Inductor, CUDA graphs disabled, precision
cast emulation enabled. Block payload verification and interruption checks stay
outside compiled compute; the BF16 control and all budgets remain unchanged.
Native retirement must precede removal of compiled forwards. The host runner's
FP8 compilation exclusion remains in place until a separate gate is approved
and passes. No compiled CUDA candidate has run yet.

The separately approved small CUDA Inductor candidate subsequently ran with
the same prepared fixture and unchanged binary/ceilings, evidence in
.internal/test-artifacts/fp8-compile-cuda-20261009. BF16 eager control passed
exact repeated outputs, interruption recovery and cleanup. FP8 compiled eight
Linear forwards, but stopped at the first exact-output check: 1,165/2,048
elements differed (56.9%), maximum absolute difference 0.00048828125, maximum
relative difference 5.0. The relative maximum alone does not establish a large
signal error because near-zero values can magnify ratios; it does not waive
the agreed exact gate. FP8 cleanup completed without a cleanup error, and the
DLL SHA256 remained unchanged. No FP8 compiled interruption/reuse acceptance
or model benchmark followed the failed first numerical check.

Compilation success is not numerical acceptance. Preserve the host FP8
compilation exclusion and existing uncompiled route; do not automatically
relax tolerances. TorchAO's selected weight-only path dequantizes the weight,
performs matmul and then a separate bias addition. Compiled fusion/rounding
boundaries are a diagnostic hypothesis, not a proven cause of this mismatch.

Read-only inspection of the failed gate's generated Inductor code subsequently
confirmed separate FP8-to-FP32 scale multiplication and BF16 weight storage,
followed by extern_kernels.addmm for the compute+bias portion. In contrast,
TorchAO eager uses matmul followed by a separate bias addition. Thus the
compiled graph did change the eager BF16 rounding boundary. This identifies
a concrete graph difference; only a passing corrected gate can establish that
it explains the observed numerical mismatch.

The small opt-in candidate now disables Inductor pattern_matcher and
epilogue_fusion, keeping fullgraph/static compilation, precision-cast emulation
and CUDA graphs disabled. Options are included in the report. This conservative
diagnostic preserves the separate matmul/bias boundary rather than changing
TorchAO math or relaxing tolerances. It does not enable compilation in the host
runner and has not run on CUDA yet. BF16 control, binary and ceilings unchanged.

The separately approved corrected small CUDA gate passed with exit 0. Evidence:
.internal/test-artifacts/fp8-compile-rounding-cuda-20261009. BF16 stayed eager;
FP8 compiled eight Linear compute forwards with pattern_matcher=false,
epilogue_fusion=false, emulate_precision_casts=true and triton.cudagraphs=false.
Both cases passed exact repeated-output equality, deliberate RuntimeError and
KeyboardInterrupt recovery, and cleanup; FP8 physical-payload verification
passed. The DLL hash and fixture ceilings remain those recorded above.

Disabling these fusion settings resolves the observed mismatch in this selected
fixture without relaxing tolerances. Because two fusion settings changed
together, the gate does not isolate their individual effects. The previous
generated addmm remains the concrete diagnosed graph difference. Acceptance is
limited to this small CUDA/public native integration contract, not LTX regional
compilation, speed, media quality, or arbitrary model support. The production
runner still excludes FP8 compilation; no video benchmark followed this gate.

### Selected 10-second no-conditioning model comparison (2026-10-09)

The explicitly approved BF16 control completed: 1280x704, 24 fps, 241 frames,
seed 43, eight steps, no source-audio conditioning, latent output. Generated
audio remains enabled. Total excluding decode was 502.3321 s; denoise wall was
462.1879 s and CUDA steady steps 2-8 averaged 57.708845 s/it. Transformer load
was 2.3029 s and attach was 8.9207 s. Sampled denoise absolute GPU usage peaked
at 7.1186 GiB (baseline-relative delta 6.0918 GiB), RSS at 28.5847 GiB.
CPU inspection confirmed finite video latents (1, 27280, 128) and audio latents
(1, 251, 128). Evidence: sibling host outputs/bf16_10s_noaudio_comparison.

The next approved FP8 compiled candidate failed before completing its first
step with FailOnRecompileLimitHit, fullgraph=True. Total until failure was
79.4792 s, transformer load 32.6908 s and attach 13.3864 s; 1,660 Linear
wrappers were configured. These are setup/failure measurements, not FP8
denoise speed or quality acceptance. Evidence: sibling host
outputs/fp8_10s_noaudio_compiled_comparison. No automatic retry or compiler
limit increase followed. The successful small eight-Linear contract did not
cover real-model geometry diversity.

CPU diagnosis reproduced the limit on the ninth distinct Linear geometry,
whereas twelve equal geometries passed. Dynamo's shared external wrapper
reported a bias-size guard mismatch. Cloning the library forward code alone
still shared that external wrapper and failed; it is not the correction.
The experimental helper now compiles the standard functional Linear entry
with a separate code/cache per weight geometry, passes current weight/bias
views at invocation, and retains TorchAO dispatch and the accepted rounding
options. Compiler limits, native allocator, BF16 and uncompiled routes remain
unchanged. This correction requires separate CUDA acceptance; CPU evidence
alone is not a model performance result.

Corrected CPU ownership/source/arena/compile contracts: 39 passed in 8.65 s,
CUDA hidden. Twelve distinct geometries each produced a graph and exact current
backend output, including changed scale storage after capture. Global compiler
limits remained unchanged. The first consolidated attempt passed 16 tests but
23 fixture setups failed on access to the default pytest temporary directory;
the corrected run used a new project-local temporary directory and disabled
pytest cache writes. The CUDA probe now invokes the same compilation helper
as the host, rather than a separate bound-forward implementation. No corrected
CUDA run or model benchmark has executed yet.

The separately approved corrected-helper CUDA gate subsequently passed, exit 0.
Evidence: .internal/test-artifacts/fp8-geometry-cuda-20261009. BF16 remained
eager; FP8 used eight Linear dispatches sharing one weight-geometry cache.
Both reported exact repeated outputs, RuntimeError/KeyboardInterrupt recovery
and completed cleanup. FP8 physical payload verification passed, including
the bounded root-state fixture. The protected DLL SHA256 remained
EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.
Weight ceilings remained 6 MiB BF16 / 8 MiB FP8, host cache 64 MiB, staging
4 MiB and allocator cache 4 MiB. Compilation options stayed unchanged.

Scope: CUDA numerical/ownership acceptance for the selected small fixture,
not real-model speed or CUDA geometry-diversity acceptance. Twelve geometries
were exercised by the CPU contract, not by this one-geometry CUDA fixture.
No corrected full-model benchmark has run. Emitted TorchAO enum-registration,
serialization-import and stable-cache-hash warnings did not fail this gate;
this does not establish durable cross-process compiler-cache reuse.

### Corrected FP8 compilation: selected full-model execution (2026-10-09)

The separately approved ten-second/eight-step candidate completed, exit 0:
sibling host outputs/fp8_10s_noaudio_geometry_comparison. Workload remains
1280x704, 24 fps/241 frames, seed 43, no source-audio conditioning, generated
audio active, latent output without decode. The same protected DLL and native
ceilings were retained. The 1,660 standard FP8 Linears used 19 weight-geometry
caches; denoise_completed and callback_retired were both true. No compiler
limit increase or fallback to eager execution was applied by the helper.

| Selected no-source-audio case | CUDA steady steps 2-8, s/it | Denoise CUDA, s | Total without decode, s |
| --- | ---: | ---: | ---: |
| BF16 control | 57.708845 | 462.089156 | 502.3321 |
| FP8 geometry-isolated Linear compile | 63.442852 | 516.996156 | 596.6240 |

FP8 CUDA step times: 72.875188, 63.910988, 64.192656, 64.001320,
63.927133, 63.920270, 64.115055, 60.032543 s. The first interval includes
lazy compilation; it is not an isolated compiler-duration measurement.
FP8 is 9.94% slower in steady-state and 18.77% slower in one-shot total than
the selected BF16 control. BF16 retains auxiliary ownership/source detachment;
FP8 uses composite/root ownership. This compares supported routes, not dtype
alone with identical ownership behavior.

The older audio-conditioned uncompiled FP8 result was 67.918261 s/it and
687.5026 s total. The new steady-state is 6.59% shorter, but audio conditioning
and ownership-index setup differ between runs; do not attribute that entire
difference to compilation. A matched no-source-audio uncompiled FP8 benchmark
has not run. BF16 remains the selected performance reference; the candidate
does not meet a speed-improvement acceptance criterion against BF16.

Reported FP8 global sampled GPU peak is 6.293381 GiB baseline-relative, not
absolute driver usage; global RSS peak is 25.480087 GiB. Transformer source DMA
was 146,452,988,032 bytes. No cache-pressure entries, failed observations,
temporary fallbacks or driver allocation-operation failures were reported.
Source tickets collected from 132 to zero, native timeline events were released,
and compiled callbacks retired after native retirement. CPU inspection confirmed
finite float32 video latents (1, 27280, 128) and audio latents (1, 251, 128),
with CUDA uninitialized during inspection. These are selected execution/cleanup
checks, not decoded visual quality or exact full-model eager parity.

The raw report retains its static validation description stating CUDA acceptance
is pending; preserve this recorded field and use completed status, CUDA timing,
compile completion and the separately recorded gates as actual execution evidence.
No automatic additional benchmark, residency adjustment or default change follows.

### Independent local host transformer: selected BF16 migration gate (2026-10-09)

The separately approved sibling-host migration check completed with exit 0:
`outputs/local_transformer_bf16_regression/metrics/modular_run.json` in the
LTX-2 host. The report records implementation
`diffusers_modules.local.transformer_ltx2_video`, loaded through the trusted
custom-block component specification. DDO and the protected native DLL were
unchanged (SHA256 EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6).

Matched requested conditions: BF16, 1280x704, 10 s at 24 fps/241 frames,
seed 43, eight steps, real prompt/image, no source-audio conditioning but
generated audio active, latent output without decode. Transformer residency
remained 1.5 GiB, encoder residency 4 GiB, staging 1.5 GiB and native source
cache/registration 24.625 GiB. Phase modulation, compact timesteps, in-place
residuals, rotary reuse, 1024-token chunks and auxiliary ownership were retained.
No compilation, custom compute kernel or quantization was enabled.

| Selected BF16 implementation | CUDA steady steps 2-8, s/it | Denoise CUDA, s | Total without decode, s |
| --- | ---: | ---: | ---: |
| Previous fork-backed control | 57.708845 | 462.089156 | 502.3321 |
| Independent local transformer | 59.715444 | 477.993750 | 518.7514 |

Steady-state was 3.48% longer and total was 3.27% longer, within the explicitly
approved 5% investigation threshold. This single sequential comparison does
not establish the cause of the timing difference or an improvement; retain the
previous control as the performance reference. Sampled baseline-relative GPU
peak was 6.390427 GiB versus 6.091766 GiB (about 0.30 GiB higher); the new run's
denoise baseline-adjusted absolute estimate was 7.4173 GiB. Global sampled RSS
peak was 28.587563 GiB. These are sampled observations, not allocation censuses.

CPU inspection with CUDA uninitialized confirmed exact equality of both saved
float32 tensors against the preserved control: video (1, 27280, 128), audio
(1, 251, 128), maximum and mean absolute difference zero, all finite.
Transformer source tickets were zero at final collection; native timeline
pending/failures/dropped were zero and event_cleanup was released. No temporary
fallback, fault/admission/source-read failure or source-pressure episode was
reported. Allocation diagnostics completed without failure. The selected BF16
migration gate passed; it does not validate FP8, optional kernel/compile variants,
decoded visual quality, the image route, or arbitrary models. No additional CUDA
run or budget adjustment is automatically authorized by this result.

### Local FP8 chunk-reuse candidate: CPU only (2026-10-09)

The sibling host implements opt-in per-FeedForward converted-weight reuse,
without changing the native allocator, accepted BF16 route or existing FP8
default. Retained BF16 temporaries are explicitly capped and event-owned; no
full-model BF16 cache is added. Native projection boundaries remain installed.
The selected CPU fixture demonstrated two conversions instead of 54 over 27
token chunks, with exact backend outputs and no retained converted outputs.
Consolidated final CPU acceptance: 92 tests and 11 subtests, 12.89 s. The local
saved-checkpoint preflight also passed exact outputs, interruption recovery
and cleanup with 128 KiB retained-weight peak and CUDA uninitialized
(.internal/test-artifacts/fp8-chunk-preflight-20261009/report.json).

This is not CUDA ownership, capacity or performance acceptance. Compiled compute
was stubbed in the CPU configuration contract; event lifecycle contracts used
test events. The preflight native budgets are requested controls, not executed
native observations. A small compiled/native CUDA fixture is prepared in the
host but has not been executed or approved. No generation benchmark, budget
increase or speed-improvement claim follows from these CPU results.

### FP8 chunk-reuse CUDA gate: setup failure (2026-10-09)

The subsequently approved `outputs/fp8_chunk_reuse_cuda_gate_01` attempt stopped
before candidate model execution: checkpoint conversion ran after native allocator
bootstrap and correctly refused with `Run conversion in a fresh CPU-only process`.
The partial checkpoint directory is preserved. No report was generated because
preparation preceded the harness exception handler. This is a harness setup
failure, not CUDA acceptance or evidence of a compute regression.

The corrected harness requires a saved fixture from a passed CPU-only preflight
for CUDA execution, records its file hashes, and includes preparation failures
in report handling. The converter safety guard and native runtime are unchanged.
A corrected CUDA attempt requires renewed approval; no video benchmark follows
automatically.

Corrected fixture-reuse CPU validation passed with CUDA uninitialized, exact
outputs, interruption recovery, complete cleanup and a 128 KiB retained-weight
peak. File hashes are recorded in
`.internal/test-artifacts/fp8-chunk-fixture-reuse-cpu-20261009/report.json`.
This does not validate compiled compute or native CUDA ownership.

### Local FP8 chunk reuse: selected small native gate passed (2026-10-09)

The separately approved corrected host gate completed with exit 0. Evidence:
`E:/LTX-2/outputs/fp8_chunk_reuse_cuda_gate_02/report.json`. It reused the passed
CPU fixture without conversion after native bootstrap. The protected DLL hash
remained EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.

Eight compiled graphs were observed. Repeated outputs matched the backend
reference exactly; deliberate RuntimeError and KeyboardInterrupt recovery passed.
Across normal, interrupted and recovery calls: 22 FF scopes, 44 conversions and
120 reuse hits. Retained-weight peak was 131072 bytes (the explicit 128 KiB cap),
with zero pending/quarantined bytes after removal. Two event-specific waits took
0.0001157 host seconds; this is not GPU timing. Cleanup completed and parameters
returned to meta. The gate asserted zero native fault failures, temporary
fallbacks and cache admission denials, with residency within the 6 MiB ceiling.
Requested staging/source/cache limits remained 4 MiB/64 MiB/4 MiB respectively.

This accepts only the small compiled/native ownership contract, not LTX quality,
real-model memory headroom or performance. No image/linear/NVMe route, allocator
implementation, default or full-model BF16 cache was changed. A real-model
comparison remains separately approval-gated; no speedup is claimed.

### Local FP8 FF chunk reuse: selected 10-second comparison (2026-10-09)

The approved sibling-host comparison completed with exit 0. Evidence:
`E:/LTX-2/outputs/fp8_10s_noaudio_chunk_reuse/metrics/modular_run.json`.
Conditions matched the saved compiled FP8 control: 1280x704, 10 s at 24 fps
(241 frames), seed 43, eight steps, real prompt/image, no source-audio conditioning,
generated audio active and latent-only output. Native transformer/encoder
residency remained 1.5/4 GiB, staging 1.5 GiB, source cache/registration
24.625 GiB and the protected DLL was unchanged. FF/normalization chunks were
1024 tokens; phase modulation, compact timesteps, in-place residuals and rotary
reuse were retained. The candidate additionally used explicit 256 MiB scoped
converted-weight retention and the independent local transformer. The older
control used the fork-backed transformer, so this is not an isolated same-build
toggle comparison, although saved outputs are exact.

| Selected FP8 route | CUDA steady steps 2-8, s/it | Denoise CUDA, s | Total without decode, s | Sampled baseline-relative GPU peak, GiB |
| --- | ---: | ---: | ---: | ---: |
| Preserved compiled Linear control | 63.442852 | 516.996156 | 596.6240 | 6.293381 |
| Local FF converted-weight reuse | 62.768249 | 510.155531 | 588.9593 | 5.920681 |

Observed reductions were 1.06% steady and 1.28% total (7.66 s total), with
0.372700 GiB lower sampled GPU delta. This single comparison does not establish
a reproducible speedup. Candidate sampled baseline was 1.026855 GiB; global RSS
peak was 25.477177 GiB. GPU observations are sampled baseline-relative deltas,
not allocation censuses. The BF16 references remain faster in steady state.

CPU-only inspection confirmed exact equality against the preserved FP8 output:
float32 video latents (1, 27280, 128), audio latents (1, 251, 128), maximum/mean
absolute difference zero, finite tensors and matching saved metadata. CUDA stayed
uninitialized during this inspection; decoded visual quality was not tested.

Actual reuse metrics: 192 configured projections, 768 FF scopes, 1536 conversions,
19968 reuse hits, peak retained weights exactly 268435456 bytes. Pending and
quarantined bytes were zero after removal; denoise completed and callbacks retired.
There were 384 event-specific waits totaling 467.962575 host seconds. These waits
overlap execution of queued GPU work and are already included in wall time; do
not sum them with denoise time or interpret them as removable extra overhead.
The single-FF-sized cap imposes frequent completion waits rather than retaining
a full-model BF16 cache. No default, allocator or protected runtime was changed.
Further capacity changes or benchmarks require separate approval.

Native fault failures, temporary fallbacks, cache admission denials and pressure
entries were zero; final source tickets were zero. Native timeline completed
with zero pending/failures/dropped entries and allocation diagnostics completed
without failure. These are selected-run observations, not arbitrary-model support.

### Scoped FP8 operator diagnostic: compute attribution (2026-10-09)

The separately approved diagnostic completed with exit 0, keeping the same
10-second workload, 256 MiB retained-weight ceiling and native budgets, but using
three denoise steps and two profiler captures of `transformer_blocks.0`.
Evidence: `E:/LTX-2/outputs/fp8_chunk_reuse_operator_diagnostic/metrics/modular_run.json`
and `operator_traces/operators_invocation_02.json` in that output directory.
Operator diagnostics report complete, two samples and benchmark_eligible=false.
The first capture includes compilation; use the third-invocation capture for
the following attribution, not the instrumented pipeline timing as a benchmark.

The third capture contained 1222 CUDA activity events, including 1179 kernels.
Sum of kernel durations was 1.286211 s; this is a sampled kernel-duration sum,
not an exclusive pipeline-wall-time decomposition or a driver-paging census.
Attention operator device duration was 0.641213 s (six calls), with the two
largest attention kernels totaling 0.636344 s. Matrix multiplication operator
device duration was 0.484956 s (86 calls). These CPU-operator device aggregates
refer to underlying kernels: never add operator and kernel values together.

GPU child annotations placed video self-attention at 0.789409 s and video FF
at 0.254316 s, both inclusive of their projections/other child kernels.
The identified fused conversion kernel family (`triton_poi_fused__to_copy_mul_t_0`)
totaled 0.002862 s over 34 calls; this name-based identification is not a census
of every possible quantization-related operation. The two compiled add families
totaled 0.016762 s over 86 calls. Aggregate attention and matmul costs dominate
the sampled block, explaining why FF conversion reuse gave only a small gain.
This is not evidence that an attention-backend switch would help or that the
same costs explain a difference against an unmatched external implementation.

The audio FF CPU range included a 1.223307 s cudaEventSynchronize call; its GPU
annotation was only 0.001324 s. Waiting in that range retires queued earlier GPU
work under the retained-weight cap, not 1.22 s of audio FF computation or extra
time to add to the kernels. Reported reuse totals were 288 scopes, 576 conversions,
7488 hits, 144 event waits and zero pending/quarantined bytes after retirement.
Native faults, temporary fallbacks, source-cache admission denials, pressure
entries and final source tickets were zero. Timeline completed without pending,
failed or dropped entries; allocation diagnostics completed without failure.
No additional benchmark, budget/backend change or default promotion follows.

### Five-second FP8 group-offload control: unsupported pinning (2026-10-09)

The separately approved A/B sequence stopped in its first case, before the first
denoise step completed. Evidence:
`E:/LTX-2/outputs/fp8_5s_group_control/metrics/modular_run.json` (status failed,
exit 1). The standard Diffusers loader successfully loaded the ten-shard FP8
checkpoint and the explicit recipe guard passed. Encoder/connectors/VAEs completed
on official group routes, then transformer leaf-level group offloading installed.
No native allocator, FP8 compute compilation or scoped weight reuse was requested.

The first transformer projection triggered group offload's
`_pinned_memory_tensors`: it calls `tensor.pin_memory()` for its streamed CPU
parameter dictionary. TorchAO 0.18 Float8Tensor dispatch refused
`aten.pin_memory.default` with NotImplementedError. This is an incompatibility
of the selected streamed group/TorchAO route, not OOM, failed FP8 reconstruction,
model-forward failure or native DDO evidence. Source inspection confirms the
selected group implementation avoids that pinning branch when use_stream=false;
that alternative has not been executed or accepted for this workload.

Partial total was 45.0354 s, sampled global GPU delta 2.251953 GiB and RSS
25.983688 GiB; these are failed-run/setup observations, not denoise benchmarks or
capacity limits. There are no matched throughput numbers or saved video latents.
The native second case was not started, honoring stop-on-first-failure approval.
No Diffusers/TorchAO patch, stream-setting change or retry was performed. The
working native FP8 route remains unchanged; any amended comparison requires
renewed approval and explicit labeling of the offload stream difference.

### FP8 five-second streamless group control — failed (2026-10-09)

The separately approved amendment kept the same checkpoint, 1280x704, 5 s at
24 fps, eight steps, seed 43 and local forward options. Only transformer official
leaf-group `use_stream` changed to false; auxiliary events confirm true. Evidence:
`E:/LTX-2/outputs/fp8_5s_group_no_stream_control/metrics/modular_run.json`.

The first projection failed before completing step one: standard Float8Tensor
CPU-to-CUDA `.to()` reached TorchAO alias correction and PyTorch rejected assigning
CPU storage to a CUDA tensor. This bypassed pinning but did not establish a usable
control. No OOM was reported. Partial total 31.8042 s, GPU delta peak 2.251953 GiB,
RSS peak 25.983658 GiB are setup/failure observations, not throughput results.
No saved latents/parity or matched speed numbers are available. The native second
case was not run, respecting stop-on-first-failure approval. Existing native,
BF16/image and allocator defaults were not changed; no backend dispatch or
dependency patch was introduced. Both failures apply to the selected installed
stack, not all official group/TorchAO versions or platforms.

### Isolated FP8 CUDA transfer diagnostic — passed (2026-10-09)

`E:/LTX-2/outputs/fp8_transfer_probe_cuda_01.json`: ordinary CUDA, Torch
2.12.1+cu132 / TorchAO 0.18.0, one freshly quantized 16x16 Linear, no LTX or DDO.
Dense control and four parameter/data default/copy=True transfers passed; FP8
bytes/scales were exact, outputs finite, synchronization successful. Peak PyTorch
allocation 8,524,800 bytes is a tiny diagnostic, not generation capacity evidence.
The runner failure was not reproduced: fresh quantization and transfer invocation
do not cover the saved-checkpoint/official-hook lifecycle. Do not claim copy=True
fixes the failed group control or that generic CUDA transfer is broken. No
benchmark retry, dispatch patch or accepted-runtime change followed this result.

### Saved FP8 official-hook diagnostic — passed (2026-10-09)

`E:/LTX-2/outputs/fp8_saved_group_cuda_01/report.json`: one saved/reloaded 16x16
Diffusers Linear, standard loader and official streamless leaf hooks. Direct
group-style transfer and two forwards passed with exact CUDA-reference outputs
and restored CPU FP8 bytes/scales. Peak PyTorch allocation was 8,525,312 bytes.
No DDO runtime, dependency patch or full-model retry. The LTX failure remains
unreproduced; host `torch.inference_mode()` was not covered by this initial probe.
This is a diagnostic limitation, not proof the failed comparison now works.

### FP8 hook inference-context reproduction — failed (2026-10-09)

`E:/LTX-2/outputs/fp8_saved_group_inference_cuda_01/report.json` reuses exactly
the passing saved-checkpoint fixture hashes. Direct transfer outside
inference_mode passed; the first official streamless hook forward inside
torch.inference_mode reproduced the full runner's CPU/CUDA storage-alias error
in TorchAO `aten.to.dtype_layout` / PyTorch `_functionalize_unsafe_set`.
The selected hook contract therefore has an isolated execution-context trigger,
not an established native allocator, LTX-forward or capacity defect. The streamed
pin_memory failure remains separate. No runtime changes or full-model retry
followed this probe. A group-control-only no_grad context is a proposed workaround,
not yet a validated generation baseline or fix for streamed offload.

### FP8 five-second official group/native A/B — completed (2026-10-09)

Matched saved FP8 checkpoint/local transformer: 1280x704, 5 s at 24 fps,
121 frames, eight steps, seed 43, real prompt/image, no source audio but generated
audio active, latent-only output. Same 1024-token FF/normalization chunks,
phase/compact modulation, residual/RoPE reuse; no FP8 compilation or chunk weight
reuse. Evidence: `E:/LTX-2/outputs/fp8_5s_group_no_grad_control/metrics/modular_run.json`
and `E:/LTX-2/outputs/fp8_5s_native_control/metrics/modular_run.json`.

| Measurement | Official leaf group, no stream/no_grad | DDO native, inference_mode |
| --- | ---: | ---: |
| Transformer load/prepare + attach (s) | 15.7939 + 0.2060 | 34.3141 + 12.3416 |
| Denoise phase (s) | 411.2244 | 235.2354 |
| Total tracked execution, latent output (s) | 443.4385 | 316.2860 |
| Sampled global GPU delta peak (GiB) | 2.283203 | 5.523609 |
| Sampled absolute driver GPU peak (GiB) | 3.3101 | 6.5505 |
| Sampled global process RSS peak (GiB) | 27.634579 | 25.981510 |

Native denoise time fell 42.7963% (1.7481x speedup); total fell 28.6742%, saving
127.1525 s despite greater setup cost. Single selected A/B, not a claim against
working streamed FP8 group offload or repeated-run statistics. The control uses
no_grad to avoid the reproduced alias defect; native retains inference_mode.
Auxiliary components use official streamed hooks in both, with native global
allocator installation an intended difference. No per-step CUDA timer: phase/8
is not steady-state. Total excludes VAE decode/MP4 encoding.

Saved Float32 video `[1,14080,128]` and audio `[1,126,128]` latents and metadata
were exactly equal in CPU comparison, finite, max/mean absolute error and relative
RMS zero. Native fault failures, temporary fallbacks, source-cache admission
denials/pressure entries, source-read failures and quarantine bytes were zero;
132 source tickets collected to zero. GPU deltas use the 1.026855 GiB baseline,
not a weight-residency measurement. Native kept the selected DLL hash unchanged,
1.5 GiB weight/staging and 24.625 GiB source/registration ceilings. This is selected
FP8 compatibility/performance evidence, not generic recipe support or BF16 quality
equivalence. No further benchmark or default change was made.

### SDNQ INT4 video control and preparation correction (2026-10-10)

The official Diffusers streamed leaf control completed a 5-second, 24-fps,
1280x704, eight-step generation (seed 43; real prompt/image, no source audio,
generated audio enabled). This is normal PyTorch allocation and official leaf
offload, not DDO SDNQ parameter swapping or native VMM. The modular host used
explicit GPU temporal video decoding followed by separate audio VAE/vocoder
stages. No block tracing was enabled.

| Measurement | Completed leaf control |
| --- | ---: |
| Denoise (s) | 649.6985 |
| Decode and save (s) | 27.9491 |
| Total tracked execution (s) | 713.2001 |
| Denoise sampled GPU delta peak (GiB) | 6.9688 |
| Denoise sampled absolute driver GPU peak (GiB) | 7.9995 |
| Decode sampled GPU delta peak (GiB) | 3.6875 |
| Decode sampled absolute driver GPU peak (GiB) | 4.7183 |
| Global sampled process RSS peak (GiB) | 26.0283 |

Current-stream CUDA step times were 64.1410, 81.7814, 135.9691, 77.8015,
97.6200, 76.5921, 65.8113 and 49.7024 seconds. The runner verified 121 frames
at 1280x704 and finite, nonempty generated audio; exit code was zero and all
phase memory-monitor errors were null. No visual quality assessment was made.
Evidence: sibling host `ddo-examples/outputs/sdnq_int4_leaf_streaming/metrics/
sdnq_int4_leaf_run.json` and `outputs/ddo_modular_sdnq_video/
int4-leaf-streaming-console.log`. GPU deltas and absolute driver usage are
different measurements, not weight-residency estimates.

GPU streaming decode passed functionally. Denoise performance is **not accepted**:
the large variability also occurs with official leaf offload. The earlier traced
run completed denoise in approximately 262.5 seconds but was interrupted during
CPU decode; it produced no complete video. Tracing changes execution timing, so
that result is diagnostic evidence, not a matched performance baseline.

Separately, DDO's SDNQ preparation skipped parameters already on the offload
device, bypassing pinning and pinned/unpinned accounting. Running the new CPU
contract against the HEAD implementation reproduced this defect (expected test
failure: zero pin calls instead of four). The correction accounts for CPU packed
parameters and scales, deduplicates shared Parameter objects, and bounds new
pinning by the resolved byte budget, ratio, maximum and known RAM headroom.
Existing loader-owned pins remain intact and any excess is reported. A permitted
registration failure disables subsequent pin attempts without skipping remaining
parameter migration/accounting; strict failures still propagate.

The existing residency planner and its dense candidate estimate were not changed.
`sdnq_candidate_parameter_bytes` reports the separate packed-parameter population;
the resolved pin ceiling may therefore remain conservative until measured profile
validation. Backend math/forward, dense linear and native VMM paths are unchanged.
This is not proof of improved GPU performance or generic tensor-subclass storage
accounting. CPU contracts simulate pinning and forbid CUDA initialization: 77 tests
and two subtests passed across preparation, presets, named profiles, profiles,
backend routing and cleanup. Video validation and image regression require an
explicitly approved CUDA sequence; neither was run after this correction.

### SDNQ INT4 corrected preparation GPU attempt — interrupted (2026-10-10)

Approved sequence: the same 5-second, 24-fps, 1280x704, eight-step INT4 video,
seed 43, real default prompt/image, no source audio, GPU streaming decode;
DDO `auto` / `sdnq_runtime` instead of the official leaf control. An INT4 image
generation was conditional on video acceptance and was **not started**.

Video setup retained the automatic 5.00 GiB resident budget and 0.36 GiB pin
budget. Text encoder took 17.1 s, connectors 7.7 s, video encoder 4.1 s,
audio encoder 0.2 s, transformer loading 5.5 s and attachment 6.8137 s.
Attachment sampled GPU delta peak was 6.30 GiB and RSS peak 11.60 GiB.
During denoise, sampled dedicated driver usage remained approximately
7,944–7,957 MiB with high GPU utilization. Standby purge was skipped because
the process token lacked SeProfileSingleProcessPrivilege; no privilege or
purge policy was changed during this attempt.

At the user's request the identified runner process was terminated, retaining
`ddo-examples/outputs/sdnq_int4_ddo_corrected/console.log`. Before termination,
one host callback was recorded at **525.2814 s**. This is a host callback interval,
not a resolved CUDA step time or complete-denoise benchmark. There is no final
metrics JSON, completed video, measured actual pinned-byte total, or image
regression acceptance from this attempt. The console reported no CUDA exception;
termination was a user-approved diagnostic interruption, not an OOM result.

Read-only investigation of the cached transformer safetensors headers found
7,105 tensors: 9.140258789 GiB U8 plus 1.603125095 GiB BF16, totaling
10.743383884 GiB physical file payload. This includes scales and auxiliary
tensors, not only offloaded eligible weights. For example, one attention weight
is U8 `[2097152,5]`, a packed layout rather than its original dense Linear shape.
The configuration specifies dynamic mixed quantization, `is_training=false`,
`use_quantized_matmul=false` and `dequantize_fp32=false`. Its default uint4 label
is not a claim that every tensor uses four bits.

The shared automatic budget still calls the dense Linear candidate collector,
which rejects packed shapes. A guarded CPU reproduction with one packed Linear
and one dense Linear reported 16 dense candidate bytes, 50 SDNQ candidate bytes,
and a resolved pin ceiling of only 16 bytes. This confirms an incomplete pin
budget population, separate from the corrected already-CPU preparation defect.
It does **not** establish that this alone caused the slow step: workload-unprofiled
GPU residency and missing overlap remain separate questions. No further budget
change, backend-forward change, GPU run or code correction followed this
investigation. Next work should separate SDNQ pin-capacity accounting from dense
planning, retain RAM/byte limits, and validate video workload capacity before
accepting residence or performance settings.

### SDNQ automatic pin-capacity correction — CPU contracts (2026-10-10)

The SDNQ automatic budget now collects the same identity-deduplicated direct
packed parameters, scales and auxiliaries as SDNQ preparation. It no longer uses
dense Linear shape eligibility or includes dense fallback weights that this
runtime keeps on the execution device. Other runtimes retain their existing
dense collector. Shared residency heuristics and backend forward math were not
rewritten; correcting their capacity input can change automatic decisions, so
GPU performance and image regression acceptance remain pending.

Before the correction, CPU tests reproduced a 16-byte automatic pin budget for
32 bytes of SDNQ candidates and incorrect RAM admission with only 24 bytes
available. Both tests now pass, along with explicit byte/maximum/ratio limits,
shared-Parameter deduplication, failure policy, existing pins and unchanged dense
planning. Calibration still forces zero residents with the corrected pin
capacity. The public report renders actual setup pins/unpinned bytes and the pin
ceiling; historical reports without that ceiling do not invent zero values.
Units and limitations are documented in `docs/metrics.md` and profile guidance
in `docs/capacity-profiles.md`.

CPU-only validation: 85 tests and two subtests passed across preparation,
presets, named profiles, profiles, backend routing and cleanup. Pin registration
was simulated and CUDA initialization forbidden by the preparation contracts.
No further GPU run or residency recommendation has been accepted after the
interrupted attempt. Proposed next gate: the same five-second video workload
with two steps and a new zero-resident named calibration, followed by image
functional/regression observation only after successful video admission.

### SDNQ corrected-capacity video calibration and image observation (2026-10-10)

The approved two-case sequence completed with exit code zero in both processes.
No allocator binary, backend forward, extra residency probe or benchmark beyond
these cases was changed/run. The video gate uses the new named profile
`sdnq-int4-video-5s-capacity-v2`, with calibration forcing zero residents.
Conditions: INT4 mixed SDNQ transformer, BF16 auxiliaries, 1280x704, 5 seconds at
24 fps, **two steps**, seed 43, real default prompt/image, no source audio,
generated audio enabled, streaming GPU decode. The image uses its own INT4
checkpoint, default real text prompt and BF16 text encoder, 1280x704, **eight
steps**, seed 43, `auto`, no profile/PAG/LoRA. These are different workloads and
checkpoints, not an image/video speed comparison.

| Measurement | Video calibration, 2 steps | Image observation, 8 steps |
| --- | ---: | ---: |
| Transformer setup/attachment (s) | 8.0303 | 18.3296 |
| Denoise phase/call (s) | 118.5485 | 16.4873 |
| Full tracked total (s) | 191.2700 | 106.2028 |
| Selected extra module residency (GiB) | 0.0000 | 1.7650 |
| Actual CPU SDNQ pinned payload (GiB) | 8.5473 | 5.2907 |
| CPU SDNQ unpinned payload (GiB) | 1.5083 | 0.0000 |
| Sampled global GPU delta peak (GiB) | 6.968750 | 6.1627 |
| Sampled absolute driver GPU peak (GiB) | 7.999512 | 7.193504 |
| Sampled global process memory peak (GiB) | 27.852203 RSS | 10.0194 USS |

Video current-stream CUDA step times: **34.6961, 83.4489 seconds**. Corresponding
host callback intervals: 34.6940 and 83.4445 seconds. The first callback satisfied
the approved 180-second stop threshold. Exact offload candidate capacity was
10,797,139,968 bytes, pin ceiling 9,177,568,972 bytes; the actual pin count was
5,326, with 869 unpinned parameters. The rounded pin GiB value must not be used
to infer exact byte equality, but selection and the reported payload remained
within the ceiling. All phase memory-monitor errors were null. Video decode/save
took 28.0511 seconds; planned and completed stages were video, audio_mel and audio.
Runner validation checked 121 frames at the requested resolution and finite,
nonempty generated audio before successful completion.

Calibration saved `sdnq_baseline_safe`, recommending **0.0 GiB** extra residency.
Its cache-inclusive conservative peak estimate was 9.593261719 GiB against
7.999511719 GiB reported device capacity, with a 1.0 GiB margin. This estimate
combines allocator reservation and initial non-PyTorch usage; it is **not** a
claim that 9.59 GiB of physical dedicated VRAM was occupied. No pending resident
probe was recommended or executed. The second step remained much slower than
the first; steady-state performance for eight-step video remains unaccepted.
The prior 525-second attempt also had different residency, so its timing cannot
be attributed solely to the pin-capacity correction.

Image step observations were 4.8344, 1.5548, 1.5126, 1.5643, 1.5669, 1.5429,
1.5366 and 1.5192 seconds (existing runner timing, not the video CUDA-event
protocol). The saved PNG header confirms 1280x704; output size is 1,589,120 bytes.
All 1,548 eligible offloaded parameters were pinned. The setup's initial automatic
pin budget was 7.0543 GiB; after resident selection the resolved candidate/ceiling
was 5,680,894,464 bytes. No functional regression was observed for this selected
image case; no matched pre-change image A/B or visual/numerical equivalence test
was performed. Image text encoding took 55.7981 seconds, including a harmless
official group-offload warning about the unused vision branch; USS/RSS and
different text workloads must not be compared as equivalent memory/time evidence.

Evidence in sibling `ddo-examples`:

- `outputs/sdnq_int4_capacity_v2/console.log`
- `outputs/sdnq_int4_capacity_v2/metrics/sdnq_int4_ddo_run.json`
- `outputs/sdnq_int4_image_corrected/console.log`
- `outputs/sdnq_int4_image_corrected/metrics/ltx23_image_distilled_modular_sdnq_ddo_runtime_sdnq4_text_encoder_original_nopag_nolora_1280x704_steps8_seed43.json`

The named video profile is stored locally under `.ddo/profiles`, key
`ddo-capacity-v1-b09358a953df579846d25469`. Profile names are workload-specific;
do not apply this five-second recommendation to longer video or image models.
Next performance evidence would require an explicitly approved eight-step video
using this zero-resident profile, not an automatic budget increase.

### SDNQ INT4: approved eight-step saved-profile video (2026-10-10)

The approved primary-host run completed with exit 0 using the same five-second,
24-fps, 1280x704, seed-43 workload, real prompt/image, no source audio and
generated audio enabled. It reused `sdnq-int4-video-5s-capacity-v2`, without
recalibration or extra resident-module budget. This is `sdnq_runtime` with the
normal PyTorch allocator, not native VMM. GPU temporal streaming decode remained
enabled. No CUDA error or 180-second callback stop threshold was encountered.

| Recorded route | Denoise phase, s | Full runner total, s | Decode/save, s |
| --- | ---: | ---: | ---: |
| Official streamed leaf INT4 control | 649.6985 | 713.2001 | 27.9491 |
| Corrected DDO SDNQ INT4, saved zero-resident profile | 319.6422 | 389.3677 | 28.0043 |
| Historical BF16 native VMM full-video gate (2026-10-06) | 197.0318 | 266.5138 | 30.1365 |

The selected DDO SDNQ run reduced denoise time by about 50.8% relative to the
recorded leaf control. Historical BF16 VMM still took about 38.4% less denoise
time than this SDNQ run. BF16 and INT4 use different checkpoints, host revisions
and runtime implementations; their row comparison is an observed workload-level
reference, not an isolated allocator, quantization or repeatability claim.

CUDA event intervals were 24.6510, 54.3126, 38.3030, 40.2564, 40.2516, 39.7300,
42.7081 and 39.0847 seconds. The phase average was about 39.96 s/step; steps
2-8 retained variation, so do not infer a universal steady-state speed.
Transformer attachment took 7.4599 seconds (setup event 7.4494 seconds).
Reported CPU parameter pinning remained 8.5473 GiB pinned and 1.5083 GiB
unpinned, within the 8.5473 GiB resolved ceiling. Peak sampled process RSS was
27.3831 GiB; GPU peak was 6.96875 GiB baseline-relative and 7.999511719 GiB
absolute driver usage. All phase monitor error fields were null.

Decode completed video, audio_mel and audio on `cuda:0`. The runner completed
its frame/audio checks and saved an 838,042-byte MP4. This is functional/media
structure acceptance, not a visual-quality or BF16 numerical-parity claim.
Evidence in sibling `ddo-examples`:

- `outputs/sdnq_int4_ddo_8steps_profiled/console.log`
- `outputs/sdnq_int4_ddo_8steps_profiled/metrics/sdnq_int4_ddo_run.json`
- `outputs/sdnq_int4_ddo_8steps_profiled/ltx23_sdnq_int4_ddo_1280x704_5.0s_seed43.mp4`

The approved sequence is complete. No additional benchmark, budget change or
native/image regression run follows automatically.


### Primary-host BF16 native VMM full-video migration (2026-10-10)

The separately approved gate completed in `ddo-examples`' root uv environment,
exit 0. Workload: BF16, 1280x704, 5 seconds at 24 fps/121 frames, eight steps,
seed 43, real default prompt/image, no source audio and generated audio enabled.
The trusted local custom-block transformer was
`diffusers_modules.local.transformer_ltx2_video`. Native encoder ceiling 4 GiB,
transformer ceiling 1.5 GiB, total staging 1.5 GiB and source/registration ceilings
24.625 GiB were explicit. FF/RMSNorm chunks 1024, phase modulation, compact
timesteps, in-place residuals and rotary reuse were enabled; no compile or
experimental compute kernels. CPU host-cache release ran before the transformer.
GPU temporal streaming decode was enabled.

The selected DLL remained unchanged, SHA256
`EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
Command: from the host root, after unsetting child `SSLKEYLOGFILE` and setting
`HF_HOME=F:/hf_cache`:

```powershell
uv run --no-sync python -u scripts/ltx2_video/run_dynamic_modular_vmm_distilled.py --ddo-backend vmm --ddo-preset auto --ddo-vmm-library-path E:/ProjetosIA/diffusers-dynamic-offloader/ddo-vmm-allocator/build-joined-spans-cu128-20261006/Release/ddo_vmm.dll --custom-blocks modular_diffusers/custom_blocks/ltx2_video --local-files-only --no-use-audio-condition --width 1280 --height 704 --seconds 5 --frame-rate 24 --steps 8 --seed 43 --text-encoder-native-vmm --text-encoder-vmm-resident-weight-budget-gb 4 --ddo-vmm-resident-weight-budget-gb 1.5 --ddo-vmm-staging-buffer-gb 1.5 --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625 --feed-forward-chunk-size 1024 --normalization-chunk-size 1024 --phase-modulation --compact-timesteps --inplace-residuals --release-pytorch-host-cache-before-transformer --stream-video-decode --decode-device cuda:0 --show-profile --show-metrics --save-metrics --output-dir outputs/migration_video_bf16_vmm
```

| Measurement | Current primary host | Historical 2026-10-06 full-video gate |
| --- | ---: | ---: |
| Denoise phase, s | 203.7925 | 197.0318 |
| Decode/save, s | 27.6654 | 30.1365 |
| Full tracked total, s | 266.9297 | 266.5138 |

Denoise was approximately 3.43% longer, while total was approximately 0.16%
longer. The historical case used a different DLL/host/forward revision and
2 GiB transformer ceiling; this is a workload-level reference, not an isolated
migration regression, matched speedup or repeatability claim. Current phase/8
is 25.4741 s; progress intervals are not separately resolved CUDA step timings.

Encoder: 13.2534 s; transformer load/prepare: 1.8849 s; attach: 6.7815 s.
Global sampled GPU delta: 4.115234375 GiB; absolute driver peak: 5.142089844 GiB.
Denoise absolute driver peak: 4.571777344 GiB. Peak RSS: 28.415805817 GiB.
These are driver/process samples, not physical weight residency or WDDM shared
accounting. All phase monitor error fields were null.

Encoder and transformer snapshots reported zero fault/guard/CUDA/capacity/read/
source-admission/pool failures, plan mismatches, temporary fallback tensors and
quarantined bytes. Source live tickets and transfer-owned events were zero.
The encoder collected eight tickets to zero; transformer collection was zero
to zero. Cleanup and process exit succeeded, not a post-exit allocation census.

Runner media checks found 121 frames at 1280x704 and finite stereo generated
audio, shape [2,240480], 48000 Hz. MP4 SHA256:
`56BD4CF1D6F20CD2C978FDA6280902197A24FC179E303AC98F65EE02EB7FD891`.
This accepts selected full-video migration functionality, not visual quality,
exact historical output parity, arbitrary models or Linux execution.

Evidence: `outputs/migration_video_bf16_vmm/console.log`,
`outputs/migration_video_bf16_vmm/metrics/modular_run.json` and
`outputs/migration_video_bf16_vmm/ltx23_modular_1280x704_5.0s_seed43.mp4`.
Saved FP8 execution in the primary host remains a separate approval gate.

### Same video runner with historical interpreter (2026-10-10)

The separately approved control ran the same primary-host VMM runner, local
custom-block clone, DLL and complete arguments as the preceding gate. Only
the interpreter/environment and output directory changed: direct
`E:/LTX-2/.venv-vmm/Scripts/python.exe -u` instead of `uv run --no-sync python -u`,
with `--output-dir outputs/migration_video_bf16_old_venv`. The host working
directory remained `ddo-examples`, child SSLKEYLOGFILE was unset and HF_HOME
remained F:/hf_cache. No dependencies, source code or residency limits changed.

| Measurement | Primary environment | Historical environment |
| --- | ---: | ---: |
| Denoise, s | 203.7925 | 204.7288 |
| Encoder, s | 13.2534 | 13.1415 |
| Decode/save, s | 27.6654 | 27.6269 |
| Tracked total, s | 266.9297 | 268.3470 |

The historical environment did not recover the old 197.0318-second denoise;
its denoise was approximately 0.46% longer than the primary environment in
this single sequential control. This does not isolate each dependency or
prove equivalence, but provides no observed old-environment performance gain
for this selected video. Image performance remains a separate investigation.

Both interpreters are Python 3.12.11 with identical recorded Torch/CUDA core
pins. `uv pip freeze` found the main dependency difference: the historical
environment uses editable F:/projetos/diffusers at be07836d91c3dfee7147445863a8a28dd8a3d186;
the primary environment pins the DEVAIEXP fork at e71ebc5858961e2a37d6a7b161d00ae0063c3ab7,
not a moving upstream main. Other differing versions: filelock, hf-xet,
imageio, iniconfig, pycparser and typer; primary-only packages include SDNQ,
sentencepiece and ruff. Fork changes between these revisions are the timestep
fix and LTX activation/modulation changes, already migrated into local custom
blocks. Comparing these video transformer copies found only newline differences;
the new host does not rely on the modified Diffusers video forward.

The control completed exit 0, 121 frames at 1280x704, finite stereo generated
audio [2,240480] at 48000 Hz and successful cleanup. Both component snapshots
have zero fault failures, temporary fallback tensors, source admission denials,
quarantine, live source tickets and transfer-owned events. All monitor errors
were null. Global absolute driver peak 5.142089844 GiB; sampled RSS peak
28.414672852 GiB. Media structure is accepted, not perceptual/numerical parity.
Evidence: outputs/migration_video_bf16_old_venv/console.log,
metrics/modular_run.json and ltx23_modular_1280x704_5.0s_seed43.mp4 under that
output directory. No dependency change or automatic follow-up run follows.

### Image migration repeat after Windows reboot (2026-10-10)

The approved repeat used the primary root uv environment and the same BF16
image runner, model, DLL and settings as outputs/migration_image_gate. No code,
dependency or budget changed. The only intended differences were the reported
Windows reboot and a fresh output directory, outputs/migration_image_gate_after_reboot.
This single repeat recovers selected historical image performance; it does not
identify the earlier slowdown's root cause or prove general driver recovery.

| Measurement | Historical image | Before reboot | After reboot |
| --- | ---: | ---: | ---: |
| CUDA warm steps 2-8, s/it | 1.395602 | 1.524931 | 1.387940 |
| CUDA transformer phase with dependencies, s | 16.498256 | 18.657049 | 15.397565 |
| Tracked encoder phase, s | 12.6624 | 20.6153 | 11.5705 |
| Tracked pipeline total, s | 38.5135 | 50.7519 | 35.2571 |

The after-reboot case is BF16, 1280x704, seed 43, eight steps, default real
prompt, no PAG, one generation and one transformer preparation, with native
encoder and transformer. Both use a 4 GiB weight budget; the image staging CLI
value 0.5 resolves to 1.5 GiB total. Source cache and registered ceilings remain
24.625 GiB. FP8 is disabled. Windows standby purge was skipped because the
process token lacked SeProfileSingleProcessPrivilege; it was not a successful
purge intervention. All sampled monitor error fields are null.

Sampled absolute GPU driver peak is 6.878520966 GiB, baseline 1.026855469 GiB,
reported baseline-relative peak 5.8517 GiB, sampled RSS peak 27.366 GiB.
Both component snapshots have zero fault, guard/CUDA/capacity admission,
source-read, source-cache admission, pool and execution-plan failures, zero
temporary fallback tensors, quarantine, live source tickets and transfer-owned
events. Encoder/transformer upload bytes remain 25,545,825,792 / 187,588,136,576.
Host DMA enqueue counters are 4.199564 / 5.526979 seconds, not pure GPU DMA time.

Execution exited 0 and saved the PNG. Its SHA256 is
877876a4a8e1143a59e51a737ef3eb7f05a27980a6a44d20fed80d2ab581cced,
identical to the pre-reboot image. DLL SHA256 remains
755fcdd8dd1a72b371402750c071ca4b2146c0ac6bfe6b94c9b78f50f1ebb729.
Evidence is the console.log, PNG and metrics JSON beneath the fresh output
directory. Selected image functionality and performance are accepted; this
does not extend video, FP8, arbitrary-model or Linux acceptance. No further
CUDA run or dependency change is authorized by this result.

### Video BF16 repeat after Windows reboot (2026-10-10)

The separately approved repeat retained the primary root uv environment,
video runner, local custom blocks and joined-span DLL from
outputs/migration_video_bf16_vmm. It used 5 seconds, 1280x704, 24 fps,
121 frames, 8 steps, seed 43, the default real prompt/image and no conditioning
audio. Native encoder ceiling remained 4 GiB, transformer ceiling 1.5 GiB,
staging 1.5 GiB and source cache/registration ceilings 24.625 GiB. Chunk sizes
were 1024 with phase modulation, compact timesteps and in-place residuals.
PyTorch host-cache release and CUDA streamed decode remained enabled.
No code, dependency, budget or optimization changed.

| Tracked measurement, s | Before reboot | After reboot |
| --- | ---: | ---: |
| Encoder | 13.2534 | 11.4153 |
| Denoise | 203.7925 | 206.5028 |
| Decode/save | 27.6654 | 27.9140 |
| Total | 266.9297 | 266.7800 |

Denoise was approximately 1.33% longer; the total was effectively unchanged
in this single sequential comparison. Unlike the image repeat, no meaningful
video acceleration was observed after reboot. This does not isolate transient
driver state or prove a software regression. The 197.0318-second historical
video remains unmatched in DLL, budget and forward settings.

The runner calls the Windows standby helper before the transformer, but the
approved command did not enable standby purge and no successful purge event
was recorded. PyTorch host-cache release did execute; these are different
operations. An explicit standby-purge comparison remains a separate gate,
requiring successful privilege/event verification rather than assuming purge.

Execution exited 0 with status completed, 121 frames at 1280x704 and finite
generated stereo audio [2,240480] at 48000 Hz. MP4 SHA256 is
56bd4cf1d6f20cd2c978fda6280902197a24fc179e303ac98f65ee02eb7fd891,
identical to the pre-reboot video. Encoder and transformer snapshots show zero
fault/guard/CUDA/capacity admission, source read/cache admission, pool and
execution-plan failures; temporary fallback tensors, quarantine, live source
tickets and transfer-owned events are zero. All monitor errors are null.
Sampled absolute driver peak is 5.142089844 GiB, baseline 1.026855469 GiB,
relative peak 4.115234375 GiB and sampled RSS peak 28.416580200 GiB.

Evidence: outputs/migration_video_bf16_after_reboot/console.log,
metrics/modular_run.json and ltx23_modular_1280x704_5.0s_seed43.mp4.
Selected video functionality is accepted; timing is comparable to the previous
primary-host run, not improved historical performance or general model support.
No automatic follow-up test or change is authorized by this result.

### Administrative Windows standby-purge full-video results (2026-10-10)

User-run primary-host BF16 and saved TorchAO FP8 cases both completed a
5-second, 1280x704, 24-fps, 121-frame, 8-step video, seed 43, real prompt/image,
no input audio, generated audio enabled. Native encoder ceiling 4 GiB;
transformer weight/staging ceilings 1.5 GiB; source/registration ceilings
24.625 GiB. FF/normalization chunks 1024, phase/compact modulation and
in-place residuals enabled; PyTorch host-cache release and CUDA streamed decode
enabled. Compute compilation and FP8 chunk weight reuse were disabled.

| Tracked measurement | BF16 + purge | Saved FP8 + purge |
| --- | ---: | ---: |
| Transformer load/prepare, s | 1.4042 | 32.4251 |
| Transformer attach, s | 6.5249 | 10.4584 |
| Encoder phase, s | 12.2153 | 9.7655 |
| Denoise, s | 199.3626 | 218.5919 |
| Decode/save, s | 26.8720 | 26.9401 |
| Successful standby purge, s | 3.3066 | 3.9073 |
| Total including purge, s | 262.4741 | 312.9841 |
| Sampled absolute driver GPU peak, GiB | 5.142089844 | 5.923339844 |
| Sampled RSS peak, GiB | 28.418369293 | 25.378753662 |

Both report the successful before-transformer NtSetSystemInformation event,
class 80, command 4, with SeProfileSingleProcessPrivilege. Native encoder
and transformer snapshots have zero fault/guard/CUDA/capacity admission,
source-read, cache-admission, pool and execution-plan failures, temporary
fallback tensors, quarantine, live source tickets and transfer-owned events.
All phase monitor errors are null. Both report 121 frames, 1280x704 and finite
stereo audio [2,240480] at 48000 Hz. FP8 transport is executed for the FP8
transformer only, not the BF16 encoder. Its model argument resolves the root
F:/models/ltx23-distilled-fp8-weight-only; the source format remains saved
TorchAO weight-only FP8. DLL path is build-joined-spans-cu128-20261006/Release/ddo_vmm.dll.

Relative to the prior no-purge video, BF16 denoise fell from 206.5028 to
199.3626 s (3.46%); total fell from 266.7800 to 262.4741 s. One sequential
comparison does not isolate purge causality from system/cache state.
FP8 denoise was 9.65% longer than BF16 and total 19.24% longer, while sampled
RSS was about 3.04 GiB lower. Setup accounts for most additional total time;
the metrics do not isolate the underlying cause. Full-video functionality is
accepted for both selected cases; perceptual parity was not scored.

Artifacts: outputs/migration_video_bf16_standby_purge and
outputs/migration_video_fp8_standby_purge, each with metrics/modular_run.json
and ltx23_modular_1280x704_5.0s_seed43.mp4. User-pasted source reports were
39d34940-d442-4482-aa26-de3c7c13bd46 and
a9babd70-e81b-417e-95aa-7e48457ce9e7. No new CUDA execution or source change
was performed while recording these results.

## Qwen Image 2.1 native VAE: exact output, higher memory than leaf

User-run artifact: `outputs/qwen21_vae_native_vmm_20steps/metrics.json` and
`image.png`; compare with `outputs/qwen21_vae_leaf_trim_enabled`. Both completed
base BF16 1024x1024, 20 steps, seed 42, CFG 1, native KV cache, prompt
`A capybara wearing a wizard hat, reading by candlelight, oil painting`.
Untiled local single-frame decode and residual aliasing enabled; direct padding
disabled. Native transformer ceiling 4 GiB, staging 0.625 GiB, source cache
16 GiB. Successful standby purge before transformer and simple native idle trim
before VAE; no cuBLAS workspace clear. Same idle-trim candidate DLL SHA256
`468A854EB550C6299C3DE785B65ECD4F4DF382C488B6F1C83AA6899983F12CFE`.

| Measurement | Official leaf VAE + trim | Native VMM VAE + trim |
| --- | ---: | ---: |
| Total pipeline, s | 72.5433 | 70.6738 |
| Denoise through CPU readback, s | 52.8167 | 51.9337 |
| VAE load, s | 0.8469 | 0.0540 |
| Separate native VAE attach, s | N/A (leaf hooks included in load) | 0.1220 |
| VAE decode, s | 1.8943 | 1.4262 |
| Decode absolute sampled driver GPU peak, GiB | 3.92724609375 | 5.94677734375 |
| Decode sampled RSS peak, GiB | 4.5544 | 5.0889 |

GPU peaks are absolute driver usage, not baseline-relative deltas; sampled at
0.1-second intervals with RSS monitoring and no monitor errors. New run's
baseline was 1.02685546875 GiB. Its 6.12060546875 GiB global absolute peak was
during denoise, not decode. The legacy top-level `global_peak_vram_gb` is a
baseline-relative delta and must not replace the absolute values above.

The native VAE loaded a separately saved BF16 component at
`F:/models/qwen-image-21-vae-bf16`, rather than casting original FP32 weights
during load. Conversion verified all 238 tensors against that same source cast
and preserved the original. Saved weight SHA256:
`38fab976bd9bcd73fa0982736a2fb4f2549d1aee6a4a65b026f34fbc19d71f7b`.
This saved-source preparation affects load time; the comparison is not a pure
allocator-only setup A/B. Native VAE weight ceiling was explicitly 0.375 GiB,
separate from the transformer; two page-aligned execution slots occupied 332 MiB.
Twelve buffer uploads transferred 518,096,424 parameter bytes; arena residency
was zero. All 238 bindings were file-backed, with zero CPU source bindings,
fault/admission failures, temporary fallback, pool/source-read failures, live
source tickets or transfer-owned events. Two remaining completion tickets were
collected before normal cleanup. No real-VAE cancellation test was performed.

Both PNGs have the accepted 20-step SHA256
`1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45`.
Functionality and exact image equality passed. **The memory-reduction objective
failed:** native decode saved 0.4681 s but its sampled GPU peak was 2.01953125 GiB
higher (about 51.4%). Keep official leaf + trim as the selected capacity route;
native VAE remains experimental, not a replacement default. Slot residency
does not explain the entire difference; these reports do not isolate pool,
activation or workspace causality. Sequential timing variation is not proof
of a transformer speedup or a systematic end-to-end advantage. No additional
CUDA execution or parameter tuning was performed while recording this result.

### Qwen VAE linear versus leaf, normal allocator (2026-10-11)

User-executed full modular base BF16 pair: 1024 x 1024, seed 42, 20 steps,
CFG 1, native KV cache, prompt `A capybara wearing a wizard hat, reading by
candlelight, oil painting`, transformer preset one_shot_fast. Both used accepted
single-frame cacheless/residual-alias VAE options, no tiling or direct padding,
and successful Windows standby purge before the transformer. This pair used
the normal PyTorch allocator, not native VMM.

| VAE route | Total wall s | Denoise through CPU readback s | Decode s | Sampled absolute decode GPU GiB | Decode process RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official leaf, no stream | 77.6654 | 53.7452 | 1.5958 | 6.469936 | 14.5509 |
| DDO linear convolution | 77.3947 | 54.2367 | 1.3642 | 6.503990 | 14.8634 |

Artifacts: `outputs/qwen21_linear_vae_leaf/metrics.json` and
`outputs/qwen21_linear_vae_linear/metrics.json`, with image.png in each directory.
Both reports have status completed, 20 executed steps and successful save/cleanup
phases. Both PNGs match the accepted SHA256
`1FF1ECA154725130C6205ADF273CE069CD192BA8F8F385C07FEA792FF7753B45`.
Linear decode was 0.2316 s shorter (14.5%) in this pair, with no demonstrated
capacity improvement: the sampled peak was 0.034054 GiB higher. Sampling interval
was 100 ms. This single sequential pair is not proof of a repeatable pipeline
speedup; no additional runs were made during analysis.

Do not directly rank these absolute memory numbers against the native-transformer
leaf+trim/native-VAE pair above: process allocator and preceding component routes
differ. Keep native transformer plus leaf and explicit idle trim as the selected
capacity combination. Linear and native VMM VAE are compatible for this selected
checkpoint, not every VAE. DDO now owns VAE preset routing and the host delegates
the selection; this subsequent policy migration has CPU tests only, not new
performance acceptance. Historical artifacts were not rewritten.

## 2026-10-11: LTX image/video VAE resident versus no-stream leaf

Selected Windows user-run gates in the primary host root uv environment,
with the local Diffusers fork and independent local custom-block clones.
Only the VAE decode route and output directory change within each pair;
conditioning encode, transformer math/budgets, checkpoint dtype and audio
routes are unchanged. Leaf uses DDO's public API and official Diffusers
leaf-level group offload, use_stream=false, low_cpu_mem_usage=false.
This is not native VMM VAE execution or DDO linear convolution offload.

Both workloads use BF16, 1280x704, seed 43, eight steps and native text
encoding. Image uses the real default prompt, no PAG, a 4 GiB transformer
ceiling and 1.5 GiB total staging. Video uses five seconds at 24 fps
(121 frames), the existing guitar image/prompt, no conditioning audio,
generated output audio, the 1.5 GiB transformer ceiling, 4 GiB encoder
ceiling, 1.5 GiB staging, chunk sizes 1024, phase modulation, compact
timesteps and in-place residuals. Source/cache registration limits are
24.625 GiB. Video retains the existing temporal streaming decode, spatial
tiling setting, blend geometry and CPU stitching; image adds no tiling.
Successful Windows standby purge before the transformer is recorded in
all four artifacts, and its cost remains in tracked totals.

Selected DLL: build-joined-spans-cu128-20261006/Release/ddo_vmm.dll,
SHA256 EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.
These gates do not validate another binary or platform. The VAE/telemetry
candidate was an uncommitted working-tree change; no immutable candidate
revision is claimed. Missing installed inventory fields remain unavailable.

### Image phases

| Phase | Resident time (s) | Leaf time (s) | Resident VRAM (GiB) | Leaf VRAM (GiB) | Resident RSS (GiB) | Leaf RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pass 0: Encode prompts | 14.29 | 14.65 | 6.79 | 6.81 | 25.12 | 25.12 |
| Pass 1: Generate at 1280x704 | 21.72 | 22.28 | 7.15 | 7.17 | 27.37 | 27.37 |
| Load VAE for decode | 1.03 | 0.06 | 2.88 | 1.68 | 2.86 | 1.61 |
| Decode VAE | 0.33 | 1.12 | 3.19 | 2.23 | 1.65 | 3.17 |
| Cleanup VAE | 0.14 | 0.16 | 3.19 | 2.03 | 1.66 | 3.19 |
| Save Image | 0.07 | 0.07 | 2.07 | 2.03 | 1.66 | 2.43 |

Tracked total: 39.18 s resident / 40.34 s leaf.
Global sampled absolute GPU peak: 7.15 / 7.17 GiB;
global process RSS: 27.37 / 27.37 GiB.
Load+decode+cleanup aggregate: 1.4978 / 1.3483 s. The decode-only
phase shifts upload cost into leaf execution and must not be used alone
as an end-to-end comparison. The total-time difference includes earlier
phases and is not attributable entirely to the VAE.

Both saved PNGs match the historical image SHA256:
877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED.

### Video phases

| Phase | Resident time (s) | Leaf time (s) | Resident VRAM (GiB) | Leaf VRAM (GiB) | Resident RSS (GiB) | Leaf RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text_encoder | 12.08 | 12.15 | 5.14 | 4.96 | 25.29 | 25.29 |
| connectors | 6.50 | 6.47 | 2.82 | 2.82 | 11.31 | 11.36 |
| video_encoder | 3.85 | 3.21 | 1.45 | 1.45 | 8.09 | 8.09 |
| audio_encoder | 0.15 | 0.14 | 1.45 | 1.45 | 6.73 | 6.73 |
| load_and_prepare_transformer | 1.81 | 1.99 | 1.71 | 1.71 | 6.88 | 6.88 |
| attach_transformer | 6.23 | 6.66 | 1.72 | 1.72 | 3.75 | 3.75 |
| denoise | 198.58 | 199.30 | 4.57 | 4.57 | 28.42 | 28.42 |
| load_decode_video | 0.96 | 0.06 | 2.41 | 1.52 | 3.57 | 2.27 |
| decode_video | 22.66 | 32.88 | 4.81 | 3.02 | 7.16 | 9.27 |
| cleanup_decode_video | 0.25 | 0.29 | 2.89 | 1.57 | 5.89 | 7.88 |
| load_decode_audio_mel | 0.14 | 0.17 | 1.54 | 1.57 | 5.95 | 7.10 |
| decode_audio_mel | 0.07 | 0.06 | 1.54 | 1.57 | 5.89 | 7.06 |
| cleanup_decode_audio_mel | 0.28 | 0.22 | 1.54 | 1.57 | 5.90 | 7.08 |
| load_decode_audio | 0.47 | 0.50 | 1.54 | 1.57 | 6.14 | 7.29 |
| decode_audio | 0.48 | 0.48 | 1.54 | 1.57 | 6.02 | 7.18 |
| cleanup_decode_audio | 0.24 | 0.24 | 1.54 | 1.57 | 6.02 | 7.19 |
| save_video | 1.34 | 1.33 | 1.54 | 1.57 | 6.83 | 7.99 |

Tracked total: 261.67 s resident / 271.83 s leaf.
Global sampled absolute GPU peak: 5.14 / 4.96 GiB;
global process RSS: 28.42 / 28.42 GiB.
Decode+save aggregate (including video/audio load and cleanup):
26.9105 / 36.2375 s. Both reports completed with 121 frames at
1280x704 and finite two-channel audio, shape [2, 240480], 48000 Hz.
Both saved MP4s match the historical video SHA256:
56BD4CF1D6F20CD2C978FDA6280902197A24FC179E303AC98F65EE02EB7FD891.
This confirms identical delivered media, not a perceptual score or a
separate capture of pre-encoding tensors.

### Interpretation and artifacts

Leaf lowers the VAE decode GPU peak by approximately 0.96 GiB for image
and 1.79 GiB for video. Decode RSS rises by approximately 1.52 and 2.11 GiB.
Video decode is 10.22 s longer; retain resident as the existing default,
with leaf as an explicit VAE capacity option. Global peaks are dominated
by earlier components and must not be advertised as the VAE savings.
These are single sequential observations, not statistically established
speedups or low-system-RAM acceptance.

Image artifacts: outputs/ltx_image_vae_resident_gate and
outputs/ltx_image_vae_leaf_gate (PNG plus metrics/<run_slug>.json).
Video artifacts: outputs/ltx_video_vae_resident_gate and
outputs/ltx_video_vae_leaf_gate (MP4 plus metrics/modular_run.json).
Phase memory is sampled absolute CUDA driver usage and process RSS,
in GiB, at 100 ms intervals; it is not total system commit or shared WDDM
occupancy. New load/decode/cleanup/save phases are separate; legacy combined
decode names are aggregate events, not extra phases to sum. Historical
JSON is preserved unchanged. Denoise phase averages are not CUDA steady-state.
See docs/benchmark-summary.md for VAE-only hardware recommendations.

## 2026-10-11: native Linux LTX image resident VAE control

The approved single control completed with exit 0, image and metrics saved.
Leaf/video comparisons remain pending. Primary host `.venv-linux`, Python
3.12.14, Torch 2.12.1+cu132, local editable Diffusers 0.41.0.dev0 at revision
7d047b180, local DDO/allocator and independent local image blocks. Ubuntu 26.04.1,
RTX 3060 Ti 8192 MiB. Candidate build-linux-idle-trim-20261011/libddo_vmm.so,
SHA256 db63138da2fdcca328a244ff885ca44ecd8b33683825b2ab0f6f39a95b3cba79.
Span pool ON; joined spans/probes/dispatcher/test hooks/diagnostics/independent
kernels OFF. Uncommitted source candidate; no immutable revision or promotion.

Preflight GPU 463 MiB used/7377 MiB free; MemAvailable 58,719,080 kB,
CommitLimit 40,090,756 kB, Committed_AS 15,131,200 kB; overcommit mode 0.
No kernel policy or capacity adjustment. Output directory absent before the run.
BF16, real default prompt, 1280x704, seed 43, eight steps, one generation, no PAG.
Native encoder then native transformer, resident VAE; active ceiling 4 GiB,
total staging 1.5 GiB, source/registration ceilings 24.625 GiB. No automatic trim,
cuBLAS clear, added tiling or Linux standby purge. VAE executed route none/resident.
Checkpoint: /run/media/eliseu/IA/hf_cache/hub/models--elismasilva--ltx2.3-image-distilled-1.1/snapshots/5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9.
The selected Windows image recipe was retained with Linux interpreter, library,
cache paths and this explicit Linux snapshot. No workload tuning occurred.

| Phase | Time (s) | Sampled absolute GPU peak (GiB) | Process RSS peak (GiB) |
| --- | ---: | ---: | ---: |
| Prompt encoding | 15.6269 | 6.259766 | 25.1432 |
| Generate including connectors/transformer setup | 23.2872 | 6.559631 | 27.3297 |
| VAE load | 0.6600 | 2.435242 | 2.8960 |
| VAE decode | 0.3418 | 2.758850 | 1.6525 |
| VAE cleanup | 0.1675 | 2.758850 | 1.6843 |
| Image save | 0.2106 | 1.643616 | 1.6595 |

Tracked total 40.2959 s excludes bootstrap/import startup. Global absolute GPU
peak 6.559631 GiB, RSS 27.3297 GiB. Combined VAE event 1.1702 s is an aggregate,
not an additional phase. Denoise modular-call wall time 17.1203 s; wall phase
average 2.1400 s/step includes first-step warmup. Completed CUDA compute-stream
phase with dependencies 17.399541 s; first interval 10.006696 s; CUDA mean steps
2-8 1.056121 s/step. All eight intervals complete, timing.failed=false.
Host callbacks are not completed CUDA intervals.

No reported native failures, temporary fallbacks, quarantine, admission denials,
source-cache bypasses or monitor errors. Encoder source committed/registered
23,555,211,264 bytes; transformer 26,038,239,232 bytes, within fixed ceilings.
Transformer completion collection retired 411 tickets to zero. Final recorded
source committed/registered bytes and live tickets are zero. This is one selected
staged workload, not arbitrary-model or driver-loss acceptance.

Saved PNG: 1280x704, visually plausible; SHA256
dfbf135a833b5a77f4574a98cebd3be84d19fc569b79de8dd5f18457c52d64c1.
It differs from Windows; no cross-platform pixel equality is inferred. Use this
Linux image as the resident control for the pending leaf comparison.
Artifacts: outputs/ltx_image_vae_resident_linux_gate, PNG plus metrics/<run_slug>.json.
Raw console: .internal/logs/ltx_image_vae_resident_linux_gate.log, local/ignored.
GPU/RSS peaks are sampled, not instantaneous enforced caps, system commitment,
WDDM/shared usage or model residency. Raw artifacts remain unchanged.

## 2026-10-11: native Linux LTX image VAE pair closed

The approved leaf run completed with exit 0, retaining the resident control's
exact command, checkpoint, native binary, budgets and model settings. Only
`--vae-leaf-offload` and the output directory changed. Preflight: GPU 460 MiB
used/7380 MiB free, MemAvailable 58,773,140 kB; candidate hash unchanged.
Leaf executed official Diffusers leaf-level group offload through DDO,
use_stream=false and low_cpu_mem_usage=false, not native VMM VAE transport.

| Phase | Resident (s) | Leaf (s) | Resident absolute GPU (GiB) | Leaf absolute GPU (GiB) | Resident RSS (GiB) | Leaf RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt encoding | 15.6269 | 14.9689 | 6.259766 | 6.234192 | 25.1432 | 25.1297 |
| Generate including setup | 23.2872 | 21.5114 | 6.559631 | 6.534973 | 27.3297 | 27.3595 |
| VAE load | 0.6600 | 0.0206 | 2.435242 | 1.269348 | 2.8960 | 1.5980 |
| VAE decode | 0.3418 | 0.7482 | 2.758850 | 1.821838 | 1.6525 | 3.1866 |
| VAE cleanup | 0.1675 | 0.2049 | 2.758850 | 1.596924 | 1.6843 | 3.2335 |
| Image save | 0.2106 | 0.1980 | 1.643616 | 1.596924 | 1.6595 | 1.6871 |

Tracked total 40.2959 / 37.6539 s (resident/leaf); global sampled absolute
GPU 6.559631 / 6.534973 GiB, RSS 27.3297 / 27.3595 GiB. Denoise modular-call
wall time 17.1203 / 17.0253 s; wall phase average 2.1400 / 2.1282 s/step,
including first-step warmup. Completed CUDA phase with dependencies
17.399541 / 17.303381 s; CUDA mean steps 2-8 1.056121 / 1.056260 s/step.
Both have all eight completed CUDA intervals and timing.failed=false.
Host callback intervals are not steady-state CUDA measurements.

Leaf VAE combined load/decode/cleanup aggregate is 0.9746 s versus resident
1.1702 s. GPU decode peak falls by 0.937012 GiB; decode RSS rises by 1.5341 GiB.
Leaf uploads occur during decode rather than initial GPU loading. Earlier
prompt/setup phases also differ, so the 2.6420 s shorter pipeline total is not
an allocator-only or VAE-only speedup. These are single sequential observations;
residency defaults remain unchanged, with leaf an explicit capacity alternative.

Both PNGs have the exact same SHA256:
dfbf135a833b5a77f4574a98cebd3be84d19fc569b79de8dd5f18457c52d64c1.
No reported native failures, temporary fallbacks, quarantine, admission denials,
source-cache bypasses or monitor errors. Local delivered-image parity and the
selected native encoder/transformer plus leaf VAE route pass. This does not
establish pixel equality across Windows/Linux, video acceptance or WSL support.

Artifacts: outputs/ltx_image_vae_resident_linux_gate and
outputs/ltx_image_vae_leaf_linux_gate, each with PNG and metrics/<run_slug>.json.
Raw leaf console: .internal/logs/ltx_image_vae_leaf_linux_gate.log, local/ignored.
No idle trim or cuBLAS clear was enabled in either image workload. Measurements
retain sampled driver GPU/RSS scope and fixed 100 ms sampling; global peaks
remain dominated by earlier components, not the VAE. Video comparison is pending.

## 2026-10-11: native Linux LTX video VAE pair closed

Both approved five-second/eight-step runs completed, exit 0, with 121 frames
at 1280x704 and finite two-channel generated audio, shape [2, 240480], 48000 Hz.
PyAV independently decoded all 121 encoded frames and confirmed 24 fps,
1280x704, stereo 48000 Hz. Both delivered MP4 SHA256 values are identical:
2c0c9491b0394ec3ee472a45b69cbf03f5afecd8ded3bca5b8ae509d56d270ef.
This is within-Linux output parity, not cross-platform byte equality or a
perceptual score; Windows delivers a different encoded hash.

The primary host environment and isolated candidate are the same as the image
pair above: Torch 2.12.1+cu132, local merged Diffusers 7d047b180, local editable
DDO/allocator, Linux span pool ON with joined spans/probes/dispatcher/test hooks/
diagnostics/independent kernels OFF. Candidate SHA256
 db63138da2fdcca328a244ff885ca44ecd8b33683825b2ab0f6f39a95b3cba79.
The source working tree remains uncommitted; no immutable revision/promotion.

Checkpoint: OzzyGT/LTX-2.3-Distilled-1.1, local cache snapshot
f070fb11c2af41e90ad9012d1f495b5131753a4d. BF16, default guitar prompt, local
conditioning image .internal/inputs/man_guitar.png (1280x704 RGBA), SHA256
213964e961f05ee39a9a21ac07ec93f146e73ab135b79cda57f773928a81db90.
This was retrieved from the same public image URL used on Windows; its Windows
input hash was not recorded, so exact cross-platform input identity is not proven.
Five seconds, 24 fps, seed 43, eight steps, no input audio, generated audio.
Native encoder ceiling 4 GiB; transformer/staging ceilings 1.5 GiB each;
source and registration 24.625 GiB. Preserve chunk sizes 1024, phase modulation,
compact timesteps, in-place residuals, host cache release and temporal streaming.
No automatic idle trim/cuBLAS clear, budget change, new compute kernel or Linux
standby purge. Only --vae-leaf-offload and output directory differ within the pair.
Leaf is official no-stream Diffusers leaf via DDO, low_cpu_mem_usage=false;
audio VAE/vocoder and conditioning encode routes remain unchanged.

Preflight: resident GPU 449 MiB used/7391 MiB free, MemAvailable 58,683,288 kB,
CommitLimit 40,090,756 kB, Committed_AS 15,354,740 kB, overcommit mode 0.
Leaf GPU 460 MiB used/7380 MiB free, MemAvailable 58,752,948 kB. Output directories
were absent. No kernel policy adjustment, failed-gate retry or new case was added.

| Phase | Resident (s) | Leaf (s) | Resident GPU (GiB) | Leaf GPU (GiB) | Resident RSS (GiB) | Leaf RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text_encoder | 14.7144 | 14.6544 | 4.777649 | 4.787964 | 25.2879 | 25.2893 |
| connectors | 5.4066 | 4.6470 | 2.429382 | 2.426880 | 11.0601 | 11.1483 |
| video_encoder | 1.8310 | 1.6078 | 1.038818 | 1.036255 | 7.7762 | 7.7701 |
| audio_encoder | 0.1606 | 0.1302 | 1.039001 | 1.036255 | 6.4272 | 6.4202 |
| load_and_prepare_transformer | 2.5370 | 2.0084 | 1.468689 | 1.465942 | 6.8975 | 6.8915 |
| attach_transformer | 1.5457 | 1.5644 | 1.220642 | 1.219849 | 3.7563 | 3.7504 |
| denoise | 209.7483 | 209.5043 | 4.141418 | 4.138306 | 28.4240 | 28.4177 |
| load_decode_video | 0.2335 | 0.0212 | 2.027344 | 1.108826 | 2.5040 | 1.8548 |
| decode_video | 24.3894 | 32.5026 | 4.871399 | 2.589294 | 6.2289 | 7.5149 |
| cleanup_decode_video | 0.2587 | 0.2959 | 2.470703 | 1.130737 | 2.2552 | 3.8219 |
| load_decode_audio_mel | 0.0888 | 0.0296 | 1.113281 | 1.130737 | 2.2552 | 2.5335 |
| decode_audio_mel | 0.0466 | 0.0287 | 1.113281 | 1.130737 | 2.2552 | 2.5335 |
| cleanup_decode_audio_mel | 0.2320 | 0.2296 | 1.113342 | 1.130737 | 2.2608 | 2.5375 |
| load_decode_audio | 0.3003 | 0.1737 | 1.113342 | 1.130615 | 2.3941 | 2.5477 |
| decode_audio | 0.4537 | 0.3903 | 1.164429 | 1.132568 | 2.3780 | 2.5830 |
| cleanup_decode_audio | 0.2568 | 0.2586 | 1.164429 | 1.156006 | 2.3791 | 2.5830 |
| save_video | 1.2484 | 1.1073 | 1.154297 | 1.145935 | 3.1913 | 3.2174 |

| Aggregate scope | Resident | Leaf |
| --- | ---: | ---: |
| Denoise wall (s) | 209.7483 | 209.5043 |
| Wall phase average per step (s, includes warmup) | 26.2185 | 26.1880 |
| Decode/save aggregate (s) | 27.5134 | 35.0427 |
| Tracked total (s) | 266.5132 | 272.2485 |
| Global sampled absolute GPU peak (GiB) | 4.871399 | 4.787964 |
| Global RSS peak (GiB) | 28.423954 | 28.417744 |

Leaf decode saves 2.282104 GiB of sampled absolute GPU usage, with 8.1132 s
additional decode time and 1.2860 GiB more decode RSS. Native denoise differs by
only 0.2440 s between these single observations; no denoise optimization is
attributed to the VAE. Keep resident default, with leaf as a capacity alternative.
Global memory is not the VAE-specific savings. All reported native failure/
admission/quarantine/fallback and monitor-error observations are zero/absent.
This closes selected Linux image/video VAE compatibility, not arbitrary models,
FP8/SDNQ, VAE linear/native variants, WSL or driver-loss support.

No synchronized per-step CUDA timing was requested in this video benchmark:
wall averages include warmup and are not completed CUDA steady-state intervals.
Driver GPU/RSS peaks are sampled at 100 ms, not enforced instantaneous caps or
system commitment. Decode/save events are aggregates, not phases to double-count.
The streaming helper disables VAE use_tiling/use_framewise_decoding during each
window and restores them afterward; temporal streaming is not spatial tiling.
No new spatial-tiling fallback was tested in this gate.

Artifacts: outputs/ltx_video_vae_{resident,leaf}_linux_gate, MP4 plus
metrics/modular_run.json. Raw console: .internal/logs/ltx_video_vae_{resident,leaf}_linux_gate.log,
local/ignored. The prior Linux .so and all raw artifacts remain unchanged.

## 2026-10-11: Qwen Image 2.1 Linux shared 512-square trio

The approved three-run sequence completed with exit 0 and 20 executed steps in
all reports. Base Qwen/Qwen-Image-2.1 BF16, 512x512, seed 42, default capybara
prompt, CFG 1, native KV cache and stock resident AutoencoderKLQwenImage21 VAE.
No tiling, cacheless VAE, residual aliasing, direct padding, idle trim, cuBLAS
clear, compute compilation, Turbo, FP8 or SDNQ. Exact delivered PNG SHA256 in
all three cases: f53ebe128845d491a98a02b56e693d1b993fecf6f2777239df84d920612eb11f.
The inspected control image has the expected size and plausible content; no
perceptual score is claimed. Historical Windows 1024 cases are separate workloads.

Native Linux primary host .venv-linux, Python 3.12.14, Torch 2.12.1+cu132, local
merged Diffusers 7d047b180 / 0.41.0.dev0, local editable DDO/allocator and local
Qwen blocks. Local checkpoint snapshot d26bb61231c349cf6b7896fa83353113880e1ba3.
VMM uses the isolated candidate SHA256
 db63138da2fdcca328a244ff885ca44ecd8b33683825b2ab0f6f39a95b3cba79,
span pool ON / joined spans OFF. Code is a local working tree, not an immutable
published checkpoint. Conventional and linear runs use normal PyTorch allocation.

The conventional control retains its full pipeline and applies official streamed
leaf to encoder/transformer: use_stream=true, record_stream=false,
low_cpu_mem_usage=true, non_blocking=false; VAE resident. DDO runs are phase-staged
and release encoder/pipeline references before transformer execution. Linear
auto resolves one_shot_fast; observed auto plan resident 4.66 GiB and pinned
weight budget 11.24 GiB. Native active/staging/source/registration ceilings are
4/0.625/16/16 GiB; no budget adjustment or failed-case retry occurred. The native
encoder route is explicit Diffusers compatibility, not native encoder ownership.

Preflight: GPU 461 MiB used/7379 MiB free, MemAvailable 58,376,832 kB,
CommitLimit 40,090,756 kB, Committed_AS 15,457,584 kB, overcommit mode 0. Output
directories absent. GPU used 459 MiB/free 7381 MiB before VMM. No kernel policy
or standby purge on Linux. 27 Qwen/block/local-VAE CPU contracts passed first.

| Linux route | Denoise/readback (s) | Phase average (s/step) | Tracked total (s) | Global GPU (GiB) | Global RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official streamed leaf (full pipeline) | N/A | N/A | 45.3132 | 4.800171 | 32.598476 |
| DDO linear auto (staged) | 18.1429 | 0.9071 | 34.7060 | 5.580688 | 18.897018 |
| DDO VMM fast (staged) | 15.6808 | 0.7840 | 23.2622 | 5.719055 | 18.867943 |

Conventional generate phase 42.8842 s includes encoding, denoise and decoding;
its isolated denoise/VAE times are unavailable, not zero. The other two denoise
values include CPU latent readback and first-step warmup. No --ddo-gpu-timing was
requested; synchronized CUDA warm means are N/A. Console callbacks remain
asynchronous host observations and are not CUDA steady-state measurements.

Native total is 48.7% lower than this conventional control and
33.0% lower than linear in these single sequential observations. These are
system/lifecycle-inclusive comparisons, not allocator-only gains. Native setup
is shorter than linear pin/residency preparation; staged encoder cleanup also
explains why conventional retained-pipeline RSS is not a pure allocator metric.
Native GPU peak is 0.92 GiB higher than the conventional control;
process RSS is 13.73 GiB lower. Do not extrapolate to a low-RAM host.


### leaf phase observations

| Phase | Wall (s) | Sampled absolute GPU (GiB) | RSS (GiB) |
| --- | ---: | ---: | ---: |
| load_pipeline | 1.2388 | 0.592896 | 2.5912 |
| attach_leaf | 0.0957 | 0.601013 | 1.6282 |
| generate | 42.8842 | 4.800110 | 32.5842 |
| cleanup_pipeline | 0.9828 | 4.800171 | 32.5985 |
| save_image | 0.1100 | 0.669312 | 4.0103 |

### linear phase observations

| Phase | Wall (s) | Sampled absolute GPU (GiB) | RSS (GiB) |
| --- | ---: | ---: | ---: |
| metadata | 0.0085 | 0.592590 | 0.8509 |
| load_text_encoder | 0.7228 | 0.592590 | 0.9729 |
| attach_text_encoder | 0.0225 | 0.592224 | 1.0043 |
| encode | 3.3761 | 1.854065 | 18.7393 |
| cleanup_text_encoder | 0.5763 | 1.854126 | 18.8970 |
| purge_before_transformer | 0.0002 | 0.670532 | 3.5644 |
| load_transformer | 0.0333 | 0.670532 | 3.5644 |
| attach_transformer | 9.3609 | 5.188049 | 16.6348 |
| prepare | 0.1612 | 5.188049 | 9.3809 |
| denoise | 18.1430 | 5.580688 | 9.5071 |
| collect_transformer | 0.0073 | 5.580688 | 9.5077 |
| cleanup_transformer | 1.3487 | 5.580688 | 9.7076 |
| load_vae | 0.2328 | 1.012146 | 11.1273 |
| decode | 0.3825 | 3.135315 | 10.3877 |
| collect_vae | 0.0001 | 3.275940 | 10.3908 |
| cleanup_vae | 0.1851 | 3.275940 | 10.3908 |
| save_image | 0.1341 | 0.689819 | 9.9694 |

### vmm phase observations

| Phase | Wall (s) | Sampled absolute GPU (GiB) | RSS (GiB) |
| --- | ---: | ---: | ---: |
| metadata | 0.0087 | 0.585571 | 0.8221 |
| load_text_encoder | 0.6308 | 0.585571 | 0.9665 |
| attach_text_encoder | 0.0228 | 0.582397 | 0.9749 |
| encode | 3.4170 | 1.808899 | 18.8489 |
| cleanup_text_encoder | 0.5660 | 1.803589 | 18.8679 |
| purge_before_transformer | 0.0002 | 1.803589 | 3.5312 |
| load_transformer | 0.0562 | 1.803589 | 3.5312 |
| attach_transformer | 0.6126 | 1.805542 | 4.1932 |
| prepare | 0.0095 | 1.797119 | 4.1791 |
| denoise | 15.6808 | 5.719055 | 17.6165 |
| collect_transformer | 0.0028 | 5.705322 | 17.6166 |
| cleanup_transformer | 1.4444 | 5.705322 | 17.6167 |
| load_vae | 0.2188 | 1.812561 | 5.3344 |
| decode | 0.3391 | 3.834290 | 4.2052 |
| collect_vae | 0.0001 | 2.314697 | 4.2054 |
| cleanup_vae | 0.1383 | 2.314697 | 4.2054 |
| save_image | 0.1085 | 2.312744 | 3.9176 |

All three reports have no monitoring errors; native snapshots report no failure,
quarantine, admission denial or temporary fallback. Native transformer completion
collection retired 258 tickets to zero. The upstream processor logged min_frames/
max_frames docstring diagnostics and group-offload warned about vision branches
not executed in the text-only encoder. Those logs were preserved, not suppressed;
there was no corresponding runtime exception or device mismatch.

Artifacts: outputs/qwen21_{leaf,linear,vmm}_linux_512, image.png plus metrics.json.
Raw console: .internal/logs/qwen21_{leaf,linear,vmm}_linux_512.log, local/ignored.
All time/resource scopes remain in raw JSON. RAM is process RSS, GPU sampled
absolute driver usage at 100 ms, not system commit, model residency or a hard
instantaneous cap. Totals exclude bootstrap/import startup. No Windows speed
ratio, 1024 Linux, other presets, quantized recipe, VAE offload variant, WSL or
arbitrary-model support is inferred. The finite approved Linux trio is closed.

### Official PyPI Diffusers 0.41.0 Windows compatibility gate (2026-10-11)

The user executed three finite native BF16 generations from the primary host's
Windows .venv after replacing only Diffusers with its official 0.41.0 PyPI
package. Torch remained 2.12.1+cu132; no allocator source, binary, driver or
residency ceiling changed. PYTHONPATH no longer injected the local fork.
CPU checks preceded generation: 95 host tests plus 10 subtests, five LTX
custom-block compatibility tests and 34 selected DDO CPU contracts passed.

The first image attempt failed before transformer loading because the default
AutoModel dispatch looked for the fork-only LTX2ImageTransformer2DModel export
in installed Diffusers. Image minimal/linear/VMM runners were corrected to
pass their custom-block transformer class explicitly to DDO's model_loader.
Ten selected CPU checks, including an exact tiny-checkpoint round-trip with
AutoModel/CUDA forbidden, passed before the successful user repeat.
This was a host loading correction, not a generic DDO or allocator change.

| Official package case | Denoise wall (s) | Tracked total (s) | Successful purge event (s) | Global sampled absolute GPU peak (GiB) | Global process RSS peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LTX image BF16, 1280x704, 8 steps, seed 43 | 13.7679 | 35.2343 | 1.6156 | 7.114185 | 27.3695 |
| LTX video BF16, 1280x704, 5 s / 24 fps / 121 frames, 8 steps, seed 43 | 197.3890 | 259.0260 | 2.5028 | 5.142090 | 28.420605 |
| Qwen-Image 2.1 base BF16, 512x512, 20 steps, seed 42 | 17.6734 | 36.639479 | 3.6663 | 6.120605 | 18.970501 |

All three executed the Windows standby purge before the transformer using
SeProfileSingleProcessPrivilege. Purge cost remains in tracked totals. These
are staged-run totals, not shell wall time including import/bootstrap. GPU
observations are sampled absolute driver usage, not allocator weight residency;
RAM is process RSS, not system RAM or commit. Denoise is host phase wall time,
not a pure kernel sum or steady-state per-step measurement.

LTX image used real prompt, no PAG, native encoder then transformer, 4 GiB
weight ceiling, total staging 1.5 GiB (image CLI 0.5), and 24.625 GiB source
storage/registration. Resident VAE, no added tiling. Prior resident-VAE gate:
13.9869 s denoise / 39.1770 s tracked total. Output PNG SHA256 is identical:
877876A4A8E1143A59E51A737EF3EB7F05A27980A6A44D20FED80D2AB581CCED.

LTX video retained native encoder ceiling 4 GiB, transformer ceiling 1.5 GiB,
staging 1.5 GiB, source storage/registration 24.625 GiB, chunk sizes 1024,
phase modulation, compact timesteps and in-place residuals. It used real
image/prompt, no input audio, generated output audio, temporal CUDA streaming
decode and resident VAE. Prior administrative-purge record:
199.3626 s denoise / 262.4741 s total; recent resident-VAE gate:
198.5835 s / 261.6728 s. MP4 SHA256 is identical to both:
56BD4CF1D6F20CD2C978FDA6280902197A24FC179E303AC98F65EE02EB7FD891.
Current report is completed, error=null; output audio [2,240480] at 48000 Hz
is finite. Both LTX runs used the preserved joined-spans CUDA 12.8 DLL,
SHA256 EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6.

Qwen used the base checkpoint and capybara/wizard/candlelight prompt, CFG 1,
native KV cache, source ceiling/registration 16 GiB, weight ceiling 4 GiB,
staging 0.625 GiB, stock resident untiled VAE and no automatic cache trim.
Text encoder remains official leaf compatibility. Its local independent
custom blocks were loaded through trust_remote_code, not from published Hub
code (the local code commit has not been pushed). The report records the
official installed Diffusers path and version, completed status and error=null.
The selected idle-trim DLL is unchanged, SHA256
468A854EB550C6299C3DE785B65ECD4F4DF382C488B6F1C83AA6899983F12CFE.
Image SHA256 matches the historical Windows 512-square VMM image:
550C28BB34648ABE520D59C5C5506CB55488F221E8CCFB6F22D8E13CCD1B6CF1.
That older run used full/USS monitoring and different phase reporting, so its
43.6096 s total is not a matched package-release speedup. Linux timings are
also not a matched Windows dependency comparison.

Artifacts (raw evidence unchanged):
- outputs/official_diffusers_041_image_purge: PNG and metrics/<run_slug>.json.
- outputs/official_diffusers_041_video_purge: MP4 and metrics/modular_run.json.
- outputs/official_diffusers_041_qwen512_purge: image.png and metrics.json.

The selected Windows BF16 gate passed with identical generated files and no
observed performance regression in the LTX cases. Single runs do not isolate
small timing changes from cache/OS/observer variation. This does not validate
all presets, FP8/SDNQ, other resolutions, Linux/WSL, or the historical conventional
LTX image apps. See docs/official-diffusers.md for transition and limitations.

## 2026-10-12: native Linux official Diffusers/cgroup candidate targeted image gate

User-approved single generation, not a rerun of the closed platform matrix.
Official Diffusers 0.41.0 replaced only the editable fork in host `.venv-linux`;
Torch remains 2.12.1+cu132 and TorchAudio 2.11.0+cpu. Candidate allocator
build-linux-cgroup-review-20261012/libddo_vmm.so SHA256
5a49e0a6acd672186af116ee31fe1409bd33c605b2846bfc5f9d4ae42d79907c.
The cgroup-root fix passed native CPU, compilation/linkage, fresh-process trim
and three small source/cache contracts. The changed package environment passed
152 CPU tests and 10 subtests before this generation.

Host HEAD 153d79852d21c29de18eb566740c07526e87d8eb; local image custom-block
HEAD 1eba529b4a838a8590fbc362ad2249dd8f8b01ea, with existing dirty metadata/docs.
Allocator HEAD bf3280600f0a8695547b1847459989e75b8eac9a with reviewed dirty
cgroup header/test. No immutable source checkpoint or promotion is claimed.
Use the direct host .venv-linux interpreter and explicit candidate environment
selection; native bootstrap installs before CUDA. Checkpoint snapshot remains
5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9 from the local cache.

BF16 native encoder/transformer, resident VAE; real default prompt, 1280x704,
seed 43, 8 steps, one generation, no PAG. Active weights 4 GiB, staging
1.5 GiB total (runner --vmm-staging-buffer-gb 0.5), persistent source and
registration 24.625 GiB each. No trim, cuBLAS clear, new tiling, compute tuning
or Linux standby purge. VAE executed route none/resident. Offline cached load.

| Phase | Wall time (s) | Sampled absolute GPU peak (GiB) | Process RSS peak (GiB) |
| --- | ---: | ---: | ---: |
| Prompt encoding | 11.2894 | 6.485229 | 25.1058 |
| Generate including setup | 18.8913 | 6.723999 | 27.3354 |
| VAE load | 0.6703 | 2.613708 | 2.8936 |
| VAE decode | 0.3389 | 2.953552 | 1.6572 |
| VAE cleanup | 0.1486 | 2.953796 | 1.6889 |
| Save image | 0.1981 | 1.838562 | 1.6619 |

Tracked total 31.5387 s; global absolute GPU peak 6.723999 GiB, RSS 27.3354 GiB.
GPU baseline 0.782227 GiB; baseline-relative global delta 5.9418 GiB, not the
absolute peak. Sampling interval 100 ms. No monitor errors. Nested VAE aggregate
1.1588 s is not another phase. Denoise modular call 12.8468 s; completed CUDA
phase 13.124732 s, first CUDA step 5.775473 s, mean steps 2–8 1.049894 s/step.
Eight complete intervals, timing.failed=false. Callbacks are not CUDA timing.

Previous fork-based resident control: total 40.2959 s, denoise wall 17.1203 s,
first CUDA step 10.006696 s, warm mean 1.056121 s/step, absolute GPU 6.559631 GiB,
RSS 27.3297 GiB. The difference is concentrated in encoder/first-step setup;
single runs across changed package/binary/source state do not isolate a
Diffusers-release speedup. GPU baseline also differs; no sampled-memory guarantee
or allocator-only timing attribution. Steady-state is approximately preserved.

Exit 0, plausible 1280x704 PNG inspected and byte-identical to the Linux control:
SHA256 dfbf135a833b5a77f4574a98cebd3be84d19fc569b79de8dd5f18457c52d64c1.
No reported native fault/pool/admission/read failures, temporary fallbacks,
quarantine or monitor errors. Transformer tickets collected 409 to zero.
Pre-unload snapshots still contain source bytes; process success does not
constitute a measured zero-backing allocator teardown assertion.

Artifacts: outputs/ltx_image_official_linux_cgroup_gate_20261012, PNG and
metrics/<run_slug>.json. Raw artifacts remain unchanged. Targeted native Linux
official-package/candidate compatibility PASS for this workload only. No video,
Qwen, FP8/SDNQ or general-model acceptance is inherited; no further matrix run,
commit, default binary promotion or publication occurred.
