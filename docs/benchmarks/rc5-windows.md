# RC5 validation on Windows

These single runs cover selected RC5 workloads on an RTX 3060 Ti with 8 GiB
of VRAM and 64 GiB of system RAM. Times are runner wall times in seconds.
Memory values are sampled absolute GPU peaks and process RSS in GiB. The
results complement the model comparison tables, which use different recipes.
Native Linux and WSL measurements are pending.

## LTX 2.3 video

These runs used BF16, image and audio conditioning, seed 43, 24 fps, eight
denoise steps, cuDNN video self-attention and exact streaming video decode.
The VMM profile selected approximately 0.63 GiB for transformer residency.

| Resolution and duration | Text encoder (s) | Attach transformer (s) | Denoise (s) | Video decode (s) | Save video (s) | Total (s) | Peak GPU (GiB) | Peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1280×704, 5 s | 12.00 | 6.83 | 152.44 | 11.93 | 1.37 | 202.05 | 5.17 | 33.12 |
| 1280×704, 8 s | 11.65 | 7.19 | 263.51 | 18.68 | 2.02 | 321.21 | 5.24 | 33.19 |
| 1280×704, 10 s | 12.32 | 6.70 | 352.13 | 24.10 | 2.51 | 416.02 | 5.30 | 33.29 |
| 1920×1056, 10 s | 11.43 | 7.10 | 1181.38 | 55.13 | 4.93 | 1278.87 | 8.00 | 34.12 |

The 1920×1056 run also enabled frame modulation and used chunks of 1024 for
feed-forward and normalization. Denoising peaked at 7.03 GiB. During video
decode the allocator released 892 MiB of unused backing and satisfied the
original allocation on retry. The saved file had 241 frames and an audio
stream, and completed a full FFmpeg decode check. The two profile calibration
runs took 476.92 s and 476.22 s. Their timings are separate from inference.

In a separate 1280×704 frame modulation comparison at 8 s, denoise peak went
from 4.61 to 4.10 GiB, denoise time from 265.56 to 262.63 s, and total time
from 323.20 to 319.40 s. The saved videos matched byte for byte. At 5 s,
denoise peak went from 3.56 to 3.32 GiB and total time from 224.87 to 208.73 s.
Source cache pressure differed between the 5 s runs, so the wall time change
cannot be attributed to modulation alone.

## Image profiles and text encoders

| Model and workload | Text encoder route | Denoise (s) | Total (s) | Peak GPU (GiB) | Peak RSS (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| LTX 2.3 image, 1280×704, 8 steps | Native | 13.00 | 38.94 | 7.02 | 27.37 |
| Qwen Image 2.1, 512×512, 20 steps | Native | 18.27 | 31.74 | 3.56 | 18.16 |
| Qwen Image 2.1, 1024×1024, 20 steps | Native | 49.57 | 65.40 | 8.00 | 18.16 |

In paired encoder runs, the LTX image prompt phase took 23.55 s with leaf
offload and 12.58 s with the native encoder. Total time was 50.59 s and
38.99 s, respectively. In the Qwen 512×512 pair, prompt encoding took
13.69 s and 3.18 s, respectively, with totals of 40.45 s and 31.18 s.
The saved image matched within each pair. See
[Qwen image VAE feature measurements](qwen-image-21-features.md) for the
separate decoder comparison.

## Profile preparation for benchmark automation

The planned benchmark suite will check for a compatible profile before each
case that uses automatic VMM residency. Compatibility covers the workload,
model, transport and hardware. When a compatible complete profile is missing,
run both calibration passes and verify the selected profile before starting
the timed inference. Record calibration and inference durations separately.
Fixed-budget cases use their specified budget. Leaf and linear cases follow
their own runtime setup.

The current benchmark script previews individual recipes and supports selected
executions. Automatic calibration and full suite execution are pending.
See [benchmark reproduction](reproduction.md) for the current entry points.
