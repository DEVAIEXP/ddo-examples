# Test hardware and environments

Use this reference alongside the workload at the top of each model's benchmark
page. Different hardware, drivers, cache state and memory limits can change both
time and peak memory; these results are not performance guarantees.

## Reference machine

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Ti |
| Dedicated GPU memory | 8 GiB (8192 MiB) |
| CPU | AMD Ryzen 5 5600X, 6 cores / 12 logical processors |
| Installed system RAM | 64 GiB |

The GPU and its capacity are recorded in the test records. CPU and installed
RAM were confirmed on the reference Windows host on 2026-10-15; this inspection
does not reconstruct a complete hardware inventory for every earlier run.
Storage model, interface and measured disk throughput were not captured
consistently, so no storage-equivalence claim is made.

## Platform conditions

| Platform | Recorded environment and limits |
| --- | --- |
| Windows | Same reference GPU; full host RAM rather than the WSL memory limit. Selected comparisons record successful standby purge before the transformer. Current host inspection identifies Windows 10 Enterprise, build 19045; not every result recorded its exact OS build. |
| Native Linux | Ubuntu 26.04.1 x86_64 is documented in the native-Linux environment records. Selected protected NVMe observations use NVIDIA driver 595.99.02 and a different reboot/cache session from the earlier leaf/linear/fast-VMM comparisons. |
| WSL2 | Ubuntu 22.04.3, kernel 6.18.33.2-microsoft-standard-WSL2; approximately 39 GiB guest RAM and 32 GiB configured swap. GPU access uses the Windows driver. Guest RAM is not the machine's full 64 GiB, and configured swap is not evidence that a run used it. |

Windows and WSL share the physical GPU, not an identical execution environment.
Native Linux is a separate target. Exact driver versions were not recorded
uniformly across all rows; do not assign one version to the entire comparison.
Primary comparisons use official Diffusers 0.41.0; software and component-route
exceptions are described in each model's measurement details.

## Match your comparison

Record your GPU/VRAM, CPU, installed RAM, OS, driver, storage and any VM/container
memory limits. Then match resolution, precision, steps, seed, conditioning,
encoder/VAE routes and runtime capacities before comparing results.

- [LTX image conditions and results](ltx23-image.md)
- [LTX video conditions and results](ltx23-video.md)
- [Qwen Image 2.1 conditions and results](qwen-image-21.md)
- [Reproduction guide](reproduction.md)

The model pages state resolution and generation settings separately for primary
and supplementary workloads. Compare within the matching group, not across
different resolutions, video durations or quantization formats.
