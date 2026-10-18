# Benchmark labels and memory glossary

Use this page before comparing tables. Units and measurement scope matter as
much as the values. The model page fixes workload, platform and executed routes.

## Table labels

| Label | Meaning |
| --- | --- |
| Platform | Operating environment: Windows, native Linux or WSL. |
| Transformer runtime | Executed denoising implementation: linear_runtime, vmm_fast_runtime, protected vmm_runtime, sdnq_runtime or the official control. |
| DDO selector | Requested backend/preset and its resolution; auto is backend-dependent. See [runtime mapping](runtimes.md). |
| VAE route | Decoder placement/strategy, independently from the transformer: resident or official leaf, for example. Resident keeps decoder weights on the GPU; leaf uploads offloaded modules as needed. |
| Platform / VAE | Compact label used in decode-only tables, with transformer identity inherited from the parent workload table. |
| Route | Executed transformer/offload implementation, such as DDO linear, native VMM or official Diffusers leaf. It is not necessarily the VAE strategy. |
| Denoise wall (s) | Elapsed host-clock time for the named denoising region. Includes compute and waits inside that region, not just GPU kernels. Some runners include latent CPU readback; each table identifies its scope. |
| Wall average / phase average (s/step) | Denoise wall time divided by executed steps. Includes first-step setup/warmup and any readback in the region. It is not the speed of a typical warmed-up step. |
| CUDA phase (s) | Completed GPU-event duration on the observed compute stream, including stream dependencies. It is not summed pure-kernel time or automatically identical to wall time. |
| CUDA warm (s/step) | Mean of completed CUDA step intervals after the first step; selected eight-step image cases use steps 2–8. Available only when those event intervals were recorded successfully. |
| Tracked total (s) | Total of the runner's tracked pipeline interval, including its component loading/setup, encoding, denoise, cleanup, decode/save and recorded purge where executed. Not complete command startup: imports, bootstrap and first downloads are outside the stated scope. |
| Peak GPU (absolute, GiB) | Highest sampled driver-reported used GPU memory during the stated interval, without subtracting the pre-run baseline. Main comparison rows normally show the global tracked-run peak. |
| Peak RAM (RSS, GiB) | Highest sampled resident memory of the Python process during the stated interval. Main comparison rows normally show the global tracked-run peak. |
| Decode peak GPU / RAM | Same memory definitions, but sampled only during decode. These are phase peaks, not the global peaks. |
| Source registration (MiB) | Configured ceiling for registering retained source bytes with CUDA. Separate from committed source storage, staging registrations, active GPU weights and observed memory peaks. |
| N/A | Not measured, unavailable or not applicable as explained by the table. Never zero or an implicit successful test. |

## RSS is a metric; peak is a statistic

RSS means **Resident Set Size**: the process memory currently resident in physical
system RAM as reported by the operating system. It includes resident shared pages;
it is not necessarily memory unique to the process.

RSS does not inherently mean peak. **Peak RAM (RSS)** means the largest RSS sample
observed during the run or named phase. A value of 27 GiB therefore means the
process reached approximately that resident footprint in the samples, not that
the entire computer used 27 GiB or that every machine with 27 GiB RAM will fit.

RSS is not total system RAM usage, virtual address reservation, committed bytes,
swap/pagefile usage, free RAM, standby cache or a guaranteed minimum RAM requirement.
Those are different observations. Other processes and operating-system headroom
also need memory.

**USS** (Unique Set Size) counts memory private to the process; it is not RSS.
The legacy full monitor uses USS on Windows and anonymous RSS with a USS fallback
on Linux. Reports identify the actual RAM metric. Never relabel those older
measurements as RSS or compare them as identical quantities.

## Peak RAM and Peak VRAM in console and JSON

The shared helper currently prints some historical console units as `GB`, but
converts bytes by `1024**3`: these measurements are **GiB**. One GiB is
1,073,741,824 bytes; one decimal GB is 1,000,000,000 bytes.

Console **Peak RAM** uses the selected monitor's RAM metric. With
`--memory-monitor rss`, it is the sampled RSS peak described above.

The shared RunTracker console **Peak VRAM** is generally a **baseline-relative
GPU delta**, not the absolute usage shown in these comparison tables:
`sampled absolute driver usage - pre-run baseline`.

| Recorded field | Interpretation |
| --- | --- |
| `global_peak_vram_absolute_gb` | Global sampled absolute driver GPU usage, in GiB |
| `global_peak_vram_gb` | Global sampled GPU usage minus the recorded baseline, in GiB |
| `vram_baseline_gb` | Driver GPU usage recorded before the tracked phases |
| Phase `peak_vram_absolute_gb` | Absolute driver GPU peak for that phase |
| Phase `peak_vram_gb` | Baseline-relative GPU delta for that phase |
| `global_peak_ram_gb` / phase `peak_ram_gb` | Sampled RAM peak; inspect `ram_metric` to identify RSS, USS or another recorded metric |
| `memory_monitor.interval_sec` | Requested sampling interval, commonly 0.1 s in selected cases |

For example, one accepted Linux image report recorded 6.72 GiB absolute GPU
peak and a 0.78 GiB baseline. Its console/delta peak is approximately 5.94 GiB.
These describe the same observation with different baselines, not contradictory
memory usage or an extra 0.78 GiB allocated by the model.

Driver usage is obtained through `torch.cuda.mem_get_info` (total minus free).
It is not process-isolated PyTorch tensor memory: desktop/other processes and
driver/platform accounting can influence it. PyTorch `memory_allocated`,
`memory_reserved`, native virtual arena size and native pool/cache counters
are distinct. The pluggable allocator may not provide PyTorch device statistics.

The Windows Task Manager's dedicated/shared GPU and WDDM counters are different
measurements. These benchmark columns do not separately measure WDDM shared
memory, prove that every virtual weight is resident, or identify who owns every byte.

## Text encoder and phase boundaries

A phase label describes the measured interval, not just a model forward.

| Runner | Text encoder timing shown in the phase table |
| --- | --- |
| LTX image | Monitored prompt phase: encoder/tokenizer loading, encoding and cleanup, including purge where it occurs inside that phase. Connectors are in the separate generate aggregate. |
| LTX video | Staged text-encoder phase, including its loading/encoding/cleanup. Connectors and conditioning encoders are separate. |
| Qwen | Separate encoder load, attachment, encode and cleanup rows. The encode row excludes its neighboring setup/cleanup phases. |

Do not compare these as identical pure-forward timers. Phase memory values are
sampled within the named interval; they cannot be assigned to a smaller event
inside it. In particular, image generate memory is not isolated denoise memory.
Model detail pages identify the source boundaries and any decode aggregates.

## Global and phase peaks

The global peak can occur during encoding, denoise or decode. A leaf VAE may
reduce decode memory without reducing the global peak, if encoding was larger.
Do not add encoder, denoise and VAE peaks: they are maxima at different times.
Likewise, do not add nested aggregate timings to their constituent phases.

Peaks are sampled, not exact instantaneous hardware maxima. Short spikes may be
missed; scheduling/query overhead can alter the effective interval. A successful
sampled peak below capacity is not a guarantee that another workload will fit.
A monitor error or missing historical metric cannot count as zero usage.

Use [recording rules](recording.md), [reproduction](reproduction.md) and the model's
workload/route notes. Compare like timing, memory metric, interval and scope;
callback intervals are not substitutes for synchronized CUDA warm timing.
