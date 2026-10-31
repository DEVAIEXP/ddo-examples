# DDO examples

Model-specific Modular Diffusers runners and measured comparisons using
[DDO](https://github.com/DEVAIEXP/diffusers-dynamic-offloader), covering LTX 2.3
image, LTX 2.3 video/audio and Qwen Image 2.1.

DDO manages weight loading and offloading while Diffusers/backend code computes
the model. The **linear** route uses PyTorch-backed offloading; the optional
**native VMM** route uses CUDA Virtual Memory Management to keep pointer
addresses stable while reusing bounded physical GPU backing safely. VMM is not
activation compression or a guarantee of local-VRAM placement. See the
[library overview](https://devaiexp.github.io/diffusers-dynamic-offloader/).

## Use the examples

Read the [documentation website](https://devaiexp.github.io/ddo-examples/), including
[setup](https://devaiexp.github.io/ddo-examples/docs/user/setup/) and
[benchmarks](https://devaiexp.github.io/ddo-examples/docs/benchmarks/).

Start with the [user guide](docs/user/README.md) to install packages and choose a
model runner. This path does not require a DDO source checkout or native compiler.
Experimental packages `0.1.0rc3` are available on PyPI. Follow the package setup
guide to install DDO, the allocator and runner dependencies in one environment.

## Compare performance and memory

Start with this existing **Windows LTX 2.3 video** comparison: BF16, 1280×704,
5 seconds/121 frames at 24 fps, 8 steps, seed 43, RTX 3060 Ti 8 GiB and 64 GiB
system RAM. All three use a shared Diffusers leaf text encoder and resident
temporal VAE. The method is the transformer route. Times are seconds; memory
peaks are sampled absolute driver GPU usage and process RSS in GiB.

Before comparing rows, read the [labels and memory glossary](docs/benchmarks/glossary.md)
for runtime labels, latency scope and peak-memory definitions.

| Runtime method | Denoise (s) | Video VAE decode (s) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diffusers streamed leaf | 392.91 | 22.52 | 452.33 | 4.72 | 41.34 |
| DDO linear (`one_shot_fast`) | 262.22 | 23.47 | 336.44 | 4.72 | 47.87 |
| DDO native VMM (`vmm_fast`) | 207.40 | 24.00 | 277.09 | 5.21 | 32.65 |

Video VAE decode excludes decoder loading, audio decode and vocoder work. The
tracked total already includes these phases; it is not denoise plus decode.

Tracked total excludes command startup and downloads. These single observations
compare configured pipelines, not the isolated effect of the allocator.

| Workload | What to compare | Full tables and conditions |
| --- | --- | --- |
| LTX video | Denoise, decoder latency and memory; native is fastest in this primary five-second set. | [Video benchmarks](docs/benchmarks/ltx23-video.md) |
| LTX image | Linear has shorter Windows denoise; native has shorter tracked total. | [Image benchmarks](docs/benchmarks/ltx23-image.md) |
| Qwen Image 2.1 | Compare resolution, encoder/VAE routes and observation sessions before ranking methods. | [Qwen benchmarks](docs/benchmarks/qwen-image-21.md) |

Each model page separates **phase tables** and finishes each primary platform
set with a **pipeline-total table**. Phases can use different offload methods;
Diffusers leaf VAE may reduce decode GPU peaks at extra latency. No single method
wins every phase or memory metric. Read the [summary](docs/benchmarks/summary.md),
[hardware](docs/benchmarks/hardware.md) and [metric definitions](docs/benchmarks/glossary.md).
Native Linux and WSL results stay in separate groups. These are experimental
prereleases; further model architectures and hardware need validation.

On the website, top menus select pages; the left contents list selects sections
of the current page.

## Develop or contribute

Use the [developer guide](docs/developer/README.md) for editable source setup,
architecture, validation, documentation preview and release procedures.
Generic allocation/runtime policy stay in DDO and its allocator; model math
belongs to independent custom-block repositories.
