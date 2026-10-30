# DDO examples

Model-specific Modular Diffusers runners and measured comparisons using
[DDO](https://github.com/DEVAIEXP/diffusers-dynamic-offloader), covering LTX 2.3
image, LTX 2.3 video/audio and Qwen Image 2.1.

## Use the examples

Read the [documentation website](https://devaiexp.github.io/ddo-examples/), including
[setup](https://devaiexp.github.io/ddo-examples/docs/user/setup/) and
[benchmarks](https://devaiexp.github.io/ddo-examples/docs/benchmarks/).

Start with the [user guide](docs/user/README.md) to install packages and choose a
model runner. This path does not require a DDO source checkout or native compiler.
Experimental packages `0.1.0rc3` are available on PyPI. Follow the package setup
guide to install DDO, the allocator and runner dependencies in one environment.

## Compare performance and memory

See [benchmarks](docs/benchmarks/README.md) for model-specific results, memory
tradeoffs and recommendations. Conditions and platform boundaries accompany each
comparison; no arbitrary-model or all-preset speedup is promised.

## Develop or contribute

Use the [developer guide](docs/developer/README.md) for editable source setup,
architecture, validation, documentation preview and release procedures.
Generic allocation/runtime policy stay in DDO and its allocator; model math
belongs to independent custom-block repositories.
