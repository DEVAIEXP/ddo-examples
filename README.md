# DDO examples

Model-specific Modular Diffusers runners and measured comparisons using
[DDO](https://github.com/DEVAIEXP/diffusers-dynamic-offloader), covering LTX 2.3
image, LTX 2.3 video/audio and Qwen Image 2.1.

DDO manages how model weights are loaded and moved into GPU memory, while
Diffusers and the model's backend perform the calculations. DDO provides two
execution paths:

- **Linear** uses PyTorch to move groups of weights between CPU and GPU memory
  as the model needs them.
- **Native VMM** uses a native component built on CUDA Virtual Memory Management,
  NVIDIA's mechanism for managing GPU memory addresses separately from the
  memory that holds the data. This lets DDO keep each weight at a stable address
  while changing which weights occupy GPU memory. It keeps selected weights
  available and reuses a configured GPU memory budget to load others as needed,
  waiting for ongoing GPU work to finish before reusing that memory.

See the [library overview](https://devaiexp.github.io/diffusers-dynamic-offloader/)
for implementation details and supported configurations.

## Use the examples

Read the [documentation website](https://devaiexp.github.io/ddo-examples/), including
[setup](https://devaiexp.github.io/ddo-examples/docs/user/setup/) and
[benchmarks](https://devaiexp.github.io/ddo-examples/docs/benchmarks/).

Start with the [user guide](docs/user/README.md) to install packages and choose a
model runner. This path does not require a DDO source checkout or native compiler.
Experimental packages `0.1.0rc5` are available on PyPI. Follow the package setup
guide to install DDO, the allocator and runner dependencies in one environment.

Optional LTX video Sol-Attn uses the reviewed source bundled under `kernels/`.
See [optional kernels](docs/user/kernels.md) and [optional setup](docs/user/setup.md#optional-sol-attn-for-ltx-video)
for its dependencies and selected validation scope.

## Benchmarks

See the [benchmark overview](docs/benchmarks/README.md) for measured latency
and memory comparisons across LTX 2.3 image, LTX 2.3 video/audio and Qwen Image
2.1. It brings together summaries, recommendations and links to platform results,
phase measurements and reproducible recipes.

## Develop or contribute

Use the [developer guide](docs/developer/README.md) for editable source setup,
architecture, validation, documentation preview and release procedures.
Generic allocation/runtime policy stay in DDO and its allocator; model math
belongs to independent custom-block repositories.
