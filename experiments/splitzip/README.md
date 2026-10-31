# Experimental SplitZip Qwen host

These maintainer sources are separate from the normal runners, installed
packages and public benchmark recipes. They are not an enabled inference
feature or a performance recommendation. Native codec, backing ownership and
budget enforcement remain in the allocator's `experiments/splitzip` sources.

`qwen_exclusive_cycle` contains the current two-block Qwen observer, cold
configuration/snapshots, one-cycle adapter, controller, runner and CPU tests.
It observes a single 32.1875 MiB BF16 storage, proves the first declared read
completed, parks its exclusive backing and restores verified bytes before its
next declared access. It stops after block 1; it does not produce an image or
establish full-model, cross-step, external-consumer or performance acceptance.
The experimental isolation policy is bounded and separate from weight budgets.

`owner_observation` preserves earlier host-side preparation and observation
templates. It is not another supported runner. Its preparation scripts refer to
local accepted fixtures and must not be run as standalone user commands.

The captured sources, manifests, artifacts and evidence are preserved separately
without modifying the original receipts. No machine-specific manifest, model
payload, build artifact or raw result is distributed here. The controller
requires a separately prepared and reviewed sealed candidate, including fixed
paths, package identities, native artifacts, parameters and limits. Copying these
sources does not grant execution approval, and the checked-in controller cannot
run without that manifest. Create a fresh candidate rather than editing an
accepted capture or reusing its seal after source changes.

CPU coverage uses real CPU Torch operations and injected native callbacks:

```text
python -I -B experiments/splitzip/qwen_exclusive_cycle/tests_cpu.py
```

These tests must not initialize CUDA and are not hardware acceptance. Preserve
the normal runner recipe and obtain separate approval for benchmark parameter
changes and CUDA execution. No compression policy or default runner is changed.
