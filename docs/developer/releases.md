# Release procedure

## Repository responsibilities

This repository delivers example runners, model integration and benchmark
documentation. DDO and its allocator deliver their Python/native packages.
Custom-block code is maintained in independent model repositories.

## Publication checklist

- Verify installation instructions in a clean package environment.
- Check the supported DDO/allocator package pair and native platform artifact.
- Verify documentation links and the commands shown in the model guides.
- Record the environment lock, custom-block revisions and model revisions used
  for validation.
- Keep benchmark workloads, measurement boundaries and tested configurations
  identifiable; link to the model comparison rather than duplicating results.
- Document changes to arguments, defaults or dependencies in the release notes.

Current package availability belongs in the project README and release notes.
Native package build and distribution follow the
[DDO release procedure](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/releases.md).
