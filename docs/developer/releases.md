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
- Record the effective native artifact and its hash as well as the Python
  package versions; editable-source acceptance does not validate a clean wheel.
- Keep benchmark workloads, measurement boundaries and tested configurations
  identifiable; link to the model comparison rather than duplicating results.
- Document changes to arguments, defaults or dependencies in the release notes.
- Preserve bundled kernel licenses, attribution and the source manifest.
  Optional attention dependencies remain separate from the ordinary install.

`constraints.txt` constrains dependency versions; it is not a hash-verified
environment lock. Retain the actual installed environment used for acceptance.
Keep Windows, native Linux and WSL validation separate. Experimental changes
ship as prereleases before stable release.

Local validation does not authorize publication. Tags, push, registry or Hub
uploads and CI publication require explicit authorization. Commit independent
custom-block repositories, DDO/allocator changes and this host separately;
do not include weights, outputs, environments or local internal records.

Current package availability belongs in the project README and release notes.
Native package build and distribution follow the
[DDO release procedure](https://github.com/DEVAIEXP/diffusers-dynamic-offloader/blob/main/docs/developer/releases.md).
