# Repository guidance

DDO examples is the primary host for model-specific Modular Diffusers runners.
The sibling diffusers-dynamic-offloader repository owns generic runtime logic.

- Write documentation, comments and docstrings in English.
- Keep scripts organized by model under scripts/. Keep model math in custom blocks.
- Preserve accepted runner behavior and settings during migration.
- Independent custom-block clones are ignored under modular_diffusers/custom_blocks/.
  They are never submodules or vendored source.
- Use the root uv-managed .venv. Historical host repositories are optional.
- Install DDO and its allocator; never import them from historical host directories.
- Native bootstrap must precede CUDA initialization.
- Before CUDA tests, benchmarks or parameter changes, present the command,
  hypothesis and acceptance criterion and wait for user confirmation.
- CPU checks are not generation or performance acceptance.
- Preserve compatibility controls and historical evidence.
- Keep weights, outputs, caches and exploratory logs untracked.
- Do not publish this repository without explicit user authorization.
