# Repository guidance

DDO examples is the primary host for model-specific Modular Diffusers runners.
The sibling diffusers-dynamic-offloader repository owns generic runtime logic.

- Write documentation, comments and docstrings in English.
- Keep scripts organized by model under scripts/. Keep model math in custom blocks.
- Use scripts/helpers/inference_utils.py for all runner telemetry. Do not fork
  RunTracker or the memory-monitor schema inside a model directory.
- Keep parsing and phase orchestration visible in each public runner. Do not
  replace runners with wrappers that only select a mode in a shared dispatcher.
  Minimal runners are complete didactic examples; intentional duplication of
  essential loading, modular calls and DDO lifecycle is preferable to hiding it.
- Share telemetry and small compute adapters under scripts/helpers/.
  Keep LTX-specific adapters under helpers/ltx2; sharing a file does not imply
  validated image support. Video entry points separate minimal, linear,
  VMM and diagnostics; never enable experiments as migration defaults.
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
- Own pipeline benchmarks in experiments/dynamic_offload_results.md and
  reproduction guidance in docs/benchmarks.md. Record exact workload, revisions,
  timing/memory scopes and acceptance boundaries; never rewrite raw evidence.
  Generic runtime/allocator contracts and technical gates remain in DDO.
- Keep weights, outputs, caches and exploratory logs untracked.
- Do not publish this repository without explicit user authorization.
