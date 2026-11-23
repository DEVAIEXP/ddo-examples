# Runner interface contract

Recipes use the vocabulary from the [shared controls](../user/runner-controls.md).
Public LTX image/video and Qwen entries expose the controls implemented by each route.
The two native image entries' complete
shared DDO vocabulary is in the [native image interface](../user/runner-controls.md#public-native-image-interface).
Supported modular linear, VMM and SDNQ runners expose persistent capacity
profiles through `--ddo-profile` and `--build-ddo-profile`. Repeat and native
allocation experiments use dedicated entries or benchmark families; minimal
runners do not expose profile calibration. Public LTX image entries share the custom-block
loader and unified LoRA interface with public LTX VMM; SDNQ preserves its packed
forward and exposes no unimplemented dynamic encoder switch.

An explicit transformer selector takes precedence over an environment preset.
For Windows purge, explicit positive/negative CLI wins, then the corresponding
`DDO_PURGE_WINDOWS_STANDBY_<PHASE>` variable, then the DDO lifecycle default.
Current source defaults leave purge disabled; positive switches enable selected
phases. An older installed wheel can have different defaults, so record the
effective package identity and observed purge events.
Unsupported phase switches must be rejected, not accepted as no-ops.

Image `--ddo-vmm-total-staging-gb` specifies total reusable host staging in GiB.
The runner converts that capacity to the slots required by the selected route.
Video uses the same total-capacity vocabulary and delegates staging conversion
to the public DDO backend configuration. Public native video uses the installed
allocator bootstrap; isolated DLL selection belongs to diagnostics. Video source
roles and supported subsets are in the [video interface matrix](../user/ltx2-video-parameters.md#interface-scope-by-runner).

Preserve official Diffusers and quantized backend forwards when selecting
compatibility runtime routes. Parser tests must verify supported controls,
unsupported-option errors and unchanged runtime destinations.

## Custom code identity

A Hub revision selects downloaded executable code; it does not switch an existing
local clone. Reports distinguish the requested revision, resolved snapshot and
local clone HEAD. Public LTX video loaders use Hub `main` when the revision is
omitted, and pass an explicit branch, tag or commit through unchanged.
Benchmarks explicitly select immutable code revisions.

## Video decode defaults

Decode defaults depend on the entry. The official Leaf video control enables
GPU temporal streaming; other entries can require explicit
`--decode-device cuda:0 --stream-video-decode`. When streaming is enabled,
`exact-streaming` is the default decoder. Select
`--video-decoder streaming-tiles` explicitly for overlapping temporal windows.
Do not infer these settings from the transformer's precision or offload preset.
The benchmark preview records the effective selection. Refer to the
[video guide](../user/ltx2-video.md) for model-specific controls.
