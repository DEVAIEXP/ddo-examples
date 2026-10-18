# Runner interface compatibility

New recipes use canonical names from the [shared controls](../user/runner-controls.md).
The aliases below are currently supported. Use canonical names in new recipes.

| Existing name | Canonical name |
| --- | --- |
| `--show-ddo-profile` | `--show-profile` |
| `--vmm-resident-weight-budget-gb` | `--ddo-vmm-resident-weight-budget-gb` |
| Linear `--preset` | `--ddo-preset` |
| Image VMM auxiliary `--preset` | `--auxiliary-preset` |

An explicit transformer selector takes precedence over an environment preset.
For Windows purge, explicit positive/negative CLI wins, then the corresponding
`DDO_PURGE_WINDOWS_STANDBY_<PHASE>` variable, then the preset default.
Unsupported phase switches must be rejected, not accepted as no-ops.

Legacy staging arguments retain their units. Image `--vmm-staging-buffer-gb`
is per slot: two slots in protected NVMe and three in the native harness.
The canonical total-staging argument converts at the interface boundary.
Do not combine both spellings in one reproduction recipe.

Compatibility runtime routes are not obsolete CLI aliases: preserve official
Diffusers and quantized backend forwards. Removing an alias requires an explicit
scope, migration of callers and parser tests; it must not remove those runtimes.

## Custom code identity

A Hub revision selects downloaded executable code; it does not switch an existing
local clone. Reports distinguish the requested revision, resolved snapshot and
local clone HEAD. LTX video loaders use the fallback revision
`f5c712e93c52283cb101da26d4a4544c808666bf` when the Hub revision is omitted.
Benchmarks explicitly select immutable code revisions.

## Video decode defaults

Video runners currently default to CPU decode without explicit temporal streaming.
Select `--decode-device cuda:0 --stream-video-decode` for the GPU temporal decoder.
This is independent of transformer precision and offload preset. Refer to the
[video guide](../user/ltx2-video.md) for model-specific controls.
