# DDO examples repository guidance
## Purpose and repository boundaries
This is the primary model-specific Modular Diffusers runner/benchmark host.
Read docs/README.md, user/setup.md, developer/architecture.md, validation.md and
benchmarks/README.md before changing execution or benchmark conclusions.
Sibling DDO owns generic settings, bootstrap, runtime lifecycle, transport,
profiles and public metrics; its pinned allocator owns native C++/CUDA policy.
Independent custom-block clones under modular_diffusers/custom_blocks are ignored,
not submodules or vendored source. Read each clone's AGENTS.md before editing it.
Historical LTX hosts and the reference Diffusers fork are optional, not import roots.
## Language and evidence
Write all new/updated documentation, comments and explanatory docstrings in English.
Conversation may use the user's language. Preserve verbatim logs, prompts, quotes
and required license/attribution text; never alter evidence to translate or sanitize it.
Use docs/user for current integration, docs/developer for contributor contracts,
and ignored .internal/documentation for handoffs/plans/intermediate investigations.
README must index public docs. Never link public instructions to required ignored files.
Keep a separate backup of local internal records; .gitignore is not a backup and
does not remove previously tracked content from Git history. No implicit history rewrite.
## Runner and helper rules
Keep scripts/<model> entry points explicit and didactic. Minimal runners contain
the essential parsing/loading/modular orchestration instead of delegating to a
hidden mode dispatcher. Separate minimal, linear, VMM, SDNQ/controls and diagnostics.
Shared controls must use the canonical vocabulary in docs/user/runner-controls.md
and scripts/helpers/runner_controls.py; preserve legacy aliases and reject
unsupported controls rather than accepting no-op arguments.
Never enable diagnostic/compile/kernel flags as migration defaults.
Use scripts/helpers/inference_utils.py for one shared RunTracker/memory schema;
no forked model-specific metrics standards. Share only genuinely generic helpers.
LTX helpers remain model-specific; sharing code does not prove image/video parity.
Keep math/checkpoint layout in independent custom blocks, not generic DDO.
Use native Diffusers/backend forwards and authoritative execution devices, even
when weights remain on CPU. Preserve prompt/connector intermediate cleanup and
avoid duplicate weight/embedding retention. Stage phases when memory requires it.
Native bootstrap precedes CUDA-owning imports. Hosts consume public DDO reports,
never allocator/planner internals. Generic source lifecycle/pressure stays native.
## Environment and model controls
Use official version-constrained Diffusers, not fork imports/PYTHONPATH. Root Windows .venv,
native Linux .venv-linux and separate WSL environment must never replace each other.
Torch remains CUDA; Linux/WSL CPU TorchAudio index is audio decoding only.
FFmpeg system shared libraries and native wheel/build are separate prerequisites.
Optional fp8/kernels/sdnq extras do not become mandatory ordinary dependencies.
Default dependency metadata must not select editable sibling checkouts. Users
install wheel packages and model dependencies into the same environment using
the package setup guide. Source development selects editable DDO/allocator
installs explicitly. Keep third-party versions in constraints.txt; do not restore
the old source-based uv.lock as a user installation default.
Setup custom-block clones only at reviewed revisions; refuse to overwrite local
edits/unrelated checkouts. trust_remote_code executes code, not inert metadata.
Preserve selected checkpoint, input/prompt, seed, shape, precision, steps, schedules,
audio scope, budgets and compute options during comparisons. A Turbo saved schedule
must not be replaced by base defaults. SDNQ keeps its own compatible forward route.
## Benchmarks and acceptance
Public model comparisons live in docs/benchmarks: measured metrics, comparison
conditions, recommendations and runnable usage instructions. Do not publish a
separate evidence registry or references to ignored outputs, local report paths
or private artifact hashes. Preserve original reports and full investigations
under .internal with a separate backup. Never rewrite original evidence.
Keep each model's main benchmark page focused on workload summaries, tables and
recommendations, grouped by platform. Link scenario labels in tables directly to
a separate model measurement-details page. Include recorded text-encoder phases
without relabeling aggregate phases as pure forwards; do not rerun closed gates
just to fill a presentation table.
Record per-phase wall, denoise/readback scope, warm CUDA timing only if completed,
RSS/USS identity, absolute versus baseline-relative GPU, actual route, purge state,
code/environment/binary/model identity and output acceptance.
Benchmark rows must identify executed transformer runtime, requested backend/preset
and VAE route separately. Include protected vmm/nvme_baseline → vmm_runtime in
coverage, without fabricating matched numbers. Resident/leaf alone is not a runtime.
Before benchmark tables, explain column labels and link docs/benchmarks/glossary.md.
Memory headers must state peak, unit, metric (RSS/USS) and absolute/delta scope;
never label all legacy Peak RAM fields RSS or console Peak VRAM absolute.
N/A is not zero. Single observations are not a statistical/all-model speed guarantee.
Display benchmark measurements with at most two decimal places. Preserve exact
raw artifacts, identifiers and executable reproduction parameters; rounding is
presentation only and must not alter recorded conditions or future test controls.
Do not sum nested aggregate timings. WSL registration limits and conditioned audio
mean runs may not match native-Linux/Windows controls. Windows purge is no-op elsewhere.
Selected platform/model gates are closed: do not reopen a Cartesian matrix.
## Tests and prereleases
CPU tests use the root plus modular_diffusers imports and do not initialize CUDA.
Native gates belong to DDO/allocator, not fake passes in this host.
Experimental prereleases solicit model feedback, not stable universal support.
CI/package publication is distinct from local runner development.
Commit independent clones, DDO pin and examples separately when explicitly authorized.
Keep outputs, model weights/media, caches, environments and internal records ignored.
## Execution, validation and release authority
Read this file and relevant public docs before acting. Inspect git status and preserve
user changes. Use apply_patch for manual edits; scope searches/edits to the task.
Before CUDA, benchmarks, benchmark parameter changes, compatibility removal or
history cleanup, show exact command, hypothesis, fixed limits and criteria and
wait for user approval. CPU tests/builds do not authorize GPU execution.
Stop at the first failure/anomaly. No silent retries, budget growth, driver resets,
OS memory-policy changes, binary promotion or full Cartesian test matrix.
Never claim general support from selected acceptance. Treat missing fields as
unavailable, not zero; preserve raw result identity and rejected experiments.
Experimental prereleases precede stable release. Tags, push, Hub/registry upload,
CI publication and Git-history rewrites require explicit permission.
Do not remove required licenses/attribution or suggest that hiding names establishes
independent implementation/license compatibility. Do not commit user scratch files,
weights, environments, build products, outputs or internal records.
