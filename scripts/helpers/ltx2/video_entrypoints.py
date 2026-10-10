"""Limit each video CLI to its responsibility without duplicating phase logic."""

DIAGNOSTICS = {
    "compile_repeated_blocks", "compile_eager_rmsnorm", "compile_eager_torch_rmsnorm",
    "external_qk_rope_library", "independent_qk_rope", "compile_triton_execution",
    "compile_fp8_linears", "fp8_feed_forward_reuse", "fp8_feed_forward_temporary_mb",
    "fp8_group_offload_control", "fp8_group_offload_stream", "diagnose_denoise",
    "diagnose_operators", "diagnose_operator_module", "inspect_host_memory_phases",
    "detach_transformer_auxiliary_cpu_sources",
}
NATIVE = DIAGNOSTICS | {
    "transformer_fp8_weight_only", "text_encoder_native_vmm",
    "text_encoder_vmm_resident_weight_budget_gb", "feed_forward_chunk_size",
    "normalization_chunk_size", "phase_modulation", "compact_timesteps",
    "inplace_residuals", "fused_rms_modulation", "fused_rotary_pair",
    "release_pytorch_host_cache_before_transformer", "ddo_vmm_library_path",
}
MINIMAL = {
    "help", "dry_run", "local_files_only", "distilled_model", "custom_blocks",
    "custom_blocks_revision", "prompt", "image", "audio", "use_audio_condition",
    "width", "height", "seconds", "frame_rate", "steps", "seed", "output_dir",
    "show_profile", "show_metrics", "save_metrics", "ddo_preset",
}


def configure_parser(parser, entrypoint):
    """Keep defaults for hidden internal options, but reject them at the CLI."""
    if entrypoint not in {"legacy", "diagnostics", "vmm", "linear", "minimal"}:
        raise ValueError(f"Unknown video entrypoint: {entrypoint}")
    if entrypoint in {"legacy", "diagnostics"}:
        return
    for action in list(parser._actions):
        hidden = (
            action.dest not in MINIMAL if entrypoint == "minimal" else
            action.dest in DIAGNOSTICS if entrypoint == "vmm" else
            action.dest in NATIVE or action.dest.startswith("ddo_vmm_")
        )
        if hidden:
            parser.set_defaults(**{action.dest: action.default})
            parser._remove_action(action)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
            for option in action.option_strings:
                parser._option_string_actions.pop(option, None)
    backend = "vmm" if entrypoint == "vmm" else "linear"
    parser.set_defaults(ddo_backend=backend)
    for action in parser._actions:
        if action.dest == "ddo_backend":
            action.choices = (backend,)
            action.help = f"This entry point uses the {backend} DDO backend."
        if action.dest == "ddo_preset":
            action.choices = ("auto", "nvme_baseline") if backend == "vmm" else (
                "auto", "one_shot_fast", "low_ram_safe", "wsl_compat", "warm_process",
                "diffusers_offload_compat", "diffusers_leaf_offload_compat", "off",
            )
