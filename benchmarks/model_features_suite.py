"""Cumulative model features with one calibrated VMM Fast budget per series."""

import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess


CONTROL = "vmm-fast-control"
VIDEO = "ltx23-video-8s"
QWEN = "qwen-image21"
VIDEO_FEATURES = (
    ("rope-reuse", "reuse_rotary_embeddings"),
    ("compact-timesteps", "compact_timesteps"),
    ("inplace-residuals", "inplace_residuals"),
    ("fused-rms-modulation", "fused_rms_modulation"),
    ("fused-rotary-pair", "fused_rotary_pair"),
    ("phase-modulation", "phase_modulation"),
    ("frame-modulation", "frame_modulation"),
)
QWEN_FEATURES = (
    ("single-frame", "vae_no_temporal_cache", True),
    ("residual-alias", "vae_alias_residuals", True),
    ("direct-padding", "vae_direct_padding", True),
    ("bands-256", "vae_band_rows", 256),
)
QWEN_BRANCHES = (
    ("bands-128", "direct-padding", {"vae_band_rows": 128}),
    ("bands-256-no-direct-padding", "residual-alias", {"vae_band_rows": 256}),
    ("bands-128-no-direct-padding", "residual-alias", {"vae_band_rows": 128}),
    ("tiling-only", CONTROL, {"vae_tiling": True}),
)
VIDEO_CHUNKS = (("feed-forward-1024", "feed_forward_chunk_size", 1024),
                ("normalization-1024", "normalization_chunk_size", 1024))


def case_names(recipe):
    if recipe not in (VIDEO, QWEN):
        raise ValueError("No reviewed model-feature series is available for LTX Image.")
    additions = (*VIDEO_FEATURES, *VIDEO_CHUNKS) if recipe == VIDEO else QWEN_FEATURES
    return (CONTROL, *(item[0] for item in additions),
            *(() if recipe == VIDEO else (item[0] for item in QWEN_BRANCHES)))


def set_option(command, name, value):
    """Replace both boolean spellings instead of depending on argument order."""
    flag = "--" + name.replace("_", "-")
    result = list(command)
    for spelling in (flag, "--no-" + flag[2:]):
        if spelling in result:
            index = result.index(spelling)
            del result[index:index + (1 if type(value) is bool else 2)]
    if type(value) is bool:
        if value:
            result.append(flag)
        elif name in ("reuse_rotary_embeddings", "vae_no_temporal_cache"):
            result.append("--no-" + flag[2:])
    else:
        result.extend((flag, str(value)))
    return result


def build_plan(suite, video, args):
    if args.platform not in ("windows", "linux"):
        raise ValueError("Use Windows or native Linux for model features.")
    names = case_names(args.recipe)
    if args.case and args.case not in names:
        raise ValueError("The selected case belongs to a different model-feature series.")
    base_args = argparse.Namespace(**{**vars(args), "family": "runtime", "case": "vmm-fast",
                                     "attention_runtime": None, "profile_file": None,
                                     "expandable_segments": False, "nvme_resident_gib": 1.5,
                                     "no_purge": False, "purge_before_transformer_only": False,
                                     "component_reference_manifest": None})
    if args.recipe == VIDEO:
        base_args.custom_blocks = args.custom_blocks or suite.ROOT / "modular_diffusers/custom_blocks/ltx2_video"
        base_args.fixtures_dir = args.fixtures_dir or suite.ROOT / "benchmarks/inputs/ltx23-video-5s"
        plan = video.video_runtime_plan(base_args)
        for stage in plan["cases"]["vmm-fast"]["stages"]:
            stage["command"][3] = str(suite.ROOT / "scripts/ltx2_video/run_model_features_distilled.py")
        defaults = {field: False for _, field in VIDEO_FEATURES}
        defaults.update(feed_forward_chunk_size=0, normalization_chunk_size=0)
        variants = [(name, field, True) for name, field in VIDEO_FEATURES] + list(VIDEO_CHUNKS)
    else:
        # A preview placeholder never authorizes loading a different VAE source.
        base_args.vae_model_path = args.vae_model_path or Path("elismasilva/qwen-image-2.1-bf16-vae")
        plan = suite.build_plan(base_args)
        defaults = {"vae_no_temporal_cache": False, "vae_alias_residuals": False,
                    "vae_direct_padding": False, "vae_band_rows": None, "vae_tiling": False}
        variants = (*QWEN_FEATURES, *((name, None, None) for name, _, _ in QWEN_BRANCHES))
    base = plan["cases"]["vmm-fast"]
    profile = base["profile"] + "-model-features"
    for stage in base["stages"]:
        command = stage["command"]
        command[command.index("--ddo-profile") + 1] = profile
    base.update(profile=profile, hold=None)
    cumulative = defaults.copy()
    cases = {}
    previous = None
    for name, field, value in [(CONTROL, None, None), *variants]:
        parent = previous
        branch = next((item for item in QWEN_BRANCHES if args.recipe == QWEN and item[0] == name), None)
        if branch:
            _, parent, overrides = branch
            cumulative = {**cases[parent]["model_features"], **overrides}
            field = next(iter(overrides))
        elif field:
            cumulative[field] = value
        case = deepcopy(base)
        case.update(model_features=cumulative.copy(), added_feature=field,
                    previous_case=parent, feature_control_case=CONTROL)
        if branch:
            case["feature_branch"] = True
        if name == CONTROL and (args.recipe == QWEN or args.platform == "linux"):
            case["feature_control_exact_budget"] = True
        if previous:
            if args.recipe == VIDEO:
                case["profile"] = profile + "-" + name
                case["feature_fixed_budget_case"] = CONTROL
                for stage in case["stages"]:
                    stage["command"] = set_option(stage["command"], "ddo_profile", case["profile"])
            else:
                case["stages"] = [case["stages"][-1]]
                case["feature_profile_case"] = CONTROL
                case["feature_fixed_budget_case"] = CONTROL
        for stage in case["stages"]:
            for feature, selected in cumulative.items():
                if selected is not None:
                    stage["command"] = set_option(stage["command"], feature, selected)
        cases[name] = case
        previous = name
        if args.case == name:
            break
    plan.update(family="model-features", cases=cases,
                feature_comparison="cumulative additions or explicit branch parent; unchanged-budget SDPA control",
                profile_policy=("calibrate each video case; measure at the control budget" if args.recipe == VIDEO
                                else "calibrate the control only; reuse its final profile and budget"))
    return plan


def control_profile(state, name):
    """Require an intact, completed control before sharing its final recommendation."""
    if __package__:
        from . import image_suite as suite
    else:
        import image_suite as suite
    record = state["cases"][name]
    receipt = record.get("profile_receipt", {})
    if (record.get("status") != "completed" or not suite.receipts_valid(record.get("artifacts", {}))
            or len(receipt) != 1 or not suite.receipts_valid(receipt)):
        raise ValueError("The feature control or its profile is incomplete or changed.")
    saved = suite.read_json(Path(next(iter(receipt))))
    recommendation = saved.get("recommendation", {})
    budget = recommendation.get("vmm_resident_weight_budget_bytes")
    if (recommendation.get("vmm_build_stage") != "final" or type(budget) is not int or budget <= 0
            or saved.get("context", {}).get("name") != state["identity"]["plan"]["cases"][name]["profile"]):
        raise ValueError("The feature control requires a matching final residency profile.")
    report_path = next(Path(p) for p in record["artifacts"] if Path(p).suffix == ".json")
    applied = suite.read_json(report_path).get("ddo_profile") or {}
    exact_control = state["identity"]["plan"]["cases"][name].get("feature_control_exact_budget", False)
    if (applied.get("status") != ("explicit" if exact_control else "applied")
            or applied.get("shape_check") != "match"
            or (not exact_control and
                applied.get("recommendation", {}).get("vmm_resident_weight_budget_bytes") != budget)):
        raise ValueError("The control did not apply the final profile with matching shapes and budget.")
    if exact_control:
        verify_fixed_budget({"profile": saved["context"]["name"], "feature_fixed_budget_bytes": budget},
                            suite.read_json(report_path))
    return {"profile_receipt": receipt, "resident_budget_bytes": budget}


def fixed_budget_measurement(state, case, record, command):
    """Use an intact case-specific shape profile and the control's exact budget."""
    if __package__:
        from . import image_suite as suite
    else:
        import image_suite as suite
    receipt = record.get("profile_receipt", {})
    shared = None
    if case.get("feature_profile_case"):
        shared = control_profile(state, case["feature_profile_case"])
        if receipt and receipt != shared["profile_receipt"]:
            raise ValueError("The shared feature profile changed.")
        receipt = shared["profile_receipt"]
    if len(receipt) != 1 or not suite.receipts_valid(receipt):
        raise ValueError("A fixed-budget feature measurement requires its own intact profile.")
    saved = suite.read_json(Path(next(iter(receipt))))
    if (saved.get("context", {}).get("name") != case["profile"]
            or saved.get("recommendation", {}).get("vmm_build_stage") != "final"):
        raise ValueError("The feature shape profile must be final and belong to this case.")
    if case.get("feature_control_exact_budget"):
        budget = saved["recommendation"].get("vmm_resident_weight_budget_bytes")
        if type(budget) is not int or budget <= 0:
            raise ValueError("The feature control requires a positive exact byte budget.")
    else:
        budget = (shared or control_profile(state, case["feature_fixed_budget_case"]))["resident_budget_bytes"]
    effective = {**case, "feature_fixed_budget_bytes": budget}
    return effective, set_option(command, "ddo_vmm_resident_weight_budget_gb", budget / 2**30)


def verify_fixed_budget(case, report):
    profile = report.get("ddo_profile", {})
    budget = case["feature_fixed_budget_bytes"] / 2**30
    config = report.get("dynamic_offload_reports", {}).get("transformer", {}).get("configuration", {})
    if (profile.get("status") != "explicit" or profile.get("name") != case["profile"]
            or profile.get("shape_check") != "match" or profile.get("resident_budget_gb") != budget
            or config.get("dynamic_offload_vmm_resident_weight_budget_gb") != budget):
        raise ValueError("Feature measurement did not preserve its own shapes and the control budget.")


def verify_features(recipe, case, report, *, calibration=False):
    expected = case["model_features"]
    if recipe == QWEN:
        if calibration:
            return  # Calibration stops before the VAE is loaded.
        actual = report.get("executed_controls", {})
        if any(actual.get(k, "unavailable") != v or type(actual.get(k)) is not type(v)
               for k, v in expected.items()):
            raise ValueError("The Qwen VAE did not apply the cumulative features.")
        if (actual.get("vae_class") != ("QwenImage21ImageVAE" if expected["vae_no_temporal_cache"]
                                        else "AutoencoderKLQwenImage21")
                or actual.get("vae_band_blocks") != (2 if expected["vae_band_rows"] is not None else None)):
            raise ValueError("The Qwen decoder class or band count differs from the plan.")
        return
    controls = report.get("requested_controls", {})
    if controls.get("benchmark_feature_entrypoint") is not True:
        raise ValueError("The video did not use the explicit model-feature entry.")
    if any(type(controls.get(key)) is not type(value) or controls.get(key) != value
           for key, value in expected.items()):
        raise ValueError("The video feature requests differ from the cumulative plan.")
    applied = {"reuse_rotary_embeddings": "rotary_reuse_prepared",
               "compact_timesteps": "compact_timesteps_executed",
               "inplace_residuals": "inplace_residuals_executed",
               "phase_modulation": "phase_modulation_configuration_applied",
               "frame_modulation": "frame_modulation_executed"}
    for key, field in applied.items():
        if expected[key] and report.get(field) is not True:
            raise ValueError(f"No application evidence for {key}.")
        if not expected[key] and report.get(field) is True:
            raise ValueError(f"Unexpected feature application: {key}.")
    for _, field, _ in VIDEO_CHUNKS:
        if report.get(field + "_configuration_applied", 0) != expected[field]:
            raise ValueError(f"No matching chunk configuration evidence for {field}.")
    for key, container, field in (
        ("fused_rms_modulation", "execution_ops_submissions", "rms_modulate_submissions"),
        ("fused_rotary_pair", "video_self_attention", "fused_rotary_pair_submissions"),
    ):
        value = report.get(container, {}).get(field)
        if expected[key] and (type(value) is not int or value <= 0):
            raise ValueError(f"No observed submissions for {key}.")
        if not expected[key] and value not in (None, 0):
            raise ValueError(f"Unexpected fused submissions for {key}.")


def feature_summary(state, name, artifacts):
    if __package__:
        from . import image_suite as suite
    else:
        import image_suite as suite
    case = state["identity"]["plan"]["cases"][name]
    summary = {"enabled_features": case["model_features"], "added_feature": case["added_feature"],
               "comparison_scope": "single observation; cumulative feature interaction is included"}
    current = suite.measurement_summary(artifacts)
    for label, target in (("parent" if case.get("feature_branch") else "previous", case["previous_case"]),
                          ("control", case["feature_control_case"])):
        if not target or target == name:
            continue
        reference = state["cases"][target]
        if reference.get("status") != "completed" or not suite.receipts_valid(reference.get("artifacts", {})):
            raise ValueError("A cumulative feature reference is incomplete or changed.")
        old = reference["summary"]
        keys = ("total_sec", "peak_gpu_absolute_gib", "peak_ram_rss_gib")
        delta = {key: current[key] - old[key] for key in keys}
        before = {row["name"]: row["elapsed_sec"] for row in old["phases"]}
        delta["phase_latency_sec"] = {row["name"]: row["elapsed_sec"] - before[row["name"]]
                                      for row in current["phases"] if row["name"] in before}
        if any(Path(p).suffix == ".png" for p in artifacts):
            delta["output"] = suite.compare_pngs(reference["artifacts"], artifacts)
        else:
            current_hashes = [sha for p, sha in artifacts.items() if Path(p).suffix.lower() == ".mp4"]
            reference_hashes = [sha for p, sha in reference["artifacts"].items()
                                if Path(p).suffix.lower() == ".mp4"]
            if len(current_hashes) != 1 or len(reference_hashes) != 1:
                raise ValueError("Video hash comparison requires exactly one MP4 per measurement.")
            delta["output"] = {"mp4_sha256": current_hashes[0],
                               "reference_mp4_sha256": reference_hashes[0],
                               "mp4_sha256_identical": reference_hashes[0] == current_hashes[0],
                               "visual_acceptance": "pending; container hashes do not establish pixel quality"}
        summary[label] = {"case": target, "delta": delta}
    return {"model_features": summary}


def main(suite, video, run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows", "linux"), default="windows")
    parser.add_argument("--recipe", choices=(VIDEO, QWEN, "ltx23-image"), required=True)
    parser.add_argument("--family", choices=("model-features",), required=True)
    parser.add_argument("--case", help="Stop after this cumulative case, including its prerequisites.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--custom-blocks", type=Path)
    parser.add_argument("--vae-model-path", type=Path)
    parser.add_argument("--fixtures-dir", type=Path)
    parser.add_argument("--ffmpeg-bin", type=Path)
    parser.add_argument("--output-root", type=Path, default=suite.OUTPUT)
    parser.add_argument("--vram-gib", type=int, choices=(8,), default=8)
    parser.add_argument("--expected-dll-sha256")
    parser.add_argument("--expected-library-sha256")
    parser.add_argument("--session-action", choices=("continue", "restart"))
    parser.add_argument("--list-cases", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.recipe == QWEN and args.vae_model_path is None:
        args.vae_model_path = Path("elismasilva/qwen-image-2.1-bf16-vae")
    try:
        names = case_names(args.recipe)
        if args.list_cases:
            if args.execute or args.preflight:
                parser.error("Case listing cannot execute or preflight a series.")
            print(json.dumps(names))
            return
        plan = build_plan(suite, video, args)
    except ValueError as error:
        parser.error(str(error))
    if not args.execute and not args.preflight:
        print(json.dumps(plan, indent=2))
        for case in plan["cases"].values():
            for stage in case["stages"]:
                print(subprocess.list2cmdline(stage["command"]))
        return
    if video.current_platform() != args.platform:
        parser.error("Use the matching native platform RC5 environment for this series.")
    expected_hash = args.expected_library_sha256 if args.platform == "linux" else args.expected_dll_sha256
    if not expected_hash:
        parser.error("Supply --expected-library-sha256 on Linux or --expected-dll-sha256 on Windows.")
    if args.recipe == QWEN and args.vae_model_path is None:
        parser.error("Supply the reviewed BF16 native --vae-model-path, as in the runtime control.")
    if args.recipe == VIDEO:
        if args.execute and args.platform == "windows" and not ctypes.windll.shell32.IsUserAnAdmin():
            parser.error("Use an elevated terminal for the transformer standby purge.")
        if {Path(p).name: suite.digest(Path(p)) for p in plan["fixtures"]} != video.REFERENCE_FIXTURE_SHA256:
            parser.error("Video conditioning files differ from the reviewed fixtures.")
        ffmpeg_bin = args.ffmpeg_bin or os.getenv("DDO_RUNNER_FFMPEG_BIN")
        search = os.pathsep.join((str(ffmpeg_bin), os.environ.get("PATH", ""))) if ffmpeg_bin else None
        tools = {name: shutil.which(name, path=search) for name in ("ffmpeg", "ffprobe")}
        if not all(tools.values()):
            parser.error("Pass --ffmpeg-bin containing FFmpeg and ffprobe.")
        plan["ffprobe"] = str(Path(tools["ffprobe"]).resolve())
        plan["environment_overrides"]["DDO_RUNNER_FFMPEG_BIN"] = str(Path(tools["ffmpeg"]).resolve().parent)
    identity = suite.execution_identity(plan, expected_hash.lower())
    if args.recipe == QWEN:
        command = next(iter(plan["cases"].values()))["stages"][-1]["command"]
        source = Path(command[command.index("--vae-model-path") + 1])
        identity["vae_source"] = {str(p): suite.digest(p) for p in
                                  (source / "config.json", *sorted(source.glob("*.safetensors")))}
    if args.recipe == VIDEO:
        identity.update(fixture_sha256={p: suite.digest(Path(p)) for p in plan["fixtures"]},
                        ffmpeg_sha256={name: suite.digest(Path(p)) for name, p in tools.items()})
    if args.preflight:
        print(json.dumps(identity, indent=2))
        return

    def verify(case, stage, output, profile_dir):
        report = suite.read_json(suite.metrics_file(output))
        verify_features(args.recipe, case, report, calibration=stage.startswith("profile-"))
        return (video.verify_video_stage(plan, case, stage, output, profile_dir) if args.recipe == VIDEO else
                suite.verify_stage(suite.RECIPES[args.recipe], case, stage, output, profile_dir))

    suite.run_session(parser, args, identity, run, verify=verify)
