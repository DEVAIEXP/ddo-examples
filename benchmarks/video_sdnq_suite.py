"""Saved INT4 video recipes using the shared resumable benchmark controller."""

import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import json

if __package__:
    from .sdnq_suite import CASES, verify_profile
    from .run_benchmark import SNAPSHOT, VIDEO_BLOCKS_REVISION, REFERENCE_FIXTURE_SHA256
else:
    from sdnq_suite import CASES, verify_profile
    from run_benchmark import SNAPSHOT, VIDEO_BLOCKS_REVISION, REFERENCE_FIXTURE_SHA256

MODEL = "OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int4"
REVISION = "9f4973696b009ef66bfc7357b82d5e336f271b16"
PROFILE = "ltx23-video-1280x704-8s-8steps-sdnq-int4"
PROMPT = "the man plays the guitar and sings on an old western town"


def build_plan(suite, args):
    if args.platform != "windows":
        raise ValueError("This SDNQ video pair is prepared only for Windows.")
    cache = Path("F:/hf_cache/hub")
    checkpoint = args.checkpoint or cache / "models--OzzyGT--LTX-2.3-Distilled-1.1/snapshots" / SNAPSHOT
    packed = args.sdnq_model_path or cache / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    blocks = args.custom_blocks or suite.ROOT / "modular_diffusers/custom_blocks/ltx2_video"
    fixtures = suite.ROOT / "benchmarks/inputs/ltx23-video-5s"
    image, audio = fixtures / "man_guitar.png", fixtures / "diff_vocal_song.mp3"
    command = [sys.executable, "-B", "-u", str(suite.ROOT / "scripts/ltx2_video/run_dynamic_modular_sdnq_distilled.py"),
               "--distilled-model", str(checkpoint), "--model-path", str(packed), "--bits", "4",
               "--custom-blocks", str(blocks), "--local-files-only", "--image", str(image),
               "--audio", str(audio), "--use-audio-condition", "--prompt", PROMPT,
               "--width", "1280", "--height", "704", "--seconds", "8", "--frame-rate", "24",
               "--steps", "8", "--seed", "43", "--video-self-attention", "default",
               "--no-reuse-rotary-embeddings", "--stream-video-decode", "--video-decoder", "exact-streaming",
               "--decode-device", "cuda:0", "--memory-monitor", "rss", "--save-metrics"]
    cases = {}
    for name in CASES:
        if args.case and args.case != name:
            continue
        leaf = name == CASES[0]
        preset = "diffusers_leaf_offload_compat" if leaf else "one_shot_fast"
        cmd = command + ["--ddo-preset", preset]
        stages = []
        if not leaf:
            cmd += ["--ddo-profile", PROFILE]
            stages = [{"name": "profile-1", "command": cmd + ["--build-ddo-profile"]},
                      {"name": "profile-2", "command": cmd + ["--validate-ddo-profile"]}]
        stages.append({"name": "measure", "command": cmd})
        cases[name] = {"recipe": args.recipe, "preset": preset, "profile": None if leaf else PROFILE,
                       "dispatcher": None, "hold": None, "stages": stages,
                       "sdnq_checkpoint": str(packed), "checkpoint": str(checkpoint), "custom_blocks": str(blocks),
                       "image": str(image), "audio": str(audio),
                       "routes": {"encoder": "diffusers_leaf", "conditioning": "diffusers_leaf",
                                  "transformer": "official_diffusers_leaf" if leaf else "sdnq_runtime",
                                  "vae": "resident"}}
    return {"schema": 1, "platform": args.platform, "recipe": args.recipe, "family": "precision",
            "nominal_vram_gib": 8, "checkpoint": str(checkpoint), "checkpoint_revision": SNAPSHOT,
            "custom_blocks": str(blocks), "blocks_revision": VIDEO_BLOCKS_REVISION,
            "sdnq_checkpoint": str(packed), "sdnq_revision": REVISION, "expected_frames": 193,
            "fixtures": [str(image), str(audio)], "cases": cases,
            "environment_overrides": {"DIFFUSERS_ATTN_BACKEND": "native", "HF_HUB_OFFLINE": "1",
                                      "PYTHONNOUSERSITE": "1"}}


def prepare_media_identity(suite, plan, args):
    """Resolve installed codecs and verify fixed inputs without loading a model."""
    search = os.pathsep.join(filter(None, (str(args.ffmpeg_bin or ""), os.environ.get("PATH", ""))))
    tools = {name: shutil.which(name, path=search) for name in ("ffmpeg", "ffprobe")}
    if not all(tools.values()):
        raise ValueError("Install shared FFmpeg with ffprobe or pass --ffmpeg-bin.")
    plan["environment_overrides"].update(PATH=search, DDO_RUNNER_FFMPEG_BIN=str(Path(tools["ffmpeg"]).parent))
    plan["ffprobe"] = str(Path(tools["ffprobe"]).resolve())
    hashes = {}
    for name in plan["fixtures"]:
        path = Path(name)
        hashes[name] = suite.digest(path)
        if hashes[name] != REFERENCE_FIXTURE_SHA256[path.name]:
            raise ValueError("The fixed video input changed.")
    for case in plan["cases"].values():
        case["ffprobe"] = plan["ffprobe"]
    return {"fixtures_sha256": hashes, "ffprobe": plan["ffprobe"],
            "ffmpeg": tools["ffmpeg"]}


def verify_stage(suite, case, stage, output, profile_dir):
    path = suite.metrics_file(output)
    report = suite.read_json(path)
    calibration = stage.startswith("profile-")
    count = 2 if calibration else 8
    leaf = case["profile"] is None
    controls = report.get("requested_controls", {})
    expected = {"distilled_model": case["checkpoint"], "model_path": case["sdnq_checkpoint"],
                "custom_blocks": case["custom_blocks"], "local_files_only": True,
                "image": case["image"], "audio": case["audio"], "use_audio_condition": True,
                "prompt": PROMPT, "width": 1280, "height": 704, "seconds": 8,
                "frame_rate": 24, "steps": 8, "seed": 43, "bits": 4,
                "ddo_preset": case["preset"], "ddo_backend": "linear", "ddo_profile": case["profile"],
                "video_self_attention": "default", "reuse_rotary_embeddings": False,
                "stream_video_decode": True, "video_decoder": "exact-streaming", "decode_device": "cuda:0",
                "build_ddo_profile": stage == "profile-1", "validate_ddo_profile": stage == "profile-2"}
    if any(controls.get(key) != value for key, value in expected.items()):
        raise ValueError("The SDNQ video controls differ from the fixed recipe.")
    if (report.get("status") != "completed" or report.get("active_inference_steps") != count
            or report.get("runtime") != ("plan" if leaf else "sdnq_runtime")
            or report.get("memory_monitor", {}).get("ram_metric") != "rss"):
        raise ValueError("Incomplete SDNQ video execution or unexpected runtime.")
    workload = report.get("workload", {})
    for key, value in {"width": 1280, "height": 704, "frames": 193, "seconds": 8, "frame_rate": 24,
                       "steps_requested": 8, "seed": 43, "prompt": PROMPT, "dtype": "bfloat16",
                       "guidance_scale": 1, "guidance_rescale": 0.7, "audio_guidance_scale": 1}.items():
        if workload.get(key) != value:
            raise ValueError(f"Unexpected SDNQ video workload: {key}")
    recipe = report.get("quantization_recipe", {})
    if (recipe.get("quant_method") != "sdnq" or recipe.get("weights_dtype") != "uint4"
            or not report.get("quantized_backend", {}).get("sdnq_module_count", 0)):
        raise ValueError("Expected saved packed SDNQ INT4 modules.")
    features = report.get("model_features", {})
    for key in ("phase_modulation", "inplace_residuals", "compact_timesteps", "feed_forward_chunk_size",
                "normalization_chunk_size", "reuse_rotary_embeddings"):
        if key not in features or features[key] not in (False, 0):
            raise ValueError(f"Unexpected model feature: {key}")
    for phase in ("before_run", "after_text_encoder", "before_transformer", "after_run"):
        if controls.get("purge_" + phase) is not False:
            raise ValueError("The INT4 pair must not request standby purge.")
    if any(value is True for key in ("requested", "purged")
           for value in suite.walk_field(report.get("standby_purges", {}), key)):
        raise ValueError("Unexpected standby purge.")
    reports = report.get("dynamic_offload_reports", {})
    for component in ("text_encoder", "connectors", "vae", "audio_vae", "transformer"):
        expected_route = "dynamic_offload" if component == "transformer" and not leaf else "diffusers_group_offload"
        if reports.get(component, {}).get("route") != expected_route:
            raise ValueError(f"Unexpected component route: {component}")
    if not leaf and reports["transformer"].get("configuration", {}).get("dynamic_offload_execution_mode") != "sdnq_runtime":
        raise ValueError("The packed SDNQ runtime was not observed.")
    finite = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
    events = report.get("events", [])
    for suffix in ("host", "cuda"):
        rows = [row for row in events if row.get("name", "").startswith("denoise_step_")
                and row["name"].endswith("_" + suffix)]
        if len(rows) != count or any(row.get("step_index") != i or not finite(row.get("elapsed_sec"))
                                     for i, row in enumerate(rows)):
            raise ValueError("Incomplete video step timing observations.")
    verify_profile(suite, case, stage, report, profile_dir)
    artifacts = [path]
    phases = {row.get("name"): row.get("elapsed_sec") for row in report.get("steps", [])}
    decode_aggregate = next((event.get("elapsed_sec") for event in report.get("events", [])
                             if event.get("name") == "decode_and_save"), phases.get("decode_and_save"))
    if not {"text_encoder", "connectors", "video_encoder", "audio_encoder", "load_and_prepare_transformer",
            "attach_transformer", "denoise"}.issubset(phases) or not all(map(finite, phases.values())):
        raise ValueError("Incomplete video phase metrics.")
    videos = list(output.glob("*.mp4"))
    if calibration:
        if videos or decode_aggregate is not None or report.get("decode_route"):
            raise ValueError("SDNQ calibration unexpectedly decoded video.")
    else:
        decode = report.get("decode_route", {})
        media = report.get("decoded_media", {})
        if (decode.get("device") != "cuda:0" or decode.get("video_decoder") != "exact-streaming"
                or decode.get("stream_video_decode") is not True
                or decode.get("completed_stages") != ["video", "audio_mel", "audio"]
                or any(media.get(key) != value for key, value in
                       {"frames": 193, "width": 1280, "height": 704}.items())
                or media.get("audio_finite") is not True):
            raise ValueError("Incomplete exact-streaming output validation.")
        for key in ("total_elapsed_sec", "global_peak_vram_absolute_gb", "global_peak_ram_gb"):
            if not finite(report.get(key)):
                raise ValueError(f"Missing video metric: {key}")
        if not finite(decode_aggregate) or len(videos) != 1 or Path(report.get("output_path", "")).resolve() != videos[0].resolve():
            raise ValueError("Expected one reported MP4.")
        probe = json.loads(subprocess.check_output([case["ffprobe"], "-v", "error", "-show_entries",
                    "stream=codec_type,width,height,nb_frames", "-of", "json", str(videos[0])], text=True, timeout=30))
        streams = probe.get("streams", [])
        visual = [row for row in streams if row.get("codec_type") == "video"]
        if (len(visual) != 1 or (visual[0].get("width"), visual[0].get("height"), str(visual[0].get("nb_frames"))) != (1280, 704, "193")
                or not any(row.get("codec_type") == "audio" for row in streams)):
            raise ValueError("MP4 dimensions, frame count or audio stream differs.")
        artifacts += videos
    return {str(p): suite.digest(p) for p in artifacts}
