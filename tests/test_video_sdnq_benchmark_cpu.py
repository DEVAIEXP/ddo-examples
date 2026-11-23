"""CPU contracts for the saved INT4 video pair; no model loading."""

import ast
import json
import pytest

from benchmarks import image_suite as suite
from benchmarks import sdnq_suite as sdnq
from benchmarks import video_sdnq_suite as video
from tests.test_sdnq_benchmark_cpu import options, write_profile


def plan():
    return sdnq.build_plan(suite, options(recipe="ltx23-video-8s"))


def test_real_parsers_schedule_and_purge(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from scripts.ltx2_video.run_dynamic_modular_sdnq_distilled import parse_args, prepare_profile_steps
    value = plan()
    assert value["sdnq_revision"] == video.REVISION
    for case in value["cases"].values():
        for stage in case["stages"]:
            args = parse_args(stage["command"][4:])
            assert (args.width, args.height, args.seconds, args.steps, args.bits) == (1280, 704, 8, 8, 4)
            assert args.video_decoder == "exact-streaming" and args.decode_device == "cuda:0"
            assert not args.reuse_rotary_embeddings
            assert all(getattr(args, "purge_" + p) is False for p in
                       ("before_run", "after_text_encoder", "before_transformer", "after_run"))
            state = {"timesteps": torch.arange(8), "sigmas": list(range(9)), "num_inference_steps": 8}
            count = prepare_profile_steps(state, args)
            assert count == (8 if stage["name"] == "measure" else 2)
            assert state["timesteps"].tolist() == list(range(count))
            assert state["sigmas"] == list(range(9)) and state["num_inference_steps"] == 8
    with pytest.raises(SystemExit): parse_args(["--validate-ddo-profile"])
    with pytest.raises(SystemExit): parse_args(["--ddo-preset", "diffusers_leaf_offload_compat", "--ddo-profile", "bad"])
    assert not torch.cuda.is_initialized()


def fixture(case, stage):
    from scripts.ltx2_video.run_dynamic_modular_sdnq_distilled import parse_args
    command = next(row["command"] for row in case["stages"] if row["name"] == stage)
    count = 8 if stage == "measure" else 2
    leaf = case["profile"] is None
    return {"requested_controls": vars(parse_args(command[4:])), "status": "completed",
            "runtime": "plan" if leaf else "sdnq_runtime", "active_inference_steps": count,
            "memory_monitor": {"ram_metric": "rss"},
            "workload": {"width": 1280, "height": 704, "frames": 193, "seconds": 8, "frame_rate": 24,
                         "steps_requested": 8, "seed": 43, "prompt": video.PROMPT, "dtype": "bfloat16",
                         "guidance_scale": 1, "guidance_rescale": .7, "audio_guidance_scale": 1},
            "model_features": {k: False for k in ("phase_modulation", "inplace_residuals", "compact_timesteps",
                  "feed_forward_chunk_size", "normalization_chunk_size", "reuse_rotary_embeddings")},
            "quantization_recipe": {"quant_method": "sdnq", "weights_dtype": "uint4"},
            "quantized_backend": {"sdnq_module_count": 10},
            "dynamic_offload_reports": {k: {"route": "dynamic_offload" if k == "transformer" and not leaf else "diffusers_group_offload",
                    "configuration": {"dynamic_offload_execution_mode": "sdnq_runtime"}}
                    for k in ("text_encoder", "connectors", "vae", "audio_vae", "transformer")},
            "events": [{"name": f"denoise_step_{i+1}_{kind}", "step_index": i, "elapsed_sec": 1}
                       for i in range(count) for kind in ("host", "cuda")],
            "steps": [{"name": k, "elapsed_sec": 1} for k in ("text_encoder", "connectors", "video_encoder", "audio_encoder",
                       "load_and_prepare_transformer", "attach_transformer", "denoise")],
            "ddo_profile": None if leaf else {"name": case["profile"], "status": "applied", "resident_budget_gb": 2}}


@pytest.mark.parametrize("mutation", [None, "purge", "route", "schedule", "feature", "decode"])
def test_calibration_evidence(tmp_path, monkeypatch, mutation):
    import torch
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    case = plan()["cases"]["sdnq-int4-ddo"]
    report = fixture(case, "profile-2")
    profile = tmp_path / "profiles/packed.json"
    write_profile(profile, case)
    report["ddo_profile"].update(status="saved", path=str(profile), profile_measurement_kind="sdnq_probe", build_stage="final")
    if mutation == "purge": report["standby_purges"] = {"before_run": {"requested": True}}
    if mutation == "route": report["dynamic_offload_reports"]["connectors"]["route"] = "dynamic_offload"
    if mutation == "schedule": report["active_inference_steps"] = 8
    if mutation == "feature": report["model_features"]["phase_modulation"] = True
    if mutation == "decode": report["decode_route"] = {"device": "cuda:0"}
    output = tmp_path / "output"
    output.mkdir()
    (output / "metrics.json").write_text(json.dumps(report))
    monkeypatch.setattr(suite, "metrics_file", lambda _: output / "metrics.json")
    if mutation:
        with pytest.raises(ValueError): sdnq.verify_stage(suite, case, "profile-2", output, profile.parent)
    else:
        assert sdnq.verify_stage(suite, case, "profile-2", output, profile.parent)


def test_calibration_retires_transformer_before_early_return():
    source = (suite.ROOT / "scripts/ltx2_video/run_dynamic_modular_sdnq_distilled.py").read_text()
    ast.parse(source)
    assert source.index('finish_phase("cleanup_transformer"') < source.index("del result, transformer, scheduler") < source.index('if profile_only:\n            metrics["status"]')
    assert source.index('if profile_only:\n            metrics["status"]') < source.index('decode_and_save_start = time.time()')


def test_shared_controller_and_fp8_dispatch_preserved():
    source = (suite.ROOT / "benchmarks/run_benchmark.py").read_text()
    assert 'selected.case in (None, "sdnq-int4-leaf", "sdnq-int4-ddo")' in source
    assert 'video_runtime_main(image_suite)' in source
    source = (suite.ROOT / "benchmarks/sdnq_suite.py").read_text()
    assert "suite.run_session(parser, args, identity, run," in source
    assert all(len(case["stages"]) == (1 if name.endswith("leaf") else 3)
               for name, case in plan()["cases"].items())


@pytest.mark.parametrize("bad_frames", [False, True])
def test_measurement_requires_verified_media(tmp_path, monkeypatch, bad_frames):
    import torch
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    case = plan()["cases"]["sdnq-int4-leaf"]
    case["ffprobe"] = "ffprobe"
    report = fixture(case, "measure")
    movie = tmp_path / "result.mp4"
    movie.write_bytes(b"fixture; subprocess output is mocked")
    report.update(output_path=str(movie), total_elapsed_sec=20, global_peak_vram_absolute_gb=7,
                  global_peak_ram_gb=20, decoded_media={"frames": 193, "width": 1280, "height": 704,
                                                       "audio_finite": True, "audio_samples": 768960},
                  decode_route={"device": "cuda:0", "video_decoder": "exact-streaming", "stream_video_decode": True,
                                "completed_stages": ["video", "audio_mel", "audio"]})
    report["steps"].append({"name": "decode_and_save", "elapsed_sec": 1})
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(suite, "metrics_file", lambda _: path)
    monkeypatch.setattr(video.subprocess, "check_output", lambda *a, **k: json.dumps({"streams": [
        {"codec_type": "video", "width": 1280, "height": 704, "nb_frames": "192" if bad_frames else "193"},
        {"codec_type": "audio"}]}))
    if bad_frames:
        with pytest.raises(ValueError, match="frame count"): video.verify_stage(suite, case, "measure", tmp_path, tmp_path / "profiles")
    else:
        assert str(movie) in video.verify_stage(suite, case, "measure", tmp_path, tmp_path / "profiles")
