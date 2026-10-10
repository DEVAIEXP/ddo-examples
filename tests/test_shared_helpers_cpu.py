"""Shared telemetry and CLI ownership contracts; no CUDA initialization."""
import ast
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.helpers import inference_utils as metrics
from scripts.ltx2_video import run_dynamic_minimal as minimal
from scripts.ltx2_video import run_dynamic_modular_distilled as linear
from scripts.ltx2_video import run_dynamic_modular_vmm_distilled as native
from scripts.ltx2_video import run_dynamic_modular_diagnostics_distilled as diagnostics


class SharedContracts(unittest.TestCase):
    def test_single_metrics_implementation(self):
        for model in ("ltx2_image", "ltx2_video"):
            self.assertFalse((ROOT / "scripts" / model / "inference_utils.py").exists())
        for path in (ROOT / "scripts/ltx2_image").glob("run_*.py"):
            self.assertIn("from scripts.helpers.inference_utils import", path.read_text())
        for runner in (minimal, linear, native, diagnostics):
            self.assertIn("from scripts.helpers.inference_utils import", Path(runner.__file__).read_text())
        self.assertTrue(callable(metrics.release_pinned_host_cache))
        self.assertTrue(callable(metrics.get_pinned_host_memory_stats))

    def test_monitor_validation(self):
        for interval in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                metrics.PeakMemoryMonitor(interval=interval)

    def test_metrics_schema_and_public_report(self):
        data = {"events": [], "steps": []}
        with patch.object(metrics.torch.cuda, "init"), patch.object(metrics, "get_gpu_used_gb", return_value=1):
            tracker = metrics.RunTracker("cuda:0", data, show_metrics=False, memory_monitor="rss")
        class Report:
            def as_dict(self):
                return {"version": 1, "capacity": {"bytes": 12}}
        tracker.record_ddo_metrics(Report())
        with patch.object(metrics, "get_ram_rss_gb", return_value=2), patch.object(metrics, "get_gpu_used_gb", return_value=7):
            tracker.sample_memory()
        with patch.object(tracker.monitor, "stop", return_value=(7, 2)):
            phase = tracker.step_end("denoise", metrics.time.time())
        self.assertEqual(data["global_peak_vram_absolute_gb"], 7)
        self.assertEqual(phase["peak_vram_gb"], 6)
        self.assertEqual(phase["peak_vram_absolute_gb"], 7)
        self.assertEqual(phase["ram_metric"], "rss")
        self.assertEqual(phase["memory_monitor"]["memory_unit"], "GiB")
        self.assertEqual(data["dynamic_offload_reports"]["transformer"]["version"], 1)

    def test_monitor_errors_are_not_silent(self):
        monitor = metrics.PeakMemoryMonitor(mode="rss")
        with patch.object(monitor, "sample_once", side_effect=ValueError("sample failed")):
            monitor._poll()
        with self.assertRaisesRegex(RuntimeError, "peaks are incomplete"):
            monitor.stop()

    def test_minimal_and_linear_routes(self):
        for runner in (minimal, linear):
            for preset in ("auto", "low_ram_safe", "diffusers_leaf_offload_compat", "off"):
                args = runner.parse_args(["--dry-run", "--ddo-preset", preset])
                self.assertEqual(getattr(args, "ddo_backend", "linear"), "linear")
                self.assertFalse(hasattr(args, "compile_repeated_blocks"))
            with self.assertRaises(SystemExit):
                runner.parse_args(["--diagnose-denoise"])
            with self.assertRaises(SystemExit):
                runner.parse_args(["--ddo-preset", "vmm_fast"])

    def test_vmm_separation_and_preserved_defaults(self):
        args = native.parse_args(["--dry-run"])
        self.assertEqual(native.describe_plan(args)["effective_preset"], "vmm_fast")
        legacy = diagnostics.parse_args(["--dry-run", "--ddo-backend", "vmm"])
        for name, value in vars(legacy).items():
            if name != "runner_entrypoint":
                self.assertEqual(getattr(args, name), value, name)
        with self.assertRaises(SystemExit):
            native.parse_args(["--compile-repeated-blocks"])
        with self.assertRaises(SystemExit):
            native.parse_args(["--ddo-backend", "linear"])
        diagnostic = diagnostics.parse_args(["--ddo-backend", "vmm", "--compile-repeated-blocks"])
        self.assertTrue(diagnostic.compile_repeated_blocks)

    def test_bootstrap_precedes_parser_and_runtime(self):
        class ReachedBootstrap(Exception):
            pass
        with patch.object(sys, "argv", ["runner.py"]), patch("ddo_vmm_bootstrap.ensure_vmm_fast_process", side_effect=ReachedBootstrap):
            with patch.object(native, "parse_args", side_effect=AssertionError("parser before bootstrap")):
                with self.assertRaises(ReachedBootstrap):
                    native.main()

    def test_public_runners_own_the_modular_workflow(self):
        self.assertFalse((ROOT / "scripts/helpers/ltx2/video_workflow.py").exists())
        for runner in (minimal, linear, native, diagnostics):
            tree = ast.parse(Path(runner.__file__).read_text())
            functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            self.assertIn("parse_args", functions)
            self.assertIn("main", functions)
            self.assertTrue("run" in functions or "run_stage" in functions)
            calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
            for name in ("load_components", "update_components", "unload_components"):
                self.assertTrue(any(isinstance(call, ast.Attribute) and call.attr == name for call in calls))
            self.assertTrue(any(isinstance(call, ast.Name) and call.id == "enable_offload" for call in calls))

    def test_basic_generation_defaults_match_native(self):
        reference = native.parse_args(["--dry-run"])
        for runner in (minimal, linear):
            args = runner.parse_args(["--dry-run"])
            for name in ("distilled_model", "prompt", "image", "audio", "use_audio_condition",
                         "width", "height", "seconds", "frame_rate", "steps", "seed"):
                self.assertEqual(getattr(args, name), getattr(reference, name), name)

    def test_linear_profile_is_explicit(self):
        self.assertFalse(linear.parse_args([]).build_ddo_profile)
        with self.assertRaises(SystemExit):
            linear.parse_args(["--build-ddo-profile"])
        args = linear.parse_args(["--build-ddo-profile", "--ddo-profile", "video-workload"])
        self.assertTrue(args.build_ddo_profile)
        self.assertEqual(args.ddo_profile, "video-workload")

    def test_all_scripts_parse(self):
        for path in (ROOT / "scripts").rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
