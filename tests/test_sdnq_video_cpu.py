"""Finite SDNQ video integration contracts; no weights or CUDA execution."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ltx2_video import run_dynamic_modular_sdnq_distilled as runner


class SDNQVideoContracts(unittest.TestCase):
    def test_defaults_and_sources(self):
        args = runner.parse_args([])
        self.assertEqual((args.width, args.height, args.seconds, args.steps, args.seed),
                         (1280, 704, 5, 8, 43))
        self.assertFalse(args.use_audio_condition)
        for bits in (4, 8):
            args = runner.parse_args(["--bits", str(bits)])
            self.assertTrue(runner.transformer_source(args).endswith(f"int{bits}"))
        args = runner.parse_args(["--model-path", "local-sdnq"])
        self.assertEqual(runner.transformer_source(args), "local-sdnq")

    def test_rejects_native_and_control_routes(self):
        for arguments in (["--ddo-backend", "vmm"], ["--ddo-preset", "off"],
                          ["--ddo-preset", "diffusers_offload_compat"], ["--bits", "3"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(arguments)

    def test_settings_select_sdnq_without_cuda(self):
        import torch
        with patch.dict(os.environ, {"DDO_PRESET": "off", "DDO_EXECUTION_MODE": "linear_runtime"}), \
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            settings = runner.sdnq_settings(runner.parse_args([]), "cuda:0")
        self.assertEqual(settings.execution_mode, "sdnq_runtime")
        self.assertEqual(settings.requested_preset, "auto")
        self.assertTrue(settings.enabled)
        self.assertFalse(settings.plan)

    def test_explicit_leaf_control_without_cuda(self):
        import torch
        args = runner.parse_args(["--bits", "4", "--ddo-preset", "diffusers_leaf_offload_compat"])
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            settings = runner.sdnq_settings(args, "cuda:0")
        self.assertEqual(runner.transformer_route(args), "diffusers_group_offload")
        self.assertEqual(settings.execution_mode, "plan")
        self.assertTrue(settings.plan)
        self.assertIn("int4", runner.transformer_source(args))

    def test_step_observer_uses_existing_phase_sync(self):
        source = Path(runner.__file__).read_text()
        tree = ast.parse(source)
        observer = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "observe_step")
        self.assertNotIn("synchronize", ast.unparse(observer))
        self.assertIn('state["callback_on_step_end"] = observe_step', source)
        self.assertIn("previous.elapsed_time(current)", source)

    def test_dry_run_does_not_import_backend_or_torch(self):
        captured = io.StringIO()
        with patch.object(sys, "argv", [runner.__file__, "--dry-run", "--bits", "4"]), \
                patch.dict(sys.modules, {"torch": None, "sdnq": None}), contextlib.redirect_stdout(captured):
            runner.main()
        plan = json.loads(captured.getvalue())
        self.assertEqual(plan["runtime"], "sdnq_runtime")
        self.assertEqual(plan["num_frames"], 121)
        self.assertEqual(plan["bits"], 4)

    def test_owns_workflow_and_preserves_backend_loading(self):
        source = Path(runner.__file__).read_text()
        tree = ast.parse(source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [ast.unparse(node.func) for node in ast.walk(main) if isinstance(node, ast.Call)]
        self.assertIn("transformer_type.from_pretrained", calls)
        self.assertIn("enable_offload", calls)
        self.assertIn("run_stage", calls)
        self.assertNotIn("from_pretrained_with_dynamic_offload", calls)
        self.assertNotIn("ensure_vmm_fast_process", source)
        self.assertNotIn(".forward =", source)
        self.assertIn("from scripts.helpers.inference_utils import RunTracker", source)


if __name__ == "__main__":
    unittest.main()
