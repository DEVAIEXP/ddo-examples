"""Finite CPU-only contracts for the independent example host."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "scripts/ltx2_video/ltx23_modular_vmm_video.py"
spec = importlib.util.spec_from_file_location("video_runner", VIDEO)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MigrationContracts(unittest.TestCase):
    def test_manifest(self):
        manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(manifest["project"]["name"], "ddo-examples")
        sources = manifest["tool"]["uv"]["sources"]
        self.assertEqual(sources["diffusers-dynamic-offloader"]["path"], "../diffusers-dynamic-offloader")
        self.assertNotIn("LTX-2", str(sources))
        self.assertNotIn("ltx2.3-image-diffusers", str(sources))

    def test_clones_are_ignored_not_submodules(self):
        self.assertFalse((ROOT / ".gitmodules").exists())
        ignored = (ROOT / ".gitignore").read_text()
        self.assertIn("modular_diffusers/custom_blocks/", ignored)
        for name in ("ltx2_image", "ltx2_video"):
            self.assertTrue((ROOT / "modular_diffusers/custom_blocks" / name / ".git").is_dir())

    def test_video_defaults_and_output_root(self):
        args = runner.parse_args(["--ddo-backend", "vmm", "--dry-run"])
        plan = runner.describe_plan(args)
        self.assertEqual((plan["width"], plan["height"], plan["num_frames"], plan["steps"], plan["seed"]),
                         (1280, 704, 121, 8, 43))
        self.assertFalse(args.use_audio_condition)
        self.assertEqual(plan["effective_preset"], "vmm_fast")
        self.assertEqual(runner.ROOT, ROOT)
        self.assertTrue(Path(args.output_dir).is_relative_to(ROOT / "outputs"))

    def test_explicit_audio(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--use-audio-condition", "--dry-run"])
        args = runner.parse_args(["--use-audio-condition", "--audio", "input.wav", "--dry-run"])
        self.assertTrue(args.use_audio_condition)
        self.assertEqual(args.audio, "input.wav")

    def test_saved_fp8_stays_explicit(self):
        from diffusers_dynamic_offloader import DynamicOffloadConfig
        base = DynamicOffloadConfig(execution_mode="vmm_fast_runtime")
        self.assertIs(runner.build_transformer_config(runner.parse_args([]), base), base)
        args = runner.parse_args(["--ddo-backend", "vmm", "--ddo-preset", "auto",
                                  "--transformer-fp8-weight-only", "--model-path", "saved-model",
                                  "--local-files-only", "--dry-run"])
        config = runner.build_transformer_config(args, base)
        self.assertTrue(config.vmm_float8_weight_only)
        self.assertFalse(args.compile_fp8_linears)
        self.assertFalse(base.vmm_float8_weight_only)

    def test_hub_download_is_code_only_and_pinned(self):
        args = runner.parse_args(["--local-files-only", "--dry-run"])
        with patch("huggingface_hub.snapshot_download", return_value="code-snapshot") as download:
            with patch("diffusers.ModularPipelineBlocks.from_pretrained", return_value="blocks") as load:
                self.assertEqual(runner.load_custom_blocks(args), "blocks")
        self.assertEqual(download.call_args.kwargs["allow_patterns"], ["*.py", "modular_config.json"])
        self.assertEqual(download.call_args.kwargs["revision"], "f5c712e93c52283cb101da26d4a4544c808666bf")
        self.assertTrue(download.call_args.kwargs["local_files_only"])
        load.assert_called_once_with("code-snapshot", trust_remote_code=True, local_files_only=True)

    def test_local_video_code_identity_without_weights(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        args = runner.parse_args(["--custom-blocks",
                                  str(ROOT / "modular_diffusers/custom_blocks/ltx2_video"),
                                  "--local-files-only", "--dry-run"])
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            blocks = runner.load_custom_blocks(args)
            exports = runner.custom_block_exports(blocks)
            self.assertTrue(exports.LTX2ImageCondition.__module__.startswith("diffusers_modules."))
            self.assertIsNotNone(exports.LTX2VideoStreamingDecoderStep)
            pipe = blocks.init_pipeline()
            for name in ("transformer", "text_encoder", "vae", "audio_vae", "connectors", "vocoder"):
                self.assertIsNone(pipe.components[name])
        self.assertFalse(torch.cuda.is_initialized())

    def test_image_clone_exports(self):
        import torch
        sys.path.insert(0, str(ROOT / "modular_diffusers"))
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            from custom_blocks.ltx2_image import LTX2ImageDistilledBlocks
            from custom_blocks.ltx2_image.connectors_ltx2_image import LTX2ImageTextConnectors
            from custom_blocks.ltx2_image.modular_blocks_ltx2_image import LTX2ImageDenoiseStep
            self.assertTrue(issubclass(LTX2ImageDistilledBlocks, object))
            self.assertIsNotNone(LTX2ImageTextConnectors)
            self.assertIsNotNone(LTX2ImageDenoiseStep)
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_old_host_paths(self):
        for script in (ROOT / "scripts").rglob("*.py"):
            text = script.read_text(encoding="utf-8-sig")
            ast.parse(text)
            self.assertNotIn("E:\\LTX-2", text)
            self.assertNotIn("E:/LTX-2", text)
            self.assertNotIn("ltx2.3-image-diffusers", text)
            self.assertNotIn("E:\\ProjetosIA", text)

    def test_help_all_runners_without_cuda(self):
        scripts = list((ROOT / "scripts/ltx2_image").glob("run_*.py")) + list((ROOT / "scripts/ltx2_video").glob("run_*.py")) + [VIDEO]
        for script in scripts:
            with self.subTest(script=script.name):
                code = (
                    "import sys,runpy,torch;"
                    "torch.cuda._lazy_init=lambda: (_ for _ in ()).throw(AssertionError('CUDA forbidden'));"
                    "sys.path.insert(0,str(__import__('pathlib').Path(sys.argv[1]).parent));"
                    "sys.argv=[sys.argv[1],'--help'];"
                    "runpy.run_path(sys.argv[0],run_name='__main__')"
                )
                result = subprocess.run([sys.executable, "-B", "-c", code, str(script)],
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), cwd=ROOT,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_video_dry_run(self):
        result = subprocess.run([sys.executable, "-B", str(VIDEO), "--dry-run",
                                 "--ddo-backend", "vmm", "--ddo-preset", "auto"],
                                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["effective_preset"], "vmm_fast")


if __name__ == "__main__":
    unittest.main()
