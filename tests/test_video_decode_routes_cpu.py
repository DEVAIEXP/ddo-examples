"""Shared decode-route contracts; no model weights or CUDA execution."""

import ast
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.ltx2_video import run_dynamic_minimal as minimal
from scripts.ltx2_video import run_dynamic_modular_distilled as linear
from scripts.ltx2_video import run_dynamic_modular_sdnq_distilled as sdnq


class VideoDecodeRoutes(unittest.TestCase):
    def test_public_code_revision_defaults_to_main_and_preserves_explicit_revision(self):
        from scripts.ltx2_video import run_dynamic_modular_vmm_distilled as native
        for runner in (minimal, linear, sdnq, native):
            for revision in (None, "main", "explicit-commit"):
                command = [] if revision is None else ["--custom-blocks-revision", revision]
                args = runner.parse_args(command)
                self.assertEqual(args.custom_blocks_revision, revision)
                tree = ast.parse(Path(runner.__file__).read_text())
                function_name = "load_custom_blocks" if runner is native else "main"
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
                # Execute just the real source-resolution branch, without model loading.
                branch = next(n for n in function.body if isinstance(n, ast.If)
                              and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                                      and c.func.id == "snapshot_download" for c in ast.walk(n)))
                with patch("huggingface_hub.snapshot_download", return_value="snapshot") as download:
                    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(runner.__file__), "exec"),
                         {"Path": Path, "source": "owner/code", "args": args})
                self.assertEqual(download.call_args.kwargs["revision"], revision)

    def test_preserves_cpu_defaults(self):
        for runner in (minimal, linear, sdnq):
            args = runner.parse_args([])
            self.assertEqual(args.decode_device, "cpu")
            self.assertFalse(args.stream_video_decode)

    def test_streaming_requires_explicit_gpu_device(self):
        for runner in (minimal, linear, sdnq):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(["--stream-video-decode"])
            args = runner.parse_args(["--stream-video-decode", "--decode-device", "cuda:0"])
            self.assertTrue(args.stream_video_decode)
            self.assertEqual(args.decode_device, "cuda:0")

    def test_decode_orchestration_matches_between_runners(self):
        routes = []
        for runner in (minimal, linear, sdnq):
            tree = ast.parse(Path(runner.__file__).read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            # Import compatibility can introduce earlier try blocks. Identify
            # the unique decode loop by its contract, not statement position.
            loops = [n for n in ast.walk(main) if isinstance(n, ast.For)
                     and ast.unparse(n.iter) == "decoder_stages"]
            self.assertEqual(len(loops), 1)
            loop = loops[0]
            routes.append(ast.dump(loop, include_attributes=False))
            text = ast.unparse(loop)
            self.assertIn('state.pop(\'generated_mel_spectrograms\', None)', text)
            self.assertIn('pipe.unload_components(component_names)', text)
            self.assertIn('input_device', text)
        self.assertEqual(routes[0], routes[1])
        self.assertEqual(routes[1], routes[2])


if __name__ == "__main__":
    unittest.main()
