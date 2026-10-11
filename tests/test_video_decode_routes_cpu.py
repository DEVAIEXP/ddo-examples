"""Shared decode-route contracts; no model weights or CUDA execution."""

import ast
import contextlib
import io
from pathlib import Path
import unittest

from scripts.ltx2_video import run_dynamic_minimal as minimal
from scripts.ltx2_video import run_dynamic_modular_distilled as linear
from scripts.ltx2_video import run_dynamic_modular_sdnq_distilled as sdnq


class VideoDecodeRoutes(unittest.TestCase):
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
