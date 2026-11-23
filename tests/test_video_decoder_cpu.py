"""CPU contracts for the LTX `--video-decoder` selection."""
import argparse
from pathlib import Path
import types
import unittest

from scripts.helpers.ltx2 import video_decoder as vd

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ("run_dynamic_minimal.py", "run_dynamic_modular_distilled.py", "run_dynamic_modular_sdnq_distilled.py",
           "run_dynamic_modular_vmm_distilled.py", "run_dynamic_modular_diagnostics_distilled.py")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--stream-video-decode", action="store_true")
    vd.add_video_decoder_argument(p)
    return p


class VideoDecoderCpu(unittest.TestCase):
    def test_default_decoder_when_streaming(self):
        p = parser()
        args = p.parse_args(["--stream-video-decode"])
        vd.validate_video_decoder(p, args)
        self.assertEqual(args.video_decoder, "exact-streaming")

        args = p.parse_args(["--stream-video-decode", "--video-decoder", "streaming-tiles"])
        vd.validate_video_decoder(p, args)
        self.assertEqual(args.video_decoder, "streaming-tiles")

    def test_default_without_streaming(self):
        p = parser()
        args = p.parse_args([])
        vd.validate_video_decoder(p, args)
        self.assertEqual(args.video_decoder, "streaming-tiles")

    def test_exact_streaming_requires_streaming_decode(self):
        p = parser()
        args = p.parse_args(["--video-decoder", "exact-streaming"])
        with self.assertRaises(SystemExit):
            vd.validate_video_decoder(p, args)
        vd.validate_video_decoder(p, p.parse_args(["--stream-video-decode", "--video-decoder", "exact-streaming"]))
        with self.assertRaises(SystemExit):
            parser().parse_args(["--video-decoder", "tiled"])

    def test_step_selection(self):
        class Tiles: pass
        class Exact: pass
        exports = types.SimpleNamespace(LTX2VideoStreamingDecoderStep=Tiles, LTX2VideoExactStreamingDecoderStep=Exact)
        self.assertIsInstance(vd.streaming_video_decoder_step(exports, "streaming-tiles"), Tiles)
        self.assertIsInstance(vd.streaming_video_decoder_step(exports, "exact-streaming"), Exact)
        old = types.SimpleNamespace(LTX2VideoStreamingDecoderStep=Tiles)
        with self.assertRaisesRegex(RuntimeError, "do not export"):
            vd.streaming_video_decoder_step(old, "exact-streaming")
        with self.assertRaises(ValueError):
            vd.streaming_video_decoder_step(exports, "other")

    def test_runners_wire_the_option(self):
        for name in RUNNERS:
            text = (ROOT / "scripts/ltx2_video" / name).read_text(encoding="utf-8")
            with self.subTest(runner=name):
                self.assertIn("add_video_decoder_argument(parser", text)
                self.assertIn("validate_video_decoder(parser, args)", text)
                self.assertIn("streaming_video_decoder_step(exports, args.video_decoder)", text)
                self.assertNotIn("LTX2VideoStreamingDecoderStep(), [\"vae\"]", text)
                self.assertIn('"video_decoder": args.video_decoder', text)


if __name__ == "__main__":
    unittest.main()
