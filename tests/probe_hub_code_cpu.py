"""Load trusted video code online/offline without weights or CUDA."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/ltx2_video"))
import torch
import ltx23_modular_vmm_video as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-files-only", action="store_true")
    options = parser.parse_args()
    revision = "f5c712e93c52283cb101da26d4a4544c808666bf"
    values = ["--custom-blocks-revision", revision]
    if options.local_files_only:
        values += ["--local-files-only"]
    args = runner.parse_args(values)
    assert not torch.cuda.is_initialized()
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        blocks = runner.load_custom_blocks(args)
        exports = runner.custom_block_exports(blocks)
        assert runner.resolve_local_transformer_loader(blocks) is exports.LTX2VideoTransformer3DModel
        prefix = type(blocks).__module__.rsplit(".", 1)[0] + "."
        for name in ("LTX2ImageCondition", "LTX2AudioCondition", "LTX2VideoPrepareRotaryStep",
                     "LTX2VideoAudioGuidance", "configure_float8_chunk_reuse",
                     "LTX2VideoStreamingDecoderStep", "LTX2VideoAudioMelDecoderStep", "LTX2VideoVocoderStep"):
            assert getattr(exports, name).__module__.startswith(prefix)
        pipe = blocks.init_pipeline()
        assert pipe.transformer is None and pipe.text_encoder is None
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status="passed", revision=revision, local_files_only=options.local_files_only,
                          weights_loaded=False, cuda_initialized=False)))


if __name__ == "__main__":
    main()
