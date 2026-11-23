"""Selection of the LTX video streaming decoder step (`--video-decoder`)."""

VIDEO_DECODERS = ("streaming-tiles", "exact-streaming")
_STEPS = {
    "streaming-tiles": "LTX2VideoStreamingDecoderStep",
    "exact-streaming": "LTX2VideoExactStreamingDecoderStep",
}


def add_video_decoder_argument(parser):
    parser.add_argument(
        "--video-decoder", choices=VIDEO_DECODERS,
        help="Video decoder used with --stream-video-decode (default: exact-streaming). "
             "Select streaming-tiles for overlapping temporal windows and lower peak VRAM.")


def validate_video_decoder(parser, args):
    if args.video_decoder is None:
        args.video_decoder = "exact-streaming" if args.stream_video_decode else "streaming-tiles"
    if args.video_decoder != "streaming-tiles" and not args.stream_video_decode:
        parser.error("--video-decoder exact-streaming requires --stream-video-decode.")


def streaming_video_decoder_step(exports, name):
    """Instantiate the selected streaming decoder step from the custom-block exports."""
    if name not in _STEPS:
        raise ValueError(f"Unknown video decoder: {name}")
    step = getattr(exports, _STEPS[name], None)
    if step is None:
        raise RuntimeError(f"The loaded custom blocks do not export {_STEPS[name]}; "
                           "use custom blocks that include it or --video-decoder streaming-tiles.")
    return step()
