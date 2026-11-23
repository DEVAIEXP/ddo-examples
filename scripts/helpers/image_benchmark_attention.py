"""Transformer attention controls for image runners."""


LTX_BACKENDS = {"default": "native", "cudnn": "_native_cudnn", "flash-attn2": "flash"}
QWEN_BACKENDS = {"default": "native", "cudnn": "_native_cudnn"}


def add_image_attention_arguments(parser, *, qwen=False):
    backends = QWEN_BACKENDS if qwen else LTX_BACKENDS
    flag = "--image-joint-attention" if qwen else "--image-self-attention"
    legacy = {backend: name for name, backend in backends.items()}
    parser.add_argument(
        flag, "--transformer-attention", dest="image_attention", choices=tuple(backends),
        type=lambda value: legacy.get(value, value), default="default",
        help="Qwen joint text/image attention backend." if qwen else "LTX image self-attention backend.",
    )
    parser.add_argument("--trace-transformer-attention", action="store_true",
                        help="Record operators from the first transformer forward. Diagnostic only.")


def configure_image_attention(transformer, args, metrics, *, qwen=False):
    backend = (QWEN_BACKENDS if qwen else LTX_BACKENDS)[args.image_attention]
    metrics["transformer_attention_requested"] = backend
    metrics["transformer_attention_scope"] = "joint" if qwen else "self"
    if backend != "native":
        if qwen:
            transformer.set_attention_backend(backend)
        else:
            from diffusers.models.attention_dispatch import AttentionBackendName, _check_attention_backend_requirements

            selected = AttentionBackendName(backend)
            _check_attention_backend_requirements(selected)
            pairs = [(block.attn1.processor, block.attn2.processor)
                     for block in transformer.transformer_blocks]
            if not pairs or any(not hasattr(processor, "_attention_backend") or
                                getattr(processor, "_parallel_config", None) is not None
                                for pair in pairs for processor in pair):
                raise ValueError("Expected LTX image attention processors without context parallelism.")
            if {id(self_attention) for self_attention, _ in pairs} & {id(cross) for _, cross in pairs}:
                raise ValueError("LTX image self and cross-attention must use separate processors.")
            for self_attention, cross_attention in pairs:
                self_attention._attention_backend = selected
                cross_attention._attention_backend = AttentionBackendName.NATIVE
    if not args.trace_transformer_attention:
        return

    import torch

    profiler = None

    def start(_module, _inputs):
        nonlocal profiler
        before.remove()
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        profiler.start()

    def finish(_module, _inputs, _output):
        after.remove()
        if profiler is not None:
            profiler.stop()
            metrics["transformer_attention_operators"] = sorted(
                event.key for event in profiler.key_averages()
                if "attention" in event.key.lower() or "flash_attn" in event.key.lower()
            )

    before = transformer.register_forward_pre_hook(start)
    after = transformer.register_forward_hook(finish, always_call=True)
