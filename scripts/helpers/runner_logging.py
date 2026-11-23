"""Filter known dependency notices while preserving runtime warnings and errors."""

import logging
import warnings
from contextlib import contextmanager
from pathlib import Path


_CUSTOM_BLOCKS = Path(__file__).resolve().parents[2] / "modular_diffusers" / "custom_blocks"
_TRUST_PREFIX = "`trust_remote_code` is enabled. Downloading code from "
_TRUST_SUFFIX = ". Please ensure you trust the contents of this repository"
_TORCHAO_NOTICE = (
    "Unable to import `torchao` Tensor objects. "
    "This may affect loading checkpoints serialized with `torchao`"
)


def _keep_record(record):
    if record.levelno != logging.WARNING:
        return True
    message = record.getMessage()
    if record.name == "torch.utils._pytree":
        return not (
            message.startswith(("<enum 'KernelPreference'> is an Enum subclass",
                                "<enum 'ScaleCalculationMode'> is an Enum subclass"))
            and "register_constant() on Enum subclasses is deprecated" in message
        )
    if record.name == "diffusers.quantizers.torchao.torchao_quantizer":
        return message != _TORCHAO_NOTICE
    if message.startswith(_TRUST_PREFIX) and message.endswith(_TRUST_SUFFIX):
        source = Path(message[len(_TRUST_PREFIX):-len(_TRUST_SUFFIX)])
        return not (source.is_absolute() and source.parent == _CUSTOM_BLOCKS and source.is_dir())
    return True


@contextmanager
def quiet_qwen_processor_doc_notices():
    from importlib import import_module

    auto_docstring = import_module("transformers.utils.auto_docstring")
    original_print = getattr(auto_docstring, "print", print)
    had_override = "print" in vars(auto_docstring)
    prefixes = tuple(
        f"[ERROR] `{name}` is part of Qwen3VLVideoProcessorInitKwargs, but not documented."
        for name in ("min_frames", "max_frames")
    )

    def print_other_notices(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            lines = args[0].split("\n")
            kept = [
                line for line in lines
                if not (
                    line.startswith(prefixes)
                    and line.endswith("video_processing_qwen3_vl.py.")
                )
            ]
            if kept != lines:
                if kept:
                    original_print("\n".join(kept), **kwargs)
                return
        original_print(*args, **kwargs)

    auto_docstring.print = print_other_notices
    try:
        yield
    finally:
        if had_override:
            auto_docstring.print = original_print
        else:
            del auto_docstring.print


def configure_runner_logging():
    warnings.filterwarnings(
        "ignore",
        message=r"^invalid escape sequence '\\\.'$",
        category=SyntaxWarning,
        module=r"^(?:torchao\.quantization\.quant_api|.*[/\\]torchao[/\\]quantization[/\\]quant_api(?:\.py)?)$",
    )
    for name in (
        "torch.utils._pytree",
        "diffusers.quantizers.torchao.torchao_quantizer",
        "diffusers.utils.dynamic_modules_utils",
    ):
        logging.getLogger(name).addFilter(_keep_record)
