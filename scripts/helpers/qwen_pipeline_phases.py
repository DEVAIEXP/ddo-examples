"""Observe Qwen Image 2.1's official loop and decode without replacing its math."""

from contextlib import ExitStack, contextmanager
from functools import wraps


@contextmanager
def measure_qwen_pipeline_phases(pipe, tracker, metrics, *, before_decode=None):
    """Keep the full pipeline while measuring its public loop/output boundaries.

    The reviewed pipeline enters progress_bar only around its sampling loop.
    Decode starts after that loop and ends after image postprocessing, including
    latent unpacking/scaling and CPU image conversion, but excluding hook cleanup
    and file saving. An optional host cleanup runs in a separate interval before
    decode. The observer itself adds no synchronization or replacement forward.
    """
    original_progress = pipe.progress_bar
    processor = pipe.image_processor
    original_postprocess = processor.postprocess
    saved = [(obj, key, key in vars(obj), vars(obj).get(key))
             for obj, key in ((pipe, "progress_bar"), (processor, "postprocess"))]
    loop_count = decode_count = 0
    with ExitStack() as decode_scope:
        @contextmanager
        def progress_bar(*args, **kwargs):
            nonlocal loop_count
            loop_count += 1
            if loop_count != 1:
                raise RuntimeError("Expected exactly one Qwen denoising loop")
            with tracker.subphase("denoise"):
                with original_progress(*args, **kwargs) as bar:
                    yield bar
            if before_decode is not None:
                with tracker.subphase("cleanup_before_decode"):
                    before_decode()
                metrics["pre_decode_cleanup"] = {
                    "status": "completed", "scope": "after_sampling_before_decode",
                }
            decode_scope.enter_context(tracker.subphase("decode"))

        @wraps(original_postprocess)
        def postprocess(*args, **kwargs):
            nonlocal decode_count
            if loop_count != 1 or decode_count != 0:
                raise RuntimeError("Unexpected Qwen decode/postprocess boundary")
            result = original_postprocess(*args, **kwargs)
            decode_scope.close()
            decode_count += 1
            return result

        pipe.progress_bar = progress_bar
        processor.postprocess = postprocess
        try:
            yield
            if loop_count != 1 or decode_count != 1:
                raise RuntimeError("Qwen did not complete isolated denoise and decode measurements")
            metrics["isolated_phase_measurements"] = {
                "schema": 1,
                "denoise": "official sampling loop including scheduler and callbacks",
                "decode": "latent preparation through image postprocess, excluding hook cleanup and saving",
                "cuda_synchronization_added": before_decode is not None,
            }
        finally:
            for obj, key, had_instance_attribute, value in reversed(saved):
                if had_instance_attribute:
                    setattr(obj, key, value)
                else:
                    delattr(obj, key)
