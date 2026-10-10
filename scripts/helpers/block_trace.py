"""Opt-in host-only block observations, without replacing model forwards."""

from contextlib import contextmanager
import time


@contextmanager
def trace_blocks(blocks, tracker, step_index, *, clock=time.perf_counter):
    """Log boundaries without retaining tensors or synchronizing CUDA.

    Host durations include whatever waits the existing forward performs, not
    asynchronous GPU completion. Hooks do not isolate leaf upload/dequantization
    and can perturb Python execution; this is diagnostic, not benchmark timing.
    """
    handles = []
    pending = {}
    origin = clock()

    def before(index):
        def observe(module, inputs):
            now = clock()
            step = step_index()
            pending.setdefault(index, []).append((step, now))
            tracker.record_event(f"block_{index}_enter_host", 0.0,
                                 step_index=step, block_index=index, boundary="enter",
                                 trace_elapsed_sec=now - origin,
                                 timing_kind="host_forward_boundary_no_sync")
            print(f"  [block trace] step={step + 1} block={index} enter", flush=True)
        return observe

    def after(index):
        def observe(module, inputs, output):
            if not pending.get(index):
                return
            step, start = pending[index].pop()
            now = clock()
            tracker.record_event(f"block_{index}_exit_host", now - start,
                                 step_index=step, block_index=index, boundary="exit",
                                 trace_elapsed_sec=now - origin,
                                 timing_kind="host_forward_boundary_no_sync")
            print(f"  [block trace] step={step + 1} block={index} exit host={now - start:.4f}s", flush=True)
        return observe

    try:
        for index, block in enumerate(blocks):
            handles.append(block.register_forward_pre_hook(before(index)))
            handles.append(block.register_forward_hook(after(index), always_call=True))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        pending.clear()
