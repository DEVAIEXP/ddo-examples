"""Block tracing contracts with fake modules; no Torch or CUDA is required."""

import contextlib
import io
import unittest

from scripts.helpers.block_trace import trace_blocks


class Handle:
    def __init__(self, hooks, hook):
        self.hooks, self.hook = hooks, hook

    def remove(self):
        self.hooks.remove(self.hook)


class Block:
    def __init__(self):
        self.pre, self.post = [], []

    def register_forward_pre_hook(self, hook):
        self.pre.append(hook)
        return Handle(self.pre, hook)

    def register_forward_hook(self, hook, *, always_call):
        assert always_call
        self.post.append(hook)
        return Handle(self.post, hook)

    def __call__(self, value):
        for hook in self.pre:
            assert hook(self, (value,)) is None
        for hook in self.post:
            assert hook(self, (value,), value) is None
        return value


class Tracker:
    def __init__(self):
        self.events = []

    def record_event(self, name, elapsed, **metadata):
        self.events.append((name, elapsed, metadata))


class BlockTraceTests(unittest.TestCase):
    def test_output_preserved_step_and_duration_recorded_hooks_removed(self):
        block, tracker = Block(), Tracker()
        ticks = iter([10.0, 11.0, 13.5])
        token = object()
        with contextlib.redirect_stdout(io.StringIO()), trace_blocks(
                [block], tracker, lambda: 2, clock=lambda: next(ticks)):
            self.assertIs(block(token), token)
        self.assertEqual(len(tracker.events), 2)
        self.assertEqual(tracker.events[1][1], 2.5)
        self.assertEqual(tracker.events[0][2]["step_index"], 2)
        self.assertEqual((block.pre, block.post), ([], []))

    def test_exception_removes_hooks(self):
        block = Block()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with trace_blocks([block], Tracker(), lambda: 0):
                raise RuntimeError("failure")
        self.assertEqual((block.pre, block.post), ([], []))

    def test_partial_registration_removes_installed_hooks(self):
        class BrokenBlock(Block):
            def register_forward_hook(self, hook, *, always_call):
                raise RuntimeError("registration failure")
        block = BrokenBlock()
        with self.assertRaisesRegex(RuntimeError, "registration failure"):
            with trace_blocks([block], Tracker(), lambda: 0):
                pass
        self.assertEqual(block.pre, [])


if __name__ == "__main__":
    unittest.main()
