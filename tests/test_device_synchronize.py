"""``synchronize`` -- the timing-attribution device barrier.

CUDA launches are asynchronous, so a ``launch`` span measured without a
barrier reads as microseconds and the real wall silently lands on whichever
device-to-host pull blocks next. That is how job 508's first SOBBH telemetry
could only say "kernel = 100.0%" without naming WHICH part of the call.
:func:`lisatools.utils.device.synchronize` is the barrier the scoring
telemetry places at that span boundary (always immediately before a blocking
D2H, so it costs nothing it did not already cost).

Here we lock the CPU contract -- a strict no-op that never raises -- and the
dispatch rule (drive ``xp.cuda.runtime.deviceSynchronize`` when, and only
when, ``xp`` has a ``cuda`` attribute).
"""

import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.utils.device import synchronize


class SynchronizeCpuTest(unittest.TestCase):
    def test_numpy_is_a_silent_noop(self):
        self.assertIsNone(synchronize(np))

    def test_none_xp_is_a_silent_noop(self):
        """Call sites read ``xp`` off a holder that may not carry one."""
        self.assertIsNone(synchronize(None))

    def test_object_without_cuda_is_a_noop(self):
        self.assertIsNone(synchronize(SimpleNamespace()))


class SynchronizeCudaDispatchTest(unittest.TestCase):
    def _fake_xp(self, calls):
        return SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(
                    deviceSynchronize=lambda: calls.append(1)
                )
            )
        )

    def test_drives_device_synchronize_once(self):
        calls = []
        synchronize(self._fake_xp(calls))
        self.assertEqual(calls, [1])

    def test_each_call_is_its_own_barrier(self):
        calls = []
        xp = self._fake_xp(calls)
        for _ in range(3):
            synchronize(xp)
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
