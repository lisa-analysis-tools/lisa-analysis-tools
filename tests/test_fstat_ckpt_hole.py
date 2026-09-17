"""A cleared/rewritten F-stat checkpoint payload must not kill the sweep.

2026-09-16: two jobs fitting the same epoch in the same directory -- one
finished its comb scan and ran ``ckpt_clear`` while the other was mid-sweep;
the second one then hit ``ckpt_save``'s hole refusal and aborted the job.
The sweep's result is in memory, so a hole only costs the checkpoint: the
write is skipped with a warning, and a later resume restarts the sweep.
"""

import os
import tempfile
import unittest

import numpy as np

from lisatools.sampling import fstat_gridfit as fg


class CheckpointHoleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ckpt = os.path.join(self._tmp.name, "parts", "comb_nsky8")
        self.fp = "fingerprint-1"

    def _pair(self):
        return (self.ckpt + fg._CKPT_HEADER_SUFFIX,
                self.ckpt + fg._CKPT_PAYLOAD_SUFFIX)

    def test_a_cleared_payload_skips_the_write_with_a_warning(self):
        n = 40
        # rows [0, 10) checkpointed normally
        fg.ckpt_save(self.ckpt, np.arange(10, dtype=float), 10, n, self.fp,
                     start=0)
        header, dat = self._pair()
        self.assertEqual(fg._ckpt_payload_rows(dat), 10)
        # another process clears the parts under us
        os.remove(dat)
        # the next append starts at 10 -> no rows behind it -> hole
        with self.assertLogs(fg.logger, level="WARNING") as cm:
            fg.ckpt_save(self.ckpt, np.arange(10, 20, dtype=float), 20, n,
                         self.fp, start=10)
        self.assertTrue(any("skipping this checkpoint write" in m
                            for m in cm.output))
        # nothing was written: no payload, header still at the old cursor
        self.assertFalse(os.path.exists(dat))
        with np.load(header) as d:
            self.assertEqual(int(d["done"]), 10)

    def test_a_later_resume_restarts_the_sweep_instead_of_trusting_the_header(self):
        n = 40
        fg.ckpt_save(self.ckpt, np.arange(10, dtype=float), 10, n, self.fp,
                     start=0)
        _, dat = self._pair()
        os.remove(dat)
        with self.assertLogs(fg.logger, level="WARNING"):
            fg.ckpt_save(self.ckpt, np.arange(10, 20, dtype=float), 20, n,
                         self.fp, start=10)
        F, done = fg.ckpt_resume(self.ckpt, n, self.fp)
        # "no usable checkpoint" is (None, 0): the sweep restarts from row 0
        self.assertEqual(done, 0)
        self.assertIsNone(F)

    def test_a_longer_payload_is_still_truncated_and_appended(self):
        # the pre-existing "run died before its header landed" repair
        n = 40
        fg.ckpt_save(self.ckpt, np.arange(10, dtype=float), 10, n, self.fp,
                     start=0)
        _, dat = self._pair()
        with open(dat, "ab") as f:
            np.full(5, -1.0).tofile(f)          # 5 orphan rows past the header
        fg.ckpt_save(self.ckpt, np.arange(10, 20, dtype=float), 20, n, self.fp,
                     start=10)
        F, done = fg.ckpt_resume(self.ckpt, n, self.fp)
        self.assertEqual(done, 20)
        np.testing.assert_array_equal(np.asarray(F)[:20], np.arange(20.0))


if __name__ == "__main__":
    unittest.main()
