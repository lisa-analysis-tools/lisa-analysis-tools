"""Per-leaf scoring-cost telemetry on ``SOBBHChunkedLikeMove.compute_like``.

2026-09-16: production measures 190.5 s/leaf = 25 x ~7.35 s compute_like
calls = 61 ms/row against the kernel's own in-code reference of 2.78 ms/row
(job 373, same Tobs / m_band_half_width / shard count) -- a ~22x per-row
regression with log-silent leaf windows. These tests pin the telemetry that
makes the next production log answer it: per-leaf-window call/row counts and
a host-stage vs kernel wall split, flushed as one [SOBBH_LL_TIMING] line at
the next ``setup_likelihood_here`` (= next leaf).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.sobbhspecialmove import SOBBHChunkedLikeMove
from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

LOGGER = "lisatools.globalfit.moves.sobbhspecialmove"


def _bare_move(n_walkers=4):
    """Minimal instance without the heavy ctor: only what compute_like and
    the stats plumbing touch."""
    inst = object.__new__(SOBBHChunkedLikeMove)
    inst._dcga = None
    inst._exposed_offset = np.zeros(n_walkers)
    inst._f_band_lo = 0.0
    inst._f_band_hi = 1.0
    inst._kernel_ll = mock.Mock(
        side_effect=lambda params, idx: (
            np.zeros(len(params)), np.zeros(len(params)),
            np.zeros(len(params))))
    return inst


def _rows(n, f_low=0.5):
    """(n, 11) waveform-basis rows; f_low sits at input column 6."""
    out = np.ones((n, 11))
    out[:, 6] = f_low
    return out


class LLStatsAccumulationTest(unittest.TestCase):
    def test_two_calls_accumulate_calls_rows_and_walls(self):
        inst = _bare_move()
        inst.compute_like(_rows(6), np.zeros(6, dtype=int))
        inst.compute_like(_rows(4), np.zeros(4, dtype=int))
        st = inst._ll_stats
        self.assertEqual(st["calls"], 2)
        self.assertEqual(st["rows"], 10)
        self.assertGreater(st["total_s"], 0.0)
        self.assertGreaterEqual(st["host_s"], 0.0)
        self.assertGreaterEqual(st["kernel_s"], 0.0)
        self.assertGreaterEqual(
            st["total_s"], st["host_s"] + st["kernel_s"] - 1e-9)

    def test_invalid_rows_still_counted(self):
        inst = _bare_move()
        inst.compute_like(_rows(5, f_low=5.0), np.zeros(5, dtype=int))
        self.assertEqual(inst._ll_stats["calls"], 1)
        self.assertEqual(inst._ll_stats["rows"], 5)
        inst._kernel_ll.assert_not_called()


class LLStatsFlushTest(unittest.TestCase):
    def test_flush_logs_one_line_and_resets(self):
        inst = _bare_move()
        inst.compute_like(_rows(6), np.zeros(6, dtype=int))
        inst.compute_like(_rows(6), np.zeros(6, dtype=int))
        with self.assertLogs(LOGGER, level="INFO") as cm:
            inst._flush_ll_stats()
        joined = "\n".join(cm.output)
        self.assertIn("LL_TIMING", joined)
        self.assertIn("calls=2", joined)
        self.assertIn("rows=12", joined)
        self.assertEqual(inst._ll_stats["calls"], 0)

    def test_flush_with_no_calls_is_silent(self):
        inst = _bare_move()
        with self.assertNoLogs(LOGGER, level="INFO"):
            inst._flush_ll_stats()

    def test_setup_likelihood_here_flushes_previous_leaf(self):
        inst = _bare_move()
        inst.acs = SimpleNamespace(likelihood=lambda: np.zeros(4))
        inst._flush_ll_stats = mock.Mock()
        with mock.patch.object(
                ResidualAddOneRemoveOneMove, "setup_likelihood_here",
                return_value=None) as base_setup:
            inst.setup_likelihood_here(np.zeros((4, 11)))
        inst._flush_ll_stats.assert_called_once()
        base_setup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
