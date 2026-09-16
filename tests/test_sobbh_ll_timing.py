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

import time
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


# ----------------------------------------------------------------------
# _kernel_ll sub-spans (2026-09-16 follow-up). The first telemetry put
# 100.0% of the leaf wall inside _kernel_ll with host_stage == 0.00 and a
# NEGATIVE marginal row cost (288 rows @ 0.8 s in job 373 vs 54 rows @
# 3.48 s now), i.e. the expensive part of a call is row-INDEPENDENT. These
# tests pin the breakdown that names it: the comp's own per-call spans
# (static-geometry re-assert, layer grouping, comp-group wrap construction,
# kernel launch) plus the move's shard dispatch, the post-launch device
# sync, and the D2H pulls.
# ----------------------------------------------------------------------

_STUB_SPANS = {
    "stage": 0.01, "geom": 0.02, "wrap": 0.003, "launch": 0.5, "total": 0.533,
    "n_groups": 7,
}


class _StubComp:
    """Comp stand-in exposing the chunked-het per-call span record."""

    def __init__(self, spans=_STUB_SPANS, with_spans=True):
        self._spans = spans
        self._with_spans = with_spans
        self.calls = 0
        self.d_h_out = np.zeros(0)
        self.h_h_out = np.zeros(0)
        if with_spans:
            self.last_call_spans = dict(spans)

    def get_ll_wdm(self, params, holder, data_index=None, noise_index=None,
                   m_band_half_width=1):
        self.calls += 1
        n = len(params)
        self.d_h_out = np.zeros(n)
        self.h_h_out = np.zeros(n)
        if self._with_spans:
            self.last_call_spans = dict(self._spans)
        else:
            # measurable wall so the "unattributed" bucket is testable
            time.sleep(0.002)
        return np.zeros(n)


def _kernel_move(n_walkers=4, comp=None):
    """Bare move that runs the REAL _kernel_ll against a stub comp."""
    inst = object.__new__(SOBBHChunkedLikeMove)
    inst._dcga = None
    inst._exposed_offset = np.zeros(n_walkers)
    inst._f_band_lo = 0.0
    inst._f_band_hi = 1.0
    inst.m_band_half_width = 3
    inst.comp = comp if comp is not None else _StubComp()
    inst.acs = SimpleNamespace(linear_data_arr=[object()], xp=np)
    return inst


class KernelSubSpanTest(unittest.TestCase):
    def test_comp_spans_accumulate_across_calls(self):
        inst = _kernel_move()
        inst.compute_like(_rows(6), np.zeros(6, dtype=int))
        inst.compute_like(_rows(4), np.zeros(4, dtype=int))
        st = inst._ll_stats
        self.assertAlmostEqual(st["geom_s"], 2 * _STUB_SPANS["geom"], places=9)
        self.assertAlmostEqual(st["launch_s"], 2 * _STUB_SPANS["launch"],
                               places=9)
        self.assertAlmostEqual(st["wrap_s"], 2 * _STUB_SPANS["wrap"], places=9)
        self.assertAlmostEqual(st["stage_s"], 2 * _STUB_SPANS["stage"],
                               places=9)

    def test_shard_calls_counted(self):
        inst = _kernel_move()
        inst.compute_like(_rows(6), np.zeros(6, dtype=int))
        self.assertEqual(inst._ll_stats["shard_calls"], 1)
        self.assertEqual(inst.comp.calls, 1)

    def test_layer_group_count_accumulates(self):
        """Kernel work scales with GROUPS (m-band x data_index), not rows."""
        inst = _kernel_move()
        inst.compute_like(_rows(6), np.zeros(6, dtype=int))
        inst.compute_like(_rows(4), np.zeros(4, dtype=int))
        self.assertEqual(inst._ll_stats["groups"], 2 * _STUB_SPANS["n_groups"])

    def test_move_side_spans_are_recorded(self):
        inst = _kernel_move()
        inst.compute_like(_rows(6), np.zeros(6, dtype=int))
        st = inst._ll_stats
        for key in ("dispatch_s", "sync_s", "pull_s"):
            self.assertIn(key, st)
            self.assertGreaterEqual(st[key], 0.0)

    def test_comp_without_span_record_charges_the_whole_call_to_launch(self):
        inst = _kernel_move(comp=_StubComp(with_spans=False))
        out = inst.compute_like(_rows(3), np.zeros(3, dtype=int))
        self.assertEqual(out.shape, (3,))
        self.assertEqual(inst._ll_stats["shard_calls"], 1)
        self.assertGreaterEqual(inst._ll_stats["launch_s"], 0.002)
        self.assertEqual(inst._ll_stats["geom_s"], 0.0)

    def test_invalid_only_batch_records_no_shard_call(self):
        inst = _kernel_move()
        inst.compute_like(_rows(4, f_low=5.0), np.zeros(4, dtype=int))
        self.assertEqual(inst._ll_stats["shard_calls"], 0)
        self.assertEqual(inst._ll_stats["launch_s"], 0.0)


class KernelSubSpanFlushTest(unittest.TestCase):
    def test_companion_line_names_the_slowest_component(self):
        inst = _kernel_move()
        inst.compute_like(_rows(6), np.zeros(6, dtype=int))
        with self.assertLogs(LOGGER, level="INFO") as cm:
            inst._flush_ll_stats()
        joined = "\n".join(cm.output)
        self.assertIn("LL_TIMING", joined)
        self.assertIn("launch=", joined)
        self.assertIn("geom=", joined)
        self.assertIn("slowest=launch", joined)
        self.assertIn("groups/call=7.0", joined)
        self.assertIn("ms/group", joined)

    def test_flush_resets_the_sub_spans(self):
        inst = _kernel_move()
        inst.compute_like(_rows(6), np.zeros(6, dtype=int))
        with self.assertLogs(LOGGER, level="INFO"):
            inst._flush_ll_stats()
        self.assertEqual(inst._ll_stats["launch_s"], 0.0)
        self.assertEqual(inst._ll_stats["shard_calls"], 0)


if __name__ == "__main__":
    unittest.main()
