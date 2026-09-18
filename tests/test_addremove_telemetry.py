"""MBH / EMRI / SOBBH record acceptance telemetry on their sub-state.

``PerLeafLadderState`` declares ``in_model_proposed`` /
``in_model_accepted``, per-rung-pair swap counters and per-leaf
``log_like`` / ``log_prior``, and the HDF backend persists all of them --
but nothing in ``addremovemove`` ever wrote them, so every MBH / EMRI /
SOBBH store on disk recorded zeros. Found 2026-09-17 in the 4-GPU store:
53 iterations, every one of those fields exactly zero, while the same
fields were populated for gb, vgb, psd and galfor. With no acceptance
rate for the three branches there is no way to tune the inner moves or
to see a collapse, and SOBBH alone is 43% of an iteration.

These tests pin the writer (``_record_leaf_telemetry``) and the property
that makes it correct under the walker-block fan-out: the counters are
DELTAS that ``merge_walkers`` sums across ranks, while ``log_like`` and
``log_prior`` are merged by walker column.
"""

import types
import unittest

import numpy as np

from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
from lisatools.globalfit.state import SOBBHState

NT, NW, NLEAVES, NDIM = 4, 6, 3, 11


def _sub(ntemps=NT, nwalkers=NW, nleaves=NLEAVES):
    sub = SOBBHState(None, betas_all=np.tile(
        1.0 / (1.5 ** np.arange(ntemps)), (nleaves, 1)))
    sub.initialize_tempered(ntemps, nwalkers, nleaves, NDIM)
    return sub


def _move(ntemps=NT, nwalkers=NW):
    move = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name = "sobbh"
    move.ntemps = ntemps
    move.nwalkers = nwalkers
    move._fanout_swap_tally = {}
    return move


def _state(sub):
    return types.SimpleNamespace(sub_states={"sobbh": sub})


class RecordLeafTelemetryTest(unittest.TestCase):
    def test_counters_lnl_and_swaps_are_written(self):
        sub, move = _sub(), _move()
        move._fanout_swap_tally[1] = (
            np.array([3.0, 1.0, 0.0]), np.array([12.0, 12.0, 12.0]))
        proposed = np.full(NT, 20 * NW, dtype=np.int64)
        accepted = np.array([11, 24, 37, 58], dtype=np.int64)
        ll = np.arange(NT * NW, dtype=float).reshape(NT, NW)
        lp = -ll

        move._record_leaf_telemetry(_state(sub), 1, proposed, accepted, ll, lp)

        np.testing.assert_array_equal(sub.in_model_proposed[1], proposed)
        np.testing.assert_array_equal(sub.in_model_accepted[1], accepted)
        np.testing.assert_array_equal(sub.swaps_accepted[1], [3, 1, 0])
        np.testing.assert_array_equal(sub.swaps_proposed[1], [12, 12, 12])
        np.testing.assert_allclose(sub.log_like[1], ll)
        np.testing.assert_allclose(sub.log_prior[1], lp)
        # untouched leaves stay zero
        self.assertEqual(sub.in_model_proposed[0].sum(), 0)
        self.assertEqual(sub.in_model_proposed[2].sum(), 0)

    def test_counters_accumulate_but_lnl_is_replaced(self):
        """A leaf visited twice in one propose sums counts, not likelihoods."""
        sub, move = _sub(), _move()
        p = np.full(NT, 5, dtype=np.int64)
        a = np.ones(NT, dtype=np.int64)
        first = np.full((NT, NW), 1.0)
        second = np.full((NT, NW), 2.0)
        move._record_leaf_telemetry(_state(sub), 0, p, a, first, first)
        move._record_leaf_telemetry(_state(sub), 0, p, a, second, second)
        np.testing.assert_array_equal(sub.in_model_proposed[0], 2 * p)
        np.testing.assert_array_equal(sub.in_model_accepted[0], 2 * a)
        np.testing.assert_allclose(sub.log_like[0], second)

    def test_no_swap_tally_leaves_swap_counters_alone(self):
        sub, move = _sub(), _move()
        move._record_leaf_telemetry(
            _state(sub), 0, np.ones(NT, dtype=np.int64),
            np.zeros(NT, dtype=np.int64), np.zeros((NT, NW)), np.zeros((NT, NW)))
        self.assertEqual(sub.swaps_proposed.sum(), 0)

    def test_missing_or_bare_sub_state_is_a_no_op(self):
        move = _move()
        p = np.ones(NT, dtype=np.int64)
        z = np.zeros((NT, NW))
        move._record_leaf_telemetry(types.SimpleNamespace(sub_states={}), 0, p, p, z, z)
        move._record_leaf_telemetry(types.SimpleNamespace(sub_states=None), 0, p, p, z, z)
        bare = SOBBHState(None, betas_all=np.ones((NLEAVES, NT)))
        move._record_leaf_telemetry(_state(bare), 0, p, p, z, z)  # not initialized


class FanoutMergeTest(unittest.TestCase):
    """What the head ends up with after two walker blocks report."""

    def test_counters_sum_over_blocks_and_lnl_lands_in_its_columns(self):
        block = NW // 2
        full = _sub()
        parts = []
        for w0 in (0, block):
            part = _sub(nwalkers=block)
            move = _move(nwalkers=block)
            move._fanout_swap_tally[2] = (
                np.array([1.0, 1.0, 1.0]), np.array([2.0 * block] * 3))
            # each rank proposes num_repeats * ITS OWN block width per rung
            proposed = np.full(NT, 10 * block, dtype=np.int64)
            accepted = np.full(NT, 1 + w0, dtype=np.int64)
            ll = np.full((NT, block), float(w0 + 1))
            move._record_leaf_telemetry(
                _state(part), 2, proposed, accepted, ll, -ll)
            parts.append((part, w0))

        for part, w0 in parts:
            full.merge_walkers(part, w0, w0 + block)

        # proposals pool to the ENSEMBLE count: num_repeats * nwalkers
        np.testing.assert_array_equal(full.in_model_proposed[2], 10 * NW)
        np.testing.assert_array_equal(full.in_model_accepted[2], 1 + (1 + block))
        np.testing.assert_array_equal(full.swaps_accepted[2], [2, 2, 2])
        np.testing.assert_array_equal(full.swaps_proposed[2], [2 * NW] * 3)
        # lnL: each rank's own columns, nothing overwritten
        np.testing.assert_allclose(full.log_like[2, :, :block], 1.0)
        np.testing.assert_allclose(full.log_like[2, :, block:], 1.0 + block)
        # a branch-wide acceptance rate is now computable
        rate = full.in_model_accepted[2] / full.in_model_proposed[2]
        self.assertTrue(np.all((rate > 0) & (rate < 1)))


if __name__ == "__main__":
    unittest.main()
