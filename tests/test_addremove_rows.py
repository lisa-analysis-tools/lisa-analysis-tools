"""addremove in one-walker replica mode: compute_like scatters rows, replays reach every rank."""

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.communication.rowfanout import RowFanout
from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove


def _stub(tag):
    m = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    m.branch_name = "mbh"
    m.gf_move_name = "mbh_pe"
    m._dcga = None
    m.waveform_like_kwargs = {}
    m.waveform_gen = object()
    m.likelihood_fanout = True
    m.row_fanout = None
    m.nwalkers = 1
    m.ntemps = 4
    m.num_repeats = 3
    m.permute_every = 1
    m._fancy_swap_clock = 1
    m.log = []
    m.compute_like_local = lambda coords, idx, _t=tag: np.asarray(coords).sum(axis=1) + _t
    m.compute_acs_like = lambda coords, data_index=None, signal_gen=None, **kw: (
        np.asarray(coords).sum(axis=1) * 10.0 + tag
    )
    m.remove_cold_chain_sources = lambda c: m.log.append(("expose", m._current_leaf, np.asarray(c).copy()))
    m.setup_likelihood_here = lambda c: m.log.append(("setup", m._current_leaf, np.asarray(c).copy()))
    m.add_back_in_cold_chain_sources = lambda c: m.log.append(("fold", m._current_leaf, np.asarray(c).copy()))
    return m


def _world(head_fn, knob=None):
    world = FakeWorld(3, nodes=[0, 0, 0])
    moves = {}

    def fn(rank, comm):
        layout = build_layout(comm, 1, [0, 1], legacy=False)
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return "saver"
        fo = WalkerFanout(fcomm, layout, rank, model=None)
        move = _stub(100.0 * rank)
        move.row_fanout = RowFanout(fo, move)
        if knob is not None:
            move.likelihood_fanout = knob
        moves[rank] = move
        if layout.role_of(rank) == RankRole.HEAD:
            fo.enter_stage("pe", "pe")
            try:
                return head_fn(move)
            finally:
                fo.stop()
        return ComputeService(fcomm, layout, rank, registry={("pe", "mbh_pe"): move}).serve()

    return world.run(fn), moves


COORDS = np.arange(10.0).reshape(5, 2)
IDX = np.zeros(5, dtype=np.int32)


class AddremoveRowsTest(unittest.TestCase):
    def test_compute_like_scatters_rows_contiguously(self):
        out, _ = _world(lambda m: (m.compute_like(COORDS, IDX), m._last_d_h))
        ll, d_h = out[0]
        tags = np.array([0, 0, 0, 100, 100], dtype=float)  # array_split(5, 2) -> 3 + 2
        np.testing.assert_array_equal(ll, COORDS.sum(axis=1) + tags)
        self.assertEqual(d_h.shape, (5,))
        self.assertTrue(np.all(np.isnan(d_h)))  # the base scorer has no side outputs

    def test_knob_off_scores_everything_on_the_head(self):
        out, _ = _world(lambda m: m.compute_like(COORDS, IDX), knob=False)
        np.testing.assert_array_equal(out[0], COORDS.sum(axis=1))

    def test_check_batch_scatters_through_the_container_path(self):
        out, _ = _world(lambda m: m.compute_check_like(COORDS, IDX))
        tags = np.array([0, 0, 0, 100, 100], dtype=float)
        np.testing.assert_array_equal(out[0], COORDS.sum(axis=1) * 10.0 + tags)

    def test_replays_reach_every_rank_in_order(self):
        c = np.array([[1.0, 2.0]])

        def head(m):
            m._replay_cold_chain("expose", c, 7)
            m._replay_cold_chain("setup", c, 7)
            m._replay_cold_chain("fold", c, 7)
            return m.log

        out, moves = _world(head)
        kinds = [(k, leaf) for k, leaf, _ in out[0]]
        self.assertEqual(kinds, [("expose", 7), ("setup", 7), ("fold", 7)])
        self.assertEqual([(k, leaf) for k, leaf, _ in moves[1].log], kinds)
        np.testing.assert_array_equal(moves[1].log[0][2], c)
        with self.assertRaises(ValueError):
            moves[1]._apply_cold_chain_replay({"kind": "bogus", "coords": c, "leaf": 0})

    def test_knob_off_still_replays_on_every_rank(self):
        """F1: {PREFIX}_LIKELIHOOD_FANOUT=0 must not stop the cold-chain replay.

        The knob only routes SCORING to the head; expose/setup/fold must
        still reach every replica so their residuals stay aligned.
        """
        c = np.array([[1.0, 2.0]])

        def head(m):
            m._replay_cold_chain("expose", c, 7)
            return m.log

        out, moves = _world(head, knob=False)
        self.assertEqual([(k, leaf) for k, leaf, _ in out[0]], [("expose", 7)])
        self.assertEqual([(k, leaf) for k, leaf, _ in moves[1].log], [("expose", 7)])
        np.testing.assert_array_equal(moves[1].log[0][2], c)

    def test_fancy_swap_never_fires_with_one_walker(self):
        m = _stub(0.0)
        self.assertFalse(m._fancy_swap_fires(m.num_repeats - 1))
        m.nwalkers = 2
        self.assertTrue(m._fancy_swap_fires(m.num_repeats - 1))
        self.assertFalse(m._fancy_swap_fires(0))

    def test_single_process_is_the_local_scorer(self):
        m = _stub(5.0)
        np.testing.assert_array_equal(m.compute_like(COORDS, IDX), COORDS.sum(axis=1) + 5.0)
        m._replay_cold_chain("expose", COORDS[:1], 2)
        self.assertEqual(m.log[0][:2], ("expose", 2))


if __name__ == "__main__":
    unittest.main()
