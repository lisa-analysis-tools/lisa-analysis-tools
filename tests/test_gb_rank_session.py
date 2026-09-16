"""The GB per-propose rank session and its three ``gf_serve`` commands.

Skeleton-level: a bare ``GBSpecialBase`` carrying only the attributes the
rank plumbing reads, with the three device-binding calls
(``pin_main_device`` / ``_configure_domain`` / ``_bind_parent_acs``)
patched out. Nothing here needs a GPU, a build or a ``BandSorter`` -- the
neutral path is exactly the "head decided this block has nothing to do"
branch, which must still reply with correctly shaped arrays so the
command count stays symmetric across ranks.
"""

import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as gbs
from lisatools.globalfit.moves.globalfitmove import GlobalFitMove
from lisatools.globalfit.state import GBState, GFState

NTEMPS, B, NLEAVES, NDIM = 2, 3, 2, 8
BAND_EDGES = np.linspace(1e-3, 2e-3, 5)
NUM_BANDS = len(BAND_EDGES) - 1


class _FakeFanout:
    def __init__(self, single, is_head):
        self.single = single
        self.is_head = is_head


class _Curr:
    def __init__(self, fanout, rank):
        self.fanout = fanout
        self.rank = rank


class _FakeACS:
    gpus = None


class _FakeModel:
    def __init__(self):
        self.analysis_container_arr = _FakeACS()


def make_move():
    """A skeleton GB move with only the attributes the rank plumbing reads."""
    move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
    move.branch_name = "gb"
    move.name = "gb_test"
    # ``xp`` / ``backend`` are derived properties (deepcopy-safety rule)
    move._backend_name = "lisatools_cpu"
    move.band_edges = BAND_EDGES
    move.num_bands = NUM_BANDS
    move.nwalkers = 4
    move.ntemps = 2
    move.time = 3
    move.num_proposals = 7
    move.is_rj_prop = False
    move._reseed_firing = False
    move.temper_vertical = False
    move._cap_leaf_cap = None
    move._band_leaf_cap = None
    move._rj_band_shutoff = None
    move._gb_session = None
    move.fanout = None
    move.gf_rank = None
    move.eigen_store_path = "x.h5"
    move.mempool = gbs._NoOpMempool()
    # the three heavy binding calls are no-ops on a skeleton
    move._configure_domain = lambda acs: None
    move._bind_parent_acs = lambda acs: None
    return move


def make_slice(nwalkers=B):
    """What ``slice_state(state, w0, w1, sub_states=['gb'])`` hands a rank."""
    rng = np.random.default_rng(11)
    coords = {"gb": rng.standard_normal((NTEMPS, nwalkers, NLEAVES, NDIM))}
    inds = {"gb": np.ones((NTEMPS, nwalkers, NLEAVES), dtype=bool)}
    part = GFState(
        coords,
        inds=inds,
        log_like=rng.standard_normal((NTEMPS, nwalkers)),
        log_prior=rng.standard_normal((NTEMPS, nwalkers)),
        sub_state_bases={"gb": GBState},
    )
    part.sub_states["gb"].pull_from_main(part, "gb")
    return part


def neutral_payload(**over):
    payload = {
        "ntemps": NTEMPS,
        "nwalkers": B,
        "neutral": True,
        "state": make_slice(),
        "clock_vals": {
            "time": 11,
            "num_proposals": 5,
            "reseed_firing": True,
            "temper_vertical": True,
            "branch_propose_count": 9,
        },
        "tables": {"cap_leaf_cap": None, "band_leaf_cap": None, "rj_band_shutoff": None},
        "rank_seed": 4242,
        "directive": {"epoch": None, "ctr_table": False},
        "band_temps": np.ones((NUM_BANDS, NTEMPS)),
        "keep_all_inds": True,
        "scan_schedule": None,
        "engine_ntemps": NTEMPS,
    }
    payload.update(over)
    return payload


class RankBlockTest(unittest.TestCase):
    def setUp(self):
        self.move = make_move()
        self.model = _FakeModel()
        self._census = dict(gbs.GBSpecialBase._branch_propose_counts)
        self.addCleanup(self._restore_census)
        patcher = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _restore_census(self):
        gbs.GBSpecialBase._branch_propose_counts.clear()
        gbs.GBSpecialBase._branch_propose_counts.update(self._census)

    def test_enter_sets_the_block_and_exit_restores_it(self):
        payload = neutral_payload()
        saved = self.move._enter_rank_block(payload, {"seq": 3, "call_index": 1}, self.model)
        self.assertEqual(self.move.nwalkers, B)
        self.assertEqual(self.move.ntemps, NTEMPS)
        self.assertEqual(self.move.time, 11)
        self.assertEqual(self.move.num_proposals, 5)
        self.assertTrue(self.move._reseed_firing)
        self.assertTrue(self.move.temper_vertical)
        self.assertEqual(self.move._rank_rng_seed, 4242)
        self.assertIsNone(self.move._temper_rng)
        self.assertEqual(gbs.GBSpecialBase._branch_propose_counts["gb"], 9)

        self.move._exit_rank_block(saved)
        self.assertEqual(self.move.nwalkers, 4)
        self.assertEqual(self.move.ntemps, 2)
        self.assertEqual(self.move.time, 3)
        self.assertEqual(self.move.num_proposals, 7)
        self.assertFalse(self.move._reseed_firing)
        self.assertFalse(self.move.temper_vertical)
        self.assertIsNone(self.move._rank_rng_seed)

    def test_enter_takes_the_block_width_from_the_slice(self):
        payload = neutral_payload()
        payload.pop("nwalkers")
        saved = self.move._enter_rank_block(payload, {"seq": 1, "call_index": 1}, self.model)
        self.assertEqual(self.move.nwalkers, B)
        self.move._exit_rank_block(saved)

    def test_exit_restores_the_shipped_tables(self):
        cap = np.arange(NUM_BANDS)
        shut = np.zeros(NUM_BANDS, dtype=bool)
        payload = neutral_payload(
            tables={"cap_leaf_cap": cap, "band_leaf_cap": cap, "rj_band_shutoff": shut}
        )
        saved = self.move._enter_rank_block(payload, {"seq": 1, "call_index": 1}, self.model)
        self.assertIs(self.move._cap_leaf_cap, cap)
        self.assertIs(self.move._rj_band_shutoff, shut)
        self.move._exit_rank_block(saved)
        self.assertIsNone(self.move._cap_leaf_cap)
        self.assertIsNone(self.move._band_leaf_cap)
        self.assertIsNone(self.move._rj_band_shutoff)


class InstallWalkerFanoutTest(unittest.TestCase):
    def test_single_rank_is_inactive_and_keeps_the_sidecar(self):
        move = make_move()
        move.install_walker_fanout(_Curr(_FakeFanout(single=True, is_head=True), 0))
        self.assertFalse(move.fanout_active)
        self.assertEqual(move.eigen_store_path, "x.h5")
        self.assertEqual(move.gf_rank, 0)

    def test_no_fanout_at_all_is_inactive(self):
        move = make_move()
        move.install_walker_fanout(object())
        self.assertIsNone(move.fanout)
        self.assertFalse(move.fanout_active)

    def test_worker_rank_clears_the_eigen_sidecar(self):
        move = make_move()
        move.install_walker_fanout(_Curr(_FakeFanout(single=False, is_head=False), 2))
        self.assertTrue(move.fanout_active)
        self.assertIsNone(move.eigen_store_path)
        self.assertEqual(move.gf_rank, 2)

    def test_head_keeps_the_eigen_sidecar(self):
        move = make_move()
        move.install_walker_fanout(_Curr(_FakeFanout(single=False, is_head=True), 0))
        self.assertTrue(move.fanout_active)
        self.assertEqual(move.eigen_store_path, "x.h5")


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.move = make_move()
        self.model = _FakeModel()

    def test_ops_constant(self):
        self.assertEqual(
            gbs.GB_OPS, ("gb_run_proposal", "gb_run_tempering", "gb_finish")
        )

    def test_gb_moves_are_no_longer_fanout_unready(self):
        # run.py::_fanout_unready_moves keys off exactly this identity
        self.assertIsNot(gbs.GBSpecialBase.gf_serve, GlobalFitMove.gf_serve)

    def test_unknown_op_raises(self):
        with self.assertRaises(ValueError):
            self.move.gf_serve("nope", {}, {"seq": 1, "call_index": 1}, self.model)

    def test_tempering_without_a_session_is_stale(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.move.gf_serve(
                "gb_run_tempering", {"session": 99}, {"seq": 1, "call_index": 1}, self.model
            )
        self.assertIn("stale", str(ctx.exception))

    def test_finish_without_a_session_is_stale(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.move.gf_serve(
                "gb_finish", {"session": 99}, {"seq": 1, "call_index": 1}, self.model
            )
        self.assertIn("stale", str(ctx.exception))

    def test_setup_from_directive_is_a_no_op_on_the_base_class(self):
        self.assertIsNone(self.move._setup_from_directive({"epoch": None, "ctr_table": False}))
        self.assertIsNone(self.move._setup_from_directive({"epoch": 3, "ctr_table": True}))
        self.assertIsNone(self.move._setup_from_directive(None))


class NeutralBlockTest(unittest.TestCase):
    """A block the HEAD marked neutral still answers all three commands."""

    def setUp(self):
        self.move = make_move()
        self.model = _FakeModel()
        self._census = dict(gbs.GBSpecialBase._branch_propose_counts)
        self.addCleanup(self._restore_census)
        patcher = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clock = {"seq": 7, "call_index": 2}
        self.token = (7, 2)

    def _restore_census(self):
        gbs.GBSpecialBase._branch_propose_counts.clear()
        gbs.GBSpecialBase._branch_propose_counts.update(self._census)

    def test_the_three_commands_round_trip(self):
        rep = self.move.gf_serve(
            "gb_run_proposal", neutral_payload(), self.clock, self.model
        )
        sess = self.move._gb_session
        self.assertIsNotNone(sess)
        self.assertEqual(sess.token, self.token)
        self.assertIsNone(sess.band_sorter)
        # the head's N is restored the moment the command returns
        self.assertEqual(self.move.nwalkers, 4)

        self.assertEqual(np.shape(rep["log_like_cold"]), (B,))
        self.assertEqual(np.shape(rep["prop_counts"]), (2, NTEMPS, NUM_BANDS))
        self.assertEqual(np.shape(rep["acc_counts"]), (2, NTEMPS, NUM_BANDS))
        self.assertEqual(np.shape(rep["cold_counts"]), (2, 2, B, NUM_BANDS))
        self.assertEqual(np.shape(rep["start_diffs"]), (B,))
        self.assertEqual(list(rep["alive_per_temp"]), [0] * NTEMPS)
        self.assertEqual(rep["n_alive"], 0)
        self.assertEqual(rep["drift"], 0.0)
        self.assertIsNone(rep["rj_at_cap"])
        self.assertIn("timing", rep)
        self.assertTrue(np.all(np.asarray(rep["prop_counts"]) == 0))

        rep2 = self.move.gf_serve(
            "gb_run_tempering",
            neutral_payload(session=self.token, tmp_start=0),
            {"seq": 7, "call_index": 3},
            self.model,
        )
        self.assertEqual(np.shape(rep2["log_like_cold"]), (B,))
        self.assertEqual(np.shape(rep2["band_swaps_accepted"]), (NUM_BANDS, NTEMPS - 1))
        self.assertEqual(np.shape(rep2["band_swaps_proposed"]), (NUM_BANDS, NTEMPS - 1))
        self.assertEqual(np.shape(rep2["ll_change_sum_temp_cold"]), (B,))
        self.assertEqual(rep2["drift"], 0.0)
        self.assertIsNone(rep2["census"])
        self.assertIsNotNone(self.move._gb_session)

        rep3 = self.move.gf_serve(
            "gb_finish",
            neutral_payload(session=self.token, want_cap_stats=False),
            {"seq": 7, "call_index": 4},
            self.model,
        )
        self.assertEqual(np.shape(rep3["alive_coords"]), (0, NDIM))
        self.assertEqual(np.shape(rep3["alive_twl"]), (0, 3))
        self.assertEqual(np.shape(rep3["d_h"]), (B, NLEAVES))
        self.assertEqual(np.shape(rep3["h_h"]), (B, NLEAVES))
        self.assertEqual(np.shape(rep3["band_counts"]), (NTEMPS, B, NUM_BANDS))
        self.assertEqual(np.shape(rep3["log_like_final"]), (B,))
        self.assertIsNone(rep3["cap_stats"])
        self.assertEqual(rep3["fstat_ctr_fallback_rows"], 0)
        for key in ("band_dof", "rj_split", "replace_census", "timing"):
            self.assertIn(key, rep3)
        # the session is gone and the head's block width is back
        self.assertIsNone(self.move._gb_session)
        self.assertEqual(self.move.nwalkers, 4)

    def test_a_wrong_session_token_is_stale(self):
        self.move.gf_serve("gb_run_proposal", neutral_payload(), self.clock, self.model)
        with self.assertRaises(RuntimeError) as ctx:
            self.move.gf_serve(
                "gb_run_tempering",
                neutral_payload(session=(1, 1)),
                {"seq": 8, "call_index": 1},
                self.model,
            )
        self.assertIn("stale", str(ctx.exception))
        self.move._gb_session = None


if __name__ == "__main__":
    unittest.main()
