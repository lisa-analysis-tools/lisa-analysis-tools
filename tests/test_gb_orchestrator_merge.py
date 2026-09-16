"""The GB head orchestrator: three commands over the walker blocks, one merge.

Skeleton-level, like ``tests/test_gb_rank_session.py``: a bare
``GBSpecialBase`` carrying only what ``_propose_orchestrated``'s prologue and
merge read, with ``gf_serve`` REPLACED by a deterministic stub. No GB kernel,
no ``BandSorter``, no GPU and no build runs here -- what is under test is the
head: the command order and the session token, the payload contract, and the
merge rules (walker-order concatenation, pooled counters, ONE ladder
adaptation from the SUMMED swap counters, and the neutral-block skip).

The two compute ranks are two THREADS of a ``FakeWorld``, each with its OWN
move object -- the real run has one move instance per process.
"""

import os
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.moves import gbspecialstretch as gbs

from tests.test_gb_rank_session import make_move as make_rank_move
from tests.test_gb_rank_session import neutral_payload
from tests.test_gf_substate_roundtrip import (
    BAND_EDGES,
    BRANCH_SHAPES,
    NTEMPS,
    NUM_BANDS,
    NWALKERS,
    make_state,
)

NLEAVES, NDIM = BRANCH_SHAPES["gb"]
MOVE_NAME = "gb_test"
#: block of each compute rank at ``FakeWorld(3)`` / ``NWALKERS = 4``
BLOCKS = {0: (0, 2), 1: (2, 4)}

#: EXACTLY the reply keys ``_propose_orchestrated`` indexes, per command --
#: read off every ``rep[...]`` / ``rep.get(...)`` access in the merge. The
#: stub below is checked against this list on every reply it builds, and
#: ``ReplyKeyContractTest`` checks the REAL rank bodies against it, so a
#: rename on either side fails loudly instead of merging silence.
ORCHESTRATOR_REPLY_KEYS = {
    "gb_run_proposal": frozenset({
        "prop_counts", "acc_counts", "cold_counts", "alive_per_temp",
        "drift", "rj_at_cap", "log_like_cold",
    }),
    "gb_run_tempering": frozenset({
        "band_swaps_accepted", "band_swaps_proposed", "drift", "log_like_cold",
    }),
    "gb_finish": frozenset({
        "block_coords", "block_inds", "d_h", "h_h", "band_counts",
        "log_like_final", "cap_stats", "band_dof", "fstat_ctr_fallback_rows",
        "timing",
    }),
}


def _checked(op, reply):
    """A stub reply must carry every key the orchestrator reads."""
    missing = ORCHESTRATOR_REPLY_KEYS[op] - set(reply)
    assert not missing, f"{op}: the stub reply is missing {sorted(missing)}"
    return reply


# ----------------------------------------------------------------------
# the deterministic rank stub
# ----------------------------------------------------------------------
def _stub_gf_serve(move, op, payload, clock, model):
    """Answer one command with rank- and walker-tagged synthetic arrays."""
    rank, (w0, w1) = move._test_rank, move._test_block
    B, ntemps, nb = int(payload["nwalkers"]), int(payload["ntemps"]), NUM_BANDS
    move.served.append(op)
    move.payloads.append(payload)
    # which ``model`` object the body was handed, and the propose timer it
    # found bound (a real body rebinds it to the session's -- see below)
    move.seen_models.append(model)
    move.seen_timers.append(getattr(move, "_prop_timer", None))
    move._prop_timer = move._test_session_timer

    # --- the contract EVERY command must satisfy ----------------------
    assert payload["ntemps"] == NTEMPS
    assert set(payload["tables"]) == {
        "cap_leaf_cap", "band_leaf_cap", "rj_band_shutoff"}
    assert "keep_all_inds" in payload
    assert isinstance(payload["rank_seed"], int)
    assert B == w1 - w0

    if op == "gb_run_proposal":
        move.token = (clock["seq"], clock["call_index"])
        move.neutral = bool(payload["neutral"])
        part = payload["state"]
        assert part is not None, "the slice ships for neutral blocks too"
        assert int(part.branches["gb"].nwalkers) == B
        sched = payload["scan_schedule"]
        assert isinstance(sched, tuple) and len(sched) == 2
        # per-walker schedules are sliced to the block; a scalar schedule
        # (one class for the whole ensemble) ships as it was drawn
        assert np.ndim(sched[0]) == 0 or len(sched[0]) == B
        assert payload["directive"] == {"epoch": None, "ctr_table": False}
    else:
        assert payload["session"] == move.token, (
            f"{op}: {payload['session']} != {move.token}")

    if op == "gb_run_proposal":
        if move.neutral:
            return _checked(op, {
                "log_like_cold": np.zeros(B),
                "prop_counts": np.zeros((2, ntemps, nb), dtype=int),
                "acc_counts": np.zeros((2, ntemps, nb), dtype=int),
                "cold_counts": np.zeros((2, 2, B, nb), dtype=int),
                "alive_per_temp": [0] * ntemps,
                "n_alive": 0,
                "start_diffs": np.zeros(B),
                "drift": 0.0,
                "rj_at_cap": None,
                "timing": None,
            })
        return _checked(op, {
            "log_like_cold": 100.0 + w0 + np.arange(B, dtype=float),
            "prop_counts": np.full((2, ntemps, nb), rank + 1, dtype=int),
            "acc_counts": np.ones((2, ntemps, nb), dtype=int),
            # [prop, acc][rj, in-model] cold rows, per LOCAL walker
            "cold_counts": np.stack([
                np.full((2, B, nb), rank + 2, dtype=int),
                np.ones((2, B, nb), dtype=int),
            ]),
            "alive_per_temp": [rank + 1] * ntemps,
            "n_alive": (rank + 1) * ntemps,
            "start_diffs": np.zeros(B),
            "drift": 0.001 * (rank + 1),
            "rj_at_cap": rank,
            "timing": {"stages": {"run_proposal": 1.0 + rank}, "counts": {}},
        })

    if op == "gb_run_tempering":
        if move.neutral:
            return _checked(op, {
                "log_like_cold": np.zeros(B),
                "band_swaps_accepted": np.zeros((nb, ntemps - 1), dtype=int),
                "band_swaps_proposed": np.zeros((nb, ntemps - 1), dtype=int),
                "ll_change_sum_temp_cold": np.zeros(B),
                "drift": 0.0,
                "census": None,
                "timing": None,
            })
        # accepted VARIES with the rung so the pooled ratio really adapts
        acc = (rank + 1) * np.tile(np.arange(1, ntemps), (nb, 1))
        return _checked(op, {
            "log_like_cold": 200.0 + w0 + np.arange(B, dtype=float),
            "band_swaps_accepted": acc.astype(int),
            "band_swaps_proposed": np.full((nb, ntemps - 1), 10, dtype=int),
            "ll_change_sum_temp_cold": np.zeros(B),
            "drift": 0.0,
            "census": None,
            "timing": {"stages": {"run_tempering": 2.0}, "counts": {}},
        })

    if op == "gb_finish":
        walker_tag = float(w0) + np.arange(B, dtype=float)
        cap = None
        if payload["want_cap_stats"]:
            rows = np.repeat(walker_tag[:, None], nb, axis=1)
            cap = {
                "band_lls": rows,
                "lls": rows,
                "dof": 7.0,
                "band_dof": np.full(nb, 5.0),
                "is_cells": False,
            }
        if move.neutral:
            # A NEUTRAL block ships its cap rows too (Task 3 fix round):
            # with no sources they are the residual-window values, so the
            # head's walker-axis concatenation is still N rows.
            return _checked(op, {
                # a neutral block ships no branch block at all
                "block_coords": None,
                "block_inds": None,
                "d_h": np.full((B, NLEAVES), np.nan),
                "h_h": np.full((B, NLEAVES), np.nan),
                "band_counts": np.zeros((ntemps, B, nb), dtype=int),
                "log_like_final": np.zeros(B),
                "cap_stats": cap,
                "fstat_ctr_fallback_rows": 0,
                "band_dof": None if cap is None else np.full(nb, 5.0),
                "rj_split": None,
                "replace_census": None,
                "timing": None,
            })
        # the WHOLE block comes back: one alive source at (temp 0, LOCAL
        # walker 0, leaf 0) and a rank-tagged fill everywhere else, which is
        # what the rejected-birth fill in the dead slots looks like
        block_coords = np.full((ntemps, B, NLEAVES, NDIM), -1.0 - rank)
        block_coords[0, 0, 0] = float(rank)
        block_inds = np.zeros((ntemps, B, NLEAVES), dtype=bool)
        block_inds[0, 0, 0] = True
        return _checked(op, {
            "block_coords": block_coords,
            "block_inds": block_inds,
            "d_h": np.full((B, NLEAVES), 10.0 + rank),
            "h_h": np.full((B, NLEAVES), 20.0 + rank),
            "band_counts": np.full((ntemps, B, nb), rank + 1, dtype=int),
            "log_like_final": 300.0 + walker_tag,
            "cap_stats": cap,
            "fstat_ctr_fallback_rows": rank + 1,
            "band_dof": np.full(nb, 5.0),
            "rj_split": None,
            "replace_census": None,
            "timing": {"stages": {"write_back": 0.5}, "counts": {}},
        })
    raise AssertionError(f"unexpected op {op!r}")


# ----------------------------------------------------------------------
# skeleton move / model
# ----------------------------------------------------------------------
class _Periodic:
    def wrap(self, coords_dict):
        return coords_dict


class _TC:
    """Just enough of eryn's ``TemperatureControl`` for the move's setter."""

    adaptation_lag = 10000.0
    adaptation_time = 100.0
    ntemps = NTEMPS
    nsamplers = 1

    def __init__(self):
        self.swaps_accepted = None
        self.swaps_proposed = None

    @staticmethod
    def compute_log_posterior_tempered(logl, logp):
        return logl + logp


class _FakeACS:
    gpus = None


class _FakeModel:
    def __init__(self, seed=0):
        self.analysis_container_arr = _FakeACS()
        self.random = np.random.RandomState(seed)


def make_move(rank, block, *, use_prior_removal=False, leaf_cap_update=True,
              temper_fire=True, per_walker_schedule=False):
    """A skeleton GB move with a stubbed ``gf_serve``."""
    move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
    move.branch_name = "gb"
    move.name = MOVE_NAME
    move.gf_move_name = MOVE_NAME
    move._backend_name = "lisatools_cpu"
    move.band_edges = BAND_EDGES
    move.num_bands = NUM_BANDS
    move.band_units = 4
    move.band_unit_start_per_walker = per_walker_schedule
    move.band_unit_dir_per_walker = per_walker_schedule
    move.nwalkers = NWALKERS
    move.ntemps = NTEMPS
    move.time = 5
    move.num_proposals = 7
    move.is_rj_prop = True
    move.rj_replace = False
    move.rj_removal_only = False
    move.use_prior_removal = use_prior_removal
    move.rj_proposal_distribution = {"gb": object()}
    move.run_swaps = True
    move.swap_on_in_model = True
    move.temperature_control = _TC()
    move.periodic = _Periodic()
    move.mempool = gbs._NoOpMempool()
    move._reseed_firing = False
    move.temper_vertical = False
    move._cap_leaf_cap = None
    move._band_leaf_cap = None
    move._rj_band_shutoff = np.zeros(NUM_BANDS, dtype=bool)
    move._gb_session = None
    move.fanout = None
    move.gf_rank = rank
    # caps: divisor 1, no stagger -> the cap grid IS the band grid
    move._leaf_cap_enabled = True
    move.leaf_cap_update = leaf_cap_update
    move.leaf_cap_start = 2
    move.cap_divisor = 1
    move.cap_stagger = False
    move.num_cap_cells = NUM_BANDS
    move.cap_edges = BAND_EDGES
    move.cap_overlap_frac = 0.0
    # heavy / head-only machinery the skeleton cannot run
    move._configure_domain = lambda acs: None
    move._bind_parent_acs = lambda acs: None
    move._check_substate_consistency = lambda state, names=None: None
    move.setup = lambda model, branches: None
    move._temper_cadence_fire = lambda: temper_fire
    move._band_shutoff_enabled = lambda: True
    move.shutoff_calls = []
    move._update_band_shutoff = lambda occ, st=None: move.shutoff_calls.append(occ)
    move.cap_calls = []
    move._update_band_leaf_caps = (
        lambda model, st, counts, precomputed=None: move.cap_calls.append(precomputed)
    )
    # the rank side under test is the HEAD's orchestration, not the bodies
    move.served, move.payloads, move.token, move.neutral = [], [], None, False
    move.seen_models, move.seen_timers = [], []
    # what a REAL body leaves bound on ``_prop_timer`` (the rank session's
    # timer); the head must not keep it for its own [GB_TIMING] line
    move._test_session_timer = gbs._ProposeTimer()
    move._test_rank, move._test_block = rank, block
    move.gf_serve = types.MethodType(_stub_gf_serve, move)
    return move


def run_propose(state, *, use_prior_removal=False, leaf_cap_update=True,
                temper_fire=True, per_walker_schedule=False, size=3):
    """Drive ``_propose_orchestrated`` over the compute ranks; return the pieces.

    ``size=1`` is the single-compute-rank world: ``WalkerFanout.run`` is a
    direct call and ``propose`` only reaches the orchestrator when
    ``GB_PROPOSE_ORCHESTRATE=1`` is set by the caller.
    """
    world = FakeWorld(size)
    moves = {}
    gpu_pool = [0, 1] if size >= 3 else [0]

    def fn(rank, comm):
        layout = build_layout(comm, NWALKERS, gpu_pool, legacy=False)
        fcomm = layout.make_fanout_comm(comm)
        role = layout.role_of(rank)
        if role == RankRole.SAVER:
            return "saver-idle"
        move = make_move(
            rank, layout.block_of(rank), use_prior_removal=use_prior_removal,
            leaf_cap_update=leaf_cap_update, temper_fire=temper_fire,
            per_walker_schedule=per_walker_schedule,
        )
        moves[rank] = move
        model = _FakeModel(seed=0)
        move._test_propose_model = model
        if role == RankRole.HEAD:
            # run.py binds ``fanout.model`` ONCE, at setup: a DISTINCT object
            # here proves the head's own body runs against propose's ``model``
            fanout_model = _FakeModel(seed=99)
            move._test_fanout_model = fanout_model
            move.fanout = WalkerFanout(fcomm, layout, rank, model=fanout_model)
            move.fanout.enter_stage("gb", "gb_kind")
            try:
                return move.propose(model, state)
            finally:
                move.fanout.stop()
        service = ComputeService(
            fcomm, layout, rank, registry={MOVE_NAME: move}, model=model
        )
        service.serve()
        return "worker-done"

    results = world.run(fn)
    return results[0], moves


def expected_ladder(band_temps, acc, prop, *, times):
    """``_adapt_band_temps`` applied ``times`` times on a fresh skeleton."""
    helper = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
    helper._backend_name = "lisatools_cpu"
    helper.time = 5
    helper.temperature_control = _TC()
    out = np.array(band_temps, copy=True)
    for _ in range(times):
        helper._adapt_band_temps(out, np.asarray(acc), np.asarray(prop))
    return out


class GBOrchestratorMergeTest(unittest.TestCase):
    def setUp(self):
        if gbs.cp is not np:  # pragma: no cover - GPU box
            self.skipTest("the ladder helper runs on the host array module")
        self.state = make_state(np.random.default_rng(7))
        self.band_temps0 = np.array(
            self.state.sub_states["gb"].band_info["band_temps"], copy=True)
        self._census = dict(gbs.GBSpecialBase._branch_propose_counts)
        self.addCleanup(self._restore_census)
        patcher = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _restore_census(self):
        gbs.GBSpecialBase._branch_propose_counts.clear()
        gbs.GBSpecialBase._branch_propose_counts.update(self._census)

    # -- scenario A: both blocks live ---------------------------------
    def test_three_commands_in_order_on_every_rank(self):
        (_new_state, _acc), moves = run_propose(self.state)
        want = ["gb_run_proposal", "gb_run_tempering", "gb_finish"]
        self.assertEqual(moves[0].served, want)
        self.assertEqual(moves[1].served, want)
        # the session token is the OPENING command's (seq, call_index) and
        # rides on the two later payloads
        for move in moves.values():
            self.assertEqual(
                [p.get("session") for p in move.payloads[1:]],
                [move.token, move.token],
            )

    def test_log_like_is_the_blocks_concatenated_in_walker_order(self):
        (new_state, accepted), _moves = run_propose(self.state)
        np.testing.assert_allclose(
            new_state.log_like[0], 300.0 + np.arange(NWALKERS))
        # every engine rung carries the cold-chain value
        for t in range(new_state.log_like.shape[0]):
            np.testing.assert_allclose(
                new_state.log_like[t], new_state.log_like[0])
        self.assertEqual(accepted.shape, (NTEMPS, NWALKERS))
        self.assertFalse(accepted.any())

    def test_block_writes_land_at_w0_plus_local_walker(self):
        (new_state, _acc), moves = run_propose(self.state)
        work = moves[0]._work_branch(new_state)
        self.assertEqual(int(work.inds.sum()), 2)  # one per block
        for rank, (w0, w1) in BLOCKS.items():
            self.assertTrue(work.inds[0, w0, 0])
            np.testing.assert_allclose(
                work.coords[0, w0, 0], np.full(NDIM, float(rank)))
            # the WHOLE block is written, dead slots included -- a
            # ``keep_all_inds`` sorter's rejected-birth fill lives there and
            # ``_propose_legacy`` returns a state carrying it (fix round 4)
            np.testing.assert_allclose(
                work.coords[0, w0, 1], np.full(NDIM, -1.0 - rank))
            np.testing.assert_allclose(
                work.coords[1, w1 - 1, 0], np.full(NDIM, -1.0 - rank))
        sub = new_state.sub_states["gb"]
        np.testing.assert_allclose(sub.d_h[0:2], 10.0)
        np.testing.assert_allclose(sub.d_h[2:4], 11.0)
        np.testing.assert_allclose(sub.h_h[2:4], 21.0)

    def test_band_counts_and_pooled_counters(self):
        (new_state, _acc), _moves = run_propose(self.state)
        bi = new_state.sub_states["gb"].band_info
        np.testing.assert_array_equal(bi["band_num_binaries"][:, 0:2], 1)
        np.testing.assert_array_equal(bi["band_num_binaries"][:, 2:4], 2)
        # prop_counts are walker-summed per rank; the head sums the ranks
        # and transposes to (num_bands, ntemps)
        np.testing.assert_array_equal(bi["band_num_proposed_rj"], 1 + 2)
        np.testing.assert_array_equal(bi["band_num_accepted_rj"], 2)
        np.testing.assert_array_equal(bi["band_num_proposed"], 1 + 2)
        np.testing.assert_array_equal(bi["band_num_accepted"], 2)
        # swaps: summed over the blocks
        acc = np.tile(np.arange(1, NTEMPS), (NUM_BANDS, 1))
        np.testing.assert_array_equal(bi["band_swaps_accepted"], 3 * acc)
        np.testing.assert_array_equal(bi["band_swaps_proposed"], 20)

    def test_band_temps_adapt_once_from_the_summed_counters(self):
        (new_state, _acc), _moves = run_propose(self.state)
        got = new_state.sub_states["gb"].band_info["band_temps"]
        acc = 3 * np.tile(np.arange(1, NTEMPS), (NUM_BANDS, 1))
        prop = np.full((NUM_BANDS, NTEMPS - 1), 20)
        once = expected_ladder(self.band_temps0, acc, prop, times=1)
        np.testing.assert_allclose(got, once)
        self.assertFalse(np.allclose(got, self.band_temps0),
                         "the ladder must actually have moved")
        # the pooled step is NOT two per-rank steps
        twice = expected_ladder(self.band_temps0, acc, prop, times=2)
        self.assertFalse(np.allclose(once, twice))

    def test_clock_shutoff_and_cap_statistics(self):
        (_new_state, _acc), moves = run_propose(self.state)
        head = moves[0]
        self.assertEqual(head.time, 6)          # advanced by exactly one
        self.assertEqual(head.nwalkers, NWALKERS)  # restored after every fan-out
        self.assertEqual(head.ntemps, NTEMPS)
        self.assertEqual(len(head.shutoff_calls), 1)
        self.assertEqual(len(head.cap_calls), 1)
        stats = head.cap_calls[0]
        self.assertEqual(stats["lls"].shape, (NWALKERS, NUM_BANDS))
        # walker order: row i carries walker i's tag
        np.testing.assert_allclose(stats["lls"][:, 0], np.arange(NWALKERS))
        np.testing.assert_allclose(stats["band_lls"][:, 0], np.arange(NWALKERS))
        # the pooled [FSTAT_CTR] census is the sum over the blocks
        self.assertEqual(head._fstat_ctr_fallback_rows, 1 + 2)

    def test_the_head_body_runs_against_the_propose_model(self):
        """The head's own block is served with ``propose``'s ``model``.

        ``fanout.model`` is bound ONCE in ``run_global_fit`` and never
        refreshed, so the object eryn hands ``propose`` is the only one that
        is guaranteed current -- the head must close over IT, not over the
        one ``WalkerFanout.run`` passes its ``local_body``.
        """
        (_new_state, _acc), moves = run_propose(self.state)
        head = moves[0]
        self.assertEqual(len(head.seen_models), 3)
        for got in head.seen_models:
            self.assertIs(got, head._test_propose_model)
            self.assertIsNot(got, head._test_fanout_model)

    def test_the_head_keeps_its_own_propose_timer_across_the_commands(self):
        """``_prop_timer`` is restored after every fan-out (M4).

        A real body rebinds it to the rank session's timer and
        ``_exit_rank_block`` does not put it back, so without the restore the
        head's ``[GB_TIMING]`` line -- and any later head-side span -- would
        land on a closed session's timer.
        """
        (_new_state, _acc), moves = run_propose(self.state)
        head = moves[0]
        self.assertEqual(len(head.seen_timers), 3)
        self.assertIsNotNone(head.seen_timers[0])
        self.assertIs(head.seen_timers[1], head.seen_timers[0])
        self.assertIs(head.seen_timers[2], head.seen_timers[0])
        self.assertIs(head._prop_timer, head.seen_timers[0])
        self.assertIsNot(head._prop_timer, head._test_session_timer)

    def test_the_scan_schedule_is_sliced_per_block(self):
        """Per-walker schedules ship as each block's slice of the head draw."""
        starts = np.array([3, 1, 2, 0])
        dirs = np.array([1, -1, -1, 1])
        with mock.patch.object(
            gbs, "_draw_unit_scan_schedule",
            lambda random_state, n, units, pws, pwd: (starts, dirs),
        ):
            (_new_state, _acc), moves = run_propose(
                self.state, per_walker_schedule=True)
        for rank, (w0, w1) in BLOCKS.items():
            got = moves[rank].payloads[0]["scan_schedule"]
            np.testing.assert_array_equal(got[0], starts[w0:w1])
            np.testing.assert_array_equal(got[1], dirs[w0:w1])
        # every command of one propose carries the SAME block schedule
        self.assertEqual(len(moves[0].payloads), 3)

    def test_a_scalar_scan_schedule_ships_unchanged(self):
        """A whole-ensemble (scalar) schedule is passed through, not sliced."""
        with mock.patch.object(
            gbs, "_draw_unit_scan_schedule", lambda *a, **k: (2, 1),
        ):
            (_new_state, _acc), moves = run_propose(self.state)
        for move in moves.values():
            self.assertEqual(move.payloads[0]["scan_schedule"], (2, 1))

    def test_a_non_firing_tempering_gate_skips_the_command(self):
        """No ``gb_run_tempering``: zero swaps accumulate and the ladder holds."""
        (_new_state, _acc), moves = run_propose(self.state, temper_fire=False)
        for move in moves.values():
            self.assertEqual(move.served, ["gb_run_proposal", "gb_finish"])
        bi = _new_state.sub_states["gb"].band_info
        np.testing.assert_array_equal(bi["band_swaps_proposed"], 0)
        np.testing.assert_array_equal(bi["band_swaps_accepted"], 0)
        np.testing.assert_allclose(bi["band_temps"], self.band_temps0)
        self.assertEqual(moves[0].time, 6)

    def test_one_compute_rank_through_the_orchestrate_env_knob(self):
        """``GB_PROPOSE_ORCHESTRATE=1`` takes the orchestrator at one rank."""
        with mock.patch.dict(os.environ, {"GB_PROPOSE_ORCHESTRATE": "1"}):
            (new_state, _acc), moves = run_propose(self.state, size=1)
        head = moves[0]
        # the stub is only reachable through _propose_orchestrated
        self.assertEqual(
            head.served,
            ["gb_run_proposal", "gb_run_tempering", "gb_finish"])
        # single mode: ``seq`` never advances, ``call_index`` counts per
        # (move, op) -- so the session token is the opening command's (0, 1)
        self.assertEqual(head.token, (0, 1))
        self.assertEqual(
            [p.get("session") for p in head.payloads[1:]], [(0, 1), (0, 1)])
        # the single block IS the whole ensemble
        self.assertEqual(head.payloads[0]["nwalkers"], NWALKERS)
        np.testing.assert_allclose(
            new_state.log_like[0], 300.0 + np.arange(NWALKERS))

    def test_no_block_is_neutral_when_every_walker_has_sources(self):
        (_new_state, _acc), moves = run_propose(self.state)
        for move in moves.values():
            self.assertFalse(move.payloads[0]["neutral"])
            self.assertTrue(move.payloads[0]["keep_all_inds"])
        # per-propose, per-rank seeds
        seeds = {r: m.payloads[0]["rank_seed"] for r, m in moves.items()}
        self.assertNotEqual(seeds[0], seeds[1])
        for move in moves.values():
            self.assertEqual(
                {p["rank_seed"] for p in move.payloads}, {seeds[move._test_rank]}
            )

    # -- scenario B: the second block is empty -------------------------
    def test_empty_block_is_neutral_and_keeps_its_input_log_like(self):
        sub = self.state.sub_states["gb"]
        sub.branch.inds[:, 2:4] = False
        sub.sync_cold_row(self.state, "gb")
        before = np.array(self.state.log_like[0], copy=True)
        before_coords = np.array(sub.branch.coords[:, 2:4], copy=True)
        (new_state, _acc), moves = run_propose(self.state, use_prior_removal=True)
        self.assertFalse(moves[0].payloads[0]["neutral"])
        self.assertTrue(moves[1].payloads[0]["neutral"])
        self.assertFalse(moves[0].payloads[0]["keep_all_inds"])
        # rank 1 still answered all three commands (command-count symmetry)
        self.assertEqual(
            moves[1].served,
            ["gb_run_proposal", "gb_run_tempering", "gb_finish"],
        )
        # its walkers keep the log_like they came in with; rank 0's are merged
        np.testing.assert_allclose(new_state.log_like[0, 0:2], 300.0 + np.arange(2))
        np.testing.assert_allclose(new_state.log_like[0, 2:4], before[2:4])
        work = moves[0]._work_branch(new_state)
        self.assertEqual(int(work.inds[:, 2:4].sum()), 0)
        # a neutral block ships no branch block, so its coords are exactly
        # the ones the head sliced for it -- untouched, not zeroed
        np.testing.assert_array_equal(work.coords[:, 2:4], before_coords)
        bi = new_state.sub_states["gb"].band_info
        np.testing.assert_array_equal(bi["band_num_binaries"][:, 2:4], 0)
        # ...but its cap rows DO ride back, so the caps still advance on the
        # full N-walker statistic (Task 3 fix round; there is no
        # neutral-block cap skip on the head any more)
        self.assertEqual(len(moves[0].cap_calls), 1)
        stats = moves[0].cap_calls[0]
        self.assertEqual(stats["lls"].shape, (NWALKERS, NUM_BANDS))
        np.testing.assert_allclose(
            stats["lls"][:, 0], np.arange(NWALKERS, dtype=float))


class ReplyKeyContractTest(unittest.TestCase):
    """The keys the head READS are the keys the real rank bodies PRODUCE.

    The merge test above drives a stub, so a rename on the rank side would
    leave it green while the run broke. This drives the REAL
    ``_gb_serve_run_proposal`` / ``_gb_serve_run_tempering`` /
    ``_gb_serve_finish`` over their NEUTRAL paths -- the one branch that
    produces a full reply with no kernel, no ``BandSorter`` and no GPU --
    and checks every key in :data:`ORCHESTRATOR_REPLY_KEYS` is there. The
    non-neutral replies are built by the same three methods a few lines
    below their neutral returns.
    """

    def setUp(self):
        self.move = make_rank_move()
        self.model = _FakeModel()
        self._census = dict(gbs.GBSpecialBase._branch_propose_counts)
        self.addCleanup(self._restore_census)
        patcher = mock.patch.object(gbs, "pin_main_device", lambda xp, gpus: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _restore_census(self):
        gbs.GBSpecialBase._branch_propose_counts.clear()
        gbs.GBSpecialBase._branch_propose_counts.update(self._census)

    def test_every_key_the_orchestrator_reads_is_in_the_real_replies(self):
        token = (7, 2)
        replies = {}
        replies["gb_run_proposal"] = self.move.gf_serve(
            "gb_run_proposal", neutral_payload(),
            {"seq": 7, "call_index": 2}, self.model)
        replies["gb_run_tempering"] = self.move.gf_serve(
            "gb_run_tempering", neutral_payload(session=token, tmp_start=0),
            {"seq": 7, "call_index": 3}, self.model)
        replies["gb_finish"] = self.move.gf_serve(
            "gb_finish", neutral_payload(session=token, want_cap_stats=False),
            {"seq": 7, "call_index": 4}, self.model)
        for op, want in ORCHESTRATOR_REPLY_KEYS.items():
            missing = want - set(replies[op])
            self.assertEqual(
                set(), missing,
                f"{op}: the head reads {sorted(missing)}, the rank body does "
                "not ship it")


if __name__ == "__main__":
    unittest.main()
