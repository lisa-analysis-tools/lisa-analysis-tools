"""Search LEGS: one stored row per leg of the search cycle (user design 2026-09-30).

A search stage's move list is walked by a cursor. One sampler iteration runs
from the cursor up to and including the next LEG-ENDER (every ``in_model*``
move by default), asking each move's cadence gate whether it is due, and
then eryn saves a row. The state carries the name of the leg-ender it was
saved after; the saver writes that name into the row and stamps the stage's
ordered move list on the recipe group. At startup the step reads the last
row's name, finds it in the live list and sets the cursor to the move after
it -- a NAME, never a count, so a gated move that skipped a cycle cannot
mis-align the resume. Cadences and the galfor ratchet count CYCLES (cursor
wraps), not rows.

Construction-level: fakes for the moves, a tiny h5 for the backend.
"""

from __future__ import annotations

import contextlib
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "scripts", "fstat_proposal"))

from lisatools.globalfit.legs import LegCursor, leg_ends_from_names  # noqa: E402


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


ORDER = ["vgb_pe", "noise_ratchet_search", "rj_warm_search", "in_model",
         "rj_fstat_search", "in_model_fstat",
         "rj_prior_removal", "in_model_removal", "gb_ridge_gibbs", "mbh_pe"]


# ======================================================================
# 1. the cursor
# ======================================================================

class LegEndsTest(unittest.TestCase):

    def test_default_is_every_in_model_move_in_order(self):
        self.assertEqual(leg_ends_from_names(ORDER),
                         ["in_model", "in_model_fstat", "in_model_removal"])

    def test_explicit_list_must_exist_in_the_order(self):
        self.assertEqual(leg_ends_from_names(ORDER, ["in_model_removal"]),
                         ["in_model_removal"])
        with self.assertRaises(ValueError):
            leg_ends_from_names(ORDER, ["in_model_replace"])
        with self.assertRaises(ValueError):
            leg_ends_from_names(["a", "b"])          # no in_model at all


class LegCursorTest(unittest.TestCase):

    def _c(self):
        return LegCursor(ORDER, leg_ends_from_names(ORDER))

    def test_three_legs_then_wrap(self):
        c = self._c()
        idx, end, last = c.plan()
        self.assertEqual([ORDER[i] for i in idx], ORDER[:4]); self.assertEqual(end, "in_model")
        self.assertFalse(last)
        c.advance()
        idx, end, last = c.plan()
        self.assertEqual([ORDER[i] for i in idx], ["rj_fstat_search", "in_model_fstat"])
        c.advance()
        idx, end, last = c.plan()
        # the last leg runs THROUGH the tail of the list (ridge, sources)
        self.assertEqual([ORDER[i] for i in idx],
                         ["rj_prior_removal", "in_model_removal", "gb_ridge_gibbs", "mbh_pe"])
        self.assertEqual(end, "in_model_removal"); self.assertTrue(last)
        self.assertEqual(c.cycles, 0)
        c.advance()
        self.assertEqual(c.cursor, 0); self.assertEqual(c.cycles, 1)

    def test_set_after_a_name_resumes_at_the_next_move(self):
        c = self._c()
        self.assertTrue(c.set_after("in_model_fstat"))
        self.assertEqual(ORDER[c.cursor], "rj_prior_removal")
        self.assertTrue(c.set_after("in_model_removal"))
        self.assertEqual(c.cursor, 0)          # after the last leg: head of the list

    def test_unknown_or_missing_name_starts_at_the_head(self):
        c = self._c()
        c.set_after("in_model_fstat")
        self.assertFalse(c.set_after("in_model_replace"))
        self.assertEqual(c.cursor, 0)
        self.assertFalse(c.set_after(None))
        self.assertEqual(c.cursor, 0)

    def test_cycles_from_history_counts_last_leg_rows(self):
        c = self._c()
        hist = ["in_model", "in_model_fstat", "in_model_removal",
                "in_model", "in_model_fstat", "in_model_removal", "in_model"]
        self.assertEqual(c.cycles_from_history(hist), 2)
        self.assertEqual(c.cycles_from_history([]), 0)
        self.assertEqual(c.cycles_from_history([None, "", "in_model"]), 0)


# ======================================================================
# 2. the combine walks the legs
# ======================================================================

class _Move:
    def __init__(self, name, log):
        self.gf_move_name = name
        self.log = log
        self.moves = []

    def propose(self, model, state):
        self.log.append(self.gf_move_name)
        return state, np.zeros(np.shape(state.log_like), dtype=bool)


def _state():
    return SimpleNamespace(log_like=np.zeros((1, 4)), sub_states=None)


class CombineLegsTest(unittest.TestCase):

    def _combine(self, every=None):
        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        log = []
        moves = [_Move(n, log) for n in ORDER]
        kw = dict(share_temperature_control=False, leg_ends="auto")
        if every is not None:
            kw["move_every"] = every
        cm = GFCombineMove(moves=moves, **kw)
        cm.gf_stage_name = "gb_search_3"
        return cm, log

    def test_one_leg_per_propose_and_the_state_carries_the_name(self):
        cm, log = self._combine()
        st, _ = cm.propose(None, _state())
        self.assertEqual(log, ORDER[:4])
        self.assertEqual(st.gf_saved_after, "in_model")
        self.assertEqual(list(st.gf_move_order), ORDER)
        self.assertEqual(st.gf_stage_name, "gb_search_3")
        log.clear(); st, _ = cm.propose(None, _state())
        self.assertEqual(log, ["rj_fstat_search", "in_model_fstat"])
        self.assertEqual(st.gf_saved_after, "in_model_fstat")
        log.clear(); st, _ = cm.propose(None, _state())
        self.assertEqual(log, ["rj_prior_removal", "in_model_removal", "gb_ridge_gibbs", "mbh_pe"])
        self.assertEqual(st.gf_saved_after, "in_model_removal")
        self.assertEqual(cm.gf_legs.cycles, 1)

    def test_cadence_counts_cycles_not_rows(self):
        every = [1] * len(ORDER)
        every[ORDER.index("rj_warm_search")] = 2      # every 2nd CYCLE
        cm, log = self._combine(every=every)
        for _ in range(3):
            cm.propose(None, _state())               # cycle 0: warm runs
        self.assertIn("rj_warm_search", log)
        log.clear()
        for _ in range(3):
            cm.propose(None, _state())               # cycle 1: warm skipped
        self.assertNotIn("rj_warm_search", log)
        self.assertIn("in_model", log)               # leg-enders are never gated
        log.clear()
        cm.propose(None, _state())                   # cycle 2, leg 0: warm back
        self.assertIn("rj_warm_search", log)

    def test_resume_after_a_name_runs_the_next_leg_first(self):
        cm, log = self._combine()
        self.assertTrue(cm.gf_legs.set_after("in_model_fstat"))
        st, _ = cm.propose(None, _state())
        self.assertEqual(log[0], "rj_prior_removal")
        self.assertEqual(st.gf_saved_after, "in_model_removal")

    def test_a_leg_end_missing_from_the_list_is_refused(self):
        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        with self.assertRaises(ValueError):
            GFCombineMove(moves=[_Move("rj_fstat_search", [])],
                          share_temperature_control=False, leg_ends=["in_model"])

    def test_a_move_that_changed_the_noise_ends_the_leg_early(self):
        """User ruling 2026-09-30: a save after the in-model noise step
        WHENEVER IT RUNS. The gated noise head sets ``gf_leg_end_now`` when
        it nudged or released (and ran its in-model pass); the leg ends
        there and the row is saved after the gate's name. On a hold the flag
        stays False and the leg runs on to in_model as usual."""
        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        order = ["noise_ratchet_search", "vgb_pe", "rj_warm_search", "in_model",
                 "rj_fstat_search", "in_model_fstat",
                 "rj_prior_removal", "in_model_removal", "mbh_pe"]
        log = []

        class _Gate(_Move):
            changed = True

            def propose(self, model, state):
                self.gf_leg_end_now = self.changed
                return super().propose(model, state)

        moves = [(_Gate if n == "noise_ratchet_search" else _Move)(n, log) for n in order]
        cm = GFCombineMove(moves=moves, share_temperature_control=False, leg_ends="auto")
        cm.gf_stage_name = "gb_search_3"
        # nudge cycle: the gate ends leg 1 on its own; the rest of leg 1 follows
        st, _ = cm.propose(None, _state())
        self.assertEqual(log, ["noise_ratchet_search"])
        self.assertEqual(st.gf_saved_after, "noise_ratchet_search")
        log.clear(); st, _ = cm.propose(None, _state())
        self.assertEqual(log, ["vgb_pe", "rj_warm_search", "in_model"])
        self.assertEqual(st.gf_saved_after, "in_model")
        log.clear(); cm.propose(None, _state()); log.clear(); cm.propose(None, _state())
        self.assertEqual(cm.gf_legs.cycles, 1)        # 4 rows, ONE cycle
        # hold cycle: the gate ran but changed nothing -> no extra row
        moves[0].changed = False
        log.clear(); st, _ = cm.propose(None, _state())
        self.assertEqual(log, ["noise_ratchet_search", "vgb_pe", "rj_warm_search", "in_model"])
        self.assertEqual(st.gf_saved_after, "in_model")
        # a stale flag can never end a later leg: it is consumed when read
        self.assertFalse(getattr(moves[0], "gf_leg_end_now", False))

    def test_resume_after_the_gate_lands_on_vgb(self):
        c = LegCursor(["noise_ratchet_search", "vgb_pe", "in_model", "rj_fstat_search",
                       "in_model_fstat"], ["in_model", "in_model_fstat"])
        self.assertTrue(c.set_after("noise_ratchet_search"))
        self.assertEqual(c.order[c.cursor], "vgb_pe")
        # rows saved after the gate do not count as cycles
        self.assertEqual(c.cycles_from_history(
            ["noise_ratchet_search", "in_model", "in_model_fstat", "noise_ratchet_search"]), 1)

    def test_no_leg_ends_means_the_old_one_row_per_cycle(self):
        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        log = []
        cm = GFCombineMove(moves=[_Move(n, log) for n in ORDER],
                           share_temperature_control=False)
        st, _ = cm.propose(None, _state())
        self.assertEqual(log, ORDER)
        self.assertFalse(hasattr(st, "gf_saved_after"))


# ======================================================================
# 3. the saver writes the name and the order; the readers read them
# ======================================================================

class BackendSavedAfterTest(unittest.TestCase):

    def _store(self, tmp, iteration=3):
        import h5py

        path = os.path.join(tmp, "s_testing.h5")
        with h5py.File(path, "w") as f:
            g = f.create_group("global_fit")
            g.attrs["iteration"] = iteration
            r = g.create_group("recipe")
            s = r.create_group("gb_search_3")
            s.attrs["status"] = False
            s.attrs["order num"] = 0
        return path

    def test_write_then_read(self):
        import tempfile

        import h5py

        from lisatools.globalfit.hdfbackend import (
            GFHDFBackend, write_saved_after, stamp_move_order)

        with tempfile.TemporaryDirectory() as tmp:
            path = self._store(tmp, iteration=3)
            with h5py.File(path, "a") as f:
                g = f["global_fit"]
                # row 2 (the newest) was saved after in_model_fstat; earlier rows unknown
                write_saved_after(g, 3, "in_model_fstat")
                stamp_move_order(g, "gb_search_3", ORDER)
                stamp_move_order(g, "gb_search_3", ["changed"])   # never overwritten
            be = GFHDFBackend(path)
            self.assertEqual(be.saved_after(2), "in_model_fstat")
            self.assertIsNone(be.saved_after(1))                # written as ""
            self.assertEqual(be.saved_after_history(0, 3), [None, None, "in_model_fstat"])
            self.assertEqual(be.stage_move_order("gb_search_3"), ORDER)
            self.assertIsNone(be.stage_move_order("full_pe"))

    def test_old_store_without_the_dataset_reads_none(self):
        import tempfile

        from lisatools.globalfit.hdfbackend import GFHDFBackend

        with tempfile.TemporaryDirectory() as tmp:
            be = GFHDFBackend(self._store(tmp))
            self.assertIsNone(be.saved_after(2))
            self.assertEqual(be.saved_after_history(0, 3), [None, None, None])

    def test_stage_flag_round_trips_and_is_none_when_absent(self):
        """The galfor ratchet's stop stamp (user ruling 2026-10-02: a relaunch
        must not ratchet again because it forgot)."""
        import tempfile

        from lisatools.globalfit.hdfbackend import GFHDFBackend

        with tempfile.TemporaryDirectory() as tmp:
            be = GFHDFBackend(self._store(tmp))
            self.assertIsNone(be.stage_flag("gb_search_3", "galfor_ratchet_done"))
            self.assertTrue(be.stamp_stage_flag("gb_search_3", "galfor_ratchet_done", 1))
            self.assertEqual(int(be.stage_flag("gb_search_3", "galfor_ratchet_done")), 1)
            # an unknown stage: no write, no read, no exception
            self.assertFalse(be.stamp_stage_flag("full_pe", "galfor_ratchet_done", 1))
            self.assertIsNone(be.stage_flag("full_pe", "galfor_ratchet_done"))
            # the stamp leaves the other stamps alone
            self.assertIsNone(be.stage_start_iteration("gb_search_3"))


# ======================================================================
# 4. the step resumes by NAME and drives the ratchet in cycles
# ======================================================================

class _FakeCombine:
    """The stage combine as the step sees it: carries gf_legs."""

    def __init__(self):
        self.gf_legs = LegCursor(ORDER, leg_ends_from_names(ORDER))
        self.moves = []


class _FakeGate:
    is_noise_ratchet_gate = True

    def __init__(self):
        self.moves = []
        self.modes = []
        self.mode = "release"

    def set_mode(self, m):
        self.mode = m
        self.modes.append(m)


class _FakeBackend:
    def __init__(self, iteration, history, stage_start=None, order=None):
        self.iteration = iteration
        self._h = list(history)          # saved_after per row, index = row
        self._start = stage_start
        self._order = order

    def get_nleaves(self, branch_names=None, temp_index=0):
        return {branch_names[0]: np.arange(self.iteration)[:, None]}

    def stage_start_iteration(self, name):
        return self._start

    def saved_after(self, it):
        return self._h[it] if 0 <= it < len(self._h) else None

    def saved_after_history(self, a, b):
        return [self.saved_after(i) for i in range(a, b)]

    def stage_move_order(self, name):
        return self._order


class _FakeSampler:
    def __init__(self, backend, moves):
        self.backend = backend
        self.moves = moves
        self.periodic = None
        self.temperature_control = None
        self.weights = None


class StepResumesByNameTest(unittest.TestCase):

    def _step(self, ratchet=None):
        from lisatools.globalfit.recipe import SearchStageProfileStep

        cm, gate = _FakeCombine(), _FakeGate()
        cm.moves = [gate]
        st = SearchStageProfileStep(
            moves=[cm], convergence_iter=2, plateau_branch="gb", profile={},
            stage_name="gb_search_3", legs=True, ratchet=ratchet,
            ratchet_delta=None if ratchet is None else np.zeros(5))
        return st, cm, gate

    def _enter(self, st, cm, backend):
        st.setup_run(backend.iteration, None, _FakeSampler(backend, [cm]))
        st.note_recipe_step(3)

    def test_resume_lands_on_the_move_after_the_saved_name(self):
        st, cm, _ = self._step()
        # stage started at 47; rows 47, 48 saved after in_model, in_model_fstat
        hist = [None] * 47 + ["in_model", "in_model_fstat"]
        self._enter(st, cm, _FakeBackend(49, hist, stage_start=47, order=ORDER))
        self.assertEqual(ORDER[cm.gf_legs.cursor], "rj_prior_removal")
        self.assertEqual(cm.gf_legs.cycles, 0)

    def test_cycles_are_counted_from_the_stored_names(self):
        st, cm, _ = self._step()
        hist = [None] * 47 + ["in_model", "in_model_fstat", "in_model_removal",
                              "in_model", "in_model_fstat", "in_model_removal",
                              "in_model"]
        self._enter(st, cm, _FakeBackend(54, hist, stage_start=47, order=ORDER))
        self.assertEqual(cm.gf_legs.cycles, 2)
        self.assertEqual(ORDER[cm.gf_legs.cursor], "rj_fstat_search")

    def test_fresh_stage_or_old_store_starts_at_the_head(self):
        st, cm, _ = self._step()
        self._enter(st, cm, _FakeBackend(47, [None] * 47, stage_start=47))
        self.assertEqual(cm.gf_legs.cursor, 0)
        self.assertEqual(cm.gf_legs.cycles, 0)

    # ---- the ratchet's gain check and clock reset UNDER LEGS ---------------
    # (user ruling 2026-10-02: "please double check those gates")

    @staticmethod
    def _sample(mx, nw=4):
        return SimpleNamespace(
            log_like=np.array([[mx - 300.0, mx, mx - 50.0, mx - 120.0][:nw],
                               [0.0] * nw]),
            branches_coords={"galfor": np.zeros((2, nw, 1, 5))})

    def test_gain_check_runs_once_per_cycle_not_once_per_leg(self):
        """Under legs stopping_function fires at EVERY leg end. The release
        gain must be read once per completed cycle (at the wrap), or the
        second leg of the first release cycle would be compared with its
        first and stop the ratchet on a within-cycle wobble."""
        from lisatools.globalfit.noise_ratchet import RatchetSchedule
        from lisatools.globalfit.recipe import SearchStageProfileStep

        cm, gate = _FakeCombine(), _FakeGate()
        cm.moves = [gate]
        st = SearchStageProfileStep(
            moves=[cm], convergence_iter=2, plateau_branch="gb", profile={},
            stage_name="gb_search_3", legs=True,
            ratchet=RatchetSchedule(hold=1, release=1, cycles=20, release_first=True),
            ratchet_delta=np.zeros(5), ratchet_min_gain=200.0)
        be = _FakeBackend(47, [None] * 47, stage_start=47)
        with env(GALFOR_RATCHET_HOLD_STAGE=None, GALFOR_RATCHET_CLOCK_RESET=None):
            self._enter(st, cm, be)
            self.assertEqual(gate.modes, ["release"])            # release first, k = 0
            smp = _FakeSampler(be, [cm])
            # three leg ends inside cycle 0 (the release cycle): no wrap yet
            for mx in (1000.0, 1000.0, 1000.0):
                st.stopping_function(48, self._sample(mx), smp)
            self.assertEqual(st._ratchet_release_maxes, [])
            self.assertEqual(gate.modes, ["release"])
            cm.gf_legs.cycles = 1                                 # the wrap
            st.stopping_function(49, self._sample(1000.0), smp)
            self.assertEqual(st._ratchet_release_maxes, [(0, 1000.0)])
            self.assertEqual(gate.modes[-1], "nudge")             # k = 1
            cm.gf_legs.cycles = 2                                 # nudge cycle done
            st.stopping_function(50, self._sample(-9000.0), smp)
            self.assertEqual(gate.modes[-1], "release")           # k = 2
            # within the release cycle the lnL wobbles: NOT a gain reading
            for mx in (1100.0, 1050.0):
                st.stopping_function(51, self._sample(mx), smp)
            self.assertEqual(len(st._ratchet_release_maxes), 1)
            self.assertFalse(st._ratchet_stopped)
            cm.gf_legs.cycles = 3                                 # the wrap of k = 2
            st.stopping_function(52, self._sample(1500.0), smp)
            self.assertEqual(st._ratchet_release_maxes, [(0, 1000.0), (2, 1500.0)])
            self.assertFalse(st._ratchet_stopped)                 # +500: a real step
            self.assertEqual(gate.modes[-1], "nudge")

    def test_clock_reset_mid_cycle_waits_for_the_next_head(self):
        """A resume in the middle of a cycle has already passed that cycle's
        noise head, so k0 is the NEXT cycle: nothing is driven at entry and
        the first gate run after the wrap is the release (k = 0)."""
        from lisatools.globalfit.noise_ratchet import RatchetSchedule
        from lisatools.globalfit.recipe import SearchStageProfileStep

        cm, gate = _FakeCombine(), _FakeGate()
        cm.moves = [gate]
        st = SearchStageProfileStep(
            moves=[cm], convergence_iter=2, plateau_branch="gb", profile={},
            stage_name="gb_search_3", legs=True,
            ratchet=RatchetSchedule(hold=1, release=1, cycles=20, release_first=True),
            ratchet_delta=np.zeros(5), ratchet_min_gain=200.0)
        # rows 47, 48 saved after in_model, in_model_fstat: mid-cycle, 0 cycles done
        hist = [None] * 47 + ["in_model", "in_model_fstat"]
        be = _FakeBackend(49, hist, stage_start=47, order=ORDER)
        with env(GALFOR_RATCHET_HOLD_STAGE=None, GALFOR_RATCHET_CLOCK_RESET="1"):
            self._enter(st, cm, be)
            self.assertEqual(ORDER[cm.gf_legs.cursor], "rj_prior_removal")
            self.assertEqual(st._ratchet_k0, 1)
            self.assertEqual(st._ratchet_k, -1)
            self.assertEqual(gate.modes, [])                      # nothing driven
            smp = _FakeSampler(be, [cm])
            st.stopping_function(49, self._sample(900.0), smp)   # a leg end, no wrap
            self.assertEqual(gate.modes, [])
            self.assertEqual(st._ratchet_release_maxes, [])
            cm.gf_legs.cycles = 1                                 # the wrap
            st.stopping_function(50, self._sample(900.0), smp)
            self.assertEqual(st._ratchet_k, 0)
            self.assertEqual(gate.modes, ["release"])             # release first
            self.assertEqual(st._ratchet_release_maxes, [])       # k = -1 never read

    def test_clock_start_continues_with_the_nudge_after_an_interrupted_release_cycle(self):
        """6mo job 685: the first release ran (row 65), its cycle was cut off
        mid-way. GALFOR_RATCHET_CLOCK_START=1 makes the in-progress cycle
        schedule k = 0, so the wrap records its lnL as the baseline and the
        next gate run is the NUDGE, not a second release."""
        from lisatools.globalfit.noise_ratchet import RatchetSchedule
        from lisatools.globalfit.recipe import SearchStageProfileStep

        cm, gate = _FakeCombine(), _FakeGate()
        cm.moves = [gate]
        st = SearchStageProfileStep(
            moves=[cm], convergence_iter=2, plateau_branch="gb", profile={},
            stage_name="gb_search_3", legs=True,
            ratchet=RatchetSchedule(hold=1, release=1, cycles=20, release_first=True),
            ratchet_delta=np.zeros(5), ratchet_min_gain=200.0)
        # one full cycle, then a cycle whose noise head + in_model have run
        hist = [None] * 47 + ["in_model", "in_model_fstat", "in_model_removal",
                              "noise_ratchet_search", "in_model"]
        be = _FakeBackend(52, hist, stage_start=47, order=ORDER)
        with env(GALFOR_RATCHET_HOLD_STAGE=None, GALFOR_RATCHET_CLOCK_RESET="1",
                 GALFOR_RATCHET_CLOCK_START="1"):
            self._enter(st, cm, be)
            self.assertEqual(ORDER[cm.gf_legs.cursor], "rj_fstat_search")
            self.assertEqual(cm.gf_legs.cycles, 1)
            self.assertEqual(st._ratchet_k0, 1)                 # (1 + 1 mid) - 1
            self.assertEqual(st._ratchet_k, 0)                  # the cycle in progress IS k = 0
            self.assertEqual(gate.modes, ["release"])           # harmless: the head already ran
            smp = _FakeSampler(be, [cm])
            st.stopping_function(52, self._sample(1000.0), smp)   # a leg end, no wrap
            self.assertEqual(st._ratchet_release_maxes, [])
            cm.gf_legs.cycles = 2                                 # the wrap of the release cycle
            st.stopping_function(53, self._sample(1000.0), smp)
            self.assertEqual(st._ratchet_release_maxes, [(0, 1000.0)])   # baseline from THIS cycle
            self.assertEqual(st._ratchet_k, 1)
            self.assertEqual(gate.modes[-1], "nudge")           # part 1 of 2 next
        with env(GALFOR_RATCHET_CLOCK_START="-1", GALFOR_RATCHET_CLOCK_RESET="1"):
            st2 = SearchStageProfileStep(
                moves=[cm], convergence_iter=2, plateau_branch="gb", profile={},
                stage_name="gb_search_3", legs=True,
                ratchet=RatchetSchedule(hold=1, release=1, cycles=20, release_first=True),
                ratchet_delta=np.zeros(5))
            with self.assertRaises(ValueError):
                self._enter(st2, cm, be)

    def test_a_stamped_stop_survives_a_relaunch(self):
        """The data-driven stop is stamped in the store; a fresh process that
        re-enters the stage must not nudge again."""
        from lisatools.globalfit.noise_ratchet import RatchetSchedule
        from lisatools.globalfit.recipe import SearchStageProfileStep

        class _StampedBackend(_FakeBackend):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.flags = {"galfor_ratchet_done": 1}

            def stage_flag(self, name, key):
                return self.flags.get(key)

            def stamp_stage_flag(self, name, key, value):
                self.flags[key] = value
                return True

        cm, gate = _FakeCombine(), _FakeGate()
        cm.moves = [gate]
        st = SearchStageProfileStep(
            moves=[cm], convergence_iter=2, plateau_branch="gb", profile={},
            stage_name="gb_search_3", legs=True,
            ratchet=RatchetSchedule(hold=1, release=1, cycles=20, release_first=True),
            ratchet_delta=np.zeros(5), ratchet_min_gain=200.0)
        be = _StampedBackend(60, [None] * 60, stage_start=47)
        with env(GALFOR_RATCHET_HOLD_STAGE=None, GALFOR_RATCHET_CLOCK_RESET="1"):
            self._enter(st, cm, be)
            self.assertTrue(st._ratchet_stopped)
            self.assertEqual(gate.modes, [])                      # no nudge, no release drive
            self.assertFalse(st._ratchet_schedule_pending(0))     # the stage may end
            smp = _FakeSampler(be, [cm])
            cm.gf_legs.cycles = 1
            st.stopping_function(61, self._sample(5.0), smp)
            self.assertEqual(gate.modes, [])
            self.assertEqual(st._ratchet_release_maxes, [])

    def test_changed_composition_starts_at_the_head(self):
        st, cm, _ = self._step()
        hist = [None] * 47 + ["in_model", "in_model_fstat"]
        self._enter(st, cm, _FakeBackend(49, hist, stage_start=47,
                                         order=ORDER[:-1]))     # a move dropped
        self.assertEqual(cm.gf_legs.cursor, 0)

    def test_ratchet_clock_counts_cycles_when_legs_are_on(self):
        from lisatools.globalfit.noise_ratchet import RatchetSchedule

        st, cm, gate = self._step(ratchet=RatchetSchedule(hold=2, release=1, cycles=1))
        be = _FakeBackend(47, [None] * 47, stage_start=47)
        self._enter(st, cm, be)
        self.assertEqual(gate.modes, ["nudge"])
        # three rows = one cycle; the combine wraps its cycle count at the third
        for name in ("in_model", "in_model_fstat", "in_model_removal"):
            be.iteration += 1
            if name == "in_model_removal":
                cm.gf_legs.cycles += 1
            st.stopping_function(be.iteration, None, _FakeSampler(be, [cm]))
        self.assertEqual(gate.modes, ["nudge", "hold"])       # one drive per CYCLE
        for name in ("in_model", "in_model_fstat", "in_model_removal"):
            be.iteration += 1
            if name == "in_model_removal":
                cm.gf_legs.cycles += 1
            st.stopping_function(be.iteration, None, _FakeSampler(be, [cm]))
        self.assertEqual(gate.modes, ["nudge", "hold", "release"])


# ======================================================================
# 5. composition + launcher knob
# ======================================================================

class CompositionTest(unittest.TestCase):

    _BASE = dict(
        GB_SEARCH_IN_MODEL="1", GB_SEARCH_RJ_REPLACE="1",
        GB_SEARCH_IN_MODEL_REPLACE="1",
        GB_WARM_START_COMPONENTS="/nonexistent/warm.npz",
        STAGE_V9_SEARCH="1", GB_SEARCH_3_WARM_EVERY="5",
        MBHB_IDS="2,5", STAGE_SKIP_SOURCE_SEARCH="1", VGB_CHIRP_MASS_BASIS="1",
        PSD_START_PARAMS=None, GALFOR_START_PARAMS=None,
        GALFOR_RATCHET=None, GB_SEARCH_LEGS=None,
    )

    def _full(self, **ov):
        import run_combined_staged as R

        with env(**{**self._BASE, **ov}):
            return R.build_fit()

    def test_armed_search_stages_run_one_leg_per_row(self):
        fit = self._full(GB_SEARCH_LEGS="1")
        by = {s.name: s for s in fit.recipe.stages}
        for s in ("gb_search_1", "gb_search_2", "gb_search_3"):
            self.assertEqual(by[s].combine_kwargs.get("leg_ends"), "auto", s)
            self.assertTrue(by[s].step_kwargs.get("legs"), s)
        self.assertNotIn("leg_ends", by["gb_search_seed"].combine_kwargs)
        self.assertNotIn("leg_ends", by["full_pe"].combine_kwargs)

    def test_unarmed_is_unchanged(self):
        fit = self._full()
        for s in fit.recipe.stages:
            self.assertNotIn("leg_ends", s.combine_kwargs, s.name)
            self.assertFalse(s.step_kwargs.get("legs", False), s.name)


if __name__ == "__main__":
    unittest.main()
