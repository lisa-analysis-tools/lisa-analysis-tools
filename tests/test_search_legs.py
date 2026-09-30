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
