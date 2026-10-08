"""The two stage-advance knobs added 2026-09-26, both DEFAULT OFF.

``GB_SEARCH_STAGE_END_ON_SHUTOFF`` (DEFAULT ON) makes "every occupied
(walker, band) pair has shut off" the WHOLE stage criterion instead of one
half of an AND with the nleaves plateau (user ruling: "we want to end the
recipe stage when all the (band, walker) pairs are shutoff from converged
logL per (band, walker)"). =0 restores the composed rule.

``GF_PERSIST_STAGE_START`` keeps a stage's convergence clock across a resume
by stamping its start iteration in the backend, instead of resetting it to
the live iteration on every launch.

``GF_PERSIST_STAGE_START`` stays DEFAULT OFF in the code; the v9 production
launchers export it ON since 2026-10-07 (user ruling: "keep
GF_PERSIST_STAGE_START, but it should reset at the end of a stage" -- the
boundary reset is ``GFHDFBackend.completed_recipe_step``, pinned in
tests/test_noise_ratchet.py::NextStepStartStampTest; here: a stored start
PAST the live iteration, a rewound store, is ignored and re-stamped).
"""
import os
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.recipe import SearchStageProfileStep


class _FakeMove:
    """A move publishing the per-(walker, band) valve's two readouts."""

    def __init__(self, pending):
        self._shutoff_w_pending = pending
        self._rj_band_shutoff_w = object()
        self.moves = None


class _FakeBackend:
    def __init__(self, iteration, nleaves):
        self.iteration = iteration
        self._nleaves = np.asarray(nleaves)
        self.stamped = {}

    def get_nleaves(self, branch_names=None, temp_index=0):
        return {branch_names[0]: self._nleaves}

    def stage_start_iteration(self, name):
        return self.stamped.get(name)

    def stamp_stage_start(self, name, it):
        self.stamped[name] = int(it)


def _step(**kw):
    step = SearchStageProfileStep.__new__(SearchStageProfileStep)
    step.convergence_iter = 20
    step.plateau_branch = "gb"
    step.stage_name = "gb_search_1"
    step.st = None
    step.convergence_fn = None
    step.search_shutoff_per_walker = True
    step._stage_start_iter = 0
    for k, v in kw.items():
        setattr(step, k, v)
    return step


def _sampler(iteration, nleaves, pending):
    return SimpleNamespace(
        backend=_FakeBackend(iteration, nleaves),
        moves=[_FakeMove(pending)],
    )


class StageEndOnShutoffTest(unittest.TestCase):
    def setUp(self):
        for k in ("GB_SEARCH_STAGE_END_ON_SHUTOFF", "GB_SEARCH_CAP_QUIESCENT"):
            self.addCleanup(os.environ.pop, k, None)
        # the cap veto is a separate gate; keep it out of these cases
        os.environ["GB_SEARCH_CAP_QUIESCENT"] = "0"

    def test_DEFAULT_ON_ends_as_soon_as_every_occupied_pair_shut_off(self):
        """Unset, the shutoff rule alone ends the stage (user ruling
        2026-09-26, raised from the default-off it first shipped with).
        The stage here is well short of the 41-iteration plateau window,
        which under the old composed rule could not even be evaluated."""
        os.environ.pop("GB_SEARCH_STAGE_END_ON_SHUTOFF", None)
        s = _sampler(iteration=30, nleaves=np.full(30, 500), pending=0)
        self.assertTrue(_step().stopping_function(30, None, s))

    def test_explicit_zero_restores_the_composed_plateau_AND_shutoff_rule(self):
        for off in ("0", ""):
            with self.subTest(value=off):
                os.environ["GB_SEARCH_STAGE_END_ON_SHUTOFF"] = off
                s = _sampler(iteration=30, nleaves=np.full(30, 500), pending=0)
                self.assertFalse(_step().stopping_function(30, None, s))

    def test_armed_explicitly_behaves_the_same_as_the_default(self):
        os.environ["GB_SEARCH_STAGE_END_ON_SHUTOFF"] = "1"
        s = _sampler(iteration=30, nleaves=np.full(30, 500), pending=0)
        self.assertTrue(_step().stopping_function(30, None, s))

    def test_armed_holds_while_any_occupied_pair_is_still_active(self):
        os.environ["GB_SEARCH_STAGE_END_ON_SHUTOFF"] = "1"
        s = _sampler(iteration=30, nleaves=np.full(30, 500), pending=36)
        self.assertFalse(_step().stopping_function(30, None, s))

    def test_a_stage_with_NO_leaves_never_ends_on_this_rule(self):
        """The trap: gb_search_1 opens at ZERO leaves, so no pair is
        occupied and ``pending`` is 0 at entry. Without this guard the
        stage would end on its first check, before a single birth."""
        os.environ["GB_SEARCH_STAGE_END_ON_SHUTOFF"] = "1"
        s = _sampler(iteration=30, nleaves=np.zeros(30, dtype=int), pending=0)
        self.assertFalse(_step().stopping_function(30, None, s))

    def test_the_valve_must_be_ARMED_or_pending_zero_means_nothing(self):
        """``pending`` is 0 both when everything converged and when the
        feature is off; only ``band_shutoff_w_armed`` separates them."""
        os.environ["GB_SEARCH_STAGE_END_ON_SHUTOFF"] = "1"
        s = _sampler(iteration=30, nleaves=np.full(30, 500), pending=0)
        s.moves = [SimpleNamespace(moves=None)]          # valve not armed
        self.assertFalse(_step().stopping_function(30, None, s))


class PersistStageStartTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.environ.pop, "GF_PERSIST_STAGE_START", None)

    def _setup_run(self, backend, iteration):
        step = _step()
        step.moves, step.weights, step.thin_by = [], None, 1
        sampler = SimpleNamespace(
            backend=backend, moves=[], weights=None, yield_step=None,
            checkpoint_step=None, periodic=None, temperature_control=None)
        backend.iteration = iteration
        step.setup_run(iteration, None, sampler)
        return step._stage_start_iter

    def test_default_off_resets_the_clock_on_every_launch(self):
        os.environ.pop("GF_PERSIST_STAGE_START", None)
        be = _FakeBackend(25, np.zeros(1))
        self.assertEqual(self._setup_run(be, 25), 25)
        # a resume at 30 moves the start to 30 -- the pre-2026-09-26 behaviour
        self.assertEqual(self._setup_run(be, 30), 30)
        self.assertEqual(be.stamped, {})

    def test_armed_keeps_the_ORIGINAL_start_across_a_resume(self):
        os.environ["GF_PERSIST_STAGE_START"] = "1"
        be = _FakeBackend(25, np.zeros(1))
        self.assertEqual(self._setup_run(be, 25), 25)
        self.assertEqual(be.stamped["gb_search_1"], 25)
        # job 640 -> 644 resumed at 30; the clock must NOT restart there
        self.assertEqual(self._setup_run(be, 30), 25)
        self.assertEqual(be.stamped["gb_search_1"], 25)

    def test_armed_ignores_a_stored_start_past_the_live_iteration(self):
        """A rewound store (rows dropped below the stamp) carries a start the
        stage cannot have had; restoring it would make the clock read as
        already elapsed. Ignore it and re-stamp the live iteration (user ruling
        2026-10-07: the stamp resets, it is never trusted blindly)."""
        os.environ["GF_PERSIST_STAGE_START"] = "1"
        be = _FakeBackend(30, np.zeros(1))
        be.stamped["gb_search_1"] = 50
        self.assertEqual(self._setup_run(be, 30), 30)
        self.assertEqual(be.stamped["gb_search_1"], 30)
        # and from then on it is the ordinary resume case
        self.assertEqual(self._setup_run(be, 35), 30)

    def test_a_backend_without_the_api_falls_back_silently(self):
        """A backend predating the two methods must not raise -- it just
        keeps the pre-2026-09-26 behaviour."""
        os.environ["GF_PERSIST_STAGE_START"] = "1"
        be = SimpleNamespace(iteration=42)                # no stamp/read API
        self.assertEqual(self._setup_run(be, 42), 42)


if __name__ == "__main__":
    unittest.main()
