import os
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.globalfitmove import MaxLogLCombineMove


class _ScriptedMaxLogL(MaxLogLCombineMove):
    """Exercises ONLY the plateau loop: GFCombineMove.__init__ is skipped and
    _propose_moves_once returns a scripted cold-chain max-lnL sequence.

    Each scripted entry is either a scalar (one walker) or a sequence (one
    value per walker).

    ⚠ IT MUST RETURN A NON-None ``accepted``. The loop condition is
    ``num_so_far < num_checks OR accepted is None`` -- the second clause
    guarantees at least one inner iteration -- so a harness returning None
    never lets the plateau govern and every case here silently ran to the
    MAXLOGL_ITERS_PER_STEP cap of 10 instead. That is what had this whole
    module failing (2 failures + 2 errors) before 2026-09-26; the assertions
    were right and the harness was wrong.
    """

    def __init__(self, seq, num_checks=5, tol=5.0, max_iter=0,
                 stage_kind="search"):
        self.num_checks = num_checks
        self.tol = tol
        self.max_iter = max_iter
        self.seq = list(seq)
        self.calls = 0
        self.gf_stage_kind = stage_kind

    def _propose_moves_once(self, model, state):
        val = np.atleast_1d(np.asarray(self.seq[self.calls], dtype=float))
        self.calls += 1
        ll = val.reshape(1, -1)
        return SimpleNamespace(log_like=ll, log_prior=np.zeros_like(ll),
                               branches={}), True


def _run(seq, **kwargs):
    move = _ScriptedMaxLogL(seq, **kwargs)
    n = np.atleast_1d(np.asarray(seq[0], dtype=float)).size
    state0 = SimpleNamespace(log_like=np.full((1, n), -np.inf),
                             log_prior=np.zeros((1, n)), branches={})
    move._propose_moves(model=None, state=state0)
    return move.calls


class MaxLogLPlateauTest(unittest.TestCase):
    def setUp(self):
        os.environ["MAXLOGL_LOG_EVERY"] = "0"
        # These single-walker cases are the SCALAR rule by construction; pin
        # the knob so they read the same under either default.
        os.environ["MAXLOGL_PER_WALKER"] = "0"
        # These cases are about the PLATEAU rule, not the per-call chunking.
        # Unset, MAXLOGL_ITERS_PER_STEP defaults to 10 and silently truncates
        # every sequence longer than that -- the other half of why this module
        # was failing before 2026-09-26.
        os.environ["MAXLOGL_ITERS_PER_STEP"] = "500"
        for k in ("MAXLOGL_LOG_EVERY", "MAXLOGL_PER_WALKER",
                  "MAXLOGL_ITERS_PER_STEP"):
            self.addCleanup(os.environ.pop, k, None)

    def test_soft_tol_ignores_noise_floor_twitches(self):
        # Strict sub-tol improvements (the +2-scale polishing observed on the
        # cluster) must count as flat: baseline 1000, everything <= 1005.
        seq = [1000, 1001, 1002, 1003, 1000, 1000, 1000, 1000, 1000]
        self.assertEqual(_run(seq, num_checks=5, tol=5.0), 6)

    def test_tol_zero_restores_strict_rule(self):
        # Same sequence, tol=0: every new max resets the counter.
        seq = [1000, 1001, 1002, 1003, 1000, 1000, 1000, 1000, 1000]
        self.assertEqual(_run(seq, num_checks=5, tol=0.0), 9)

    def test_significant_climb_keeps_stage_alive(self):
        # Super-tol jumps reset; the plateau tail then ends it.
        seq = [1000, 1010, 1020, 1030] + [1030] * 6
        # 1030 == baseline exactly -> flat; needs changed_once (set at 1010).
        self.assertEqual(_run(seq, num_checks=5, tol=5.0), 9)

    def test_slow_accumulated_climb_is_progress(self):
        # +2/iter forever: each step is sub-tol but the baseline does NOT
        # advance on sub-tol steps, so the accumulated climb (+10 per
        # 5-window) keeps resetting -- the stage must not exit early.
        seq = [1000 + 2 * i for i in range(30)]
        move = _ScriptedMaxLogL(seq, num_checks=5, tol=5.0, max_iter=20)
        state0 = SimpleNamespace(log_like=np.array([[-np.inf]]))
        move._propose_moves(model=None, state=state0)
        self.assertEqual(move.calls, 20)  # only the ceiling stops it

    def test_max_iter_ceiling(self):
        seq = [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070]
        self.assertEqual(_run(seq, num_checks=5, tol=5.0, max_iter=3), 3)


class PerWalkerPlateauTest(unittest.TestCase):
    """The laggard decides, not the best walker (user ruling 2026-09-26).

    The rule this replaces took ``state.log_like[0].max()`` -- the BEST
    walker -- so "joint" meant joint over the wrapped MOVES and never over
    the walkers. Measured cost on the first 3-month v9 run (job 636, stored
    row 0): plateau declared at best=52494067.95 while the other three cold
    walkers sat 71,050 / 128,080 / 1,600 lnL below it, and because
    gb_search_1/2 then froze the noise, the four walkers searched against
    four different noise floors for the rest of the run.
    """

    def setUp(self):
        os.environ["MAXLOGL_LOG_EVERY"] = "0"
        os.environ["MAXLOGL_PER_WALKER"] = "1"
        # default OFF since it shipped broken; these cases arm it
        os.environ["MAXLOGL_FREEZE_CONVERGED"] = "1"
        os.environ["MAXLOGL_ITERS_PER_STEP"] = "500"
        for k in ("MAXLOGL_LOG_EVERY", "MAXLOGL_PER_WALKER",
                  "MAXLOGL_ITERS_PER_STEP", "MAXLOGL_FREEZE_CONVERGED"):
            self.addCleanup(os.environ.pop, k, None)

    @staticmethod
    def _seq(n, best_flat_at, lag_flat_at):
        """Best walker tops out early; a LOWER walker keeps climbing."""
        return [(2000.0 + 20.0 * min(i, best_flat_at),
                 1000.0 + 20.0 * min(i, lag_flat_at)) for i in range(n)]

    def test_a_still_climbing_laggard_holds_the_stage_open(self):
        seq = self._seq(80, best_flat_at=3, lag_flat_at=30)
        # The best walker is flat from round 4, so the scalar rule would stop
        # at 9. The laggard climbs to round 31 and then needs its own 5 flat.
        self.assertEqual(_run(seq, num_checks=5, tol=5.0), 36)

    def test_the_old_scalar_rule_stops_at_the_best_walker(self):
        """The paired control: same sequence, MAXLOGL_PER_WALKER=0."""
        os.environ["MAXLOGL_PER_WALKER"] = "0"
        seq = self._seq(80, best_flat_at=3, lag_flat_at=30)
        self.assertEqual(_run(seq, num_checks=5, tol=5.0), 9)

    def test_a_walker_that_never_moves_cannot_deadlock_the_stage(self):
        """``changed_once`` is a GLOBAL latch, and this is why.

        Armed per-walker, a walker flat from round 1 never arms, so its
        counter never increments, so the laggard rule never clears and the
        stage runs to MAXLOGL_MAX_ITER -- which defaults to 0, unbounded.
        """
        seq = [(1000.0, 1000.0 + 20.0 * min(i, 12)) for i in range(80)]
        self.assertEqual(_run(seq, num_checks=5, tol=5.0), 18)

    def test_every_walker_must_clear_its_own_flat_window(self):
        seq = [(1000.0 + 20.0 * min(i, 4), 1000.0 + 20.0 * min(i, 9))
               for i in range(60)]
        # walker 1 tops out at round 10, then 5 flat rounds -> 15.
        self.assertEqual(_run(seq, num_checks=5, tol=5.0), 15)

    def test_a_frozen_walker_restores_BOTH_representations(self):
        """The regression for what killed the first relaunch.

        A branch with a tempered ModuleSubState is stored twice -- the main
        engine state and ``state.sub_states[name]`` -- and
        ``_check_substate_consistency`` compares the main cold row against
        the sub-state's row 0 on EVERY propose. The first version of the
        freeze restored only the main state, so rank 1 died with
        "[psd] cold-chain coords mismatch between the main state and its
        sub-state (1 of 2 alive leaves differ)".
        """
        ntemps, nw, nleaves, ndim = 2, 2, 2, 3

        def _holder(fill):
            return SimpleNamespace(
                coords=np.full((ntemps, nw, nleaves, ndim), fill, dtype=float),
                inds=np.ones((ntemps, nw, nleaves), dtype=bool),
                tempered_initialized=True)

        main, sub = _holder(1.0), _holder(1.0)
        state = SimpleNamespace(
            log_like=np.full((1, nw), -np.inf), log_prior=np.zeros((1, nw)),
            branches={"psd": main}, sub_states={"psd": sub})
        mask = np.array([True, False])          # freeze walker 0 only
        snap = MaxLogLCombineMove._snapshot_walkers(state, mask)
        self.assertIsNotNone(snap)
        self.assertIn("psd", snap["subs"], "the sub-state must be captured")

        # A move now scribbles on BOTH representations for both walkers.
        main.coords[...] = 9.0
        sub.coords[...] = 9.0
        MaxLogLCombineMove._restore_walkers(state, snap)

        # Walker 0 is back to 1.0 in BOTH; walker 1 keeps the move's 9.0.
        for label, holder in (("main", main), ("sub", sub)):
            self.assertTrue((holder.coords[:, 0] == 1.0).all(),
                            f"{label}: frozen walker not restored")
            self.assertTrue((holder.coords[:, 1] == 9.0).all(),
                            f"{label}: laggard must NOT be restored")
        # ... and the two agree, which is what check_cold_row enforces.
        np.testing.assert_array_equal(main.coords[0], sub.coords[0])

    def test_converged_walkers_are_frozen_in_a_search_stage(self):
        """A walker past its flat window stops moving; the laggard does not."""
        move = _ScriptedMaxLogL(
            [(1000.0, 1000.0 + 20.0 * min(i, 20)) for i in range(80)],
            num_checks=5, tol=5.0, stage_kind="search")
        n = 2
        state0 = SimpleNamespace(log_like=np.full((1, n), -np.inf),
                                 log_prior=np.zeros((1, n)), branches={})
        move._propose_moves(model=None, state=state0)
        ml = move._ml_state
        self.assertEqual(ml["max_logl"].size, n, "per-walker state expected")
        self.assertTrue((ml["num_so_far"] >= move.num_checks).all())

    def test_no_freeze_when_riding_inside_gb_search(self):
        """Restoring rows there would write a STALE log_like: the GB residual
        moves between calls, so the freeze is search-kind only."""
        # Something must MOVE, or the changed_once guard correctly keeps the
        # stage open: a move that has never taken effect may not trip the exit.
        seq = [(1000.0, 1000.0 + 20.0 * min(i, 2)) for i in range(40)]
        move = _ScriptedMaxLogL(seq, num_checks=5, tol=5.0,
                                stage_kind="gb_search")
        state0 = SimpleNamespace(log_like=np.full((1, 2), -np.inf),
                                 log_prior=np.zeros((1, 2)), branches={})
        move._propose_moves(model=None, state=state0)
        self.assertTrue(move.maxlogl_plateau_done)


if __name__ == "__main__":
    unittest.main()
