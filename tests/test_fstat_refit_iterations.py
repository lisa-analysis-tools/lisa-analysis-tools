"""``GB_FSTAT_REFIT_EVERY`` counts global-fit ITERATIONS (ruling 2026-09-18).

It used to count the shared per-branch PROPOSE census, which is a
stage-dependent multiple of the iteration count, so the same ``=50`` meant
a different cadence in every stage. Measured on the two production runs:

* 3-month ``gb_search``: **2.00** GB-branch proposes/iteration
  (``rj_fstat_search`` + ``rj_prior_removal``) -> refit every 25 iterations;
* 6-month ``gb_search``: **3.00** (warm start adds ``rj_warm_search``)
  -> refit every 17 iterations;
* randomized ``full_pe``: the GB branch is proposed roughly a sixth as
  often -> ~150 iterations.

Both runs' epoch manifests show Delta(clock) == 50 exactly at every gap, so
the knob was firing precisely -- on the wrong unit. No fixed divisor can
reconcile three stage multiplicities, which is why the clock now counts
iterations directly off the stamp the stage combine mints.

The propose census is deliberately LEFT ALONE here: ``_temper_cadence_fire``
reads it, and retuning the tempering cadence as a side effect of an F-stat
change would be exactly the kind of silent coupling this repo keeps biting
on.
"""

import unittest

import numpy as np


class _Clock:
    """``_fstat_clock``'s tick logic, lifted off the move without its ctor."""

    from lisatools.globalfit.moves.gbspecialstretch import (
        GBSpecialBase, GBSpecialRJFStatGridMove,
    )

    def __init__(self, branch="gb"):
        self.branch_name = branch
        type(self).GBSpecialBase._branch_iteration_counts.clear()
        type(self).GBSpecialBase._branch_iteration_seen.clear()
        type(self).GBSpecialBase._branch_propose_counts.clear()

    def tick(self, gf_iteration):
        """The exact body of the new clock's counting half."""
        B = type(self).GBSpecialBase
        branch = self.branch_name
        counts, seen = B._branch_iteration_counts, B._branch_iteration_seen
        if gf_iteration is None:
            counts[branch] = int(counts.get(branch, 0)) + 1
        elif seen.get(branch) != gf_iteration:
            seen[branch] = gf_iteration
            counts[branch] = int(counts.get(branch, 0)) + 1
        return int(counts.get(branch, 0))


class CountsIterationsNotProposesTest(unittest.TestCase):
    def test_three_proposes_in_one_iteration_tick_once(self):
        """The 6-month gb_search shape: 3 GB moves per iteration."""
        c = _Clock()
        for _ in range(3):
            c.tick(7)
        self.assertEqual(c.tick(7), 1)

    def test_two_proposes_in_one_iteration_tick_once(self):
        """The 3-month gb_search shape."""
        c = _Clock()
        c.tick(4)
        self.assertEqual(c.tick(4), 1)

    def test_fifty_iterations_give_exactly_fifty_ticks(self):
        """Whatever the per-iteration propose multiplicity is."""
        for per_iter in (1, 2, 3, 6):
            c = _Clock()
            for it in range(50):
                for _ in range(per_iter):
                    c.tick(it)
            self.assertEqual(c.tick(49), 50, f"{per_iter} proposes/iteration")

    def test_the_OLD_behaviour_would_have_differed_by_the_multiplicity(self):
        """Pin the bug being fixed, so a regression is visible."""
        old_3mo, old_6mo = 50 * 2, 50 * 3      # propose census after 50 its
        self.assertNotEqual(old_3mo, 50)
        self.assertNotEqual(old_6mo, 50)
        # ... and the intervals those produced at REFIT_EVERY=50
        self.assertEqual(round(50 / 2.0), 25)   # 3-month measured
        self.assertEqual(round(50 / 3.0), 17)   # 6-month measured

    def test_a_skipped_iteration_still_ticks_only_once(self):
        """Stages where GB is not proposed every iteration must not
        back-fill: the clock counts iterations the branch RAN in."""
        c = _Clock()
        for it in (0, 6, 12, 18):
            c.tick(it)
        self.assertEqual(c.tick(18), 4)

    def test_an_unstamped_move_falls_back_to_per_visit_counting(self):
        """A direct call with no combine above it must stay monotone."""
        c = _Clock()
        for _ in range(5):
            c.tick(None)
        self.assertEqual(c.tick(None), 6)

    def test_branches_count_independently(self):
        gb, vgb = _Clock("gb"), _Clock("vgb")   # second ctor clears; re-tick
        gb.tick(0); gb.tick(0); gb.tick(1)
        vgb.tick(0)
        B = _Clock.GBSpecialBase
        self.assertEqual(B._branch_iteration_counts["gb"], 2)
        self.assertEqual(B._branch_iteration_counts["vgb"], 1)

    def test_the_propose_census_is_UNTOUCHED_by_the_iteration_clock(self):
        """_temper_cadence_fire reads that census; it must not move here."""
        c = _Clock()
        for it in range(10):
            c.tick(it)
        self.assertEqual(
            _Clock.GBSpecialBase._branch_propose_counts.get("gb", 0), 0)


class StampPlumbingTest(unittest.TestCase):
    """The stamp is minted once per iteration by the STAGE combine."""

    def test_only_the_stage_combine_increments(self):
        from lisatools.globalfit.moves.globalfitmove import GFCombineMove

        class _M:
            pass

        stage, nested = _M(), _M()
        stage.gf_is_stage_combine = True
        # the body of GFCombineMove.propose's counter, applied to each
        for obj in (stage, nested):
            if getattr(obj, "gf_is_stage_combine", False):
                obj.gf_iteration = int(getattr(obj, "gf_iteration", -1)) + 1
        self.assertEqual(stage.gf_iteration, 0)
        self.assertFalse(hasattr(nested, "gf_iteration"))
        self.assertTrue(hasattr(GFCombineMove, "propose"))

    def test_the_stage_combine_starts_at_zero_and_advances_by_one(self):
        class _M:
            gf_is_stage_combine = True

        m = _M()
        seen = []
        for _ in range(4):
            m.gf_iteration = int(getattr(m, "gf_iteration", -1)) + 1
            seen.append(m.gf_iteration)
        self.assertEqual(seen, [0, 1, 2, 3])

    def test_recipe_marks_the_stage_combine(self):
        import inspect

        from lisatools.globalfit import recipe

        src = inspect.getsource(recipe)
        self.assertIn("gf_is_stage_combine = True", src)

    def test_prepare_child_stamps_the_iteration(self):
        import inspect

        from lisatools.globalfit.moves import globalfitmove

        src = inspect.getsource(globalfitmove)
        self.assertIn("move.gf_iteration = it", src)


class MultiRankShipTest(unittest.TestCase):
    """Ranks never see a combine, so the stamp rides in the payload."""

    def test_clock_vals_ships_and_applies_gf_iteration(self):
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch

        src = inspect.getsource(gbspecialstretch)
        self.assertIn('"gf_iteration": (None if getattr(self, "gf_iteration"', src)
        self.assertIn('self.gf_iteration = int(cv["gf_iteration"])', src)

    def test_a_rank_applying_the_stamp_ticks_like_the_head(self):
        head, rank = _Clock(), None
        B = _Clock.GBSpecialBase
        for it in range(5):
            head.tick(it)
        head_total = B._branch_iteration_counts["gb"]
        B._branch_iteration_counts.clear()
        B._branch_iteration_seen.clear()
        rank = _Clock()
        for it in range(5):          # same stamps, 3 commands each
            for _ in range(3):
                rank.tick(it)
        self.assertEqual(B._branch_iteration_counts["gb"], head_total)


if __name__ == "__main__":
    unittest.main()


class ClockJournalGranularityTest(unittest.TestCase):
    """The refit clock must not lose a short job's progress.

    REGRESSION (2026-09-27). ``_FSTAT_CLOCK_WRITE_EVERY`` was 10 and the
    clock is SEEDED from ``clock.json`` on every process start, so a job
    shorter than 10 iterations contributed nothing -- the next process
    re-seeded from the same stale value. Measured on 6mo v9 job 650:
    epoch 4 stamped clock=5 at 21:11 and the next refit came EIGHTEEN
    HOURS later, because jobs 646/648 and the first third of 650 were
    spent climbing back to a value the run had already reached.
    """

    def test_the_journal_is_written_every_tick(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove as M)
        self.assertEqual(
            M._FSTAT_CLOCK_WRITE_EVERY, 1,
            "a granularity above 1 loses every job shorter than it")

    def test_the_clamp_cannot_leave_the_clock_below_its_last_fit(self):
        """A clock behind its own last-fit mark can never refit again.

        ``clock - _fstat_last_fit_hit`` goes negative and no cadence
        elapses, silently and with no error -- the shape of the 18-hour
        stall.
        """
        import inspect
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove as M)
        src = inspect.getsource(M._fstat_clock)
        self.assertIn("_fstat_last_fit_hit", src)
        self.assertIn("counts[branch] = _lf", src)


class ForcedRefitStalenessTest(unittest.TestCase):
    """A forced refit declines only for a FRESH epoch, not any epoch.

    The decline exists because the arming decision is process-global and
    so fires on every restart, which would otherwise open a fresh epoch
    (a full comb scan) per resume. But it used to key on the peak FLOOR
    alone, so an epoch fitted against a residual many iterations old was
    reloaded whatever its age -- handing the search a birth grid for a
    model that had moved on.
    """

    def test_it_measures_age_with_the_refit_cadence(self):
        import inspect
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove as M)
        src = inspect.getsource(M._consume_forced_refit)
        # the two halves of the new condition
        self.assertIn("_floor_same", src)
        self.assertIn("_stale", src)
        # staleness is the SAME cadence the ordinary refit uses, so the
        # two cannot disagree about what "old" means
        self.assertIn("self.fstat_refit_every", src)
        self.assertIn("_epoch_fit_clock", src)

    def test_a_fresh_epoch_at_the_same_floor_is_still_reloaded(self):
        """The restart case the decline exists for must still work."""
        import inspect
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove as M)
        src = inspect.getsource(M._consume_forced_refit)
        self.assertIn("if _floor_same and not _stale:", src)
        self.assertIn("return None",
                      src.split("if _floor_same and not _stale:")[1][:800])
