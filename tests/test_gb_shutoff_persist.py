"""The per-(walker, band) RJ shutoff valve and its WINDOW are persisted.

Two halves, because the feature has two halves that fail independently:

1. **The state layer.** ``band_shutoff_best_w`` / ``band_shutoff_streak_w``
   live in ``band_info``, so the backend's ordinary band-array channel
   carries them; the move holds references to those very objects, so its
   in-place updates are what gets saved; and a store written before they
   existed backfills them WITHOUT losing the boolean valve it had earned.

2. **The migration.** ``PerBranchHDFBackend.save_step`` silently skips any
   array whose dataset the file does not already have, so a store created
   before the family existed drops it on every save forever. That is the
   state the v9 production runs are in (measured 2026-09-27: the
   ``sub_backend/gb`` group has ``band_rj_shutoff`` but none of the ``_w``
   family). ``migrate_gb_shutoff_persist.py`` adds the datasets.

Together these are what stops the valve re-earning its whole window on
every restart -- which, at the 6mo run's ~1.8 h iteration and a 3-iteration
patience window, meant no pair could shut in any job under about seven
hours.
"""
import os
import subprocess
import sys
import tempfile
import unittest

import h5py
import copy

import numpy as np

from lisatools.globalfit.state import (SEARCH_SHUTOFF_WINDOW_FIELDS,
                                       ensure_search_shutoff_fields,
                                       ensure_search_shutoff_window)

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "fstat_proposal", "migrate_gb_shutoff_persist.py",
)

NW, NB = 3, 5


def _band_info(with_window=True, with_valve=True):
    bi = {"nwalkers": NW, "num_bands": NB}
    if with_valve:
        bi["band_rj_shutoff_w"] = np.zeros((NW, NB), dtype=bool)
        bi["band_shutoff_w_step"] = np.full(1, -1, dtype=np.int64)
    if with_window:
        bi["band_shutoff_best_w"] = np.full((NW, NB), -np.inf)
        bi["band_shutoff_streak_w"] = np.zeros((NW, NB), dtype=np.int64)
        bi["band_cold_logl_max_w"] = np.full((NW, NB), -np.inf)
        bi["band_cold_logl_w"] = np.full((NW, NB), -np.inf)
        bi["band_shutoff_reset_w"] = np.zeros((NW, NB), dtype=np.int64)
        bi["band_cold_logl_peak_w"] = np.full((NW, NB), -np.inf)
    return bi


class ShutoffWindowStateTest(unittest.TestCase):

    def test_fresh_window_is_minus_inf_and_zero(self):
        bi = _band_info(with_window=False)
        self.assertEqual(
            ensure_search_shutoff_window(bi, NB, NW), "fresh")
        self.assertTrue(np.all(np.isneginf(bi["band_shutoff_best_w"])))
        self.assertTrue(np.all(bi["band_shutoff_streak_w"] == 0))

    def test_minus_inf_is_the_load_bearing_fresh_best(self):
        """A fresh window must make the FIRST measured lnL an improvement.

        If the best started at 0.0 (or anything finite and low) a band
        whose lnL happened to sit below it would begin its patience count
        on iteration one, and could shut a whole window early.
        """
        bi = _band_info(with_window=False)
        ensure_search_shutoff_window(bi, NB, NW)
        self.assertTrue(
            np.all(bi["band_shutoff_best_w"] < -1e300),
            "a finite fresh best would let a band start its patience "
            "count before any improvement had been observed")

    def test_an_old_store_keeps_its_valve_and_gets_a_fresh_window(self):
        """THE MIGRATION-COMPATIBILITY CASE.

        A band_info from a store written before 2026-09-27 has the boolean
        and the step stamp but neither window array. It must NOT be
        treated as an incomplete record: reopening a valve the run
        legitimately earned would throw away real convergence.
        """
        bi = _band_info(with_window=False)
        bi["band_rj_shutoff_w"][0, 1] = True
        origin = ensure_search_shutoff_fields(bi, NB, per_walker=True)
        self.assertTrue(origin.startswith("restored"), origin)
        self.assertIn("window fresh", origin)
        self.assertTrue(bool(bi["band_rj_shutoff_w"][0, 1]),
                        "the earned valve was discarded")
        for name in SEARCH_SHUTOFF_WINDOW_FIELDS:
            self.assertEqual(np.shape(bi[name]), (NW, NB))

    def test_a_complete_record_round_trips_untouched(self):
        bi = _band_info()
        bi["band_rj_shutoff_w"][2, 3] = True
        bi["band_shutoff_best_w"][2, 3] = -17.5
        bi["band_shutoff_streak_w"][2, 3] = 2
        self.assertEqual(
            ensure_search_shutoff_fields(bi, NB, per_walker=True), "restored")
        self.assertEqual(float(bi["band_shutoff_best_w"][2, 3]), -17.5)
        self.assertEqual(int(bi["band_shutoff_streak_w"][2, 3]), 2)

    def test_dtypes_are_pinned_after_a_float_round_trip(self):
        """A migrated / hand-built band_info may hand back float streaks."""
        bi = _band_info()
        bi["band_shutoff_streak_w"] = np.zeros((NW, NB), dtype=np.float32)
        bi["band_shutoff_best_w"] = np.full((NW, NB), -np.inf,
                                            dtype=np.float32)
        ensure_search_shutoff_window(bi, NB, NW)
        self.assertEqual(bi["band_shutoff_streak_w"].dtype, np.int64)
        self.assertEqual(bi["band_shutoff_best_w"].dtype, np.float64)

    def test_a_grid_change_restarts_the_window(self):
        bi = _band_info()
        bi["band_shutoff_best_w"] = np.full((NW, NB + 2), -3.0)
        bi["band_shutoff_streak_w"] = np.ones((NW, NB + 2), dtype=np.int64)
        bi["band_cold_logl_max_w"] = np.full((NW, NB + 2), -3.0)
        bi["band_cold_logl_w"] = np.full((NW, NB + 2), -3.0)
        bi["band_shutoff_reset_w"] = np.zeros((NW, NB + 2), dtype=np.int64)
        bi["band_cold_logl_peak_w"] = np.full((NW, NB + 2), -3.0)
        self.assertEqual(
            ensure_search_shutoff_window(bi, NB, NW), "reset(shape)")
        self.assertEqual(np.shape(bi["band_shutoff_best_w"]), (NW, NB))
        self.assertTrue(np.all(bi["band_shutoff_streak_w"] == 0))

    def test_zeroing_the_valve_also_zeroes_the_window(self):
        """A release must clear BOTH or the next step re-freezes at once.

        The model is already fitted when a step ends, so an inherited
        running best is unbeatable: every pair would post "no improvement"
        on its first patience window and read as converged.
        """
        from lisatools.globalfit.state import _zero_search_shutoff
        bi = _band_info()
        bi["band_rj_shutoff_w"][:] = True
        bi["band_shutoff_best_w"][:] = 500.0
        bi["band_shutoff_streak_w"][:] = 9
        _zero_search_shutoff(bi, NB, NW)
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self.assertTrue(np.all(np.isneginf(bi["band_shutoff_best_w"])))
        self.assertTrue(np.all(bi["band_shutoff_streak_w"] == 0))

    def test_the_window_reaches_storage_arrays(self):
        """The whole point: the saver must actually see them.

        ``GBState.storage_arrays`` returns every ndarray in band_info bar
        the statics, so this is really a guard that nothing has quietly
        started filtering the band_info dict by an allow-list.
        """
        from lisatools.globalfit.state import GBState
        st = GBState.make_template(
            NW, 2, num_bands=NB,
            band_edges=np.linspace(1e-3, 2e-3, NB + 1),
        )
        bi = st.band_info
        bi.setdefault("nwalkers", NW)
        ensure_search_shutoff_window(bi, NB, NW)
        names = set(st.storage_arrays())
        for name in SEARCH_SHUTOFF_WINDOW_FIELDS:
            self.assertIn(
                name, names,
                f"{name} is in band_info but not in storage_arrays(), so "
                f"the saver would never write it")


def _make_store(path, nstep=4, with_family=False):
    with h5py.File(path, "w") as f:
        root = f.create_group("global_fit")
        root.attrs["iteration"] = nstep - 1
        gb = root.create_group("sub_backend").create_group("gb")
        gb.attrs["num_bands"] = NB
        gb.attrs["num_cap_cells"] = NB
        gb.attrs["ntemps"] = 2
        gb.attrs["nwalkers"] = NW
        gb.create_dataset("band_edges", data=np.linspace(1e-3, 2e-3, NB + 1))
        # the 1-D valve the v9 stores DO have -- also the sibling the
        # migration takes its step axis from
        gb.create_dataset("band_rj_shutoff",
                          data=np.zeros((nstep, NB), dtype=bool),
                          maxshape=(None, NB))
        if with_family:
            gb.create_dataset("band_rj_shutoff_w",
                              data=np.zeros((nstep, NW, NB), dtype=bool),
                              maxshape=(None, NW, NB))


def _run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args],
                          capture_output=True, text=True)


class MigrationTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "store.h5")

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_the_four_datasets_with_the_fresh_fill(self):
        _make_store(self.path)
        r = _run("--force", self.path)
        self.assertEqual(r.returncode, 0, r.stderr)
        with h5py.File(self.path, "r") as f:
            gb = f["global_fit"]["sub_backend"]["gb"]
            self.assertEqual(gb["band_rj_shutoff_w"].shape, (4, NW, NB))
            self.assertEqual(gb["band_shutoff_best_w"].shape, (4, NW, NB))
            self.assertEqual(gb["band_shutoff_streak_w"].shape, (4, NW, NB))
            self.assertEqual(gb["band_shutoff_w_step"].shape, (4, 1))
            self.assertTrue(np.all(np.isneginf(gb["band_shutoff_best_w"][-1])))
            self.assertTrue(np.all(gb["band_shutoff_streak_w"][-1] == 0))
            self.assertFalse(gb["band_rj_shutoff_w"][-1].any())
            self.assertEqual(int(gb["band_shutoff_w_step"][-1][0]), -1)

    def test_first_axis_stays_growable(self):
        """``grow()`` resizes every non-static dataset in the group.

        A fixed first axis here would make the FIRST save after the
        migration raise and take the run down -- the opposite of the
        point.
        """
        _make_store(self.path)
        self.assertEqual(_run("--force", self.path).returncode, 0)
        with h5py.File(self.path, "r+") as f:
            gb = f["global_fit"]["sub_backend"]["gb"]
            for name in ("band_rj_shutoff_w", "band_shutoff_w_step",
                         "band_shutoff_best_w", "band_shutoff_streak_w"):
                self.assertIsNone(gb[name].maxshape[0], name)
                gb[name].resize(9, axis=0)
            gb["band_shutoff_best_w"][8] = np.full((NW, NB), -4.25)
            self.assertEqual(float(gb["band_shutoff_best_w"][8][0, 0]), -4.25)

    def test_step_axis_comes_from_a_sibling_not_the_iteration_attr(self):
        """grow() pre-allocates ahead of the iteration counter.

        Sizing the new datasets from ``iteration`` would leave them short,
        and the next save would write past their end.
        """
        _make_store(self.path, nstep=12)   # iteration attr says 11
        self.assertEqual(_run("--force", self.path).returncode, 0)
        with h5py.File(self.path, "r") as f:
            gb = f["global_fit"]["sub_backend"]["gb"]
            self.assertEqual(gb["band_shutoff_best_w"].shape[0], 12)

    def test_idempotent(self):
        _make_store(self.path, with_family=True)
        r = _run("--force", self.path)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("present", r.stdout)
        # the one it already had is untouched; the other three are added
        self.assertIn("created band_shutoff_best_w", r.stdout)
        self.assertNotIn("created band_rj_shutoff_w", r.stdout)

    def test_dry_run_touches_nothing(self):
        _make_store(self.path)
        before = os.path.getmtime(self.path)
        r = _run("--dry-run", self.path)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dry run", r.stdout)
        self.assertEqual(os.path.getmtime(self.path), before)
        with h5py.File(self.path, "r") as f:
            self.assertNotIn(
                "band_shutoff_best_w", f["global_fit"]["sub_backend"]["gb"])

    def test_refuses_a_store_that_looks_live(self):
        """Two writers on one HDF5 file corrupt it."""
        _make_store(self.path)
        r = _run(self.path)        # just written, so it looks live
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("still writing", r.stdout + r.stderr)
        with h5py.File(self.path, "r") as f:
            self.assertNotIn(
                "band_shutoff_best_w", f["global_fit"]["sub_backend"]["gb"])

    def test_writes_a_backup_by_default(self):
        _make_store(self.path)
        self.assertEqual(_run("--force", self.path).returncode, 0)
        self.assertTrue(os.path.exists(self.path + ".bak"))

    def test_refuses_to_clobber_an_existing_backup(self):
        _make_store(self.path)
        open(self.path + ".bak", "w").close()
        r = _run("--force", self.path)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already exists", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()


class ResetDiagnosticTest(unittest.TestCase):
    """``band_shutoff_reset_w`` separates the three ways to be unshut.

    USER RULING 2026-09-27: "they are resetting? we need to diagnose that
    or add more tracking info to see what happened."

    Streak alone is ambiguous -- a pair sitting at streak 0 may be a
    chronic resetter or may have been occupied for the first time last
    iteration. The reset COUNT tells them apart, and the two together
    with ``band_cold_logl_w`` / ``band_cold_logl_max_w`` say where the
    pair is, what bar it is held to, and how it has behaved.
    """

    def test_a_reset_is_counted_only_when_a_streak_is_actually_broken(self):
        streak = np.array([0, 1, 3, 2], dtype=np.int64)
        resets = np.zeros(4, dtype=np.int64)
        improved = np.array([True, True, False, True])
        # the rule as implemented in _update_search_band_shutoff
        resets[improved & (streak > 0)] += 1
        self.assertEqual(list(resets), [0, 1, 0, 1],
                         "a pair already at streak 0 has no streak to break")

    def test_the_three_unshut_populations_are_distinguishable(self):
        """chronic resetter / still young / newly occupied."""
        streak = np.array([0, 2, 0], dtype=np.int64)
        resets = np.array([17, 0, 0], dtype=np.int64)
        chronic = (resets > 0) & (streak < 3)
        young = (resets == 0) & (streak > 0)
        fresh = (resets == 0) & (streak == 0)
        self.assertEqual(list(chronic), [True, False, False])
        self.assertEqual(list(young), [False, True, False])
        self.assertEqual(list(fresh), [False, False, True])

    def test_the_counter_is_in_the_persisted_window_group(self):
        from lisatools.globalfit.state import SEARCH_SHUTOFF_WINDOW_FIELDS
        for name in ("band_shutoff_reset_w", "band_cold_logl_w",
                     "band_cold_logl_max_w"):
            self.assertIn(name, SEARCH_SHUTOFF_WINDOW_FIELDS)

    def test_the_migration_creates_all_of_them(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            _make_store(path)
            r = _run("--force", path)
            self.assertEqual(r.returncode, 0, r.stderr)
            with h5py.File(path, "r") as f:
                gb = f["global_fit"]["sub_backend"]["gb"]
                for name in ("band_cold_logl_w", "band_cold_logl_max_w",
                             "band_shutoff_reset_w"):
                    self.assertIn(name, gb, name)
                    self.assertIsNone(gb[name].maxshape[0], name)


class ColdBandLnLTrackerTest(unittest.TestCase):
    """The tracker is owned by the SUB-STATE and not by any move.

    USER RULING 2026-09-27: "we should pass an object or set of arrays as
    part of the GB sub state object ... it should not be held by a
    specific move."
    """

    def _bi(self):
        bi = _band_info()
        from lisatools.globalfit.moves.gbspecialstretch import (
            cold_band_lnl_from_band_info)
        return bi, cold_band_lnl_from_band_info(bi, (NW, NB))

    def test_the_view_aliases_band_info_it_does_not_copy(self):
        bi, v = self._bi()
        v.peak[0, 0] = 12.5
        self.assertEqual(float(bi["band_cold_logl_peak_w"][0, 0]), 12.5,
                         "the view copied instead of aliasing, so nothing "
                         "it writes would ever reach the saver")

    def test_observe_takes_the_MAX_not_the_last_write(self):
        """A block can carry two rows for one (walker, band) after a
        vertical swap relabels rungs."""
        _, v = self._bi()
        v.observe([1, 1], [2, 2], [5.0, 3.0])
        self.assertEqual(float(v.peak[1, 2]), 5.0)

    def test_observe_ignores_non_finite(self):
        _, v = self._bi()
        v.observe([0], [0], [np.nan])
        self.assertTrue(np.isneginf(v.peak[0, 0]))

    def test_judge_consumes_the_peak_and_keeps_the_max(self):
        _, v = self._bi()
        v.observe([0], [0], [10.0])
        occ = np.ones((NW, NB), dtype=np.int64)
        v.judge(4.0, 3, occ)
        self.assertEqual(float(v.max[0, 0]), 10.0)
        self.assertTrue(np.isneginf(v.peak[0, 0]), "peak was not reset")
        self.assertEqual(float(v.value[0, 0]), 10.0)

    def test_a_cell_never_observed_advances_its_streak(self):
        """Nothing refined it, so it did not get better."""
        _, v = self._bi()
        occ = np.ones((NW, NB), dtype=np.int64)
        for _ in range(3):
            v.judge(4.0, 3, occ)
        self.assertTrue(np.all(v.streak >= 3))

    def test_release_keeps_the_all_time_max(self):
        _, v = self._bi()
        v.observe([0], [0], [77.0])
        v.judge(4.0, 3, np.ones((NW, NB), dtype=np.int64))
        v.release()
        self.assertEqual(float(v.max[0, 0]), 77.0,
                         "release cleared the all-time max")
        self.assertTrue(np.all(v.streak == 0))
        self.assertTrue(np.all(v.resets == 0))

    def test_no_move_attribute_holds_the_arrays(self):
        """Regression guard for the ownership ruling."""
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase)
        for banned in ("self._shutoff_best =", "self._cold_logl_max =",
                       "self._cold_logl_now =", "self._shutoff_resets ="):
            self.assertNotIn(
                banned, src,
                f"{banned!r} puts a gate array back on the move; the two "
                f"pure in-model moves cannot see a move-owned tracker")


class MultiRankPeakFoldTest(unittest.TestCase):
    """Every walker block's peak must reach the head.

    DEFECT found by the audit window (lisa-sprint-2026-4f) in 9c80faf7
    and confirmed here: ``observe`` wrote into the GLOBAL
    (nwalkers, nbands) peak using BLOCK-LOCAL walker indices, and
    ``_cold_lnl_view`` was bound only in the two head-side propose paths.
    The rank slice carries no band_info at all (``_make_slice_state``
    says so), so on a 4-rank x 1-walker run only walker 0's repeat groups
    ever reached the peak -- walkers 1-3 still saw the single mid-cycle
    ``cap_stats`` sample, i.e. the defect the commit claimed to fix was
    unfixed for three quarters of the ensemble, silently.
    """

    def test_observe_targets_a_block_local_array(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertIn("_cold_lnl_block_peak", src)
        self.assertIn("np.maximum.at(_cold_peak", src)
        self.assertNotIn("_cold_lnl.observe(", src,
                         "writing block-local indices into the global peak")

    def test_the_rank_allocates_the_peak_and_ships_it_in_gb_finish(self):
        """Allocated when the session opens, shipped when it closes.

        The two live in different handlers, and the head merge reads the
        ``gb_finish`` replies (``replies_f``) -- the same reply that
        carries ``cap_stats``. Shipping it from ``gb_run_proposal``
        instead would have been silently dropped.
        """
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        alloc = inspect.getsource(g.GBSpecialBase._gb_serve_run_proposal)
        self.assertIn("_cold_lnl_block_peak = (", alloc)
        ship = inspect.getsource(g.GBSpecialBase._gb_serve_finish)
        self.assertIn('"cold_lnl_peak"', ship)
        self.assertIn('"cap_stats"', ship,
                      "must ride the reply the head actually merges")

    def test_the_head_folds_every_block_at_its_offset(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g)
        self.assertIn('_rp = rep.get("cold_lnl_peak")', src)
        self.assertIn("np.maximum(_gv.peak[w0:w1], _rp, out=_gv.peak[w0:w1])",
                      src)

    def test_the_fold_is_a_max_at_the_offset_not_an_overwrite(self):
        """The arithmetic the head does, at the array level."""
        nw, nb = 4, 6
        peak = np.full((nw, nb), -np.inf)
        # rank 2 owns walker 2 and observed two cells
        blk = np.full((1, nb), -np.inf); blk[0, 1] = -3.0; blk[0, 4] = -9.0
        w0, w1 = 2, 3
        np.maximum(peak[w0:w1], blk, out=peak[w0:w1])
        self.assertEqual(float(peak[2, 1]), -3.0)
        self.assertEqual(float(peak[2, 4]), -9.0)
        self.assertTrue(np.isneginf(peak[0]).all(), "wrote outside its block")
        self.assertTrue(np.isneginf(peak[3]).all())
        # a second, worse observation must not lower it
        blk2 = np.full((1, nb), -np.inf); blk2[0, 1] = -8.0
        np.maximum(peak[w0:w1], blk2, out=peak[w0:w1])
        self.assertEqual(float(peak[2, 1]), -3.0)

    def test_a_neutral_block_cannot_lower_the_peak(self):
        """A neutral reply ships -inf; the maximum must be a no-op."""
        peak = np.full((2, 3), -5.0)
        np.maximum(peak[0:1], np.full((1, 3), -np.inf), out=peak[0:1])
        self.assertTrue((peak == -5.0).all())

    def test_the_single_process_path_folds_locally(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._fold_local_cold_peak)
        self.assertIn("fanout_active", src,
                      "both paths would claim to be the one that works")
        self.assertIn("np.maximum(view.peak, pk, out=view.peak)", src)

    def test_the_gate_line_reports_cells_OBSERVED_not_cells_finite(self):
        """The old count could not see its own defect.

        It was isfinite(value) AFTER the single cap_stats fold, so it read
        as if every occupied cell had been observed even when only one
        walker block was reporting.
        """
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._update_search_band_shutoff)
        self.assertIn("walker(s) reporting", src)
        self.assertNotIn("int(np.isfinite(_view.value).sum()),", src)


class HeadPeakAllocationTest(unittest.TestCase):
    """The head-side block peak is sized from the STATE, never ``self.nwalkers``.

    THE CRASH (6mo v9 relaunch, 2026-09-27, first move of the first
    iteration)::

        File ".../gbspecialstretch.py", line 26288, in _propose_orchestrated
            (int(self.nwalkers), int(self.num_bands)), -np.inf,
        AttributeError: 'VGBSpecialStretchMove' object has no attribute 'nwalkers'

    ``self.nwalkers`` is assigned only INSIDE the propose paths, after the
    branch shapes are read, and no constructor sets it -- so on the first
    propose of a process it does not exist at the point 65851492 allocated
    the peak. ``vgb_pe`` is the first GB-family move of the cycle, which is
    why VGB was the one to hit it.
    """

    class _Bare:
        """A move with ``num_bands`` and deliberately NO ``nwalkers``."""
        num_bands = 5

    class _State:
        def __init__(self, ntemps, nwalkers):
            self.log_like = np.zeros((ntemps, nwalkers))

    def test_a_move_without_nwalkers_still_gets_a_peak(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        move = self._Bare()
        self.assertFalse(hasattr(move, "nwalkers"), "fixture must reproduce the crash")
        pk = g.GBSpecialBase._alloc_cold_peak_from_state(move, self._State(3, 4))
        self.assertEqual(pk.shape, (4, 5), "(nwalkers from state, num_bands from move)")
        self.assertTrue(np.isneginf(pk).all())
        self.assertEqual(pk.dtype, np.float64)

    def test_it_takes_the_walker_axis_from_log_like_not_the_temperature_axis(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        pk = g.GBSpecialBase._alloc_cold_peak_from_state(
            self._Bare(), self._State(ntemps=24, nwalkers=2))
        self.assertEqual(pk.shape, (2, 5))

    def test_an_unreadable_state_yields_None_not_an_exception(self):
        """Telemetry must never take a propose down."""
        import lisatools.globalfit.moves.gbspecialstretch as g

        class _NoLogLike:
            pass

        self.assertIsNone(
            g.GBSpecialBase._alloc_cold_peak_from_state(self._Bare(), _NoLogLike()))

    def test_neither_head_side_propose_path_reads_self_nwalkers_for_the_peak(self):
        """The regression guard: both allocations go through the helper."""
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        for fn in (g.GBSpecialBase._propose_orchestrated,
                   g.GBSpecialBase._propose_legacy):
            src = inspect.getsource(fn)
            self.assertIn("_alloc_cold_peak_from_state(state)", src, fn.__name__)
            self.assertNotIn("(int(self.nwalkers), int(self.num_bands)), -np.inf",
                             src, fn.__name__)


class ValveWritesMustLandInNewStateTest(unittest.TestCase):
    """The valve wrote into the state eryn THROWS AWAY.

    ``_arm_search_stage(state)`` and the ``_cold_lnl_view`` binding both
    run BEFORE ``new_state = GFState(state, copy=True)``, and
    ``GBState.__init__`` DEEP-COPIES band_info into a new dict of new
    arrays. Every cached reference therefore pointed at the incoming
    state's dict, which is discarded the moment new_state is adopted.

    6mo v9 job 659, store row 23 (iteration 1), after a judge that had
    just run over 1902 occupied cells:
        band_cold_logl_peak_w   finite 1905 / 4928   (head folds it into
                                                      new_state -> saved)
        band_cold_logl_max_w    finite    0 / 4928   (judge -> old dict)
        band_cold_logl_w        finite    0 / 4928
        streak_w, reset_w all 0; band_rj_shutoff_w all False
    So max stays -inf forever, every pair "improves" at every judge, the
    streak never leaves 0, and no band can ever shut. 3mo jobs 653/658
    show the same across six consecutive within-job iterations.
    """

    NW, NB = 2, 3

    def _band_info(self):
        import lisatools.globalfit.state as st
        bi = {"num_bands": self.NB, "nwalkers": self.NW}
        st.ensure_search_shutoff_window(bi, self.NB, self.NW)
        bi["band_rj_shutoff_w"] = np.zeros((self.NW, self.NB), dtype=bool)
        return bi

    @staticmethod
    def _view(bi, shape):
        import lisatools.globalfit.moves.gbspecialstretch as g
        return g.cold_band_lnl_from_band_info(bi, shape)

    def test_a_view_on_the_OLD_dict_leaves_the_NEW_one_untouched(self):
        """The bug, stated as a test."""
        old = self._band_info()
        new = copy.deepcopy(old)                  # what GBState does
        v = self._view(old, (self.NW, self.NB))
        v.peak[:] = -5.0
        v.judge(4.0, 3, np.ones((self.NW, self.NB), dtype=np.int64))
        self.assertTrue(np.isfinite(old["band_cold_logl_max_w"]).all())
        self.assertFalse(np.isfinite(new["band_cold_logl_max_w"]).any(),
                         "the new dict must be untouched -- if this ever "
                         "passes, deepcopy stopped copying and the whole "
                         "premise changed")

    def test_a_view_on_the_NEW_dict_is_what_gets_saved(self):
        """The fix: rebind after the copy and the judge's work survives."""
        old = self._band_info()
        new = copy.deepcopy(old)
        v = self._view(new, (self.NW, self.NB))
        v.peak[:] = -5.0
        v.judge(4.0, 3, np.ones((self.NW, self.NB), dtype=np.int64))
        self.assertTrue(np.isfinite(new["band_cold_logl_max_w"]).all())
        np.testing.assert_allclose(new["band_cold_logl_max_w"], -5.0)
        np.testing.assert_allclose(new["band_cold_logl_w"], -5.0)

    def test_two_judges_on_the_CARRIED_dict_give_streak_1_then_2(self):
        """The 3mo signature is 0, 0 -- every judge a first judge."""
        bi = self._band_info()
        occ = np.ones((self.NW, self.NB), dtype=np.int64)
        v = self._view(bi, (self.NW, self.NB))
        v.peak[:] = -5.0
        v.judge(4.0, 3, occ)                      # first: sets the max
        self.assertTrue((bi["band_shutoff_streak_w"] == 0).all())
        v.peak[:] = -5.0                          # no improvement
        v.judge(4.0, 3, occ)
        self.assertTrue((bi["band_shutoff_streak_w"] == 1).all())
        v.peak[:] = -5.0
        v.judge(4.0, 3, occ)
        self.assertTrue((bi["band_shutoff_streak_w"] == 2).all())

    def test_conv_iter_non_improving_judges_shut_the_band(self):
        bi = self._band_info()
        occ = np.ones((self.NW, self.NB), dtype=np.int64)
        v = self._view(bi, (self.NW, self.NB))
        conv = None
        for _ in range(4):
            v.peak[:] = -5.0
            conv = v.judge(4.0, 3, occ)
        self.assertTrue(np.asarray(conv).all(),
                        "3 non-improving judges must converge at conv_iter=3")

    def test_the_peak_is_RESET_after_each_judge(self):
        """Otherwise new_state's peak accumulates across iterations."""
        bi = self._band_info()
        v = self._view(bi, (self.NW, self.NB))
        v.peak[:] = -5.0
        v.judge(4.0, 3, np.ones((self.NW, self.NB), dtype=np.int64))
        self.assertFalse(np.isfinite(v.peak).any())

    def test_the_updater_REBINDS_from_new_state_before_reading_shut(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._update_search_band_shutoff)
        self.assertIn("_bi_new = self._band_shutoff_band_info(new_state)", src)
        self.assertIn("self._rj_band_shutoff_w = _bi_new", src)
        self.assertIn("self._cold_lnl_view = _v_new", src)
        self.assertLess(src.index("_bi_new = self._band_shutoff_band_info"),
                        src.index("shut = self._rj_band_shutoff_w"),
                        "the rebind must precede the read it corrects")

    def test_the_saved_arrays_include_the_window(self):
        """storage_arrays keeps only np.ndarray entries of band_info, so
        the values have to be IN that dict to reach the store at all."""
        import lisatools.globalfit.state as st
        for name in st.SEARCH_SHUTOFF_WINDOW_FIELDS:
            bi = self._band_info()
            self.assertIsInstance(bi[name], np.ndarray, name)


class Level3ShutPairsRunNothingTest(unittest.TestCase):
    """USER RULING 2026-09-28 (as corrected): a level-3 shut
    (walker, band) pair runs NOTHING until the recipe step releases it
    -- no births, no deaths, no in-model repeats at any rung, and NO
    vertical swaps.

    The 09-27 "frozen rows still swap" ruling is unchanged but applies
    to LEVEL-1 frozen rows -- a converged source inside a column that
    is still active -- not to level-3 shut pairs.

    Job 662: 1680 of 1904 occupied pairs (88%) shut, against 2640 s of
    pure in-model work and 1381 s of level-1 windows per iteration.
    """

    NW, NB = 2, 4

    def _mask(self, shut, w_inds, b_inds):
        """The production expression, applied as production applies it."""
        return ~np.asarray(shut)[w_inds, b_inds]

    def test_a_shut_pair_contributes_no_picked_rows(self):
        shut = np.zeros((self.NW, self.NB), dtype=bool)
        shut[0, 1] = True
        w = np.array([0, 0, 1, 1]); b = np.array([1, 2, 1, 2])
        keep = self._mask(shut, w, b)
        self.assertFalse(bool(keep[0]), "the shut pair was still eligible")
        # the SAME BAND in another walker is untouched: the valve is
        # per (walker, band), not per band
        self.assertTrue(bool(keep[2]))
        # and another band in the SAME walker is untouched
        self.assertTrue(bool(keep[1]))

    def test_the_mask_is_applied_at_the_group_eligibility_point(self):
        """One cascade point: everything per-(walker, band) downstream
        runs on PICKED ROWS, so a pair never picked spends no GPU time
        in level-1 windows, cell-ll brackets, sig-het refits or the
        infomat tables."""
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._run_band_unit)
        self.assertIn('_shut3 = getattr(self, "_rj_band_shutoff_w", None)',
                      src)
        self.assertIn("eligible = eligible & ~_xp3.asarray(_shut3)[", src)

    def test_the_mask_does_NOT_depend_on_is_rj_prop(self):
        """The pure in-model moves are ~47% of the iteration; they log
        'valve requested but NOT live (is_rj_prop=False)' about the RJ
        STEP, and gating the pick mask on that would leave most of the
        saving unclaimed."""
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._run_band_unit)
        blk = src[src.index("_shut3 = getattr"):]
        blk = blk[:blk.index("eligible = eligible & ~_xp3")]
        self.assertNotIn("is_rj_prop", blk)

    def test_the_RJ_subset_still_skips_it_independently(self):
        """The RJ path has applied this valve since before the ruling;
        the new mask does not replace it."""
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g)
        self.assertIn('_shut_w = getattr(self, "_rj_band_shutoff_w", None)',
                      src)

    def test_a_shut_column_gets_NO_vertical_swap(self):
        """Not a separate exclusion: ``_vert_all_rung_tables`` derives
        its columns from the block's PICKED ROWS
        (``unique(w_i * num_bands + b_i)``), so a pair that is never
        picked has no column, no rungs in the table, zero swap
        proposals and no [GB_VERT] census entry. One mask, nothing to
        drift out of sync."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        # rows for (w=1, b=2) only -- (w=0, b=1) is shut and unpicked
        t_i = np.array([0, 1]); w_i = np.array([1, 1]); b_i = np.array([2, 2])
        cols, carrier, occ, nal = g._vert_all_rung_tables(
            t_i, w_i, b_i, 3, self.NW, self.NB,
            lambda q: np.ones(q.shape, dtype=np.int64), np)
        self.assertEqual([int(c) for c in cols], [1 * self.NB + 2])
        self.assertNotIn(0 * self.NB + 1, [int(c) for c in cols])

    def test_release_makes_a_shut_pair_pickable_again(self):
        """The step change reopens everything. Per Mike's confirmation
        the all-time MAX survives; the counters do not."""
        import lisatools.globalfit.state as st
        import lisatools.globalfit.moves.gbspecialstretch as g
        bi = {"num_bands": self.NB, "nwalkers": self.NW}
        st.ensure_search_shutoff_window(bi, self.NB, self.NW)
        shut = np.zeros((self.NW, self.NB), dtype=bool)
        v = g.cold_band_lnl_from_band_info(bi, shut.shape)
        occ = np.ones((self.NW, self.NB), dtype=np.int64)
        for _ in range(4):                     # drive it to converged
            v.peak[:] = -5.0
            conv = v.judge(4.0, 3, occ)
        shut[np.asarray(conv)] = True
        self.assertTrue(shut.all())
        self.assertFalse(self._mask(shut, np.array([0]), np.array([1]))[0])

        v.release()
        shut[:] = False
        self.assertTrue(self._mask(shut, np.array([0]), np.array([1]))[0],
                        "release must make the pair pickable again")
        self.assertTrue((bi["band_shutoff_streak_w"] == 0).all())
        self.assertTrue((bi["band_shutoff_reset_w"] == 0).all())
        self.assertFalse(np.isfinite(bi["band_cold_logl_peak_w"]).any())
        # ...and the all-time max SURVIVES (Mike, 2026-09-28)
        self.assertTrue(np.isfinite(bi["band_cold_logl_max_w"]).all())

    def test_the_mask_table_IS_BOUND_on_a_pure_in_model_move(self):
        """⚠ THE DEFECT 98428c32 SHIPPED.

        ``_search_shutoff_per_walker`` requires ``is_rj_prop``, so
        ``_arm_search_stage`` left ``_rj_band_shutoff_w`` at None on the
        two pure in-model moves -- and the pick mask reads exactly that
        attribute. The expression was not gated on is_rj_prop, but the
        table it reads was, so the mask was a silent no-op on ~47% of
        the iteration. The test written to catch it checked the TEXT of
        the block, not the binding, and passed.

        This checks the binding, on the real property.
        """
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        m.search_shutoff_per_walker = True
        m.search_mode = True
        m.is_rj_prop = False                       # a PURE in-model move
        self.assertFalse(m._search_shutoff_per_walker,
                         "owning the valve stays RJ-only")
        self.assertTrue(m._shutoff_mask_live,
                        "but reading it to skip shut pairs must not be")
        m.is_rj_prop = True
        self.assertTrue(m._search_shutoff_per_walker)
        self.assertTrue(m._shutoff_mask_live)

    def test_the_read_only_bind_grants_no_ownership(self):
        """A non-RJ move gets the table and nothing else: no window
        bind, no step stamp, no judge, no release."""
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        src = inspect.getsource(g.GBSpecialBase._arm_search_stage)
        i = src.index("_shutoff_mask_live and not self._search_shutoff")
        blk = src[i:src.index("if self._search_shutoff_per_walker:", i)]
        self.assertIn("self._rj_band_shutoff_w = bi.get", blk)
        for forbidden in ("_bind_shutoff_window", "band_shutoff_w_step",
                          "_release_search_band_shutoff"):
            self.assertNotIn(forbidden, blk, forbidden)

    def test_the_mask_is_off_outside_search_mode(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        m.search_shutoff_per_walker = True
        m.search_mode = False
        m.is_rj_prop = False
        self.assertFalse(m._shutoff_mask_live)
