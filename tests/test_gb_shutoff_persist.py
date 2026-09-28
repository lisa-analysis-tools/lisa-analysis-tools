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
