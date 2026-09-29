"""The staged-cell census: WHAT is in the staging floor?

Job 669 it=1 measured staged cells per unit at rj_fstat_search
1866-1921, rj_warm_search 846-990, in_model 26-34 -- and the logs could
not say what those cells were. The two candidate explanations call for
opposite fixes: level-3-shut OCCUPIED cells mean the staging filter has
a defect, while EMPTY open-band birth cells mean the filter is fine and
the floor needs a valve of its own.
"""

import os
import types
import unittest
from unittest import mock

import numpy as np

import lisatools.globalfit.moves.gbspecialstretch as g
from lisatools.globalfit.moves.gbbands import pack_special_index

NW = 2


def _spec(t, w, b):
    return pack_special_index(np.asarray(t), np.asarray(w),
                              np.asarray(b), NW)


class StageCellCensusTest(unittest.TestCase):

    NB = 6

    def _census(self, cells, alive=(), shut_w=None, shut_b=None,
                elig=None):
        sp = _spec(*zip(*cells)) if cells else np.asarray([], dtype=int)
        al = _spec(*zip(*alive)) if alive else np.asarray([], dtype=int)
        return g.stage_cell_census(sp, al, shut_w, shut_b, elig, NW, np)

    def test_the_buckets_are_exhaustive(self):
        cells = [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)]
        c = self._census(cells, alive=[(0, 0, 0)])
        self.assertEqual(sum(c.values()), len(cells))

    def test_an_occupied_cell_in_a_SHUT_pair_is_the_defect_bucket(self):
        shut = np.zeros((NW, self.NB), bool)
        shut[1, 3] = True
        c = self._census([(0, 1, 3)], alive=[(0, 1, 3)], shut_w=shut)
        self.assertEqual(c["shut_occ"], 1)
        self.assertEqual(c["active_occ"], 0)

    def test_an_occupied_cell_in_an_OPEN_pair_is_real_work(self):
        c = self._census([(0, 1, 3)], alive=[(0, 1, 3)],
                         shut_w=np.zeros((NW, self.NB), bool))
        self.assertEqual(c["active_occ"], 1)

    def test_occupancy_is_PER_CELL_not_per_band(self):
        """A band occupied on the cold chain can be empty on a hot rung,
        and those are different cells with different work."""
        c = self._census([(0, 0, 2), (5, 0, 2)], alive=[(0, 0, 2)])
        self.assertEqual(c["active_occ"], 1)
        self.assertEqual(c["empty_open_hi"], 1)

    def test_an_empty_cell_in_a_band_the_valve_shut_is_already_handled(self):
        sb = np.zeros(self.NB, bool)
        sb[4] = True
        c = self._census([(0, 0, 4)], shut_b=sb)
        self.assertEqual(c["empty_valved"], 1)

    def test_an_empty_cell_in_a_shut_PAIR_also_counts_as_valved(self):
        shut = np.zeros((NW, self.NB), bool)
        shut[0, 4] = True
        c = self._census([(0, 0, 4)], shut_w=shut)
        self.assertEqual(c["empty_valved"], 1)

    def test_empty_open_cells_SPLIT_on_the_valve_FMIN_floor(self):
        """The split is the whole point of the low bucket: below FMIN the
        existing per-band valve can never shut a band however barren, so
        those cells need either a lower FMIN or a new valve."""
        elig = np.array([False, False, True, True, True, True])
        c = self._census([(0, 0, 0), (0, 0, 1), (0, 0, 5)], elig=elig)
        self.assertEqual(c["empty_open_lo"], 2)
        self.assertEqual(c["empty_open_hi"], 1)

    def test_no_tables_at_all_still_classifies(self):
        c = self._census([(0, 0, 1), (0, 0, 2)], alive=[(0, 0, 1)])
        self.assertEqual(c["active_occ"], 1)
        self.assertEqual(c["empty_open_hi"], 1)

    def test_an_empty_unit_is_all_zeros_not_a_crash(self):
        self.assertEqual(sum(g.stage_cell_census(
            np.asarray([], dtype=int), None, None, None, None, NW,
            np).values()), 0)


class StageCensusWiringTest(unittest.TestCase):

    LOGGER = "lisatools.globalfit.moves.gbspecialstretch"

    def _move(self):
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        m.name = "rj_fstat_search"
        m._rj_band_shutoff_w = None
        m._rj_band_shutoff = None
        m._last_stage_removed = 7
        m._alive_cells_cache = None
        m.band_edges = np.array([1e-3, 5e-3, 11e-3, 20e-3])
        return m

    def _sorter(self):
        return types.SimpleNamespace(
            inds=np.array([True, False]),
            temp_inds=np.array([0, 0]), walker_inds=np.array([0, 0]),
            band_inds=np.array([0, 2]), nwalkers=NW)

    def test_the_line_reports_every_bucket_and_what_was_removed(self):
        m, s = self._move(), self._sorter()
        sp = _spec([0, 0], [0, 0], [0, 2])
        with mock.patch.dict(os.environ, {"GB_STAGE_CENSUS": "1"}):
            with self.assertLogs(self.LOGGER, "INFO") as cm:
                g.GBSpecialBase._log_stage_census(m, 3, s, sp, np)
        out = "\n".join(cm.output)
        self.assertIn("[GB_STAGE_CENSUS rj_fstat_search] unit 3", out)
        self.assertIn("staged 2", out)
        self.assertIn("filter removed 7", out)
        for tok in ("active-occ", "SHUT-OCC", "empty-valved",
                    "empty-open<FMIN", "empty-open>=FMIN"):
            self.assertIn(tok, out)

    def test_the_knob_turns_it_off(self):
        m, s = self._move(), self._sorter()
        with mock.patch.dict(os.environ, {"GB_STAGE_CENSUS": "0"}):
            with mock.patch.object(g, "stage_cell_census") as spy:
                g.GBSpecialBase._log_stage_census(
                    m, 0, s, _spec([0], [0], [0]), np)
            spy.assert_not_called()

    def test_a_broken_table_WARNS_and_never_breaks_the_unit(self):
        m, s = self._move(), self._sorter()
        m._rj_band_shutoff_w = np.zeros((1, 1), bool)   # wrong shape
        with mock.patch.dict(os.environ, {"GB_STAGE_CENSUS": "1"}):
            with self.assertLogs(self.LOGGER, "WARNING"):
                g.GBSpecialBase._log_stage_census(
                    m, 0, s, _spec([0, 0], [0, 0], [0, 2]), np)

    def test_the_FMIN_mask_matches_the_valve_that_uses_it(self):
        """Recomputed rather than shared, so it is pinned: lower edge at
        or above GB_RJ_BAND_SHUTOFF_FMIN_MHZ."""
        m = self._move()
        with mock.patch.dict(
            os.environ, {"GB_RJ_BAND_SHUTOFF_FMIN_MHZ": "10.0"}
        ):
            np.testing.assert_array_equal(
                m._band_shutoff_eligible_mask(), [False, False, True])
        with mock.patch.dict(
            os.environ, {"GB_RJ_BAND_SHUTOFF_FMIN_MHZ": "1.0"}
        ):
            np.testing.assert_array_equal(
                m._band_shutoff_eligible_mask(), [True, True, True])

    def test_the_alive_cell_set_is_cached_PER_PROPOSE(self):
        """The band assignment is frozen for the propose, so nine units
        must not pay for nine device reductions."""
        m, s = self._move(), self._sorter()
        first = g.GBSpecialBase._alive_cell_specials(m, s, np)
        s.inds = np.array([False, False])      # would change the answer
        again = g.GBSpecialBase._alive_cell_specials(m, s, np)
        np.testing.assert_array_equal(first, again)

    def test_the_unit_index_reaches_the_census(self):
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_band_unit)
        self.assertIn("self._log_stage_census(unit_i, band_sorter, "
                      "_sched_specials,", src)
        self.assertIn(
            "unit_i", inspect.signature(g.GBSpecialBase._run_band_unit)
            .parameters)

    def test_the_filter_stashes_what_it_removed(self):
        import inspect
        src = inspect.getsource(g.GBSpecialBase._staged_specials)
        self.assertIn("self._last_stage_removed = int(n)", src)

    def test_the_census_runs_AFTER_the_filter(self):
        """It must describe what the scheduler will actually be built
        from, not what the subset offered."""
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_band_unit)
        self.assertLess(src.index("_staged_specials("),
                        src.index("_log_stage_census("))
        self.assertLess(src.index("_log_stage_census("),
                        src.index("BandScheduler("))


if __name__ == "__main__":
    unittest.main()


class CensusSurvivesTheTarTest(unittest.TestCase):
    """A diagnostic the snapshot filter strips is a diagnostic nobody
    reads.

    ⚠ Neither ``[GB_STAGE ...] excluded from staging`` nor
    ``[GB_STAGE_CENSUS]`` matched LOG_KEEP_PATTERN before 2026-09-29, so
    every tar's ``*_filtered.log`` dropped them -- and the conclusion
    "the exclusion line never fires" was read off exactly those files.
    An absence in a filtered log proves nothing unless the prefix is
    kept.
    """

    def _keep(self):
        import re
        from lisatools.globalfit.monitor import snapshot as snap
        return re.compile(snap.LOG_KEEP_PATTERN)

    def test_the_census_line_survives_the_filter(self):
        self.assertTrue(self._keep().search(
            "[GB_STAGE_CENSUS rj_fstat_search] unit 3: staged 1900 = "
            "active-occ 12 + SHUT-OCC 0 + empty-valved 300 + "
            "empty-open<FMIN 1200 + empty-open>=FMIN 388; filter removed 0"))

    def test_the_EXISTING_exclusion_line_survives_it_too(self):
        self.assertTrue(self._keep().search(
            "[GB_STAGE rj_warm_search] 12 cell(s) in shut pairs excluded "
            "from staging (900 staged)"))

    def test_the_shell_and_python_filters_still_agree(self):
        """Covered by test_monitor_snapshot's parity test; asserted here
        too because this branch edits both copies."""
        import pathlib
        import re
        from lisatools.globalfit.monitor import snapshot as snap
        sh = (pathlib.Path(__file__).resolve().parents[1] / "scripts" /
              "fstat_proposal" / "make_snapshots.sh").read_text()
        m = re.search(r'grep -aE "([^"]+)"', sh, re.S)
        self.assertIsNotNone(m)
        self.assertEqual(
            {a for a in m.group(1).replace("\\\n", "").split("|") if a},
            {a for a in snap.LOG_KEEP_PATTERN.split("|") if a})
