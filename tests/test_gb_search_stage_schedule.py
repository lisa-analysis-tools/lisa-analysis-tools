"""Per-(walker, band) GB search schedule (user request 2026-09-23/24).

Two independently switchable features built on one shared cold-chain
census, both the per-(walker, band) twins of a mechanism that already
exists shared:

* the SEARCH-STAGE SCHEDULE -- the opt-SNR prior boundary stops being one
  number for the whole run and becomes a function of a (walker, band)
  stage, promoted COARSE -> FINE once that walker's source count in that
  band has settled;
* the PER-WALKER RJ SHUTOFF VALVE -- additional to, and composed by OR
  with, the existing per-band valve: a (walker, band) whose cold-chain
  lnL has PLATEAUED within the current recipe step takes no further RJ
  until the next step begins. (User ruling 2026-09-24: the quantity that
  has to converge is the lnL, not the leaf count -- a count can sit still
  while the sampler is still materially improving the fit of the sources
  it already has.)

THE HARD CONSTRAINT these tests exist to protect, same as the per-walker
cap suite: with the flags OFF nothing is allocated, nothing is written and
no branch is taken, so a live run can be relaunched from a build carrying
this change.

Same light-fake style as ``test_gb_cap_per_walker``: the latch, the valve
and the floor gather are all array arithmetic over band_info-shaped and
band_sorter-shaped arrays, so none of it needs a built move, an ACA or a
backend.
"""

import dataclasses
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove
from lisatools.globalfit.recipe import (
    band_shutoff_w_armed,
    band_shutoff_w_pending_total,
    begin_search_recipe_step,
)
from lisatools.globalfit.state import (
    SEARCH_STAGE_COARSE,
    SEARCH_STAGE_FINE,
    SEARCH_SHUTOFF_STEP_UNSET,
    ensure_search_shutoff_fields,
    ensure_search_stage_fields,
)
from lisatools.sampling.fstat_gridfit import select_comb_peaks
from lisatools.sampling.fstat_proposal import (
    fstat_band_min_F,
    fstat_peak_min_F,
    fstat_peak_min_F_stages,
)

from tests.test_gb_cap_cell_grid import BAND_EDGES, NUM_BANDS, _move

STAGE_ENV = "GB_SEARCH_STAGE_PER_WALKER"
SHUTOFF_ENV = "GB_SEARCH_BAND_SHUTOFF_PER_WALKER"
NWALKERS = 3


def _band_info(nwalkers=NWALKERS, num_bands=NUM_BANDS, stage=False,
               shutoff=False):
    """A band_info dict with only what these features read."""
    bi = {"nwalkers": nwalkers, "num_bands": num_bands,
          "band_edges": np.asarray(BAND_EDGES)}
    ensure_search_stage_fields(bi, num_bands, per_walker=stage)
    ensure_search_shutoff_fields(bi, num_bands, per_walker=shutoff)
    return bi


def _stage_move(nwalkers=NWALKERS, min_iters=3, coarse=8.0, fine=5.0,
                stage=True, shutoff=False, conv_iter=2, search=True):
    """A fake move carrying only the stage/valve attributes."""
    m = _move(1, ntemps=1, nwalkers=nwalkers)
    m.branch_name = "gb"
    m.is_rj_prop = True
    m.search_mode = bool(search)
    m.search_stage_per_walker = bool(stage)
    m.search_stage_min_iters = int(min_iters)
    m.opt_snr_limit_search_coarse = float(coarse)
    m.opt_snr_limit_search_fine = float(fine)
    m.opt_snr_rej_samp_limit = float(coarse)
    m.search_shutoff_per_walker = bool(shutoff)
    m.search_shutoff_conv_iter = int(conv_iter)
    m._stage_table = None
    m._snr_lim_table = None
    m._rj_band_shutoff_w = None
    m._stage_band_lls = None
    # the valve's statistic since 2026-10-10: the band GAIN, stashed next to
    # the residual term under the same stamp (see tests/test_gb_valve_gain.py)
    m._stage_band_gain = None
    m._stage_band_lls_stamp = None
    m._shutoff_w_warned_lls = False
    m.leaf_cap_ndim = 8.0           # -> D/2 = 4.0 improvement threshold
    m.num_proposals = 0
    m._shutoff_w_pending = None
    m._shutoff_w_shut = 0
    m._shutoff_w_total = 0
    m._recipe_step_serial = None
    m._shutoff_w_warned_mode = False
    m._stage_armed_logged = True
    return m


def _state(bi):
    """The two attribute hops the latch makes into a GFState."""
    return SimpleNamespace(
        sub_states={"gb": SimpleNamespace(band_info=bi)})


def _counts(occ):
    """``band_counts`` shaped (ntemps, nwalkers, nbands); only [0] is read."""
    return np.asarray(occ, dtype=np.int64)[None, ...]


class StageAllocationTest(unittest.TestCase):
    """The flag-off guarantee: no array, no key, no branch."""

    def test_flag_off_allocates_nothing(self):
        bi = _band_info(stage=False, shutoff=False)
        for key in ("band_stage_w", "band_stage_occ_last_w",
                    "band_stage_streak_w", "band_stage",
                    "band_rj_shutoff_w", "band_shutoff_w_step"):
            self.assertNotIn(key, bi, f"{key} allocated with the flag off")

    def test_flag_off_returns_the_off_token(self):
        bi = {"nwalkers": NWALKERS, "num_bands": NUM_BANDS}
        self.assertEqual(
            ensure_search_stage_fields(bi, NUM_BANDS, per_walker=False), "off")
        self.assertEqual(
            ensure_search_shutoff_fields(bi, NUM_BANDS, per_walker=False),
            "off")

    def test_flag_on_allocates_the_documented_shapes(self):
        bi = _band_info(stage=True, shutoff=True)
        for key in ("band_stage_w", "band_stage_occ_last_w",
                    "band_stage_streak_w", "band_rj_shutoff_w"):
            self.assertEqual(np.shape(bi[key]), (NWALKERS, NUM_BANDS), key)
        self.assertEqual(np.shape(bi["band_stage"]), (NUM_BANDS,))
        self.assertEqual(np.shape(bi["band_shutoff_w_step"]), (1,))

    def test_fresh_stage_is_coarse_and_occ_last_is_the_sentinel(self):
        bi = _band_info(stage=True)
        self.assertTrue(np.all(bi["band_stage_w"] == SEARCH_STAGE_COARSE))
        # -1, not 0: the first update must start a fresh streak rather than
        # credit a spurious match against a zero-filled array.
        self.assertTrue(np.all(bi["band_stage_occ_last_w"] == -1))
        self.assertTrue(np.all(bi["band_stage_streak_w"] == 0))

    def test_fresh_valve_is_open_with_the_step_sentinel(self):
        bi = _band_info(shutoff=True)
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self.assertEqual(int(bi["band_shutoff_w_step"][0]),
                         SEARCH_SHUTOFF_STEP_UNSET)

    def test_no_walker_count_refuses_rather_than_falling_back(self):
        for fn in (ensure_search_stage_fields, ensure_search_shutoff_fields):
            with self.assertRaises(ValueError):
                fn({"num_bands": NUM_BANDS}, NUM_BANDS, per_walker=True)


class FeatureIndependenceTest(unittest.TestCase):
    """The two features are separately switchable.

    USER RULING 2026-09-24: the coming run will NOT move the SNR limits
    per band -- those move together across recipe stages instead -- but it
    WILL use the per-(walker, band) RJ shutoff. So "valve on, stage
    schedule off" is the production configuration, and it has to be a
    configuration, not an accident.
    """

    def test_both_default_off_in_gbsettings(self):
        from lisatools.globalfit.stock.erebor.gb import GBSettings

        for name in ("GB_SEARCH_STAGE_PER_WALKER",
                     "GB_SEARCH_BAND_SHUTOFF_PER_WALKER"):
            self.assertIsNone(os.environ.get(name),
                              f"{name} leaked into the test environment")
        fields = {f.name: f for f in dataclasses.fields(GBSettings)}
        self.assertFalse(
            fields["search_stage_per_walker"].default_factory())
        self.assertFalse(
            fields["search_shutoff_per_walker"].default_factory())

    def test_valve_on_stage_off_is_the_production_shape(self):
        m = _stage_move(stage=False, shutoff=True, conv_iter=2)
        bi = _band_info(stage=False, shutoff=True)
        m._arm_search_stage(_state(bi))
        # the valve is live...
        self.assertIsNotNone(m._rj_band_shutoff_w)
        # ...and the SNR floor is untouched: still the scalar it always was
        self.assertIsNone(m._snr_lim_table)
        self.assertIsNone(m._stage_table)
        self.assertEqual(m._live_snr_lim(), m.opt_snr_rej_samp_limit)
        # ...and the F-stat catalog keeps its single global floor
        from lisatools.globalfit.moves.gbspecialstretch import (
            fstat_band_min_F_for,
        )
        self.assertIsNone(fstat_band_min_F_for(m))

        # the valve still works with the schedule off
        st = _state(bi)
        for _ in range(6):
            m.num_proposals += 1
            m._stage_band_gain = np.zeros((NWALKERS, NUM_BANDS))
            m._stage_band_lls_stamp = m.num_proposals
            m._update_search_band_shutoff(
                None, st, _counts(np.ones((NWALKERS, NUM_BANDS))))
        self.assertTrue(bi["band_rj_shutoff_w"].all())

    def test_stage_on_valve_off_also_works(self):
        m = _stage_move(stage=True, shutoff=False)
        bi = _band_info(stage=True, shutoff=False)
        m._arm_search_stage(_state(bi))
        self.assertIsNotNone(m._snr_lim_table)
        self.assertIsNone(m._rj_band_shutoff_w)

    def test_both_off_touches_nothing(self):
        m = _stage_move(stage=False, shutoff=False)
        bi = _band_info(stage=False, shutoff=False)
        m._arm_search_stage(_state(bi))
        self.assertIsNone(m._snr_lim_table)
        self.assertIsNone(m._stage_table)
        self.assertIsNone(m._rj_band_shutoff_w)
        self.assertEqual(m._live_snr_lim(), m.opt_snr_rej_samp_limit)
        self.assertEqual(bi.keys() - {"nwalkers", "num_bands", "band_edges"},
                         set())


class RealConstructionTest(unittest.TestCase):
    """Build an ACTUAL GBSpecialStretchMove on the DEFAULT path.

    ⚠ THIS IS THE TEST THE REST OF THIS FILE COULD NOT BE. Every other case
    here drives the stage/valve logic through a light fake that bypasses
    ``__init__`` entirely, which is what makes them fast -- and is exactly
    why 491 green tests missed an ordering bug in ``__init__`` that raised
    ``AttributeError`` on the default construction path and reached dev
    (2026-09-24: the coarse/fine block read ``self.opt_snr_rej_samp_limit``
    ~315 lines before it was assigned; kwarg None + env unset is the
    SHIPPED configuration, so every real GB run would have crashed before
    its first proposal).

    So this one pays for a real fixture. It asserts nothing clever: only
    that the constructor RUNS with no stage env and no stage kwargs, and
    that the COARSE floor ends up equal to the floor the move actually
    resolved -- which is the contract the buggy line was trying to express.
    """

    STAGE_ENVS = (
        "GB_SEARCH_STAGE_PER_WALKER", "GB_SEARCH_STAGE_MIN_ITERS",
        "GB_OPT_SNR_LIMIT_SEARCH_COARSE", "GB_OPT_SNR_LIMIT_SEARCH_FINE",
        "GB_SEARCH_BAND_SHUTOFF_PER_WALKER",
        "GB_SEARCH_BAND_SHUTOFF_CONV_ITER",
    )

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.STAGE_ENVS}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def _build_move(self, **extra):
        from tests.test_gbspecial_flow import build_fixture
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialStretchMove,
        )

        fx = build_fixture(seed=1234)
        kwargs = dict(fx["move_kwargs"])
        kwargs.update(extra)
        return GBSpecialStretchMove(
            *fx["move_args"], is_rj_prop=False,
            name="stage_ctor_check", stretch_probability=0.5, **kwargs)

    def test_default_construction_does_not_raise(self):
        move = self._build_move()
        # the default path: both features off, floor still a plain scalar
        self.assertFalse(move.search_stage_per_walker)
        self.assertFalse(move.search_shutoff_per_walker)
        self.assertEqual(np.ndim(move.opt_snr_rej_samp_limit), 0)

    def test_coarse_defaults_to_the_resolved_floor(self):
        move = self._build_move()
        self.assertEqual(
            float(move.opt_snr_limit_search_coarse),
            float(move.opt_snr_rej_samp_limit),
            "COARSE must default to the floor this move actually resolved",
        )

    def test_an_explicit_coarse_env_wins(self):
        os.environ["GB_OPT_SNR_LIMIT_SEARCH_COARSE"] = "9.5"
        move = self._build_move()
        self.assertEqual(float(move.opt_snr_limit_search_coarse), 9.5)

    def test_an_explicit_kwarg_beats_the_env(self):
        os.environ["GB_OPT_SNR_LIMIT_SEARCH_COARSE"] = "9.5"
        move = self._build_move(opt_snr_limit_search_coarse=7.25)
        self.assertEqual(float(move.opt_snr_limit_search_coarse), 7.25)

    def test_an_inverted_pair_refuses_at_construction(self):
        with self.assertRaises(ValueError):
            self._build_move(
                search_stage_per_walker=True,
                opt_snr_limit_search_coarse=5.0,
                opt_snr_limit_search_fine=9.0,
            )


class StageRestoreTest(unittest.TestCase):
    """All-or-nothing, and every degradation lands on the SAFE side."""

    def test_partial_record_is_discarded_whole(self):
        bi = _band_info(stage=True)
        bi["band_stage_w"][0, 0] = SEARCH_STAGE_FINE
        del bi["band_stage_streak_w"]
        origin = ensure_search_stage_fields(bi, NUM_BANDS, per_walker=True)
        self.assertTrue(origin.startswith("reset"))
        # discarded WHOLE: the promoted pair is gone too
        self.assertTrue(np.all(bi["band_stage_w"] == SEARCH_STAGE_COARSE))

    def test_wrong_grid_restarts_at_coarse(self):
        bi = _band_info(stage=True)
        bi["band_stage_w"][:] = SEARCH_STAGE_FINE
        bi["nwalkers"] = NWALKERS + 1
        origin = ensure_search_stage_fields(bi, NUM_BANDS, per_walker=True)
        self.assertTrue(origin.startswith("reset"))
        self.assertEqual(np.shape(bi["band_stage_w"]),
                         (NWALKERS + 1, NUM_BANDS))
        # COARSE is the TIGHTER floor -- a record we cannot trust must never
        # hand a walker the relaxed prior it did not earn.
        self.assertTrue(np.all(bi["band_stage_w"] == SEARCH_STAGE_COARSE))

    def test_intact_record_round_trips(self):
        bi = _band_info(stage=True)
        bi["band_stage_w"][1, 2] = SEARCH_STAGE_FINE
        bi["band_stage_streak_w"][1, 2] = 7
        origin = ensure_search_stage_fields(bi, NUM_BANDS, per_walker=True)
        self.assertEqual(origin, "restored")
        self.assertEqual(int(bi["band_stage_w"][1, 2]), SEARCH_STAGE_FINE)
        self.assertEqual(int(bi["band_stage_streak_w"][1, 2]), 7)

    def test_restore_rebuilds_the_min_over_walkers_mirror(self):
        bi = _band_info(stage=True)
        bi["band_stage_w"][:, 1] = SEARCH_STAGE_FINE
        bi["band_stage_w"][0, 1] = SEARCH_STAGE_COARSE  # one holdout
        ensure_search_stage_fields(bi, NUM_BANDS, per_walker=True)
        # a band reads FINE only once EVERY walker agrees
        self.assertEqual(int(bi["band_stage"][1]), SEARCH_STAGE_COARSE)
        bi["band_stage_w"][0, 1] = SEARCH_STAGE_FINE
        ensure_search_stage_fields(bi, NUM_BANDS, per_walker=True)
        self.assertEqual(int(bi["band_stage"][1]), SEARCH_STAGE_FINE)

    def test_valve_wrong_shape_reopens(self):
        bi = _band_info(shutoff=True)
        bi["band_rj_shutoff_w"][:] = True
        bi["nwalkers"] = NWALKERS + 2
        origin = ensure_search_shutoff_fields(bi, NUM_BANDS, per_walker=True)
        self.assertTrue(origin.startswith("reset"))
        # ALL-OPEN is the permissive direction: a valve we cannot trust
        # must never freeze a band the search still needs.
        self.assertFalse(bi["band_rj_shutoff_w"].any())


class StageLatchTest(unittest.TestCase):
    """Promotion on an UNCHANGED source count, and only then."""

    def _run(self, m, bi, occ_series):
        st = _state(bi)
        for occ in occ_series:
            m._update_search_stages(st, _counts(occ))

    def test_promotes_after_exactly_min_iters(self):
        m = _stage_move(min_iters=3)
        bi = _band_info(stage=True)
        occ = np.ones((NWALKERS, NUM_BANDS), dtype=np.int64)
        # update 1 seeds occ_last (streak stays 0 -- occ_last was -1);
        # updates 2,3,4 post streak 1,2,3 -> promotion lands on the 4th.
        self._run(m, bi, [occ] * 3)
        self.assertTrue(
            np.all(bi["band_stage_w"] == SEARCH_STAGE_COARSE),
            "promoted before min_iters full quiet iterations")
        self._run(m, bi, [occ])
        self.assertTrue(np.all(bi["band_stage_w"] == SEARCH_STAGE_FINE))

    def test_a_count_change_resets_the_streak_to_zero(self):
        m = _stage_move(min_iters=3)
        bi = _band_info(stage=True)
        one = np.ones((NWALKERS, NUM_BANDS), dtype=np.int64)
        two = one * 2
        self._run(m, bi, [one, one, one])          # streak 0,1,2
        self._run(m, bi, [two])                     # CHANGE -> 0, not 2
        self.assertTrue(np.all(bi["band_stage_streak_w"] == 0))
        self._run(m, bi, [two, two])                # streak 1,2
        self.assertTrue(np.all(bi["band_stage_w"] == SEARCH_STAGE_COARSE))
        self._run(m, bi, [two])                     # streak 3 -> promote
        self.assertTrue(np.all(bi["band_stage_w"] == SEARCH_STAGE_FINE))

    def test_an_empty_band_never_promotes(self):
        """The ghost-increment failure, in its stage form.

        An empty band's count is trivially unchanged forever. Without the
        ``occ > 0`` term every empty band would promote itself on a fixed
        clock -- and a band that has found nothing has reached no
        equilibrium.
        """
        m = _stage_move(min_iters=2)
        bi = _band_info(stage=True)
        occ = np.ones((NWALKERS, NUM_BANDS), dtype=np.int64)
        occ[:, 0] = 0
        self._run(m, bi, [occ] * 20)
        self.assertTrue(np.all(bi["band_stage_w"][:, 0] == SEARCH_STAGE_COARSE))
        self.assertTrue(np.all(bi["band_stage_w"][:, 1:] == SEARCH_STAGE_FINE))

    def test_promotion_is_one_way(self):
        m = _stage_move(min_iters=2)
        bi = _band_info(stage=True)
        occ = np.ones((NWALKERS, NUM_BANDS), dtype=np.int64)
        self._run(m, bi, [occ] * 5)
        self.assertTrue(np.all(bi["band_stage_w"] == SEARCH_STAGE_FINE))
        # a count change afterwards must NOT demote: FINE is a strictly
        # larger prior support, and tightening it under an assembled model
        # would freeze out sources the model already holds.
        self._run(m, bi, [occ * 3, occ * 7, occ])
        self.assertTrue(np.all(bi["band_stage_w"] == SEARCH_STAGE_FINE))

    def test_walkers_promote_independently(self):
        m = _stage_move(min_iters=2)
        bi = _band_info(stage=True)
        occ = np.ones((NWALKERS, NUM_BANDS), dtype=np.int64)
        for _ in range(4):
            churn = occ.copy()
            churn[0, 0] += 1          # walker 0 band 0 never settles
            self._run(m, bi, [churn])
            churn[0, 0] -= 1
            self._run(m, bi, [churn])
        self.assertEqual(int(bi["band_stage_w"][0, 0]), SEARCH_STAGE_COARSE)
        self.assertTrue(np.all(bi["band_stage_w"][1:, 0] == SEARCH_STAGE_FINE))

    def test_flag_off_is_a_no_op(self):
        m = _stage_move(stage=False)
        bi = _band_info(stage=False)
        m._update_search_stages(
            _state(bi), _counts(np.ones((NWALKERS, NUM_BANDS))))
        self.assertNotIn("band_stage_w", bi)

    def test_census_row_mismatch_raises_rather_than_broadcasting(self):
        m = _stage_move()
        bi = _band_info(stage=True)
        with self.assertRaises(RuntimeError):
            m._update_search_stages(
                _state(bi), _counts(np.ones((NWALKERS * 2, NUM_BANDS))))


class SnrFloorTableTest(unittest.TestCase):
    def test_scalar_passes_through_bit_identically(self):
        lim = 8.0
        out = GBSpecialStretchMove._snr_lim_for_rows(
            lim, np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.assertIs(out, lim)

    def test_table_gathers_per_walker_and_band(self):
        table = np.arange(NWALKERS * NUM_BANDS, dtype=float).reshape(
            NWALKERS, NUM_BANDS)
        w = np.array([0, 2, 1])
        b = np.array([3, 0, 2])
        np.testing.assert_array_equal(
            GBSpecialStretchMove._snr_lim_for_rows(table, w, b),
            table[w, b])

    def test_build_maps_stage_to_the_two_floors(self):
        m = _stage_move(coarse=8.0, fine=5.0)
        bi = _band_info(stage=True)
        bi["band_stage_w"][1, 2] = SEARCH_STAGE_FINE
        table = m._build_snr_lim_table(_state(bi))
        self.assertEqual(np.shape(table), (NWALKERS, NUM_BANDS))
        self.assertEqual(table[1, 2], 5.0)
        self.assertEqual(table[0, 2], 8.0)
        self.assertEqual(table[1, 1], 8.0)

    def test_out_of_range_stage_raises(self):
        m = _stage_move()
        bi = _band_info(stage=True)
        bi["band_stage_w"][0, 0] = 7
        with self.assertRaises(ValueError):
            m._build_snr_lim_table(_state(bi))

    def test_flag_off_builds_no_table(self):
        m = _stage_move(stage=False)
        self.assertIsNone(m._build_snr_lim_table(_state(_band_info())))

    def test_live_lim_falls_back_to_the_scalar(self):
        m = _stage_move()
        m._snr_lim_table = None
        self.assertEqual(m._live_snr_lim(), m.opt_snr_rej_samp_limit)

    def test_truncation_floor_takes_the_minimum(self):
        """Wider than any row needs, so no row is drawn outside a region
        its own charged density covers."""
        self.assertEqual(GBSpecialStretchMove._snr_trunc_floor(8.0), 8.0)
        table = np.array([[8.0, 5.0], [8.0, 8.0]])
        self.assertEqual(GBSpecialStretchMove._snr_trunc_floor(table), 5.0)

    def test_block_slice_matches_the_cap_rule(self):
        m = _stage_move()
        table = np.arange(NWALKERS * NUM_BANDS, dtype=float).reshape(
            NWALKERS, NUM_BANDS)
        np.testing.assert_array_equal(
            m._snr_lim_table_for_block(table, 1, 3), table[1:3])
        self.assertIsNone(m._snr_lim_table_for_block(None, 0, 2))

    def test_rank_table_row_count_is_checked(self):
        m = _stage_move()
        table = np.zeros((NWALKERS, NUM_BANDS))
        m._check_rank_stage_table("snr_lim_table", table, NWALKERS)  # ok
        with self.assertRaises(RuntimeError):
            m._check_rank_stage_table("snr_lim_table", table, NWALKERS - 1)


class SwapGateTest(unittest.TestCase):
    def test_same_stage_permits_and_different_stage_refuses(self):
        m = _stage_move()
        bi = _band_info(stage=True)
        bi["band_stage_w"][1, 2] = SEARCH_STAGE_FINE
        m._stage_table = bi["band_stage_w"]
        bands = np.array([2, 2, 1])
        # walker 0 (COARSE) <-> walker 1 (FINE) in band 2: refused
        ok = m._swap_stage_ok(np.array([0, 1, 0]), np.array([1, 2, 1]), bands)
        self.assertFalse(bool(ok[0]))
        # walker 1 (FINE) <-> walker 2 (COARSE) in band 2: also refused
        self.assertFalse(bool(ok[1]))
        # band 1: nobody promoted, both COARSE -> permitted
        self.assertTrue(bool(ok[2]))

    def test_vertical_sweep_shares_a_walker_and_is_unchanged(self):
        m = _stage_move()
        bi = _band_info(stage=True)
        bi["band_stage_w"][1, :] = SEARCH_STAGE_FINE
        m._stage_table = bi["band_stage_w"]
        w = np.array([1, 1, 1])
        ok = m._swap_stage_ok(w, w, np.array([0, 1, 2]))
        self.assertTrue(bool(np.all(ok)))

    def test_vacuous_with_the_feature_off(self):
        m = _stage_move(stage=False)
        self.assertIs(
            m._swap_stage_ok(np.array([0]), np.array([1]), np.array([0])),
            True)

    def test_shutoff_refuses_a_swap_touching_a_frozen_pair(self):
        m = _stage_move(shutoff=True)
        bi = _band_info(shutoff=True)
        bi["band_rj_shutoff_w"][1, 2] = True
        m._rj_band_shutoff_w = bi["band_rj_shutoff_w"]
        ok = m._swap_shutoff_ok(
            np.array([0, 0, 1]), np.array([1, 2, 2]), np.array([2, 2, 1]))
        self.assertFalse(bool(ok[0]))   # w1 frozen in band 2
        self.assertTrue(bool(ok[1]))    # neither w0 nor w2 frozen in band 2
        self.assertTrue(bool(ok[2]))    # band 1 untouched

    def test_shutoff_swap_gate_vacuous_when_unbound(self):
        m = _stage_move()
        self.assertIs(
            m._swap_shutoff_ok(np.array([0]), np.array([1]), np.array([0])),
            True)


class RjShutoffValveTest(unittest.TestCase):
    """The lnL-plateau test, per (walker, band), per recipe step."""

    def _run(self, m, bi, ll_series, occ=None):
        """Feed per-iteration ``(nwalkers, nbands)`` lnLs through the valve."""
        st = _state(bi)
        if occ is None:
            occ = np.ones((m.nwalkers, NUM_BANDS), dtype=np.int64)
        for lls in ll_series:
            m.num_proposals += 1
            # the stash the cap gate hands over each iteration (the valve
            # judges the GAIN stash; these series are statistic-agnostic)
            m._stage_band_gain = np.asarray(lls, dtype=float)
            m._stage_band_lls_stamp = m.num_proposals
            m._update_search_band_shutoff(None, st, _counts(occ))

    def _armed(self, conv_iter=2, nwalkers=NWALKERS):
        m = _stage_move(shutoff=True, conv_iter=conv_iter, nwalkers=nwalkers)
        bi = _band_info(nwalkers=nwalkers, shutoff=True)
        m._rj_band_shutoff_w = bi["band_rj_shutoff_w"]
        # Bind the window through the SAME path production uses, so the
        # move's arrays are the ones band_info holds. Setting only
        # ``_rj_band_shutoff_w`` left ``_shutoff_band_info`` unset, and the
        # move then allocated a DETACHED window -- the tests still passed
        # because they read the move's attributes, but nothing they
        # asserted would have reached the saver.
        m._bind_shutoff_window(bi, np.shape(bi["band_rj_shutoff_w"]))
        return m, bi

    @staticmethod
    def _flat(value=0.0, nwalkers=NWALKERS):
        return np.full((nwalkers, NUM_BANDS), float(value))

    def test_a_flat_lnl_shuts_after_exactly_conv_iter(self):
        m, bi = self._armed(conv_iter=3)
        # update 1 IS an improvement (-inf -> 0), so the streak starts at 0
        # and the next three non-improving updates earn the shutoff.
        self._run(m, bi, [self._flat()] * 3)
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self._run(m, bi, [self._flat()])
        self.assertTrue(bi["band_rj_shutoff_w"].all())

    def test_an_improving_band_stays_open(self):
        m, bi = self._armed(conv_iter=2)
        for k in range(1, 10):
            lls = self._flat()
            lls[:, 0] = 10.0 * k        # band 0 keeps paying
            self._run(m, bi, [lls])
        self.assertFalse(bi["band_rj_shutoff_w"][:, 0].any())
        self.assertTrue(bi["band_rj_shutoff_w"][:, 1:].all())

    def test_an_improvement_below_the_threshold_does_not_reset(self):
        """D/2 = 4.0: drifting up by 1.0 a step is not a new source."""
        m, bi = self._armed(conv_iter=3)
        for k in range(8):
            self._run(m, bi, [self._flat(1.0 * k)])
        self.assertTrue(bi["band_rj_shutoff_w"].all())

    def test_an_improvement_above_the_threshold_resets_the_clock(self):
        m, bi = self._armed(conv_iter=3)
        self._run(m, bi, [self._flat(0.0)] * 3)     # streak 0,1,2
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self._run(m, bi, [self._flat(100.0)])        # > 4.0 -> streak 0
        self._run(m, bi, [self._flat(100.0)] * 2)    # streak 1,2
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self._run(m, bi, [self._flat(100.0)])        # streak 3 -> shut
        self.assertTrue(bi["band_rj_shutoff_w"].all())

    def test_the_best_is_a_RUNNING_max_so_a_dip_does_not_reopen(self):
        m, bi = self._armed(conv_iter=2)
        self._run(m, bi, [self._flat(100.0)])
        # falling back is not an improvement on the running best
        self._run(m, bi, [self._flat(10.0), self._flat(20.0),
                          self._flat(30.0)])
        self.assertTrue(bi["band_rj_shutoff_w"].all())

    def test_an_empty_band_never_shuts(self):
        """A band that has found nothing has no lnL to converge -- the
        ghost-increment failure in its valve form."""
        m, bi = self._armed(conv_iter=2)
        occ = np.ones((NWALKERS, NUM_BANDS), dtype=np.int64)
        occ[:, 0] = 0
        self._run(m, bi, [self._flat()] * 12, occ=occ)
        self.assertFalse(bi["band_rj_shutoff_w"][:, 0].any())
        self.assertTrue(bi["band_rj_shutoff_w"][:, 1:].all())

    def test_the_occupancy_guard_can_be_turned_off(self):
        m, bi = self._armed(conv_iter=2)
        occ = np.zeros((NWALKERS, NUM_BANDS), dtype=np.int64)
        os.environ["GB_SEARCH_BAND_SHUTOFF_REQUIRE_OCC"] = "0"
        try:
            self._run(m, bi, [self._flat()] * 6, occ=occ)
        finally:
            os.environ.pop("GB_SEARCH_BAND_SHUTOFF_REQUIRE_OCC", None)
        self.assertTrue(bi["band_rj_shutoff_w"].all())

    def test_walkers_shut_independently(self):
        m, bi = self._armed(conv_iter=2)
        for k in range(1, 10):
            lls = self._flat()
            lls[0, 1] = 50.0 * k        # walker 0 band 1 keeps improving
            self._run(m, bi, [lls])
        self.assertFalse(bool(bi["band_rj_shutoff_w"][0, 1]))
        self.assertTrue(bool(bi["band_rj_shutoff_w"][1, 1]))
        self.assertTrue(bool(bi["band_rj_shutoff_w"][2, 1]))

    def test_nan_lnl_is_folded_to_minus_inf_not_propagated(self):
        m, bi = self._armed(conv_iter=2)
        lls = self._flat()
        lls[0, 0] = np.nan
        self._run(m, bi, [lls] * 6)
        self.assertTrue(np.isfinite(bi["band_cold_logl_max_w"][1:, :]).all())
        # a NaN band never improves, so it converges like any flat one
        self.assertTrue(bool(bi["band_rj_shutoff_w"][0, 0]))

    def test_a_new_recipe_step_releases_the_valve_and_the_window(self):
        m, bi = self._armed(conv_iter=2)
        # ANNOUNCE THE STEP THIS VALVE IS EARNED UNDER. Without it the move
        # has no previous serial and `begin_recipe_step` reads as a FRESH
        # PROCESS, which deliberately does NOT release (2026-09-26) -- see
        # test_a_fresh_process_does_NOT_release_the_persisted_valve below.
        # Production always announces: a stage's setup_run calls
        # recipe.begin_search_recipe_step(moves, serial) on entry.
        m.begin_recipe_step(16)
        self._run(m, bi, [self._flat()] * 6)
        self.assertTrue(bi["band_rj_shutoff_w"].any())
        m.begin_recipe_step(17)
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        # THE WINDOW, not only the boolean: a step that inherited the
        # previous step's running-best lnL would re-freeze immediately.
        #
        # CLEARED IN PLACE, not rebound to None (2026-09-27). The window
        # arrays are live references into band_info now that they are
        # persisted, so a release has to be WRITTEN THROUGH -- dropping the
        # reference would leave the saved arrays holding the old step's
        # best and the next bind would pick it straight back up.
        self.assertTrue(np.all(np.isneginf(bi["band_shutoff_best_w"])))
        self.assertTrue(np.all(bi["band_shutoff_streak_w"] == 0))

    def test_step_2_is_judged_against_the_ALL_TIME_max(self):
        """CONTRACT REVERSED 2026-09-27 by user ruling.

        This test used to assert the opposite: "step 2 must NOT inherit
        step 1's verdict, and must not re-freeze on step 1's already-high
        best". The reasoning was that step 2 has different moves, caps and
        floors, so step 1's best is an unfair bar.

        The ruling now is to hold "an absolute logL max value for the cold
        chain per (band, walker) ... it should persist through everything
        including saving. We want to monitor against that." The tradeoff
        is taken knowingly: if step 2's moves genuinely find something
        better they beat the all-time max and the streak resets; if they
        cannot, the band IS finished and re-freezing it is the right
        answer. A reference that resets every step is why the valve could
        not latch at all across the short jobs of 2026-09-26.

        So: the BOOLEAN still releases at the step boundary (the verdict
        is re-earned) but the BAR does not move.
        """
        m, bi = self._armed(conv_iter=3)
        m.begin_recipe_step(1)
        self._run(m, bi, [self._flat(500.0)] * 6)
        self.assertTrue(bi["band_rj_shutoff_w"].all())
        peak = bi["band_cold_logl_max_w"].copy()

        m.begin_recipe_step(2)
        # the verdict is released ...
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        # ... the per-step window is cleared ...
        self.assertTrue(np.all(np.isneginf(bi["band_shutoff_best_w"])))
        # ... but the all-time max is NOT.
        np.testing.assert_array_equal(bi["band_cold_logl_max_w"], peak)

        # Three iterations at the same lnL cannot beat the inherited max,
        # so the streak runs straight to conv_iter and the band re-shuts.
        self._run(m, bi, [self._flat(500.0)] * 3)
        self.assertTrue(
            bi["band_rj_shutoff_w"].all(),
            "step 2 failed to re-freeze a band that never beat its "
            "all-time best")

    def test_a_genuinely_better_step_resets_the_streak(self):
        """The safety valve on the ruling above: beating the all-time max
        by more than the tolerance must reopen the question."""
        m, bi = self._armed(conv_iter=3)
        m.begin_recipe_step(1)
        self._run(m, bi, [self._flat(500.0)] * 6)
        m.begin_recipe_step(2)
        # +100 lnL is far beyond the D/2 tolerance
        self._run(m, bi, [self._flat(600.0)])
        self.assertTrue(np.all(bi["band_shutoff_streak_w"] == 0))
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self.assertAlmostEqual(
            float(bi["band_cold_logl_max_w"].max()), 600.0, places=6)

    def test_the_valve_HOLDS_across_a_mid_step_resume(self):
        """User requirement 2026-09-24: the shutoff holds through a recipe
        stage until a new one starts -- including across a restart.

        This is why the step identity is the step INDEX and not the backend
        iteration: a resume re-announces the SAME step, and the stored
        stamp must match so nothing is released.
        """
        m, bi = self._armed(conv_iter=2)
        m.begin_recipe_step(2)
        bi["band_shutoff_w_step"][0] = 2
        self._run(m, bi, [self._flat()] * 6)
        shut_before = bi["band_rj_shutoff_w"].copy()
        best_before = bi["band_shutoff_best_w"].copy()
        streak_before = bi["band_shutoff_streak_w"].copy()
        self.assertTrue(shut_before.any())

        # ---- the process dies and comes back mid-step ----------------
        fresh = _stage_move(shutoff=True, conv_iter=2)
        fresh._recipe_step_serial = 2          # same step re-announced
        fresh._arm_search_stage(_state(bi))
        np.testing.assert_array_equal(
            fresh._rj_band_shutoff_w, shut_before,
            "a mid-step resume released a valve the step had earned")
        # THE WINDOW NOW HOLDS TOO (user request 2026-09-27). It used to be
        # re-earned from -inf, which is permissive and cannot lose a
        # source, but it also meant no pair could EVER shut on a job
        # shorter than conv_iter + 1 iterations -- the 6mo v9 run shut zero
        # pairs in jobs 646, 648 and 650 for exactly that reason. The step
        # stamp still releases everything when the step changes, so this
        # can never outlive the configuration it was measured under.
        # The arrays are owned by band_info, not by the move (2026-09-27),
        # so a resume that binds the same state sees the same values --
        # asserted on band_info directly, because that IS the owner now.
        np.testing.assert_array_equal(bi["band_shutoff_best_w"], best_before)
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"],
                                      streak_before)
        self.assertIsNotNone(fresh._shutoff_band_info)
        self.assertIs(fresh._shutoff_band_info, bi)

    def test_a_resume_into_a_DIFFERENT_step_releases(self):
        m, bi = self._armed(conv_iter=2)
        m.begin_recipe_step(2)
        bi["band_shutoff_w_step"][0] = 2
        self._run(m, bi, [self._flat()] * 6)
        self.assertTrue(bi["band_rj_shutoff_w"].any())

        fresh = _stage_move(shutoff=True, conv_iter=2)
        fresh._recipe_step_serial = 3          # the run advanced while down
        fresh._arm_search_stage(_state(bi))
        self.assertFalse(fresh._rj_band_shutoff_w.any())
        self.assertEqual(int(bi["band_shutoff_w_step"][0]), 3)

    def test_arming_without_the_state_array_raises(self):
        """The one way the valve could be fully configured and still do
        nothing -- it is load-bearing, so it must not pass quietly."""
        m = _stage_move(shutoff=True)
        bi = _band_info(shutoff=False)
        with self.assertRaises(RuntimeError):
            m._arm_search_stage(_state(bi))

    def test_the_same_serial_twice_does_not_wipe_an_earned_valve(self):
        m, bi = self._armed(conv_iter=2)
        m.begin_recipe_step(4)
        self._run(m, bi, [self._flat()] * 6)
        self.assertTrue(bi["band_rj_shutoff_w"].any())
        m.begin_recipe_step(4)      # same step re-announced
        self.assertTrue(bi["band_rj_shutoff_w"].any())

    def test_recipe_helper_walks_nested_move_trees(self):
        m, bi = self._armed(conv_iter=2)
        m.begin_recipe_step(1)
        nested = SimpleNamespace(moves=[m])
        self._run(m, bi, [self._flat()] * 6)
        self.assertTrue(bi["band_rj_shutoff_w"].any())
        begin_search_recipe_step([nested], 2)
        self.assertFalse(bi["band_rj_shutoff_w"].any())

    def test_a_stale_stash_is_not_replayed(self):
        """A stash from a previous iteration must not be scored again --
        an unchanged lnL is exactly what 'converged' looks like."""
        m, bi = self._armed(conv_iter=1)
        st = _state(bi)
        m.num_proposals = 5
        m._stage_band_gain = self._flat()
        m._stage_band_lls_stamp = 4          # LAST iteration's
        m._cap_stats_local = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("no residual here"))
        m._update_search_band_shutoff(None, st, _counts(
            np.ones((NWALKERS, NUM_BANDS))))
        # inert, not frozen: nothing was measured this iteration
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        # The window is now pre-bound by the fixture (it has to be, or
        # nothing the move writes reaches band_info), so "the update bailed
        # out" is asserted as "the window is untouched" rather than as
        # "the window is still None".
        self.assertTrue(np.all(np.isneginf(bi["band_shutoff_best_w"])))
        self.assertTrue(np.all(np.isneginf(bi["band_cold_logl_max_w"])))
        self.assertTrue(np.all(bi["band_shutoff_streak_w"] == 0))

    def test_pe_mode_never_arms(self):
        m = _stage_move(shutoff=True, search=False)
        self.assertFalse(m._search_shutoff_per_walker)

    def test_non_rj_move_never_arms(self):
        m = _stage_move(shutoff=True)
        m.is_rj_prop = False
        self.assertFalse(m._search_shutoff_per_walker)

    def test_flag_off_is_a_no_op(self):
        m = _stage_move(shutoff=False)
        bi = _band_info(shutoff=False)
        m._update_search_band_shutoff(
            None, _state(bi), _counts(np.ones((NWALKERS, NUM_BANDS))))
        self.assertNotIn("band_rj_shutoff_w", bi)

    def test_row_mismatch_raises_rather_than_broadcasting(self):
        m, bi = self._armed(conv_iter=2)
        m.num_proposals = 1
        m._stage_band_gain = np.zeros((NWALKERS * 2, NUM_BANDS))
        m._stage_band_lls_stamp = 1
        with self.assertRaises(RuntimeError):
            m._update_search_band_shutoff(
                None, _state(bi), _counts(np.ones((NWALKERS, NUM_BANDS))))


class StageConvergenceInterfaceTest(unittest.TestCase):
    """The counter a recipe Stage gates on (cross-session contract)."""

    def _armed(self, conv_iter=2):
        m = _stage_move(shutoff=True, conv_iter=conv_iter)
        bi = _band_info(shutoff=True)
        m._rj_band_shutoff_w = bi["band_rj_shutoff_w"]
        # Bind through the production path so the move knows WHICH
        # band_info owns the record (it holds no arrays of its own).
        m._bind_shutoff_window(bi, np.shape(bi["band_rj_shutoff_w"]))
        return m, bi

    def _step(self, m, bi, value=0.0, occ=None):
        if occ is None:
            occ = np.ones((NWALKERS, NUM_BANDS), dtype=np.int64)
        m.num_proposals += 1
        m._stage_band_gain = np.full((NWALKERS, NUM_BANDS), float(value))
        m._stage_band_lls_stamp = m.num_proposals
        m._update_search_band_shutoff(None, _state(bi), _counts(occ))

    def test_pending_is_none_until_the_first_update(self):
        m, _ = self._armed()
        self.assertIsNone(m._shutoff_w_pending)
        self.assertFalse(band_shutoff_w_armed([m]))
        # absent counter contributes 0 -- a move without the feature has no
        # opinion and must not hold a stage open
        self.assertEqual(band_shutoff_w_pending_total([m]), 0)

    def test_pending_is_published_before_anything_converges(self):
        m, bi = self._armed(conv_iter=3)
        self._step(m, bi)
        # a stage polling now must not read a stale 0 and call it converged
        self.assertEqual(m._shutoff_w_pending, NWALKERS * NUM_BANDS)
        self.assertTrue(band_shutoff_w_armed([m]))

    def test_pending_falls_to_zero_on_full_convergence(self):
        m, bi = self._armed(conv_iter=2)
        for _ in range(6):
            self._step(m, bi)
        self.assertEqual(band_shutoff_w_pending_total([m]), 0)
        self.assertEqual(m._shutoff_w_shut, NWALKERS * NUM_BANDS)

    def test_empty_pairs_are_not_counted_as_pending(self):
        """Otherwise 'everything shut off' is unreachable in any run with
        an empty band, which is every run."""
        m, bi = self._armed(conv_iter=2)
        occ = np.ones((NWALKERS, NUM_BANDS), dtype=np.int64)
        occ[:, 0] = 0
        for _ in range(8):
            self._step(m, bi, occ=occ)
        self.assertEqual(band_shutoff_w_pending_total([m]), 0)
        self.assertFalse(bi["band_rj_shutoff_w"][:, 0].any())

    def test_a_release_resets_pending_so_a_stage_cannot_read_zero(self):
        m, bi = self._armed(conv_iter=2)
        m.begin_recipe_step(8)          # the step this valve is earned under
        for _ in range(6):
            self._step(m, bi)
        self.assertEqual(band_shutoff_w_pending_total([m]), 0)
        m.begin_recipe_step(9)
        # the next stage must not inherit "converged" before it has run
        self.assertIsNone(m._shutoff_w_pending)
        self.assertFalse(band_shutoff_w_armed([m]))

    def test_a_fresh_process_does_NOT_release_the_persisted_valve(self):
        """A RESTART must not wipe what the store is carrying.

        ``prev is None`` means "this process has not announced a step yet",
        NOT "the step changed". It used to fall through to the release,
        which defeated the whole band_rj_shutoff_w / band_shutoff_w_step
        persistence channel: the bind path is written to compare the STORED
        stamp against the current serial and KEEP a mask earned in the same
        step, and setup_run's release ran first every time.

        Measured on the 3mo v9 run: occupied (walker, band) pairs went
        1387 -> 54 -> 36 over 3.5 h inside one job, then RESET to 1693 the
        moment the next job restarted -- three times in 16 h, so
        gb_search_1 never advanced.
        """
        m, bi = self._armed(conv_iter=2)
        m.begin_recipe_step(4)
        for _ in range(6):
            self._step(m, bi)
        self.assertTrue(bi["band_rj_shutoff_w"].any())
        earned = bi["band_rj_shutoff_w"].copy()

        # simulate the restart: a NEW move object over the SAME band_info,
        # announcing the SAME step it was earned under.
        m2 = _stage_move(shutoff=True, conv_iter=2)
        m2._rj_band_shutoff_w = bi["band_rj_shutoff_w"]
        self.assertIsNone(m2._recipe_step_serial)
        m2.begin_recipe_step(4)
        np.testing.assert_array_equal(bi["band_rj_shutoff_w"], earned)

    def test_a_genuine_step_change_still_releases(self):
        m, bi = self._armed(conv_iter=2)
        m.begin_recipe_step(4)
        for _ in range(6):
            self._step(m, bi)
        self.assertTrue(bi["band_rj_shutoff_w"].any())
        m.begin_recipe_step(5)
        self.assertFalse(bi["band_rj_shutoff_w"].any())

    def test_helper_walks_nested_trees_and_unwraps_weights(self):
        m, bi = self._armed(conv_iter=2)
        self._step(m, bi)
        n = m._shutoff_w_pending
        self.assertEqual(
            band_shutoff_w_pending_total([SimpleNamespace(moves=[(m, 0.5)])]),
            n)
        self.assertTrue(
            band_shutoff_w_armed([SimpleNamespace(moves=[(m, 0.5)])]))



class FstatPeakFloorTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in (
            "FSTAT_PEAK_MIN_SNR", "FSTAT_PEAK_MIN_SNR_COARSE",
            "FSTAT_PEAK_MIN_SNR_FINE", "FSTAT_PEAK_MIN_F",
            "FSTAT_PEAKS_PER_BAND")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_stages_default_coarse_to_the_existing_floor(self):
        os.environ.pop("FSTAT_PEAK_MIN_SNR_COARSE", None)
        os.environ["FSTAT_PEAK_MIN_SNR"] = "8.0"
        coarse, fine = fstat_peak_min_F_stages()
        self.assertAlmostEqual(coarse, fstat_peak_min_F())
        self.assertAlmostEqual(fine, 0.5 * 6.25 ** 2)

    def test_inverted_floors_refuse(self):
        os.environ["FSTAT_PEAK_MIN_SNR"] = "5.0"
        os.environ.pop("FSTAT_PEAK_MIN_SNR_COARSE", None)
        os.environ["FSTAT_PEAK_MIN_SNR_FINE"] = "9.0"
        with self.assertRaises(ValueError):
            fstat_peak_min_F_stages()

    def test_band_floor_takes_the_looser_over_walkers(self):
        os.environ["FSTAT_PEAK_MIN_SNR"] = "8.0"
        os.environ.pop("FSTAT_PEAK_MIN_SNR_COARSE", None)
        os.environ["FSTAT_PEAK_MIN_SNR_FINE"] = "6.25"
        stage = np.zeros((NWALKERS, NUM_BANDS), dtype=np.int8)
        stage[1, 2] = SEARCH_STAGE_FINE       # ONE walker promoted
        out = fstat_band_min_F(stage, NUM_BANDS)
        # one shared catalog -> the band must cover the loosest floor
        self.assertAlmostEqual(out[2], 0.5 * 6.25 ** 2)
        self.assertAlmostEqual(out[0], 0.5 * 8.0 ** 2)

    def test_band_floor_is_none_when_off(self):
        self.assertIsNone(fstat_band_min_F(None, NUM_BANDS))

    def test_band_floor_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fstat_band_min_F(np.zeros((NWALKERS, NUM_BANDS + 3)), NUM_BANDS)

    def test_select_comb_peaks_scalar_is_unchanged(self):
        """Explicit scalar == the env default: bit-identical selection."""
        os.environ["FSTAT_PEAK_MIN_SNR"] = "8.0"
        os.environ["FSTAT_PEAKS_PER_BAND"] = "200"
        nodes, F = self._comb()
        a = select_comb_peaks(nodes, F, BAND_EDGES, self._spacing(nodes), np)
        b = select_comb_peaks(nodes, F, BAND_EDGES, self._spacing(nodes), np,
                              min_F=0.5 * 8.0 ** 2)
        np.testing.assert_array_equal(a, b)

    def test_select_comb_peaks_per_band_vector_selects_per_band(self):
        os.environ["FSTAT_PEAKS_PER_BAND"] = "200"
        nodes, F = self._comb()
        tight = np.full(NUM_BANDS, 0.5 * 8.0 ** 2)
        loose = tight.copy()
        loose[2] = 0.5 * 3.0 ** 2          # band 2 only
        a = select_comb_peaks(nodes, F, BAND_EDGES, self._spacing(nodes), np,
                              min_F=tight)
        b = select_comb_peaks(nodes, F, BAND_EDGES, self._spacing(nodes), np,
                              min_F=loose)
        # loosening ONE band adds peaks there and nowhere else
        self.assertGreater(
            int((b[:, 3] == 2).sum()), int((a[:, 3] == 2).sum()))
        for band in (1,):
            self.assertEqual(
                int((b[:, 3] == band).sum()), int((a[:, 3] == band).sum()))

    def test_select_comb_peaks_rejects_a_mismatched_vector(self):
        nodes, F = self._comb()
        with self.assertRaises(ValueError):
            select_comb_peaks(nodes, F, BAND_EDGES, self._spacing(nodes), np,
                              min_F=np.full(NUM_BANDS + 2, 32.0))

    @staticmethod
    def _spacing(nodes):
        return float(nodes[1] - nodes[0])

    @staticmethod
    def _comb():
        """A comb with a loud peak and a quiet one in every sub-band."""
        nodes = np.arange(5.0, 9.0, 0.01)          # mHz, inside BAND_EDGES
        F = np.full(nodes.shape, 1.0)
        for k in range(NUM_BANDS):
            lo = 5.0 + k
            F[np.argmin(np.abs(nodes - (lo + 0.30)))] = 0.5 * 20.0 ** 2
            F[np.argmin(np.abs(nodes - (lo + 0.70)))] = 0.5 * 5.0 ** 2
        return nodes, F


if __name__ == "__main__":
    unittest.main()


class DropSearchMovesKnobTest(unittest.TestCase):
    """``GB_SEARCH_DROP_MOVES`` -- omit a pure in-model slot.

    Cost knob from the 2026-09-28 iteration-time package. Job 662
    iteration 5 (5564 s): ``in_model`` 1392 s + ``in_model_fstat``
    1248 s = 47% of the wall clock, and the two are the SAME CLASS --
    one pure in-model polish per RJ move. With cold births at ~1e-4
    acceptance a single group-rule polish per iteration is enough.
    """

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("GB_SEARCH_DROP_MOVES", None)
        self.addCleanup(self.env.stop)
        self.m = self._mod()
        self.m._DROP_LOGGED.clear()

    @staticmethod
    def _mod():
        import importlib.util
        import os as _os
        p = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "scripts", "fstat_proposal", "run_combined_staged.py")
        spec = importlib.util.spec_from_file_location("_rcs_drop", p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        return mod

    def test_unset_drops_nothing(self):
        self.assertEqual(self.m._dropped_search_moves(), frozenset())
        self.assertFalse(self.m._drop_slot("in_model_fstat"))

    def test_the_launcher_value_drops_that_slot_only(self):
        os.environ["GB_SEARCH_DROP_MOVES"] = "in_model_fstat"
        self.assertTrue(self.m._drop_slot("in_model_fstat"))
        self.assertFalse(self.m._drop_slot("in_model"))
        self.assertFalse(self.m._drop_slot("in_model_replace"))

    def test_a_comma_list_with_spaces_parses(self):
        os.environ["GB_SEARCH_DROP_MOVES"] = " in_model_fstat , in_model "
        self.assertEqual(self.m._dropped_search_moves(),
                         frozenset({"in_model_fstat", "in_model"}))

    def test_rj_fstat_search_is_REFUSED_loudly(self):
        """Dropping it would leave the stage unable to END, and the
        symptom is indistinguishable from a broken gate -- which this
        run has already spent days on."""
        os.environ["GB_SEARCH_DROP_MOVES"] = "rj_fstat_search"
        with self.assertRaises(SystemExit) as cm:
            self.m._dropped_search_moves()
        msg = str(cm.exception)
        self.assertIn("designated updater", msg)
        self.assertIn("stage-end", msg)

    def test_it_is_refused_even_alongside_a_legal_name(self):
        os.environ["GB_SEARCH_DROP_MOVES"] = "in_model_fstat,rj_fstat_search"
        with self.assertRaises(SystemExit):
            self.m._dropped_search_moves()

    def test_the_drop_is_logged_once_per_slot(self):
        import logging
        os.environ["GB_SEARCH_DROP_MOVES"] = "in_model_fstat"
        with self.assertLogs(self.m.logger, level=logging.INFO) as cap:
            self.m._drop_slot("in_model_fstat")
            self.m._drop_slot("in_model_fstat")
            self.m.logger.info("sentinel")
        hits = [ln for ln in cap.output if "GB_SEARCH_DROP_MOVES" in ln]
        self.assertEqual(len(hits), 1, "must log once, not once per stage")
        self.assertIn("[V9-STAGE]", hits[0])

    def test_BOTH_stage_assemblies_consult_the_knob(self):
        """The file's own warning: gating only one assembly is the
        'knob resolves, consuming path never runs' shape that has
        produced several defects in this run."""
        import inspect
        src = inspect.getsource(self.m)
        self.assertEqual(src.count("if _drop_slot(slot):"), 2)

    def test_the_move_is_still_BUILT(self):
        """Only the stage composition drops it. A listed-but-unbuilt
        move fails recipe materialization, so removing it from the
        build would be the more dangerous edit."""
        import inspect
        self.assertIn("still BUILT",
                      inspect.getsource(self.m._drop_slot))


class IterationTimePackageLauncherTest(unittest.TestCase):
    """The three exports Mike should see, in BOTH launchers."""

    def _src(self, name):
        import pathlib
        return (pathlib.Path(__file__).resolve().parents[1]
                / "scripts" / "fstat_proposal" / name).read_text()

    def test_both_launchers_carry_the_package(self):
        for name in ("submit_gf_6mo_v9_4gpu.sh", "submit_gf_3mo_v9_2gpu.sh"):
            src = self._src(name)
            self.assertIn("export GB_FSTAT_REFIT_EVERY=2", src, name)
            self.assertNotIn("export GB_FSTAT_REFIT_EVERY=1\n", src, name)
            self.assertIn("export GB_SEARCH_BAND_SHUTOFF_LL_TOL=20", src, name)
            # ⚠ ABSENT ON PURPOSE. Dropping in_model_fstat was the
            # audit window's cost recommendation and MIKE REJECTED IT
            # (2026-09-28): the move stays, so the stage compositions
            # are unchanged from before the package. The knob itself
            # remains in the code as a documented no-op when unset.
            self.assertNotIn("export GB_SEARCH_DROP_MOVES", src, name)
            self.assertIn("DELIBERATELY UNSET", src, name)

    def test_the_tolerance_is_marked_a_JUDGEMENT_not_a_formula(self):
        """20 is Mike's direct choice (2026-09-28) and is NOT
        0.5 * SNR^2 for any floor in play -- that rule would give 32 at
        floor 8 and 12.5 at floor 5. The launcher has to say so, or the
        next reader "restores" a derived value and silently moves a
        gate threshold. Both numbers stay written down as the
        superseded rule, since Mike changes this by hand per stage.
        """
        for name in ("submit_gf_6mo_v9_4gpu.sh", "submit_gf_3mo_v9_2gpu.sh"):
            src = self._src(name)
            self.assertIn("JUDGEMENT, not a", src, name)
            self.assertIn("superseded", src, name)
            self.assertIn("12.5", src, name)     # the rule's stage-2 value


class FstatSkipsShutBandsTest(unittest.TestCase):
    """USER RULING 2026-09-28: "no fstat refit for shutdown bands".

    The scan is a SINGLE SHARED PASS -- ``_run_fstat_fit`` logs
    ``walker_ref=%d GLOBAL`` and there is one catalog for every walker --
    so a band may only be skipped when the level-3 valve is shut on
    EVERY walker. ``any`` would starve the walkers still open there.

    Job 662 split: stage A 720 s, stage B 409 s of a 1129 s epoch. The
    2026-10-10 ruling extends this to stage A's SWEEP (pinned in
    test_fstat_parallel_fit.CombBandSkipSweepTest) and ORs the per-band
    empty-band valve into the skip set (the per-band tests below).
    """

    NB = 4

    def _skip(self, shut):
        from lisatools.sampling.fstat_proposal import fstat_band_skip
        return fstat_band_skip(np.asarray(shut, dtype=bool), self.NB)

    def test_shut_on_EVERY_walker_is_skipped(self):
        out = self._skip([[True, False, True, False],
                          [True, False, False, False]])
        self.assertEqual([bool(v) for v in out],
                         [True, False, False, False])

    def test_shut_on_ONE_walker_is_NOT_skipped(self):
        """The starvation case: walker 1 is still open in band 2."""
        out = self._skip([[False, False, True, False],
                          [False, False, False, False]])
        self.assertFalse(bool(out[2]))

    def test_the_valve_being_off_scans_everything(self):
        from lisatools.sampling.fstat_proposal import fstat_band_skip
        self.assertIsNone(fstat_band_skip(None, self.NB))

    def test_a_mismatched_table_raises_rather_than_skipping_wrong_bands(self):
        from lisatools.sampling.fstat_proposal import fstat_band_skip
        with self.assertRaises(ValueError) as cm:
            fstat_band_skip(np.zeros((2, self.NB + 3), dtype=bool), self.NB)
        self.assertIn("WRONG", str(cm.exception))

    def test_release_reopens_the_band(self):
        """Release clears the shut flags, so the next fit scans it."""
        shut = np.ones((2, self.NB), dtype=bool)
        shut[:, 1] = False
        self.assertTrue(bool(self._skip(shut)[0]))
        shut[:] = False                       # what release() does
        self.assertFalse(self._skip(shut).any())

    def test_ALL_bands_shut_refuses_to_skip_any(self):
        """An empty catalog proposes nothing anywhere. Refuse rather
        than ship a fit that cannot help."""
        import logging
        import lisatools.globalfit.moves.gbspecialstretch as g
        mv = SimpleNamespace(
            _rj_band_shutoff_w=np.ones((2, self.NB), dtype=bool),
            num_bands=self.NB, name="rj_fstat_search")
        with self.assertLogs(g.logger, level=logging.WARNING) as cap:
            self.assertIsNone(g.fstat_band_skip_for(mv))
        self.assertTrue(any("every band is shut" in m for m in cap.output))

    def test_the_move_side_builder_degrades_when_unbound(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        self.assertIsNone(g.fstat_band_skip_for(SimpleNamespace()))

    def test_the_skip_set_is_part_of_the_EPOCH_FINGERPRINT(self):
        """⚠ run_fstat_grid_fit short-circuits a COMPLETE epoch straight
        off disk without re-selecting peaks. A catalog selected while
        88% of bands were shut would otherwise keep being handed back
        AFTER release -- missing most of the spectrum, for the rest of
        the run."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        shut = np.zeros((2, self.NB), dtype=bool); shut[:, 0] = True
        a = g.fstat_stage_fingerprint_for(SimpleNamespace(
            _rj_band_shutoff_w=shut.copy(), num_bands=self.NB, name="m"))
        shut2 = shut.copy(); shut2[:, 1] = True
        b = g.fstat_stage_fingerprint_for(SimpleNamespace(
            _rj_band_shutoff_w=shut2, num_bands=self.NB, name="m"))
        self.assertIn("bandskip=", a)
        self.assertNotEqual(a, b, "a changed skip set must be a new epoch")

    def test_the_fingerprint_is_EMPTY_with_both_features_off(self):
        """No existing cache key may change."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        self.assertEqual(
            g.fstat_stage_fingerprint_for(SimpleNamespace()), "")

    def test_the_fit_passes_the_skip_and_logs_the_count(self):
        import inspect
        import lisatools.globalfit.moves.gbspecialstretch as g
        import lisatools.sampling.fstat_gridfit as gf
        self.assertIn("band_skip=fstat_band_skip_for(self)",
                      inspect.getsource(g.GBSpecialRJFStatGridMove._run_fstat_fit))
        src = inspect.getsource(gf.run_fstat_grid_fit)
        self.assertIn("f0 nodes dropped from the sweep", src)

    # -- the per-BAND empty-band valve joins the skip set (2026-10-10) ---
    @staticmethod
    def _mv(shut_w=None, shut_b=None, live=True, nb=4, name="rj_fstat_search"):
        """Duck-typed move: the per-walker table, the per-band valve's
        ``_rj_band_shutoff`` and its liveness gate ``_band_shutoff_enabled``
        (the SAME gate run_proposal's RJ freeze uses)."""
        kw = dict(num_bands=nb, name=name)
        if shut_w is not None:
            kw["_rj_band_shutoff_w"] = np.asarray(shut_w, dtype=bool)
        if shut_b is not None:
            kw["_rj_band_shutoff"] = np.asarray(shut_b, dtype=bool)
            kw["_band_shutoff_enabled"] = (lambda: bool(live))
        return SimpleNamespace(**kw)

    def test_per_walker_only_is_unchanged(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        from lisatools.sampling.fstat_proposal import fstat_band_skip
        shut = np.array([[True, False, False, False],
                         [True, False, True, False]])
        got = g.fstat_band_skip_for(self._mv(shut_w=shut))
        np.testing.assert_array_equal(got, fstat_band_skip(shut, self.NB))

    def test_the_per_band_valve_JOINS_the_skip_set(self):
        """A band whose births are off cannot use a peak either -- and the
        648 bands above 10 mHz it shut on job 753 are the dense, nsky=512
        end of the comb, i.e. most of stage A."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        # 6 bands: interior 1..4, so the union {1, 3} is a real skip set
        # (with 4 bands {1, 2} would be every interior band -> refused).
        shut_w = np.zeros((2, 6), dtype=bool)
        shut_w[:, 1] = True
        shut_w[0, 2] = True                   # one walker only: not skipped
        got = g.fstat_band_skip_for(self._mv(
            shut_w=shut_w, shut_b=[False, False, False, True, False, False],
            nb=6))
        self.assertEqual([bool(v) for v in got],
                         [False, True, False, True, False, False])

    def test_the_per_band_valve_alone_still_skips(self):
        """Per-walker valve unbound (feature off): the per-band one is the
        whole skip set, not ignored."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        got = g.fstat_band_skip_for(self._mv(shut_b=[False, True, False,
                                                      False]))
        self.assertIsNotNone(got)
        self.assertEqual([bool(v) for v in got], [False, True, False, False])

    def test_a_per_band_valve_that_is_NOT_live_is_ignored(self):
        """Not live on this move (wrong scope, not the designated move): its
        stale array must not shrink the sweep."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        shut_w = np.zeros((2, self.NB), dtype=bool)
        shut_w[:, 1] = True
        got = g.fstat_band_skip_for(self._mv(
            shut_w=shut_w, shut_b=[False, False, True, False], live=False))
        self.assertEqual([bool(v) for v in got], [False, True, False, False])
        self.assertIsNone(g.fstat_band_skip_for(self._mv(
            shut_b=[True, True, False, False], live=False)))

    def test_the_all_shut_guard_applies_to_the_UNION(self):
        import logging
        import lisatools.globalfit.moves.gbspecialstretch as g
        shut_w = np.zeros((2, self.NB), dtype=bool)
        shut_w[:, :2] = True
        with self.assertLogs(g.logger, level=logging.WARNING) as cap:
            self.assertIsNone(g.fstat_band_skip_for(self._mv(
                shut_w=shut_w, shut_b=[False, False, True, True])))
        self.assertTrue(any("every band is shut" in m for m in cap.output))

    def test_every_INTERIOR_band_skipped_also_refuses(self):
        """Peaks are selected in interior bands only and the comb spans
        only them, so skipping every INTERIOR band is already the empty
        catalog the guard exists to refuse -- the edge bands do not count."""
        import logging
        import lisatools.globalfit.moves.gbspecialstretch as g
        with self.assertLogs(g.logger, level=logging.WARNING):
            self.assertIsNone(g.fstat_band_skip_for(self._mv(
                shut_b=[False, True, True, False])))

    def test_a_mismatched_per_band_valve_raises(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        with self.assertRaises(ValueError) as cm:
            g.fstat_band_skip_for(self._mv(shut_b=[True, False]))
        self.assertIn("WRONG", str(cm.exception))

    def test_the_per_band_valve_is_part_of_the_EPOCH_FINGERPRINT(self):
        import lisatools.globalfit.moves.gbspecialstretch as g
        shut_w = np.zeros((2, 6), dtype=bool)
        shut_w[:, 1] = True
        a = g.fstat_stage_fingerprint_for(self._mv(shut_w=shut_w, nb=6))
        b = g.fstat_stage_fingerprint_for(self._mv(
            shut_w=shut_w, shut_b=[False, False, False, True, False, False],
            nb=6))
        self.assertIn("bandskip=", a)
        self.assertIn("bandskip=", b)
        self.assertNotEqual(a, b)

    def _rj_grid_move(self, *, search):
        """A real GBSpecialRJFStatGridMove shell with the valve attributes
        the recipe sets, in SEARCH or PE configuration (rj_fstat_search vs
        rj_fstat_pe: same class, name, ``search_mode``)."""
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialRJFStatGridMove,
        )
        mv = GBSpecialRJFStatGridMove.__new__(GBSpecialRJFStatGridMove)
        mv.name = "rj_fstat_search" if search else "rj_fstat_pe"
        mv.branch_name = "gb"
        mv.num_bands = self.NB
        mv.is_rj_prop = True
        mv.rj_removal_only = False
        mv.rj_replace = False
        mv.leaf_cap_update = True      # the worst case: designated
        mv.search_mode = bool(search)
        mv.search_shutoff_per_walker = True
        mv.search_stage_per_walker = False
        mv._stage_armed_logged = True
        # Valve state left over from the search stage: the per-band set
        # (restored from the store) and the per-walker table in band_info.
        mv._rj_band_shutoff = np.array([False, True, False, False])
        return mv

    def test_a_PE_move_skips_NOTHING(self):
        """FULL_PE: both valves are search-scoped, so the fit must keep
        sweeping every band (None -> the complete comb) and the epoch key
        must carry no skip term, even with stale shut state on board."""
        import lisatools.globalfit.moves.gbspecialstretch as g
        env = {k: v for k, v in os.environ.items()
               if k != "GB_RJ_BAND_SHUTOFF_SCOPE"}
        with mock.patch.dict(os.environ, env, clear=True):
            pe = self._rj_grid_move(search=False)
            shut_w = np.zeros((2, self.NB), dtype=bool)
            shut_w[:, 2] = True
            state = SimpleNamespace(sub_states={"gb": SimpleNamespace(
                band_info={"band_rj_shutoff_w": shut_w})})
            pe._arm_search_stage(state)       # what propose runs in PE
            self.assertIsNone(pe._rj_band_shutoff_w)
            self.assertIsNone(g.fstat_band_skip_for(pe))
            self.assertEqual(g.fstat_stage_fingerprint_for(pe), "")
            # ... and the SAME shell in search configuration does skip, so
            # the None above is the scope gate and not a dead code path.
            se = self._rj_grid_move(search=True)
            got = g.fstat_band_skip_for(se)
            self.assertIsNotNone(got)
            self.assertEqual([bool(v) for v in got],
                             [False, True, False, False])

    # -- the BEHAVIOUR, not the wiring ---------------------------------
    def _comb(self):
        """A 4-band comb with one clear peak per band."""
        # bands 0..3 over 1.0-5.0 mHz, edges in Hz
        band_edges_hz = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3])
        spacing = 1e-4                                   # mHz per node
        f0 = np.arange(1.0, 5.0, spacing)                # mHz
        F = np.full(f0.shape, 1.0)
        for c in (1.5, 2.5, 3.5, 4.5):                   # one per band
            F[np.argmin(np.abs(f0 - c))] = 1e4
        return f0, F, band_edges_hz, spacing

    def test_a_SHUT_band_yields_no_peaks_and_an_open_one_still_does(self):
        from lisatools.sampling.fstat_gridfit import select_comb_peaks
        f0, F, edges, sp = self._comb()
        base = select_comb_peaks(f0, F, edges, sp, np, min_F=10.0)
        bands_before = set(int(r[3]) for r in base)
        self.assertEqual(bands_before, {1, 2},
                         "fixture: expect interior bands only")

        skip = np.zeros(self.NB, dtype=bool)
        skip[1] = True                                   # shut band 1
        got = select_comb_peaks(f0, F, edges, sp, np, min_F=10.0,
                                band_skip=skip)
        bands_after = set(int(r[3]) for r in got)
        self.assertNotIn(1, bands_after, "a shut band still produced peaks")
        self.assertIn(2, bands_after, "an open band lost its peaks")
        self.assertLess(len(got), len(base))

    def test_no_band_skip_is_bit_identical(self):
        from lisatools.sampling.fstat_gridfit import select_comb_peaks
        f0, F, edges, sp = self._comb()
        a = select_comb_peaks(f0, F, edges, sp, np, min_F=10.0)
        b = select_comb_peaks(f0, F, edges, sp, np, min_F=10.0,
                              band_skip=None)
        np.testing.assert_array_equal(a, b)

    def test_skipping_nothing_changes_nothing(self):
        from lisatools.sampling.fstat_gridfit import select_comb_peaks
        f0, F, edges, sp = self._comb()
        a = select_comb_peaks(f0, F, edges, sp, np, min_F=10.0)
        b = select_comb_peaks(f0, F, edges, sp, np, min_F=10.0,
                              band_skip=np.zeros(self.NB, dtype=bool))
        np.testing.assert_array_equal(a, b)

    def test_a_wrong_length_band_skip_raises(self):
        from lisatools.sampling.fstat_gridfit import select_comb_peaks
        f0, F, edges, sp = self._comb()
        with self.assertRaises(ValueError):
            select_comb_peaks(f0, F, edges, sp, np, min_F=10.0,
                              band_skip=np.zeros(self.NB + 2, dtype=bool))


class InModelAfterRemovalTest(unittest.TestCase):
    """A pure in-model pass after ``rj_prior_removal``.

    USER RULING 2026-09-28. ``rj_prior_removal`` was the ONLY RJ move
    with no in-model partner, so the SAVED state -- and the residual
    the next iteration's births are drawn against -- was the post-death
    state with no group-rule pass over it. The surviving twin of a
    killed double is exactly the cell that needs readjusting.

    ⚠ NOT a doubles cure: deaths are still JUDGED on the unadjusted
    residual. Correctness of the saved state, and better births.
    """

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("GB_SEARCH_INMODEL_AFTER_REMOVAL", "GB_SEARCH_DROP_MOVES"):
            os.environ.pop(k, None)
        self.addCleanup(self.env.stop)
        self.m = self._mod()
        self.m._DROP_LOGGED.clear()

    @staticmethod
    def _mod():
        import importlib.util
        import os as _os
        p = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "scripts", "fstat_proposal", "run_combined_staged.py")
        spec = importlib.util.spec_from_file_location("_rcs_rem", p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
        return mod

    @staticmethod
    def _src():
        import pathlib
        return (pathlib.Path(__file__).resolve().parents[1] / "scripts"
                / "fstat_proposal" / "run_combined_staged.py").read_text()

    # (a) composition and ORDER
    def test_it_FOLLOWS_rj_prior_removal_in_both_assemblies(self):
        """Before the removal it would polish the pre-death state,
        which is the opposite of the point."""
        import re
        src = self._src()
        # IMMEDIATELY after: nothing but whitespace and the list "+"
        # may sit between the removal move and its in-model partner.
        # An earlier version of this test used rindex to find "the
        # previous rj_prior_removal", which still succeeded when the
        # call was moved BEFORE it -- the spec's own mutation did not
        # fail. This pattern cannot be satisfied by the wrong order.
        pat = re.compile(
            r'Move\("rj_prior_removal",\s*branch="gb"\)\]\s*\+\s*'
            r'(?:gb_only_)?in_model\("in_model_removal"\)')
        self.assertEqual(
            len(pat.findall(src)), 2,
            "in_model_removal must directly follow rj_prior_removal in "
            "BOTH stage assemblies")

    def test_it_precedes_the_ridge_move(self):
        src = self._src()
        i = src.index('gb_only_in_model("in_model_removal")')
        self.assertLess(i, src.index("ridge()),", i))

    def test_BOTH_assemblies_gate_on_the_knob(self):
        """Gating one and not the other is the shape this file warns
        about; it has already bitten twice here."""
        self.assertEqual(
            self._src().count(
                'if slot == "in_model_removal" and not '
                "_inmodel_after_removal():"), 2)

    # (b) the knob
    def test_the_knob_is_ON_by_default(self):
        self.assertTrue(self.m._inmodel_after_removal())

    def test_turning_it_OFF_removes_the_slot(self):
        os.environ["GB_SEARCH_INMODEL_AFTER_REMOVAL"] = "0"
        self.assertFalse(self.m._inmodel_after_removal())

    # (d) it composes with the drop knob, guard unaffected
    def test_DROP_MOVES_can_name_it(self):
        os.environ["GB_SEARCH_DROP_MOVES"] = "in_model_removal"
        self.assertTrue(self.m._drop_slot("in_model_removal"))
        self.assertFalse(self.m._drop_slot("in_model"))

    def test_the_undroppable_guard_is_unaffected(self):
        os.environ["GB_SEARCH_DROP_MOVES"] = "rj_fstat_search"
        with self.assertRaises(SystemExit):
            self.m._dropped_search_moves()

    # the slot has to EXIST or materialization fails
    def test_the_slot_is_registered_so_the_move_is_BUILT(self):
        """A listed-but-unbuilt move fails recipe materialization."""
        import lisatools.globalfit.recipe as r
        import inspect
        src = inspect.getsource(r)
        i = src.index("GB_IN_MODEL_SLOTS = (")
        self.assertIn("in_model_removal", src[i:i + 200])

    # (c) the stage profile reaches it
    def test_the_profile_applies_by_DISCOVERY_not_a_name_list(self):
        """_apply_profile walks gb_moves_in_tree, so a new GB move in
        the stage is covered automatically and the '[V9-STAGE] ... over
        N GB move(s) [...]' line names it. Asserted because the spec
        assumed a name list that would have had to be edited."""
        import inspect
        import lisatools.globalfit.recipe as r
        src = inspect.getsource(r.SearchStageProfileStep._apply_profile)
        self.assertIn("gb_moves_in_tree(self.moves)", src)
        self.assertNotIn('"in_model_fstat"', src)

    # (e) the launchers
    def test_BOTH_launchers_export_it(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        for name in ("submit_gf_6mo_v9_4gpu.sh", "submit_gf_3mo_v9_2gpu.sh"):
            src = (root / "scripts" / "fstat_proposal" / name).read_text()
            self.assertIn("export GB_SEARCH_INMODEL_AFTER_REMOVAL=1",
                          src, name)
            self.assertIn("NOT A DOUBLES CURE", src, name)
