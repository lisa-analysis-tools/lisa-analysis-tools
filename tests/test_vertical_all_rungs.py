"""The all-rungs vertical swap: every rung of a (walker, band) column.

Implements ``docs/superpowers/specs/2026-09-25-vertical-swap-all-rungs-design.md``
(user 2026-09-25: "swapping across all rungs ... regardless if a
source/band is picked or not"; "even if it does not have a source in that
band it has a total likelihood ... and should get swapped accordingly").

The picked-row sweep can only pair picked-vs-picked, so a rung holding an
unpicked source -- or nothing -- was never a swap partner and a cold cell
whose whole model is net-harmful had no eviction route.

Three things get tests here because the audit named them as the ways this
goes silently wrong:

1. **empty<->empty pairs must never be proposed.** Identical totals give
   ``paccept == 0.0`` and ``acc = paccept >= log u`` accepts that
   unconditionally; at ntemps=24 a sparse column would contribute ~20
   always-accept pairs against 1-3 real ones and drive the band's measured
   acceptance to ~1, collapsing its ladder via ``_adapt_band_temps``.
2. **disjointness.** A rung enumerated both as a carrier and independently
   duplicates a source cell, and ``searchsorted(side="left")`` inside
   ``exchange_cell_labels_batch`` then maps it to one arbitrary
   destination -- a half-applied relabel that ``GB_INDEX_ASSERTS`` (off in
   production) would have caught.
3. **an unpriceable rung must be dropped, not guessed.** A wrong total is
   a silently mis-weighted swap.

Plus the gate: OFF by default, and the production picked-row path
untouched when it is off.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

import lisatools.globalfit.moves.gbspecialstretch as g


class KnobTest(unittest.TestCase):
    def test_it_is_OFF_by_default(self):
        """The spec's own instruction: both silent-corruption routes are
        invisible in aggregate logs, so this ships off."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_TEMPER_ALL_RUNGS", None)
            self.assertFalse(g._vert_all_rungs_on())

    def test_it_turns_on(self):
        for v in ("1", "true", "yes", "on"):
            with mock.patch.dict(os.environ, {"GB_TEMPER_ALL_RUNGS": v}):
                self.assertTrue(g._vert_all_rungs_on(), v)

    def test_junk_does_not_turn_it_on(self):
        for v in ("0", "", "off", "no", "maybe"):
            with mock.patch.dict(os.environ, {"GB_TEMPER_ALL_RUNGS": v}):
                self.assertFalse(g._vert_all_rungs_on(), v)


class PairEnumerationTest(unittest.TestCase):
    """:func:`_vert_all_rung_pairs` -- hazards 2 and the empty<->empty one."""

    def _pairs(self, carrier, occupied, parity, ntemps):
        return g._vert_all_rung_pairs(
            np.asarray(carrier), np.asarray(occupied), parity, ntemps, np)

    def test_an_unpicked_but_occupied_rung_IS_a_partner(self):
        """The whole point: carrier -1 does not disqualify a rung."""
        carrier = [[0, -1, -1, -1]]
        occupied = [[True, True, False, False]]
        ci, tc, th = self._pairs(carrier, occupied, 0, 4)
        self.assertIn((0, 1), list(zip(tc.tolist(), th.tolist())))

    def test_an_EMPTY_rung_is_a_partner_when_the_other_side_is_not(self):
        """A sole-occupant cold cell vs an empty hot rung is the eviction
        route that was missing."""
        carrier = [[0, -1]]
        occupied = [[True, False]]
        ci, tc, th = self._pairs(carrier, occupied, 0, 2)
        self.assertEqual(tc.tolist(), [0])
        self.assertEqual(th.tolist(), [1])

    def test_EMPTY_vs_EMPTY_is_never_proposed(self):
        """Hazard 2. paccept == 0.0 is accepted unconditionally and would
        drive the band's measured acceptance to ~1."""
        carrier = [[-1] * 6]
        occupied = [[False] * 6]
        ci, tc, th = self._pairs(carrier, occupied, 0, 6)
        self.assertEqual(ci.shape[0], 0)

    def test_only_the_empty_empty_pairs_are_dropped(self):
        carrier = [[0, -1, -1, -1, -1, -1]]
        occupied = [[True, False, False, False, True, False]]
        ci, tc, th = self._pairs(carrier, occupied, 0, 6)
        got = set(zip(tc.tolist(), th.tolist()))
        self.assertEqual(got, {(0, 1), (4, 5)})   # (2,3) is empty<->empty

    def test_parity_keeps_every_rung_in_at_most_one_pair(self):
        """The disjointness contract exchange_cell_labels_batch requires."""
        nt = 8
        occupied = [[True] * nt]
        for parity in (0, 1):
            ci, tc, th = self._pairs([[-1] * nt], occupied, parity, nt)
            touched = tc.tolist() + th.tolist()
            self.assertEqual(len(touched), len(set(touched)),
                             f"parity {parity} reused a rung")

    def test_both_parities_together_cover_the_whole_ladder(self):
        nt = 8
        seen = set()
        for parity in (0, 1):
            _, tc, th = self._pairs([[-1] * nt], [[True] * nt], parity, nt)
            seen |= set(zip(tc.tolist(), th.tolist()))
        self.assertEqual(seen, {(t, t + 1) for t in range(nt - 1)})

    def test_columns_do_not_pair_with_each_other(self):
        """Same walker, same band, adjacent temps -- never across columns."""
        ci, tc, th = self._pairs([[-1, -1], [-1, -1]],
                                 [[True, True], [True, True]], 0, 2)
        self.assertEqual(ci.tolist(), [0, 1])
        self.assertTrue(all(h == c + 1 for c, h in zip(tc.tolist(),
                                                       th.tolist())))


class ScoringTest(unittest.TestCase):
    """:func:`_vert_all_rung_L` -- live rows vs the cached total."""

    def test_a_carrier_rung_is_scored_LIVE(self):
        carrier = np.array([[0, -1]])
        cached = np.array([[111.0, 222.0]])
        L = g._vert_all_rung_L(carrier, cached, np.array([10.0]),
                               np.array([3.0]), np.array([0]),
                               np.array([0]), np)
        self.assertAlmostEqual(float(L[0]), 13.0)   # base + ll_ref, not 111

    def test_a_non_carrier_rung_is_scored_from_the_cache(self):
        carrier = np.array([[0, -1]])
        cached = np.array([[111.0, 222.0]])
        L = g._vert_all_rung_L(carrier, cached, np.array([10.0]),
                               np.array([3.0]), np.array([0]),
                               np.array([1]), np)
        self.assertAlmostEqual(float(L[0]), 222.0)

    def test_mixed_rungs_in_one_call(self):
        carrier = np.array([[0, -1, 1]])
        cached = np.array([[0.0, 5.0, 0.0]])
        L = g._vert_all_rung_L(
            carrier, cached, np.array([10.0, 20.0]), np.array([1.0, 2.0]),
            np.array([0, 0, 0]), np.array([0, 1, 2]), np)
        self.assertEqual([float(x) for x in L], [11.0, 5.0, 22.0])


class CachedTotalsTest(unittest.TestCase):
    """:func:`_vert_all_rung_cached` -- the FREE bare-slab measurement."""

    def _call(self, n_alive, carrier, occupied, vert_base, t_i, w_i, b_i,
              nb=4):
        cols = np.array([0], dtype=np.int64)
        return g._vert_all_rung_cached(
            cols, np.asarray(carrier), np.asarray(occupied),
            np.asarray(n_alive), np.asarray(vert_base, dtype=float),
            np.asarray(t_i), np.asarray(w_i), np.asarray(b_i), nb, np)

    def test_a_sole_occupant_picked_row_yields_the_column_bare_value(self):
        """After its one source is removed the slab IS bare, and every rung
        of the column reads a bit-identical slab -- one number per column."""
        cached, scor = self._call(
            n_alive=[[1, 0]], carrier=[[0, -1]], occupied=[[True, False]],
            vert_base=[-7.5], t_i=[0], w_i=[0], b_i=[0])
        self.assertAlmostEqual(float(cached[0, 1]), -7.5)
        self.assertTrue(bool(scor[0, 1]))       # empty rung now priceable

    def test_a_MULTI_leaf_picked_row_does_NOT_give_the_bare_value(self):
        """Its other leaves are still subtracted, so _vert_base is not the
        bare slab. Using it would be a wrong total."""
        cached, scor = self._call(
            n_alive=[[3, 0]], carrier=[[0, -1]], occupied=[[True, False]],
            vert_base=[-7.5], t_i=[0], w_i=[0], b_i=[0])
        self.assertFalse(bool(scor[0, 1]),
                         "empty rung priced off a multi-leaf cell's base")

    def test_an_occupied_but_UNPICKED_rung_is_not_scorable(self):
        """Its total is bare + its own sources, which nothing measured.
        Needs the spec's transient pass; must be dropped, not guessed."""
        cached, scor = self._call(
            n_alive=[[1, 2]], carrier=[[0, -1]], occupied=[[True, True]],
            vert_base=[-7.5], t_i=[0], w_i=[0], b_i=[0])
        self.assertFalse(bool(scor[0, 1]))

    def test_a_carrier_rung_is_always_scorable(self):
        cached, scor = self._call(
            n_alive=[[5, 0]], carrier=[[0, -1]], occupied=[[True, False]],
            vert_base=[-7.5], t_i=[0], w_i=[0], b_i=[0])
        self.assertTrue(bool(scor[0, 0]))


class TablesTest(unittest.TestCase):
    """:func:`_vert_all_rung_tables` -- carrier vs occupied are independent."""

    def test_carrier_and_occupied_are_independent(self):
        t_i = np.array([0]); w_i = np.array([0]); b_i = np.array([2])

        def alive(spec):
            # rung 1 holds an alive-but-unpicked source
            out = np.zeros(spec.shape, dtype=np.int64)
            out[..., 0] = 1
            out[..., 1] = 4
            return out

        cols, carrier, occ, nal = g._vert_all_rung_tables(
            t_i, w_i, b_i, 3, 4, 8, alive, np)
        self.assertEqual(carrier[0].tolist(), [0, -1, -1])
        self.assertEqual(occ[0].tolist(), [True, True, False])
        self.assertEqual(nal[0].tolist(), [1, 4, 0])

    def test_columns_are_the_distinct_walker_band_of_the_picked_rows(self):
        cols, carrier, occ, nal = g._vert_all_rung_tables(
            np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([2, 2, 3]),
            2, 4, 8, lambda s: np.ones(s.shape, dtype=np.int64), np)
        self.assertEqual(int(cols.shape[0]), 2)     # (0,2) and (1,3)


class ProductionPathUntouchedTest(unittest.TestCase):
    """The gate is worth nothing if the off path changed."""

    def test_the_picked_row_sweep_is_a_SEPARATE_method(self):
        self.assertTrue(hasattr(g.GBSpecialBase, "_vertical_swap_sweep"))
        self.assertTrue(
            hasattr(g.GBSpecialBase, "_vertical_swap_sweep_all_rungs"))

    def test_the_all_rung_state_is_None_unless_the_knob_is_on(self):
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertIn("_ar_state = None", src)
        self.assertIn("_vert_all_rungs_on()", src)
        # The old sweep is still what runs when the state is None.
        #
        # Find the branch that actually CONTAINS the sweep rather than the
        # first or last one: there are now three ``if _ar_state is not
        # None:`` guards (label-window widening, the sweep, and the
        # freeze-to-cache at the poll), and indexing by position made this
        # test break the moment a fourth was added -- it was asserting
        # about whichever guard happened to be last, not about the sweep.
        _needle = "self._vertical_swap_sweep_all_rungs("
        _j = src.index(_needle)
        i = src.rindex("if _ar_state is not None:", 0, _j)
        blk = src[i:i + 1200]
        self.assertIn("self._vertical_swap_sweep_all_rungs(", blk)
        self.assertIn("else:", blk)
        self.assertIn("self._vertical_swap_sweep(", blk)
        # and the all-rung call must come first in that if/else
        self.assertLess(blk.index("_vertical_swap_sweep_all_rungs("),
                        blk.index("else:"))

    def test_the_label_window_is_widened_only_when_armed(self):
        """GB_CELL_LABEL_DEFERRED is on by default and naming a cell
        outside the window raises at the flush -- but widening it
        unconditionally would change the off path."""
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        i = src.index("_cell_window = False")
        blk = src[i:i + 1400]
        self.assertIn("if _ar_state is not None:", blk)
        self.assertIn("begin_cell_label_window(_win_spec)", blk)

    def test_the_new_sweep_refuses_to_run_without_cell_ll_base(self):
        """ll_ref alone is an add-delta against DIFFERENT residuals; the
        ratio needs whole-cell totals."""
        m = g.GBSpecialBase.__new__(g.GBSpecialBase)
        # The guard must fire BEFORE any instance state is touched -- this
        # object has no __init__ run, so anything else raises AttributeError
        # and the test would pass for the wrong reason.
        with self.assertRaises(ValueError):
            g.GBSpecialBase._vertical_swap_sweep_all_rungs(
                m, None, None, None, None, None, None, None, None, None,
                None, None, None, 0, None, cell_ll_base=None)


if __name__ == "__main__":
    unittest.main()


class _FakeBuffer:
    """Minimal buffer: slot labels, an active mask, and a slot-scored ll."""

    def __init__(self, specials, lls, active=None):
        self.slot_specials = np.asarray(specials, dtype=np.int64)
        self._lls = np.asarray(lls, dtype=float)
        self.slot_active = (np.ones(len(specials), bool) if active is None
                            else np.asarray(active, bool))
        self.calls = 0

    def band_likelihoods(self, source_only=False, slots=None):
        assert source_only, "the gate's unit is -1/2<r|r>, i.e. source_only"
        self.calls += 1
        return self._lls[np.asarray(slots, dtype=np.int64)]


class AllRungsAreMeasuredNotInferredTest(unittest.TestCase):
    """Every rung is priceable, so every rung can swap.

    USER RULING 2026-09-27: "all of the rungs should perform vertical
    tempering swaps everywhere that happens. The likelihood for non-active
    rows does not change so it does not need to persist, just needs one
    compute."

    The old code refused to measure and inferred the bare slab from a
    sole-occupant picked row, which priced empties only in columns that
    happened to carry one and dropped OCCUPIED-but-unpicked cells
    entirely -- 620.0 M of 875.0 M candidate pairs on 6mo job 650.
    """

    NB, NT, NW = 4, 3, 2

    def _setup(self, carrier, occupied):
        from lisatools.globalfit.moves.gbbands import pack_special_index
        cols = np.array([0 * self.NB + 1, 1 * self.NB + 2], dtype=np.int64)
        n_cols = len(cols)
        w = (cols // self.NB)[:, None]
        b = (cols % self.NB)[:, None]
        t = np.arange(self.NT)[None, :]
        spec = pack_special_index(t, w, b, self.NW).astype(np.int64).ravel()
        lls = -np.arange(1.0, spec.size + 1.0)          # distinct per slot
        return cols, np.asarray(carrier), np.asarray(occupied), spec, lls

    def test_every_resident_rung_becomes_scorable(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _vert_all_rung_cached)
        carrier = np.array([[0, -1, -1], [-1, 1, -1]])     # one live each
        occupied = np.array([[True, False, True],          # incl. UNPICKED
                             [False, True, False]])
        cols, carrier, occupied, spec, lls = self._setup(carrier, occupied)
        buf = _FakeBuffer(spec, lls)
        cached, scor = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np, buffer_obj=buf, nwalkers=self.NW)
        self.assertTrue(scor.all(), "a resident rung was left unpriceable")
        self.assertEqual(buf.calls, 1, "must be ONE compute for the block")

    def test_the_occupied_but_UNPICKED_bucket_is_now_priced(self):
        """The bucket that grows as the run converges: a cell holding
        sources whose rows have all frozen has no picked row."""
        from lisatools.globalfit.moves.gbspecialstretch import (
            _vert_all_rung_cached)
        carrier = np.array([[0, -1, -1], [-1, -1, -1]])
        occupied = np.array([[True, True, True], [True, True, True]])
        cols, carrier, occupied, spec, lls = self._setup(carrier, occupied)
        buf = _FakeBuffer(spec, lls)
        cached, scor = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np, buffer_obj=buf, nwalkers=self.NW)
        self.assertTrue(scor.all())
        # the live rung keeps 0 (scored elsewhere from cell_ll_base+ll_ref);
        # every other rung carries its measured slab value
        self.assertTrue((cached[carrier < 0] != 0.0).all())

    def test_a_non_resident_rung_stays_unscorable(self):
        """Residency is required, not assumed."""
        from lisatools.globalfit.moves.gbspecialstretch import (
            _vert_all_rung_cached)
        carrier = np.array([[0, -1, -1], [-1, 1, -1]])
        occupied = np.array([[True, False, False], [False, True, False]])
        cols, carrier, occupied, spec, lls = self._setup(carrier, occupied)
        spec2 = spec.copy(); spec2[-1] = 10**9        # evict the last cell
        buf = _FakeBuffer(spec2, lls)
        cached, scor = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np, buffer_obj=buf, nwalkers=self.NW)
        self.assertFalse(bool(scor.ravel()[-1]))

    def test_a_retired_slot_is_not_trusted(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _vert_all_rung_cached)
        carrier = np.array([[0, -1, -1], [-1, 1, -1]])
        occupied = np.array([[True, False, False], [False, True, False]])
        cols, carrier, occupied, spec, lls = self._setup(carrier, occupied)
        act = np.ones(len(spec), bool); act[-1] = False
        buf = _FakeBuffer(spec, lls, active=act)
        cached, scor = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np, buffer_obj=buf, nwalkers=self.NW)
        self.assertFalse(bool(scor.ravel()[-1]))

    def test_without_a_buffer_it_falls_back_to_the_old_inference(self):
        """The pre-2026-09-27 path stays reachable and still tested."""
        from lisatools.globalfit.moves.gbspecialstretch import (
            _vert_all_rung_cached)
        carrier = np.array([[0, -1, -1], [-1, 1, -1]])
        occupied = np.array([[True, False, False], [False, True, False]])
        cols, carrier, occupied, spec, lls = self._setup(carrier, occupied)
        cached, scor = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np)
        # no vert_base -> nothing inferable -> only the live rungs score
        np.testing.assert_array_equal(scor, carrier >= 0)


class FrozenRungsArePricedFromCacheTest(unittest.TestCase):
    """A converged row keeps swapping; it just stops being re-priced.

    USER RULING 2026-09-27: "you still swap the converged rows even if
    they are turned off since their likelihood is frozen and just a label
    switch."

    Numerically neutral by construction -- a frozen row is out of
    ``_half_pre`` so no proposal is ever accepted for it and its
    ``ll_ref`` cannot move. What this pins is the REPRESENTATION: the
    sweep must stop reading a frozen row's live arrays, which is the
    prerequisite for evicting frozen rows from the buffer (1-yr TODO 7/8).
    """

    NB = 4

    def _ar(self, n_cols=2, ntemps=3):
        cols = np.array([0 * self.NB + 1, 1 * self.NB + 2], dtype=np.int64)
        carrier = np.full((n_cols, ntemps), -1, dtype=np.int64)
        carrier[0, 0] = 0          # row 0 lives on column 0, rung 0
        carrier[1, 1] = 1          # row 1 lives on column 1, rung 1
        occupied = np.ones((n_cols, ntemps), bool)
        cached = np.zeros((n_cols, ntemps))
        scorable = np.ones((n_cols, ntemps), bool)
        frozen = np.zeros((n_cols, ntemps), bool)
        return [cols, carrier, occupied, cached, scorable, frozen]

    def _rows(self):
        t_i = np.array([0, 1]); w_i = np.array([0, 1]); b_i = np.array([1, 2])
        base = np.array([-10.0, -20.0]); ll = np.array([1.0, 2.0])
        return t_i, w_i, b_i, base, ll

    def test_a_frozen_carrier_is_priced_at_its_freeze_time_value(self):
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        n = g._ar_freeze_rows(ar, np.array([0]), t_i, w_i, b_i, base, ll,
                              self.NB, np)
        self.assertEqual(n, 1)
        cols, carrier, occ, cached, scor, frozen = ar
        ll[0] += 100.0                      # the live array moves...
        L = g._vert_all_rung_L(carrier, cached, base, ll,
                               np.array([0]), np.array([0]), np,
                               frozen=frozen)
        self.assertAlmostEqual(float(L[0]), -9.0)   # ...the price does not

    def test_NEGATIVE_CONTROL_an_unfrozen_carrier_follows_the_live_value(self):
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        cols, carrier, occ, cached, scor, frozen = ar
        ll[0] += 100.0
        L = g._vert_all_rung_L(carrier, cached, base, ll,
                               np.array([0]), np.array([0]), np,
                               frozen=frozen)
        self.assertAlmostEqual(float(L[0]), 91.0)

    def test_the_freeze_value_equals_the_live_value_at_freeze_time(self):
        """Identity: freezing must not shift the price."""
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        cols, carrier, occ, cached, scor, frozen = ar
        before = float(g._vert_all_rung_L(carrier, cached, base, ll,
                                          np.array([0]), np.array([0]), np,
                                          frozen=frozen)[0])
        g._ar_freeze_rows(ar, np.array([0]), t_i, w_i, b_i, base, ll,
                          self.NB, np)
        after = float(g._vert_all_rung_L(carrier, cached, base, ll,
                                         np.array([0]), np.array([0]), np,
                                         frozen=ar[5])[0])
        self.assertAlmostEqual(before, after)

    def test_freezing_a_row_that_is_not_the_carrier_is_REFUSED(self):
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        ar[1][0, 0] = 7                      # some other row is the carrier
        n = g._ar_freeze_rows(ar, np.array([0]), t_i, w_i, b_i, base, ll,
                              self.NB, np)
        self.assertEqual(n, 0)
        self.assertFalse(ar[5].any(), "wrote a cache entry on a bad belief")

    def test_freezing_is_idempotent(self):
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        self.assertEqual(g._ar_freeze_rows(ar, np.array([0]), t_i, w_i, b_i,
                                           base, ll, self.NB, np), 1)
        self.assertEqual(g._ar_freeze_rows(ar, np.array([0]), t_i, w_i, b_i,
                                           base, ll, self.NB, np), 0)

    def test_an_accepted_swap_permutes_frozen_with_cached_and_carrier(self):
        """Otherwise the arriving LIVE row is priced from a cache slot
        that now holds someone else's total -- only for pairs where
        exactly one side was frozen, which is the hard case to spot."""
        cols, carrier, occ, cached, scor, frozen = self._ar()
        ci = np.array([0]); t_c = np.array([0]); t_h = np.array([1])
        cached[0, 0], cached[0, 1] = -9.0, -3.0
        frozen[0, 0] = True
        for arr in (carrier, cached, occ, scor, frozen):
            _h = arr[ci, t_h].copy()
            arr[ci, t_h] = arr[ci, t_c]
            arr[ci, t_c] = _h
        self.assertTrue(bool(frozen[0, 1]), "the flag stayed behind")
        self.assertFalse(bool(frozen[0, 0]))
        self.assertAlmostEqual(float(cached[0, 1]), -9.0)

    def test_the_frozen_flag_does_NOT_clear_the_carrier(self):
        """t_i / beta must keep following an accepted swap for a frozen
        row: the cold-lnL observe picks cold slots with ``t_i == 0`` and
        ``_converge_gate_mask`` reads t_i at the poll and in absorb."""
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        g._ar_freeze_rows(ar, np.array([0]), t_i, w_i, b_i, base, ll,
                          self.NB, np)
        self.assertEqual(int(ar[1][0, 0]), 0,
                         "carrier cleared; t_i would go stale")

    def test_the_state_tuple_carries_six_entries(self):
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertIn("_ar_state = (_cols, _carrier, _occ, _cached, _scor,",
                      src)
        sweep = inspect.getsource(
            g.GBSpecialBase._vertical_swap_sweep_all_rungs)
        self.assertIn(
            "cols, carrier, occupied, cached, scorable, frozen = ar", sweep)
