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
        # The third return is the rung -> slot map; these cases exercise
        # the inference path, which has none, so drop it here.
        return g._vert_all_rung_cached(
            cols, np.asarray(carrier), np.asarray(occupied),
            np.asarray(n_alive), np.asarray(vert_base, dtype=float),
            np.asarray(t_i), np.asarray(w_i), np.asarray(b_i), nb, np)[:2]

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
    """A slot-scored ll, and NOTHING ELSE.

    ⚠ DELIBERATELY has no ``slot_specials`` / ``slot_active``. The real
    :class:`SubBandBuffer` has never had either -- they are
    :class:`BandScheduler` properties -- and the first version of this
    fake carried them anyway. Every test here passed while production
    raised ``AttributeError`` on its first call and fell back to the
    inference for 4,532 consecutive blocks on 6mo v9 job 654. A fake that
    is more capable than the real object tests nothing.
    """

    def __init__(self, lls):
        self._lls = np.asarray(lls, dtype=float)
        self.calls = 0

    def band_likelihoods(self, source_only=False, slots=None):
        assert source_only, "the gate's unit is -1/2<r|r>, i.e. source_only"
        self.calls += 1
        return self._lls[np.asarray(slots, dtype=np.int64)]


class _FakeScheduler:
    """Slot labels + the active mask -- where the real code reads them."""

    def __init__(self, specials, active=None):
        self.slot_specials = np.asarray(specials, dtype=np.int64)
        self.slot_active = (np.ones(len(specials), bool) if active is None
                            else np.asarray(active, bool))


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
        buf, sch = _FakeBuffer(lls), _FakeScheduler(spec)
        cached, scor, _slotmap = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np, buffer_obj=buf, scheduler=sch,
            nwalkers=self.NW)
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
        buf, sch = _FakeBuffer(lls), _FakeScheduler(spec)
        cached, scor, _slotmap = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np, buffer_obj=buf, scheduler=sch,
            nwalkers=self.NW)
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
        buf, sch = _FakeBuffer(lls), _FakeScheduler(spec2)
        cached, scor, _slotmap = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np, buffer_obj=buf, scheduler=sch,
            nwalkers=self.NW)
        self.assertFalse(bool(scor.ravel()[-1]))

    def test_a_retired_slot_is_not_trusted(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _vert_all_rung_cached)
        carrier = np.array([[0, -1, -1], [-1, 1, -1]])
        occupied = np.array([[True, False, False], [False, True, False]])
        cols, carrier, occupied, spec, lls = self._setup(carrier, occupied)
        act = np.ones(len(spec), bool); act[-1] = False
        buf, sch = _FakeBuffer(lls), _FakeScheduler(spec, active=act)
        cached, scor, _slotmap = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np, buffer_obj=buf, scheduler=sch,
            nwalkers=self.NW)
        self.assertFalse(bool(scor.ravel()[-1]))

    def test_without_a_buffer_it_falls_back_to_the_old_inference(self):
        """The pre-2026-09-27 path stays reachable and still tested."""
        from lisatools.globalfit.moves.gbspecialstretch import (
            _vert_all_rung_cached)
        carrier = np.array([[0, -1, -1], [-1, 1, -1]])
        occupied = np.array([[True, False, False], [False, True, False]])
        cols, carrier, occupied, spec, lls = self._setup(carrier, occupied)
        cached, scor, _slotmap = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np)
        # no vert_base -> nothing inferable -> only the live rungs score
        np.testing.assert_array_equal(scor, carrier >= 0)

    def test_a_scheduler_is_required_for_the_measured_path(self):
        """buffer_obj alone must NOT be treated as enough.

        Regression for job 654: the measured path was entered on
        ``buffer_obj`` alone, went looking for slot labels that only the
        scheduler has, and the blanket ``except`` converted the
        AttributeError into a silent reversion to the inference.
        """
        from lisatools.globalfit.moves.gbspecialstretch import (
            _vert_all_rung_cached)
        carrier = np.array([[0, -1, -1], [-1, 1, -1]])
        occupied = np.array([[True, True, True], [True, True, True]])
        cols, carrier, occupied, spec, lls = self._setup(carrier, occupied)
        buf = _FakeBuffer(lls)
        cached, scor, _slotmap = _vert_all_rung_cached(
            cols, carrier, occupied, None, None, np.zeros(0), np.zeros(0),
            np.zeros(0), self.NB, np, buffer_obj=buf, nwalkers=self.NW)
        self.assertEqual(buf.calls, 0, "measured without slot labels")
        np.testing.assert_array_equal(scor, carrier >= 0)


class SlotLabelsComeFromTheSchedulerTest(unittest.TestCase):
    """The contract the fake got wrong, checked against the REAL classes.

    Job 654 (2026-09-27) ran 4,532 blocks with the all-rungs measurement
    disabled because ``_vert_all_rung_cached`` read ``slot_specials`` off
    ``buffer_obj``. Every unit test passed: the fake buffer carried the
    attribute the real one does not. These assertions are cheap, need no
    instantiation, and are the ones that would have caught it.
    """

    def test_slot_specials_is_a_BandScheduler_property(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler
        self.assertIsInstance(
            getattr(BandScheduler, "slot_specials", None), property)

    def test_SubBandBuffer_does_NOT_carry_the_slot_labels(self):
        from lisatools.globalfit.moves.gbbands import SubBandBuffer
        for name in ("slot_specials", "slot_active"):
            self.assertFalse(
                hasattr(SubBandBuffer, name),
                f"SubBandBuffer grew {name!r}: if that is deliberate, the "
                "docstring warning in _vert_all_rung_cached and this test "
                "both need revisiting -- but two sources of slot labels "
                "is the ambiguity that caused the job 654 regression.")

    def test_the_measured_path_reads_the_labels_off_the_scheduler(self):
        import inspect
        from lisatools.globalfit.moves import gbspecialstretch as g
        src = inspect.getsource(g._vert_all_rung_cached)
        body = src[src.index("the measured path"):]
        self.assertIn("scheduler.slot_specials", body)
        self.assertNotIn("buffer_obj.slot_specials", body)

    def test_every_production_call_site_passes_a_scheduler(self):
        """The knob reaching nothing is the recurring failure mode."""
        import inspect, re
        from lisatools.globalfit.moves import gbspecialstretch as g
        src = inspect.getsource(g)
        calls = [m for m in re.finditer(r"_vert_all_rung_cached\(", src)]
        # one definition + the production call(s)
        self.assertGreaterEqual(len(calls), 2)
        for m in calls[1:]:
            chunk = src[m.end():m.end() + 400]
            chunk = chunk[:chunk.index(")\n")] if ")\n" in chunk else chunk
            self.assertIn("scheduler=", chunk,
                          "a _vert_all_rung_cached call site omits "
                          "scheduler=, so it silently prices nothing")

    def test_run_in_model_repeats_is_always_given_the_scheduler(self):
        """Job 654's sibling gap: _polish called it without a scheduler."""
        import inspect, re
        from lisatools.globalfit.moves import gbspecialstretch as g
        src = inspect.getsource(g)
        for m in re.finditer(r"self\._run_in_model_repeats\(", src):
            chunk = src[m.end():m.end() + 900]
            depth, end = 1, len(chunk)
            for i, ch in enumerate(chunk):
                depth += (ch == "(") - (ch == ")")
                if depth == 0:
                    end = i
                    break
            self.assertIn("scheduler=", chunk[:end],
                          "a _run_in_model_repeats call omits scheduler=, "
                          "so all-rung pricing is off for that path")


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
        # rung -> slot: identity, every cell resident (the case the
        # two-sided re-point exists for).
        ar_slot = np.arange(n_cols * ntemps, dtype=np.int64).reshape(
            n_cols, ntemps)
        return [cols, carrier, occupied, cached, scorable, frozen,
                ar_slot]

    def _rows(self):
        t_i = np.array([0, 1]); w_i = np.array([0, 1]); b_i = np.array([1, 2])
        base = np.array([-10.0, -20.0]); ll = np.array([1.0, 2.0])
        return t_i, w_i, b_i, base, ll

    def test_a_frozen_carrier_is_priced_at_its_freeze_time_value(self):
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        n = g._ar_freeze_rows(ar, np.array([0]), t_i, w_i, b_i, base, ll,
                              self.NB, np)
        self.assertEqual(n, 1)
        cols, carrier, occ, cached, scor, frozen, _slotmap = ar
        ll[0] += 100.0                      # the live array moves...
        L = g._vert_all_rung_L(carrier, cached, base, ll,
                               np.array([0]), np.array([0]), np,
                               frozen=frozen)
        self.assertAlmostEqual(float(L[0]), -9.0)   # ...the price does not

    def test_NEGATIVE_CONTROL_an_unfrozen_carrier_follows_the_live_value(self):
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        cols, carrier, occ, cached, scor, frozen, _slotmap = ar
        ll[0] += 100.0
        L = g._vert_all_rung_L(carrier, cached, base, ll,
                               np.array([0]), np.array([0]), np,
                               frozen=frozen)
        self.assertAlmostEqual(float(L[0]), 91.0)

    def test_the_freeze_value_equals_the_live_value_at_freeze_time(self):
        """Identity: freezing must not shift the price."""
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        cols, carrier, occ, cached, scor, frozen, _slotmap = ar
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
        cols, carrier, occ, cached, scor, frozen, ar_slot = self._ar()
        ci = np.array([0]); t_c = np.array([0]); t_h = np.array([1])
        cached[0, 0], cached[0, 1] = -9.0, -3.0
        frozen[0, 0] = True
        _s0, _s1 = int(ar_slot[0, 0]), int(ar_slot[0, 1])
        for arr in (carrier, cached, occ, scor, frozen, ar_slot):
            _h = arr[ci, t_h].copy()
            arr[ci, t_h] = arr[ci, t_c]
            arr[ci, t_c] = _h
        self.assertTrue(bool(frozen[0, 1]), "the flag stayed behind")
        self.assertFalse(bool(frozen[0, 0]))
        self.assertAlmostEqual(float(cached[0, 1]), -9.0)
        # ar_slot rides with them: the cell now CALLED rung t_c is the
        # one whose sources sit in the slot that was rung t_h's. If this
        # stays behind, the two-sided re-point relabels the wrong slot.
        self.assertEqual(int(ar_slot[0, 0]), _s1)
        self.assertEqual(int(ar_slot[0, 1]), _s0)

    def test_the_frozen_flag_does_NOT_clear_the_carrier(self):
        """t_i / beta must keep following an accepted swap for a frozen
        row: the cold-lnL observe picks cold slots with ``t_i == 0`` and
        ``_converge_gate_mask`` reads t_i at the poll and in absorb."""
        ar = self._ar(); t_i, w_i, b_i, base, ll = self._rows()
        g._ar_freeze_rows(ar, np.array([0]), t_i, w_i, b_i, base, ll,
                          self.NB, np)
        self.assertEqual(int(ar[1][0, 0]), 0,
                         "carrier cleared; t_i would go stale")

    def test_the_state_tuple_carries_seven_entries(self):
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertIn("_ar_state = (_cols, _carrier, _occ, _cached, _scor,",
                      src)
        self.assertIn("_ar_frozen, _ar_slot)", src)
        sweep = inspect.getsource(
            g.GBSpecialBase._vertical_swap_sweep_all_rungs)
        self.assertIn(
            "cols, carrier, occupied, cached, scorable, frozen, ar_slot = ar",
            sweep)


class TwoSidedCellLabelRepointTest(unittest.TestCase):
    """The credited-ledger bug: both sides of a swap must be relabelled.

    USER RULING 2026-09-28 (item 1). ``_cell_ll_finalize`` credits
    ``ll_change_log[spec_of_slot[slot]]``. Before this fix only CARRIER
    slots were re-pointed, so an accepted swap with a RESIDENT
    non-carrier partner left two slots claiming ONE label: that label
    was credited twice and the other never. Measured cost on 6mo v9,
    ``[GB_ORTHO_LL]`` rj_warm_search median against a 0.05 tolerance:
    0.020 feature-off (job 643) -> 29.7 (654) -> 437.8 (655), with
    ``vgb_pe`` flat at 4e-8 as the unaffected control.

    THE MODEL used throughout. Two rungs of one column, labels X and Y.
    Before the swap X's sources sit in slot 0 and Y's in slot 1, so
    ``spec == [X, Y]``. ``exchange_cell_labels_batch`` trades the
    labels, so afterwards the cell CALLED X is the one in slot 1 and the
    cell called Y is the one in slot 0 -- i.e. the correct end state is
    ``spec == [Y, X]``. ``carrier`` and ``ar_slot`` are already permuted
    by the time the re-point runs, which is why each rung is described
    here by its POST-swap slot.
    """

    X, Y = 1000, 2000
    SLOTS = np.array([0, 1], dtype=np.int64)      # row -> slot, identity

    def _repoint(self, *rungs, use_map=True):
        """``rungs`` are (carrier_row, post_swap_slot, label) triples."""
        spec = np.array([self.X, self.Y], dtype=np.int64)
        for row, slot, lab in rungs:
            g._ar_repoint_cell_ll(
                spec, np.array([row]),
                np.array([slot]) if use_map else None,
                self.SLOTS, np.array([lab]), np)
        return spec

    @staticmethod
    def _credits(spec):
        """A fake ``_cell_ll_finalize``: one credit per slot, at the
        label that slot claims."""
        out = {}
        for slot in range(len(spec)):
            out[int(spec[slot])] = out.get(int(spec[slot]), 0) + 1
        return out

    def test_a_resident_NON_CARRIER_partner_is_relabelled(self):
        # X kept a carrier (row 1, now in slot 1); Y has no carrier but
        # its cell is resident in slot 0.
        spec = self._repoint((1, 1, self.X), (-1, 0, self.Y))
        self.assertEqual([int(v) for v in spec], [self.Y, self.X])

    def test_each_label_is_credited_EXACTLY_once(self):
        spec = self._repoint((1, 1, self.X), (-1, 0, self.Y))
        self.assertEqual(self._credits(spec), {self.X: 1, self.Y: 1})

    def test_THE_BUG_one_label_credited_twice_and_one_never(self):
        """The failing control: the carrier-only path, reproduced by
        withholding the rung -> slot map."""
        spec = self._repoint((1, 1, self.X), (-1, 0, self.Y),
                             use_map=False)
        self.assertEqual(self._credits(spec), {self.X: 2},
                         "the old bug credits the carrier's label twice")
        self.assertNotIn(self.Y, self._credits(spec))

    def test_a_TRULY_non_resident_partner_stays_one_sided(self):
        """No bracket exists to relabel; skipping is correct, not a gap.
        This is also the inference path, where ar_slot is all -1."""
        spec = self._repoint((1, 1, self.X), (-1, -1, self.Y))
        self.assertEqual(int(spec[1]), self.X)
        self.assertEqual(int(spec[0]), self.X,
                         "slot 0 has no bracket to fix here")

    def test_two_carriers_are_unaffected_by_the_change(self):
        """The in_model case: every resident cell has a carrier, so both
        sides were re-pointed even before the fix. Must not change."""
        with_map = self._repoint((1, 1, self.X), (0, 0, self.Y))
        without = self._repoint((1, 1, self.X), (0, 0, self.Y),
                                use_map=False)
        self.assertEqual([int(v) for v in with_map], [self.Y, self.X])
        np.testing.assert_array_equal(with_map, without)

    def test_the_carrier_WINS_when_a_rung_has_both(self):
        """slots[row] is live per-row bookkeeping; the all-rung map is
        built once per block and can go stale across a rebind."""
        spec = np.array([self.X, self.Y], dtype=np.int64)
        g._ar_repoint_cell_ll(spec, np.array([0]), np.array([1]),
                              self.SLOTS, np.array([self.X]), np)
        self.assertEqual(int(spec[0]), self.X)
        self.assertEqual(int(spec[1]), self.Y, "the stale map was used")

    def test_the_sweep_calls_the_helper_for_BOTH_rungs(self):
        import inspect
        src = inspect.getsource(
            g.GBSpecialBase._vertical_swap_sweep_all_rungs)
        blk = src[src.index("TWO-SIDED re-point"):]
        self.assertIn("_ar_repoint_cell_ll(", blk)
        self.assertIn("for _t_new in (t_h, t_c):", blk)

    def test_ar_slot_is_permuted_in_the_sweep(self):
        import inspect
        src = inspect.getsource(
            g.GBSpecialBase._vertical_swap_sweep_all_rungs)
        self.assertIn("ar_slot[ci, t_h] = ar_slot[ci, t_c]", src)


class SwapAndRefitInTheSameRepeatTest(unittest.TestCase):
    """``GB_TEMPER_VERTICAL_AT_REFIT`` -- item 2 of the 2026-09-28 ruling.

    USER RULING, verbatim: "i just want the swap and refit to happen the
    same iteration I guess", with the order WITHIN the repeat left as it
    already is -- "sample -> refit -> swap". So this is a GATE on which
    repeats sweep, not a reordering: no hoist of the refresh out of the
    parity-half loop, because the refresh already runs at the end of the
    last half and the sweep already runs after it.

    (An earlier relayed spec asked for swap -> refit -> sample plus an
    opening sweep. The user superseded it directly; there is no opening
    sweep and nothing was hoisted.)
    """

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("GB_TEMPER_VERTICAL_AT_REFIT", None)
        self.addCleanup(self.env.stop)

    def test_the_knob_is_OFF_by_default(self):
        self.assertFalse(g._vert_at_refit_on())

    def test_the_knob_reads_the_usual_truthy_spellings(self):
        for v in ("1", "true", "True", "yes", "on"):
            os.environ["GB_TEMPER_VERTICAL_AT_REFIT"] = v
            self.assertTrue(g._vert_at_refit_on(), v)
        for v in ("0", "no", "off", ""):
            os.environ["GB_TEMPER_VERTICAL_AT_REFIT"] = v
            self.assertFalse(g._vert_at_refit_on(), v)

    # -- the gate expression itself, as production evaluates it --------
    @staticmethod
    def _gate(vert_on, at_refit, sighet_active, refit_this_repeat):
        return vert_on and not (
            at_refit and sighet_active and not refit_this_repeat)

    def test_with_the_knob_ON_only_refit_repeats_sweep(self):
        self.assertTrue(self._gate(True, True, True, True))
        self.assertFalse(self._gate(True, True, True, False))

    def test_with_the_knob_OFF_every_repeat_sweeps(self):
        """Bit-identical to today on both kinds of repeat."""
        self.assertTrue(self._gate(True, False, True, True))
        self.assertTrue(self._gate(True, False, True, False))

    def test_an_EXACT_reference_engine_is_never_gated(self):
        """chunked-het / FD never refresh, so a refit-gated sweep would
        disable vertical swapping outright. The gate ANDs on
        ``sighet_active`` precisely to avoid that."""
        self.assertTrue(self._gate(True, True, False, False))
        self.assertTrue(self._gate(True, True, False, True))

    def test_the_gate_cannot_switch_vertical_swapping_ON(self):
        self.assertFalse(self._gate(False, True, True, True))
        self.assertFalse(self._gate(False, False, True, True))

    # -- cadence over a block ------------------------------------------
    def test_exactly_one_sweep_per_refresh_tick_over_100_repeats(self):
        """every=25, n_rep=100: the refresh condition is
        ``(move_i+1) % every == 0 and move_i+1 < n_rep``, so ticks are
        repeats 24, 49, 74 -- repeat 99 is excluded by ``< n_rep``."""
        every, n_rep = 25, 100
        ticks = [i for i in range(n_rep)
                 if (i + 1) % every == 0 and i + 1 < n_rep]
        self.assertEqual(ticks, [24, 49, 74])
        swept = [i for i in range(n_rep)
                 if self._gate(True, True, True, i in ticks)]
        self.assertEqual(swept, ticks)
        self.assertEqual(len(swept), 3)

    def test_the_LAST_repeat_never_sweeps_and_that_is_documented(self):
        """``move_i + 1 < n_rep`` excludes it; the next block's own tick
        covers the gap. Documented so it is not read as a bug."""
        doc = g._vert_at_refit_on.__doc__
        self.assertIn("LAST", doc)
        self.assertIn("move_i + 1 < n_rep", doc)

    # -- wiring --------------------------------------------------------
    def test_production_gates_the_sweep_on_the_refit_flag(self):
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertIn("_refit_this_repeat = False", src)
        self.assertIn("_refit_this_repeat = True", src)
        self.assertIn("_vert_at_refit and sighet_active "
                      "and not _refit_this_repeat", src)
        self.assertIn("if _vert_now:", src)

    def test_the_flag_is_set_BEFORE_the_far_any_test(self):
        """A tick on which nothing had drifted far enough is still a
        tick; gating the swap on ``far.any()`` would make the swap
        cadence depend on drift."""
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertLess(src.index("_refit_this_repeat = True"),
                        src.index("if bool(far.any()):"))

    def test_the_sweep_count_is_printed(self):
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        self.assertIn("_vert_sweeps = 0", src)
        self.assertIn("_vert_sweeps += 1", src)
        self.assertIn("sweep(s)", src)

    def test_BOTH_launchers_carry_the_new_cadence_and_knob(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        for name in ("submit_gf_6mo_v9_4gpu.sh", "submit_gf_3mo_v9_2gpu.sh"):
            src = (root / "scripts" / "fstat_proposal" / name).read_text()
            self.assertIn("export GB_SIGHET_REFRESH_EVERY=25", src, name)
            self.assertNotIn("export GB_SIGHET_REFRESH_EVERY=50", src, name)
            self.assertIn("export GB_TEMPER_VERTICAL_AT_REFIT=1", src, name)


class CreditedEqualsDirectAcrossASwapTest(unittest.TestCase):
    """FINDING A (2026-09-28): the bracket must TRADE, not just re-point.

    ``_cell_ll_finalize`` credits ``ll_change_log[spec] = led0 + (lls -
    ll0)``. The picked-row sweep has always traded all four bracket
    fields between the two slots and sat at 0.02 on job 643. The
    all-rungs sweep re-pointed ``spec`` only and left ll0/led0/rep0 with
    the slot, which drops the swap JUMP from the credited value -- so
    [GB_ORTHO_LL] stayed at 438 / 340 on jobs 655 / 659 even after the
    two-sided re-point landed and accepted swaps fell 26x, while moves
    running no sweep at all measured ~0.005-0.009.
    """

    #: slot A holds cell A's sources, slot B holds cell B's.
    A, B = 0, 1
    X, Y = 1000, 2000          # labels: A opens as X, B opens as Y

    def _state(self):
        return {
            "spec": np.array([self.X, self.Y], dtype=np.int64),
            "ll0": np.array([-100.0, -300.0]),    # opening slab lls
            "led0": np.array([-10.0, -30.0]),     # opening ledgers
            "rep0": np.array([0, 0], dtype=np.int64),
        }

    @staticmethod
    def _finalize(st, lls):
        """The real credit rule, as _cell_ll_finalize applies it."""
        return {int(st["spec"][s]): st["led0"][s] + (lls[s] - st["ll0"][s])
                for s in range(len(lls))}

    # current slab lls: each cell moved a little since open
    LLS = np.array([-97.0, -295.0])       # A moved +3, B moved +5

    def _direct(self):
        """What ORTHO's `direct` measures after the labels swap: label Y
        now denotes cell A, so its value relative to ITS OWN opening
        baseline is led0_B + (lls_A - ll0_B), and symmetrically for X."""
        st = self._state()
        return {
            self.Y: st["led0"][self.B] + (self.LLS[self.A]
                                          - st["ll0"][self.B]),
            self.X: st["led0"][self.A] + (self.LLS[self.B]
                                          - st["ll0"][self.A]),
        }

    def test_the_FULL_TRADE_reproduces_direct(self):
        st = self._state()
        g._ar_trade_cell_ll(st, np.array([self.A]), np.array([self.B]), np)
        got = self._finalize(st, self.LLS)
        for lab, want in self._direct().items():
            self.assertAlmostEqual(got[lab], want, places=9,
                                   msg=f"label {lab}")

    def test_SPEC_ONLY_is_short_by_exactly_the_swap_jump(self):
        """The failing control: today's behaviour, and the size of the
        error it leaves -- one slab jump, not a rounding difference."""
        st = self._state()
        st["spec"][self.A], st["spec"][self.B] = self.Y, self.X
        got = self._finalize(st, self.LLS)
        direct = self._direct()
        jump = st["ll0"][self.A] - st["ll0"][self.B]     # 200.0
        self.assertNotAlmostEqual(got[self.Y], direct[self.Y], places=3)
        # short by the jump plus the ledger baseline difference
        # direct - got = (led0_B - led0_A) + (ll0_A - ll0_B)
        #              = ledger baseline difference + the swap JUMP
        self.assertAlmostEqual(
            direct[self.Y] - got[self.Y],
            (st["led0"][self.B] - st["led0"][self.A]) + jump, places=9)
        self.assertGreater(abs(direct[self.Y] - got[self.Y]), 100.0,
                           "the error should be slab-sized")

    def test_each_label_is_still_credited_exactly_once(self):
        """The item-1 control must survive the item-A change."""
        st = self._state()
        g._ar_trade_cell_ll(st, np.array([self.A]), np.array([self.B]), np)
        self.assertEqual(sorted(int(v) for v in st["spec"]),
                         sorted((self.X, self.Y)))

    def test_rep0_travels_with_the_bracket(self):
        st = self._state()
        st["rep0"][:] = [7, 9]
        g._ar_trade_cell_ll(st, np.array([self.A]), np.array([self.B]), np)
        self.assertEqual([int(v) for v in st["rep0"]], [9, 7])

    def test_a_missing_field_is_skipped_not_fatal(self):
        st = self._state(); st.pop("rep0")
        g._ar_trade_cell_ll(st, np.array([self.A]), np.array([self.B]), np)
        self.assertEqual(int(st["spec"][self.A]), self.Y)

    def test_the_field_list_matches_the_picked_row_sweep(self):
        """One list, so the two sweeps cannot drift apart again."""
        import inspect
        src = inspect.getsource(g.GBSpecialBase._vertical_swap_sweep)
        self.assertIn('for key in ("spec", "ll0", "led0", "rep0"):', src)
        self.assertEqual(g._CELL_LL_BRACKET_FIELDS,
                         ("spec", "ll0", "led0", "rep0"))

    def test_the_all_rungs_sweep_TRADES_when_both_slots_exist(self):
        import inspect
        src = inspect.getsource(
            g.GBSpecialBase._vertical_swap_sweep_all_rungs)
        self.assertIn("_ar_trade_cell_ll(st,", src)
        self.assertIn("_both = (_s_c >= 0) & (_s_h >= 0)", src)

    def test_the_one_sided_remainder_is_COUNTED(self):
        """A truly non-resident partner has no bracket to trade with, so
        that credit is still short by the jump. Documented and counted,
        not silent."""
        import inspect
        src = inspect.getsource(
            g.GBSpecialBase._vertical_swap_sweep_all_rungs)
        self.assertIn('census["one_sided_bracket"]', src)


class OpeningSweepTest(unittest.TestCase):
    """FINDING B (2026-09-28): a 25-repeat block never reaches a tick.

    ``in_model`` / ``in_model_fstat`` run n_rep = 25 with every = 25, and
    the tick carries ``move_i + 1 < n_rep``, so the only candidate is the
    final repeat and it is excluded. Job 659: all 320 in_model census
    blocks printed "(0 sweep(s))", 0 proposed -- and those two moves are
    ~2,800 s of the iteration.
    """

    @staticmethod
    def _ticks(n_rep, every):
        return [i for i in range(n_rep)
                if (i + 1) % every == 0 and i + 1 < n_rep]

    def test_a_25_repeat_block_has_NO_tick(self):
        self.assertEqual(self._ticks(25, 25), [])

    def test_which_is_why_the_opening_sweep_exists(self):
        """open + ticks: a 25-repeat block sweeps exactly once."""
        self.assertEqual(1 + len(self._ticks(25, 25)), 1)

    def test_a_255_repeat_block_sweeps_open_plus_ten(self):
        self.assertEqual(1 + len(self._ticks(255, 25)), 11)

    def test_the_opening_sweep_is_wired_and_gated(self):
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        blk = src[src.index("OPENING SWEEP"):]
        self.assertIn(
            "if _vert_on and _vert_at_refit and sighet_active and _half_pre:",
            blk)
        self.assertLess(src.index("OPENING SWEEP"),
                        src.index("for move_i in range(n_rep"))

    def test_it_does_NOT_fire_with_the_knob_off(self):
        """Knob off must stay bit-identical to before item 2."""
        import inspect
        src = inspect.getsource(g.GBSpecialBase._run_in_model_repeats)
        blk = src[src.index("OPENING SWEEP"):]
        self.assertIn("_vert_at_refit", blk.split("\n            ")[0]
                      + blk[:800])
