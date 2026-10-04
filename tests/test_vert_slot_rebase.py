"""Block-end slot-map re-base after ALL-RUNGS vertical swaps (2026-10-04).

THE BUG. ``_run_in_model_repeats`` re-based the buffer's special -> slot map
and the scheduler's slot labels from the CARRIER rows only. The all-rungs
sweep also swaps a carrier with a RESIDENT NON-CARRIER cell; the sorter
re-labels both cells but the non-carrier's slot kept its old label, so two
active slots claimed one cell and the next RJ round resolved picks into the
wrong slab (hot births scored on, and written into, a cold cell's slab).
6mo replica_pe job 715: per-propose ledger-vs-residual drift -1,360..-1,745
per walker on prior-RJ iterations; job 706 rj_prior_removal with births
-1,945 vs deaths-only (alive-only subsets, no non-carriers) -0..-58.

Three layers, each on REAL code where it matters:

* ``_ar_rebase_slot_labels`` on the all-rung state after a carrier <->
  non-carrier swap (and its refusal when table and rows disagree);
* the REAL ``BandScheduler.relabel_slots``: carriers-only leaves a duplicate
  active label (the bug, as a negative control); the full set is a clean
  permutation of labels and budgets;
* the REAL ``_run_in_model_repeats`` with ``GB_TEMPER_ALL_RUNGS=1`` and a
  REAL scheduler: after a forced carrier <-> non-carrier swap every active
  slot's label equals the label of the sources sitting in that slot.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as g
from lisatools.globalfit.moves.gbbands import BandScheduler, pack_special_index

NB = 3          # bands
NW = 2          # walkers
NT = 4          # rungs


def _spec(t, w, b):
    return pack_special_index(np.asarray(t), np.asarray(w), np.asarray(b), NW)


def _ar_one_column(carrier_rungs, nonc_rungs, w=0, b=1):
    """One column; carriers at ``carrier_rungs`` (row k on rung
    carrier_rungs[k], slot = rung), resident non-carriers at ``nonc_rungs``
    (slot = rung)."""
    cols = np.array([w * NB + b], dtype=np.int64)
    carrier = np.full((1, NT), -1, dtype=np.int64)
    for k, t in enumerate(carrier_rungs):
        carrier[0, t] = k
    ar_slot = np.full((1, NT), -1, dtype=np.int64)
    for t in nonc_rungs:
        ar_slot[0, t] = t
    occ = np.ones((1, NT), bool)
    z = np.zeros((1, NT))
    return [cols, carrier, occ, z.copy(), np.ones((1, NT), bool),
            np.zeros((1, NT), bool), ar_slot]


def _swap(ar, t_c, t_h):
    """The all-rungs sweep's permutation of the table for one accepted pair."""
    for arr in (ar[1], ar[3], ar[2], ar[4], ar[5], ar[6]):
        h = arr[0, t_h].copy()
        arr[0, t_h] = arr[0, t_c]
        arr[0, t_c] = h


class RebaseHelperTest(unittest.TestCase):
    def test_a_resident_non_carrier_slot_follows_the_swap(self):
        # carriers on rungs 1, 2, 3 (rows 0, 1, 2 in slots 1, 2, 3); the
        # cold rung 0 is a resident NON-CARRIER in slot 0
        ar = _ar_one_column([1, 2, 3], [0])
        slots = np.array([1, 2, 3])
        t_i = np.array([1, 2, 3])
        _swap(ar, 0, 1)                     # cold non-carrier <-> rung-1 carrier
        t_i[0] = 0                          # the carrier row now sits on rung 0
        w_i = np.zeros(3, int); b_i = np.ones(3, int)
        spec_final = _spec(t_i, w_i, b_i)
        s, sp = g._ar_rebase_slot_labels(
            ar, slots, spec_final, _spec, NB, np)
        got = dict(zip(s.tolist(), sp.tolist()))
        # slot 1 (carrier) is now cold; slot 0 (non-carrier) is now rung 1
        self.assertEqual(got[1], int(_spec(0, 0, 1)))
        self.assertEqual(got[0], int(_spec(1, 0, 1)))
        self.assertEqual(got[2], int(_spec(2, 0, 1)))
        self.assertEqual(got[3], int(_spec(3, 0, 1)))
        self.assertEqual(len(set(got.values())), 4, "labels must be unique")

    def test_NEGATIVE_CONTROL_carriers_only_leaves_a_duplicate(self):
        """What the 2026-09-10 re-base produced: the carrier slot takes the
        cold label, the non-carrier slot keeps it too."""
        ar = _ar_one_column([1, 2, 3], [0])
        t_i = np.array([1, 2, 3])
        _swap(ar, 0, 1)
        t_i[0] = 0
        labels = {0: int(_spec(0, 0, 1))}                    # stale slot 0
        labels.update(zip([1, 2, 3], _spec(t_i, np.zeros(3, int),
                                           np.ones(3, int)).tolist()))
        self.assertEqual(labels[0], labels[1], "the bug: two slots, one label")

    def test_no_swap_is_a_noop(self):
        ar = _ar_one_column([1, 2], [0, 3])
        slots = np.array([1, 2]); t_i = np.array([1, 2])
        s, sp = g._ar_rebase_slot_labels(
            ar, slots, _spec(t_i, [0, 0], [1, 1]), _spec, NB, np)
        self.assertEqual(dict(zip(s.tolist(), sp.tolist())),
                         {t: int(_spec(t, 0, 1)) for t in range(NT)})

    def test_disagreement_between_table_and_rows_is_refused(self):
        ar = _ar_one_column([1, 2, 3], [0])
        slots = np.array([1, 2, 3])
        bad = _spec(np.array([3, 2, 1]), np.zeros(3, int), np.ones(3, int))
        with self.assertRaises(RuntimeError):
            g._ar_rebase_slot_labels(ar, slots, bad, _spec, NB, np)


class RealSchedulerRelabelTest(unittest.TestCase):
    def _sched(self):
        sp = _spec(np.arange(NT), np.zeros(NT, int), np.ones(NT, int))
        s = BandScheduler(sp, 16, xp=np, cell_order="band", nwalkers=NW)
        s.cell_run = np.array([5, 6, 7, 8])         # distinguishable budgets
        s.cell_counts = np.array([15, 16, 17, 18])
        return s

    def test_NEGATIVE_CONTROL_carriers_only_relabel_duplicates_a_label(self):
        s = self._sched()
        s.relabel_slots(np.array([1]), np.array([_spec(0, 0, 1)]))
        act = s.active_slot_specials
        self.assertLess(len(np.unique(act)), len(act))
        with self.assertRaises(AssertionError):
            g._assert_active_slot_labels_unique(s, "test", np)

    def test_the_full_set_is_a_clean_permutation(self):
        s = self._sched()
        s.relabel_slots(np.array([0, 1]),
                        np.array([_spec(1, 0, 1), _spec(0, 0, 1)]))
        g._assert_active_slot_labels_unique(s, "test", np)
        np.testing.assert_array_equal(
            s.slot_specials,
            [_spec(1, 0, 1), _spec(0, 0, 1), _spec(2, 0, 1), _spec(3, 0, 1)])
        # budgets follow the MODEL: the cell now labelled rung 0 is the one
        # that was rung 1, so label 0 carries rung 1's counters
        np.testing.assert_array_equal(s.cell_run, [6, 5, 7, 8])
        np.testing.assert_array_equal(s.cell_counts, [16, 15, 17, 18])


# --------------------------------------------------------------------------
# the REAL _run_in_model_repeats, all-rungs on, REAL scheduler
# --------------------------------------------------------------------------
def _harness():
    try:
        from tests import test_inmodel_repeats as h
        from tests import test_vertical_swap as v
    except ImportError:  # discovered from inside tests/
        import test_inmodel_repeats as h
        import test_vertical_swap as v
    return h, v


class BlockEndRebaseIntegrationTest(unittest.TestCase):
    """One (walker, band) column with 4 rungs. Rows 1, 2, 3 are picked
    (carriers in slots 1, 2, 3); row 0 -- the COLD cell -- is resident in
    slot 0 but not picked (its pick this round was a rejected birth). Its
    measured slab total is made very poor, so the rung-0 <-> rung-1 pair is
    accepted with certainty (paccept >= 0) on the first even-parity sweep."""

    def _run(self, n_rep=2):
        h, v = _harness()
        n_src = 4
        t = np.arange(NT)
        w = np.zeros(NT, dtype=int)
        b = np.ones(NT, dtype=int)
        rng = np.random.RandomState(3)
        coords = np.zeros((n_src, 4))
        coords[:, 0] = rng.uniform(-0.5, 0.5, n_src)
        coords[:, 1] = rng.uniform(2.95, 3.05, n_src)
        coords[:, 2] = rng.uniform(-1, 1, n_src)
        coords[:, 3] = rng.uniform(-1, 1, n_src)
        sorter = v._FakeSorter(t, w, b, NW)
        sorter.inds = np.ones(n_src, dtype=bool)
        sorter.coords = coords.copy()
        sorter.leaf_inds = np.arange(n_src)

        picked_rows = np.array([1, 2, 3])
        picked = {
            "ids": picked_rows,
            "specials": _spec(t[picked_rows], w[picked_rows], b[picked_rows]),
            "slot_index": picked_rows.astype(np.int32),   # slot == rung
            "temp_inds": t[picked_rows].copy(),
            "walker_inds": w[picked_rows].copy(),
            "band_inds": b[picked_rows].copy(),
            "N_vals": np.full(3, 64),
        }
        sched = BandScheduler(sorter.special_band_inds.copy(), 16, xp=np,
                              cell_order="band", nwalkers=NW)

        class _Buf(h._FakeBuffer):
            def __init__(self, n):
                super().__init__(n)
                self.updates = []

            def band_likelihoods(self, source_only=False, slots=None):
                vals = np.zeros(NT)
                vals[0] = -1.0e6            # the cold non-carrier is terrible
                return vals if slots is None else vals[np.asarray(slots)]

            def update_special_indices(self, new, inds_fill=None):
                self.updates.append((np.asarray(inds_fill).copy(),
                                     np.asarray(new).copy()))

        buf = _Buf(NT)
        mv = h._make_move(n_rep)
        mv.ntemps, mv.nwalkers, mv.num_bands = NT, NW, NB
        mv.temper_vertical = True
        mv._temper_rng = np.random.default_rng(5)
        mv.sequential_parity_repeats = False
        ll_change = np.zeros((NT, NW, NB))
        prop = np.zeros((2, NT, NW, NB), dtype=int)
        acc = np.zeros_like(prop)
        env = {"GB_TEMPER_ALL_RUNGS": "1", "GB_TEMPER_VERTICAL_AT_REFIT": "0",
               "GB_VERT_EXACT_PRICE": "0"}
        np.random.seed(7)
        with mock.patch.dict(os.environ, env):
            mv._run_in_model_repeats(
                None, sorter, buf, v._ladder(), picked, ll_change, prop, acc,
                num_repeats=n_rep, scheduler=sched,
            )
        return sorter, sched, buf

    def test_every_active_slot_claims_the_label_of_the_sources_it_holds(self):
        sorter, sched, buf = self._run()
        # the forced swap happened: source 0 (slot 0) is no longer cold
        self.assertNotEqual(int(sorter.temp_inds[0]), 0, "no swap was accepted")
        slot_of_row = np.arange(4)                    # rows never move slots
        for r in range(4):
            self.assertEqual(
                int(sched.slot_specials[slot_of_row[r]]),
                int(sorter.special_band_inds[r]),
                f"slot {slot_of_row[r]} claims a label its sources do not carry")
        g._assert_active_slot_labels_unique(sched, "test", np)

    def test_the_buffer_map_is_re_based_for_the_non_carrier_slot_too(self):
        sorter, sched, buf = self._run()
        self.assertTrue(buf.updates, "no buffer re-base happened")
        filled = np.concatenate([u[0] for u in buf.updates])
        self.assertIn(0, filled.tolist(), "the resident non-carrier slot "
                      "was not re-based")
        last = {}
        for slots_u, specs_u in buf.updates:
            last.update(zip(slots_u.tolist(), specs_u.tolist()))
        for r in range(4):
            self.assertEqual(last.get(r, int(_spec(r, 0, 1))),
                             int(sorter.special_band_inds[r]))


if __name__ == "__main__":
    unittest.main()
