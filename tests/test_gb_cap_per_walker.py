"""PER-WALKER GB leaf caps (user request 2026-09-22).

The progressive leaf cap is stored per cap CELL and the gate collapses the
walker axis the moment it runs (``cur_max = lls.max(axis=0)``), so one
walker's lnL improvement and its occupancy-at-cap ramp the allowance for
every walker -- including laggards that never filled the cap they already
had. ``GB_LEAF_CAP_PER_WALKER=1`` gives every walker its own per-cell cap,
earned from its own evidence and enforced against its own rows.

THE HARD CONSTRAINT these tests exist to protect: with the flag OFF nothing
is allocated, nothing is written and no branch is taken, so the live 6-month
run can be relaunched from a build carrying this change.

Same light-fake style as ``test_gb_cap_cell_grid``: the cap machinery is
index arithmetic over ``band_sorter``-shaped arrays, so none of it needs a
built move, an ACA or a backend.
"""

import os
import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import tempering_swap_cap_ok
from lisatools.globalfit.state import (
    GBState,
    ensure_cap_cell_fields,
    ensure_leaf_cap_fields,
    make_cap_edges,
)

from tests.test_gb_cap_cell_grid import (
    BAND_EDGES,
    NUM_BANDS,
    _move,
    _sorter,
)

PW_ENV = "GB_LEAF_CAP_PER_WALKER"


def _pw_move(cap_divisor=2, nwalkers=3, ntemps=1, per_walker=True,
             min_iters=3):
    """A fake move with the cap attributes + the per-walker switch."""
    m = _move(cap_divisor, ntemps=ntemps, nwalkers=nwalkers)
    m.branch_name = "gb"
    m.leaf_cap_per_walker = bool(per_walker)
    m._leaf_cap_enabled = True
    m.leaf_cap_ll_improve = True
    m.leaf_cap_iter_only = False
    m.leaf_cap_ll_nsigma = 3.0
    m.leaf_cap_require_occupancy = False
    m.leaf_cap_min_iters = min_iters
    m.leaf_cap_ndim = 8  # hold threshold D/2 = 4.0
    m.cap_overlap_frac = 0.0
    m._work_branch = lambda ns: SimpleNamespace(shape=(ntemps, nwalkers, 10))
    m._band_residual_lls = lambda acs: np.zeros((nwalkers, NUM_BANDS))
    return m


def _pw_state(m, nwalkers=None, start_cap=1):
    """band_info carrying the per-walker cap family, armed at ``start_cap``."""
    nwalkers = int(m.nwalkers if nwalkers is None else nwalkers)
    bi = {"num_bands": m.num_bands, "nwalkers": nwalkers,
          "num_cap_cells": m.num_cap_cells}
    ensure_leaf_cap_fields(bi, m.num_bands)
    ensure_cap_cell_fields(bi, m.num_cap_cells,
                           staggered=m.cap_stagger,
                           per_walker=m.leaf_cap_per_walker)
    cap, iters, best = m._cap_state_arrays(bi)
    cap[:] = start_cap
    iters[:] = 0
    best[:] = -np.inf
    state = SimpleNamespace(sub_states={"gb": SimpleNamespace(band_info=bi)})
    return state, bi


def _pw_step(m, state, stats, occ):
    """One ``_update_band_leaf_caps`` call with scripted per-walker stats."""
    stats = np.asarray(stats, dtype=float)
    occ = np.asarray(occ, dtype=int)
    m._cap_cell_lls = lambda model, ns, band_lls: (
        stats, np.zeros(m.num_cap_cells)
    )
    m._cold_occupancy = lambda bc, ns: occ
    m._update_band_leaf_caps(
        SimpleNamespace(analysis_container_arr=None), state, None
    )


class PerWalkerAllocationTest(unittest.TestCase):
    """The storage layer, and the flag-off guarantee."""

    def test_flag_off_allocates_no_per_walker_array(self):
        """THE COMPAT GATE: a 6mo relaunch must see exactly today's keys."""
        bi = {"num_bands": NUM_BANDS, "nwalkers": 4}
        ensure_cap_cell_fields(bi, NUM_BANDS * 4, staggered=False)
        stray = [k for k in bi if k.endswith("_w")]
        self.assertEqual(
            stray, [],
            f"per-walker arrays leaked into a flag-off band_info: {stray}")

    def test_flag_off_state_arrays_are_the_shared_triple(self):
        m = _pw_move(cap_divisor=2, per_walker=False)
        _, bi = _pw_state(m)
        cap, iters, best = m._cap_state_arrays(bi)
        self.assertIs(cap, bi["cap_cell_leaf_cap"])
        self.assertIs(iters, bi["cap_cell_iters"])
        self.assertIs(best, bi["cap_cell_best_ll"])
        self.assertEqual(cap.ndim, 1)

    def test_per_walker_allocation_shapes_and_fills(self):
        bi = {"num_bands": NUM_BANDS, "nwalkers": 4}
        ncells = NUM_BANDS * 2
        ensure_cap_cell_fields(bi, ncells, staggered=False, per_walker=True)
        self.assertEqual(bi["cap_cell_leaf_cap_w"].shape, (4, ncells))
        self.assertEqual(bi["cap_cell_iters_w"].shape, (4, ncells))
        self.assertEqual(bi["cap_cell_best_ll_w"].shape, (4, ncells))
        self.assertTrue(np.all(bi["cap_cell_leaf_cap_w"] == -1))
        self.assertTrue(np.all(bi["cap_cell_iters_w"] == 0))
        self.assertTrue(np.all(np.isneginf(bi["cap_cell_best_ll_w"])))

    def test_per_walker_allocated_even_on_the_band_grid(self):
        """divisor 1 without stagger still needs the walker axis.

        The divisor-1 short circuit exists because cells ARE bands there and
        the shared arrays serve both. That reasoning does not survive a
        walker axis -- ``band_leaf_cap`` has none.
        """
        bi = {"num_bands": NUM_BANDS, "nwalkers": 2}
        ensure_cap_cell_fields(bi, NUM_BANDS, staggered=False, per_walker=True)
        self.assertEqual(bi["cap_cell_leaf_cap_w"].shape, (2, NUM_BANDS))

    def test_enabling_on_an_existing_store_seeds_from_the_shared_cap(self):
        """No migration script: every walker inherits the stored cap+clock."""
        bi = {"num_bands": NUM_BANDS, "nwalkers": 3}
        ncells = NUM_BANDS * 2
        ensure_cap_cell_fields(bi, ncells, staggered=True)  # the OLD store
        bi["cap_cell_leaf_cap"][:] = np.arange(ncells)
        bi["cap_cell_iters"][:] = 7
        bi["cap_cell_best_ll"][:] = -12.5
        ensure_cap_cell_fields(bi, ncells, staggered=True, per_walker=True)
        for w in range(3):
            np.testing.assert_array_equal(
                bi["cap_cell_leaf_cap_w"][w], np.arange(ncells))
            np.testing.assert_array_equal(bi["cap_cell_iters_w"][w], 7)
            np.testing.assert_allclose(bi["cap_cell_best_ll_w"][w], -12.5)

    def test_band_best_ll_per_walker_mirror_allocated(self):
        bi = {"num_bands": NUM_BANDS, "nwalkers": 5}
        ensure_leaf_cap_fields(bi, NUM_BANDS, per_walker=True)
        self.assertEqual(bi["band_best_ll_w"].shape, (5, NUM_BANDS))
        self.assertTrue(np.all(np.isneginf(bi["band_best_ll_w"])))

    def test_leaf_cap_fields_flag_off_has_no_per_walker_mirror(self):
        bi = {"num_bands": NUM_BANDS, "nwalkers": 5}
        ensure_leaf_cap_fields(bi, NUM_BANDS)
        self.assertNotIn("band_best_ll_w", bi)

    def test_state_arrays_return_the_per_walker_triple(self):
        m = _pw_move(cap_divisor=2, nwalkers=3)
        _, bi = _pw_state(m)
        cap, iters, best = m._cap_state_arrays(bi)
        self.assertIs(cap, bi["cap_cell_leaf_cap_w"])
        self.assertIs(iters, bi["cap_cell_iters_w"])
        self.assertIs(best, bi["cap_cell_best_ll_w"])
        self.assertEqual(cap.shape, (3, m.num_cap_cells))


class PerWalkerMirrorTest(unittest.TestCase):
    """The 1-D arrays stay live as max-over-walkers summaries."""

    def test_cell_mirrors_are_the_max_over_walkers(self):
        m = _pw_move(cap_divisor=2, nwalkers=3)
        _, bi = _pw_state(m)
        bi["cap_cell_leaf_cap_w"][:] = [[1, 5, 2, 1, 1, 1, 1, 1],
                                        [3, 2, 2, 1, 1, 1, 1, 1],
                                        [2, 2, 9, 1, 1, 1, 1, 1]]
        bi["cap_cell_iters_w"][:] = [[0, 4, 1, 0, 0, 0, 0, 0],
                                     [2, 1, 1, 0, 0, 0, 0, 0],
                                     [1, 1, 6, 0, 0, 0, 0, 0]]
        bi["cap_cell_best_ll_w"][:] = -np.arange(24.0).reshape(3, 8)
        m._mirror_band_leaf_cap(bi)
        np.testing.assert_array_equal(
            bi["cap_cell_leaf_cap"], [3, 5, 9, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(
            bi["cap_cell_iters"], [2, 4, 6, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(
            bi["cap_cell_best_ll"], -np.arange(8.0))

    def test_band_leaf_cap_is_the_max_over_walkers_and_cells(self):
        m = _pw_move(cap_divisor=2, nwalkers=2)
        _, bi = _pw_state(m)
        # 4 bands x 2 cells; walker 1 holds the band-2 maximum
        bi["cap_cell_leaf_cap_w"][:] = [[1, 1, 1, 1, 1, 4, 1, 1],
                                        [1, 1, 7, 1, 1, 1, 1, 1]]
        m._mirror_band_leaf_cap(bi)
        np.testing.assert_array_equal(bi["band_leaf_cap"], [1, 7, 4, 1])

    def test_band_grid_mirror_writes_band_leaf_cap(self):
        """divisor 1: band_leaf_cap is a MIRROR now, not the gate array."""
        m = _pw_move(cap_divisor=1, nwalkers=2)
        _, bi = _pw_state(m)
        bi["cap_cell_leaf_cap_w"][:] = [[1, 6, 1, 1], [3, 1, 1, 1]]
        m._mirror_band_leaf_cap(bi)
        np.testing.assert_array_equal(bi["band_leaf_cap"], [3, 6, 1, 1])


class PerWalkerGateTest(unittest.TestCase):
    """Each walker earns its own increment from its OWN series."""

    def setUp(self):
        self._old = os.environ.get("GB_LEAF_CAP_REQUIRE_IMPROVEMENT")
        os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = "1"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_LEAF_CAP_REQUIRE_IMPROVEMENT", None)
        else:
            os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = self._old

    CELL = 3

    def test_only_the_plateaued_walker_increments(self):
        """Walker 0 stagnates at cap; walker 1 keeps improving by > D/2."""
        m = _pw_move(cap_divisor=2, nwalkers=2, min_iters=3)
        state, bi = _pw_state(m, nwalkers=2, start_cap=1)
        n = m.num_cap_cells
        occ = np.zeros((2, n), dtype=int)
        occ[:, self.CELL] = 1  # both walkers at their cap of 1
        # iteration 0 engages both (occupied at first sight)
        for i in range(6):
            stats = np.full((2, n), -1000.0)
            stats[0, self.CELL] = -100.0           # flat: plateaued
            stats[1, self.CELL] = -100.0 + 10.0 * i  # +10 > D/2 each step
            _pw_step(m, state, stats, occ)
        cap_w = bi["cap_cell_leaf_cap_w"]
        self.assertGreater(
            int(cap_w[0, self.CELL]), 1,
            "the plateaued walker must have ramped")
        self.assertEqual(
            int(cap_w[1, self.CELL]), 1,
            "a walker still improving by >= D/2 must keep its cap")

    def test_occupancy_at_cap_is_per_walker(self):
        """A walker below ITS cap does not ramp on another walker's demand."""
        m = _pw_move(cap_divisor=2, nwalkers=2, min_iters=3)
        state, bi = _pw_state(m, nwalkers=2, start_cap=2)
        n = m.num_cap_cells
        occ = np.zeros((2, n), dtype=int)
        occ[0, self.CELL] = 2  # walker 0 AT its cap
        occ[1, self.CELL] = 1  # walker 1 has headroom
        for _ in range(6):
            stats = np.full((2, n), -1000.0)
            stats[:, self.CELL] = -100.0  # both flat -> both plateaued
            _pw_step(m, state, stats, occ)
        cap_w = bi["cap_cell_leaf_cap_w"]
        self.assertGreater(int(cap_w[0, self.CELL]), 2)
        self.assertEqual(
            int(cap_w[1, self.CELL]), 2,
            "a walker with headroom must not ramp on its neighbour's demand")

    def test_empty_cells_never_engage_for_any_walker(self):
        """The ghost-increment guard survives the walker axis."""
        m = _pw_move(cap_divisor=2, nwalkers=2, min_iters=3)
        state, bi = _pw_state(m, nwalkers=2, start_cap=1)
        n = m.num_cap_cells
        occ = np.zeros((2, n), dtype=int)
        rng = np.random.default_rng(0)
        for _ in range(10):
            _pw_step(m, state, -100.0 + 1e-3 * rng.standard_normal((2, n)),
                     occ)
        self.assertTrue(np.all(bi["cap_cell_leaf_cap_w"] == 1))

    def test_increment_resets_only_the_incrementing_walkers_clock(self):
        m = _pw_move(cap_divisor=2, nwalkers=2, min_iters=3)
        state, bi = _pw_state(m, nwalkers=2, start_cap=1)
        n = m.num_cap_cells
        occ = np.zeros((2, n), dtype=int)
        occ[0, self.CELL] = 1
        occ[1, self.CELL] = 1
        for i in range(4):
            stats = np.full((2, n), -1000.0)
            stats[0, self.CELL] = -100.0
            stats[1, self.CELL] = -100.0 + 10.0 * i
            _pw_step(m, state, stats, occ)
        self.assertEqual(int(bi["cap_cell_iters_w"][0, self.CELL]), 0)
        self.assertTrue(np.isneginf(bi["cap_cell_best_ll_w"][0, self.CELL]))
        self.assertGreater(float(bi["cap_cell_best_ll_w"][1, self.CELL]),
                           -np.inf)

    def test_band_best_ll_w_tracks_per_walker(self):
        m = _pw_move(cap_divisor=2, nwalkers=2, min_iters=3)
        state, bi = _pw_state(m, nwalkers=2, start_cap=1)
        ensure_leaf_cap_fields(bi, m.num_bands, per_walker=True)
        m._band_residual_lls = lambda acs: np.array(
            [[-10.0, -20.0, -30.0, -40.0], [-1.0, -2.0, -3.0, -4.0]])
        n = m.num_cap_cells
        _pw_step(m, state, np.full((2, n), -100.0),
                 np.zeros((2, n), dtype=int))
        np.testing.assert_allclose(
            bi["band_best_ll_w"],
            [[-10.0, -20.0, -30.0, -40.0], [-1.0, -2.0, -3.0, -4.0]])


class PerWalkerEnforcementTest(unittest.TestCase):
    """Every row is gated against ITS OWN walker's cap."""

    def test_cap_for_rows_shared_and_per_walker(self):
        m = _pw_move(cap_divisor=2, nwalkers=3)
        cells = np.array([0, 1, 2])
        walkers = np.array([0, 1, 2])
        shared = np.arange(m.num_cap_cells)
        np.testing.assert_array_equal(
            m._cap_for_rows(shared, walkers, cells), [0, 1, 2])
        per_w = np.arange(3 * m.num_cap_cells).reshape(3, m.num_cap_cells)
        np.testing.assert_array_equal(
            m._cap_for_rows(per_w, walkers, cells),
            [per_w[0, 0], per_w[1, 1], per_w[2, 2]])

    def test_row_at_cap_uses_the_rows_own_walker_cap(self):
        m = _pw_move(cap_divisor=2, nwalkers=2, ntemps=1)
        # one source per walker in cell 0; walker 0 capped at 1, walker 1 at 2
        cap = np.array([[1] + [1] * (m.num_cap_cells - 1),
                        [2] + [1] * (m.num_cap_cells - 1)])
        counts = np.zeros(1 * 2 * m.num_cap_cells, dtype=np.int32)
        counts[m._cap_flat_index(np.array([0]), np.array([0]),
                                 np.array([0]))] = 1
        counts[m._cap_flat_index(np.array([0]), np.array([1]),
                                 np.array([0]))] = 1
        t = np.array([0, 0])
        w = np.array([0, 1])
        cells = np.array([0, 0])
        at = m._row_at_cap(counts, cap, t, w, cells)
        np.testing.assert_array_equal(
            at, [True, False],
            "walker 0 is at its cap of 1; walker 1 has headroom to 2")

    def test_cap_new_entry_veto_uses_the_rows_own_walker_cap(self):
        m = _pw_move(cap_divisor=2, nwalkers=2, ntemps=1)
        os.environ["GB_CAP_INMODEL_HEADROOM"] = "0"
        self.addCleanup(os.environ.pop, "GB_CAP_INMODEL_HEADROOM", None)
        cap = np.array([[1] * m.num_cap_cells, [3] * m.num_cap_cells])
        counts = np.zeros(1 * 2 * m.num_cap_cells, dtype=np.int32)
        for wk in (0, 1):
            counts[m._cap_flat_index(np.array([0]), np.array([wk]),
                                     np.array([1]))] = 1
        t = np.array([0, 0])
        w = np.array([0, 1])
        cur = (np.array([0, 0]), None, None)
        new = (np.array([1, 1]), None, None)
        veto = m._cap_new_entry_veto(counts, cap, t, w, cur, new)
        np.testing.assert_array_equal(
            veto, [True, False],
            "cell 1 holds 1 leaf: at cap for walker 0 (cap 1), not for "
            "walker 1 (cap 3)")

    def test_band_saturated_flat_broadcasts_over_temps(self):
        m = _pw_move(cap_divisor=2, nwalkers=2, ntemps=3)
        n = m.num_cap_cells
        cap = np.ones((2, n), dtype=int)
        cap[1] = 2  # walker 1 is roomier everywhere
        counts = np.ones(3 * 2 * n, dtype=np.int32)  # one leaf in every cell
        sat = m._band_saturated_flat(counts, cap).reshape(3, 2, m.num_bands)
        self.assertTrue(np.all(sat[:, 0, :]),
                        "walker 0 (cap 1, one leaf per cell) is saturated")
        self.assertFalse(np.any(sat[:, 1, :]),
                         "walker 1 (cap 2) has headroom in every cell")


class PerWalkerSwapGateTest(unittest.TestCase):
    """A swap is gated against EACH SIDE'S OWN walker cap."""

    def test_asymmetric_caps_gate_each_side(self):
        # one cell affected; side A holds 2 (all from the band), side B 0
        occ_a = np.array([[2]])
        occ_b = np.array([[0]])
        from_a = np.array([[2]])
        from_b = np.array([[0]])
        # post_a = 0, post_b = 2. Side B's cap decides.
        ok = tempering_swap_cap_ok(occ_a, occ_b, from_a, from_b,
                                   np.array([[5]]), np.array([[1]]))
        self.assertFalse(bool(ok[0]), "side B's cap of 1 must veto post_b=2")
        ok = tempering_swap_cap_ok(occ_a, occ_b, from_a, from_b,
                                   np.array([[5]]), np.array([[2]]))
        self.assertTrue(bool(ok[0]), "side B's cap of 2 admits post_b=2")

    def test_cap_b_defaults_to_cap_a(self):
        occ_a = np.array([[2]])
        occ_b = np.array([[0]])
        from_a = np.array([[2]])
        from_b = np.array([[0]])
        self.assertFalse(
            bool(tempering_swap_cap_ok(occ_a, occ_b, from_a, from_b,
                                       np.array([[1]]))[0]))
        self.assertTrue(
            bool(tempering_swap_cap_ok(occ_a, occ_b, from_a, from_b,
                                       np.array([[2]]))[0]))

    def test_one_side_disarmed_is_unconstrained_on_that_side_only(self):
        occ_a = np.array([[9]])
        occ_b = np.array([[9]])
        from_a = np.array([[9]])
        from_b = np.array([[9]])
        # A disarmed (-1) -> no limit on A; B capped at 1 -> post_b = 9 vetoes
        ok = tempering_swap_cap_ok(occ_a, occ_b, from_a, from_b,
                                   np.array([[-1]]), np.array([[1]]))
        self.assertFalse(bool(ok[0]))
        ok = tempering_swap_cap_ok(occ_a, occ_b, from_a, from_b,
                                   np.array([[-1]]), np.array([[-1]]))
        self.assertTrue(bool(ok[0]), "both disarmed -> every swap admissible")


class PerWalkerFanoutSliceTest(unittest.TestCase):
    """A rank must receive ITS OWN block's rows of the cap table.

    A rank runs with ``self.nwalkers = B`` and every cap lookup uses a
    BLOCK-LOCAL walker index, so shipping the full N-row table would gate
    block-local walker 0 against global walker 0's allowance on every
    rank -- with every index still in range and nothing raising.
    """

    def test_per_walker_table_is_sliced_to_the_block(self):
        m = _pw_move(cap_divisor=2, nwalkers=4)
        cap = np.arange(4 * m.num_cap_cells).reshape(4, m.num_cap_cells)
        block = m._cap_table_for_block(cap, 2, 4)
        np.testing.assert_array_equal(block, cap[2:4])
        self.assertEqual(block.shape, (2, m.num_cap_cells))

    def test_shared_table_is_shipped_whole(self):
        m = _pw_move(cap_divisor=2, nwalkers=4, per_walker=False)
        cap = np.arange(m.num_cap_cells)
        np.testing.assert_array_equal(
            m._cap_table_for_block(cap, 2, 4), cap)

    def test_none_table_stays_none(self):
        m = _pw_move(cap_divisor=2, nwalkers=4)
        self.assertIsNone(m._cap_table_for_block(None, 0, 2))

    def test_block_slice_is_a_copy_not_a_view(self):
        """The head owns the caps: a rank's table must not alias them.

        ``_ro_table`` ships a host COPY -- "read-only" is the rank-side
        contract, not a numpy writeable flag. What must hold is that a
        rank writing its own table cannot move the head's, and that a
        head advancing the caps mid-propose cannot be seen by a rank.
        """
        m = _pw_move(cap_divisor=2, nwalkers=4)
        cap = np.zeros((4, m.num_cap_cells), dtype=int)
        block = m._cap_table_for_block(cap, 0, 2)
        block[0, 0] = 5
        self.assertEqual(int(cap[0, 0]), 0,
                         "the rank's table aliases the head's")
        cap[1, 0] = 9
        self.assertEqual(int(block[1, 0]), 0,
                         "the head's write leaked to a rank")

    def test_a_rank_rejects_a_table_that_does_not_match_its_block(self):
        """The RECEIVING end of the slice, which is otherwise silent.

        Every ``[GB_CAP_PW]`` line is head-side. A rank just takes
        ``tables["cap_leaf_cap"]`` and runs; if the head shipped the full
        N-row table (or the wrong block), every index still lands in
        range and the rank gates block-local walker 0 against global
        walker 0's allowance. On a 4-GPU run that is the one place the
        slice can be wrong with nothing in any log to say so.
        """
        m = _pw_move(cap_divisor=2, nwalkers=4)
        full = np.zeros((4, m.num_cap_cells), dtype=int)
        with self.assertRaises(RuntimeError) as ctx:
            m._check_rank_cap_table(full, 2)      # B = 2, got 4 rows
        msg = str(ctx.exception)
        self.assertIn("4", msg)
        self.assertIn("2", msg)

    def test_a_rank_accepts_its_own_block(self):
        m = _pw_move(cap_divisor=2, nwalkers=4)
        block = np.zeros((2, m.num_cap_cells), dtype=int)
        m._check_rank_cap_table(block, 2)          # no raise

    def test_a_shared_table_is_accepted_at_any_block_width(self):
        m = _pw_move(cap_divisor=2, nwalkers=4, per_walker=False)
        m._check_rank_cap_table(np.zeros(m.num_cap_cells, dtype=int), 2)

    def test_no_table_is_accepted(self):
        m = _pw_move(cap_divisor=2, nwalkers=4)
        m._check_rank_cap_table(None, 2)

    def test_the_rank_logs_what_it_received_once(self):
        m = _pw_move(cap_divisor=2, nwalkers=4)
        block = np.zeros((2, m.num_cap_cells), dtype=int)
        with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch",
                level="INFO") as cm:
            m._check_rank_cap_table(block, 2)
            m._check_rank_cap_table(block, 2)
        lines = [ln for ln in cm.output if "rank cap table" in ln]
        self.assertEqual(len(lines), 1, "must not log every propose")
        self.assertIn("2 walkers", lines[0])

    def test_every_block_covers_the_table_exactly_once(self):
        """The union of the blocks is the table, in order."""
        m = _pw_move(cap_divisor=2, nwalkers=4)
        cap = np.arange(4 * m.num_cap_cells).reshape(4, m.num_cap_cells)
        blocks = [m._cap_table_for_block(cap, w0, w1)
                  for w0, w1 in ((0, 1), (1, 3), (3, 4))]
        np.testing.assert_array_equal(np.concatenate(blocks, axis=0), cap)


class PerWalkerRowAlignmentTest(unittest.TestCase):
    """The gate's statistic must carry exactly one row per walker.

    The SHARED cap reduced the statistic with ``max(axis=0)``, which is
    indifferent to how many rows arrive and in what order -- a duplicated
    walker changed nothing. Per-walker caps index ``cap[w, c]`` against
    ``lls[w, c]`` positionally, so the row count and the row ORDER are
    now load-bearing.

    This matters for the multi-rank merge, which concatenates each
    compute rank's block on the walker axis. Today ``layout.block_of``
    partitions the walker axis, so the concatenation is exactly N rows in
    walker order. A layout where several ranks SHARE a walker block
    (GPUs > walkers) would hand the gate N x R rows -- which must fail
    loudly here rather than gate walker w on some other walker's series.
    """

    def setUp(self):
        self._old = os.environ.get("GB_LEAF_CAP_REQUIRE_IMPROVEMENT")
        os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = "1"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_LEAF_CAP_REQUIRE_IMPROVEMENT", None)
        else:
            os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = self._old

    def test_too_many_stat_rows_is_a_loud_failure(self):
        m = _pw_move(cap_divisor=2, nwalkers=2)
        state, _ = _pw_state(m)
        nc = m.num_cap_cells
        with self.assertRaises(RuntimeError) as ctx:
            _pw_step(m, state,
                     np.full((4, nc), -100.0),        # 2 walkers x R=2
                     np.zeros((4, nc), dtype=int))
        msg = str(ctx.exception)
        self.assertIn("4", msg)
        self.assertIn("2", msg)

    def test_too_few_stat_rows_is_a_loud_failure(self):
        m = _pw_move(cap_divisor=2, nwalkers=3)
        state, _ = _pw_state(m)
        nc = m.num_cap_cells
        with self.assertRaises(RuntimeError):
            _pw_step(m, state, np.full((2, nc), -100.0),
                     np.zeros((2, nc), dtype=int))

    def test_matching_rows_pass(self):
        m = _pw_move(cap_divisor=2, nwalkers=3)
        state, _ = _pw_state(m)
        nc = m.num_cap_cells
        _pw_step(m, state, np.full((3, nc), -100.0),
                 np.zeros((3, nc), dtype=int))

    def test_the_shared_cap_path_still_accepts_any_row_count(self):
        """Not a regression on the flag-off path: max() is row-count blind."""
        m = _pw_move(cap_divisor=2, nwalkers=2, per_walker=False)
        state, _ = _pw_state(m)
        nc = m.num_cap_cells
        _pw_step(m, state, np.full((5, nc), -100.0),
                 np.zeros((5, nc), dtype=int))


LOGGER = "lisatools.globalfit.moves.gbspecialstretch"


class PerWalkerDiagnosticsTest(unittest.TestCase):
    """``[GB_CAP_PW]``: the run log must show what the gate decided.

    The shared-cap gate has twice produced increments that could not be
    explained after the fact from the store (most recently snapshot 19,
    where occupancy-at-cap held for 2 of 54 observed increments under
    every census tried). These lines exist so that question is answerable
    from the log at the moment of decision. A diagnostic that never fires
    is worth nothing, so each one is pinned here.
    """

    CELL = 3

    def setUp(self):
        self._old = os.environ.get("GB_LEAF_CAP_REQUIRE_IMPROVEMENT")
        os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = "1"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("GB_LEAF_CAP_REQUIRE_IMPROVEMENT", None)
        else:
            os.environ["GB_LEAF_CAP_REQUIRE_IMPROVEMENT"] = self._old

    def _ramp(self, m, state, n=5):
        nc = m.num_cap_cells
        occ = np.zeros((m.nwalkers, nc), dtype=int)
        occ[0, self.CELL] = 1
        for _ in range(n):
            stats = np.full((m.nwalkers, nc), -1000.0)
            stats[:, self.CELL] = -100.0
            _pw_step(m, state, stats, occ)

    def test_arming_line_names_the_mode_and_the_table(self):
        m = _pw_move(cap_divisor=2, nwalkers=3)
        _, bi = _pw_state(m)
        with self.assertLogs(LOGGER, level="INFO") as cm:
            m._log_per_walker_arm(bi["cap_cell_leaf_cap_w"])
        line = "\n".join(cm.output)
        self.assertIn("[GB_CAP_PW", line)
        self.assertIn("PER WALKER", line)
        self.assertIn("3 walkers", line)
        self.assertIn(f"{m.num_cap_cells} cap cells", line)

    def test_arming_line_says_SHARED_when_the_flag_is_off(self):
        """A run whose per-walker caps failed to allocate must say so."""
        m = _pw_move(cap_divisor=2, nwalkers=3, per_walker=False)
        _, bi = _pw_state(m)
        with self.assertLogs(LOGGER, level="INFO") as cm:
            m._log_per_walker_arm(bi["cap_cell_leaf_cap"])
        self.assertIn("SHARED", "\n".join(cm.output))

    def test_arming_line_fires_once_per_process(self):
        m = _pw_move(cap_divisor=2, nwalkers=3)
        _, bi = _pw_state(m)
        with self.assertLogs(LOGGER, level="INFO") as cm:
            m._log_per_walker_arm(bi["cap_cell_leaf_cap_w"])
            m._log_per_walker_arm(bi["cap_cell_leaf_cap_w"])
            m._log_per_walker_arm(bi["cap_cell_leaf_cap_w"])
        self.assertEqual(
            sum("PER WALKER" in ln for ln in cm.output), 1,
            "the arming line must not repeat every propose")

    def test_increment_line_reports_the_per_walker_breakdown(self):
        m = _pw_move(cap_divisor=2, nwalkers=2, min_iters=3)
        state, _ = _pw_state(m, start_cap=1)
        with self.assertLogs(LOGGER, level="INFO") as cm:
            self._ramp(m, state)
        inc = [ln for ln in cm.output if "incremented for" in ln]
        self.assertTrue(inc, f"no increment line logged; got {cm.output}")
        self.assertIn("(walker, cell) pairs", inc[0])
        self.assertIn("w0:", inc[0])
        self.assertIn("w1:", inc[0])

    def test_diag_line_carries_the_decision_tuple(self):
        os.environ["GB_CAP_PW_DIAG"] = "1"
        self.addCleanup(os.environ.pop, "GB_CAP_PW_DIAG", None)
        m = _pw_move(cap_divisor=2, nwalkers=2, min_iters=3)
        state, _ = _pw_state(m, start_cap=1)
        with self.assertLogs(LOGGER, level="INFO") as cm:
            self._ramp(m, state)
        detail = [ln for ln in cm.output if "occ=" in ln]
        self.assertTrue(detail, f"GB_CAP_PW_DIAG logged nothing: {cm.output}")
        # the answer to "was occupancy-at-cap really met"
        self.assertIn("occ=1", detail[0])
        self.assertIn("cap 1 -> 2", detail[0])
        self.assertIn("mHz", detail[0])

    def test_diag_is_silent_unless_asked(self):
        m = _pw_move(cap_divisor=2, nwalkers=2, min_iters=3)
        state, _ = _pw_state(m, start_cap=1)
        with self.assertLogs(LOGGER, level="INFO") as cm:
            self._ramp(m, state)
        self.assertEqual([ln for ln in cm.output if "occ=" in ln], [])

    def test_spread_line_says_when_the_mode_is_inert(self):
        m = _pw_move(cap_divisor=2, nwalkers=3)
        _, bi = _pw_state(m, start_cap=1)
        with self.assertLogs(LOGGER, level="INFO") as cm:
            m._log_per_walker_cap_spread(bi["cap_cell_leaf_cap_w"])
        self.assertIn("inert", "\n".join(cm.output))

    def test_spread_line_reports_real_divergence(self):
        m = _pw_move(cap_divisor=2, nwalkers=3)
        _, bi = _pw_state(m, start_cap=1)
        bi["cap_cell_leaf_cap_w"][1, 2] = 6
        with self.assertLogs(LOGGER, level="INFO") as cm:
            m._log_per_walker_cap_spread(bi["cap_cell_leaf_cap_w"])
        line = "\n".join(cm.output)
        self.assertIn("1/8 cells differ", line)
        self.assertIn("max spread 5", line)
        self.assertIn("cell 2", line)

    def test_all_walkers_flag_is_ignored_with_a_warning(self):
        os.environ["GB_LEAF_CAP_ALL_WALKERS"] = "1"
        self.addCleanup(os.environ.pop, "GB_LEAF_CAP_ALL_WALKERS", None)
        m = _pw_move(cap_divisor=2, nwalkers=2, min_iters=3)
        state, _ = _pw_state(m, start_cap=1)
        with self.assertLogs(LOGGER, level="WARNING") as cm:
            self._ramp(m, state, n=2)
        line = "\n".join(cm.output)
        self.assertIn("GB_LEAF_CAP_ALL_WALKERS=1 is IGNORED", line)
        # once, not every propose
        with self.assertLogs(LOGGER, level="INFO") as cm2:
            self._ramp(m, state, n=2)
        self.assertEqual(
            sum("ALL_WALKERS" in ln for ln in cm2.output), 0,
            "the subsumed-flag warning must not repeat every iteration")


class PerWalkerStorageRoundTripTest(unittest.TestCase):
    """The schema wiring: allocate -> persist -> restore -> resume."""

    NW, NT, K = 3, 2, 4

    def _state(self, per_walker):
        st = GBState(None)
        st.initialize_band_information(
            self.NW, self.NT, BAND_EDGES, np.zeros((NUM_BANDS, self.NT)),
            cap_edges=make_cap_edges(BAND_EDGES, self.K),
            leaf_cap_per_walker=per_walker,
        )
        return st

    def test_flag_off_persists_exactly_the_historical_arrays(self):
        """THE COMPAT GATE at the storage layer.

        A 6mo relaunch from this build must write the same dataset set it
        always did -- an extra per-iteration array is a schema change the
        backend would carry into the store.
        """
        arrays = self._state(False).storage_arrays()
        stray = [k for k in arrays if k.endswith("_w")]
        self.assertEqual(
            stray, [], f"new datasets on the flag-off path: {stray}")

    def test_per_walker_arrays_are_persisted(self):
        arrays = self._state(True).storage_arrays()
        for name in ("cap_cell_leaf_cap_w", "cap_cell_iters_w",
                     "cap_cell_best_ll_w", "band_best_ll_w"):
            self.assertIn(name, arrays, name)
        self.assertEqual(
            arrays["cap_cell_leaf_cap_w"].shape, (self.NW, NUM_BANDS * self.K))
        self.assertEqual(
            arrays["band_best_ll_w"].shape, (self.NW, NUM_BANDS))

    def test_from_stored_restores_the_per_walker_caps(self):
        st = self._state(True)
        st.band_info["cap_cell_leaf_cap_w"][0] = 2
        st.band_info["cap_cell_leaf_cap_w"][1] = 5
        st.band_info["cap_cell_leaf_cap_w"][2] = 9
        arrays = {k: np.asarray(v)[None]
                  for k, v in st.storage_arrays().items()}
        back = GBState.from_stored(
            arrays, statics=st.static_arrays(), attrs={})
        got = np.asarray(back.band_info["cap_cell_leaf_cap_w"])[0]
        np.testing.assert_array_equal(got[0], 2)
        np.testing.assert_array_equal(got[1], 5)
        np.testing.assert_array_equal(got[2], 9)

    def test_resume_strips_the_step_axis(self):
        """``_bare_ndim`` must know the per-walker family's bare rank.

        Backend-loaded arrays keep a leading step axis; a name missing
        from ``_bare_ndim`` stays 3-D in band_info and every consumer then
        indexes a step as a walker.
        """
        st = self._state(True)
        arrays = {k: np.asarray(v)[None]
                  for k, v in st.storage_arrays().items()}
        back = GBState.from_stored(
            arrays, statics=st.static_arrays(), attrs={})
        back.initialize_band_information(
            self.NW, self.NT, BAND_EDGES, np.zeros((NUM_BANDS, self.NT)),
            cap_edges=make_cap_edges(BAND_EDGES, self.K),
            leaf_cap_per_walker=True,
        )
        self.assertEqual(
            back.band_info["cap_cell_leaf_cap_w"].shape,
            (self.NW, NUM_BANDS * self.K))
        self.assertEqual(
            back.band_info["band_best_ll_w"].shape, (self.NW, NUM_BANDS))

    def test_enabling_over_a_store_written_without_the_flag(self):
        """No migration script: every walker inherits the stored cap."""
        st = self._state(False)
        st.band_info["cap_cell_leaf_cap"][:] = 3
        arrays = {k: np.asarray(v)[None]
                  for k, v in st.storage_arrays().items()}
        back = GBState.from_stored(
            arrays, statics=st.static_arrays(), attrs={})
        back.initialize_band_information(
            self.NW, self.NT, BAND_EDGES, np.zeros((NUM_BANDS, self.NT)),
            cap_edges=make_cap_edges(BAND_EDGES, self.K),
            leaf_cap_per_walker=True,
        )
        np.testing.assert_array_equal(
            back.band_info["cap_cell_leaf_cap_w"],
            np.full((self.NW, NUM_BANDS * self.K), 3))

    def test_a_stored_walker_axis_that_moved_is_a_refusal(self):
        """Not a silent regrid: it would gate walker w on another's cap."""
        st = self._state(True)
        arrays = {k: np.asarray(v)[None]
                  for k, v in st.storage_arrays().items()}
        back = GBState.from_stored(
            arrays, statics=st.static_arrays(), attrs={})
        # A cap table one walker SHORT of the ladder the store declares.
        # 2-D on purpose: a 3-D array is just the backend's step axis and
        # is stripped on reload, so it would not test anything.
        back.band_info["cap_cell_leaf_cap_w"] = np.full(
            (self.NW - 1, NUM_BANDS * self.K), -1)
        with self.assertRaises(ValueError) as ctx:
            back.initialize_band_information(
                self.NW, self.NT, BAND_EDGES, np.zeros((NUM_BANDS, self.NT)),
                cap_edges=make_cap_edges(BAND_EDGES, self.K),
                leaf_cap_per_walker=True,
            )
        self.assertIn("cap_cell_leaf_cap_w", str(ctx.exception))

    def test_vgb_cap_free_branch_drops_the_per_walker_family(self):
        st = self._state(True)
        arrays = {k: np.asarray(v)[None]
                  for k, v in st.storage_arrays().items()}
        back = GBState.from_stored(
            arrays, statics=st.static_arrays(), attrs={})
        back.initialize_band_information(
            self.NW, self.NT, BAND_EDGES, np.zeros((NUM_BANDS, self.NT)),
            cap_edges=make_cap_edges(BAND_EDGES, self.K),
            leaf_caps=False,
        )
        for name in ("cap_cell_leaf_cap_w", "cap_cell_iters_w",
                     "cap_cell_best_ll_w"):
            self.assertNotIn(name, back.band_info, name)


if __name__ == "__main__":
    unittest.main()
