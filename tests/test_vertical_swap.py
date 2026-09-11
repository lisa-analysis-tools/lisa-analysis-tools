"""Per-repeat VERTICAL band-temperature swaps in the in-model loop.

Fake-based (same style as ``test_inmodel_repeats``): the swap is pure
bookkeeping -- a relabel plus a closed-form acceptance ratio -- so fakes
exercise every branch of it in milliseconds, with no waveform, no buffer
and no GPU.

Why fakes rather than an end-to-end fixture: the CPU flow fixture
(``test_gbspecial_flow.build_fixture``) carries an all-zero data array, so
its templates contribute ~2e-4 to lnL and its swap pairs are almost all
EMPTY cells. An empty-vs-empty pair has ``paccept == 0``, which passes the
Metropolis test unconditionally and moves nothing -- so it inflates the
"accepted swaps" counter while exercising none of the bookkeeping. That was
measured, not assumed (2026-08-18). End-to-end coverage belongs on the GPU
probe, where cells are genuinely occupied.

Covered here:

* pair selection (same walker, same sub-band, adjacent temperatures);
* the closed-form ratio ``(b_cold - b_hot) * (L_hot - L_cold)``;
* the relabel: block ``t_i``/``beta``, sorter labels, per-cell ledgers;
* the hoisted-array staleness hazard -- after a swap the per-half
  ``beta_s`` must still equal ``band_temps[b_i, t_i]``;
* the block-boundary barrier (``special_index_check``);
* default OFF, and no template buffer ever touched.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialStretchMove,
    _resolve_temper_vertical,
)
from lisatools.globalfit.moves.gbbands import (
    BandSorter,
    pack_special_index,
    unpack_special_index,
)

NTEMPS, NWALKERS, NBANDS = 4, 2, 3


class _FakeSorter:
    """Minimal BandSorter with the real special-index semantics."""

    # The deferred-relabel machinery is carried REAL (not faked), so the
    # repeat-block window that _run_in_model_repeats opens under
    # GB_CELL_LABEL_DEFERRED runs end-to-end here -- this suite is the
    # sweep-level integration coverage for it. See
    # tests/test_cell_label_deferred.py for the unit-level composition.
    xp = np
    _deferred_labels = None
    begin_cell_label_window = BandSorter.begin_cell_label_window
    flush_cell_labels = BandSorter.flush_cell_labels
    _defer_exchange = BandSorter._defer_exchange
    _assert_cell_labels_flushed = BandSorter._assert_cell_labels_flushed

    def __init__(self, temp_inds, walker_inds, band_inds, nwalkers):
        self.temp_inds = np.asarray(temp_inds).copy()
        self.walker_inds = np.asarray(walker_inds).copy()
        self.band_inds = np.asarray(band_inds).copy()
        self.nwalkers = nwalkers
        self.special_band_inds = self.get_special_band_index(
            self.temp_inds, self.walker_inds, self.band_inds
        )
        self.touched_template_buffer = False

    def get_special_band_index(self, t, w, b):
        return pack_special_index(t, w, b, self.nwalkers)

    @property
    def special_index_check(self):
        # mirrors the real property: the alarm must judge FLUSHED state
        self._assert_cell_labels_flushed("special_index_check")
        return np.all(
            self.special_band_inds
            == self.get_special_band_index(
                self.temp_inds, self.walker_inds, self.band_inds
            )
        )

    def exchange_cell_labels(self, sa, ta, wa, sb, tb, wb, bands=None):
        """Real semantics: both membership maps computed before mutation."""
        if self._deferred_labels is not None:
            self._defer_exchange(sa, ta, wa, sb, tb, wb)
            return
        keep_a = np.isin(self.special_band_inds, np.atleast_1d(sa))
        keep_b = np.isin(self.special_band_inds, np.atleast_1d(sb))
        self.special_band_inds[keep_a] = np.atleast_1d(sb)[0]
        self.temp_inds[keep_a] = tb
        self.special_band_inds[keep_b] = np.atleast_1d(sa)[0]
        self.temp_inds[keep_b] = ta

    def exchange_cell_labels_batch(self, sa, ta, wa, sb, tb, wb,
                                   bands=None):
        """Batch = pairwise loop over this fake's own exchange (the real
        primitive's equivalence is pinned by
        BatchExchangeEquivalenceTest against the REAL BandSorter)."""
        if self._deferred_labels is not None:
            self._defer_exchange(sa, ta, wa, sb, tb, wb)
            return
        sa, sb = np.atleast_1d(sa), np.atleast_1d(sb)
        ta, tb = np.atleast_1d(ta), np.atleast_1d(tb)
        wa, wb = np.atleast_1d(wa), np.atleast_1d(wb)
        for k in range(sa.size):
            self.exchange_cell_labels(
                sa[k:k + 1], int(ta[k]), wa[k:k + 1],
                sb[k:k + 1], int(tb[k]), wb[k:k + 1],
                bands=None if bands is None
                else np.atleast_1d(bands)[k:k + 1])

    # a vertical swap must never reach for the template twin: the in-model
    # buffer has none (use_template_arr is True only inside run_tempering)
    def swap_template_slots(self, *a, **k):  # pragma: no cover - must not run
        self.touched_template_buffer = True
        raise AssertionError(
            "vertical swap must not touch the template buffer"
        )


class _Move(GBSpecialStretchMove):
    def __init__(self):  # bypass the production ctor
        pass


def _make_move(seed=3):
    m = _Move()
    m._backend_name = "lisatools_cpu"
    m.use_gpu = False
    m.branch_name = "gb"
    m.name = "fake_vert"
    m.ntemps = NTEMPS
    m.nwalkers = NWALKERS
    m.num_bands = NBANDS
    m.temper_vertical = True
    m._temper_rng = np.random.default_rng(seed)
    return m


def _ladder():
    """(num_bands, ntemps) betas, strictly decreasing along the ladder."""
    return np.tile(
        (1.0 / 2.0 ** np.arange(NTEMPS))[None, :], (NBANDS, 1)
    )


def _rows(band=1, walker=0):
    """One picked row per temperature of a single (walker, band) column."""
    t = np.arange(NTEMPS)
    w = np.full(NTEMPS, walker)
    b = np.full(NTEMPS, band)
    return t, w, b


class VerticalKnobTest(unittest.TestCase):
    def test_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_TEMPER_VERTICAL", None)
            self.assertFalse(_resolve_temper_vertical("gb", None))

    def test_env_on(self):
        with mock.patch.dict(os.environ, {"GB_TEMPER_VERTICAL": "1"}):
            self.assertTrue(_resolve_temper_vertical("gb", None))

    def test_kwarg_wins_over_env(self):
        with mock.patch.dict(os.environ, {"GB_TEMPER_VERTICAL": "1"}):
            self.assertFalse(_resolve_temper_vertical("gb", False))

    def test_branch_prefix_isolated(self):
        with mock.patch.dict(os.environ, {"GB_TEMPER_VERTICAL": "1"}):
            self.assertFalse(_resolve_temper_vertical("vgb", None))

    def test_bad_value_rejected(self):
        with mock.patch.dict(os.environ, {"GB_TEMPER_VERTICAL": "yes-please"}):
            with self.assertRaises(ValueError):
                _resolve_temper_vertical("gb", None)


class VerticalPairsTest(unittest.TestCase):
    def test_pairs_are_same_walker_same_band_adjacent_temps(self):
        mv = _make_move()
        t, w, b = _rows()
        hot, cold = mv._vertical_pairs(t, w, b)
        self.assertEqual(len(hot), NTEMPS - 1)
        for h, c in zip(hot, cold):
            self.assertEqual(t[h], t[c] + 1)
            self.assertEqual(w[h], w[c])
            self.assertEqual(b[h], b[c])

    def test_no_pair_across_different_walkers(self):
        """Different walkers means different data slabs -- never a pair."""
        mv = _make_move()
        t = np.array([0, 1])
        w = np.array([0, 1])          # <- differs
        b = np.array([1, 1])
        hot, cold = mv._vertical_pairs(t, w, b)
        self.assertEqual(len(hot), 0)

    def test_no_pair_across_different_bands(self):
        mv = _make_move()
        t = np.array([0, 1])
        w = np.array([0, 0])
        b = np.array([1, 2])          # <- differs
        hot, cold = mv._vertical_pairs(t, w, b)
        self.assertEqual(len(hot), 0)

    def test_non_adjacent_temps_are_not_paired(self):
        mv = _make_move()
        t = np.array([0, 2])
        w = np.array([0, 0])
        b = np.array([1, 1])
        hot, cold = mv._vertical_pairs(t, w, b)
        self.assertEqual(len(hot), 0)


class _SweepFixture:
    """A single (walker, band) column with one picked row per temperature."""

    def __init__(self, ll, seed=3, base=None):
        # ``ll`` = per-row add-delta ll_ref; ``base`` = per-row source-free
        # slab likelihood L_free = -<r|r>/2 (zeros by default, so L_with ==
        # ll_ref and the ratio tests read directly in ``ll``).
        self.mv = _make_move(seed)
        self.base = (
            np.zeros(NTEMPS) if base is None else np.asarray(base, dtype=float)
        )
        self.t, self.w, self.b = _rows()
        self.slots = np.arange(NTEMPS, dtype=int)
        self.band_temps = _ladder()
        self.beta = self.band_temps[self.b, self.t].copy()
        self.ll_ref = np.asarray(ll, dtype=float)
        self.sorter = _FakeSorter(self.t, self.w, self.b, NWALKERS)
        self.ll_change = np.zeros((NTEMPS, NWALKERS, NBANDS))
        self.prop = np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int)
        self.acc = np.zeros_like(self.prop)
        self.cell_ll = {
            "spec": self.sorter.special_band_inds.copy(),
            "ll0": self.ll_ref.copy(),
            "led0": np.zeros(NTEMPS),
            "rep0": np.zeros(NTEMPS, dtype=int),
        }

    def sweep(self, parity=0):
        return self.mv._vertical_swap_sweep(
            self.sorter, self.band_temps, self.t, self.w, self.b,
            self.slots, self.beta, self.ll_ref, self.ll_change,
            self.prop, self.acc, self.cell_ll, parity,
            cell_ll_base=self.base,
        )


class VerticalSweepTest(unittest.TestCase):
    def test_favourable_swap_is_always_accepted(self):
        """A hotter rung holding the BETTER model always swaps down.

        paccept = (b_cold - b_hot) * (L_hot - L_cold) > 0 when the hot rung
        has the higher likelihood and b_cold > b_hot, and any positive
        exponent beats log(u) for u in (0, 1).
        """
        # temps 0..3, betas 1, .5, .25, .125; give t=1 a much better ll
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        n = fx.sweep(parity=0)          # pairs with cold rung even: (1,0),(3,2)
        self.assertGreaterEqual(n, 1)
        # the good model moved DOWN to the cold rung
        self.assertEqual(fx.t[1], 0)
        self.assertEqual(fx.t[0], 1)
        self.assertAlmostEqual(fx.beta[1], fx.band_temps[fx.b[1], 0])
        self.assertAlmostEqual(fx.beta[0], fx.band_temps[fx.b[0], 1])

    def test_strongly_unfavourable_swap_is_rejected(self):
        """Cold rung already far better -> exponent very negative."""
        fx = _SweepFixture(ll=[-10.0, -1e6, -10.0, -1e6])
        n = fx.sweep(parity=0)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(fx.t, np.arange(NTEMPS))

    def test_no_likelihood_and_no_template_buffer_touched(self):
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        ll_before = fx.ll_ref.copy()
        fx.sweep(parity=0)
        # the ratio is closed form: lls are READ, never recomputed
        np.testing.assert_array_equal(fx.ll_ref, ll_before)
        self.assertFalse(fx.sorter.touched_template_buffer)

    def test_sorter_labels_stay_self_consistent(self):
        """The block-boundary barrier's invariant."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        fx.sweep(parity=0)
        self.assertTrue(bool(fx.sorter.special_index_check))
        t_unpacked, _, _ = unpack_special_index(
            fx.sorter.special_band_inds, NWALKERS
        )
        np.testing.assert_array_equal(t_unpacked, fx.sorter.temp_inds)

    def test_sorter_sources_actually_change_temperature(self):
        """The SORTER must move, not merely stay self-consistent.

        Self-consistency alone is satisfied trivially by doing nothing --
        that exact gap let a `run_tempering` mutation (deleting the
        ``exchange_cell_labels`` call) survive an earlier version of this
        suite. Assert the observable change itself.
        """
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        before = fx.sorter.temp_inds.copy()
        n = fx.sweep(parity=0)
        self.assertGreaterEqual(n, 1)
        after = fx.sorter.temp_inds
        self.assertTrue(
            bool((before != after).any()),
            "accepted vertical swaps did not relabel a single source in the "
            "sorter -- exchange_cell_labels never took effect",
        )
        # rows 0 and 1 are the accepted pair: their temps must have traded
        self.assertEqual(int(after[0]), int(before[1]))
        self.assertEqual(int(after[1]), int(before[0]))
        # and the count of sources per rung is conserved by a swap
        np.testing.assert_array_equal(
            np.bincount(before, minlength=NTEMPS),
            np.bincount(after, minlength=NTEMPS),
        )

    def test_per_cell_ledgers_follow_the_model(self):
        """ll_change_log / counters are keyed by cell and must trade too."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        w0, b0 = fx.w[0], fx.b[0]
        fx.ll_change[0, w0, b0] = 7.0     # cold cell's credit
        fx.ll_change[1, w0, b0] = 9.0     # hot cell's credit
        fx.prop[1][0, w0, b0] = 3
        fx.prop[1][1, w0, b0] = 5
        n = fx.sweep(parity=0)
        self.assertGreaterEqual(n, 1)
        self.assertEqual(fx.ll_change[0, w0, b0], 9.0)
        self.assertEqual(fx.ll_change[1, w0, b0], 7.0)
        self.assertEqual(fx.prop[1][0, w0, b0], 5)
        self.assertEqual(fx.prop[1][1, w0, b0], 3)

    def test_cell_ll_slot_state_follows_the_model(self):
        """`spec` staleness (hazard 3) -- slot->cell label must move."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        spec_before = fx.cell_ll["spec"].copy()
        ll0_before = fx.cell_ll["ll0"].copy()
        n = fx.sweep(parity=0)
        self.assertGreaterEqual(n, 1)
        self.assertEqual(fx.cell_ll["spec"][0], spec_before[1])
        self.assertEqual(fx.cell_ll["spec"][1], spec_before[0])
        self.assertEqual(fx.cell_ll["ll0"][0], ll0_before[1])
        self.assertEqual(fx.cell_ll["ll0"][1], ll0_before[0])

    def test_beta_matches_the_new_temperature(self):
        """Hazard 2: a stale beta would score later repeats wrongly."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        fx.sweep(parity=0)
        np.testing.assert_allclose(
            fx.beta, fx.band_temps[fx.b, fx.t],
            err_msg="beta must be re-derived from the post-swap temperature",
        )

    def test_parity_keeps_each_row_in_at_most_one_pair(self):
        """Adjacent pairs overlap; parity is what makes the sweep disjoint."""
        mv = _make_move()
        t, w, b = _rows()
        hot, cold = mv._vertical_pairs(t, w, b)
        for parity in (0, 1):
            sel = (t[cold] % 2) == parity
            rows = np.concatenate([hot[sel], cold[sel]])
            self.assertEqual(len(rows), len(set(rows.tolist())),
                             f"parity {parity} reused a row")

    def test_ladder_is_never_modified(self):
        """_adapt_band_temps stays exclusive to run_tempering."""
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        before = fx.band_temps.copy()
        fx.sweep(parity=0)
        np.testing.assert_array_equal(fx.band_temps, before)


class CellOrderTest(unittest.TestCase):
    """``BandScheduler`` cell ordering, and why vertical swaps need it.

    A vertical pair requires ``(t, w, b)`` and ``(t-1, w, b)`` to be
    resident in the buffer at the SAME time. Under the historical
    count-ordering that is coincidence; ``cell_order="band"`` makes a
    sub-band's whole column contiguous so partners land together.
    """

    NT, NW, NB, SLOTS = 8, 6, 40, 200

    def _cells(self, seed=2, lam=4.0):
        """Synthetic per-cell source counts -> a flat special-index array."""
        rng = np.random.default_rng(seed)
        cnt = rng.poisson(lam, size=(self.NT, self.NW, self.NB))
        t, w, b = np.meshgrid(
            np.arange(self.NT), np.arange(self.NW), np.arange(self.NB),
            indexing="ij",
        )
        spec = pack_special_index(t.ravel(), w.ravel(), b.ravel(), self.NW)
        return np.repeat(spec, cnt.ravel())

    def _resident(self, order):
        from lisatools.globalfit.moves.gbbands import BandScheduler

        sch = BandScheduler(self._cells(), self.SLOTS, xp=np,
                            cell_order=order, nwalkers=self.NW)
        return sch, set(sch.slot_specials.tolist())

    def _pair_fraction(self, resident):
        step = self.NW * 1000000          # one temperature rung
        n = sum(1 for s in resident if (s - step) in resident)
        return n / max(len(resident), 1)

    def test_default_is_count_and_unchanged(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler

        cells = self._cells()
        a = BandScheduler(cells, self.SLOTS, xp=np)
        b = BandScheduler(cells, self.SLOTS, xp=np, cell_order="count")
        np.testing.assert_array_equal(a.cell_specials, b.cell_specials)
        np.testing.assert_array_equal(a.cell_counts, b.cell_counts)
        self.assertEqual(a.cell_order, "count")

    def test_count_order_is_ascending(self):
        sch, _ = self._resident("count")
        self.assertTrue(np.all(np.diff(sch.cell_counts) >= 0))

    def test_band_order_groups_each_band_contiguously(self):
        sch, _ = self._resident("band")
        bands = sch.cell_specials % 1000000
        # a band's entries must form ONE contiguous run
        self.assertTrue(np.all(np.diff(bands) >= 0), "bands not sorted")
        self.assertEqual(len(np.unique(bands)), self.NB)

    def test_band_order_puts_temperature_partners_adjacent(self):
        """Temperature LAST is what survives a buffer smaller than a column.

        A vertical pair is (t, w, b) / (t-1, w, b). Ordering by
        (band, walker, temp) makes those neighbours in the slot sequence, so
        they stay co-resident even when slots < ntemps*nwalkers.
        """
        sch, _ = self._resident("band")
        spec = sch.cell_specials
        tw = spec // 1000000
        walker, temp = tw % self.NW, tw // self.NW
        band = spec % 1000000
        # Within a (band, walker) group temperatures must run in ascending
        # order with no other group interleaved. They need not be
        # CONSECUTIVE: an empty cell has no sources, so it never enters the
        # scheduler and its rung is simply absent.
        step = np.diff(temp)
        same_group = (np.diff(band) == 0) & (np.diff(walker) == 0)
        self.assertTrue(
            np.all(step[same_group] > 0),
            "temperatures within a (band, walker) group must ascend",
        )
        # and every present partner pair IS adjacent in the slot sequence
        adjacent_pairs = int(np.sum(same_group & (step == 1)))
        self.assertGreater(
            adjacent_pairs, 0,
            "no vertical partner pair ended up adjacent -- the ordering is "
            "not delivering the property it exists for",
        )

    def test_band_order_needs_nwalkers(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler

        with self.assertRaises(ValueError):
            BandScheduler(self._cells(), self.SLOTS, xp=np, cell_order="band")

    def test_band_order_preserves_the_cell_set(self):
        """Ordering only -- no cell may be added, dropped or duplicated."""
        sa, _ = self._resident("count")
        sb, _ = self._resident("band")
        np.testing.assert_array_equal(
            np.sort(sa.cell_specials), np.sort(sb.cell_specials))
        self.assertEqual(sa.n_cells, sb.n_cells)
        # counts must travel with their own cell
        ma = dict(zip(sa.cell_specials.tolist(), sa.cell_counts.tolist()))
        mb = dict(zip(sb.cell_specials.tolist(), sb.cell_counts.tolist()))
        self.assertEqual(ma, mb)

    def test_band_order_raises_vertical_partner_availability(self):
        """The measurement that motivates the knob."""
        _, res_count = self._resident("count")
        _, res_band = self._resident("band")
        f_count = self._pair_fraction(res_count)
        f_band = self._pair_fraction(res_band)
        self.assertGreater(
            f_band, 3.0 * f_count,
            f"band ordering must materially raise vertical partner "
            f"availability (count={f_count:.3f}, band={f_band:.3f})",
        )

    def test_bad_order_rejected(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler

        with self.assertRaises(ValueError):
            BandScheduler(self._cells(), self.SLOTS, xp=np,
                          cell_order="sideways", nwalkers=self.NW)


class VerticalWiringTest(unittest.TestCase):
    """The sweep as wired INTO ``_run_in_model_repeats``.

    The unit tests above call ``_vertical_swap_sweep`` directly; this
    exercises the wiring around it -- the per-repeat call, the ``_half_pre``
    rebuild after an accepted swap, and the block-boundary barrier.
    """

    @staticmethod
    def _harness():
        """The in-model fakes, importable as ``tests.x`` or bare ``x``."""
        try:
            from tests import test_inmodel_repeats as h
        except ImportError:  # discovered from inside tests/
            import test_inmodel_repeats as h
        return h

    def _deferrable_harness(self):
        """Declare the harness's fake proposal label-safe.

        ``test_inmodel_repeats._HarnessMove`` overrides
        ``in_model_proposal``, and ``_inmodel_labels_deferrable`` treats an
        unknown override as label-reading (the conservative default). The
        fake proposal touches no sorter labels, so say so explicitly --
        otherwise the deferred window can never open under this harness and
        the tests below would pass vacuously.
        """
        h = self._harness()
        return mock.patch.object(
            h._HarnessMove, "inmodel_proposal_reads_labels", False,
            create=True)

    def _problem(self, n_rep, vertical, seed=11):
        h = self._harness()
        _FakeBuffer, _make_move = h._FakeBuffer, h._make_move

        n_src = 8
        # ONE (walker, band) column, one picked row per temperature: the
        # only shape in which vertical partners exist at all.
        t = np.arange(NTEMPS)
        w = np.zeros(NTEMPS, dtype=int)
        b = np.ones(NTEMPS, dtype=int)
        ids = np.arange(NTEMPS)
        rng = np.random.RandomState(seed)
        coords = np.zeros((n_src, 4))
        coords[:, 0] = rng.uniform(-0.5, 0.5, n_src)
        coords[:, 1] = rng.uniform(2.95, 3.05, n_src)
        coords[:, 2] = rng.uniform(-1, 1, n_src)
        coords[:, 3] = rng.uniform(-1, 1, n_src)

        picked = {
            "ids": ids, "specials": pack_special_index(t, w, b, NWALKERS),
            "slot_index": np.arange(NTEMPS, dtype=np.int32),
            "temp_inds": t.copy(), "walker_inds": w.copy(),
            "band_inds": b.copy(), "N_vals": np.full(NTEMPS, 64),
        }
        sorter = _FakeSorter(t, w, b, NWALKERS)
        sorter.inds = np.ones(n_src, dtype=bool)
        sorter.coords = coords.copy()
        sorter.leaf_inds = np.arange(n_src)

        mv = _make_move(n_rep)
        mv.ntemps, mv.nwalkers, mv.num_bands = NTEMPS, NWALKERS, NBANDS
        mv.temper_vertical = vertical
        mv._temper_rng = np.random.default_rng(5)
        mv.sequential_parity_repeats = False

        band_temps = _ladder()
        ll_change = np.zeros((NTEMPS, NWALKERS, NBANDS))
        prop = np.zeros((2, NTEMPS, NWALKERS, NBANDS), dtype=int)
        acc = np.zeros_like(prop)
        cell_ll = {
            "spec": sorter.special_band_inds.copy(),
            "ll0": np.zeros(NTEMPS), "led0": np.zeros(NTEMPS),
            "rep0": np.zeros(NTEMPS, dtype=int),
        }
        np.random.seed(seed)
        mv._run_in_model_repeats(
            None, sorter, _FakeBuffer(NTEMPS), band_temps, picked,
            ll_change, prop, acc, num_repeats=n_rep, cell_ll_state=cell_ll,
        )
        return mv, sorter, band_temps, picked

    def test_off_by_default_leaves_temperatures_alone(self):
        _, sorter, _, picked = self._problem(6, vertical=False)
        np.testing.assert_array_equal(
            sorter.temp_inds, picked["temp_inds"],
            err_msg="vertical swaps must be OFF unless asked for",
        )

    def test_on_swaps_and_leaves_state_consistent(self):
        mv, sorter, band_temps, picked = self._problem(6, vertical=True)
        # the barrier inside _run_in_model_repeats would have raised on an
        # inconsistent relabel; assert the invariant here too
        self.assertTrue(bool(sorter.special_index_check))
        # rungs are a permutation of the originals (a swap conserves them)
        np.testing.assert_array_equal(
            np.sort(sorter.temp_inds), np.sort(picked["temp_inds"]),
        )
        self.assertFalse(sorter.touched_template_buffer)

    def test_ladder_untouched_by_the_in_model_loop(self):
        _, _, band_temps, _ = self._problem(6, vertical=True)
        np.testing.assert_array_equal(band_temps, _ladder())

    # ---- deferred cell relabels (GB_CELL_LABEL_DEFERRED) ----

    def test_deferred_relabel_matches_immediate_over_the_block(self):
        """Knob ON == knob OFF for a whole repeat block.

        The block runs one sweep per repeat step; with the knob ON they
        compose into a single window flushed at the block boundary. Six
        repeats of chained relabels is exactly the case a wrong
        composition would get wrong.
        """
        with mock.patch.dict(os.environ, {"GB_CELL_LABEL_DEFERRED": "0"}):
            _, s_off, _, picked = self._problem(6, vertical=True)
        with mock.patch.dict(os.environ, {"GB_CELL_LABEL_DEFERRED": "1"}), \
                self._deferrable_harness():
            _, s_on, _, _ = self._problem(6, vertical=True)

        np.testing.assert_array_equal(s_on.temp_inds, s_off.temp_inds)
        np.testing.assert_array_equal(s_on.walker_inds, s_off.walker_inds)
        np.testing.assert_array_equal(
            s_on.special_band_inds, s_off.special_band_inds)
        # the comparison is only worth anything if the block moved rows
        self.assertFalse(
            np.array_equal(s_off.temp_inds, picked["temp_inds"]),
            "no swap was accepted -- the equivalence proves nothing",
        )
        # window closed behind it
        self.assertIsNone(s_on._deferred_labels)

    def test_one_window_per_block_not_per_repeat_step(self):
        """The saving, pinned: ONE full-table relabel for the block.

        Six repeat steps means six vertical sweeps; before this rework each
        accepted sweep paid its own full-table isin + syncing boolean
        getitem + 3 scatters.
        """
        opened, flushed = [], []
        real_open = _FakeSorter.begin_cell_label_window
        real_flush = _FakeSorter.flush_cell_labels

        def spy_open(self, cells):
            out = real_open(self, cells)
            opened.append(int(np.asarray(cells).size))
            return out

        def spy_flush(self, close=False):
            out = real_flush(self, close=close)
            if out:
                flushed.append(bool(close))
            return out

        with mock.patch.dict(os.environ, {"GB_CELL_LABEL_DEFERRED": "1"}), \
                self._deferrable_harness(), \
                mock.patch.object(
                    _FakeSorter, "begin_cell_label_window", spy_open), \
                mock.patch.object(
                    _FakeSorter, "flush_cell_labels", spy_flush):
            self._problem(6, vertical=True)

        self.assertEqual(len(opened), 1, "one window per repeat BLOCK")
        self.assertEqual(
            flushed, [True], "exactly one closing flush at the block boundary")

    def test_no_window_when_vertical_swaps_are_off(self):
        """Nothing to defer without sweeps -- do not open a window."""
        opened = []
        real_open = _FakeSorter.begin_cell_label_window

        def spy_open(self, cells):
            opened.append(1)
            return real_open(self, cells)

        with mock.patch.dict(os.environ, {"GB_CELL_LABEL_DEFERRED": "1"}), \
                self._deferrable_harness(), \
                mock.patch.object(
                    _FakeSorter, "begin_cell_label_window", spy_open):
            self._problem(6, vertical=False)
        self.assertEqual(opened, [])

    def test_label_reading_proposal_override_opts_out(self):
        """VGB reads sorter labels INSIDE the loop, so it must not defer.

        ``VGBSpecialStretchMove.in_model_proposal`` looks up
        ``band_sorter.temp_inds[source_ids]`` per repeat step; a window
        spanning the block would hand it pre-swap labels. Any override is
        treated as label-live.
        """
        from lisatools.globalfit.moves.gbspecialstretch import (
            GBSpecialBase,
            VGBSpecialStretchMove,
        )

        # the base move defers
        self.assertTrue(_make_move()._inmodel_labels_deferrable)

        # an UNKNOWN override does not (pessimistic inference)
        class _Override(GBSpecialBase):
            def in_model_proposal(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

        self.assertFalse(
            _Override.__new__(_Override)._inmodel_labels_deferrable)

        # VGB says so explicitly, at the class that carries the hazard
        self.assertIs(
            VGBSpecialStretchMove.inmodel_proposal_reads_labels, True)
        self.assertFalse(
            VGBSpecialStretchMove.__new__(
                VGBSpecialStretchMove)._inmodel_labels_deferrable)

        # ... and an override that KNOWS it is label-safe can opt back in
        class _SafeOverride(GBSpecialBase):
            inmodel_proposal_reads_labels = False

            def in_model_proposal(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

        self.assertTrue(
            _SafeOverride.__new__(_SafeOverride)._inmodel_labels_deferrable)


class _RealMethodSorter:
    """Duck-typed state carrying ONLY what the real BandSorter label
    methods touch, so the real (unbound) exchange methods can run against
    it on CPU: xp, special_band_inds, temp_inds, walker_inds, band_inds."""

    xp = np
    # the exchange primitives check for an open deferred window first
    # (GB_CELL_LABEL_DEFERRED); None keeps this fixture on the immediate
    # path, which is what this suite pins
    _deferred_labels = None

    def __init__(self, t, w, b, nwalkers):
        self.temp_inds = np.asarray(t).copy()
        self.walker_inds = np.asarray(w).copy()
        self.band_inds = np.asarray(b).copy()
        self.nwalkers = nwalkers
        self.special_band_inds = pack_special_index(
            self.temp_inds, self.walker_inds, self.band_inds, nwalkers)


class BatchExchangeEquivalenceTest(unittest.TestCase):
    """exchange_cell_labels_batch == K sequential pairwise calls.

    The orchestration audit (2026-08-27): the vertical sweep called
    exchange_cell_labels once PER ACCEPTED SWAP -- 2 full-table isin + 2
    int() syncs + 2 assert syncs each, ~51 ms/step of the 70 ms repeat
    cost. The batch primitive does ONE membership pass for all K disjoint
    pairs. Equivalence requires the 2K cells to be pairwise disjoint,
    which the sweep's parity selection guarantees.
    """

    def _states(self):
        # 3 pairs across distinct (temp, walker) cells of band 1, walkers
        # 0/1, temps 0..3; several sources per cell + bystander rows.
        t = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2])
        w = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2])
        return (_RealMethodSorter(t, w, b, NWALKERS),
                _RealMethodSorter(t, w, b, NWALKERS))

    def test_batch_matches_sequential(self):
        from lisatools.globalfit.moves.gbbands import BandSorter

        s_seq, s_bat = self._states()
        # pairs: (t=0,w=0,b=1)<->(t=1,w=0,b=1), (t=2,w=0,b=1)<->(t=3,w=0,b=1),
        #        (t=0,w=1,b=2)<->(t=2,w=1,b=2)  -- disjoint cells
        t_h = np.array([1, 3, 2])
        t_c = np.array([0, 2, 0])
        w_p = np.array([0, 0, 1])
        b_p = np.array([1, 1, 2])
        sp_h = pack_special_index(t_h, w_p, b_p, NWALKERS)
        sp_c = pack_special_index(t_c, w_p, b_p, NWALKERS)

        for k in range(3):
            BandSorter.exchange_cell_labels(
                s_seq, sp_h[k:k + 1], int(t_h[k]), w_p[k:k + 1],
                sp_c[k:k + 1], int(t_c[k]), w_p[k:k + 1],
                bands=b_p[k:k + 1])
        BandSorter.exchange_cell_labels_batch(
            s_bat, sp_h, t_h, w_p, sp_c, t_c, w_p, bands=b_p)

        np.testing.assert_array_equal(
            s_seq.special_band_inds, s_bat.special_band_inds)
        np.testing.assert_array_equal(s_seq.temp_inds, s_bat.temp_inds)
        np.testing.assert_array_equal(s_seq.walker_inds, s_bat.walker_inds)
        np.testing.assert_array_equal(s_seq.band_inds, s_bat.band_inds)
        # sanity: something actually moved, and bystanders did not
        self.assertFalse(np.array_equal(
            s_bat.temp_inds, self._states()[0].temp_inds))
        self.assertEqual(int(s_bat.temp_inds[9]), 1)  # untouched bystander

    def test_batch_empty_is_noop(self):
        from lisatools.globalfit.moves.gbbands import BandSorter

        s, ref = self._states()
        e = np.array([], dtype=np.int64)
        BandSorter.exchange_cell_labels_batch(
            s, e, e, e, e, e, e, bands=None)
        np.testing.assert_array_equal(s.temp_inds, ref.temp_inds)
        np.testing.assert_array_equal(
            s.special_band_inds, ref.special_band_inds)


class VerticalCensusDeSyncTest(unittest.TestCase):
    """Rung counters accumulate DEVICE-side per sweep and flush ONCE at the
    block-end log (orchestration audit 2026-08-27 candidate 6): the two
    per-sweep host bincount pulls were 2 forced syncs x one sweep per
    repeat step, spent purely on a log line."""

    def _swept_census(self):
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        cn = fx.mv._vertical_census_new(NTEMPS)
        n_acc = fx.mv._vertical_swap_sweep(
            fx.sorter, fx.band_temps, fx.t, fx.w, fx.b,
            fx.slots, fx.beta, fx.ll_ref, fx.ll_change,
            fx.prop, fx.acc, fx.cell_ll, 0, census=cn,
            cell_ll_base=fx.base,
        )
        return fx, cn, n_acc

    def test_host_rung_arrays_untouched_per_sweep(self):
        _, cn, _ = self._swept_census()
        self.assertGreater(cn["proposed"], 0)
        self.assertEqual(int(cn["prop_by_rung"].sum()), 0)
        self.assertIsNotNone(cn.get("prop_by_rung_dev"))
        self.assertGreater(int(np.asarray(cn["prop_by_rung_dev"]).sum()), 0)

    def test_flush_merges_exactly_once_and_matches_totals(self):
        fx, cn, n_acc = self._swept_census()
        fx.mv._vertical_census_flush(cn)
        self.assertEqual(int(cn["prop_by_rung"].sum()), cn["proposed"])
        self.assertEqual(int(cn["acc_by_rung"].sum()), cn["accepted"])
        self.assertEqual(cn["accepted"], n_acc)
        self.assertIsNone(cn.get("prop_by_rung_dev"))
        self.assertIsNone(cn.get("acc_by_rung_dev"))
        # idempotent: flushing again changes nothing
        before = cn["prop_by_rung"].copy()
        fx.mv._vertical_census_flush(cn)
        np.testing.assert_array_equal(cn["prop_by_rung"], before)


class VerticalLWithTest(unittest.TestCase):
    """The ratio compares WHOLE-cell likelihoods, L_with = L_free + ll_ref.

    ``ll_ref`` is the picked source's add-delta ``<r|h> - <h|h>/2`` against
    the cell's SOURCE-FREE residual ``r``; ``L_free = -<r|r>/2`` is the
    slab term the block measures once after the removal. Inside a cell
    L_free cancels; between two cells it does not (2026-09-10 correction).
    """

    def test_missing_base_is_refused(self):
        fx = _SweepFixture(ll=[-100.0, -10.0, -100.0, -100.0])
        fx.base = None
        with self.assertRaises(ValueError):
            fx.sweep(parity=0)

    def test_equal_deltas_do_not_mean_equal_cells(self):
        """Cold cell {A, B} (picked A) vs hot cell {A'} (picked A').

        Both cells see the SAME add-delta for A (A improves each cell by
        SNR_A^2/2 = 800), so the 2026-08-18 statistic was 0 and the swap
        was accepted at every draw. The source-free residual of the hot
        cell still contains B (L_free = -SNR_B^2/2 = -450); the cold
        cell's is clean (L_free = 0). L_with: cold 800, hot 350 -> the
        cold model is better by 450 and the swap must be rejected.
        """
        ll = [800.0, 800.0, -10.0, -1e6]         # rows: t0 (cold), t1 (hot), ...
        base = [0.0, -450.0, 0.0, 0.0]
        fx = _SweepFixture(ll=ll, base=base)
        n = fx.sweep(parity=0)                  # pair (t1, t0)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(fx.t, np.arange(NTEMPS))

    def test_equal_deltas_swap_when_the_hot_residual_is_cleaner(self):
        """Mirror image: the hot cell holds the extra (good) companion."""
        ll = [800.0, 800.0, -10.0, -1e6]
        base = [-450.0, 0.0, 0.0, 0.0]          # cold residual still has B
        fx = _SweepFixture(ll=ll, base=base)
        n = fx.sweep(parity=0)
        self.assertEqual(n, 1)
        self.assertEqual(fx.t[1], 0)
        self.assertEqual(fx.t[0], 1)

    def test_base_and_ll_ref_stay_with_their_rows(self):
        """Rows keep their slots, so neither per-row array is swapped."""
        ll = [800.0, 800.0, -10.0, -1e6]
        base = [-450.0, 0.0, 0.0, 0.0]
        fx = _SweepFixture(ll=ll, base=base)
        ll_before, base_before = fx.ll_ref.copy(), fx.base.copy()
        self.assertEqual(fx.sweep(parity=0), 1)
        np.testing.assert_array_equal(fx.ll_ref, ll_before)
        np.testing.assert_array_equal(fx.base, base_before)

    def test_sole_occupants_reduce_to_the_delta_form(self):
        """Two sole-occupant cells share the bare data slab: equal bases, so
        the decision is the 2026-08-18 one (exact there)."""
        ll = [-100.0, -10.0, -100.0, -100.0]
        fx_a = _SweepFixture(ll=ll, base=[-5.0, -5.0, -5.0, -5.0], seed=3)
        fx_b = _SweepFixture(ll=ll, base=None, seed=3)
        self.assertEqual(fx_a.sweep(parity=0), fx_b.sweep(parity=0))
        np.testing.assert_array_equal(fx_a.t, fx_b.t)


class ColumnStagingTest(unittest.TestCase):
    """Every rung of a (walker, band) column is staged together
    (user requirement 2026-09-10)."""

    @staticmethod
    def _pool(t, w, b):
        t, w, b = (np.asarray(x, dtype=int) for x in (t, w, b))
        n = t.size
        return {
            "ids": np.arange(n), "temp_inds": t, "walker_inds": w,
            "band_inds": b, "slot_index": np.arange(n, dtype=np.int32),
            "specials": pack_special_index(t, w, b, NWALKERS),
        }

    def test_order_is_band_walker_temp(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _order_pool_by_column,
        )
        # pick order: scrambled rungs of two columns
        pool = self._pool(t=[2, 0, 1, 3, 1, 0], w=[1, 0, 1, 0, 0, 1],
                          b=[2, 2, 2, 2, 2, 2])
        out = _order_pool_by_column(pool, np, NBANDS)
        key = out["walker_inds"] * NBANDS + out["band_inds"]
        # columns contiguous, temperature ascending inside each
        self.assertEqual(key.tolist(), [2, 2, 2, 5, 5, 5])
        self.assertEqual(out["temp_inds"].tolist(), [0, 1, 3, 0, 1, 2])
        # identity when already ordered (same object)
        self.assertIs(_order_pool_by_column(out, np, NBANDS), out)
        # every parallel array permuted together
        np.testing.assert_array_equal(
            out["specials"],
            pack_special_index(out["temp_inds"], out["walker_inds"],
                               out["band_inds"], NWALKERS))

    def test_chunks_never_split_a_column(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _column_chunks, _order_pool_by_column,
        )
        # three columns of 4, 3, 4 rows; width 6 -> [4], [3], [4]
        t = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3]
        w = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        b = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        pool = _order_pool_by_column(self._pool(t, w, b), np, NBANDS)
        chunks = list(_column_chunks(pool, 6, NBANDS))
        sizes = [int(c["ids"].size) for c in chunks]
        self.assertEqual(sizes, [4, 3, 4])
        for c in chunks:
            key = c["walker_inds"] * NBANDS + c["band_inds"]
            self.assertEqual(len(np.unique(key)), 1)
        # rows are conserved and in order
        np.testing.assert_array_equal(
            np.concatenate([c["ids"] for c in chunks]), pool["ids"])
        # width >= pool -> the same object back
        self.assertIs(next(_column_chunks(pool, 11, NBANDS)), pool)

    def test_newborn_class_is_lifted_to_the_column(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _column_atomic_newborn,
        )
        pool = self._pool(t=[0, 1, 2, 0, 1], w=[0, 0, 0, 1, 1],
                          b=[1, 1, 1, 1, 1])
        pool["newborn"] = np.array([False, True, False, False, False])
        out = _column_atomic_newborn(pool, np, NBANDS)
        self.assertEqual(out["newborn"].tolist(),
                         [True, True, True, False, False])

    def test_capacity_is_a_multiple_of_ntemps_with_swaps_on(self):
        mv = _make_move()
        mv.num_band_preload = 1000
        mv.gb = SimpleNamespace(gpus=None)      # CPU backend: gpus unused
        mv.temper_vertical = True          # NTEMPS = 4 -> 1000
        self.assertEqual(mv.num_band_preload_total, 1000)
        mv.ntemps = 3                      # -> 999
        self.assertEqual(mv.num_band_preload_total, 999)
        mv.num_band_preload = 2            # never below one column
        self.assertEqual(mv.num_band_preload_total, 3)
        mv.temper_vertical = False         # untouched with swaps off
        self.assertEqual(mv.num_band_preload_total, 2)

    def test_band_order_is_the_default_with_swaps_on(self):
        from lisatools.globalfit.moves.gbspecialstretch import (
            _resolve_temper_cell_order,
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_TEMPER_CELL_ORDER", None)
            self.assertEqual(
                _resolve_temper_cell_order("gb", None, default="band"), "band")
            self.assertEqual(
                _resolve_temper_cell_order("gb", None, default="count"),
                "count")
        with mock.patch.dict(os.environ, {"GB_TEMPER_CELL_ORDER": "count"}):
            self.assertEqual(
                _resolve_temper_cell_order("gb", None, default="band"),
                "count")


class SchedulerRelabelTest(unittest.TestCase):
    """``BandScheduler.relabel_slots``: counters and slot->cell follow the
    model through a vertical relabel."""

    def _sched(self):
        from lisatools.globalfit.moves.gbbands import BandScheduler
        # one column, 3 rungs; cell t0 has 3 sources, t1 has 1, t2 has 2
        t = np.array([0, 0, 0, 1, 2, 2])
        w = np.zeros(6, dtype=int)
        b = np.ones(6, dtype=int)
        spec = pack_special_index(t, w, b, NWALKERS)
        return BandScheduler(spec, 3, xp=np, cell_order="band",
                             nwalkers=NWALKERS), spec

    def test_counters_and_slots_follow_the_model(self):
        sch, spec = self._sched()
        s0 = pack_special_index(0, 0, 1, NWALKERS)
        s1 = pack_special_index(1, 0, 1, NWALKERS)
        sch.record_picks(np.array([s0, s1]))     # one pick each
        slot0 = int(np.flatnonzero(sch.slot_specials == s0)[0])
        slot1 = int(np.flatnonzero(sch.slot_specials == s1)[0])
        # swap t0 <-> t1: slot0's model is now labelled s1, slot1's s0
        sch.relabel_slots(np.array([slot0, slot1]), np.array([s1, s0]))
        self.assertEqual(int(sch.slot_specials[slot0]), s1)
        self.assertEqual(int(sch.slot_specials[slot1]), s0)
        # the 3-source model (now labelled s1) keeps its budget 3 / run 1
        p1 = int(sch._cells_of(np.array([s1]))[0])
        p0 = int(sch._cells_of(np.array([s0]))[0])
        self.assertEqual(int(sch.cell_counts[p1]), 3)
        self.assertEqual(int(sch.cell_run[p1]), 1)
        self.assertEqual(int(sch.cell_counts[p0]), 1)
        self.assertEqual(int(sch.cell_run[p0]), 1)
        # advance retires exactly the finished (1-source) model's slot
        fin = sch.slot_active & (
            sch.cell_run[sch.slot_cell] >= sch.cell_counts[sch.slot_cell])
        self.assertEqual(np.flatnonzero(fin).tolist(), [slot1])

    def test_unchanged_labels_are_a_noop(self):
        sch, spec = self._sched()
        before = (sch.slot_cell.copy(), sch.cell_run.copy(),
                  sch.cell_counts.copy())
        sch.relabel_slots(np.arange(3), sch.slot_specials.copy())
        np.testing.assert_array_equal(sch.slot_cell, before[0])
        np.testing.assert_array_equal(sch.cell_run, before[1])
        np.testing.assert_array_equal(sch.cell_counts, before[2])


if __name__ == "__main__":
    unittest.main()
