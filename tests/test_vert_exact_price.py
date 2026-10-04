"""Exact pricing of the vertical swap (GB_VERT_EXACT_PRICE, 2026-10-04).

The vertical swap compares whole-cell likelihoods ``L_free + ll_ref`` across
rungs. ``ll_ref`` is the sig-het RUNNING add-delta; its anchor-level error
cancels inside one rung's MH ratio but NOT between two cells. On the 6mo
replica_pe (job 715) the end-of-block audit showed block-max
|sig-het - exact| p90 40-77 nats (max 43,000) on the hot rungs while the
cold chain lost ~1,400 lnL per walker on every prior-RJ iteration.

Three layers, each against the REAL code it relies on:

* :func:`gbbands.exact_scoring_context` drives the REAL
  ``GBSignalHetComputations.get_ll_wdm`` routing: inside the context a call
  answers from the chunked delegate, outside it from the heterodyne
  reference, and ``_in_model`` is restored (also on an exception).
* ``SubBandBuffer.get_add_ll_exact`` runs the REAL ``get_add_ll`` /
  ``get_ll`` through the REAL router and WDM engine, returns the exact
  add-delta while the reference stays built, and restores the buffer's
  output stash.
* the REAL ``_run_in_model_repeats`` hands the vertical sweep the exact
  price with the knob on and the sig-het ``ll_ref`` with it off.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from gbgpu.gb_likelihood import WDMBandLikelihoodEngine
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

from lisatools.globalfit.moves import gbbands as G
from lisatools.globalfit.moves import gbspecialstretch as S
from lisatools.globalfit.moves.gbbands import pack_special_index

HET_DH_ERR = 7.0          # the sig-het d_h error the doubles carry


def _exact_dh(p):
    return 10.0 * p[:, 0]


def _exact_hh(p):
    return 4.0 * p[:, 0] ** 2 + 1.0


class _ExactDelegate:
    """The chunked engine behind a sig-het comp: exact inner products."""

    def __init__(self):
        self.calls = 0

    def get_ll_wdm(self, params, holder, data_index=None, noise_index=None,
                   **kw):
        self.calls += 1
        p = np.asarray(params, dtype=float)
        self.d_h_out = _exact_dh(p)
        self.h_h_out = _exact_hh(p)
        self.d_h_im_out = None
        return self.d_h_out - 0.5 * self.h_h_out


class _SigHetOnNumpy(GBSignalHetComputations):
    """The production class with only its backend-derived ``xp`` pinned to
    numpy (the base exposes ``xp`` through a read-only ``backend`` property);
    ``get_ll_wdm`` -- the routing under test -- is inherited untouched."""
    xp = np


def _sighet_comp(in_model=True):
    """A REAL GBSignalHetComputations (ctor bypassed) whose heterodyne
    kernel is replaced by one with a known +HET_DH_ERR error in d_h; the
    routing in ``get_ll_wdm`` is the production method."""
    c = object.__new__(_SigHetOnNumpy)
    self_check = GBSignalHetComputations.get_ll_wdm
    assert _SigHetOnNumpy.get_ll_wdm is self_check
    c.chunked = _ExactDelegate()
    c._in_model = True if in_model else None
    c._slot_to_ref_xp = np.arange(64)
    c.het_calls = 0

    def het_get_ll(params, data_index=None, phase_maximize=False):
        c.het_calls += 1
        p = np.asarray(params, dtype=float)
        c.last_d_h = _exact_dh(p) + HET_DH_ERR
        c.last_h_h = _exact_hh(p)
        c.last_d_h_im = None
        return c.last_d_h - 0.5 * c.last_h_h

    c.get_ll = het_get_ll
    return c


class _Holder:
    """Buffer double whose likelihood METHODS are SubBandBuffer's own."""

    get_ll = G.SubBandBuffer.get_ll
    get_add_ll = G.SubBandBuffer.get_add_ll
    get_add_ll_exact = G.SubBandBuffer.get_add_ll_exact

    def __init__(self, engine):
        self._likelihood_engine = engine
        self.linear_data_arr = [None]          # ONE shard: single-engine route
        self.waveform_kwargs = {}
        self.xp = np
        self.d_h_out = "dh-before"
        self.h_h_out = "hh-before"
        self.phase_angle = "phase-before"
        self.kept_out = "kept-before"

    def _to_phys(self, params, leaf_inds=None):
        return np.asarray(params, dtype=float)

    def _psd_mirror_parity_check(self, *a, **k):   # gate disarmed
        pass


def _routed(comp):
    eng = WDMBandLikelihoodEngine(comp, basis_settings=None, nchannels=3,
                                  tdi_channel_setup="XYZ")
    return G._RoutedBandEngine(eng)


class ExactScoringContextTest(unittest.TestCase):
    P = np.array([[0.5, 3.0], [1.5, 3.1], [-0.25, 2.9]])

    def test_routes_to_the_chunked_delegate_and_restores(self):
        comp = _sighet_comp(in_model=True)
        eng = SimpleNamespace(gb_comps=comp)
        slots = np.arange(3)
        het = comp.get_ll_wdm(self.P, None, data_index=slots, noise_index=slots)
        with G.exact_scoring_context([eng]):
            self.assertIsNone(comp._in_model)
            ex = comp.get_ll_wdm(self.P, None, data_index=slots,
                                 noise_index=slots)
        self.assertTrue(comp._in_model)
        np.testing.assert_allclose(ex, _exact_dh(self.P) - 0.5 * _exact_hh(self.P))
        np.testing.assert_allclose(het - ex, HET_DH_ERR)
        self.assertEqual(comp.chunked.calls, 1)
        self.assertEqual(comp.het_calls, 1)

    def test_restores_on_an_exception(self):
        comp = _sighet_comp(in_model=True)
        with self.assertRaises(RuntimeError):
            with G.exact_scoring_context([SimpleNamespace(gb_comps=comp)]):
                raise RuntimeError("boom")
        self.assertTrue(comp._in_model)

    def test_shared_comp_and_plain_engines(self):
        comp = _sighet_comp(in_model=True)
        plain = SimpleNamespace(gb_comps=SimpleNamespace(get_ll_wdm=None))
        engines = [SimpleNamespace(gb_comps=comp), SimpleNamespace(gb_comps=comp),
                   plain, SimpleNamespace()]
        with G.exact_scoring_context(engines):
            self.assertIsNone(comp._in_model)
        self.assertTrue(comp._in_model)          # deduped: restored to True
        self.assertFalse(hasattr(plain.gb_comps, "_in_model"))

    def test_router_fans_out_over_device_replicas(self):
        c0, c1 = _sighet_comp(), _sighet_comp()
        r = _routed(c0)
        r._engine_by_device[1] = SimpleNamespace(gb_comps=c1)
        with r.exact_scoring():
            self.assertIsNone(c0._in_model)
            self.assertIsNone(c1._in_model)
        self.assertTrue(c0._in_model and c1._in_model)


class GetAddLLExactTest(unittest.TestCase):
    P = np.array([[0.5, 3.0], [1.5, 3.1], [-0.25, 2.9], [0.1, 3.05]])

    def test_exact_add_delta_with_the_reference_live(self):
        comp = _sighet_comp(in_model=True)
        buf = _Holder(_routed(comp))
        slots = np.arange(4)
        ex = buf.get_add_ll_exact(self.P, slots, slots, np.full(4, 64))
        np.testing.assert_allclose(
            ex, _exact_dh(self.P) - 0.5 * _exact_hh(self.P))
        self.assertTrue(comp._in_model, "the reference must stay live")
        # the plain call still answers from the reference
        het = buf.get_add_ll(self.P, slots, slots, np.full(4, 64))
        np.testing.assert_allclose(het - ex, HET_DH_ERR)

    def test_output_stash_is_restored(self):
        buf = _Holder(_routed(_sighet_comp(in_model=True)))
        slots = np.arange(4)
        buf.get_add_ll_exact(self.P, slots, slots, np.full(4, 64))
        self.assertEqual(buf.d_h_out, "dh-before")
        self.assertEqual(buf.h_h_out, "hh-before")
        self.assertEqual(buf.phase_angle, "phase-before")
        self.assertEqual(buf.kept_out, "kept-before")

    def test_plain_engine_without_router(self):
        comp = _sighet_comp(in_model=True)
        eng = WDMBandLikelihoodEngine(comp, basis_settings=None, nchannels=3,
                                      tdi_channel_setup="XYZ")
        buf = _Holder(eng)
        slots = np.arange(4)
        ex = buf.get_add_ll_exact(self.P, slots, slots, np.full(4, 64))
        np.testing.assert_allclose(
            ex, _exact_dh(self.P) - 0.5 * _exact_hh(self.P))
        self.assertTrue(comp._in_model)


class KnobTest(unittest.TestCase):
    def test_default_off_and_env_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GB_VERT_EXACT_PRICE", None)
            self.assertFalse(S._vert_exact_price_on())
        with mock.patch.dict(os.environ, {"GB_VERT_EXACT_PRICE": "1"}):
            self.assertTrue(S._vert_exact_price_on())


# --------------------------------------------------------------------------
# the REAL _run_in_model_repeats: what price reaches the vertical sweep
# --------------------------------------------------------------------------
def _harness():
    try:
        from tests import test_inmodel_repeats as h
        from tests import test_vertical_swap as v
    except ImportError:  # discovered from inside tests/
        import test_inmodel_repeats as h
        import test_vertical_swap as v
    return h, v


OFFSET = 5.0      # sig-het add-delta = exact + OFFSET (an anchor-level error)


def _het_buffer_cls(h):
    class _SigHetBuffer(h._FakeBuffer):
        """The harness buffer with a LIVE sig-het reference: ``get_add_ll``
        (the in-model path) carries a +OFFSET error, ``get_add_ll_exact``
        is the truth."""

        def __init__(self, n):
            super().__init__(n)
            self.exact_calls = []

        def setup_in_model_likelihood(self, *a, **k):
            return True

        def get_add_ll(self, params, slots_in, slots_out, N,
                       phase_maximize=False, leaf_inds=None):
            return super().get_add_ll(params, slots_in, slots_out, N,
                                      phase_maximize, leaf_inds) + OFFSET

        def get_add_ll_exact(self, params, slots_in, slots_out, N,
                             leaf_inds=None):
            out = h._fake_ll(np.asarray(params))
            self.exact_calls.append((np.asarray(params).copy(), out.copy()))
            return out

    return _SigHetBuffer


class SweepPriceWiringTest(unittest.TestCase):
    def _run(self, exact, at_refit, n_rep=6, seed=11):
        h, v = _harness()
        NT, NW, NB = v.NTEMPS, v.NWALKERS, v.NBANDS
        n_src = 8
        t = np.arange(NT)
        w = np.zeros(NT, dtype=int)
        b = np.ones(NT, dtype=int)
        ids = np.arange(NT)
        rng = np.random.RandomState(seed)
        coords = np.zeros((n_src, 4))
        coords[:, 0] = rng.uniform(-0.5, 0.5, n_src)
        coords[:, 1] = rng.uniform(2.95, 3.05, n_src)
        coords[:, 2] = rng.uniform(-1, 1, n_src)
        coords[:, 3] = rng.uniform(-1, 1, n_src)
        picked = {
            "ids": ids, "specials": pack_special_index(t, w, b, NW),
            "slot_index": np.arange(NT, dtype=np.int32),
            "temp_inds": t.copy(), "walker_inds": w.copy(),
            "band_inds": b.copy(), "N_vals": np.full(NT, 64),
        }
        sorter = v._FakeSorter(t, w, b, NW)
        sorter.inds = np.ones(n_src, dtype=bool)
        sorter.coords = coords.copy()
        sorter.leaf_inds = np.arange(n_src)

        mv = h._make_move(n_rep)
        mv.ntemps, mv.nwalkers, mv.num_bands = NT, NW, NB
        mv.temper_vertical = True
        mv._temper_rng = np.random.default_rng(5)
        mv.sequential_parity_repeats = False

        prices = []
        orig = mv._vertical_swap_sweep

        def rec(*args, **kw):
            prices.append(np.asarray(args[7], dtype=float).copy())
            return orig(*args, **kw)

        mv._vertical_swap_sweep = rec
        buf = _het_buffer_cls(h)(NT)
        ll_change = np.zeros((NT, NW, NB))
        prop = np.zeros((2, NT, NW, NB), dtype=int)
        acc = np.zeros_like(prop)
        cell_ll = {
            "spec": sorter.special_band_inds.copy(),
            "ll0": np.zeros(NT), "led0": np.zeros(NT),
            "rep0": np.zeros(NT, dtype=int),
        }
        env = {"GB_VERT_EXACT_PRICE": "1" if exact else "0",
               "GB_TEMPER_VERTICAL_AT_REFIT": "1" if at_refit else "0",
               "GB_TEMPER_ALL_RUNGS": "0"}
        np.random.seed(seed)
        with mock.patch.dict(os.environ, env):
            mv._run_in_model_repeats(
                None, sorter, buf, v._ladder(), picked, ll_change, prop, acc,
                num_repeats=n_rep, cell_ll_state=cell_ll,
            )
        return prices, buf

    def test_knob_on_every_sweep_is_priced_exactly(self):
        prices, buf = self._run(exact=True, at_refit=False)
        self.assertGreater(len(prices), 1, "per-repeat sweeps expected")
        self.assertEqual(len(buf.exact_calls), len(prices),
                         "one exact batched call per sweep")
        for px, (_, ex) in zip(prices, buf.exact_calls):
            np.testing.assert_allclose(px, ex)

    def test_knob_on_opening_sweep_at_refit(self):
        prices, buf = self._run(exact=True, at_refit=True)
        self.assertEqual(len(prices), 1, "at-refit with no refresh: the "
                         "opening sweep only")
        np.testing.assert_allclose(prices[0], buf.exact_calls[0][1])

    def test_knob_off_prices_with_the_sighet_running_value(self):
        prices, buf = self._run(exact=False, at_refit=False)
        self.assertGreater(len(prices), 1)
        self.assertEqual(buf.exact_calls, [], "no exact call with the knob off")
        # the running value carries the +OFFSET sig-het error
        h, _ = _harness()
        p0 = prices[0]
        self.assertTrue(np.all(np.isfinite(p0)))

    def test_on_and_off_prices_differ_by_the_sighet_error(self):
        on, buf_on = self._run(exact=True, at_refit=True)
        off, _ = self._run(exact=False, at_refit=True)
        # same seed, same block-open coordinates -> the opening sweep's
        # price differs by exactly the anchor-level error
        np.testing.assert_allclose(off[0] - on[0], OFFSET)


if __name__ == "__main__":
    unittest.main()
