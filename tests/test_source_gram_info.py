"""``{BRANCH}_EIGEN_INFO=gram`` on the EMRI direct and MBH batched moves.

The move's Gram matrix ``<dh_a|dh_b>`` (its own batched templates, its
per-column step tuning) against an INDEPENDENT one: central differences of
the containers' full-grid generator (``_slow_gen``, the toy of
``tests/test_mbh_batched_move.py``) at the move's steps, through the
walker's container inner product. The toy depends on m1, dist, phi_ref, inc,
psi and t_plunge only (``COLS``); the other columns are exactly flat.
SOBBH (the chunked fill) is covered in ``tests/test_sobbh_chunked_move.py``.
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from lisatools.diagnostic import inner_product
from lisatools.domains import WDMSignal

try:
    from tests import test_emri_direct_move as E
    from tests import test_mbh_batched_move as M
except ImportError:  # pragma: no cover - run from tests/
    import test_emri_direct_move as E
    import test_mbh_batched_move as M

#: the columns the toy template depends on (m1, dist, phi_ref, inc, psi, t_plunge)
COLS = [0, 4, 5, 6, 7, 10]
#: stand-in prior-box widths (the step unit)
WIDTHS = 0.2 * np.abs(M.BASE_ROW) + 0.2


def _truth(acs, x0, steps, walker):
    """Independent ``<dh_a|dh_b>`` over ``COLS``: central differences of the
    full-grid generator at ``steps``, through walker ``walker``'s container
    PSD and ``inner_product``."""
    ac = acs.acs.flatten()[walker]
    dh = []
    for i in COLS:
        e = np.zeros(x0.size)
        e[i] = steps[i]
        hp, hm = M._slow_gen(*(x0 + e)), M._slow_gen(*(x0 - e))
        dh.append(WDMSignal((hp.arr - hm.arr) / (2.0 * steps[i]), hp.settings))
    _, _, psd = ac._slice_to_template(dh[0])
    G = np.zeros((len(COLS), len(COLS)))
    for a in range(len(COLS)):
        for b in range(a, len(COLS)):
            G[a, b] = G[b, a] = float(np.real(inner_product(dh[a], dh[b], psd=psd)))
    return G


def _corr_rel(a, b):
    d = np.sqrt(np.abs(np.diag(b)))
    return np.abs(a - b) / np.outer(d, d)


class _GramCase:
    """Shared Gram checks; a concrete case sets ``self.move`` / ``self.acs``
    in ``setUp`` and the agreement bound ``TOL``."""

    TOL = None

    def _info(self, walker):
        G, steps = self.move._gram_info(M.BASE_ROW.copy(), walker, WIDTHS, return_steps=True)
        return G[np.ix_(COLS, COLS)], steps

    def test_gram_matches_the_full_grid_generator_per_walker_psd(self):
        for w in (0, 1):
            G, steps = self._info(w)
            T = _truth(self.acs, M.BASE_ROW.copy(), steps, w)
            self.assertLess(_corr_rel(G, T).max(), self.TOL, f"walker {w}")
        # NEGATIVE CONTROL: walker 1's PSD differs, so its Gram must
        G1, steps = self._info(1)
        T0 = _truth(self.acs, M.BASE_ROW.copy(), steps, 0)
        self.assertGreater(_corr_rel(G1, T0).max(), 10 * self.TOL)

    def test_flat_columns_are_exactly_zero(self):
        G, _ = self.move._gram_info(M.BASE_ROW.copy(), 0, WIDTHS, return_steps=True)
        flat = [i for i in range(11) if i not in COLS]
        np.testing.assert_array_equal(G[np.ix_(flat, flat)], 0.0)

    def test_refresh_routes_through_the_gram(self):
        from types import SimpleNamespace

        from lisatools.globalfit.moves import eigen_refresh

        self.move.eigen_info = "gram"
        work = SimpleNamespace(coords=np.tile(M.BASE_ROW, (2, 3, 1, 1)))
        with mock.patch.object(self.move, "_eigen_best_walker",
                               return_value=(0, np.tile(M.BASE_ROW, (3, 1)))), \
                mock.patch.object(eigen_refresh, "prior_box_widths", return_value=WIDTHS), \
                mock.patch.object(eigen_refresh, "eigen_table_from_ll") as ll_route:
            axes, sigmas = self.move._build_eigen_table(0, work)
        ll_route.assert_not_called()
        self.assertTrue(np.all(np.isfinite(axes)) and np.all(sigmas > 0))


class EMRIDirectGramTest(_GramCase, E._Armed):
    # the fake direct adapter IS the full-grid generator: exact
    TOL = 1e-8

    def setUp(self):
        super().setUp()
        self.acs = self.move.acs
        self.move._to_phys = lambda x: np.atleast_2d(np.asarray(x, dtype=float))  # no transform

    def test_a_refused_row_falls_back_to_the_likelihood_route(self):
        from types import SimpleNamespace

        from lisatools.globalfit.moves import eigen_refresh

        self.move.eigen_info = "gram"
        self.direct.fail_with = RuntimeError("refused")
        work = SimpleNamespace(coords=np.tile(M.BASE_ROW, (2, 3, 1, 1)))
        with mock.patch.object(self.move, "_eigen_best_walker",
                               return_value=(0, np.tile(M.BASE_ROW, (3, 1)))), \
                mock.patch.object(eigen_refresh, "prior_box_widths", return_value=WIDTHS), \
                mock.patch.object(eigen_refresh, "eigen_table_from_ll",
                                  return_value=(np.eye(11), np.ones(11))) as ll_route:
            self.move._build_eigen_table(0, work)
        ll_route.assert_called_once()


class MBHBatchedGramTest(_GramCase, unittest.TestCase):
    # windowed + padded batch vs the full grid (the scorer's own parity
    # bound is 5e-2 lnL at SNR ~30)
    TOL = 2e-2

    def setUp(self):
        self.acs, self.adapter, self.fast, self.wdm = M._build()
        self.move = M._build_move(self.acs, self.adapter)
        cold = M._cold_rows()
        self.move.remove_cold_chain_sources(cold)     # expose (sets the window)
        self.move.setup_likelihood_here(cold)
        self.move._to_phys = lambda x: np.atleast_2d(np.asarray(x, dtype=float))  # no transform

    def test_no_window_raises_into_the_fallback(self):
        self.move._leaf_windows.clear()
        with self.assertRaises(RuntimeError):
            self.move._gram_info(M.BASE_ROW.copy(), 0, WIDTHS)


if __name__ == "__main__":
    unittest.main()
