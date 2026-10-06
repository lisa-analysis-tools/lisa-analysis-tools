"""GB_CHOL_CACHE (``_CholCache`` in gbspecialstretch): host-resident in-model
proposal factors mapped to living sources, refreshed on a global ticker.

A stub move stands in for the info-matrix engine: its factor for a source is
diag(widths) with a per-source MARKER in an off-identity diagonal slot, so a
test can tell exactly whose factor a row received.
"""

import os
import unittest

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as G

NDIM = 9
SIG = np.array([0.01, 1e-6, 1e-3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


class _Sorter:
    def __init__(self, coords, walkers, temps, alive=None):
        self.coords = np.asarray(coords, float)
        self.walker_inds = np.asarray(walkers)
        self.temp_inds = np.asarray(temps)
        self.inds = np.ones(len(coords), bool) if alive is None else np.asarray(alive)


class _Move:
    name = "gb_stub"
    xp = np
    _fdot_col = None
    use_info_mat_proposal = True
    _per_leaf_fill = False
    branch_name = "gb"

    def __init__(self, obs="off"):
        self.calls = []
        self.obs = obs
        self._obs_gamma_z = None

    def _obs_eigen_mode(self):
        return self.obs

    def _eigen_axis_ready(self):
        return False

    def _compute_proposal_cholesky(self, model, bs, ids, slots=None, buffer_obj=None):
        ids = np.asarray(ids)
        self.calls.append((ids.copy(), None if slots is None else np.asarray(slots).copy()))
        self._proposal_param_scales = np.ones(NDIM)
        B = np.zeros((len(ids), NDIM, NDIM))
        B[:, np.arange(NDIM), np.arange(NDIM)] = SIG
        B[:, 3, 3] = bs.coords[ids, 3]          # the marker: col 3 is per-source
        if self.obs != "off":
            n = len(bs.inds)
            if self._obs_gamma_z is None or self._obs_gamma_z.shape[0] != n:
                self._obs_gamma_z = np.full((n, 8, 8), np.nan)
            self._obs_gamma_z[ids] = bs.coords[ids, 3][:, None, None] * np.ones((8, 8))
        return B


def _population(n=12, seed=0):
    rng = np.random.default_rng(seed)
    c = np.zeros((n, NDIM))
    c[:, 0] = 1.0
    c[:, 1] = 3.0 + 1e-3 * np.arange(n)          # f0 (mHz), well separated
    c[:, 2] = 0.3
    c[:, 3] = 100.0 + np.arange(n)               # marker
    w = np.repeat([0, 1], n // 2)
    t = np.tile([0, 1, 2], n // 3)
    return c, w, t


class CholCacheTest(unittest.TestCase):
    def setUp(self):
        self.c, self.w, self.t = _population()
        self.bs = _Sorter(self.c, self.w, self.t)
        self.m = _Move()
        self.cache = G._CholCache(("gb", "off", False))
        self.cache.every = 40
        self.assertTrue(self.cache.due(0))
        self.cache.refresh(self.m, None, self.bs)
        self.m.calls.clear()

    def _marker(self, chol):
        return chol[:, 3, 3]

    def test_refresh_then_every_source_gets_its_own_factor_without_compute(self):
        ids = np.arange(len(self.c))[::-1]
        chol = self.cache.take(self.m, None, self.bs, ids, ids + 1000, None)
        self.assertEqual(self.m.calls, [])
        np.testing.assert_array_equal(self._marker(chol), self.c[ids, 3])

    def test_map_follows_sources_through_reorder_and_accumulated_drift(self):
        """Ten steps of 0.6 x tol each: only the key TRACKING keeps the match."""
        c = self.c.copy()
        for _ in range(10):
            c[:, 1] += 0.6 * self.cache.tol * SIG[1]
            perm = np.random.default_rng(1).permutation(len(c))
            bs = _Sorter(c[perm], self.w[perm], self.t[perm])
            ids = np.arange(len(c))
            chol = self.cache.take(self.m, None, bs, ids, ids, None)
            np.testing.assert_array_equal(self._marker(chol), c[perm, 3])
        self.assertEqual(self.m.calls, [])

    def test_birth_is_computed_once_with_its_slot_then_cached(self):
        c = np.vstack([self.c, self.c[:1]])
        c[-1, 1] = 9.0                         # a new f0 nobody had
        c[-1, 3] = 777.0
        bs = _Sorter(c, np.r_[self.w, 0], np.r_[self.t, 0])
        ids = np.arange(len(c))
        chol = self.cache.take(self.m, None, bs, ids, ids + 50, None)
        self.assertEqual(len(self.m.calls), 1)
        np.testing.assert_array_equal(self.m.calls[0][0], [len(c) - 1])
        np.testing.assert_array_equal(self.m.calls[0][1], [len(c) - 1 + 50])
        self.assertEqual(chol[-1, 3, 3], 777.0)
        self.cache.take(self.m, None, bs, ids, ids, None)
        self.assertEqual(len(self.m.calls), 1)

    def test_out_of_tolerance_and_other_group_are_misses(self):
        c = self.c.copy()
        c[0, 0] += 2 * self.cache.tol * SIG[0]       # amplitude jump
        t = self.t.copy()
        t[1] = 7                                     # a rung with no entries
        bs = _Sorter(c, self.w, t)
        ids = np.arange(len(c))
        self.cache.take(self.m, None, bs, ids, ids, None)
        self.assertEqual(sorted(self.m.calls[0][0].tolist()), [0, 1])

    def test_global_ticker(self):
        self.assertFalse(self.cache.due(39))
        self.assertFalse(self.cache.due(5))          # a slower move never rewinds it
        self.assertTrue(self.cache.due(40))
        self.cache.refresh(self.m, None, self.bs)
        self.assertFalse(self.cache.due(41))
        self.assertTrue(self.cache.due(80))

    def test_refresh_rebuilds_all_alive_only(self):
        alive = np.ones(len(self.c), bool)
        alive[[2, 5]] = False
        bs = _Sorter(self.c, self.w, self.t, alive)
        self.cache.due(40)
        self.cache.refresh(self.m, None, bs)
        got = np.concatenate([ids for ids, _ in self.m.calls])
        np.testing.assert_array_equal(np.sort(got), np.nonzero(alive)[0])

    def test_observable_gamma_z_is_scattered_on_hits(self):
        m = _Move(obs="axis")
        cache = G._CholCache(("gb", "axis", False))
        cache.due(0)
        cache.refresh(m, None, self.bs)
        m._obs_gamma_z = None                        # a new propose: no stash yet
        ids = np.arange(len(self.c))
        cache.take(m, None, self.bs, ids, ids, None)
        np.testing.assert_array_equal(m._obs_gamma_z[ids, 0, 0], self.c[:, 3])

    def test_env_gate(self):
        old = os.environ.pop("GB_CHOL_CACHE", None)
        try:
            self.assertIsNone(G.GBSpecialBase._chol_cache(self.m))
            os.environ["GB_CHOL_CACHE"] = "1"
            a = G.GBSpecialBase._chol_cache(self.m)
            self.assertIs(a, G.GBSpecialBase._chol_cache(self.m))
            self.m._per_leaf_fill = True
            self.assertIsNone(G.GBSpecialBase._chol_cache(self.m))
        finally:
            os.environ.pop("GB_CHOL_CACHE", None)
            if old is not None:
                os.environ["GB_CHOL_CACHE"] = old

    def test_an_engine_error_disables_and_falls_back(self):
        def boom(*a, **k):
            raise RuntimeError("x")
        self.cache.match = boom
        ids = np.arange(len(self.c))
        old = os.environ.get("SIGHET_INFOMAT_ENGINE")
        os.environ["SIGHET_INFOMAT_ENGINE"] = "lookup"
        try:
            chol = self.cache.take(self.m, None, self.bs, ids, ids, None)
            self.assertNotIn("SIGHET_INFOMAT_ENGINE", os.environ)
        finally:
            os.environ.pop("SIGHET_INFOMAT_ENGINE", None)
            if old is not None:
                os.environ["SIGHET_INFOMAT_ENGINE"] = old
        self.assertTrue(self.cache.disabled)
        np.testing.assert_array_equal(self._marker(chol), self.c[:, 3])


if __name__ == "__main__":
    unittest.main()
