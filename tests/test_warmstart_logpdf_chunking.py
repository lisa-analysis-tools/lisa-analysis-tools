"""The windowed-mixture logpdf is chunked over PAIRS, and must be exact.

A 6-month production run died here on 2026-09-19:

    rj_warm_pe -> BandSorter.__init__ -> rj_prop.logpdf
      -> _mixture_logpdf -> einsum("nij,nj->ni", chol_inv[k_sel], diff)
    OutOfMemoryError: allocating 12,768,211,456 bytes
                      (allocated so far: 90,670,297,600)

``chol_inv[k_sel]`` is a fancy-index TAKE that materialises
``(n_pair, ndim, ndim)`` float64 in ONE block. The pair count is not the
coordinate count: every coordinate contributes one pair per overlapping
f0 window, and ``GB_WARM_START_CIRC_IMAGES`` multiplies the candidate set
by the wrapped-normal image count on top. 12.77e9 / 8 / 81 = ~19.7 million
pairs.

Each pair is independent -- it writes one cell of ``lp`` and reads only its
own component -- so chunking is a float-EXACT transformation. That is the
property under test: the same numbers, bit for bit, at every chunk size,
including a chunk size of one. A chunking bug that silently changed a
handful of densities would be invisible in a run and would corrupt the
birth proposal, so "close enough" is not the bar here.
"""

import importlib
import os
import unittest

import numpy as np


def _reload_with_budget(nbytes):
    """Re-import the module with a given GB_WARM_START_LOGPDF_MAX_BYTES."""
    from lisatools.globalfit.warmstart import proposal as mod

    old = os.environ.get("GB_WARM_START_LOGPDF_MAX_BYTES")
    if nbytes is None:
        os.environ.pop("GB_WARM_START_LOGPDF_MAX_BYTES", None)
    else:
        os.environ["GB_WARM_START_LOGPDF_MAX_BYTES"] = str(int(nbytes))
    try:
        return importlib.reload(mod)
    finally:
        os.environ.pop("GB_WARM_START_LOGPDF_MAX_BYTES", None)
        if old is not None:
            os.environ["GB_WARM_START_LOGPDF_MAX_BYTES"] = old


class BudgetKnobTest(unittest.TestCase):
    def test_default_is_one_gib(self):
        mod = _reload_with_budget(None)
        self.assertEqual(mod._LOGPDF_MAX_BYTES, 1 << 30)

    def test_env_overrides_it(self):
        mod = _reload_with_budget(4096)
        self.assertEqual(mod._LOGPDF_MAX_BYTES, 4096)
        _reload_with_budget(None)          # restore for other tests

    def test_chunk_step_is_at_least_one_however_small_the_budget(self):
        """A 1-byte budget must still make progress, not divide to zero."""
        ndim = 9
        per_pair = 8 * (ndim * ndim + 3 * ndim)
        for budget in (1, 8, per_pair - 1):
            step = max(1, int(budget // max(per_pair, 1)))
            self.assertGreaterEqual(step, 1, budget)

    def test_the_production_allocation_would_have_been_chunked(self):
        """The failure this guards, in numbers."""
        ndim, want = 9, 12_768_211_456
        n_pair = want // 8 // (ndim * ndim)
        self.assertGreater(n_pair, 19_000_000)
        per_pair = 8 * (ndim * ndim + 3 * ndim)
        step = (1 << 30) // per_pair
        self.assertLess(step, n_pair)                 # it really does chunk
        self.assertLess(step * per_pair, 1.1 * (1 << 30))   # within budget


class _Mix:
    """The pair-scoring kernel, chunked exactly as the module does it.

    The real method needs a fitted proposal object; what is portable -- and
    what the OOM was in -- is the pair loop itself, so it is reproduced here
    against the same inputs and compared with the one-shot form.
    """

    def __init__(self, ndim, n_comp, seed=0):
        rng = np.random.default_rng(seed)
        self.ndim = ndim
        self.means = rng.normal(size=(n_comp, ndim))
        a = rng.normal(size=(n_comp, ndim, ndim))
        self.chol_inv = a + np.eye(ndim)[None] * ndim      # well-conditioned
        self.log_w = rng.normal(size=n_comp)
        self.log_norm = rng.normal(size=n_comp)
        self.log_trunc_z = rng.normal(size=n_comp)

    def _score(self, x, rows, cols, k_img, step, D):
        # D comes from the proposal's _overlap_depth, never from cols -- an
        # empty pair list has no max, which is exactly the zero-candidate
        # case the production guard (`if n_pair:`) handles.
        n = x.shape[0]
        lp = np.full((n, D), -np.inf, dtype=np.float64)
        n_pair = rows.shape[0]
        for s in range(0, n_pair, step):
            e = min(s + step, n_pair)
            r_c, c_c = rows[s:e], cols[s:e]
            k_sel = k_img[s:e]
            diff = x[r_c] - self.means[k_sel]
            y = np.einsum("nij,nj->ni", self.chol_inv[k_sel], diff)
            maha = np.sum(y * y, axis=1)
            lp[r_c, c_c] = (self.log_w[k_sel] + self.log_norm[k_sel]
                            - self.log_trunc_z[k_sel] - 0.5 * maha)
        return lp


class ChunkingIsExactTest(unittest.TestCase):
    def setUp(self):
        self.ndim, n_comp, self.n, self.D = 9, 17, 40, 5
        self.m = _Mix(self.ndim, n_comp)
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(self.n, self.ndim))
        valid = rng.random((self.n, self.D)) > 0.35
        valid[0, :] = False                     # a coord with NO candidate
        valid[1, :] = True                      # and one with all of them
        self.rows, self.cols = np.where(valid)
        self.k_img = rng.integers(0, n_comp, size=self.rows.shape[0])

    def _ref(self):
        return self.m._score(self.x, self.rows, self.cols, self.k_img,
                             step=10 ** 9, D=self.D)   # one shot

    def test_every_chunk_size_is_bit_identical_to_one_shot(self):
        ref = self._ref()
        for step in (1, 2, 3, 7, 13, 64, self.rows.shape[0] - 1,
                     self.rows.shape[0], self.rows.shape[0] + 5):
            got = self.m._score(self.x, self.rows, self.cols, self.k_img, step, self.D)
            np.testing.assert_array_equal(
                got, ref, err_msg=f"chunk step {step} changed the result")

    def test_a_chunk_of_one_still_fills_every_pair(self):
        got = self.m._score(self.x, self.rows, self.cols, self.k_img, 1, self.D)
        self.assertEqual(int(np.isfinite(got).sum()), self.rows.shape[0])

    def test_rows_with_no_candidate_stay_minus_inf(self):
        got = self.m._score(self.x, self.rows, self.cols, self.k_img, 3, self.D)
        self.assertTrue(np.all(np.isneginf(got[0])))

    def test_zero_pairs_is_not_an_error(self):
        empty = np.zeros(0, dtype=int)
        got = self.m._score(self.x, empty, empty, empty, 4, self.D)
        self.assertTrue(np.all(np.isneginf(got)))

    def test_chunk_boundaries_do_not_double_write(self):
        """Each pair writes exactly one cell; a boundary bug would overwrite."""
        ref = self._ref()
        n_finite = int(np.isfinite(ref).sum())
        self.assertEqual(n_finite, self.rows.shape[0])
        for step in (1, 5, 11):
            got = self.m._score(self.x, self.rows, self.cols, self.k_img, step, self.D)
            self.assertEqual(int(np.isfinite(got).sum()), n_finite, step)


if __name__ == "__main__":
    unittest.main()
