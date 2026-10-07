"""A PE F-stat refit releases the installed grid before building the next.

6mo job 730 (2026-10-05): epoch 2's stacked-proposal build OOMed on rank 0
at 88.4 GB allocated while epoch 1's grid was still resident (the move's own
reference: ``RefitReleaseTest``). 6mo job 738 (2026-10-06): the same at 88.7
GB, from the process-wide epoch registries (evicted now) and the stacked
proposal's device CDF (host-resident now: ``HostCdfTest``).
"""
import inspect
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as g
from lisatools.sampling import fstat_proposal as fp


class RefitReleaseTest(unittest.TestCase):
    def test_old_grid_dropped_and_pool_freed_before_the_fit(self):
        src = inspect.getsource(g.GBSpecialRJFStatGridMove)
        i_fit = src.index("stacked, n_peaks = self._run_fstat_fit(")
        pre = src[:i_fit][-900:]
        self.assertIn("self.rj_proposal_distribution = None", pre)
        self.assertIn("self._free_inmodel_batch_pools(", pre)

    def test_registries_evicted_before_the_fit(self):
        """job 738 (2026-10-06): the process registries kept EVERY epoch."""
        src = inspect.getsource(g.GBSpecialRJFStatGridMove)
        pre = src[:src.index("stacked, n_peaks = self._run_fstat_fit(")][-1500:]
        self.assertIn("_evict_fstat_epochs(self._fstat_root, keep=self._epoch_dir(k))", pre)
        root, other = "/r/shared", "/r/shared_search"
        keys = [f"{root}/epoch_0001", f"{root}/epoch_0002", f"{other}/epoch_0001"]
        with mock.patch.dict(g._FSTAT_GRID_REGISTRY, {k: (object(), 0, 0) for k in keys},
                             clear=True), \
                mock.patch.dict(g._FSTAT_CTR_TABLE_REGISTRY, {k: {} for k in keys},
                                clear=True):
            g._evict_fstat_epochs(root, keep=f"{root}/epoch_0002")
            for reg in (g._FSTAT_GRID_REGISTRY, g._FSTAT_CTR_TABLE_REGISTRY):
                self.assertEqual(sorted(reg), [f"{root}/epoch_0002", f"{other}/epoch_0001"])

    def test_the_fit_never_reads_the_old_proposal(self):
        src = inspect.getsource(g.GBSpecialRJFStatGridMove._run_fstat_fit)
        self.assertNotIn("rj_proposal_distribution", src)


class _DeviceXP:
    """numpy standing in for cupy, except the CDF ops a device must not do."""

    def __getattr__(self, name):
        if name in ("cumsum", "searchsorted"):
            raise AssertionError(f"device xp.{name}: the CDF must live on the host")
        return getattr(np, name)


class HostCdfTest(unittest.TestCase):
    """6mo job 738 (2026-10-06): the PE refit OOMed at 88.7 GB in the
    stacked proposal's device cumsum. The CDF is host-resident now."""

    def _build(self):
        rng = np.random.default_rng(0)
        grids = rng.normal(size=(3, 5, 4, 3, 3))
        ax = [np.linspace(0, 1, n) for n in (4, 3, 3)]
        return fp.StackedFStatProposal4D(
            grids, np.array([1.0, 1.1, 1.2]), np.full(3, 0.01), *ax, seed=1)

    def test_cdf_built_and_searched_on_the_host(self):
        ref = self._build()
        ref_draws = ref.rvs(200)
        with mock.patch("lisatools.utils.utility.get_array_module",
                        return_value=_DeviceXP()):
            dev = self._build()
            draws = dev.rvs(200)
        for ch in dev._chunks:
            self.assertIs(type(ch["cdf"]), np.ndarray)
        np.testing.assert_array_equal(draws, ref_draws)
        np.testing.assert_allclose(dev.logpdf(draws), ref.logpdf(ref_draws))


if __name__ == "__main__":
    unittest.main()
