"""A PE F-stat refit releases the installed grid before building the next.

6mo job 730 (2026-10-05): epoch 2's stacked-proposal build OOMed on rank 0
at 88.4 GB allocated while epoch 1's grid was still resident.
"""
import inspect
import unittest

from lisatools.globalfit.moves import gbspecialstretch as g


class RefitReleaseTest(unittest.TestCase):
    def test_old_grid_dropped_and_pool_freed_before_the_fit(self):
        src = inspect.getsource(g.GBSpecialRJFStatGridMove)
        i_fit = src.index("stacked, n_peaks = self._run_fstat_fit(")
        pre = src[:i_fit][-900:]
        self.assertIn("self.rj_proposal_distribution = None", pre)
        self.assertIn("self._free_inmodel_batch_pools(", pre)

    def test_the_fit_never_reads_the_old_proposal(self):
        src = inspect.getsource(g.GBSpecialRJFStatGridMove._run_fstat_fit)
        self.assertNotIn("rj_proposal_distribution", src)


if __name__ == "__main__":
    unittest.main()
