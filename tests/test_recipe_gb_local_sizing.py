"""GB/VGB builders size MOVES at the local walker block; the band tables stay global."""

import re
import unittest

from lisatools.globalfit import recipe as recipe_mod


def _function_source(name):
    src = open(recipe_mod.__file__).read()
    start = src.index(f"\ndef {name}(")
    nxt = re.search(r"\n(def |class )", src[start + 1:])
    return src[start: start + 1 + (nxt.start() if nxt else len(src))]


class GBLocalSizingTest(unittest.TestCase):
    def test_gb_builder_moves_are_local_and_band_tables_global(self):
        body = _function_source("build_gb_moves")
        self.assertIn("nwalkers_local = _local_nwalkers(acs)", body)
        # every move-level size reads the local count ...
        self.assertEqual(body.count("np.zeros((ntemps, nwalkers))"), 0)
        # the ONE remaining global-width accepted array is gb_ridge_gibbs: a plain
        # eryn move that runs head-only on the full engine state under fan-out
        self.assertEqual(body.count("np.zeros((1, nwalkers))"), 1)
        self.assertGreaterEqual(body.count("np.zeros((ntemps, nwalkers_local))"), 13)
        self.assertIn("nfriends=nwalkers_local", body)
        self.assertRegex(body, r"TemperatureControl\(\s*effective_ndim,\s*nwalkers_local")
        # ... while the state-level band tables keep the GLOBAL count
        self.assertRegex(body, r"initialize_band_information\(\s*nwalkers,")

    def test_vgb_builder_moves_are_local_and_band_tables_global(self):
        body = _function_source("build_vgb_moves")
        self.assertIn("nwalkers_local = _local_nwalkers(acs)", body)
        self.assertEqual(body.count("np.zeros((ntemps, nwalkers))"), 0)
        self.assertIn("np.zeros((ntemps, nwalkers_local))", body)
        self.assertIn("nfriends=nwalkers_local", body)
        self.assertRegex(body, r"TemperatureControl\(\s*effective_ndim,\s*nwalkers_local")
        self.assertRegex(body, r"initialize_band_information\(\s*nwalkers,")


if __name__ == "__main__":
    unittest.main()
