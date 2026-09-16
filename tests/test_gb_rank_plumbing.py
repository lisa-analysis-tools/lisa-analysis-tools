"""Default-preserving plumbing GB needs for the rank commands (no GPU, no build)."""

import inspect
import unittest

import numpy as np

from lisatools.globalfit.moves import gbspecialstretch as gbs


class SignatureTest(unittest.TestCase):
    def test_run_proposal_accepts_a_shipped_scan_schedule(self):
        sig = inspect.signature(gbs.GBSpecialBase.run_proposal)
        self.assertIn("scan_schedule", sig.parameters)
        self.assertIs(sig.parameters["scan_schedule"].default, None)
        self.assertEqual(sig.parameters["scan_schedule"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_run_tempering_accepts_tmp_start_and_adapt_flag(self):
        sig = inspect.signature(gbs.GBSpecialBase.run_tempering)
        self.assertIs(sig.parameters["tmp_start"].default, None)
        self.assertIs(sig.parameters["adapt_band_temps"].default, True)

    def test_update_band_leaf_caps_accepts_precomputed(self):
        sig = inspect.signature(gbs.GBSpecialBase._update_band_leaf_caps)
        self.assertIs(sig.parameters["precomputed"].default, None)
        self.assertTrue(callable(gbs.GBSpecialBase._cap_stats_local))

    def test_install_accepts_sync_shutoff(self):
        sig = inspect.signature(gbs.GBSpecialRJFStatGridMove._install)
        self.assertIs(sig.parameters["sync_shutoff"].default, True)

    def test_write_back_state_returns_inds_and_alive(self):
        src = inspect.getsource(gbs.GBSpecialBase._write_back_state)
        # both branches return the written (temp, walker, leaf) triple and the alive mask
        self.assertEqual(src.count("return inds_new, alive"), 2)


class TemperRngSeedTest(unittest.TestCase):
    def test_rank_seed_makes_the_vertical_swap_rng_deterministic(self):
        # _temper_rng is created lazily in the in-model block; exercise the factory only
        make = gbs.GBSpecialBase._make_temper_rng
        a = make(type("M", (), {"_rank_rng_seed": 123})())
        b = make(type("M", (), {"_rank_rng_seed": 123})())
        c = make(type("M", (), {"_rank_rng_seed": None})())
        self.assertEqual(a.random(), b.random())
        self.assertIsInstance(c, np.random.Generator)


if __name__ == "__main__":
    unittest.main()
