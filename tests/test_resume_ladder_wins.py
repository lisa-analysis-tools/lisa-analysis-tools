"""On a RESUME the per-leaf-ladder branches follow the STORE's rung count.

The 2026-09-17 relaunch crash: a store born under ``SOBBH_NTEMPS=8``
(d3c0d6ee era) was resumed with the script back at 12. The single-source
PE builder sized the move (TemperatureControl, coords_shape, betas_all)
off the configured 12 while the sub-state's coords / per-leaf log_like /
counters stayed 8-rung, and the SOBBH per-walker eigen sweep died with
``cannot reshape array of size 88 into shape (12,newaxis)`` (8 rungs x 1
walker x 11 dims against nt*nw = 12). Snapshot 7 (2026-09-16) was the
mirror case (12-rung store, config 8: mass eigen-table rebuilds).

``recipe.resume_ladder_wins`` applies the banded branches' rule
(``GBState.initialize_band_information`` step 3) to mbh/emri/sobbh: the
stored ladder wins, the configuration is reported and ignored, and the
branch info is rewritten so every later consumer reads the store's count.
"""

import logging
import types
import unittest

import numpy as np

from lisatools.globalfit.recipe import resume_ladder_wins


def _info(ndim=11, ntemps=12, betas=None):
    return types.SimpleNamespace(ndim=ndim, ntemps=ntemps, betas=betas)


def _sub(ntemps, nleaves=6, betas_all="geometric", initialized=True):
    if isinstance(betas_all, str) and betas_all == "geometric":
        row = 1.0 / (1.5 ** np.arange(ntemps))
        betas_all = np.tile(row, (nleaves, 1))
    return types.SimpleNamespace(
        tempered_initialized=initialized, ntemps=ntemps, betas_all=betas_all
    )


class ResumeLadderWinsTest(unittest.TestCase):
    def test_fresh_start_keeps_the_configuration(self):
        info = _info()
        self.assertEqual(resume_ladder_wins("sobbh", info, None, 12), (12, None))
        self.assertEqual(info.ntemps, 12)
        # A sub-state that is not tempered-initialized yet is a fresh start too.
        nt, betas = resume_ladder_wins(
            "sobbh", info, _sub(8, initialized=False), 12
        )
        self.assertEqual((nt, betas), (12, None))

    def test_matching_store_keeps_the_configuration(self):
        info = _info()
        nt, betas = resume_ladder_wins("sobbh", info, _sub(12), 12)
        self.assertEqual(nt, 12)
        self.assertIsNone(betas)
        self.assertEqual(info.ntemps, 12)

    def test_eight_rung_store_under_a_twelve_rung_config(self):
        """The relaunch crash: the store wins with its own leaf-0 ladder."""
        info = _info(ntemps=12)
        sub = _sub(8)
        with self.assertLogs("lisatools.globalfit.recipe", level="WARNING") as cm:
            nt, betas = resume_ladder_wins("sobbh", info, sub, 12)
        self.assertEqual(nt, 8)
        np.testing.assert_array_equal(betas, np.asarray(sub.betas_all)[0])
        self.assertEqual(len(betas), 8)
        # Every later consumer (run.py::_branch_ntemps reads betas first,
        # then ntemps) now sees the store's count.
        self.assertEqual(info.ntemps, 8)
        self.assertEqual(len(info.betas), 8)
        msg = "\n".join(cm.output)
        self.assertIn("STORED 8-rung", msg)
        self.assertIn("SOBBH_NTEMPS", msg)
        self.assertIn("configured 12-rung", msg)

    def test_twelve_rung_store_under_an_eight_rung_config(self):
        """Snapshot 7's case: the live ladder must NOT follow the config."""
        info = _info(ntemps=8, betas=1.0 / (2.0 ** np.arange(8)))
        nt, betas = resume_ladder_wins("sobbh", info, _sub(12), 8)
        self.assertEqual(nt, 12)
        self.assertEqual(len(betas), 12)
        self.assertEqual(len(info.betas), 12)

    def test_returned_ladder_is_a_copy(self):
        info = _info()
        sub = _sub(8)
        _, betas = resume_ladder_wins("mbh", info, sub, 12)
        betas[0] = -1.0
        self.assertNotEqual(np.asarray(sub.betas_all)[0, 0], -1.0)

    def test_unusable_stored_betas_all_falls_back_to_make_ladder(self):
        """A store whose betas_all disagrees with its own rung count."""
        info = _info(ndim=11)
        sub = _sub(8, betas_all=np.ones((6, 12)))
        nt, betas = resume_ladder_wins("emri", info, sub, 12)
        self.assertEqual(nt, 8)
        self.assertEqual(len(betas), 8)
        self.assertEqual(betas[0], 1.0)
        self.assertTrue(np.all(np.diff(betas) < 0))
        sub_none = _sub(8, betas_all=None)
        nt, betas = resume_ladder_wins("emri", info, sub_none, 12)
        self.assertEqual((nt, len(betas)), (8, 8))


if __name__ == "__main__":
    unittest.main()
