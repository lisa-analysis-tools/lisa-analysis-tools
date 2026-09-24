"""The stage-B cache must not silently keep a peak list selected at another threshold.

``fstat_gridfit``'s stage-B load path is "load and return, nothing
recomputed". The peak LIST inside it was selected at whatever
``FSTAT_PEAK_MIN_SNR`` was in force when it was fitted, and nothing else on
that path re-reads the knob -- so lowering the threshold on a resume used to
be a SILENT NO-OP: the run kept proposing from the old, stricter list while
its log and its submit script both claimed the new value.

Found 2026-09-23 while lowering the 6mo run's floor 8.0 -> 6.25. Same class
of trap the ``grid_basis`` and ``band_edges`` stamps already guard against,
and given the same treatment: stamp it, refuse a mismatch.

The migration named in the error is the cheap one -- the COMB cache stores
``F_max`` for every node, so deleting only ``*_peaks_stacked.npz`` while
keeping ``*_comb.npz`` re-selects peaks at the new threshold and reruns stage
B alone, with no full refit.
"""
import os
import unittest

import numpy as np

from lisatools.sampling.fstat_gridfit import _check_cached_peak_threshold
from lisatools.sampling.fstat_proposal import fstat_peak_min_F

STACKED = "/fit/fstat_grid_peaks_stacked.npz"
COMB = "/fit/fstat_grid_comb.npz"


def _cache(min_F=None):
    """A stand-in for the loaded npz: only the stamp matters here."""
    return {} if min_F is None else {"peak_min_F": np.array(float(min_F))}


class PeakThresholdCacheStampTest(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("FSTAT_PEAK_MIN_SNR")
        os.environ.pop("FSTAT_PEAK_MIN_SNR", None)

    def tearDown(self):
        os.environ.pop("FSTAT_PEAK_MIN_SNR", None)
        if self._env is not None:
            os.environ["FSTAT_PEAK_MIN_SNR"] = self._env

    def test_matching_threshold_loads(self):
        _check_cached_peak_threshold(_cache(fstat_peak_min_F()), STACKED, COMB)

    def test_matching_threshold_loads_when_set_explicitly(self):
        os.environ["FSTAT_PEAK_MIN_SNR"] = "6.25"
        _check_cached_peak_threshold(_cache(0.5 * 6.25 ** 2), STACKED, COMB)

    def test_lowered_threshold_is_refused(self):
        """The silent no-op this exists to catch: cache at 8.0, run at 6.25."""
        os.environ["FSTAT_PEAK_MIN_SNR"] = "6.25"
        with self.assertRaises(ValueError) as cm:
            _check_cached_peak_threshold(_cache(0.5 * 8.0 ** 2), STACKED, COMB)
        msg = str(cm.exception)
        # the error has to name BOTH thresholds and the cheap migration, or
        # the reader deletes the whole epoch dir and pays for the comb again
        self.assertIn("8.0", msg)
        self.assertIn("6.25", msg)
        self.assertIn(STACKED, msg)
        self.assertIn(COMB, msg)

    def test_raised_threshold_is_refused_too(self):
        """Direction-agnostic: a stricter run must not inherit a looser list."""
        os.environ["FSTAT_PEAK_MIN_SNR"] = "8.0"
        with self.assertRaises(ValueError):
            _check_cached_peak_threshold(_cache(0.5 * 6.25 ** 2), STACKED, COMB)

    def test_unstamped_legacy_cache_warns_but_loads(self):
        """Every cache written before the stamp would otherwise be unusable."""
        os.environ["FSTAT_PEAK_MIN_SNR"] = "6.25"
        with self.assertLogs("lisatools.sampling.fstat_gridfit", level="WARNING") as log:
            _check_cached_peak_threshold(_cache(None), STACKED, COMB)
        self.assertTrue(any("peak_min_F" in line for line in log.output))

    def test_stage_b_writer_stamps_the_threshold(self):
        """The golden fixtures are the schema; both must carry the stamp."""
        import pathlib

        data = pathlib.Path(__file__).resolve().parent / "data"
        for name in ("fstat_stage_b_golden_single.npz",
                     "fstat_stage_b_golden_grouped.npz"):
            path = data / name
            if not path.exists():          # fixtures are optional in a slim tree
                continue
            with np.load(path, allow_pickle=False) as d:
                self.assertIn("peak_min_F", d.files, msg=f"{name} lost the stamp")


if __name__ == "__main__":
    unittest.main()
