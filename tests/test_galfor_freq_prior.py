"""The galfor frequency-prior override (``GALFOR_FREQ_PRIOR`` / ``freq_range``).

User ruling 2026-10-08 for the 9mo and 1yr runs: "adjust the frequency
parameters (fk, f1, f2) in the foreground model['s prior] to go from 1e-4 to
1e-2". The stock support is fk (0.8 mHz, 10 mHz), f_1 and f_2 (10 uHz, 10 mHz);
the knob moves the THREE frequency columns to one shared ``[lo, hi]`` box and
touches nothing else. The default leaves every bound bit-identical (the 6mo
relaunch keeps the stock prior), and the two places that re-derive the support
outside the prior dict -- the start-pin window in ``run.py`` and the
``warmstart.noise_pin`` refusal -- must see the same box, or a pin inside the
new prior would be refused by the old one.
"""
import os
import unittest

import numpy as np

from lisatools.globalfit.stock.erebor.noise import (
    GALFOR_BASIS, GALFOR_FREQ_PARAMS, GALFOR_PRIOR_RANGE, galfor_prior_dict,
    galfor_prior_ranges)

FREQ_COLS = [GALFOR_BASIS.index(n) for n in GALFOR_FREQ_PARAMS]
OTHER_COLS = [i for i in range(len(GALFOR_BASIS)) if i not in FREQ_COLS]


class _EnvMixin:
    KNOBS = ("GALFOR_FREQ_PRIOR", "GALFOR_ALPHA_MAX", "GALFOR_FK_PRIOR",
             "GALFOR_F1_PRIOR", "GALFOR_F2_PRIOR")

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.KNOBS}

    def tearDown(self):
        for k in self.KNOBS:
            os.environ.pop(k, None)
            if self._saved[k] is not None:
                os.environ[k] = self._saved[k]


class GalforFreqPriorTest(_EnvMixin, unittest.TestCase):
    def test_the_three_frequency_columns_are_fk_f1_f2(self):
        self.assertEqual(GALFOR_FREQ_PARAMS, ("fk", "f_1", "f_2"))

    def test_default_is_the_stock_support(self):
        self.assertEqual(galfor_prior_ranges(), tuple(tuple(map(float, r)) for r in GALFOR_PRIOR_RANGE))
        d = galfor_prior_dict()
        self.assertEqual((d[GALFOR_BASIS.index("fk")].minimum, d[GALFOR_BASIS.index("fk")].maximum), (0.8e-3, 1e-2))
        self.assertEqual((d[GALFOR_BASIS.index("f_1")].minimum, d[GALFOR_BASIS.index("f_1")].maximum), (1e-5, 1e-2))
        self.assertEqual((d[GALFOR_BASIS.index("f_2")].minimum, d[GALFOR_BASIS.index("f_2")].maximum), (1e-5, 1e-2))

    def test_env_moves_only_the_frequency_columns(self):
        os.environ["GALFOR_FREQ_PRIOR"] = "1e-4,1e-2"
        base = tuple(tuple(map(float, r)) for r in GALFOR_PRIOR_RANGE)
        rngs = galfor_prior_ranges()
        for i in FREQ_COLS:
            self.assertEqual(rngs[i], (1e-4, 1e-2), GALFOR_BASIS[i])
        for i in OTHER_COLS:
            self.assertEqual(rngs[i], base[i], GALFOR_BASIS[i])
        d = galfor_prior_dict()
        for i in FREQ_COLS:
            self.assertEqual((d[i].minimum, d[i].maximum), (1e-4, 1e-2), GALFOR_BASIS[i])
        for i in OTHER_COLS:
            self.assertEqual((d[i].minimum, d[i].maximum), base[i], GALFOR_BASIS[i])

    def test_kwarg_beats_env_and_tolerates_spaces(self):
        os.environ["GALFOR_FREQ_PRIOR"] = " 2e-4 , 5e-3 "
        self.assertEqual(galfor_prior_ranges()[FREQ_COLS[0]], (2e-4, 5e-3))
        self.assertEqual(galfor_prior_ranges(freq_range=(1e-4, 1e-2))[FREQ_COLS[0]], (1e-4, 1e-2))
        self.assertEqual(galfor_prior_dict(freq_range=(1e-4, 1e-2))[FREQ_COLS[1]].maximum, 1e-2)

    def test_log_sampling_maps_the_new_box_to_log10(self):
        d = galfor_prior_dict(log_sampling=True, freq_range=(1e-4, 1e-2))
        for i in FREQ_COLS:
            self.assertAlmostEqual(d[i].minimum, -4.0, places=12)
            self.assertAlmostEqual(d[i].maximum, -2.0, places=12)
        self.assertEqual(d[GALFOR_BASIS.index("alpha")].maximum, 5.0)      # alpha stays linear

    def test_composes_with_the_alpha_cap(self):
        os.environ["GALFOR_FREQ_PRIOR"] = "1e-4,1e-2"
        os.environ["GALFOR_ALPHA_MAX"] = "20"
        rngs = galfor_prior_ranges()
        self.assertEqual(rngs[GALFOR_BASIS.index("alpha")], (1e-3, 20.0))
        self.assertEqual(rngs[GALFOR_BASIS.index("fk")], (1e-4, 1e-2))

    def test_bad_values_are_refused(self):
        for bad in ("1e-2,1e-4", "0,1e-2", "-1e-4,1e-2", "1e-4", "a,b", "1e-4,1e-2,3"):
            with self.subTest(bad=bad):
                os.environ["GALFOR_FREQ_PRIOR"] = bad
                with self.assertRaises(ValueError):
                    galfor_prior_ranges()
        os.environ.pop("GALFOR_FREQ_PRIOR", None)
        with self.assertRaises(ValueError):
            galfor_prior_ranges(freq_range=(1e-2, 1e-4))

    def test_the_9mo_start_pin_sits_inside_the_new_box(self):
        # the 6mo/9mo GALFOR_START_PARAMS (physical): fk 2.53 mHz, f_1 10 mHz, f_2 1.41 mHz
        pin = [1.436180605904e-44, 2.533915614978e-03, 5.0, 1.0e-02, 1.405721657329e-03]
        rngs = galfor_prior_ranges(freq_range=(1e-4, 1e-2))
        for i, (lo, hi) in enumerate(rngs):
            self.assertTrue(lo <= pin[i] <= hi, (GALFOR_BASIS[i], pin[i], lo, hi))


class GalforPerColumnPriorTest(_EnvMixin, unittest.TestCase):
    """User ruling 2026-10-09: "adjust the foreground prior for f_k and f_2 to go
    from 0.8 mHz to 10 mHz. Keep f_1 as is." -- per-column knobs that override
    the shared box for ONE column each."""

    FK, F1, F2 = (GALFOR_BASIS.index(n) for n in ("fk", "f_1", "f_2"))

    def test_the_9mo_box(self):
        # the 9mo / 1yr launchers: f_1 keeps the 10-08 box, fk and f_2 take 0.8-10 mHz
        os.environ["GALFOR_FK_PRIOR"] = "0.8e-3,1e-2"
        os.environ["GALFOR_F1_PRIOR"] = "1e-4,1e-2"
        os.environ["GALFOR_F2_PRIOR"] = "0.8e-3,1e-2"
        r = galfor_prior_ranges()
        self.assertEqual(r[self.FK], (0.8e-3, 1e-2))
        self.assertEqual(r[self.F1], (1e-4, 1e-2))
        self.assertEqual(r[self.F2], (0.8e-3, 1e-2))
        base = tuple(tuple(map(float, x)) for x in GALFOR_PRIOR_RANGE)
        for i in OTHER_COLS:
            self.assertEqual(r[i], base[i], GALFOR_BASIS[i])

    def test_a_column_knob_beats_the_shared_box_for_its_column_only(self):
        os.environ["GALFOR_FREQ_PRIOR"] = "1e-4,1e-2"
        os.environ["GALFOR_F2_PRIOR"] = "0.8e-3,1e-2"
        r = galfor_prior_ranges()
        self.assertEqual(r[self.FK], (1e-4, 1e-2))
        self.assertEqual(r[self.F1], (1e-4, 1e-2))
        self.assertEqual(r[self.F2], (0.8e-3, 1e-2))

    def test_kwargs_beat_env_and_reach_the_prior_dict(self):
        os.environ["GALFOR_FK_PRIOR"] = "2e-4,5e-3"
        self.assertEqual(galfor_prior_ranges(fk_range=(0.8e-3, 1e-2))[self.FK], (0.8e-3, 1e-2))
        d = galfor_prior_dict(f2_range=(0.8e-3, 1e-2))
        self.assertEqual((d[self.F2].minimum, d[self.F2].maximum), (0.8e-3, 1e-2))
        d = galfor_prior_dict(log_sampling=True, f2_range=(0.8e-3, 1e-2))
        self.assertAlmostEqual(d[self.F2].minimum, np.log10(0.8e-3), places=12)

    def test_bad_column_values_are_refused(self):
        for bad in ("1e-2,0.8e-3", "0,1e-2", "1e-3"):
            os.environ["GALFOR_F2_PRIOR"] = bad
            with self.assertRaises(ValueError):
                galfor_prior_ranges()
        os.environ.pop("GALFOR_F2_PRIOR", None)
        with self.assertRaises(ValueError):
            galfor_prior_ranges(fk_range=(1e-2, 0.8e-3))

    def test_the_9mo_start_pin_sits_inside_the_column_box(self):
        # fk 2.53 mHz, f_1 10 mHz, f_2 1.41 mHz (physical) vs fk/f_2 0.8-10 mHz, f_1 0.1-10 mHz
        pin = [1.436180605904e-44, 2.533915614978e-03, 5.0, 1.0e-02, 1.405721657329e-03]
        r = galfor_prior_ranges(fk_range=(0.8e-3, 1e-2), f1_range=(1e-4, 1e-2), f2_range=(0.8e-3, 1e-2))
        for i, (lo, hi) in enumerate(r):
            self.assertTrue(lo <= pin[i] <= hi, (GALFOR_BASIS[i], pin[i], lo, hi))

    def test_both_pin_windows_follow_the_column_knobs(self):
        from lisatools.globalfit.run import GlobalFit
        from lisatools.globalfit.warmstart.noise_pin import _windows

        os.environ["GALFOR_F2_PRIOR"] = "0.8e-3,1e-2"
        self.assertEqual(_windows("galfor"), galfor_prior_ranges())
        self.assertEqual(GlobalFit._noise_pin_window("galfor"), galfor_prior_ranges())
        self.assertEqual(_windows("galfor")[self.F2], (0.8e-3, 1e-2))


class SupportWindowsShareTheKnobTest(_EnvMixin, unittest.TestCase):
    """The start-pin window (run.py) and the noise-pin refusal must read the
    SAME box as the prior dict, knob included."""

    def test_noise_pin_windows_follow_the_knob(self):
        from lisatools.globalfit.warmstart.noise_pin import _windows

        self.assertEqual(_windows("galfor"), galfor_prior_ranges())
        os.environ["GALFOR_FREQ_PRIOR"] = "1e-4,1e-2"
        w = _windows("galfor")
        self.assertEqual(w, galfor_prior_ranges())
        for i in FREQ_COLS:
            self.assertEqual(w[i], (1e-4, 1e-2))

    def test_run_start_pin_window_follows_the_knob(self):
        from lisatools.globalfit.run import GlobalFit

        win = GlobalFit._noise_pin_window
        self.assertEqual(win("galfor"), galfor_prior_ranges())
        os.environ["GALFOR_FREQ_PRIOR"] = "1e-4,1e-2"
        w = win("galfor")
        self.assertEqual(w, galfor_prior_ranges())
        for i in FREQ_COLS:
            self.assertEqual(w[i], (1e-4, 1e-2))
        # psd untouched, unknown branch unchecked
        self.assertTrue(len(win("psd")) > 0)
        self.assertEqual(win("sgwb"), ())

    def test_a_pin_inside_the_new_box_but_outside_the_old_one_is_not_refused(self):
        # fk = 0.5 mHz: outside the stock (0.8 mHz, 10 mHz), inside (1e-4, 1e-2)
        from lisatools.globalfit.warmstart.noise_pin import _windows

        fk = GALFOR_BASIS.index("fk")
        self.assertFalse(_windows("galfor")[fk][0] <= 5e-4 <= _windows("galfor")[fk][1])
        os.environ["GALFOR_FREQ_PRIOR"] = "1e-4,1e-2"
        lo, hi = _windows("galfor")[fk]
        self.assertTrue(lo <= 5e-4 <= hi)


if __name__ == "__main__":
    unittest.main()
