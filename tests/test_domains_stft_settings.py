"""``get_stft_settings``: the segment length must not depend on the last bits of the time stamps.

At absolute times of 1e8 s neighbouring doubles are 1.5e-8 s apart, and time stamps that went through a few
additions can be one such spacing off a regular grid. Measuring the sample step from two of them and
truncating ``big_dt / dt`` then loses a whole sample per segment (86395 s instead of 86400 s), and the settings
no longer fit the data's.
"""

import types
import unittest

import numpy as np

from lisatools.domains import STFTSettings, get_stft_settings
from lisatools.sources.waveformbase import TDWaveformBase

T0 = 107560699.827664  # a data start as in the runs: one float spacing is 1.49e-8 s here
DT = 5.0
BIG_DT = 86400.0
NPERSEG = 17280
N = 3 * NPERSEG + 100


def stamps(first_step_ulps=0):
    """``N`` time stamps from ``T0``; the first step is ``DT`` plus that many float spacings."""
    times = T0 + np.arange(N) * DT
    if first_step_ulps:
        times[1:] += first_step_ulps * np.spacing(T0)
    return times


class SampleStepTest(unittest.TestCase):
    def check(self, settings):
        self.assertEqual(settings.dt, BIG_DT)
        self.assertEqual((settings.NT, settings.NF), (3, NPERSEG // 2 + 1))
        self.assertEqual(settings.df, 1.0 / BIG_DT)
        self.assertEqual(settings.t0, T0)
        self.assertEqual(settings.sample_cadence, DT)
        self.assertEqual(settings.get_nperseg(DT), NPERSEG)

    def test_the_stamps_are_one_spacing_off(self):
        for ulps in (1, -1):
            with self.subTest(ulps=ulps):
                times = stamps(ulps)
                measured = float(times[1] - times[0])
                self.assertNotEqual(measured, DT)
                # what used to happen for a step one spacing too long: a sample lost per segment
                if ulps > 0:
                    self.assertEqual(int(BIG_DT / measured), NPERSEG - 1)
                self.check(get_stft_settings(times, big_dt=BIG_DT, dt=DT, force_backend="cpu"))

    def test_measured_step_does_not_lose_a_sample(self):
        for ulps in (1, -1):
            with self.subTest(ulps=ulps):
                settings = get_stft_settings(stamps(ulps), big_dt=BIG_DT, force_backend="cpu")
                self.assertEqual(round(settings.dt / DT), NPERSEG)
                self.assertEqual((settings.NT, settings.NF), (3, NPERSEG // 2 + 1))
                self.assertAlmostEqual(settings.dt, BIG_DT, delta=1e-3)

    def test_exact_stamps_are_unchanged(self):
        times = stamps()
        with_dt = get_stft_settings(times, big_dt=BIG_DT, min_freq=1e-4, max_freq=1e-1, dt=DT, force_backend="cpu")
        without = get_stft_settings(times, big_dt=BIG_DT, min_freq=1e-4, max_freq=1e-1, force_backend="cpu")
        self.check(with_dt)
        self.assertEqual(with_dt, without)
        # the formulas this function had before
        old_big_dt = int(BIG_DT / DT) * DT
        self.assertEqual(with_dt.dt, old_big_dt)
        self.assertEqual(with_dt.NT, int(np.floor(N / (old_big_dt / DT))))
        self.assertEqual(with_dt.NF, int(old_big_dt / DT) // 2 + 1)
        self.assertEqual((with_dt.ind_min, with_dt.ind_max), (without.ind_min, without.ind_max))

    def test_segment_is_fitted_down_to_whole_samples(self):
        times = np.arange(1000) * DT
        for big_dt, nperseg in ((640.0, 128), (642.0, 128), (644.9, 128), (645.0, 129)):
            with self.subTest(big_dt=big_dt):
                for kwargs in ({}, {"dt": DT}):
                    settings = get_stft_settings(times, big_dt=big_dt, force_backend="cpu", **kwargs)
                    self.assertEqual(settings.dt, nperseg * DT)
                    self.assertEqual(settings.dt, int(big_dt / DT) * DT)  # as before
                    self.assertEqual(settings.NT, 1000 // nperseg)

    def test_segment_shorter_than_a_sample_raises(self):
        with self.assertRaisesRegex(ValueError, "shorter than one sample"):
            get_stft_settings(np.arange(100) * DT, big_dt=2.0, dt=DT, force_backend="cpu")

    def test_factory_passes_its_sample_step(self):
        factory = STFTSettings.make_factory(BIG_DT, 1e-4, 1e-1)
        self.check(factory(stamps(1), DT, "cpu"))

    def test_waveform_output_settings_use_the_data_sample_step(self):
        # the few attributes TDWaveformBase.get_output_settings reads, on a stand-in
        waveform = types.SimpleNamespace(
            analysis_domain="STFT", nperseg=NPERSEG, dt=DT, freq_min=1e-4, freq_max=1e-1, force_backend="cpu"
        )
        for ulps in (1, -1, 0):
            with self.subTest(ulps=ulps):
                self.check(TDWaveformBase.get_output_settings(waveform, stamps(ulps)))


if __name__ == "__main__":
    unittest.main()
