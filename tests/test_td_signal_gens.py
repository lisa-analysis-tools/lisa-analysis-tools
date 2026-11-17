"""Time-domain TDI generators: grid checks, ``TDWaveformBase.get_td_signals``, TD AnalysisContainers.

CPU only. The waveform is a stub whose ``compute_tdi_channels`` returns fixed ``(times, channels)``, so the
tests check the placement on the data grid and the consistency with the analysis-domain path
(``get_signals_for_residuals``), not a waveform model.
"""

import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import unittest

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.domains import DomainBaseArray, TDSettings, TDSignal, WDMSettings, td_grid_offset
from lisatools.sources.waveformbase import TDWaveformBase
from lisatools.utils.utility import tukey

T0 = 97729089.327664  # an awkward, non-integer data start (the mojito reference time)
DT = 5.0
N = 4096


def data_grid(n=N):
    return TDSettings(n, DT, t0=T0, force_backend="cpu")


class StubWaveform(TDWaveformBase):
    """``compute_tdi_channels`` returns the ``(times, channels)`` it was given."""

    def __init__(self, times, channels, **kwargs):
        super().__init__(
            waveform_t0=T0, data_td_settings=data_grid(), orbits="unused", force_backend="cpu", **kwargs
        )
        self.out = (times, channels)

    def compute_tdi_channels(self, *args, **kwargs):
        return self.out


def source(k0, n, seed=0):
    """A 3-channel source of ``n`` samples starting ``k0`` samples after the data start."""
    times = T0 + (k0 + np.arange(n)) * DT
    return times, np.random.default_rng(seed).standard_normal((3, n))


class GridOffsetTest(unittest.TestCase):
    def test_on_grid(self):
        self.assertEqual(td_grid_offset(T0 + 7 * DT, T0, DT), 7)
        self.assertEqual(td_grid_offset(T0 - 3 * DT, T0, DT), -3)

    def test_off_grid_raises(self):
        with self.assertRaisesRegex(ValueError, "not on the time grid"):
            td_grid_offset(T0 + 0.5 * DT, T0, DT)

    def test_td_add_rejects_an_off_grid_template(self):
        data = TDSignal(np.zeros((3, N)), data_grid())
        template = TDSignal(np.ones((3, 4)), TDSettings(4, DT, t0=T0 + 10.3 * DT))
        with self.assertRaisesRegex(ValueError, "not on the time grid"):
            data.add_signal(template)

    def test_td_add_partial_overlap(self):
        data = TDSignal(np.zeros((3, 10)), TDSettings(10, DT, t0=T0))
        data.add_signal(TDSignal(np.ones((3, 5)), TDSettings(5, DT, t0=T0 - 2 * DT)))  # samples -2..2
        data.add_signal(TDSignal(2 * np.ones((3, 4)), TDSettings(4, DT, t0=T0 + 8 * DT)), sign=-1)  # 8..11
        np.testing.assert_array_equal(data.arr[0], [1, 1, 1, 0, 0, 0, 0, 0, -2, -2])


class GetTDSignalsTest(unittest.TestCase):
    def test_native_span(self):
        times, channels = source(100, 50)
        out = StubWaveform(times, channels).get_td_signals()
        self.assertIsInstance(out, TDSignal)
        self.assertEqual((out.N, td_grid_offset(out.t0, T0, DT)), (50, 100))
        self.assertEqual(out.t0, T0 + 100 * DT)  # exactly on the grid, not the source's float time
        np.testing.assert_array_equal(out.arr, channels)

    def test_native_span_is_clipped_to_the_data(self):
        for k0, n, (start, stop) in ((-20, 50, (0, 30)), (N - 10, 50, (N - 10, N))):
            with self.subTest(k0=k0):
                times, channels = source(k0, n)
                out = StubWaveform(times, channels).get_td_signals()
                self.assertEqual((td_grid_offset(out.t0, T0, DT), out.N), (start, stop - start))
                np.testing.assert_array_equal(out.arr, channels[:, start - k0:stop - k0])

    def test_native_span_added_to_zeros_is_the_full_grid_output(self):
        times, channels = source(-20, 300)
        wf = StubWaveform(times, channels)
        full = wf.get_td_signals(td_settings=data_grid())
        placed = TDSignal(np.zeros((3, N)), data_grid()).add_signal(wf.get_td_signals())
        self.assertEqual((full.N, full.t0), (N, T0))
        np.testing.assert_array_equal(placed.arr, full.arr)

    def test_sub_span(self):
        times, channels = source(100, 300)
        wf = StubWaveform(times, channels)
        sub = TDSettings(64, DT, t0=T0 + 380 * DT)  # covers the source's tail (100..399) and beyond
        out = wf.get_td_signals(td_settings=sub)
        full = wf.get_td_signals(td_settings=data_grid())
        self.assertEqual((out.N, out.t0), (64, sub.t0))
        np.testing.assert_array_equal(out.arr, full.arr[:, 380:444])
        self.assertFalse(out.arr[:, 20:].any())  # past the source's end

    def test_off_grid_inputs_raise(self):
        times, channels = source(100, 50)
        with self.subTest("source times"):
            with self.assertRaisesRegex(ValueError, "not on the time grid"):
                StubWaveform(times + 0.3 * DT, channels).get_td_signals()
        with self.subTest("non-uniform times"):
            bad = times.copy()
            bad[-1] += DT
            with self.assertRaisesRegex(ValueError, "uniformly spaced"):
                StubWaveform(bad, channels).get_td_signals()
        wf = StubWaveform(times, channels)
        with self.subTest("td_settings t0"):
            with self.assertRaisesRegex(ValueError, "not on the time grid"):
                wf.get_td_signals(td_settings=TDSettings(16, DT, t0=T0 + 2.5 * DT))
        with self.subTest("td_settings dt"):
            with self.assertRaisesRegex(ValueError, "differs from the data dt"):
                wf.get_td_signals(td_settings=TDSettings(16, 2 * DT, t0=T0))

    def test_batch(self):
        times = T0 + (np.array([[10], [200]]) + np.arange(30)) * DT
        channels = np.random.default_rng(1).standard_normal((2, 3, 30))
        out = StubWaveform(times, channels).get_td_signals()
        self.assertIsInstance(out, DomainBaseArray)
        self.assertEqual([td_grid_offset(s.t0, T0, DT) for s in out], [10, 200])
        np.testing.assert_array_equal(out[1].arr, channels[1])


class ConsistencyWithAnalysisDomainTest(unittest.TestCase):
    """``td_gen(p)`` windowed and transformed == ``get_signals_for_residuals(p)``: TD residuals stay consistent."""

    def assertSameSignal(self, a, b):
        self.assertEqual(a.settings, b.settings)
        scale = np.abs(b.arr).max()
        self.assertLess(np.abs(a.arr - b.arr).max() / scale, 1e-12)

    def test_fd(self):
        times, channels = source(37, 900, seed=2)
        wf = StubWaveform(times, channels, freq_min=1e-4, freq_max=0.09)
        reference = wf.get_signals_for_residuals()
        td = wf.get_td_signals(td_settings=data_grid())
        window = tukey(N, alpha=wf.tukey_alpha, xp=np)
        self.assertSameSignal(td.transform(reference.settings, window=window), reference)

    def test_wdm(self):
        times, channels = source(37, 900, seed=3)
        wdm = WDMSettings(64, 64, DT, force_backend="cpu")
        wf = StubWaveform(times, channels, output_domain_settings=wdm)
        reference = wf.get_signals_for_residuals()
        td = wf.get_td_signals(td_settings=data_grid())
        window = tukey(N, alpha=wf.tukey_alpha, xp=np)
        self.assertSameSignal(td.transform(wdm, window=window), reference)


class TDAnalysisContainerTest(unittest.TestCase):
    def test_needs_no_sensitivity_for_td_data_only(self):
        AnalysisContainer(TDSignal(np.zeros((3, N)), data_grid()))
        with self.assertRaisesRegex(TypeError, "requires a ``sens_mat``"):
            AnalysisContainer(TDSignal(np.zeros((3, N)), data_grid()).transform(WDMSettings(64, 64, DT)))

    def test_build_template_places_native_spans_on_the_data_grid(self):
        spans = {1.0: (-5, 40), 2.0: (1000, 64)}  # parameter -> (start sample, length)

        def gen(amp):
            k0, n = spans[amp]
            times = T0 + (k0 + np.arange(n)) * DT
            return StubWaveform(times, amp * np.ones((3, n))).get_td_signals()

        ac = AnalysisContainer(TDSignal(np.zeros((3, N)), data_grid()), signal_gen={"mbh": gen})
        template = ac.build_template({"mbh": np.array([[1.0], [2.0]])})
        expected = np.zeros(N)
        expected[0:35] = 1.0
        expected[1000:1064] = 2.0
        self.assertEqual(template.settings, data_grid())
        np.testing.assert_array_equal(template.arr[2], expected)


if __name__ == "__main__":
    unittest.main()
