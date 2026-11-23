"""Placement of time-domain templates on the data's STFT segments (``TDWaveformBase.find_bin_edges``,
``build_common_grid``) when a template starts at or before the data start, and the template with no sample
inside the data in ``_td_to_output_domain``."""

import types
import unittest

import numpy as np

from lisatools.domains import STFTSignal
from lisatools.sources.waveformbase import TDWaveformBase

DT = 5.0
NPERSEG = 16
NT = 6
T0 = 107558994.827664  # data start of a real run: neighbouring doubles are 1.49e-8 s apart here
SEGMENT = NPERSEG * DT
ULP = np.spacing(T0)


def stand_in():
    """The attributes the placement methods read, on a stand-in (no response, no waveform model)."""
    waveform = types.SimpleNamespace(
        xp=np, analysis_domain="STFT", nperseg=NPERSEG, dt=DT, data_t0=T0, force_backend="cpu",
        freq_min=0.0, freq_max=None, tukey_alpha=0.1, output_domain_settings=None, _common_grid_hint=None,
        domain_settings=types.SimpleNamespace(N=NT * NPERSEG, dt=DT, t0=T0),
        stft_t_arr=T0 + np.arange(0, NT * NPERSEG, NPERSEG) * DT,
    )
    for name in ("find_bin_edges", "build_common_grid", "_pad_td_signal", "get_output_settings"):
        setattr(waveform, name, types.MethodType(getattr(TDWaveformBase, name), waveform))
    return waveform


def stamps(first_sample, num, offset=0.0):
    """``num`` stamps on the data grid from sample ``first_sample`` (negative: before the data start)."""
    return T0 + (first_sample + np.arange(num)) * DT + offset


def reference_grid(first_sample, values, left_segment, num_segments):
    """Where the samples belong: sample ``k`` of the data at index ``k - left_segment * NPERSEG``."""
    out = np.zeros(num_segments * NPERSEG)
    for k, value in zip(first_sample + np.arange(values.size), values):
        index = k - left_segment * NPERSEG
        if k >= 0 and 0 <= index < out.size:
            out[index] = value
    return out


class BinEdgesTest(unittest.TestCase):
    def place(self, first_sample, num, offset=0.0):
        waveform = stand_in()
        times = stamps(first_sample, num, offset)[None, :]
        values = 1.0 + np.arange(num, dtype=float)
        left_edges, padded = waveform.build_common_grid(times, values[None, None, :])
        return left_edges[0], padded[0, 0], values

    def test_start_at_or_before_the_data_start_is_the_first_segment(self):
        num = 3 * NPERSEG
        for first_sample, offset in ((0, 0.0), (0, -ULP), (0, ULP), (-1, 0.0), (-3, 0.0), (-3, -ULP), (-NPERSEG - 2, 0.0)):
            with self.subTest(first_sample=first_sample, offset=offset):
                left_edge, padded, values = self.place(first_sample, num, offset)
                self.assertEqual(left_edge, T0)
                num_segments = (first_sample + num - 1) // NPERSEG + 1
                np.testing.assert_array_equal(padded, reference_grid(first_sample, values, 0, num_segments))

    def test_start_inside_the_data_is_unchanged(self):
        num = 2 * NPERSEG
        for first_sample in (NPERSEG, NPERSEG + 5, 2 * NPERSEG - 1, 3 * NPERSEG):
            with self.subTest(first_sample=first_sample):
                left_edge, padded, values = self.place(first_sample, num)
                segment = first_sample // NPERSEG
                self.assertEqual(left_edge, T0 + segment * SEGMENT)
                num_segments = (first_sample + num - 1) // NPERSEG - segment + 1
                np.testing.assert_array_equal(padded, reference_grid(first_sample, values, segment, num_segments))

    def test_rows_of_one_batch_with_different_starts(self):
        # as _apply_response leaves them: cut at the earliest start, the other rows keep stamps before the data
        waveform = stand_in()
        num = 2 * NPERSEG
        firsts = np.array([0, -1, -4])
        times = np.stack([stamps(first, num) for first in firsts])
        values = 1.0 + np.arange(num, dtype=float)
        left_edges, padded = waveform.build_common_grid(times, np.tile(values, (3, 1, 1)))
        np.testing.assert_array_equal(left_edges, np.full(3, T0))
        for row, first in enumerate(firsts):
            np.testing.assert_array_equal(padded[row, 0], reference_grid(first, values, 0, 2))

    def test_end_after_the_data_and_row_outside_the_data(self):
        waveform = stand_in()
        # ends two segments after the data: the grid stops at the last segment
        left_edges, grid_length = waveform.find_bin_edges(stamps((NT - 1) * NPERSEG, 3 * NPERSEG))
        self.assertEqual(left_edges[0], T0 + (NT - 1) * SEGMENT)
        self.assertEqual(int(grid_length[0]), NPERSEG)
        # entirely before the data: one segment at the data start, nothing in it
        left_edge, padded, _ = self.place(-4 * NPERSEG, 2 * NPERSEG)
        self.assertEqual(left_edge, T0)
        np.testing.assert_array_equal(padded, np.zeros(NPERSEG))


class EmptyTemplateTest(unittest.TestCase):
    def test_no_sample_inside_the_data_gives_a_zero_segment(self):
        waveform = stand_in()
        signal = TDWaveformBase._td_to_output_domain(waveform, times_in=np.empty(0), signal_in=np.empty((3, 0)))
        self.assertIsInstance(signal, STFTSignal)
        self.assertEqual(signal.settings.t0, T0)
        self.assertEqual(signal.settings.NT, 1)
        self.assertEqual(signal.settings.dt, SEGMENT)
        self.assertEqual(np.count_nonzero(np.asarray(signal.arr)), 0)


if __name__ == "__main__":
    unittest.main()
