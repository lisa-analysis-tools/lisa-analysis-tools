"""The MBH harness's noise weighting includes the fitted galactic confusion foreground.

``scripts/mbh/mbh_harness.py::harness_box`` builds the EMRI harness's ``RunBox``
(scripts/emri/emri_batch_speed.py, origin/dev >= ddaad46f) on the MBH run grid; its ``sens``
weights every inner product of the speed / accuracy / data harness
(mbh_batched_gpu_benchmark.py's containers, mbh_batched_accuracy.py, mbh_cd1l_campaign.py):
``XYZ2SensitivityMatrix(dom, model="scirdv1", stochastic_params=(Tobs,))`` -- SciRD v1 XYZ
(TDI-2) plus ``FittedHyperbolicTangentGalacticForeground`` at the grid's ``Tobs = Nf * Nt *
dt`` (the stock erebor convention). Pinned on the harness grid (Nf 1440, dt 2.5 s, BAND
0.25-25 mHz) at three Tobs, 180 / 360 / 720 d -- a wide edge crop leaves 8 active
time layers so the matrices stay small: the foreground IS the fitted tanh model, raises the
PSD at 0.3-3 mHz, is absent above 10 mHz, falls with Tobs at the knee (the fitted knee moves
down as more binaries are resolved), uses the grid's own Tobs and is recorded so, ``off`` is
exactly the plain instrument, and a box that does not match the grid is refused. CPU only,
seconds. Part of the MBH unittest command::

    python -m unittest tests.test_mbh_windowed_signal_gen tests.test_mbh_batched_move \\
        tests.test_mbh_harness_noise -v
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

_MOD = Path(__file__).resolve().parents[1] / "scripts" / "mbh" / "mbh_harness.py"
_spec = importlib.util.spec_from_file_location("mbh_harness", _MOD)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

DAY = 86400.0
NF, DT = 1440, 2.5          # the harness grid: 3600-s layers, layer_df = 1.39e-4 Hz
LAYER = NF * DT
DURATIONS = (180, 360, 720)
KEEP = 8                    # active time layers left by the wide edge crop


def _grid(days, edge=None):
    """The harness WDM grid of ``days`` (Tobs = Nf * Nt * dt), cropped to KEEP time layers."""
    from lisatools.domains import WDMSettings

    nt = days * 24
    edge = (nt - KEEP) // 2 if edge is None else edge
    wdm = WDMSettings(NF, nt, DT, min_freq=2.5e-4, max_freq=2.5e-2, min_time=edge * LAYER,
                      max_time=(nt - edge) * LAYER, force_backend="cpu")
    return wdm, edge


def _xx(sens):
    """The XX PSD per active frequency layer (time layer 0: the model is stationary)."""
    return np.asarray(sens.sens_mat)[0, 0][:, 0]


class HarnessForegroundTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boxes = {}
        for d in DURATIONS:
            wdm, edge = _grid(d)
            cls.boxes[d] = (wdm, edge, harness.harness_box(wdm, edge, "on"), harness.harness_box(wdm, edge, "off"))
        cls.f = np.asarray(cls.boxes[180][0].f_arr)
        # each duration against its own instrument-only PSD (the WDM fold of the instrument
        # model differs between grids at the 1e-16 level)
        cls.off = {d: _xx(b[3].sens) for d, b in cls.boxes.items()}
        cls.fg = {d: _xx(b[2].sens) for d, b in cls.boxes.items()}

    def test_box_is_the_run_grid_and_records_its_tobs(self):
        for d, (wdm, edge, on, off) in self.boxes.items():
            self.assertEqual(on.tobs, d * DAY)                        # Tobs = Nf * Nt * dt
            self.assertEqual(tuple(np.asarray(on.sens.sens_mat).shape[-2:]), tuple(wdm.basis_shape_active))
            rec = harness.noise_record(on)
            self.assertEqual((rec["tobs_s"], rec["foreground"]), (d * DAY, True))
            self.assertIn("fitted tanh galactic foreground", rec["noise"])
            self.assertEqual(harness.noise_record(off)["foreground"], False)

    def test_off_is_the_plain_instrument(self):
        from lisatools.sensitivity import XYZ2SensitivityMatrix

        wdm, edge, on, off = self.boxes[360]
        np.testing.assert_array_equal(np.asarray(off.sens.sens_mat),
                                      np.asarray(XYZ2SensitivityMatrix(wdm, model="scirdv1").sens_mat))
        # and the instrument alone carries no Tobs: the 180-d grid's to round-off
        np.testing.assert_allclose(self.off[360], self.off[180], rtol=1e-14, atol=0)

    def test_on_is_the_fitted_tanh_foreground_at_the_grid_tobs(self):
        """``stochastic_params=(Tobs,)`` alone selects the fitted tanh model (the EMRI form)."""
        from lisatools.sensitivity import XYZ2SensitivityMatrix
        from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground

        wdm, edge, on, off = self.boxes[360]
        explicit = XYZ2SensitivityMatrix(wdm, model="scirdv1", stochastic_params=(360 * DAY,),
                                         stochastic_function=FittedHyperbolicTangentGalacticForeground)
        np.testing.assert_array_equal(_xx(explicit), self.fg[360])
        self.assertEqual(harness.FOREGROUND_MODEL, FittedHyperbolicTangentGalacticForeground.__name__)

    def test_foreground_raises_the_psd_between_0p3_and_3_mhz(self):
        band = (self.f >= 3e-4) & (self.f <= 3e-3)
        self.assertGreater(int(band.sum()), 10)
        for d, s in self.fg.items():
            ratio = s[band] / self.off[d][band]
            print(f"[harness noise] Tobs {d} d: PSD(on) / PSD(off) over 0.3-3 mHz in "
                  f"[{ratio.min():.3f}, {ratio.max():.3g}]")
            self.assertTrue(np.all(ratio > 1.0 + 1e-3), (d, float(ratio.min())))

    def test_foreground_is_absent_above_10_mhz(self):
        hi = self.f >= 1e-2
        self.assertGreater(int(hi.sum()), 10)
        for d, s in self.fg.items():
            # the tanh cut-off: >= 10 transition widths past the knee at 10 mHz
            np.testing.assert_allclose(s[hi], self.off[d][hi], rtol=1e-6, atol=0, err_msg=f"Tobs {d} d")

    def test_longer_tobs_lowers_the_foreground_at_the_knee(self):
        knee = (self.f >= 2.4e-3) & (self.f <= 3.5e-3)
        self.assertGreater(int(knee.sum()), 3)
        excess = {d: s[knee] - self.off[d][knee] for d, s in self.fg.items()}
        for short, long_ in zip(DURATIONS[:-1], DURATIONS[1:]):
            self.assertTrue(np.all(excess[short] > excess[long_]), (short, long_))

    def test_a_box_that_is_not_the_grid_is_refused(self):
        wdm, edge, _, _ = self.boxes[180]
        with self.assertRaisesRegex(ValueError, "active layers"):
            harness.harness_box(wdm, edge + 1, "on")
        for bad in ("tanh", "none", "true"):
            with self.assertRaises(ValueError):
                harness.harness_box(wdm, edge, bad)


if __name__ == "__main__":
    unittest.main()
