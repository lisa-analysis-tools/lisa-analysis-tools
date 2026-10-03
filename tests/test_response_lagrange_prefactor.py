"""pyResponseTDI's Lagrange fractional-delay prefactor is EXACT (LISAResponse.cu
``lagrange_prefactor``), not linearly interpolated from a 1001-entry table.

The table gave every delayed term a fractional-delay-dependent relative gain error up to
1.7e-7, which TDI does not cancel while it cancels the signal at low frequency: measured
(2026-10-02, TDI-2 XYZ, equal-arm, order 8, plane waves) as a relative TDI error of 5-10 % at
0.1 mHz, 0.2-0.4 % at 0.3 mHz, 6e-5-1.3e-4 at 1 mHz against the exact prefactor, and -- since
it follows the fractional delays -- a different template at every sampling step: dt 2.5 s vs
5 s disagreed by 18-39 % at 0.1 mHz and 0.7-1.6 % at 0.3 mHz (exact: 1e-7 / 1e-8). This is the
low-frequency template error behind the MBH window-decimation mismatch (5e-7..3e-6 for heavy
sources) and part of the heavy-MBH residual against the mojito data below 1 mHz.

The check: the same plane wave responded at dt 2.5 and 5 s agrees at the common samples.
Needs a rebuilt C++ module (the old table build fails it). CPU, ~15 s.
"""
import unittest

import numpy as np

T_SPAN, T_BUF, ORDER = 2 * 86400.0, 10000.0, 8


def _tdi(f, dt, orbits):
    from lisatools.response.directresponse import pyResponseTDI

    t = np.arange(0.0, T_SPAN, dt)
    h = 1e-20 * np.exp(1j * 2 * np.pi * f * t)
    resp = pyResponseTDI(1.0 / dt, len(t), orbits=orbits, order=ORDER, tdi="2nd generation",
                         tdi_chan="XYZ", force_backend="cpu")
    resp.get_projections(h, 1.0, 0.3, t_buffer=T_BUF)
    return np.asarray(resp.get_tdi_delays())


class LagrangePrefactorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits

        # the CPU backend explicitly: a bare EqualArmlengthOrbits() picks CUDA on a GPU node,
        # which pyResponseTDI(force_backend="cpu") refuses (orbits/response backend assertion)
        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")

    def test_low_frequency_tdi_does_not_depend_on_the_sampling_step(self):
        for f, bound in ((1e-4, 1e-5), (3e-4, 1e-6), (1e-3, 1e-7)):
            a = _tdi(f, 2.5, self.orbits)[:, ::2]
            b = _tdi(f, 5.0, self.orbits)
            m = min(a.shape[-1], b.shape[-1])
            sl = slice(int(2 * T_BUF / 5.0), m - int(T_BUF / 5.0))
            for c in range(3):
                rel = float(np.sqrt(np.mean((a[c, sl] - b[c, sl]) ** 2)) / np.sqrt(np.mean(b[c, sl] ** 2)))
                print(f"[lagrange prefactor] f {f * 1e3:.2f} mHz ch {c}: dt 2.5 vs 5 rel rms {rel:.2e}")
                self.assertLess(rel, bound, (f, c))


if __name__ == "__main__":
    unittest.main()
