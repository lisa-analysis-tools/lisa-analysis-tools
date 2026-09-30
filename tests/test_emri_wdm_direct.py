"""EMRI direct-to-WDM building blocks (Tasks A5, A7, A8).

A5: harmonic tracks (A, Phi, f, fdot, fddot) at pixel centres from a FEW sparse
holder, exact on a synthetic holder/integrator, with FEW's conventions pinned:
retrograde sign(xI0) on Phi_phi, the backwards-integration knot offset, and the
exact -m partner (-1)^l Y_{l,-m} conj(A) e^{+i Phi}.
"""

import unittest
from types import SimpleNamespace

import numpy as np


class _FakeIntegrator:
    """Phi_phi = w t + wd t^2 / 2 + wdd t^3 / 6, Phi_theta = 0, Phi_r = 0.3 w t."""

    def __init__(self, w=1e-3, wd=2e-9, wdd=3e-15):
        self.w, self.wd, self.wdd = w, wd, wdd
        self.generating_trajectory = False
        self.massratio = 1.0

    def eval_integrator_spline(self, t):
        out = np.zeros((t.size, 6))
        out[:, 3] = self.w * t + 0.5 * self.wd * t ** 2 + self.wdd * t ** 3 / 6
        out[:, 5] = 0.3 * self.w * t
        return out

    def eval_integrator_derivative_spline(self, t, order=1):
        out = np.zeros((t.size, 6))
        if order == 1:
            out[:, 3] = self.w + self.wd * t + 0.5 * self.wdd * t ** 2
            out[:, 5] = 0.3 * self.w
        elif order == 2:
            out[:, 3] = self.wd + self.wdd * t
        elif order == 3:
            out[:, 3] = self.wdd
        return out


def _fake_holder(t_knots, minus_m=False):
    teuk = (np.linspace(1.0, 2.0, t_knots.size) * np.exp(1j * 0.1 * t_knots / t_knots[-1]))[:, None]
    ylms = np.array([0.5 + 0.2j, -0.3 + 0.1j]) if minus_m else np.array([0.5 + 0.2j])
    return SimpleNamespace(t_arr=t_knots, teuk_modes=teuk, ylms=ylms,
                           ls=np.array([3]), ms=np.array([2]), ks=np.array([0]), ns=np.array([1]),
                           phases=None, freqs=None, integrate_backwards=False)


class HarmonicTrackTest(unittest.TestCase):
    def setUp(self):
        from lisatools.sources.emri.wdm_direct import harmonic_tracks_from_holder

        self.fn = harmonic_tracks_from_holder
        self.t_knots = np.linspace(0.0, 1e6, 50)
        self.t_pix = np.arange(10, 250) * 3600.0

    def _phi(self, integ, t):
        return integ.eval_integrator_spline(t)[:, 3]

    def test_phase_and_derivatives_are_exact(self):
        integ = _FakeIntegrator()
        tr = self.fn(_fake_holder(self.t_knots), integ, self.t_pix, a=0.9, xI0=1.0)[0]
        t, w = self.t_pix, integ.w
        np.testing.assert_allclose(tr.phase, 2 * self._phi(integ, t) + 0.3 * w * t, rtol=1e-12)
        np.testing.assert_allclose(tr.f, (2 * (w + integ.wd * t + 0.5 * integ.wdd * t ** 2) + 0.3 * w) / (2 * np.pi), rtol=1e-12)
        np.testing.assert_allclose(tr.fdot, 2 * (integ.wd + integ.wdd * t) / (2 * np.pi), rtol=1e-12)
        np.testing.assert_allclose(tr.fddot, 2 * integ.wdd / (2 * np.pi), rtol=1e-12)

    def test_amplitude_is_spline_of_teuk_times_ylm(self):
        from scipy.interpolate import CubicSpline

        h = _fake_holder(self.t_knots)
        tr = self.fn(h, _FakeIntegrator(), self.t_pix, a=0.9, xI0=1.0)[0]
        re = CubicSpline(h.t_arr, h.teuk_modes[:, 0].real)(self.t_pix)
        im = CubicSpline(h.t_arr, h.teuk_modes[:, 0].imag)(self.t_pix)
        np.testing.assert_allclose(tr.amp, (re + 1j * im) * h.ylms[0], rtol=1e-12)

    def test_retrograde_flips_phi_phi_sign(self):
        integ = _FakeIntegrator()
        retro = self.fn(_fake_holder(self.t_knots), integ, self.t_pix, a=0.9, xI0=-1.0)[0]
        t = self.t_pix
        np.testing.assert_allclose(retro.phase, -2 * self._phi(integ, t) + 0.3 * integ.w * t, rtol=1e-12)

    def test_backwards_offset_applied(self):
        integ = _FakeIntegrator()
        h = _fake_holder(self.t_knots)
        h.integrate_backwards = True
        tr = self.fn(h, integ, self.t_pix, a=0.9, xI0=1.0)[0]
        ph = integ.eval_integrator_spline(h.t_arr)
        offset = 2 * (ph[-1, 3] + ph[0, 3]) + 1 * (ph[-1, 5] + ph[0, 5])
        t = self.t_pix
        np.testing.assert_allclose(tr.phase, 2 * self._phi(integ, t) + 0.3 * integ.w * t + offset, rtol=1e-12)

    def test_minus_m_partner_is_few_exact(self):
        integ = _FakeIntegrator()
        h = _fake_holder(self.t_knots, minus_m=True)
        plus, minus = self.fn(h, integ, self.t_pix, a=0.9, xI0=1.0)
        self.assertEqual(minus.lmkn, (3, -2, 0, -1))
        np.testing.assert_allclose(minus.phase, -plus.phase, rtol=1e-12)
        np.testing.assert_allclose(minus.f, -plus.f, rtol=1e-12)
        np.testing.assert_allclose(minus.fdot, -plus.fdot, rtol=1e-12)
        a_plus = plus.amp / h.ylms[0]                               # the splined teuk amplitude
        np.testing.assert_allclose(minus.amp, (-1.0) ** 3 * h.ylms[1] * np.conj(a_plus), rtol=1e-12)

    def test_track_sum_equals_few_mode_sum(self):
        # sum_tracks amp e^{-i phase} == FEW's w1 + w2 at the pixel times
        integ = _FakeIntegrator()
        h = _fake_holder(self.t_knots, minus_m=True)
        tracks = self.fn(h, integ, self.t_pix, a=0.9, xI0=1.0)
        h_sum = sum(tr.amp * np.exp(-1j * tr.phase) for tr in tracks)
        A = tracks[0].amp / h.ylms[0]
        Phi = tracks[0].phase
        ref = h.ylms[0] * A * np.exp(-1j * Phi) + (-1.0) ** 3 * h.ylms[1] * np.conj(A) * np.exp(1j * Phi)
        np.testing.assert_allclose(h_sum, ref, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
