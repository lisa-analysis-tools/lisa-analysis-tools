"""Reference-epoch vs coalescence phase convention of the SOBBH TaylorT3 TD waveform."""

# Pre-existing env shims (see tests/test_l2d_ports.py): the installed scienceplots raises at import,
# and eryn.utils.plot uses a bare ``typing`` name. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import unittest

import numpy as np

from lisatools.sources.sobbh.waveform import (
    _pn_amp_phase_core,
    _pn_amp_phase_masked,
    _pn_intrinsics,
    phase,
    tau_to_x,
    waveform_generate_h_plus_cross,
)

try:
    from pygwtf.models.taylort3spin.common import _get_phi_f_fdot

    HAVE_PYGWTF = True
except ImportError:
    HAVE_PYGWTF = False

T_PN = np.arange(0.0, 3.0e7, 5.0e4)  # PN time from the reference epoch (s)


def _sources(n=20, seed=3):
    rng = np.random.default_rng(seed)
    m = rng.uniform(10.0, 60.0, (n, 2))
    return dict(
        m1=m.max(axis=1), m2=m.min(axis=1),
        s1=rng.uniform(-0.9, 0.9, n), s2=rng.uniform(-0.9, 0.9, n),
        D=rng.uniform(0.1, 1.0, n) * 1e9,  # pc
        inc=rng.uniform(0.0, np.pi, n),
        f_low=10 ** rng.uniform(-2.3, -1.9, n),  # tc of a few yr or more: no merger on T_PN
        phi0=rng.uniform(0.0, np.pi, n),
    )


def _intrinsics(m1, m2, f_low, s1, s2):
    """``(M, eta, sigma, delta, s, tc, x_ref)`` as plain floats; ``x_ref`` is x at the reference epoch."""
    M, eta, sigma, s, delta, tc = (float(v) for v in _pn_intrinsics(m1, m2, f_low, s1, s2))
    x_ref = float(tau_to_x(eta * tc / (5 * M), sigma, delta, eta, s))
    return M, eta, sigma, delta, s, tc, x_ref


def _phase_at_reference(m1, m2, f_low, s1, s2):
    """``phase(x_ref)``: phi_coal = phi0 + phase(x_ref) maps the reference-epoch phase to the coalescence phase."""
    M, eta, sigma, delta, s, tc, x_ref = _intrinsics(m1, m2, f_low, s1, s2)
    return float(phase(x_ref, sigma, delta, eta, s))


class PhaseConventionTest(unittest.TestCase):
    def setUp(self):
        self.src = _sources()

    def _rows(self):
        d = self.src
        return zip(d["m1"], d["m2"], d["s1"], d["s2"], d["D"], d["inc"], d["f_low"], d["phi0"])

    def test_default_is_reference_epoch_formula(self):
        """Flag off: the pre-flag formula ``phi0 - phase(x) + phase(x_ref)``, anchored at the reference epoch.

        The phase is a small difference of ~1e8 rad terms, so an independent re-evaluation agrees to
        ~1e-8 rad, not bitwise.
        """
        for m1, m2, s1, s2, D, _, f_low, phi0 in self._rows():
            M, eta, sigma, delta, s, tc, x_ref = _intrinsics(m1, m2, f_low, s1, s2)
            x = tau_to_x(eta * (tc - T_PN) / (5 * M), sigma, delta, eta, s)
            ref = phi0 - phase(x, sigma, delta, eta, s) + phase(x_ref, sigma, delta, eta, s)
            _, Phi, _ = _pn_amp_phase_core(m1, m2, D, f_low, s1, s2, T_PN, reference_phase=phi0)
            np.testing.assert_allclose(np.asarray(Phi), np.asarray(ref), rtol=0, atol=1e-6)
            self.assertAlmostEqual(float(Phi[0]), phi0, delta=1e-6)

    def test_coalescence_mode_reproduces_reference_mode(self):
        """phi_coal = phi0 + phase(x_ref) gives the same polarizations in both conventions."""
        for m1, m2, s1, s2, D, inc, f_low, phi0 in self._rows():
            phi_coal = phi0 + _phase_at_reference(m1, m2, f_low, s1, s2)
            hp, hx, _ = waveform_generate_h_plus_cross(m1, m2, D, inc, f_low, s1, s2, T_PN, phi0)
            hp_c, hx_c, _ = waveform_generate_h_plus_cross(
                m1, m2, D, inc, f_low, s1, s2, T_PN, phi_coal, phase_at_coalescence=True
            )
            # phase(x_ref) ~ 1e7-1e8 rad: float64 cancellation leaves ~1e-8 rad
            scale = float(np.max(np.abs(hp)) + np.max(np.abs(hx)))
            np.testing.assert_allclose(np.asarray(hp_c), np.asarray(hp), rtol=0, atol=1e-7 * scale)
            np.testing.assert_allclose(np.asarray(hx_c), np.asarray(hx), rtol=0, atol=1e-7 * scale)

    @unittest.skipUnless(HAVE_PYGWTF, "pygwtf not installed")
    def test_catalogue_phase_mapping(self):
        """``sobhb_coalescence_phase`` (pygwtf-based) maps the reference-epoch phase to the TD coalescence phase."""
        from lisatools.sources.utils import sobhb_coalescence_phase

        d = self.src
        phi_coal = sobhb_coalescence_phase(d["m1"], d["m2"], d["s1"], d["s2"], d["f_low"], d["phi0"])
        self.assertTrue(np.all((phi_coal >= 0) & (phi_coal < 2 * np.pi)))
        for i, (m1, m2, s1, s2, D, inc, f_low, phi0) in enumerate(self._rows()):
            self.assertIsInstance(sobhb_coalescence_phase(m1, m2, s1, s2, f_low, phi0), float)
            hp, hx, _ = waveform_generate_h_plus_cross(m1, m2, D, inc, f_low, s1, s2, T_PN, phi0)
            hp_c, hx_c, _ = waveform_generate_h_plus_cross(
                m1, m2, D, inc, f_low, s1, s2, T_PN, phi_coal[i], phase_at_coalescence=True
            )
            scale = float(np.max(np.abs(hp)) + np.max(np.abs(hx)))
            np.testing.assert_allclose(np.asarray(hp_c), np.asarray(hp), rtol=0, atol=1e-6 * scale)
            np.testing.assert_allclose(np.asarray(hx_c), np.asarray(hx), rtol=0, atol=1e-6 * scale)

    def test_masked_core_follows_flag(self):
        d = self.src
        args = (d["m1"], d["m2"], d["D"], d["f_low"], d["s1"], d["s2"], T_PN)
        for flag in (False, True):
            _, Phi_m, _ = _pn_amp_phase_masked(*args, reference_phase=d["phi0"], phase_at_coalescence=flag)
            for i, (m1, m2, s1, s2, D, _, f_low, phi0) in enumerate(self._rows()):
                _, Phi, _ = _pn_amp_phase_core(
                    m1, m2, D, f_low, s1, s2, T_PN, reference_phase=phi0, phase_at_coalescence=flag
                )
                # vectorized vs scalar FP order: ~1e-8 rad on ~1e8 rad terms
                np.testing.assert_allclose(np.asarray(Phi_m[i]), np.asarray(Phi), rtol=0, atol=1e-6)

    @unittest.skipUnless(HAVE_PYGWTF, "pygwtf not installed")
    def test_coalescence_mode_matches_pygwtf_phase(self):
        """With the same tc, the coalescence-mode GW phase is pygwtf's ``2*(phi_coal - phase(x))``."""
        for m1, m2, s1, s2, D, _, f_low, phi_coal in self._rows():
            M, eta, sigma, delta, s, tc, _ = _intrinsics(m1, m2, f_low, s1, s2)
            params = np.zeros(12)
            params[[0, 1, 7, 8, 9, 10, 11]] = M, eta, phi_coal, tc, delta, sigma, s
            gw = np.array([_get_phi_f_fdot(t, params)[0] for t in T_PN])
            _, Phi, _ = _pn_amp_phase_core(
                m1, m2, D, f_low, s1, s2, T_PN, reference_phase=phi_coal, phase_at_coalescence=True
            )
            np.testing.assert_allclose(2 * np.asarray(Phi), gw, rtol=1e-12)


class WaveformKwargsTest(unittest.TestCase):
    def test_flag_survives_kwargs_round_trip(self):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.domains import TDSettings
        from lisatools.sources.sobbh import SOBBHTaylorT3TDIWaveform

        common = dict(
            waveform_t0=0.0, data_td_settings=TDSettings(N=4096, dt=5.0, t0=1.0e4, force_backend="cpu"),
            tdi_generation="2nd generation", tdi_channels="XYZ", sampling_frequency=0.2,
            orbits=EqualArmlengthOrbits(force_backend="cpu"), force_backend="cpu",
        )
        self.assertFalse(SOBBHTaylorT3TDIWaveform(**common).kwargs["phase_at_coalescence"])
        gen = SOBBHTaylorT3TDIWaveform(phase_at_coalescence=True, **common)
        self.assertTrue(gen.kwargs["phase_at_coalescence"])
        self.assertTrue(SOBBHTaylorT3TDIWaveform(**gen.kwargs).phase_at_coalescence)


if __name__ == "__main__":
    unittest.main()
