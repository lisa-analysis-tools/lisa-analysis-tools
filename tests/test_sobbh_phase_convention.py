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


@unittest.skipUnless(HAVE_PYGWTF, "pygwtf not installed")
class GWTFMatchesTDTest(unittest.TestCase):
    """``GWTF_generator`` and ``SOBBHTaylorT3TDIWaveform`` give the same signal for the same parameters, in both phase
    conventions (``phase_at_coalescence`` True: phi_coal; False: phi_0, the orbital phase at the data start).
    CPU, 3 days; the GPU version with more sources is scripts/sobbh/sobbh_gwtf_td_convention_check.py.

    The only expected phase difference is gwtf's unrefined coalescence time: arg <gwtf|TD> = 2 pi f0 (tc_gwtf - tc_TD).
    A strain-sign mismatch (e.g. pygwtf's POL_CONVENTION not at the mojito value -1) gives arg ~ pi; a phase-convention
    mismatch between the two generators gives an arbitrary arg (P(x0) mod 2 pi).
    """

    DT, BIG_DT, T0, DAYS = 5.0, 86400.0, 1.0e5, 3
    # m1, m2, s1, s2, dist [Gpc], inc, f0 [Hz, data start], phase, psi, ra, dec
    SOURCE = np.array([25.0, 18.0, -0.5, 0.4, 0.05, 2.1, 0.018, 0.3, 1.3, 4.5, -0.8])

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.domains import TDSettings, get_stft_settings
        from lisatools.sources.sobbh import SOBBHTaylorT3TDIWaveform
        from lisatools.sources.sobbh.waveform_gwtf import GWTF_generator

        times = cls.T0 + np.arange(int(cls.DAYS * 86400 / cls.DT)) * cls.DT
        cls.settings = get_stft_settings(times, big_dt=cls.BIG_DT, min_freq=4e-3, max_freq=4e-2, force_backend="cpu")
        n = int(round(cls.settings.NT * cls.settings.dt / cls.DT))
        cls.grid = TDSettings(N=n, dt=cls.DT, t0=cls.settings.t0, force_backend="cpu")
        common = dict(
            data_td_settings=cls.grid, tdi_generation="2nd generation", tdi_channels="XYZ",
            sampling_frequency=1.0 / cls.DT, orbits=EqualArmlengthOrbits(force_backend="cpu"), order=25,
            buffer_time=1.0e4, force_backend="cpu", waveform_t0=cls.settings.t0,
        )
        # convention -> (gwtf, TD) pair taking the same phase parameter
        cls.pairs = {
            flag: (
                GWTF_generator(
                    settings=cls.settings, orbits=EqualArmlengthOrbits(force_backend="cpu"), T_obs=n * cls.DT,
                    fresnel_kernel_width=16, use_GPU=False, phase_at_coalescence=flag,
                ),
                SOBBHTaylorT3TDIWaveform(phase_at_coalescence=flag, **common),
            )
            for flag in (False, True)
        }

    def overlap(self, gwtf, gwtf_params, td, td_params):
        g = np.asarray(gwtf.get_signals_for_residuals(*gwtf_params).arr)
        t = np.asarray(td.get_td_signals(*td_params, td_settings=self.grid).transform(self.settings).arr)
        self.assertEqual(g.shape, t.shape)
        return np.sum(np.conj(g) * t) / np.sqrt(np.sum(np.abs(g) ** 2) * np.sum(np.abs(t) ** 2))

    def expected_arg(self, m1, m2, s1, s2, f0):
        from pygwtf.models.taylort3spin.common import time_to_merger

        M, eta, sigma, s, delta, tc_td = (float(v) for v in _pn_intrinsics(m1, m2, f0, s1, s2))
        tc_gwtf = time_to_merger((np.pi * M * f0) ** (2 / 3), sigma, delta, eta, s) * M
        return 2 * np.pi * f0 * (tc_gwtf - tc_td)

    def assertSameSignal(self, ov, m1, m2, s1, s2, f0):
        self.assertGreater(abs(ov), 0.95)  # gwtf's time-frequency approximation: ~1-2 % mismatch
        self.assertAlmostEqual(np.angle(ov), self.expected_arg(m1, m2, s1, s2, f0), delta=0.02)

    def test_flag_is_kept_for_replicas(self):
        """MultiGPUResidualAddRemoveMove rebuilds GWTF_generator(**kwargs): the convention must survive."""
        for flag, (gwtf, _) in self.pairs.items():
            self.assertIs(gwtf.kwargs["phase_at_coalescence"], flag)

    def test_same_parameters_same_signal(self):
        m1, m2, s1, s2, _, _, f0 = self.SOURCE[:7]
        for flag, (gwtf, td) in self.pairs.items():
            with self.subTest(phase_at_coalescence=flag):
                self.assertSameSignal(self.overlap(gwtf, self.SOURCE, td, self.SOURCE), m1, m2, s1, s2, f0)

    def test_catalogue_phase_mapping(self):
        """TD in the reference-epoch convention (phi0 = TrueAnomaly at the catalogue epoch = data start here) vs gwtf
        seeded through sobhb_catalogue_to_gwtf_basis, in both conventions."""
        from lisatools.globalfit.searches import sobhb_catalogue_to_gwtf_basis

        m1, m2, s1, s2, dist, inc, f0, _, psi, ra, dec = self.SOURCE
        true_anomaly = 2.2
        entry = dict(
            PrimaryMassSSBFrame=m1, SecondaryMassSSBFrame=m2, PrimarySpinCompZ=s1, SecondarySpinCompZ=s2,
            LuminosityDistance=dist * 1e3, InclinationAngle=inc, GW22FrequencySSBFrame=f0, TrueAnomaly=true_anomaly,
            PolarisationAngle=psi, RightAscension=ra, Declination=dec,
        )
        td_params = [m1, m2, s1, s2, dist, inc, f0, true_anomaly, psi, ra, dec]
        td_reference = self.pairs[False][1]
        for flag, (gwtf, _) in self.pairs.items():
            with self.subTest(phase_at_coalescence=flag):
                seed = sobhb_catalogue_to_gwtf_basis(entry, phase_at_coalescence=flag)
                self.assertSameSignal(self.overlap(gwtf, seed, td_reference, td_params), m1, m2, s1, s2, f0)

    def test_phi_0_sampling_basis(self):
        """SOBHBPrior samples phi_0 and passes it through: the default gwtf matches the TD waveform anchored at the
        data start (waveform_t0 = settings.t0) with phi0 = phi_0."""
        from ereborpriors.stock.mojitolight import SOBHBPrior

        m1, m2, s1, s2, dist, inc, f0, _, psi, ra, dec = self.SOURCE
        transform = SOBHBPrior.prior_factory().get_transforms()
        gwtf, td_reference = self.pairs[False]
        for phi_0 in (0.4, 2.0, 5.1):
            gwtf_params = transform.both_transforms(np.array([[
                (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2, m1 * m2 / (m1 + m2) ** 2, s1, s2, dist, np.cos(inc), f0,
                phi_0, psi, ra, np.sin(dec),
            ]]))[0]
            td_params = [m1, m2, s1, s2, dist, inc, f0, phi_0, psi, ra, dec]
            self.assertSameSignal(self.overlap(gwtf, gwtf_params, td_reference, td_params), m1, m2, s1, s2, f0)


if __name__ == "__main__":
    unittest.main()
