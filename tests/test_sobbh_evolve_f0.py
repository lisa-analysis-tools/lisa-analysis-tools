"""The shared SOBBH f0-evolution helper ``evolve_sobhb``."""

# Pre-existing env shims (see tests/test_l2d_ports.py): the installed scienceplots raises at import,
# and eryn.utils.plot uses a bare ``typing`` name. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import unittest

import numpy as np

try:
    import pygwtf  # noqa: F401

    HAVE_PYGWTF = True
except ImportError:
    HAVE_PYGWTF = False


def _draws(n=64, seed=0):
    rng = np.random.default_rng(seed)
    m1, m2 = rng.uniform(5, 100, n), rng.uniform(5, 100, n)
    return dict(
        m1=np.maximum(m1, m2), m2=np.minimum(m1, m2),
        s1=rng.uniform(-0.99, 0.99, n), s2=rng.uniform(-0.99, 0.99, n),
        f0=10 ** rng.uniform(-2.5, -1.5, n), dt=rng.uniform(0.0, 1e7, n),
    )


@unittest.skipUnless(HAVE_PYGWTF, "pygwtf not installed")
class EvolveSobhbTest(unittest.TestCase):
    def setUp(self):
        from lisatools.sources.utils import evolve_sobhb

        self.evolve = evolve_sobhb
        self.d = _draws()

    def _reference(self, m1, m2, s1, s2, f0, dt):
        """The pre-consolidation ``recompute_f0`` of ``recipe.sobhb_catalogue_to_sampling_basis``."""
        from pygwtf import constants, models

        m1, m2 = m1 * constants.MTsun, m2 * constants.MTsun
        M = m1 + m2
        eta = m1 * m2 / M**2
        sigma, s, delta = (m2 * s2 - m1 * s1) / M, (m1**2 * s1 + m2**2 * s2) / M**2, (m1 - m2) / M
        common = models.taylort3spin.common
        tc = common.time_to_merger((np.pi * M * f0) ** (2 / 3), sigma, delta, eta, s) * M
        x = common.tau_to_x(eta * (tc - dt) / (5 * M), sigma, delta, eta, s)
        return x**1.5 / (np.pi * M)

    def test_matches_previous_recipe_implementation(self):
        d = self.d
        out = self.evolve(0.0, d["dt"], d["m1"], d["m2"], d["s1"], d["s2"], d["f0"])
        ref = np.array([self._reference(*a) for a in zip(d["m1"], d["m2"], d["s1"], d["s2"], d["f0"], d["dt"])])
        np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
        np.testing.assert_allclose(out, ref, rtol=1e-12, equal_nan=True)

    def test_scalar_and_array_agree(self):
        d = self.d
        arr = self.evolve(0.0, d["dt"], d["m1"], d["m2"], d["s1"], d["s2"], d["f0"])
        for i in range(5):
            one = self.evolve(0.0, d["dt"][i], d["m1"][i], d["m2"][i], d["s1"][i], d["s2"][i], d["f0"][i])
            self.assertIsInstance(one, float)
            np.testing.assert_equal(one, arr[i])

    def test_zero_interval_keeps_frequency(self):
        # time_to_merger and tau_to_x are truncated PN series, inverse only to ~1e-7
        d = self.d
        out = self.evolve(3.0, 3.0, d["m1"], d["m2"], d["s1"], d["s2"], d["f0"])
        np.testing.assert_allclose(out, d["f0"], rtol=1e-6)

    def test_evolves_upward_in_frequency(self):
        d = self.d
        out = self.evolve(0.0, 1e5, d["m1"], d["m2"], d["s1"], d["s2"], d["f0"])
        self.assertTrue(np.all(out[np.isfinite(out)] > d["f0"][np.isfinite(out)]))

    def test_catalogue_conversion_uses_helper(self):
        from lisatools.globalfit.recipe import sobhb_catalogue_to_sampling_basis

        entry = dict(
            PrimaryMassSSBFrame=20.0, SecondaryMassSSBFrame=35.0,  # swapped on purpose: m1 >= m2 enforced
            PrimarySpinCompZ=0.1, SecondarySpinCompZ=-0.3, InclinationAngle=0.4, LuminosityDistance=300.0,
            GW22FrequencySSBFrame=0.012, PolarisationAngle=4.0, RightAscension=7.0, Declination=-0.2,
            TrueAnomaly=1.3,
        )
        out = sobhb_catalogue_to_sampling_basis(entry, trim_plus_shift=7.2e5)
        f0 = self.evolve(0.0, 7.2e5, 35.0, 20.0, -0.3, 0.1, 0.012)
        self.assertEqual(out.shape, (11,))
        self.assertEqual(out[6], f0)  # SOBHBPrior sampling basis: f0 is column 6


if __name__ == "__main__":
    unittest.main()
