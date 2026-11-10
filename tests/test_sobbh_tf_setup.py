"""gwtf SOBBH branch on the erebor-priors ``SOBHBPrior``: setup defaults, catalogue conversion, search seeding."""

# Pre-existing env shims (see tests/test_l2d_ports.py): the installed scienceplots raises at import,
# and eryn.utils.plot uses a bare ``typing`` name. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

try:
    import pygwtf  # noqa: F401
    from ereborpriors.stock.mojitolight import SOBHBPrior

    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

WAVEFORM_BASIS = ["m1", "m2", "s1", "s2", "d_L", "iota", "f0", "phi_coal", "psi", "alpha", "delta"]

ENTRY = dict(
    PrimaryMassSSBFrame=20.0, SecondaryMassSSBFrame=35.0,  # swapped on purpose: m1 >= m2 enforced
    PrimarySpinCompZ=0.1, SecondarySpinCompZ=-0.3, InclinationAngle=2.4, LuminosityDistance=300.0,
    GW22FrequencySSBFrame=0.012, TrueAnomaly=1.3, PolarisationAngle=4.0, RightAscension=7.0,
    Declination=-0.2, TimeReferenceSSBFrame=1.0e4,
)


@unittest.skipUnless(HAVE_DEPS, "pygwtf / ereborpriors not installed")
class SOBBHTFSetupTest(unittest.TestCase):
    def _setup(self, **kw):
        from lisatools.globalfit.stock.erebor import SOBBHTFSettings, SOBBHTFSetup

        common = dict(Tobs=1e6, dt=5.0, initialize_kwargs={}, nleaves_max=2, nleaves_min=2)
        with tempfile.TemporaryDirectory() as d:
            return SOBBHTFSetup(SOBBHTFSettings(log_dir=d, **common, **kw))

    def test_defaults_from_sobhb_prior(self):
        setup = self._setup()
        prior = setup.priors["sobbh"]
        self.assertIsInstance(prior, SOBHBPrior)
        self.assertEqual(setup.ndim, 11)
        self.assertEqual(list(setup.transform.output_basis), WAVEFORM_BASIS)
        self.assertEqual(list(setup.transform.input_basis), prior.get_input_basis())
        self.assertEqual(setup.periodic, {"sobbh": prior.get_periodic()})

    def test_lims_reach_the_factory(self):
        setup = self._setup(f0_lims=[0.01, 0.011], Mc_lims=[20.0, 30.0])
        x = setup.priors["sobbh"].rvs(500)
        names = setup.transform.input_basis
        f0, Mc = x[:, names.index("f0")], x[:, names.index("Mc")]
        self.assertTrue(np.all((f0 >= 0.01) & (f0 <= 0.011)))
        self.assertTrue(np.all((Mc >= 20.0) & (Mc <= 30.0)))

    def test_injected_priors_are_kept(self):
        prior = SOBHBPrior.prior_factory(f0_min=0.02, f0_max=0.021)
        setup = self._setup(priors={"sobbh": prior}, transform=prior.get_transforms(), ndim=prior.ndim)
        self.assertIs(setup.priors["sobbh"], prior)
        with self.assertRaises(ValueError):
            self._setup(priors={"sobbh": prior})  # no transform


@unittest.skipUnless(HAVE_DEPS, "pygwtf / ereborpriors not installed")
class CatalogueConversionTest(unittest.TestCase):
    def test_recipe_sampling_basis_maps_to_waveform_basis(self):
        """The recipe output is in the SOBHBPrior sampling basis: its transform gives the helper's waveform basis."""
        from lisatools.globalfit.recipe import sobhb_catalogue_to_sampling_basis
        from lisatools.globalfit.searches import sobhb_catalogue_to_gwtf_basis

        transform = SOBHBPrior.prior_factory().get_transforms()
        sampling = sobhb_catalogue_to_sampling_basis(ENTRY, trim_plus_shift=7.2e5)
        waveform = sobhb_catalogue_to_gwtf_basis(ENTRY, 7.2e5)
        np.testing.assert_allclose(transform.both_transforms(sampling[None].copy())[0], waveform, rtol=1e-12)

    def test_waveform_basis_values(self):
        from lisatools.globalfit.searches import sobhb_catalogue_to_gwtf_basis
        from lisatools.sources.utils import evolve_sobhb, sobhb_coalescence_phase

        out = sobhb_catalogue_to_gwtf_basis(ENTRY, 7.2e5)
        m1, m2, s1, s2 = 35.0, 20.0, -0.3, 0.1
        phi_coal = sobhb_coalescence_phase(m1, m2, s1, s2, 0.012, 1.3)
        expected = [m1, m2, s1, s2, 0.3, 2.4, evolve_sobhb(0.0, 7.2e5, m1, m2, s1, s2, 0.012), phi_coal,
                    4.0, 7.0 % (2 * np.pi), -0.2]
        np.testing.assert_allclose(out, expected, rtol=1e-14)
        np.testing.assert_equal(sobhb_catalogue_to_gwtf_basis(ENTRY)[6], 0.012)  # no evolution


@unittest.skipUnless(HAVE_DEPS, "pygwtf / ereborpriors not installed")
class SOBHBSearchInterfaceTest(unittest.TestCase):
    def test_catalogue_seed_round_trips_through_prior(self):
        from lisatools.globalfit.searches import SOBHBSearchInterface
        from lisatools.globalfit.searches import sobhb_catalogue_to_gwtf_basis

        prior = SOBHBPrior.prior_factory()
        iface = SOBHBSearchInterface(search_type="prior", transform=prior.get_transforms())
        self.assertEqual(len(iface.physical_parameters), 11)
        curr = SimpleNamespace(general_info=SimpleNamespace(data_t0=ENTRY["TimeReferenceSSBFrame"] + 7.2e5))
        seed = iface.unpack_catalog_entries(ENTRY, curr)
        np.testing.assert_allclose(seed, sobhb_catalogue_to_gwtf_basis(ENTRY, 7.2e5), rtol=1e-14)
        sampling = iface.transform_to_sampling_basis(seed[None])
        self.assertTrue(np.all(np.isfinite(sampling)))
        np.testing.assert_allclose(prior.get_transforms().both_transforms(sampling.copy())[0], seed, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
