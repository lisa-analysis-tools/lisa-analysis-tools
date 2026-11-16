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

WAVEFORM_BASIS = ["m1", "m2", "s1", "s2", "d_L", "iota", "f0", "phi_0", "psi", "alpha", "delta"]

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


def _circular_diff(a, b):
    """``a - b`` wrapped to ``(-pi, pi]``."""
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


# phi_0 <-> phi_coal adds or subtracts P(x) ~ 1e7-1e9 rad: float64 leaves up to ~1e-7 rad on the phase
PHASE_ATOL = 1e-6
PHASE = 7  # column of the phase in the sampling and waveform bases


def assert_waveform_rows_close(actual, desired, phase=PHASE):
    """Waveform-basis rows equal to 1e-12 (relative), the phase column to PHASE_ATOL modulo 2 pi."""
    actual, desired = np.atleast_2d(actual), np.atleast_2d(desired)
    others = [i for i in range(actual.shape[-1]) if i != phase]
    np.testing.assert_allclose(actual[:, others], desired[:, others], rtol=1e-12)
    assert np.max(np.abs(_circular_diff(actual[:, phase], desired[:, phase]))) < PHASE_ATOL


@unittest.skipUnless(HAVE_DEPS, "pygwtf / ereborpriors not installed")
class PriorPhaseTest(unittest.TestCase):
    """``SOBHBPrior`` samples ``phi_0`` and hands it to the waveforms unchanged (no phase transform)."""

    @classmethod
    def setUpClass(cls):
        cls.prior = SOBHBPrior.prior_factory(f0_min=5e-3, f0_max=2e-2, chirp_mass_min=20.0, chirp_mass_max=75.0)
        cls.transform = cls.prior.get_transforms()
        cls.x = cls.prior.rvs(2000)

    def test_basis_and_periodicity(self):
        self.assertEqual(self.prior.get_input_basis()[PHASE], "phi_0")
        self.assertEqual(list(self.transform.output_basis), WAVEFORM_BASIS)
        self.assertIn("phi_0", self.prior.get_periodic())
        self.assertNotIn("phi_coal", self.prior.get_periodic())

    def test_phase_passes_through(self):
        y = self.transform.both_transforms(self.x.copy())
        np.testing.assert_array_equal(y[:, PHASE], self.x[:, PHASE])
        back = self.transform.both_inverse_transforms(y.copy())
        np.testing.assert_allclose(back, self.x, rtol=1e-12, atol=1e-15)

    def test_per_source_priors_sample_phi_0(self):
        for prior in SOBHBPrior.per_source_prior_factory():
            self.assertEqual(prior.get_input_basis()[PHASE], "phi_0")
            self.assertEqual(list(prior.get_transforms().output_basis), WAVEFORM_BASIS)


@unittest.skipUnless(HAVE_DEPS, "pygwtf / ereborpriors not installed")
class PhaseHelperTest(unittest.TestCase):
    """``sobhb_reference_phase`` inverts ``sobhb_coalescence_phase`` at the same frequency."""

    def test_round_trip(self):
        from lisatools.sources.utils import sobhb_coalescence_phase, sobhb_reference_phase

        rng = np.random.default_rng(1)
        m = rng.uniform(10.0, 80.0, (500, 2))
        m1, m2 = m.max(axis=1), m.min(axis=1)
        s1, s2 = rng.uniform(-0.9, 0.9, (2, 500))
        f, phi_0 = 10 ** rng.uniform(-2.3, -1.5, 500), rng.uniform(0, 2 * np.pi, 500)
        phi_coal = sobhb_coalescence_phase(m1, m2, s1, s2, f, phi_0)
        back = sobhb_reference_phase(m1, m2, s1, s2, f, phi_coal)
        self.assertLess(np.max(np.abs(_circular_diff(back, phi_0))), PHASE_ATOL)
        self.assertTrue(np.all((back >= 0) & (back < 2 * np.pi)))
        self.assertGreater(np.max(np.abs(_circular_diff(phi_coal, phi_0))), 1.0)  # P(x) is not trivial
        self.assertIsInstance(sobhb_reference_phase(m1[0], m2[0], s1[0], s2[0], f[0], phi_coal[0]), float)


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
        """phi_0 (default) is the TrueAnomaly carried to the evolved f0; phase_at_coalescence gives phi_coal."""
        from lisatools.globalfit.searches import sobhb_catalogue_to_gwtf_basis
        from lisatools.sources.utils import evolve_sobhb, sobhb_coalescence_phase, sobhb_reference_phase

        m1, m2, s1, s2 = 35.0, 20.0, -0.3, 0.1
        f0 = evolve_sobhb(0.0, 7.2e5, m1, m2, s1, s2, 0.012)
        phi_coal = sobhb_coalescence_phase(m1, m2, s1, s2, 0.012, 1.3)
        common = [m1, m2, s1, s2, 0.3, 2.4, f0]
        tail = [4.0, 7.0 % (2 * np.pi), -0.2]

        out = sobhb_catalogue_to_gwtf_basis(ENTRY, 7.2e5)
        expected = common + [sobhb_reference_phase(m1, m2, s1, s2, f0, phi_coal)] + tail
        np.testing.assert_allclose(out, expected, rtol=1e-14)
        out_coal = sobhb_catalogue_to_gwtf_basis(ENTRY, 7.2e5, phase_at_coalescence=True)
        np.testing.assert_allclose(out_coal, common + [phi_coal] + tail, rtol=1e-14)

        unevolved = sobhb_catalogue_to_gwtf_basis(ENTRY)
        np.testing.assert_equal(unevolved[6], 0.012)  # no evolution
        # no evolution: the data start is the catalogue epoch, so phi_0 is the TrueAnomaly itself
        self.assertLess(abs(_circular_diff(unevolved[PHASE], 1.3)), PHASE_ATOL)


@unittest.skipUnless(HAVE_DEPS, "pygwtf / ereborpriors not installed")
class SOBHBSearchInterfaceTest(unittest.TestCase):
    CURR = SimpleNamespace(general_info=SimpleNamespace(data_t0=ENTRY["TimeReferenceSSBFrame"] + 7.2e5))

    def test_catalogue_seed_round_trips_through_prior(self):
        from lisatools.globalfit.searches import SOBHBSearchInterface, sobhb_catalogue_to_gwtf_basis

        prior = SOBHBPrior.prior_factory()
        iface = SOBHBSearchInterface(search_type="prior", transform=prior.get_transforms())
        self.assertEqual(len(iface.physical_parameters), 11)
        seed = iface.unpack_catalog_entries(ENTRY, self.CURR)
        np.testing.assert_allclose(seed, sobhb_catalogue_to_gwtf_basis(ENTRY, 7.2e5), rtol=1e-14)
        sampling = iface.transform_to_sampling_basis(seed[None])
        self.assertTrue(np.all(np.isfinite(sampling)))
        np.testing.assert_allclose(prior.get_transforms().both_transforms(sampling.copy())[0], seed, rtol=1e-12)

    def test_phase_convention_reaches_the_seeds(self):
        from lisatools.globalfit.searches import SOBHBSearchInterface, sobhb_catalogue_to_gwtf_basis

        transform = SOBHBPrior.prior_factory().get_transforms()
        for flag in (False, True):
            iface = SOBHBSearchInterface(search_type="prior", transform=transform, phase_at_coalescence=flag)
            np.testing.assert_allclose(
                iface.unpack_catalog_entries(ENTRY, self.CURR),
                sobhb_catalogue_to_gwtf_basis(ENTRY, 7.2e5, phase_at_coalescence=flag), rtol=1e-14,
            )


@unittest.skipUnless(HAVE_DEPS, "pygwtf / ereborpriors not installed")
class SOBBHBlockPhaseTest(unittest.TestCase):
    """The block's ``phase_at_coalescence`` sets the gwtf waveform, the TD generator and the seeds together."""

    @staticmethod
    def block(**raw):
        import warnings

        from lisatools.globalfit.stock.l2d.blocks.sobbh import SOBBHBlock

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # no search: the branch-starts-empty warning
            return SOBBHBlock.model_validate({"type": "sobbh", **raw})

    def test_one_setting_reaches_every_consumer(self):
        general = SimpleNamespace(window_type="tukey", window_alpha=0.01, domain_settings=None, gpu_orbits=None,
                                  orbits=None, force_backend="cpu", gpu_backend="cuda12x", Tobs=1.0)
        for raw, flag in (({}, False), ({"phase_at_coalescence": True}, True)):
            block = self.block(**raw)
            self.assertIs(block.phase_at_coalescence, flag)
            self.assertIs(block.engine_waveform_kwargs(SimpleNamespace(general=general))["phase_at_coalescence"], flag)
            self.assertIs(block.td_generator().init_kwargs["phase_at_coalescence"], flag)
            self.assertEqual(block.search_interface_kwargs(), {"phase_at_coalescence": flag})

    def test_waveform_level_flags_are_rejected(self):
        from pydantic import ValidationError

        with self.assertRaisesRegex(ValidationError, "filled in by the builder"):
            self.block(waveform={"class": "GWTF_generator", "kwargs": {"phase_at_coalescence": True}})
        with self.assertRaisesRegex(ValidationError, "set the block's phase_at_coalescence"):
            self.block(td_waveform={"class": "SOBBHTaylorT3TDIWaveform", "kwargs": {"phase_at_coalescence": True}})


if __name__ == "__main__":
    unittest.main()
