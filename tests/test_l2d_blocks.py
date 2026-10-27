"""Tests for the L2D block layer (``lisatools.globalfit.stock.l2d.blocks``): schema, priors, psd build."""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pydantic import ValidationError

from eryn.priors import ProbDistContainer, UniformDistribution
from ereborpriors.base import GFBasePrior
from lisatools.globalfit.postprocessing import StochasticMetadata
from lisatools.globalfit.stock.l2d import load_settings
from lisatools.globalfit.stock.l2d.io import RunConfig
from lisatools.globalfit.stock.l2d.load_settings import load_blocks, load_general_setup

from tests.test_l2d_load_settings import FakeL1, run_1_raw

NOISE_PRIOR = {"class": "mojitolight.stochastic.NoisePrior", "kwargs": {"model_type": "parametric-symmetric"}}


def cpu_raw(blocks=None):
    """A small CPU config (1 day, no preprocessing, composite sensitivity) with ``blocks``."""
    data = {k: v for k, v in run_1_raw()["data"].items() if k not in ("highpass", "lowpass", "downsample", "trim")}
    data["Tobs"] = "1 day"
    raw = run_1_raw(
        data=data,
        compute=dict(backend="cpu"),
        window=dict(type="tukey", taper_duration="auto"),
        sensitivity={"class": "CompositeSensitivityBackend"},
    )
    raw["blocks"] = blocks or {}
    return raw


def error_keys(exc: ValidationError):
    """``(type, dotted.loc)`` for every error pydantic reported."""
    return [(err["type"], ".".join(map(str, err["loc"]))) for err in exc.errors()]


class _Blocks(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run_dir = Path(tmp.name)

    def config(self, **psd):
        """RunConfig with a psd block ``{type: psd, **psd}``."""
        raw = cpu_raw({"psd": {"type": "psd", **psd}})
        return RunConfig.model_validate(raw, context={"run_dir": self.run_dir})

    def psd(self, **fields):
        return self.config(**fields).blocks["psd"]

    def assertBlockError(self, err_type, loc, msg=None, **psd):
        """Validating a psd block with ``psd`` fails with an ``err_type`` error at ``blocks.psd.psd.<loc>``."""
        with self.assertRaises(ValidationError) as ctx:
            self.config(**psd)
        want = "blocks.psd.psd." + loc if loc else "blocks.psd.psd"
        matches = [
            err for err in ctx.exception.errors()
            if err["type"] == err_type and ".".join(map(str, err["loc"])) == want
        ]
        self.assertTrue(matches, f"no {err_type} at {want!r}; got {error_keys(ctx.exception)}")
        if msg is not None:
            self.assertTrue(
                any(msg in err["msg"] for err in matches),
                f"{msg!r} not in {[err['msg'] for err in matches]}",
            )


class PSDSchemaTest(_Blocks):
    def test_priors_are_required(self):
        self.assertBlockError("missing", "priors")

    def test_minimal_block_and_defaults(self):
        block = self.psd(priors=NOISE_PRIOR)
        self.assertEqual(block.type, "psd")
        self.assertTrue(block.enabled)
        self.assertEqual(block.priors.class_name, "mojitolight.stochastic.NoisePrior")
        self.assertIsNone(block.tempering)
        self.assertIsNone(block.inner_moves)
        self.assertIsNone(block.search)
        self.assertIsNone(block.injection)
        self.assertEqual(block.num_prop_repeats, 100)

    def test_fields(self):
        block = self.psd(
            priors=NOISE_PRIOR,
            enabled=False,
            num_prop_repeats=500,
            injection={"S_oms": 15e-12, "S_tm": 3e-15},
            tempering={"ntemps": 3, "ratio": 1.5},
            search={"search_type": "prior", "spread": [1e-4, 1e-3]},
        )
        self.assertFalse(block.enabled)
        self.assertEqual(block.num_prop_repeats, 500)
        self.assertEqual(block.injection, {"S_oms": 15e-12, "S_tm": 3e-15})
        self.assertEqual((block.tempering.ntemps, block.tempering.ratio), (3, 1.5))
        self.assertEqual(block.search.search_type, "prior")
        self.assertEqual(block.search.spread, [1e-4, 1e-3])

    def test_ladder_may_be_a_subset_but_not_longer_than_the_sampler(self):
        sampler_ntemps = self.config(priors=NOISE_PRIOR).sampler.ntemps
        for ntemps in (1, 2, sampler_ntemps):
            with self.subTest(ntemps=ntemps):
                self.psd(priors=NOISE_PRIOR, tempering={"ntemps": ntemps})
        with self.assertRaises(ValidationError) as ctx:
            self.config(priors=NOISE_PRIOR, tempering={"ntemps": sampler_ntemps + 1})
        self.assertTrue(any("is above sampler.ntemps" in e["msg"] for e in ctx.exception.errors()))

    def test_unknown_field(self):
        self.assertBlockError("extra_forbidden", "tempring", tempring={})

    def test_wrong_values(self):
        self.assertBlockError("greater_than", "num_prop_repeats", num_prop_repeats=0)
        self.assertBlockError("greater_than", "tempering.ratio", tempering={"ratio": 1.0})
        self.assertBlockError("greater_than", "tempering.ntemps", tempering={"ntemps": 0})
        self.assertBlockError("literal_error", "search.search_type", search={"search_type": "guess"})


class PriorSpecTest(_Blocks):
    def test_stock_class(self):
        spec = self.psd(priors=NOISE_PRIOR).priors
        self.assertEqual(spec.class_name, "mojitolight.stochastic.NoisePrior")
        self.assertEqual(spec.kwargs, {"model_type": "parametric-symmetric"})
        self.assertIsNone(spec.output_file)

    def test_class_needs_a_module(self):
        self.assertBlockError("value_error", "priors.class", msg="use <module>.<Class>", priors={"class": "NoisePrior"})

    def test_unknown_module(self):
        self.assertBlockError(
            "value_error", "priors.class", msg="no module ereborpriors.stock.nonesuch.",
            priors={"class": "nonesuch.NoisePrior"},
        )

    def test_unknown_class(self):
        self.assertBlockError(
            "value_error", "priors.class", msg="no class 'Nonesuch'",
            priors={"class": "mojitolight.stochastic.Nonesuch"},
        )

    def test_class_must_be_a_gf_prior(self):
        # UniformDistribution is importable from that module but is not a GFBasePrior.
        self.assertBlockError(
            "value_error", "priors.class", msg="not a GFBasePrior subclass",
            priors={"class": "mojitolight.stochastic.UniformDistribution"},
        )

    def test_kwargs_are_checked_against_the_factory(self):
        self.assertBlockError(
            "value_error", "priors.class", msg="does not accept ['model_typ']",
            priors={"class": "mojitolight.stochastic.NoisePrior", "kwargs": {"model_typ": "x"}},
        )

    def test_engine_filled_kwarg(self):
        self.assertBlockError(
            "value_error", "priors.class", msg="filled in by the builder",
            priors={"class": "mojitolight.mbhb.MBHBPrior", "kwargs": {"observation_time": 1.0}},
        )

    def test_prior_file_must_exist(self):
        self.assertBlockError(
            "value_error", "priors.file.file", msg="does not exist", priors={"file": "missing.yaml"}
        )

    def test_prior_file_relative_to_run_folder(self):
        (self.run_dir / "psd_prior.yaml").write_text("branch_name: psd\npriors: []\n")
        spec = self.psd(priors={"file": "psd_prior.yaml"}).priors
        self.assertEqual(spec.file, self.run_dir / "psd_prior.yaml")

    def test_file_and_class_are_mutually_exclusive(self):
        (self.run_dir / "psd_prior.yaml").write_text("branch_name: psd\npriors: []\n")
        self.assertBlockError(
            "extra_forbidden", "priors.file.class", priors={"file": "psd_prior.yaml", **NOISE_PRIOR}
        )

    def test_output_file_relative_to_run_folder(self):
        spec = self.psd(priors={**NOISE_PRIOR, "output_file": "out/psd_prior.yaml"}).priors
        self.assertEqual(spec.output_file, self.run_dir / "out" / "psd_prior.yaml")


class _Built(_Blocks):
    """A real CPU GeneralSetup (only the mojito file I/O is faked), shared by the build tests."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name)
        cfg = RunConfig.model_validate(cpu_raw(), context={"run_dir": cls.run_dir})
        with mock.patch.dict(load_settings.LOADERS, {"mojito_l1": FakeL1}):
            cls.general = load_general_setup(cfg)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):  # the shared run folder, not a per-test one
        pass

    def build(self, **psd):
        raw = cpu_raw({"psd": {"type": "psd", **psd}})
        cfg = RunConfig.model_validate(raw, context={"run_dir": self.run_dir})
        return cfg, load_blocks(cfg, self.general)


class PSDBuildTest(_Built):
    def test_setup_and_settings(self):
        from lisatools.globalfit.stock.erebor import PSDSetup  # imported here: schema tests don't need it

        cfg, built = self.build(priors=NOISE_PRIOR, num_prop_repeats=500, injection={"S_oms": 15e-12, "S_tm": 3e-15})
        build = built["psd"]
        setup = build.setup
        self.assertIsInstance(setup, PSDSetup)
        self.assertIsNone(build.waveform_class)
        self.assertEqual(setup.ndim, 2)
        self.assertEqual(setup.num_prop_repeats, 500)
        self.assertEqual(setup.Tobs, self.general.Tobs)
        self.assertEqual(setup.dt, self.general.dt)
        self.assertEqual(setup.log_dir, self.general.artifacts_file_dir)

    def test_priors_are_keyed_by_block_type(self):
        _, built = self.build(priors=NOISE_PRIOR)
        priors = built["psd"].setup.priors
        self.assertEqual(list(priors), ["psd"])
        self.assertIsInstance(priors["psd"], GFBasePrior)
        self.assertEqual(priors["psd"].get_input_basis(), ["S_oms", "S_tm"])

    def test_injection_follows_the_prior_basis(self):
        # Given in the "wrong" order on purpose: the array is ordered by the prior basis.
        _, built = self.build(priors=NOISE_PRIOR, injection={"S_tm": 3e-15, "S_oms": 15e-12})
        np.testing.assert_array_equal(built["psd"].setup.injection, [15e-12, 3e-15])

    def test_injection_keys_must_match_the_basis(self):
        with self.assertRaisesRegex(ValueError, "do not match prior basis"):
            self.build(priors=NOISE_PRIOR, injection={"S_oms": 15e-12})

    def test_no_injection(self):
        _, built = self.build(priors=NOISE_PRIOR)
        self.assertIsNone(built["psd"].setup.injection)

    def test_tempering_ladder(self):
        cfg, built = self.build(priors=NOISE_PRIOR, tempering={"ratio": 1.5})
        ntemps = cfg.sampler.ntemps  # tempering.ntemps defaults to sampler.ntemps
        np.testing.assert_allclose(built["psd"].setup.betas, 1.5 ** -np.arange(ntemps, dtype=float))

    def test_explicit_tempering_ntemps(self):
        _, built = self.build(priors=NOISE_PRIOR, tempering={"ntemps": 3, "ratio": 2.0})
        np.testing.assert_allclose(built["psd"].setup.betas, [1.0, 0.5, 0.25])

    def test_no_tempering_keeps_the_setup_default_ladder(self):
        _, built = self.build(priors=NOISE_PRIOR)
        betas = built["psd"].setup.betas
        self.assertEqual(len(betas), 24)  # PSDSetup's own make_ladder(ndim * 10, ntemps=24)

    def test_metadata(self):
        _, built = self.build(priors=NOISE_PRIOR)
        meta = built["psd"].metadata
        self.assertIsInstance(meta, StochasticMetadata)
        self.assertEqual(meta.prior_model, "NoisePrior")
        self.assertEqual(meta.frequency_ranges, [(self.general.start_freq, self.general.end_freq)])
        self.assertEqual(meta.model_config, dict(use_splines=False, num_params=2))

    def test_links_reach_the_metadata(self):
        _, built = self.build(
            priors=NOISE_PRIOR, links={"prior_model_code_link": "https://example.org/priors"}
        )
        self.assertEqual(built["psd"].metadata.prior_model_code_link, "https://example.org/priors")

    def test_prior_output_file_is_written_and_linked(self):
        out = self.run_dir / "written" / "psd_prior.yaml"
        out.parent.mkdir(exist_ok=True)
        _, built = self.build(priors={**NOISE_PRIOR, "output_file": str(out)})
        self.assertTrue(out.is_file())
        self.assertEqual(built["psd"].metadata.prior_model_config_file, str(out))
        reloaded = GFBasePrior.from_yaml(str(out))
        self.assertEqual(reloaded.get_input_basis(), ["S_oms", "S_tm"])

    def test_prior_files_are_not_supported_yet(self):
        # TODO(l2d): from_yaml returns a bare GFBasePrior without transforms; fail clearly until it is done.
        prior_file = self.run_dir / "psd_prior_file.yaml"
        prior_file.write_text("branch_name: psd\npriors: []\n")
        with self.assertRaisesRegex(NotImplementedError, "not supported yet"):
            self.build(priors={"file": str(prior_file)})

    def test_disabled_blocks_are_not_built(self):
        _, built = self.build(priors=NOISE_PRIOR, enabled=False)
        self.assertEqual(built, {})

    def test_config_is_not_changed_by_the_build(self):
        # build() records the written prior file in links; the validated spec itself stays as loaded.
        out = self.run_dir / "written" / "psd_prior_2.yaml"
        out.parent.mkdir(exist_ok=True)
        cfg, _ = self.build(priors={**NOISE_PRIOR, "output_file": str(out)})
        self.assertEqual(cfg.blocks["psd"].priors.kwargs, {"model_type": "parametric-symmetric"})


class PSDSearchSchemaTest(_Blocks):
    """``search:`` is the arguments of the branch's search interface; the interface's own rules hold at load."""

    POINT = {"search_type": "point_estimate", "spread": [1e-13, 1e-16]}

    def point_file(self, name="psd_point.npy"):
        path = self.run_dir / name
        np.save(path, np.array([1.5e-11, 3e-15]))
        return path

    def test_no_search_by_default(self):
        self.assertIsNone(self.psd(priors=NOISE_PRIOR).search)

    def test_available_types(self):
        path = self.point_file()
        search = self.psd(priors=NOISE_PRIOR, search={**self.POINT, "search_path": str(path)}).search
        self.assertEqual(search.search_type, "point_estimate")
        self.assertEqual(search.spread, [1e-13, 1e-16])
        self.assertEqual(search.parameters_searched, "all")
        self.assertFalse(search.is_eryn_backend)
        self.psd(priors=NOISE_PRIOR, search={"search_type": "prior"})

    def test_noise_has_no_catalogue_and_no_distribution_search_yet(self):
        for search_type in ("catalog", "distribution"):
            with self.subTest(search_type=search_type):
                self.assertBlockError(
                    "value_error", "", msg="not available for the 'psd' block",
                    priors=NOISE_PRIOR, search={"search_type": search_type, "spread": 1e-3, "search_path": "."},
                )

    def test_samples_and_point_estimates_need_a_path(self):
        for search_type in ("samples", "point_estimate"):
            with self.subTest(search_type=search_type):
                self.assertBlockError(
                    "value_error", "search", msg="needs search_path",
                    priors=NOISE_PRIOR, search={"search_type": search_type, "spread": 1e-3},
                )

    def test_point_estimate_needs_a_spread(self):
        path = self.point_file()
        self.assertBlockError(
            "value_error", "search", msg="needs spread",
            priors=NOISE_PRIOR, search={"search_type": "point_estimate", "search_path": str(path)},
        )

    def test_path_must_exist(self):
        self.assertBlockError(
            "value_error", "search", msg="does not exist",
            priors=NOISE_PRIOR, search={**self.POINT, "search_path": "missing.npy"},
        )

    def test_path_is_relative_to_the_run_folder(self):
        path = self.point_file()
        search = self.psd(priors=NOISE_PRIOR, search={**self.POINT, "search_path": path.name}).search
        self.assertEqual(search.search_path, path)

    def test_spread_forms(self):
        # scalar, per-parameter, full covariance, one covariance per leaf
        for spread in (1e-3, [1e-3, 1e-3], [[1e-6, 0.0], [0.0, 1e-6]], [[[1e-6, 0.0], [0.0, 1e-6]]]):
            with self.subTest(spread=spread):
                path = self.point_file()
                search = self.psd(
                    priors=NOISE_PRIOR, search={"search_type": "point_estimate", "search_path": str(path), "spread": spread}
                ).search
                self.assertEqual(search.spread, spread)


class PSDSearchBuildTest(_Built):
    POINT = (1.5e-11, 3e-15)

    def point_search(self, **extra):
        path = self.run_dir / "psd_point_build.npy"
        np.save(path, np.array(self.POINT))
        return {"search_type": "point_estimate", "search_path": str(path), "spread": [1e-13, 1e-16], **extra}

    def test_no_search_means_no_interface(self):
        _, built = self.build(priors=NOISE_PRIOR)
        self.assertIsNone(built["psd"].setup.search_interface)

    def test_interface_gets_the_prior_transform_and_the_search_arguments(self):
        from lisatools.globalfit.searches import NoiseSearchInterface

        _, built = self.build(priors=NOISE_PRIOR, search=self.point_search())
        setup = built["psd"].setup
        iface = setup.search_interface
        self.assertIsInstance(iface, NoiseSearchInterface)
        self.assertIs(iface.transform, setup.transform)
        self.assertEqual(iface.search_type, "point_estimate")
        self.assertEqual(iface.search_path, str(self.run_dir / "psd_point_build.npy"))
        np.testing.assert_array_equal(iface.spread, [1e-13, 1e-16])
        self.assertIsNone(iface.parameters_searched)  # "all"
        self.assertIsNone(iface.periodic)  # the noise prior has none
        self.assertEqual(iface.sampling_basis, ["S_oms", "S_tm"])

    def test_eryn_backend_flag_reaches_the_interface(self):
        path = self.run_dir / "psd_point_build.npy"
        np.save(path, np.array(self.POINT))
        for flag in (False, True):
            with self.subTest(is_eryn_backend=flag):
                search = {"search_type": "samples", "search_path": str(path), "is_eryn_backend": flag}
                _, built = self.build(priors=NOISE_PRIOR, search=search)
                self.assertEqual(built["psd"].setup.search_interface.is_eryn_backend, flag)

    def test_parameters_searched(self):
        _, built = self.build(priors=NOISE_PRIOR, search=self.point_search(parameters_searched=["S_tm"]))
        np.testing.assert_array_equal(built["psd"].setup.search_interface.parameters_searched, [1])

    def test_unknown_parameter_searched(self):
        with self.assertRaisesRegex(ValueError, r"parameters_searched \['S_xyz'\] are not in the prior basis"):
            self.build(priors=NOISE_PRIOR, search=self.point_search(parameters_searched=["S_xyz"]))

    def test_interface_survives_the_settings_deepcopy(self):
        # CurrentInfoGlobalFit deep-copies the settings
        from copy import deepcopy

        _, built = self.build(priors=NOISE_PRIOR, search=self.point_search())
        copy = deepcopy(built["psd"].setup.search_interface)
        self.assertEqual(copy.search_type, "point_estimate")

    def test_the_interface_seeds_walkers_around_the_point_inside_the_prior(self):
        _, built = self.build(priors=NOISE_PRIOR, search=self.point_search())
        setup = built["psd"].setup
        draws = setup.search_interface.generate_initial_draws(
            curr=None, num_walkers=6, betas=np.array([1.0, 0.5]), priors=setup.priors
        )
        self.assertEqual(draws.shape, (2, 6, 1, 2))  # (ntemps, nwalkers, nleaves, ndim)
        flat = draws.reshape(-1, 2)
        self.assertTrue(np.isfinite(setup.priors["psd"].logpdf(flat)).all())
        # hotter chains are scattered wider (covariance / beta), but all stay near the point
        self.assertTrue(np.all(np.abs(flat[:, 0] - self.POINT[0]) < 2e-12))
        self.assertTrue(np.all(np.abs(flat[:, 1] - self.POINT[1]) < 2e-15))


class ReferenceEquivalenceTest(_Built):
    """The psd block reproduces ``get_psd_erebor_settings`` (mojito_input/emri_mbh_psd_settings.py).

    Only the sampler side and the metadata are compared: the priors come from ereborpriors, whose
    stock ``NoisePrior`` is the ground truth (its bounds differ from the old settings file on purpose).
    """

    def reference(self):
        from lisatools.globalfit.stock.erebor import PSDSettings, PSDSetup

        g = self.general
        priors_psd = {
            r"$S_{\rm oms}$": UniformDistribution(6.0e-12, 20.0e-11),
            r"$S_{\rm tm}$": UniformDistribution(1.0e-15, 20.0e-14),
        }
        settings = PSDSettings(
            Tobs=g.Tobs, dt=g.dt, initialize_kwargs={}, log_dir=g.artifacts_file_dir,
            priors={"psd": ProbDistContainer(priors_psd)}, ndim=2,
            injection=np.array([15e-12, 3e-15]), num_prop_repeats=500,
        )
        metadata = StochasticMetadata(
            model_config=dict(use_splines=False, num_params=2),
            frequency_ranges=[(g.start_freq, g.end_freq)],
            prior_model="uniform",
            prior_model_code_link="",
            prior_model_config_file="",
        )
        return PSDSetup(settings), metadata

    def block(self):
        _, built = self.build(
            priors=NOISE_PRIOR, num_prop_repeats=500, injection={"S_oms": 15e-12, "S_tm": 3e-15}
        )
        return built["psd"]

    def test_sampler_side_matches(self):
        ref, _ = self.reference()
        got = self.block().setup
        self.assertEqual(got.ndim, ref.ndim)
        self.assertEqual(got.num_prop_repeats, ref.num_prop_repeats)
        self.assertEqual((got.Tobs, got.dt), (ref.Tobs, ref.dt))
        np.testing.assert_array_equal(got.injection, ref.injection)
        np.testing.assert_array_equal(got.betas, ref.betas)
        self.assertEqual(got.nleaves_max, ref.nleaves_max)
        self.assertEqual(got.nleaves_min, ref.nleaves_min)

    def test_metadata_matches(self):
        _, ref = self.reference()
        got = self.block().metadata
        self.assertEqual(got.model_config, ref.model_config)
        self.assertEqual(got.frequency_ranges, ref.frequency_ranges)



if __name__ == "__main__":
    unittest.main()
