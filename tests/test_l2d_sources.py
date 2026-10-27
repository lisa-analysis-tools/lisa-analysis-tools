"""Tests for the L2D source blocks (mbh, emri): schema, run-level checks, shared build, sky wiring."""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pydantic import ValidationError

from eryn.moves import StretchMove
from eryn.priors import UniformDistribution
from eryn.utils.transform import TransformContainer
from ereborpriors.base import GFBasePrior
from ereborpriors.stock import mojitolight
from lisatools.detector import EqualArmlengthOrbits
from lisatools.globalfit.postprocessing import SourceMetadata
from lisatools.globalfit.stock.l2d import load_settings
from lisatools.globalfit.stock.l2d.blocks import base as blocks_base
from lisatools.globalfit.stock.l2d.blocks.mbh import MBHBlock, make_sky_wrappers, sky_ind_map
from lisatools.globalfit.stock.l2d.io import RunConfig
from lisatools.globalfit.stock.l2d.load_settings import load_blocks, load_general_setup
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.sources.waveformbase import TDPyResponseWaveformBase
from lisatools.utils.constants import YRSID_SI

from tests.test_l2d_blocks import cpu_raw, error_keys
from tests.test_l2d_load_settings import FakeL1

REFERENCE_TIME = 97729089.327664  # MOJITO_REFERENCE_TIME
EMRI_WAVEFORM_MODULE = "lisatools.sources.emri.waveform"

MBH_BLOCK = {
    "type": "mbh",
    "priors": {"class": "mojitolight.MBHBPrior"},
    "waveform": {
        "class": "PhenomTHMTDIWaveform",
        "kwargs": {
            "waveform_kwargs": {
                "higher_modes": [21, 33, 44], "include_negative_modes": True, "t_low_fit": True,
                "coarse_grain": False, "atol": 1e-12, "rtol": 1e-12,
            },
            "tdi_generation": "2nd generation", "tdi_channels": "XYZ", "order": 30,
            "Tobs": YRSID_SI / 12.0, "start_freq": 7e-5, "use_reference_time": True,
            "buffer_time": 15_000, "fft_batch_size": 2,
        },
    },
    "num_prop_repeats": 50,
    "tempering": {},
    "inner_moves": [
        {"class": "StretchMove", "weight": 0.35},
        {"class": "SkyMove", "weight": 0.05, "kwargs": {"which": "lat"}},
        {"class": "SkyMove", "weight": 0.05, "kwargs": {"which": "both"}},
    ],
}

EMRI_BLOCK = {
    "type": "emri",
    "priors": {"class": "mojitolight.FakeEMRIPrior"},
    "waveform": {
        "class": "EMRITDIWaveform",
        "kwargs": {
            "waveform_kwargs": {"sum_kwargs": {"pad_output": True}},
            "tdi_generation": "2nd generation", "tdi_channels": "XYZ", "order": 30,
            "buffer_time": 15_000, "fft_batch_size": 1,
        },
        "runtime_kwargs": {"mode_selection_threshold": 1e-4},
    },
    "tempering": {"ntemps": 1},
}


class FakeEMRIPrior(GFBasePrior):
    """Stands in for the (not yet written) ``EMRIPrior``: branch ``emri``, one periodic angle."""

    @classmethod
    def prior_factory(cls):
        priors_in = {
            "log_m1": UniformDistribution(np.log(1e5), np.log(5e6)),
            "m2": UniformDistribution(1.0, 200.0),
            "phiK": UniformDistribution(0.0, 2 * np.pi, boundary="periodic"),
        }
        return cls(branch_name="emri", priors_in=priors_in)

    def get_transforms(self):
        basis = self.get_input_basis()
        return TransformContainer(input_basis=basis, output_basis=basis, parameter_transforms={})

    def get_periodic(self):
        return {"phiK": 2 * np.pi}


def make_fake_emri_waveform_module():
    """A stand-in for ``lisatools.sources.emri.waveform`` (it needs ``few``, not always installed)."""

    class EMRITDIWaveform(TDPyResponseWaveformBase):
        def __init__(self, waveform_class="FastKerrEccentricEquatorialFlux", waveform_kwargs=None, *args, **kwargs):
            raise RuntimeError("stand-in class: it is looked up, never constructed")

    module = types.ModuleType(EMRI_WAVEFORM_MODULE)
    module.EMRITDIWaveform = EMRITDIWaveform
    return module


def source_raw(blocks=None, **data_changes):
    raw = cpu_raw(blocks)
    raw["data"] = {
        **raw["data"],
        "source_types": ["noise", "mbhb", "emri"],
        "source_ids": {"mbhb": [18, 5, 16], "emri": [0, 3]},
        "reference_time": REFERENCE_TIME,
        **data_changes,
    }
    return raw


class _Sources(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run_dir = Path(tmp.name)

        # the emri waveform module and prior: stand-ins (see their docstrings)
        previous = _sys.modules.get(EMRI_WAVEFORM_MODULE)
        _sys.modules[EMRI_WAVEFORM_MODULE] = make_fake_emri_waveform_module()
        self.addCleanup(
            lambda: _sys.modules.pop(EMRI_WAVEFORM_MODULE, None) if previous is None
            else _sys.modules.__setitem__(EMRI_WAVEFORM_MODULE, previous)
        )
        real_prior_class = blocks_base.prior_class
        patcher = mock.patch.object(
            blocks_base, "prior_class",
            lambda name: FakeEMRIPrior if name == "mojitolight.FakeEMRIPrior" else real_prior_class(name),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def config(self, blocks, **data_changes):
        raw = source_raw(blocks, **data_changes)
        return RunConfig.model_validate(raw, context={"run_dir": self.run_dir})

    def assertConfigError(self, err_type, loc, msg=None, blocks=None, **data_changes):
        with self.assertRaises(ValidationError) as ctx:
            self.config(blocks, **data_changes)
        matches = [
            err for err in ctx.exception.errors()
            if err["type"] == err_type and ".".join(map(str, err["loc"])) == loc
        ]
        self.assertTrue(matches, f"no {err_type} at {loc!r}; got {error_keys(ctx.exception)}")
        if msg is not None:
            self.assertTrue(
                any(msg in err["msg"] for err in matches), f"{msg!r} not in {[e['msg'] for e in matches]}"
            )


def mbh_block(**changes):
    return {"mbh": {**MBH_BLOCK, **changes}}


def with_waveform_kwargs(block, **extra):
    waveform = {**block["waveform"], "kwargs": {**block["waveform"]["kwargs"], **extra}}
    return {**block, "waveform": waveform}


class SourceSchemaTest(_Sources):
    def test_blocks_load(self):
        cfg = self.config({"mbh": MBH_BLOCK, "emri": EMRI_BLOCK})
        self.assertIsInstance(cfg.blocks["mbh"], MBHBlock)
        self.assertEqual(cfg.blocks["mbh"].nleaves, "auto")
        self.assertEqual(cfg.blocks["emri"].waveform.runtime_kwargs, {"mode_selection_threshold": 1e-4})
        self.assertEqual(cfg.data.reference_time, REFERENCE_TIME)

    def test_unknown_waveform_class(self):
        block = {**MBH_BLOCK, "waveform": {"class": "Nope", "kwargs": {}}}
        self.assertConfigError("value_error", "blocks.mbh.mbh", msg="no class 'Nope'", blocks={"mbh": block})

    def test_engine_filled_waveform_kwarg(self):
        block = with_waveform_kwargs(MBH_BLOCK, waveform_t0=1.0)
        self.assertConfigError(
            "value_error", "blocks.mbh.mbh", msg="filled in by the builder", blocks={"mbh": block}
        )

    def test_misspelled_waveform_kwarg(self):
        block = with_waveform_kwargs(MBH_BLOCK, fft_batchsize=2)
        self.assertConfigError(
            "value_error", "blocks.mbh.mbh", msg="does not accept ['fft_batchsize']", blocks={"mbh": block}
        )

    def test_skymove_geometry_is_filled_by_the_builder(self):
        moves = [{"class": "SkyMove", "kwargs": {"ind_map": {"psi": 7}}}]
        self.assertConfigError(
            "value_error", "blocks.mbh.mbh", msg="filled in by the builder", blocks=mbh_block(inner_moves=moves)
        )

    def test_flow_moves_are_not_supported_yet(self):
        moves = [{"class": "ConditionalFlowMove"}]
        self.assertConfigError(
            "value_error", "blocks.mbh.mbh", msg="flow moves are not supported yet",
            blocks=mbh_block(inner_moves=moves),
        )

    def test_unimportable_waveform_module_is_a_validation_error(self):
        # e.g. `few` missing on this machine: a clean config error instead of a ModuleNotFoundError
        _sys.modules[EMRI_WAVEFORM_MODULE] = None
        self.assertConfigError("value_error", "blocks.emri.emri", msg="cannot import", blocks={"emri": EMRI_BLOCK})

    def test_reference_time_is_required(self):
        self.assertConfigError(
            "value_error", "", msg="needs data.reference_time", blocks=mbh_block(), reference_time=None
        )

    def test_nleaves_auto_needs_source_ids(self):
        self.assertConfigError(
            "value_error", "", msg="nleaves: auto needs data.source_ids['mbhb']",
            blocks=mbh_block(), source_ids={"emri": [0]},
        )

    def test_explicit_nleaves_needs_no_source_ids(self):
        cfg = self.config(mbh_block(nleaves=3), source_ids={})
        self.assertEqual(cfg.blocks["mbh"].nleaves, 3)

    def test_source_type_must_be_in_data_source_types(self):
        self.assertConfigError(
            "value_error", "", msg="needs 'mbhb' in data.source_types", blocks=mbh_block(), source_types=["noise"]
        )

    def test_source_types_compare_case_insensitively(self):
        self.config(mbh_block(), source_types=["NOISE", "MBHB"])

    def test_tempering_defaults_to_a_ladder(self):
        block = {k: v for k, v in MBH_BLOCK.items() if k != "tempering"}
        spec = self.config({"mbh": block}).blocks["mbh"].tempering
        self.assertEqual((spec.ntemps, spec.ratio), (None, 1.2))  # None -> sampler.ntemps

    def test_tempering_cannot_be_null_for_a_source(self):
        # the Setup's own default ladder (24 rungs) cannot seed a state with fewer temperatures
        with self.assertRaises(ValidationError) as ctx:
            self.config({"mbh": {**MBH_BLOCK, "tempering": None}})
        self.assertIn("blocks.mbh.mbh.tempering", [loc for _, loc in error_keys(ctx.exception)])

    def test_ladder_may_use_a_subset_of_the_sampler_temperatures(self):
        sampler_ntemps = self.config({}).sampler.ntemps
        for ntemps in (1, 2, sampler_ntemps):
            with self.subTest(ntemps=ntemps):
                self.config({"mbh": {**MBH_BLOCK, "tempering": {"ntemps": ntemps}}})

    def test_ladder_cannot_have_more_rungs_than_the_sampler(self):
        too_many = self.config({}).sampler.ntemps + 1
        self.assertConfigError(
            "value_error", "", msg="is above sampler.ntemps",
            blocks={"mbh": {**MBH_BLOCK, "tempering": {"ntemps": too_many}}},
        )

    def test_disabled_blocks_skip_the_run_checks(self):
        cfg = self.config(mbh_block(enabled=False), reference_time=None, source_ids={}, source_types=["noise"])
        self.assertFalse(cfg.blocks["mbh"].enabled)


class _Built(_Sources):
    """A real CPU GeneralSetup (only the mojito file I/O is faked), shared by the build tests."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.shared_dir = Path(cls._tmp.name)
        cfg = RunConfig.model_validate(source_raw(), context={"run_dir": cls.shared_dir})
        with mock.patch.dict(load_settings.LOADERS, {"mojito_l1": FakeL1}):
            cls.general = load_general_setup(cfg)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def build(self, blocks, **data_changes):
        cfg = self.config(blocks, **data_changes)
        return cfg, load_blocks(cfg, self.general)


class MBHBuildTest(_Built):
    def test_setup_and_settings(self):
        from lisatools.globalfit.stock.erebor import MBHSetup

        cfg, built = self.build({"mbh": MBH_BLOCK})
        setup = built["mbh"].setup
        self.assertIsInstance(setup, MBHSetup)
        self.assertEqual((setup.nleaves_min, setup.nleaves_max), (3, 3))  # len(source_ids["mbhb"])
        self.assertEqual(setup.ndim, 11)
        self.assertEqual(setup.num_prop_repeats, 50)
        self.assertEqual((setup.Tobs, setup.dt), (self.general.Tobs, self.general.dt))
        self.assertEqual(setup.log_dir, self.general.artifacts_file_dir)
        self.assertEqual(setup.waveform_kwargs, {})

    def test_priors_transform_and_periodic_are_keyed_by_the_branch(self):
        _, built = self.build({"mbh": MBH_BLOCK})
        setup = built["mbh"].setup
        self.assertEqual(list(setup.priors), ["mbh"])
        self.assertIsInstance(setup.priors["mbh"], mojitolight.MBHBPrior)
        self.assertEqual(list(setup.periodic), ["mbh"])
        self.assertEqual(set(setup.periodic["mbh"]), {"phi_ref", "psi", "alpha"})
        self.assertIsNotNone(setup.transform)

    def test_prior_of_another_branch_is_rejected(self):
        block = {**MBH_BLOCK, "priors": {"class": "mojitolight.FakeEMRIPrior"}}
        with self.assertRaisesRegex(ValueError, "prior of the 'emri' branch, but this is the 'mbh' block"):
            self.build({"mbh": block})

    def test_tempering_ladder(self):
        cfg, built = self.build({"mbh": MBH_BLOCK})  # tempering: {} -> ratio 1.2 over sampler.ntemps
        np.testing.assert_allclose(
            built["mbh"].setup.betas, 1.2 ** -np.arange(cfg.sampler.ntemps, dtype=float)
        )

    def test_no_tempering_key_gives_the_same_ladder_not_the_setups_default(self):
        block = {k: v for k, v in MBH_BLOCK.items() if k != "tempering"}
        cfg, built = self.build({"mbh": block})
        betas = built["mbh"].setup.betas
        self.assertEqual(len(betas), cfg.sampler.ntemps)  # the Setup default would have 24 rungs
        np.testing.assert_allclose(betas, 1.2 ** -np.arange(cfg.sampler.ntemps, dtype=float))

    def test_subset_ladder(self):
        _, built = self.build({"mbh": {**MBH_BLOCK, "tempering": {"ntemps": 3, "ratio": 2.0}}})
        np.testing.assert_allclose(built["mbh"].setup.betas, [1.0, 0.5, 0.25])

    def test_waveform_init_kwargs_get_the_engine_values(self):
        _, built = self.build({"mbh": MBH_BLOCK})
        init = built["mbh"].setup.initialize_kwargs
        g = self.general
        self.assertEqual(init["waveform_t0"], REFERENCE_TIME)
        self.assertEqual(init["sampling_frequency"], 1.0 / g.dt)
        self.assertIs(init["data_td_settings"], g.data_td_settings)
        self.assertIs(init["orbits"], g.orbits)  # CPU run: not gpu_orbits
        self.assertEqual(init["tukey_alpha"], g.window_alpha)
        self.assertEqual(init["stft_dt"], g.stft_dt)
        self.assertEqual((init["freq_min"], init["freq_max"]), (g.start_freq, g.end_freq))
        self.assertEqual(init["force_backend"], g.force_backend)
        # the user's kwargs are kept
        self.assertEqual(init["fft_batch_size"], 2)
        self.assertEqual(init["waveform_kwargs"]["higher_modes"], [21, 33, 44])

    def test_config_is_not_changed_by_the_build(self):
        cfg, _ = self.build({"mbh": MBH_BLOCK})
        kwargs = cfg.blocks["mbh"].waveform.kwargs
        self.assertNotIn("waveform_t0", kwargs)
        self.assertNotIn("orbits", kwargs)

    def test_waveform_class_is_resolved_not_constructed(self):
        from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform

        _, built = self.build({"mbh": MBH_BLOCK})
        self.assertIs(built["mbh"].waveform_class, PhenomTHMTDIWaveform)

    def test_metadata(self):
        _, built = self.build({"mbh": MBH_BLOCK})
        meta = built["mbh"].metadata
        self.assertIsInstance(meta, SourceMetadata)
        self.assertEqual(meta.source_type, "MBHB")
        self.assertEqual(meta.waveform_model, "PhenomTHMTDIWaveform")
        self.assertEqual(meta.frequency_ranges, [(self.general.start_freq, self.general.end_freq)])
        config = meta.waveform_model_config
        self.assertEqual(config["force_backend"], "cpu")
        self.assertNotIn("data_td_settings", config)
        self.assertNotIn("orbits", config)

    def test_default_inner_moves_are_the_setups(self):
        _, built = self.build({"mbh": {k: v for k, v in MBH_BLOCK.items() if k != "inner_moves"}})
        moves = built["mbh"].setup.inner_moves
        self.assertEqual([type(m).__name__ for m, _ in moves], ["StretchMove"])

    def test_skymove_gets_the_geometry_from_the_prior_basis(self):
        _, built = self.build({"mbh": MBH_BLOCK})
        moves = built["mbh"].setup.inner_moves
        self.assertEqual([type(m).__name__ for m, _ in moves], ["StretchMove", "SkyMove", "SkyMove"])
        self.assertEqual([w for _, w in moves], [0.35, 0.05, 0.05])
        basis = mojitolight.MBHBPrior.prior_factory(observation_time=1.0).get_input_basis()
        expected = {"cosinc": basis.index("cos_iota"), "psi": basis.index("psi"), "lam": basis.index("alpha"),
                    "sinbeta": basis.index("sin_delta"), "t_ref": basis.index("t_c")}
        for move, _ in moves[1:]:
            self.assertIsInstance(move, SkyMove)
            self.assertEqual(move.ind_map, expected)
            # the sampled sky coordinates are ICRS; the hop itself runs in the LISA frame through
            # the two callables, so SkyMove's own (default) frame converters are never used
            self.assertEqual(move.coord_frame, "icrs")
            self.assertIn("t_ref", move.ind_map)  # required by SkyMove when coord_frame != "lisa"
        self.assertEqual([m.which for m, _ in moves[1:]], ["lat", "both"])

    def test_builds_do_not_share_state(self):
        # the old mbhb.py filled SkyMove from a module-global dict that every build mutated
        _, first = self.build({"mbh": MBH_BLOCK})
        _, second = self.build({"mbh": MBH_BLOCK})
        a = first["mbh"].setup.inner_moves[1][0]
        b = second["mbh"].setup.inner_moves[1][0]
        self.assertIsNot(a.ind_map, b.ind_map)
        self.assertIsNot(a.to_lisa_frame, b.to_lisa_frame)


MBH_CATALOGUE_ENTRY = dict(  # m1 = 8e5, m2 = 2e5: total mass 1e6, Q = m1 / m2 = 4; distance in Mpc
    PrimaryMassSSBFrame=8e5, SecondaryMassSSBFrame=2e5, PrimarySpinCompZ=0.3, SecondarySpinCompZ=-0.2,
    LuminosityDistance=20000.0, PhaseReferenceSourceFrame=1.0, InclinationAngle=0.7,
    RightAscension=2.0, Declination=0.3, PolarisationAngle=1.1,
    TimeCoalescencePhenomTPHMSSBFrame=5.0e4,  # inside the prior's t_c window: [0, Tobs] and Tobs is 1 day here
)


def mbh_with_search(**search):
    return {**MBH_BLOCK, "search": {"search_type": "catalog", "spread": [1e-4, 1e-3] + [1e-3] * 9, **search}}


class SourceSearchSchemaTest(_Sources):
    def test_catalog_needs_a_spread(self):
        block = {**MBH_BLOCK, "search": {"search_type": "catalog"}}
        self.assertConfigError("value_error", "blocks.mbh.mbh.search", msg="needs spread", blocks={"mbh": block})

    def test_distribution_search_is_not_available_yet(self):
        block = {**MBH_BLOCK, "search": {"search_type": "distribution", "search_path": "."}}
        self.assertConfigError(
            "value_error", "blocks.mbh.mbh", msg="not available for the 'mbh' block", blocks={"mbh": block}
        )

    def test_search_loads(self):
        cfg = self.config({"mbh": mbh_with_search()})
        self.assertEqual(cfg.blocks["mbh"].search.search_type, "catalog")


class SourceSearchBuildTest(_Built):
    def test_no_search_means_no_interface(self):
        _, built = self.build({"mbh": MBH_BLOCK, "emri": EMRI_BLOCK})
        self.assertIsNone(built["mbh"].setup.search_interface)
        self.assertIsNone(built["emri"].setup.search_interface)

    def test_mbh_interface(self):
        from lisatools.globalfit.searches import MBHBSearchInterface

        _, built = self.build({"mbh": mbh_with_search()})
        setup = built["mbh"].setup
        iface = setup.search_interface
        self.assertIsInstance(iface, MBHBSearchInterface)
        self.assertIs(iface.transform, setup.transform)
        self.assertEqual(iface.search_type, "catalog")
        self.assertEqual(iface.spread.shape, (11,))
        # the periodicity reaches the interface as {sampling-basis column: period}
        basis = iface.sampling_basis
        self.assertEqual(
            iface.periodic, {basis.index(n): 2 * np.pi for n in ("phi_ref", "psi", "alpha")}
        )
        # the interface names the same catalogue and branch as the block
        self.assertEqual(iface.catalog_source_name, MBHBlock.CATALOGUE_KEY)
        self.assertEqual(iface.branch_name, "mbh")

    def test_emri_interface(self):
        from lisatools.globalfit.searches import EMRISearchInterface

        block = {**EMRI_BLOCK, "search": {"search_type": "catalog", "spread": 1e-3}}
        _, built = self.build({"emri": block})
        iface = built["emri"].setup.search_interface
        self.assertIsInstance(iface, EMRISearchInterface)
        self.assertEqual(iface.branch_name, "emri")
        self.assertEqual(iface.periodic, {2: 2 * np.pi})  # phiK
        self.assertEqual(iface.catalog_source_name, "EMRI")

    def test_catalogue_seeds_through_the_prior_inverse_transform(self):
        # The MBH catalogue holds physical values; the interface must bring them to the sampling basis of
        # MBHBPrior (ln M, ln q, d_L in Gpc, cos iota, sin delta, ...), where q = m1 / m2.
        _, built = self.build({"mbh": mbh_with_search()})
        setup = built["mbh"].setup
        iface = setup.search_interface
        curr = types.SimpleNamespace(general_info=types.SimpleNamespace(catalogue={"MBHB": {18: MBH_CATALOGUE_ENTRY}}))

        seed = iface.get_catalog_values(curr)
        e = MBH_CATALOGUE_ENTRY
        expected = [
            np.log(1e6), np.log(4.0), e["PrimarySpinCompZ"], e["SecondarySpinCompZ"], 20.0,
            e["PhaseReferenceSourceFrame"], np.cos(e["InclinationAngle"]), e["PolarisationAngle"],
            e["RightAscension"], np.sin(e["Declination"]), e["TimeCoalescencePhenomTPHMSSBFrame"],
        ]
        np.testing.assert_allclose(seed, [expected], rtol=1e-12)
        self.assertTrue(np.isfinite(setup.priors["mbh"].logpdf(seed)).all())

    def test_scattered_walkers_stay_in_the_prior(self):
        _, built = self.build({"mbh": mbh_with_search()})
        setup = built["mbh"].setup
        iface = setup.search_interface
        curr = types.SimpleNamespace(general_info=types.SimpleNamespace(catalogue={"MBHB": {18: MBH_CATALOGUE_ENTRY}}))
        draws = iface.generate_initial_draws(
            curr=curr, num_walkers=4, betas=np.array([1.0, 0.5]), priors=setup.priors
        )
        self.assertEqual(draws.shape, (2, 4, 1, 11))
        self.assertTrue(np.isfinite(setup.priors["mbh"].logpdf(draws.reshape(-1, 11))).all())


class SkyWiringTest(unittest.TestCase):
    BASIS = mojitolight.MBHBPrior.prior_factory(observation_time=1.0).get_input_basis()

    def test_ind_map_from_basis(self):
        ind_map = sky_ind_map(self.BASIS, MBHBlock.SKY_BASIS)
        self.assertEqual(ind_map, dict(cosinc=6, psi=7, lam=8, sinbeta=9, t_ref=10))

    def test_ind_map_follows_a_reordered_basis(self):
        basis = list(reversed(self.BASIS))
        ind_map = sky_ind_map(basis, MBHBlock.SKY_BASIS)
        self.assertEqual(ind_map["t_ref"], 0)
        self.assertEqual(basis[ind_map["lam"]], "alpha")

    def test_missing_basis_name(self):
        with self.assertRaisesRegex(ValueError, r"lacks \['alpha'\]"):
            sky_ind_map([n for n in self.BASIS if n != "alpha"], MBHBlock.SKY_BASIS)

    def test_wrappers_round_trip_and_touch_only_the_sky_columns(self):
        ind_map = sky_ind_map(self.BASIS, MBHBlock.SKY_BASIS)
        to_lisa, from_lisa = make_sky_wrappers(EqualArmlengthOrbits(), REFERENCE_TIME, ind_map)

        rng = np.random.default_rng(3)
        n = 50
        coords = rng.uniform(0.1, 0.9, size=(n, len(self.BASIS)))
        coords[:, ind_map["t_ref"]] = rng.uniform(0.0, 3e7, n)
        coords[:, ind_map["lam"]] = rng.uniform(0.2, 2 * np.pi - 0.2, n)
        coords[:, ind_map["sinbeta"]] = rng.uniform(-0.9, 0.9, n)
        coords[:, ind_map["psi"]] = rng.uniform(0.2, np.pi - 0.2, n)
        before = coords.copy()

        lisa = to_lisa(coords)
        np.testing.assert_array_equal(coords, before)  # the input is not modified
        sky = [ind_map[k] for k in ("t_ref", "lam", "sinbeta", "psi")]
        other = [i for i in range(len(self.BASIS)) if i not in sky]
        np.testing.assert_array_equal(lisa[:, other], coords[:, other])
        self.assertFalse(np.allclose(lisa[:, sky], coords[:, sky]))  # the frame change did something

        back = from_lisa(lisa)
        np.testing.assert_allclose(back[:, other], coords[:, other])
        for column in (ind_map["t_ref"], ind_map["sinbeta"]):
            np.testing.assert_allclose(back[:, column], coords[:, column], rtol=1e-9, atol=1e-9)
        # angles come back modulo their period
        np.testing.assert_allclose(np.cos(back[:, ind_map["lam"]]), np.cos(coords[:, ind_map["lam"]]), atol=1e-9)
        np.testing.assert_allclose(np.cos(2 * back[:, ind_map["psi"]]), np.cos(2 * coords[:, ind_map["psi"]]), atol=1e-9)


class MBHTransformTest(unittest.TestCase):
    """The stock MBHBPrior transform reproduces the old settings file's sampling -> waveform maps."""

    def test_matches_the_reference_formulas(self):
        prior = mojitolight.MBHBPrior.prior_factory(observation_time=1e7)
        transform = prior.get_transforms()
        samples = prior.rvs(size=(200,))
        out = transform.transform_base_parameters(samples)

        logM, logq, s1z, s2z, d_L, phi_ref, cos_iota, psi, alpha, sin_delta, t_c = samples.T
        M, Q = np.exp(logM), np.exp(logq)  # Q = m1/m2 >= 1
        m2 = M / (1 + Q)  # reference `mT_Q`
        m1 = Q * m2
        expected = np.column_stack([
            m1, m2, s1z, s2z, d_L * 1e3, phi_ref, np.arccos(cos_iota), psi, alpha, np.arcsin(sin_delta), t_c,
        ])
        np.testing.assert_allclose(out, expected, rtol=1e-12)
        self.assertTrue(np.all(out[:, 0] >= out[:, 1]))  # m1 >= m2

    def test_inverse_round_trip(self):
        prior = mojitolight.MBHBPrior.prior_factory(observation_time=1e7)
        transform = prior.get_transforms()
        samples = prior.rvs(size=(200,))
        back = transform.inverse_transform_base_parameters(transform.transform_base_parameters(samples))
        np.testing.assert_allclose(back, samples, rtol=1e-9, atol=1e-12)


class EMRIBuildTest(_Built):
    def test_emri_is_the_same_build(self):
        from lisatools.globalfit.stock.erebor import EMRISetup

        cfg, built = self.build({"emri": EMRI_BLOCK})
        setup = built["emri"].setup
        self.assertIsInstance(setup, EMRISetup)
        self.assertEqual((setup.nleaves_min, setup.nleaves_max), (2, 2))  # len(source_ids["emri"])
        self.assertEqual(setup.ndim, 3)
        self.assertEqual(list(setup.priors), ["emri"])
        self.assertEqual(setup.periodic, {"emri": {"phiK": 2 * np.pi}})
        np.testing.assert_allclose(setup.betas, [1.0])  # tempering: {ntemps: 1}
        self.assertEqual(setup.waveform_kwargs, {"mode_selection_threshold": 1e-4})
        self.assertEqual(built["emri"].metadata.source_type, "EMRI")
        self.assertEqual(built["emri"].metadata.waveform_model, "EMRITDIWaveform")
        self.assertEqual(built["emri"].waveform_class.__name__, "EMRITDIWaveform")

    def test_same_engine_kwargs_as_mbh(self):
        _, built = self.build({"mbh": MBH_BLOCK, "emri": EMRI_BLOCK})
        engine = blocks_base.ENGINE_WAVEFORM_KWARGS
        mbh_init = built["mbh"].setup.initialize_kwargs
        emri_init = built["emri"].setup.initialize_kwargs
        self.assertEqual({k: mbh_init[k] for k in engine}.keys(), {k: emri_init[k] for k in engine}.keys())
        for key in engine:
            self.assertEqual(mbh_init[key] is emri_init[key] or mbh_init[key] == emri_init[key], True, key)

    def test_a_second_block_type_orders_alphabetically(self):
        cfg = self.config({"mbh": MBH_BLOCK, "emri": EMRI_BLOCK})
        self.assertEqual(sorted(cfg.blocks), ["emri", "mbh"])


@unittest.skipUnless(hasattr(mojitolight, "EMRIPrior"), "ereborpriors has no EMRIPrior yet")
class EMRIPriorContractTest(unittest.TestCase):
    """What the emri block expects from ``ereborpriors.stock.mojitolight.EMRIPrior``."""

    INPUT_BASIS = ["log_m1", "m2", "a", "p0", "e0", "dist", "cosqK", "phiK", "Phi_phi0", "Phi_r0", "alpha", "sin_delta"]

    def prior(self):
        try:
            return mojitolight.EMRIPrior.prior_factory()
        except TypeError:
            return mojitolight.EMRIPrior.prior_factory(observation_time=1e7)

    def test_branch_basis_and_periodicity(self):
        prior = self.prior()
        self.assertEqual(prior.branch_name, "emri")
        self.assertEqual(prior.get_input_basis(), self.INPUT_BASIS)
        periodic = prior.get_periodic()
        self.assertTrue({"phiK", "Phi_phi0", "Phi_r0", "alpha"} <= set(periodic))
        self.assertTrue(set(periodic) <= set(self.INPUT_BASIS))

    def test_transform_reaches_the_waveform_basis(self):
        prior = self.prior()
        out = prior.get_transforms().transform_base_parameters(prior.rvs(size=(50,)))
        self.assertEqual(out.shape, (50, 14))  # adds the filled xI0 and Phi_theta0


if __name__ == "__main__":
    unittest.main()
