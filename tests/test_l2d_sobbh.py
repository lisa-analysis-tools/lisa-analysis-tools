"""Tests for the L2D ``sobbh`` block (gwtf waveform): schema, builder-filled kwargs, shared build, PE move."""

# Pre-existing env shims (see tests/test_l2d_ports.py): the installed scienceplots raises at import,
# and eryn.utils.plot uses a bare ``typing`` name. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import types
import unittest
import warnings
from unittest import mock

import numpy as np
from pydantic import ValidationError

import lisatools.globalfit.moves as gf_moves
from lisatools.globalfit.stock.l2d.blocks import base as blocks_base
from lisatools.globalfit.stock.l2d.blocks.base import BuildContext, make_proposal_priors, waveform_metadata_config
from lisatools.globalfit.stock.l2d.blocks.sobbh import SOBBHBlock
from lisatools.globalfit.stock.l2d.build_recipe import RecipeContext, build_block_moves
from lisatools.globalfit.stock.l2d.generators import SignalGen, WaveformHandle
from lisatools.globalfit.stock.l2d.io import RunConfig

from tests.test_l2d_recipe import NWALKERS, FakeWaveform, Recorder, _Runtime, fake_info
from tests.test_l2d_sources import _Built, _Sources, source_raw
from tests.test_l2d_time_reference import DATA_T0, fake_general

try:  # the real GWTF_generator module needs cupy + pygwtf
    from lisatools.sources.sobbh.waveform_gwtf import GWTF_generator

    HAVE_GWTF = True
except ImportError:
    HAVE_GWTF = False

SOBBH_BLOCK = {"type": "sobbh", "search": {"search_type": "catalog", "spread": [1e-4] * 11}}
SOBBH_IDS = {"source_types": ["noise", "mbhb", "emri", "sobhb"], "source_ids": {"mbhb": [18], "emri": [0], "sobhb": [0, 3, 4]}}


def sobbh_raw(block=None, **data_changes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return source_raw({"sobbh": block or SOBBH_BLOCK}, **{**SOBBH_IDS, **data_changes})


@unittest.skipUnless(HAVE_GWTF, "cupy / pygwtf not installed")
class SOBBHSchemaTest(_Sources):
    def config(self, block=None, **data_changes):
        return RunConfig.model_validate(sobbh_raw(block, **data_changes), context={"run_dir": self.run_dir})

    def test_defaults(self):
        block = self.config().blocks["sobbh"]
        self.assertIsInstance(block, SOBBHBlock)
        self.assertEqual(block.priors.class_name, "mojitolight.SOBHBPrior")
        self.assertEqual((block.waveform.class_name, block.waveform.kwargs), ("GWTF_generator", {"fresnel_kernel_width": 16}))
        self.assertEqual(block.pe_move.class_name, "SOBBHSpecialMove")
        self.assertEqual(block.time_reference, "data_start")
        td = block.td_generator()
        self.assertEqual((td.waveform_class.__name__, td.method), ("SOBBHTaylorT3TDIWaveform", "get_td_signals"))
        self.assertFalse(td.init_kwargs["phase_at_coalescence"])  # the phase is phi_0, at the data start

    def test_catalog_search_at_the_data_start_is_allowed(self):
        self.assertEqual(self.config().blocks["sobbh"].search.search_type, "catalog")

    def test_rejects(self):
        cases = {
            "runtime kwargs": {**SOBBH_BLOCK, "waveform": {"class": "GWTF_generator", "runtime_kwargs": {"a": 1}}},
            "engine-filled kwarg": {**SOBBH_BLOCK, "waveform": {"class": "GWTF_generator", "kwargs": {"T_obs": 1.0}}},
            "unknown kwarg": {**SOBBH_BLOCK, "waveform": {"class": "GWTF_generator", "kwargs": {"waveform_t0": 1.0}}},
            "mojito time reference": {**SOBBH_BLOCK, "time_reference": "mojito"},
            "generic builder": {**SOBBH_BLOCK, "pe_move": {"class": "SOBBHMoveBuilder"}},
        }
        for name, block in cases.items():
            with self.subTest(name), self.assertRaises(ValidationError):
                self.config(block)

    def test_engine_waveform_kwargs(self):
        cfg = self.config()
        block = cfg.blocks["sobbh"]
        for window_type, alpha, expected in (("tukey", 0.01, 0.01), ("tukey", 0.0, None), ("hann", 0.01, None)):
            with self.subTest(window_type=window_type, alpha=alpha):
                g = fake_general()
                g.window_type, g.window_alpha, g.domain_settings, g.Tobs = window_type, alpha, object(), 3.0e6
                kwargs = block.engine_waveform_kwargs(BuildContext(cfg, g))
                self.assertEqual(set(kwargs), set(SOBBHBlock.ENGINE_WAVEFORM_KWARGS))
                self.assertIs(kwargs["settings"], g.domain_settings)
                self.assertEqual((kwargs["T_obs"], kwargs["use_GPU"], kwargs["tukey_alpha"]), (3.0e6, False, expected))
                self.assertIs(kwargs["orbits"], g.orbits)
        # the TD generator keeps the shared engine kwargs, referenced to the data start
        g = fake_general()
        td = blocks_base.resolve_td_generator(block, BuildContext(cfg, g))
        self.assertEqual(td.init_kwargs["waveform_t0"], DATA_T0)
        self.assertIs(td.init_kwargs["data_td_settings"], g.data_td_settings)
        self.assertNotIn("settings", td.init_kwargs)

    def test_metadata_drops_the_live_objects(self):
        init = dict(settings=object(), orbits=object(), T_obs=1.0, use_GPU=False, tukey_alpha=None, fresnel_kernel_width=16)
        config = waveform_metadata_config(init, "data_start")
        self.assertEqual(set(config), {"T_obs", "use_GPU", "tukey_alpha", "fresnel_kernel_width", "force_backend", "time_reference"})


@unittest.skipUnless(HAVE_GWTF, "cupy / pygwtf not installed")
class SOBBHBuildTest(_Built):
    def test_setup_and_settings(self):
        from ereborpriors.stock.mojitolight import SOBHBPrior

        from lisatools.globalfit.searches import SOBHBSearchInterface
        from lisatools.globalfit.stock.erebor import SOBBHTFSetup

        block = {**SOBBH_BLOCK, "search": {"search_type": "prior"}}  # the fake data has no catalogue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, built = self.build({"sobbh": block}, **SOBBH_IDS)
        build = built["sobbh"]
        setup = build.setup
        self.assertIsInstance(setup, SOBBHTFSetup)
        self.assertIsInstance(setup.priors["sobbh"], SOBHBPrior)
        self.assertEqual((setup.ndim, setup.nleaves_max, setup.nleaves_min), (11, 3, 3))
        self.assertIsInstance(setup.search_interface, SOBHBSearchInterface)
        self.assertIs(build.waveform_class, GWTF_generator)
        # the waveform is built lazily, from the block-filled kwargs
        gen = setup.signal_gen
        self.assertIsInstance(gen, SignalGen)
        self.assertEqual(gen.method, "get_signals_for_residuals")
        self.assertIsInstance(gen.source, WaveformHandle)
        self.assertFalse(gen.source.built)
        self.assertEqual(
            set(gen.source.init_kwargs), {"fresnel_kernel_width"} | set(SOBBHBlock.ENGINE_WAVEFORM_KWARGS)
        )
        self.assertIs(gen.source.init_kwargs["settings"], self.general.domain_settings)
        self.assertEqual(gen.source.init_kwargs["T_obs"], self.general.Tobs)
        self.assertEqual(setup.initialize_kwargs, gen.source.init_kwargs)
        config = build.metadata.waveform_model_config
        self.assertNotIn("settings", config)
        self.assertNotIn("orbits", config)
        self.assertEqual(config["time_reference"], "data_start")
        self.assertEqual(build.td_generator.init_kwargs["waveform_t0"], self.general.data_t0)


@unittest.skipUnless(HAVE_GWTF, "cupy / pygwtf not installed")
class SOBBHMoveTest(_Runtime):
    def test_special_move(self):
        gen = FakeWaveform()
        info = fake_info([1.0, 0.5], nleaves=3, ndim=11)
        cfg = RunConfig.model_validate(sobbh_raw(), context={"run_dir": self.run_dir})
        curr, state = self.runtime(cfg, {"sobbh": info})
        engine_info = types.SimpleNamespace(ndims={"sobbh": 11})
        acs, priors = object(), {"stand-in": "priors"}
        recorder = type("SOBBHSpecialMove", (Recorder,), {"moves": []})
        with mock.patch.multiple(gf_moves, SOBBHSpecialMove=recorder):
            moves = build_block_moves(RecipeContext(cfg, {"sobbh": gen}), engine_info, curr, acs, priors, state)
        (move,) = moves["sobbh"].pe
        kw = move.kwargs
        self.assertIs(kw.pop("dcga"), acs)
        self.assertIs(kw.pop("waveform_gen"), gen)
        self.assertIs(kw.pop("priors"), priors)
        self.assertIs(kw.pop("transform_fn"), info.transform)
        self.assertIs(kw.pop("inner_moves"), info.inner_moves)
        np.testing.assert_array_equal(kw.pop("betas_all"), np.tile([1.0, 0.5], (3, 1)))
        self.assertEqual(kw.pop("waveform_gen_kwargs"), info.waveform_kwargs)
        # no waveform_gen_method / waveform_like_method: SOBBHSpecialMove sets its own
        self.assertEqual(kw, dict(
            branch_name="sobbh", coords_shape=(2, NWALKERS, 3, 11), waveform_like_kwargs={}, num_repeats=7,
            permute_every=0, pad_out_of_prior=True, run_async=True, run_threaded=True, randomize_split=True,
            tc_min_days=60,
        ))


PROPOSAL_PRIORS = {"class": "mojitolight.SOBHBPrior"}


@unittest.skipUnless(HAVE_GWTF, "cupy / pygwtf not installed")
class SOBBHProposalPriorsTest(_Sources):
    """``proposal_priors``: the per-source tiles of SOBHBPrior, mapped to the leaves by source id."""

    def config(self, block_changes=None, **data_changes):
        block = {**SOBBH_BLOCK, "proposal_priors": PROPOSAL_PRIORS, **(block_changes or {})}
        return RunConfig.model_validate(sobbh_raw(block, **data_changes), context={"run_dir": self.run_dir})

    def test_schema(self):
        spec = self.config().blocks["sobbh"].proposal_priors
        self.assertEqual((spec.class_name, spec.factory, spec.kwargs), ("mojitolight.SOBHBPrior", "per_source_prior_factory", {}))
        cases = {
            "unknown factory": {"proposal_priors": {**PROPOSAL_PRIORS, "factory": "no_such_factory"}},
            "unknown kwarg": {"proposal_priors": {**PROPOSAL_PRIORS, "kwargs": {"f0_min": 1e-3}}},
            "chirp-mass order": {"proposal_priors": {**PROPOSAL_PRIORS, "kwargs": {"sort_with_chirp_mass": True}}},
            "fixed nleaves": {"nleaves": 3},
            "prior search": {"search": {"search_type": "prior"}},
        }
        for name, changes in cases.items():
            with self.subTest(name), self.assertRaises(ValidationError):
                self.config(changes)

    def test_leaves_get_the_tiles_of_their_source_ids(self):
        from ereborpriors.stock.mojitolight import SOBHBPrior, sobhb

        cfg = self.config()  # source ids [0, 3, 4]
        leaf_priors = make_proposal_priors(cfg.blocks["sobbh"], cfg, SOBHBPrior.prior_factory())
        self.assertEqual(len(leaf_priors), 3)
        for source_id, prior in zip((0, 3, 4), leaf_priors):
            f0 = prior.rvs(200)[:, prior.get_input_basis().index("f0")]
            lo, hi = sobhb.TILES_F0_MIN[source_id], sobhb.TILES_F0_MAX[source_id]
            self.assertTrue(np.all((f0 >= lo) & (f0 <= hi)), source_id)

    def test_rejects_missing_tiles_and_other_bases(self):
        from ereborpriors.stock.mojitolight import SOBHBPrior

        cfg = self.config(source_ids={"mbhb": [18], "emri": [0], "sobhb": [0, 7]})
        with self.assertRaisesRegex(ValueError, r"source ids \[7\]"):
            make_proposal_priors(cfg.blocks["sobbh"], cfg, SOBHBPrior.prior_factory())
        cfg = self.config()
        other = types.SimpleNamespace(get_input_basis=lambda: ["Mc", "eta"])
        with self.assertRaisesRegex(ValueError, "samples"):
            make_proposal_priors(cfg.blocks["sobbh"], cfg, other)

    def test_only_this_branch_s_move_gets_them(self):
        from ereborpriors.stock.mojitolight import SOBHBPrior

        cfg = self.config()
        info = fake_info([1.0], nleaves=3, ndim=11)
        curr = types.SimpleNamespace(general_info=types.SimpleNamespace(nwalkers=NWALKERS, ntemps=cfg.sampler.ntemps),
                                     source_info={"sobbh": info})
        state = types.SimpleNamespace(sub_states={"sobbh": types.SimpleNamespace()})
        shared = {"sobbh": SOBHBPrior.prior_factory(), "mbh": object()}
        recorder = type("SOBBHSpecialMove", (Recorder,), {"moves": []})
        with mock.patch.multiple(gf_moves, SOBBHSpecialMove=recorder):
            moves = build_block_moves(
                RecipeContext(cfg, {"sobbh": FakeWaveform()}), types.SimpleNamespace(ndims={"sobbh": 11}),
                curr, object(), shared, state,
            )
        priors = moves["sobbh"].pe[0].kwargs["priors"]
        self.assertEqual(len(priors["sobbh"]), 3)
        self.assertIs(priors["mbh"], shared["mbh"])
        self.assertIsInstance(shared["sobbh"], SOBHBPrior)  # the shared dict is not changed


if __name__ == "__main__":
    unittest.main()
