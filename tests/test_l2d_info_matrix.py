"""Tests for the information matrix in the L2D blocks: ``search.spread: info_matrix`` and the block's
``info_matrix`` settings, from the YAML to the Settings and the search interface.

The information matrix only scatters the starting points: ``InfoMatrixMove`` and a proposal basis are refused.
"""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import unittest
from copy import deepcopy
from unittest import mock

import numpy as np

from lisatools.globalfit import searches
from lisatools.globalfit.infomatrix import InfoMatrixGenerator
from lisatools.globalfit.stock.l2d.io import dump_run, load_run

from tests.test_l2d_galfor import GALFOR_BLOCK
from tests.test_l2d_gb import _GBConfig, gb_search
from tests.test_l2d_recipe import PSD_BLOCK, _Recipe

from tests.test_l2d_sources import EMRI_BLOCK, MBH_BLOCK, MBH_CATALOGUE_ENTRY, _Built

INFO_SEARCH = {"search_type": "catalog", "spread": "info_matrix"}
INFO_MOVES = [
    {"class": "StretchMove", "weight": 0.9},
    {"class": "InfoMatrixMove", "kwargs": {"scale": 2.0}, "weight": 0.1},
]
SPREAD = [1e-4, 1e-3] + [1e-3] * 9  # the mbh basis has 11 columns


def mbh(**changes):
    return {"mbh": {**MBH_BLOCK, **changes}}


class InfoMatrixSchemaTest(_Recipe):
    def assertBlockError(self, msg, block, loc="blocks.mbh.mbh"):
        self.assertRunError("value_error", loc, msg=msg, blocks=block)

    def test_spread_loads_for_point_seeds_over_all_parameters(self):
        point = self.run_dir / "point.npy"
        np.save(point, np.zeros(11))
        for search in (INFO_SEARCH, {"search_type": "point_estimate", "search_path": str(point), "spread": "info_matrix"}):
            with self.subTest(search_type=search["search_type"]):
                block = self.config(mbh(search=search)).blocks["mbh"]
                self.assertEqual(block.search.spread, "info_matrix")
                self.assertIsNone(block.info_matrix)  # the settings are optional

    def test_numeric_spreads_load_as_before(self):
        for spread in (1e-3, SPREAD, np.eye(11).tolist()):
            block = self.config(mbh(search={"search_type": "catalog", "spread": spread})).blocks["mbh"]
            self.assertEqual(block.search.spread, spread)

    def test_another_string_is_not_a_spread(self):
        with self.assertRaises(ValueError):
            self.config(mbh(search={"search_type": "catalog", "spread": "infomatrix"}))

    def test_spread_is_for_point_seeds_only(self):
        samples = self.run_dir / "samples.npy"
        np.save(samples, np.zeros((5, 11)))
        for search in (
            {"search_type": "prior", "spread": "info_matrix"},
            {"search_type": "samples", "search_path": str(samples), "spread": "info_matrix"},
        ):
            with self.subTest(search_type=search["search_type"]):
                self.assertBlockError(
                    "scatters around one point per source", mbh(search=search), loc="blocks.mbh.mbh.search"
                )

    def test_spread_needs_all_parameters(self):
        search = {**INFO_SEARCH, "parameters_searched": ["logM", "Q"]}
        self.assertBlockError("needs parameters_searched: all", mbh(search=search), loc="blocks.mbh.mbh.search")

    def test_settings_alone_are_refused(self):
        for search in (None, {"search_type": "catalog", "spread": SPREAD}):
            with self.subTest(search=search):
                self.assertBlockError(
                    "info_matrix: the block uses no information matrix",
                    mbh(search=search, info_matrix={"eps_every": 5}),
                )

    def test_settings_fields(self):
        block = self.config(mbh(search=INFO_SEARCH, info_matrix={})).blocks["mbh"]
        self.assertEqual(
            block.info_matrix.model_dump(), dict(more_accurate=True, steps=None, eps_every=None, distance_cap=None),
        )
        steps = [[1e-3, 1e-4], [1e-5, 1e-6]]
        given = dict(more_accurate=False, steps=steps, eps_every=3, distance_cap=0.5)
        block = self.config(mbh(search=INFO_SEARCH, info_matrix=given)).blocks["mbh"]
        self.assertEqual(block.info_matrix.model_dump(), given)

    def test_bad_settings(self):
        loc = "blocks.mbh.mbh.info_matrix"
        self.assertRunError(
            "extra_forbidden", f"{loc}.num_prior_draws", blocks=mbh(search=INFO_SEARCH, info_matrix={"num_prior_draws": 5})
        )
        self.assertRunError(
            "greater_than", f"{loc}.eps_every", blocks=mbh(search=INFO_SEARCH, info_matrix={"eps_every": 0})
        )
        for cap in (0, 0.0, -1.0):
            with self.subTest(distance_cap=cap):
                self.assertRunError(
                    "greater_than", f"{loc}.distance_cap",
                    blocks=mbh(search=INFO_SEARCH, info_matrix={"distance_cap": cap}),
                )

    def test_distance_cap_loads_for_the_starting_scatter(self):
        block = self.config(mbh(search=INFO_SEARCH, info_matrix={"distance_cap": 1.0})).blocks["mbh"]
        self.assertEqual(block.info_matrix.distance_cap, 1.0)

    def test_distance_cap_needs_the_starting_scatter(self):
        for search in (None, {"search_type": "catalog", "spread": SPREAD}):
            with self.subTest(search=search):
                self.assertBlockError(
                    "info_matrix: the block uses no information matrix",
                    mbh(search=search, info_matrix={"distance_cap": 1.0}),
                )

    def test_inner_move_is_refused(self):
        # the information matrix only seeds: the PE move does not supply the move's covariances
        for search in (None, INFO_SEARCH, {"search_type": "catalog", "spread": SPREAD}):
            with self.subTest(search=search):
                self.assertBlockError(
                    "InfoMatrixMove is not available as an inner move", mbh(search=search, inner_moves=INFO_MOVES)
                )

    def test_proposal_basis_is_refused(self):
        self.assertRunError(
            "extra_forbidden", "blocks.mbh.mbh.info_matrix.proposal_basis",
            blocks=mbh(search=INFO_SEARCH, info_matrix={"proposal_basis": "chirp"}),
        )

    def test_generator_is_not_a_pe_move_argument(self):
        for cls in ("TDMBHSpecialMove", "MBHMoveBuilder"):
            with self.subTest(cls=cls):
                with self.assertRaises(ValueError):
                    self.config(mbh(pe_move={"class": cls, "kwargs": {"info_matrix_gen": 1}}))

    def test_per_leaf_priors_load_with_the_information_matrix_spread(self):
        # the starting scatter takes each leaf's own prior
        proposal = {"class": "mojitolight.MBHBPrior", "factory": "prior_factory"}
        block = self.config(mbh(proposal_priors=proposal, search=INFO_SEARCH)).blocks["mbh"]
        self.assertEqual(block.search.spread, "info_matrix")

    def test_fallback_spread_loads_with_the_information_matrix_spread(self):
        search = {**INFO_SEARCH, "fallback_spread": SPREAD}
        block = self.config(mbh(search=search)).blocks["mbh"]
        self.assertEqual(block.search.fallback_spread, SPREAD)
        self.assertIsNone(self.config(mbh(search=INFO_SEARCH)).blocks["mbh"].search.fallback_spread)
        self.assertRunError(
            "value_error", "blocks.mbh.mbh.search", msg="fallback_spread is the numeric spread of the sources whose",
            blocks=mbh(search={"search_type": "catalog", "spread": SPREAD, "fallback_spread": SPREAD}),
        )

    def test_psd_takes_no_information_matrix(self):
        point = self.run_dir / "psd_point.npy"
        np.save(point, np.array([1.5e-11, 3e-15]))
        search = {"search_type": "point_estimate", "search_path": str(point), "spread": "info_matrix"}
        self.assertRunError(
            "value_error", "blocks.psd.psd", msg="the 'psd' block has no information matrix",
            blocks={"psd": {**PSD_BLOCK, "search": search}},
        )
        self.assertRunError(
            "value_error", "blocks.psd.psd", msg="InfoMatrixMove is not available as an inner move",
            blocks={"psd": {**PSD_BLOCK, "inner_moves": [{"class": "InfoMatrixMove"}]}},
        )
        self.assertRunError(
            "value_error", "blocks.galfor.galfor", msg="the 'galfor' block has no information matrix",
            blocks={"psd": PSD_BLOCK, "galfor": {**GALFOR_BLOCK, "search": search}},
        )

    def test_dump_reloads_the_keys(self):
        info_matrix = {"more_accurate": False, "steps": [1e-3, 1e-5], "eps_every": 4, "distance_cap": 1.5}
        cfg = self.config(mbh(search=INFO_SEARCH, info_matrix=info_matrix))
        again = load_run(dump_run(cfg, self.run_dir / "resolved" / "run.yaml"))
        block = again.blocks["mbh"]
        self.assertEqual(block.search.spread, "info_matrix")
        self.assertEqual(block.inner_moves, cfg.blocks["mbh"].inner_moves)
        self.assertEqual(block.info_matrix.model_dump(), info_matrix)
        self.assertEqual(block, cfg.blocks["mbh"])

    def test_dump_reloads_blocks_without_the_keys(self):
        # the snapshot writes `info_matrix: null` for every source block: that must not count as "given"
        cfg = self.config({"psd": PSD_BLOCK, "mbh": {**MBH_BLOCK, "search": {"search_type": "catalog", "spread": SPREAD}}})
        again = load_run(dump_run(cfg, self.run_dir / "resolved" / "run.yaml"))
        self.assertIsNone(again.blocks["mbh"].info_matrix)
        self.assertEqual(again.blocks["mbh"], cfg.blocks["mbh"])


class GBInfoMatrixSchemaTest(_GBConfig):
    """GBGPU registers no ``Settings.signal_gen``."""

    def test_spread(self):
        self.assertConfigError("the 'gb' block has no information matrix", {"search": gb_search(spread="info_matrix")})

    def test_settings(self):
        self.assertConfigError(
            "the 'gb' block has no information matrix", {"search": gb_search(), "info_matrix": {"eps_every": 2}}
        )

    def test_inner_move(self):
        self.assertConfigError(
            "InfoMatrixMove is not available as an inner move",
            {"search": gb_search(), "inner_moves": [{"class": "InfoMatrixMove"}]},
        )

    def test_dump_reloads_a_gb_block(self):
        cfg = self.config({"search": gb_search()})
        again = load_run(dump_run(cfg, self.run_dir / "resolved" / "run.yaml"))
        self.assertEqual(again.blocks["gb"], cfg.blocks["gb"])


class InfoMatrixBuildTest(_Built):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.general.catalogue = {"MBHB": {18: MBH_CATALOGUE_ENTRY}}

    def test_no_feature_no_generator(self):
        search = {"search_type": "catalog", "spread": SPREAD}
        _, built = self.build({"mbh": {**MBH_BLOCK, "search": search}, "emri": EMRI_BLOCK})
        self.assertIsNone(built["mbh"].setup.info_matrix_gen)
        self.assertIsNone(built["emri"].setup.info_matrix_gen)
        self.assertEqual(built["mbh"].setup.search_interface.spread.shape, (11,))

    def test_spread_gives_the_settings_a_generator(self):
        _, built = self.build(mbh(search=INFO_SEARCH))
        setup = built["mbh"].setup
        gen = setup.info_matrix_gen
        self.assertIsInstance(gen, InfoMatrixGenerator)
        # the defaults of the generator, and the branch prior
        self.assertEqual((gen.more_accurate, gen.steps, gen.eps_every), (True, None, None))
        self.assertIsNone(gen.proposal_basis)
        self.assertIs(gen.prior, setup.priors["mbh"])

    def test_fallback_spread_reaches_the_interface(self):
        _, built = self.build(mbh(search={**INFO_SEARCH, "fallback_spread": SPREAD}))
        interface = built["mbh"].setup.search_interface
        np.testing.assert_array_equal(interface.fallback_spread, np.asarray(SPREAD))
        self.assertIsNone(interface.leaf_priors)
        _, built = self.build(mbh(search=INFO_SEARCH))
        self.assertIsNone(built["mbh"].setup.search_interface.fallback_spread)

    def test_per_leaf_priors_reach_the_interface_and_the_generator(self):
        from lisatools.globalfit.stock.l2d.blocks import base

        proposal = {"class": "mojitolight.MBHBPrior", "factory": "prior_factory"}
        tiles = [object(), object(), object()]  # stand-ins: the build hands the list on, it does not use it
        with mock.patch.object(base, "make_proposal_priors", return_value=tiles) as factory:
            _, built = self.build(mbh(proposal_priors=proposal, search=INFO_SEARCH))
        setup = built["mbh"].setup
        factory.assert_called()
        # one list for the two: the seeds of a leaf are drawn inside the prior that bounds its matrix
        self.assertIs(setup.search_interface.leaf_priors, tiles)
        self.assertIs(setup.info_matrix_gen.prior, tiles)
        self.assertTrue(setup.info_matrix_gen.per_leaf_priors)
        # a numeric spread: the interface still redraws into the leaves' priors, and there is no generator
        with mock.patch.object(base, "make_proposal_priors", return_value=tiles):
            _, built = self.build(mbh(proposal_priors=proposal, search={"search_type": "catalog", "spread": SPREAD}))
        self.assertIs(built["mbh"].setup.search_interface.leaf_priors, tiles)
        self.assertIsNone(built["mbh"].setup.info_matrix_gen)

    def test_generator_takes_the_block_settings(self):
        steps = [1e-3, 1e-5, 1e-7]
        info_matrix = {"more_accurate": False, "steps": steps, "eps_every": 4}
        cfg, built = self.build(mbh(search=INFO_SEARCH, info_matrix=info_matrix))
        gen = built["mbh"].setup.info_matrix_gen
        self.assertEqual((gen.more_accurate, gen.steps, gen.eps_every), (False, steps, 4))
        self.assertIsNot(gen.steps, cfg.blocks["mbh"].info_matrix.steps)  # the config is not shared with the run

    def test_distance_cap_reaches_the_generator(self):
        _, built = self.build(mbh(search=INFO_SEARCH, info_matrix={"distance_cap": 1.5}))
        self.assertEqual(built["mbh"].setup.info_matrix_gen.distance_cap, 1.5)

    def test_no_distance_cap_by_default(self):
        for changes in ({"search": INFO_SEARCH}, {"search": INFO_SEARCH, "info_matrix": {}}):
            with self.subTest(changes=sorted(changes)):
                _, built = self.build(mbh(**changes))
                self.assertIsNone(built["mbh"].setup.info_matrix_gen.distance_cap)

    def test_generator_prior_gives_a_prior_precision(self):
        _, built = self.build(mbh(search=INFO_SEARCH))
        precision = built["mbh"].setup.info_matrix_gen.prior_precision
        self.assertEqual(precision.shape, (11,))
        self.assertTrue(np.all(np.isfinite(precision) & (precision > 0)))

    def test_emri_is_the_same_build(self):
        point = self.run_dir / "emri_point.npy"
        np.save(point, np.zeros(1))  # read only when the run seeds
        search = {"search_type": "point_estimate", "search_path": str(point), "spread": "info_matrix"}
        _, built = self.build({"emri": {**EMRI_BLOCK, "search": search, "info_matrix": {"eps_every": 2}}})
        setup = built["emri"].setup
        self.assertIsInstance(setup.info_matrix_gen, InfoMatrixGenerator)
        self.assertEqual(setup.info_matrix_gen.eps_every, 2)
        self.assertIs(setup.info_matrix_gen.prior, setup.priors["emri"])

    def test_search_interface_gets_the_string(self):
        _, built = self.build(mbh(search=INFO_SEARCH, info_matrix={"eps_every": 3}))
        setup = built["mbh"].setup
        self.assertEqual(setup.search_interface.spread, "info_matrix")
        self.assertIsInstance(setup.search_interface.spread, str)
        # what the interface reads from curr.source_info[branch] when it seeds
        self.assertIsNotNone(setup.signal_gen)
        self.assertEqual(setup.info_matrix_gen.eps_every, 3)
        # the inner moves are untouched
        self.assertEqual([type(m).__name__ for m, _ in setup.inner_moves], ["StretchMove", "SkyMove", "SkyMove"])

    def test_generator_after_the_engines_deep_copy(self):
        # CurrentInfoGlobalFit deep-copies the settings once: the copy's generator still holds the copy's prior
        _, built = self.build(mbh(search=INFO_SEARCH))
        setup = built["mbh"].setup
        copy = deepcopy(setup)
        self.assertIsNot(copy.info_matrix_gen, setup.info_matrix_gen)
        self.assertIs(copy.info_matrix_gen.prior, copy.priors["mbh"])


class SearchInterfaceSpreadTest(unittest.TestCase):
    """``make_search_interface`` hands the interface the string as it is, and numbers as an array."""

    def interface(self, spread):
        from lisatools.globalfit.stock.l2d.blocks import base

        made = {}

        class Interface:
            def __init__(self, **kwargs):
                made.update(kwargs)

        spec = base.SearchSpec(search_type="catalog", spread=spread)
        block = mock.Mock(search=spec, SEARCH_INTERFACE="Interface")
        info = dict(basis=["a", "b"], transform_fn=None, periodic=None)
        with mock.patch.object(searches, "Interface", Interface, create=True):
            base.make_search_interface(block, info)
        return made

    def test_string(self):
        self.assertEqual(self.interface("info_matrix")["spread"], "info_matrix")

    def test_numbers(self):
        spread = self.interface([1.0, 2.0])["spread"]
        self.assertIsInstance(spread, np.ndarray)
        np.testing.assert_array_equal(spread, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
