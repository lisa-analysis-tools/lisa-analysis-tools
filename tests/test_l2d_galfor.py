"""The galfor block: no move of its own, sampled by the psd move on the psd ladder, through the galactic grid.

Schema and recipe tests reuse the recipe fixtures (stand-in infos and moves); the build tests run the real
``build_curr_info`` on ``FakeL1`` with a small galactic grid on CPU, and the real ``GalacticForegroundPrior``.
"""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import contextlib
import io
import types
import unittest
import warnings

import numpy as np
import yaml
from eryn.moves.tempering import make_ladder
from pydantic import ValidationError

from lisatools.globalfit.stock.l2d.blocks.galfor import GalForBlock
from lisatools.globalfit.stock.l2d.build_recipe import seed_branches
from lisatools.globalfit.stock.l2d.io import RunConfig, load_run
from lisatools.globalfit.stock.l2d.load_settings import build_curr_info

from tests.test_l2d_blocks import NOISE_PRIOR
from tests import test_l2d_recipe  # the module, not MoveBuildTest: its tests would run here too
from tests.test_l2d_recipe import PSD_BLOCK, _Runtime, fake_info
from tests.test_l2d_run import _Run, load_script, write_run_folder
from tests.test_l2d_sources import source_raw

GALFOR_PRIOR = {"class": "mojitolight.GalacticForegroundPrior"}
GALFOR_BLOCK = {"type": "galfor", "priors": GALFOR_PRIOR}
GRID = {"R_d": 2.18, "z_d": 0.48, "N_lambda": 9, "N_beta": 6}  # small: built on CPU in the build tests
XYZ_WITH_GRID = {"class": "XYZSensitivityBackend", "kwargs": {"tdi_generation": 2, "galactic_grid_kwargs": GRID}}
GALFOR_BASIS = ["log10_A", "alpha_gal", "log10_f1", "log10_fknee", "log10_f2"]


def point_search(run_dir, name, ndim):
    path = run_dir / f"{name}_point.npy"
    np.save(path, np.zeros(ndim))
    return {"search_type": "point_estimate", "search_path": str(path), "spread": 1e-3}


class _GalFor(_Runtime):
    def config(self, blocks, recipe=None, sensitivity=XYZ_WITH_GRID):
        raw = source_raw(blocks)
        raw["sensitivity"] = sensitivity
        if recipe is not None:
            raw["recipe"] = recipe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return RunConfig.model_validate(raw, context={"run_dir": self.run_dir})

    def assertGalforError(self, msg, blocks, sensitivity=XYZ_WITH_GRID):
        with self.assertRaises(ValidationError) as ctx:
            self.config(blocks, sensitivity=sensitivity)
        self.assertTrue(any(msg in e["msg"] for e in ctx.exception.errors()), ctx.exception.errors())


class GalForSchemaTest(_GalFor):
    def test_loads_next_to_psd(self):
        block = self.config({"psd": PSD_BLOCK, "galfor": GALFOR_BLOCK}).blocks["galfor"]
        self.assertIsInstance(block, GalForBlock)
        self.assertFalse(hasattr(block, "pe_move"))

    def test_move_fields_are_rejected(self):
        for field, value in (
            ("pe_move", {"class": "MultiGPUPSDMove"}), ("tempering", {"ratio": 2.0}), ("num_prop_repeats", 5),
            ("inner_moves", []), ("nleaves", 1),
        ):
            with self.subTest(field=field):
                self.assertRunError(
                    "extra_forbidden", f"blocks.galfor.galfor.{field}",
                    blocks={"psd": PSD_BLOCK, "galfor": {**GALFOR_BLOCK, field: value}},
                )

    def test_no_catalogue_search(self):
        self.assertGalforError(
            "is not available for the 'galfor' block",
            {"psd": PSD_BLOCK, "galfor": {**GALFOR_BLOCK, "search": {"search_type": "catalog", "spread": 1.0}}},
        )

    def test_needs_an_enabled_psd_block(self):
        for blocks in ({"galfor": GALFOR_BLOCK}, {"psd": {**PSD_BLOCK, "enabled": False}, "galfor": GALFOR_BLOCK}):
            with self.subTest(blocks=sorted(blocks)):
                self.assertGalforError("add an enabled psd block", blocks)

    def test_needs_the_galactic_grid(self):
        blocks = {"psd": PSD_BLOCK, "galfor": GALFOR_BLOCK}
        for sensitivity, msg in (
            ({"class": "XYZSensitivityBackend", "kwargs": {"tdi_generation": 2}}, "galactic_grid_kwargs"),
            ({"class": "XYZSensitivityBackend", "kwargs": {"galactic_grid_kwargs": {}}}, "galactic_grid_kwargs"),
            ({"class": "CompositeSensitivityBackend"}, "has none"),
        ):
            with self.subTest(sensitivity=sensitivity):
                self.assertGalforError(msg, blocks, sensitivity=sensitivity)

    def test_a_disabled_galfor_needs_nothing(self):
        cfg = self.config({"galfor": {**GALFOR_BLOCK, "enabled": False}}, sensitivity={"class": "CompositeSensitivityBackend"})
        self.assertFalse(cfg.blocks["galfor"].enabled)


class GalForMovesTest(_GalFor):
    moves = test_l2d_recipe.MoveBuildTest.moves

    def infos(self, ladder=(1.0, 0.5, 0.25, 0.1, 0.01)):
        return {"psd": fake_info(ladder, nleaves=1, ndim=2), "galfor": fake_info(ladder, nleaves=1, ndim=5)}

    def test_the_psd_move_samples_galfor(self):
        infos = self.infos()
        _, _, _, moves = self.moves({"psd": PSD_BLOCK, "galfor": GALFOR_BLOCK}, infos)
        self.assertEqual(list(moves), ["psd"])  # galfor has no move of its own
        for move in moves["psd"].search + moves["psd"].pe:
            self.assertIs(move.kwargs["galfor_transform_fn"], infos["galfor"].transform)
            self.assertIs(move.kwargs["psd_transform_fn"], infos["psd"].transform)

    def test_the_search_runs_if_either_block_needs_it(self):
        psd_point = {**PSD_BLOCK, "search": point_search(self.run_dir, "psd", 2)}
        galfor_point = {**GALFOR_BLOCK, "search": point_search(self.run_dir, "galfor", 5)}
        for blocks, searched in (
            ({"psd": psd_point, "galfor": GALFOR_BLOCK}, True),
            ({"psd": PSD_BLOCK, "galfor": galfor_point}, True),
            ({"psd": psd_point, "galfor": galfor_point}, False),
            ({"psd": psd_point, "galfor": {**GALFOR_BLOCK, "enabled": False}}, False),
        ):
            with self.subTest(blocks={n: b.get("search", {}).get("search_type") for n, b in blocks.items()}):
                infos = self.infos() if blocks["galfor"].get("enabled", True) else {"psd": self.infos()["psd"]}
                _, _, _, moves = self.moves(blocks, infos)
                self.assertEqual((len(moves["psd"].search), len(moves["psd"].pe)), (int(searched), 1))

    def test_galfor_is_seeded_on_its_ladder_and_stays_on(self):
        ladder = make_ladder(7, ntemps=5, Tmax=1e6)
        infos = self.infos(ladder)
        infos["galfor"].search_interface = self.fake_interface("galfor")
        blocks = {"psd": PSD_BLOCK, "galfor": {**GALFOR_BLOCK, "search": {"search_type": "prior"}}}
        cfg = self.config(blocks)
        curr, state = self.runtime(cfg, infos)
        wave_gens = seed_branches(cfg, {}, curr, object(), {}, state)
        (args, _), = infos["galfor"].search_interface.populate_state.call_args_list
        np.testing.assert_array_equal(args[2], ladder)
        self.assertEqual((self.events, wave_gens), ([("populate", "galfor")], {}))
        self.assertTrue(state.branches_inds["galfor"].all())


class GalForBuildTest(_Run):
    def folder(self, psd=None, galfor=None):
        run = write_run_folder(self.root / "run", psd=psd, sensitivity=XYZ_WITH_GRID)
        (run / "blocks" / "galfor.yaml").write_text(yaml.safe_dump(galfor or GALFOR_BLOCK))
        return run

    def build(self, **blocks):
        cfg = load_run(self.folder(**blocks))
        return cfg, build_curr_info(cfg)

    def test_galfor_is_on_the_psd_ladder(self):
        cfg, curr = self.build()
        ladder = make_ladder(2, ntemps=cfg.sampler.ntemps, Tmax=1e6)  # the reference psd ladder
        np.testing.assert_allclose(curr.source_info["psd"].betas, ladder)
        np.testing.assert_allclose(curr.source_info["galfor"].betas, ladder)

    def test_a_disabled_galfor_is_not_built(self):
        cfg, curr = self.build(galfor={**GALFOR_BLOCK, "enabled": False})
        self.assertEqual(curr.branch_names, ["psd"])
        np.testing.assert_allclose(curr.source_info["psd"].betas, make_ladder(2, ntemps=cfg.sampler.ntemps, Tmax=1e6))

    def test_an_explicit_psd_ladder_is_kept_for_both(self):
        cfg, curr = self.build(psd={"type": "psd", "priors": NOISE_PRIOR, "tempering": {"ratio": 2.0}})
        ladder = 0.5 ** np.arange(cfg.sampler.ntemps)
        np.testing.assert_allclose(curr.source_info["psd"].betas, ladder)
        np.testing.assert_allclose(curr.source_info["galfor"].betas, ladder)

    def test_setup_and_metadata(self):
        from lisatools.globalfit.searches import GalacticForegroundSearchInterface
        from lisatools.globalfit.stock.erebor import GalForSetup

        injection = dict(zip(GALFOR_BASIS, [-44.0, 2.0, -3.0, -2.5, -3.5]))
        galfor = {**GALFOR_BLOCK, "injection": injection, "search": {"search_type": "prior"}}
        _, curr = self.build(galfor=galfor)
        setup = curr.source_info["galfor"]
        self.assertIsInstance(setup, GalForSetup)
        self.assertEqual((setup.ndim, setup.nleaves_max, setup.nleaves_min), (5, 1, 1))
        self.assertEqual(curr.engine_info.ndims, {"galfor": 5, "psd": 2})
        self.assertEqual(type(setup.priors["galfor"]).__name__, "GalacticForegroundPrior")
        self.assertEqual(setup.transform.input_basis, GALFOR_BASIS)
        np.testing.assert_array_equal(setup.injection, [-44.0, 2.0, -3.0, -2.5, -3.5])
        self.assertIsInstance(setup.search_interface, GalacticForegroundSearchInterface)

        metadata = curr.settings_dict.source_metadata["galfor"]
        self.assertEqual(metadata.model_config, dict(num_params=5, galactic_grid_kwargs=GRID))
        self.assertEqual(metadata.prior_model, "GalacticForegroundPrior")

    def test_dry_run(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = load_script().main(["--config", str(self.folder()), "--dry-run"])
        self.assertEqual(code, 0)
        out = stdout.getvalue()
        self.assertIn("galfor: 1 leaves x 5 params, ladder 5, no search, sampled by the psd move", out)
        self.assertIn("psd: 1 leaves x 2 params, ladder 5, no search, pe_move MultiGPUPSDMove", out)


if __name__ == "__main__":
    unittest.main()
