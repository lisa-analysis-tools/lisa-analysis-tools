"""Tests for the L2D recipe: ``pe_move`` / ``recipe:`` schema, seeding, move building, steps, setup function.

The engine objects the recipe works on (acs, state, curr) need a full run; here they are stand-ins that
record what the recipe does to them, and the move classes are replaced by recorders. Building the real
moves against a real acs is left to a GPU-machine dry run.
"""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
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
import lisatools.globalfit.recipe as gf_recipe
from eryn.moves.tempering import TemperatureControl, make_ladder
from lisatools.globalfit.stock.l2d import build_recipe
from lisatools.globalfit.stock.l2d.build_recipe import (
    BranchMoves, RecipeContext, StockRecipe, build_block_moves, make_setup_function, pad_ladder, psd_needs_search,
    seed_branches,
)
from lisatools.globalfit.stock.l2d.io import RecipeFile, RunConfig, StockRecipeSpec, dump_run, load_run

from tests.test_l2d_blocks import NOISE_PRIOR, error_keys
from tests.test_l2d_sources import EMRI_BLOCK, MBH_BLOCK, _Sources, mbh_with_search, source_raw

PSD_BLOCK = {"type": "psd", "priors": NOISE_PRIOR}
EMRI_SEARCH = {"search_type": "catalog", "spread": 1e-3}


def emri_with_search(**block):
    return {**EMRI_BLOCK, "search": EMRI_SEARCH, **block}


class Recorder:
    """Stands in for a move class: records its constructor arguments."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


class _Recipe(_Sources):
    def config(self, blocks, recipe=None, **data_changes):
        raw = source_raw(blocks, **data_changes)
        if recipe is not None:
            raw["recipe"] = recipe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # source blocks without a search warn
            return RunConfig.model_validate(raw, context={"run_dir": self.run_dir})

    def assertRunError(self, err_type, loc, msg=None, blocks=None, recipe=None):
        with self.assertRaises(ValidationError) as ctx:
            self.config(blocks or {}, recipe)
        matches = [e for e in ctx.exception.errors() if e["type"] == err_type and ".".join(map(str, e["loc"])) == loc]
        self.assertTrue(matches, f"no {err_type} at {loc!r}; got {error_keys(ctx.exception)}")
        if msg is not None:
            self.assertTrue(any(msg in e["msg"] for e in matches), [e["msg"] for e in matches])


# --- schema --- #

class PEMoveSchemaTest(_Recipe):
    def test_defaults_are_the_reference_moves(self):
        cfg = self.config({"psd": PSD_BLOCK, "mbh": MBH_BLOCK, "emri": EMRI_BLOCK})
        on = dict(run_async=True, run_threaded=True)
        expected = {
            "psd": ("MultiGPUPSDMove", dict(permute_every=100, live_dangerously=True, **on)),
            "mbh": ("TDMBHSpecialMove", dict(permute_every=100, pad_out_of_prior=True, randomize_split=True, **on)),
            "emri": ("EMRISpecialMove", dict(
                permute_every=25, pad_out_of_prior=True, randomize_split=True, batch_size_per_gpu=5, **on
            )),
        }
        for name, (cls, kwargs) in expected.items():
            with self.subTest(block=name):
                self.assertEqual(cfg.blocks[name].pe_move.class_name, cls)
                self.assertEqual(cfg.blocks[name].pe_move.kwargs, kwargs)

    def test_defaults_are_fresh_per_block(self):
        a = self.config({"mbh": MBH_BLOCK}).blocks["mbh"].pe_move
        b = self.config({"mbh": MBH_BLOCK}).blocks["mbh"].pe_move
        self.assertIsNot(a.kwargs, b.kwargs)

    def test_both_families(self):
        cases = {
            "psd": (PSD_BLOCK, ("MultiGPUPSDMove", "PSDMoveBuilder")),
            "mbh": (MBH_BLOCK, ("TDMBHSpecialMove", "MBHMoveBuilder")),
            "emri": (EMRI_BLOCK, ("EMRISpecialMove", "EMRIMoveBuilder")),
        }
        for name, (block, classes) in cases.items():
            for cls in classes:
                with self.subTest(block=name, cls=cls):
                    cfg = self.config({name: {**block, "pe_move": {"class": cls}}})
                    self.assertEqual(cfg.blocks[name].pe_move.kwargs, {})

    def test_class_of_another_block_or_unknown(self):
        for cls in ("EMRISpecialMove", "MultiGPUPSDMove", "StretchMove", "NoSuchMove"):
            with self.subTest(cls=cls):
                self.assertRunError(
                    "value_error", "blocks.mbh.mbh", msg="is not available here",
                    blocks={"mbh": {**MBH_BLOCK, "pe_move": {"class": cls}}},
                )

    def test_builder_filled_names_are_rejected(self):
        cases = [
            ("mbh", MBH_BLOCK, "TDMBHSpecialMove", "betas_all"),
            ("mbh", MBH_BLOCK, "TDMBHSpecialMove", "coords_shape"),
            ("mbh", MBH_BLOCK, "TDMBHSpecialMove", "Tmax"),
            ("emri", EMRI_BLOCK, "EMRISpecialMove", "waveform_gen_method"),
            ("mbh", MBH_BLOCK, "MBHMoveBuilder", "betas"),
            ("mbh", MBH_BLOCK, "MBHMoveBuilder", "wave_gen"),
            ("mbh", MBH_BLOCK, "MBHMoveBuilder", "coords_shape"),  # what the builder hands the move
            ("psd", PSD_BLOCK, "MultiGPUPSDMove", "temperature_control"),
            ("psd", PSD_BLOCK, "MultiGPUPSDMove", "use_gpu"),
            ("psd", PSD_BLOCK, "MultiGPUPSDMove", "max_logl_mode"),
            ("psd", PSD_BLOCK, "PSDMoveBuilder", "Tmax"),
            ("psd", PSD_BLOCK, "PSDMoveBuilder", "num_repeats"),
        ]
        for name, block, cls, kwarg in cases:
            with self.subTest(cls=cls, kwarg=kwarg):
                self.assertRunError(
                    "value_error", f"blocks.{name}.{name}", msg="filled in by the builder",
                    blocks={name: {**block, "pe_move": {"class": cls, "kwargs": {kwarg: 1}}}},
                )

    def test_misspelled_kwargs_are_rejected(self):
        for name, block, cls in (
            ("mbh", MBH_BLOCK, "TDMBHSpecialMove"), ("mbh", MBH_BLOCK, "MBHMoveBuilder"),
            ("psd", PSD_BLOCK, "MultiGPUPSDMove"), ("psd", PSD_BLOCK, "PSDMoveBuilder"),
        ):
            with self.subTest(cls=cls):
                self.assertRunError(
                    "value_error", f"blocks.{name}.{name}", msg="does not accept ['permute_evry']",
                    blocks={name: {**block, "pe_move": {"class": cls, "kwargs": {"permute_evry": 10}}}},
                )

    def test_generic_builder_takes_its_own_knobs_and_the_moves(self):
        # permute_every is the builder's; pad_out_of_prior goes on to ResidualAddOneRemoveOneMove
        kwargs = {"permute_every": 10, "pad_out_of_prior": True}
        cfg = self.config({"mbh": {**MBH_BLOCK, "pe_move": {"class": "MBHMoveBuilder", "kwargs": kwargs}}})
        self.assertEqual(cfg.blocks["mbh"].pe_move.kwargs, kwargs)

    def test_use_gpu_stays_a_knob_for_sources(self):
        # the reference never set it for mbh/emri (the move's default, False); only psd is filled from the backend
        block = {**MBH_BLOCK, "pe_move": {"class": "TDMBHSpecialMove", "kwargs": {"use_gpu": True}}}
        self.assertTrue(self.config({"mbh": block}).blocks["mbh"].pe_move.kwargs["use_gpu"])


class SearchWarningTest(_Recipe):
    def validate(self, blocks):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RunConfig.model_validate(source_raw(blocks), context={"run_dir": self.run_dir})
        return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]

    def test_source_without_search_warns(self):
        messages = self.validate({"mbh": MBH_BLOCK})
        self.assertTrue(any("'mbh' has no search: the branch starts empty" in m for m in messages), messages)

    def test_no_warning_with_a_search_disabled_or_for_psd(self):
        for blocks in ({"mbh": mbh_with_search()}, {"mbh": {**MBH_BLOCK, "enabled": False}}, {"psd": PSD_BLOCK}):
            with self.subTest(blocks=list(blocks)):
                self.assertFalse([m for m in self.validate(blocks) if "has no search" in m])


class RecipeSchemaTest(_Recipe):
    def write(self, text, name="my_recipe.py"):
        path = self.run_dir / name
        path.write_text(text)
        return path

    def test_default_is_the_stock_recipe(self):
        spec = self.config({}).recipe
        self.assertIsInstance(spec, StockRecipeSpec)
        self.assertEqual((spec.class_name, spec.kwargs), ("StockRecipe", {}))

    def test_stock_recipe_kwargs(self):
        spec = self.config({}, {"class": "StockRecipe", "kwargs": {"burn_in_iterations": 200}}).recipe
        self.assertEqual(spec.kwargs, {"burn_in_iterations": 200})

    def test_unknown_recipe_or_kwarg(self):
        self.assertRunError("value_error", "recipe.class", msg="unknown recipe 'Nope'", recipe={"class": "Nope"})
        self.assertRunError(
            "value_error", "recipe.class", msg="does not accept ['burn_in']",
            recipe={"class": "StockRecipe", "kwargs": {"burn_in": 10}},
        )

    def test_burn_in_must_be_a_positive_integer(self):
        for value in (0, -3, 2.5, "10", True):
            with self.subTest(value=value):
                self.assertRunError(
                    "value_error", "recipe.class", msg="burn_in_iterations must be a positive integer",
                    recipe={"class": "StockRecipe", "kwargs": {"burn_in_iterations": value}},
                )

    def test_recipe_file(self):
        path = self.write("def setup_recipe(recipe, engine_info, curr, acs, priors, state):\n    pass\n")
        spec = self.config({}, {"file": path.name}).recipe  # relative to the run folder
        self.assertIsInstance(spec, RecipeFile)
        self.assertEqual((spec.file, spec.function), (path, "setup_recipe"))

    def test_recipe_file_with_another_function_name_and_the_l2d_keyword(self):
        self.write("def my_steps(recipe, engine_info, curr, acs, priors, state, *, l2d):\n    pass\n")
        spec = self.config({}, {"file": "my_recipe.py", "function": "my_steps"}).recipe
        self.assertEqual(spec.function, "my_steps")

    def test_recipe_file_errors(self):
        cases = {
            "missing.py": (None, "does not exist"),
            "no_function.py": ("x = 1\n", "defines no function 'setup_recipe'"),
            "short.py": ("def setup_recipe(recipe, curr):\n    pass\n", "must accept (recipe, engine_info"),
            "broken.py": ("import not_a_module_anywhere\n", "cannot import recipe file"),
        }
        for name, (text, msg) in cases.items():
            with self.subTest(file=name):
                if text is not None:
                    self.write(text, name)
                self.assertRunError("value_error", "recipe.file", msg=msg, recipe={"file": name})

    def test_dump_reloads_recipe_and_pe_moves(self):
        self.write("def setup_recipe(recipe, engine_info, curr, acs, priors, state):\n    pass\n")
        blocks = {"psd": PSD_BLOCK, "mbh": {**mbh_with_search(), "pe_move": {"class": "MBHMoveBuilder"}}}
        cfg = self.config(blocks, {"file": "my_recipe.py"})
        out = dump_run(cfg, self.run_dir / "resolved" / "run.yaml")
        again = load_run(out)
        self.assertEqual(again.recipe, cfg.recipe)
        self.assertEqual(again.blocks["mbh"].pe_move, cfg.blocks["mbh"].pe_move)
        self.assertEqual(again.blocks["psd"].pe_move.class_name, "MultiGPUPSDMove")


# --- runtime stand-ins --- #

NWALKERS = 4


def fake_general(ntemps, force_backend="cpu"):
    return types.SimpleNamespace(
        nwalkers=NWALKERS, ntemps=ntemps, force_backend=force_backend, gpu_backend="cuda12x", gpus=[3, 1],
    )


def fake_info(betas, nleaves=2, ndim=3, search_interface=None):
    return types.SimpleNamespace(
        betas=np.asarray(betas, dtype=float), nleaves_max=nleaves, ndim=ndim, num_prop_repeats=7,
        waveform_kwargs={"mode_selection_threshold": 1e-4}, initialize_kwargs={"order": 30},
        transform=object(), inner_moves=[("stretch", 1.0)], search_interface=search_interface,
    )


class FakeWaveform:
    """Records its construction; ``get_signals_for_residuals`` is what the subtraction and builders use."""

    made = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeWaveform.made.append(self)

    def get_signals_for_residuals(self, *args, **kwargs):
        return []


class _Runtime(_Recipe):
    def setUp(self):
        super().setUp()
        FakeWaveform.made = []
        self.events = []  # what happened, in order

    def fake_interface(self, name):
        iface = mock.Mock()
        iface.populate_state.side_effect = lambda curr, state, betas, **kw: self.events.append(("populate", name))
        return iface

    def runtime(self, cfg, infos):
        """``(curr, state)`` stand-ins for the blocks of ``cfg`` with the given infos."""
        ntemps = cfg.sampler.ntemps
        curr = types.SimpleNamespace(general_info=fake_general(ntemps), source_info=infos)
        state = types.SimpleNamespace(
            branches_inds={n: np.ones((ntemps, NWALKERS, i.nleaves_max), bool) for n, i in infos.items()},
            sub_states={n: types.SimpleNamespace() for n in infos},
        )
        return curr, state

    def subtract_patch(self):
        def record(acs, state, wave_gen, name, info):
            self.events.append(("subtract", name))
            self.subtracted[name] = wave_gen

        self.subtracted = {}
        return mock.patch.object(gf_recipe, "subtract_initial_signal", record)


# --- seeding --- #

class LadderHelpersTest(unittest.TestCase):
    def test_pad_ladder(self):
        np.testing.assert_array_equal(pad_ladder([1.0, 0.5], 4), [1.0, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(pad_ladder([1.0, 0.5, 0.25], 3), [1.0, 0.5, 0.25])

    def test_psd_search_runs_unless_seeded_from_samples_or_a_point(self):
        block = lambda search: types.SimpleNamespace(search=search)
        self.assertTrue(psd_needs_search(block(None)))
        self.assertTrue(psd_needs_search(block(types.SimpleNamespace(search_type="prior"))))
        self.assertFalse(psd_needs_search(block(types.SimpleNamespace(search_type="samples"))))
        self.assertFalse(psd_needs_search(block(types.SimpleNamespace(search_type="point_estimate"))))


class SeedingTest(_Runtime):
    def seed(self, blocks, infos):
        cfg = self.config(blocks)
        curr, state = self.runtime(cfg, infos)
        priors = {"stand-in": "priors"}
        with self.subtract_patch():
            wave_gens = seed_branches(cfg, {"mbh": FakeWaveform, "emri": FakeWaveform}, curr, None, priors, state)
        return cfg, curr, state, priors, wave_gens

    def test_seeded_and_empty_sources(self):
        infos = {
            "mbh": fake_info([1.0, 0.8, 0.6], nleaves=3, search_interface=self.fake_interface("mbh")),
            "emri": fake_info([1.0], nleaves=2),
        }
        cfg, curr, state, priors, wave_gens = self.seed({"mbh": mbh_with_search(), "emri": EMRI_BLOCK}, infos)
        ntemps = cfg.sampler.ntemps

        # every source starts empty; populate_state (mocked here) would switch on the seeded leaves
        self.assertFalse(state.branches_inds["mbh"].any())
        self.assertFalse(state.branches_inds["emri"].any())

        # the interface gets the branch ladder padded to every temperature, and the priors
        (args, kwargs), = infos["mbh"].search_interface.populate_state.call_args_list
        self.assertIs(args[0], curr)
        self.assertIs(args[1], state)
        np.testing.assert_array_equal(args[2], [1.0, 0.8, 0.6] + [0.6] * (ntemps - 3))
        self.assertEqual(kwargs, {"priors": priors})

        # one generator per seeded branch, built from the Setup's initialize_kwargs, used for the subtraction
        self.assertEqual(list(wave_gens), ["mbh"])
        self.assertEqual([g.kwargs for g in FakeWaveform.made], [{"order": 30}])
        self.assertEqual(self.subtracted["mbh"], wave_gens["mbh"].get_signals_for_residuals)
        self.assertEqual(self.events, [("populate", "mbh"), ("subtract", "mbh")])

        # the state keeps every temperature: (nleaves, sampler.ntemps), padded with the last rung
        self.assertEqual(state.sub_states["mbh"].betas_all.shape, (3, ntemps))
        np.testing.assert_array_equal(state.sub_states["emri"].betas_all, np.ones((2, ntemps)))

    def test_all_branches_are_populated_before_any_subtraction(self):
        infos = {
            "mbh": fake_info([1.0], search_interface=self.fake_interface("mbh")),
            "emri": fake_info([1.0], search_interface=self.fake_interface("emri")),
        }
        self.seed({"mbh": mbh_with_search(), "emri": emri_with_search()}, infos)
        self.assertEqual(
            sorted(self.events[:2]), [("populate", "emri"), ("populate", "mbh")], self.events
        )
        self.assertEqual(sorted(self.events[2:]), [("subtract", "emri"), ("subtract", "mbh")])

    def test_psd_is_seeded_by_its_interface_without_subtraction(self):
        point = self.run_dir / "psd_point.npy"
        np.save(point, np.array([1.5e-11, 3e-15]))
        psd = {**PSD_BLOCK, "search": {"search_type": "point_estimate", "search_path": str(point), "spread": 1e-3}}
        ladder = make_ladder(2, ntemps=5, Tmax=1e6)
        infos = {"psd": fake_info(ladder, nleaves=1, search_interface=self.fake_interface("psd"))}
        _, _, state, _, wave_gens = self.seed({"psd": psd}, infos)
        (args, _), = infos["psd"].search_interface.populate_state.call_args_list
        np.testing.assert_array_equal(args[2], ladder)
        self.assertEqual(self.events, [("populate", "psd")])
        self.assertEqual(wave_gens, {})
        self.assertTrue(state.branches_inds["psd"].all())  # psd is never switched off

    def test_disabled_blocks_are_left_alone(self):
        infos = {"mbh": fake_info([1.0], search_interface=self.fake_interface("mbh"))}
        _, _, state, _, wave_gens = self.seed({"mbh": {**mbh_with_search(), "enabled": False}}, infos)
        self.assertTrue(state.branches_inds["mbh"].all())
        self.assertEqual((self.events, wave_gens), ([], {}))


# --- moves --- #

class MoveBuildTest(_Runtime):
    def moves(self, blocks, infos, wave_gens=None, ntemps_engine=None):
        cfg = self.config(blocks)
        curr, state = self.runtime(cfg, infos)
        engine_info = types.SimpleNamespace(ndims={n: i.ndim for n, i in infos.items()})
        self.acs, self.priors = object(), {"stand-in": "priors"}
        ctx = RecipeContext(cfg, wave_gens or {})
        recorders = {
            name: type(name, (Recorder,), {})
            for name in ("TDMBHSpecialMove", "EMRISpecialMove", "MultiGPUPSDMove")
        }
        with mock.patch.multiple(gf_moves, **recorders):
            moves = build_block_moves(ctx, engine_info, curr, self.acs, self.priors, state)
        return cfg, curr, state, moves

    def test_special_mbh_move(self):
        gen = FakeWaveform()
        info = fake_info([1.0, 0.5], nleaves=3, ndim=11)
        cfg, curr, _, moves = self.moves({"mbh": mbh_with_search()}, {"mbh": info}, {"mbh": gen})
        (move,) = moves["mbh"].pe
        self.assertEqual(moves["mbh"].search, [])
        kw = move.kwargs
        self.assertIs(kw.pop("dcga"), self.acs)
        self.assertIs(kw.pop("waveform_gen"), gen)
        self.assertIs(kw.pop("priors"), self.priors)
        self.assertIs(kw.pop("transform_fn"), info.transform)
        self.assertIs(kw.pop("inner_moves"), info.inner_moves)
        np.testing.assert_array_equal(kw.pop("betas_all"), np.tile([1.0, 0.5], (3, 1)))
        gen_kwargs = kw.pop("waveform_gen_kwargs")
        self.assertEqual(gen_kwargs, info.waveform_kwargs)
        self.assertIsNot(gen_kwargs, info.waveform_kwargs)  # the moves may add keys (e.g. synchronize)
        self.assertEqual(kw, dict(
            branch_name="mbh",
            coords_shape=(2, NWALKERS, 3, 11),  # the move runs on its own (subset) ladder
            waveform_like_kwargs={},
            num_repeats=7,
            permute_every=100, pad_out_of_prior=True, run_async=True, run_threaded=True, randomize_split=True,
        ))
        self.assertEqual(move.accepted.shape, (cfg.sampler.ntemps, NWALKERS))

    def test_special_emri_move(self):
        gen = FakeWaveform()
        info = fake_info([1.0], nleaves=2)
        _, _, _, moves = self.moves({"emri": emri_with_search()}, {"emri": info}, {"emri": gen})
        kw = moves["emri"].pe[0].kwargs
        self.assertEqual(kw["waveform_gen_method"], "get_signals_for_residuals")
        self.assertEqual(kw["waveform_like_method"], "__call__")
        self.assertEqual(kw["waveform_like_kwargs"], info.waveform_kwargs)
        self.assertIsNot(kw["waveform_like_kwargs"], info.waveform_kwargs)
        self.assertEqual((kw["permute_every"], kw["batch_size_per_gpu"]), (25, 5))
        self.assertEqual(kw["coords_shape"], (1, NWALKERS, 2, 3))

    def test_sources_without_search_get_no_moves(self):
        _, _, _, moves = self.moves({"mbh": MBH_BLOCK}, {"mbh": fake_info([1.0])})
        self.assertEqual(moves, {})

    def test_yaml_knobs_replace_the_defaults(self):
        block = {**mbh_with_search(), "pe_move": {"class": "TDMBHSpecialMove", "kwargs": {"permute_every": 7}}}
        _, _, _, moves = self.moves({"mbh": block}, {"mbh": fake_info([1.0])}, {"mbh": FakeWaveform()})
        kw = moves["mbh"].pe[0].kwargs
        self.assertEqual(kw["permute_every"], 7)
        self.assertNotIn("run_async", kw)

    def test_special_psd_moves(self):
        ladder = make_ladder(2, ntemps=5, Tmax=1e6)
        info = fake_info(ladder, nleaves=1, ndim=2)
        cfg, _, _, moves = self.moves({"psd": PSD_BLOCK}, {"psd": info})
        (search,), (pe,) = moves["psd"].search, moves["psd"].pe
        self.assertEqual(search.args, (self.acs, self.priors))
        self.assertEqual((search.kwargs["max_logl_mode"], search.kwargs["name"]), (True, "psd search move"))
        self.assertEqual((pe.kwargs["max_logl_mode"], pe.kwargs["name"]), (False, "psd pe move"))
        tc = pe.kwargs["temperature_control"]
        self.assertIs(tc, search.kwargs["temperature_control"])  # one ladder for both, as the reference
        self.assertIsInstance(tc, TemperatureControl)
        np.testing.assert_array_equal(tc.betas, ladder)
        self.assertFalse(tc.permute)
        for key in ("max_logl_mode", "name", "temperature_control"):
            pe.kwargs.pop(key)
        self.assertIs(pe.kwargs.pop("psd_transform_fn"), info.transform)
        self.assertEqual(pe.kwargs, dict(
            num_repeats=7, use_gpu=False,  # CPU backend
            permute_every=100, live_dangerously=True, run_async=True, run_threaded=True,
        ))
        self.assertEqual(pe.accepted.shape, (cfg.sampler.ntemps, NWALKERS))

    def test_psd_search_is_dropped_when_seeded_from_a_point(self):
        point = self.run_dir / "psd_point.npy"
        np.save(point, np.array([1.5e-11, 3e-15]))
        psd = {**PSD_BLOCK, "search": {"search_type": "point_estimate", "search_path": str(point), "spread": 1e-3}}
        _, _, _, moves = self.moves({"psd": psd}, {"psd": fake_info(np.ones(5), nleaves=1, ndim=2)})
        self.assertEqual((moves["psd"].search, len(moves["psd"].pe)), ([], 1))

    def test_generic_source_builder(self):
        built = []

        class FakeBuilder:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                built.append(self)

            def build(self, engine_info, curr, acs, priors, state):
                self.build_args = (engine_info, curr, acs, priors, state)
                return [], ["generic pe move"]

        gen = FakeWaveform()
        info = fake_info([1.0, 0.5])
        block = {**mbh_with_search(), "pe_move": {"class": "MBHMoveBuilder", "kwargs": {"permute_every": 9}}}
        with mock.patch.object(gf_recipe, "MBHMoveBuilder", FakeBuilder):
            _, curr, state, moves = self.moves({"mbh": block}, {"mbh": info}, {"mbh": gen})
        self.assertEqual(moves["mbh"], BranchMoves([], ["generic pe move"]))
        (builder,) = built
        kw = builder.kwargs
        self.assertEqual(kw.pop("wave_gen"), gen.get_signals_for_residuals)
        np.testing.assert_array_equal(kw.pop("betas"), [1.0, 0.5])
        self.assertEqual(kw, {"move_name": "mbh pe move", "permute_every": 9})
        self.assertIs(builder.build_args[1], curr)
        self.assertIs(builder.build_args[4], state)

    def test_generic_psd_builder(self):
        made = []

        class FakePSDBuilder:  # the config is validated against it too (patched before the load)
            def __init__(self, *, num_repeats=None, permute_every=None, betas=None):
                self.kwargs = dict(num_repeats=num_repeats, permute_every=permute_every, betas=betas)
                made.append(self)

            def build(self, *args):
                return ["search"], ["pe"]

        ladder = make_ladder(2, ntemps=5, Tmax=1e6)
        block = {**PSD_BLOCK, "pe_move": {"class": "PSDMoveBuilder", "kwargs": {"permute_every": 30}}}
        with mock.patch.object(gf_recipe, "PSDMoveBuilder", FakePSDBuilder):
            _, _, _, moves = self.moves({"psd": block}, {"psd": fake_info(ladder, nleaves=1, ndim=2)})
        self.assertEqual(moves["psd"], BranchMoves(["search"], ["pe"]))
        kw = made[0].kwargs
        np.testing.assert_array_equal(kw.pop("betas"), ladder)
        self.assertEqual(kw, {"num_repeats": 7, "permute_every": 30})


# --- steps and the setup function --- #

class FakeCombine:
    def __init__(self, moves, share_temperature_control):
        self.moves, self.share_temperature_control = moves, share_temperature_control


class StockRecipeTest(unittest.TestCase):
    def steps(self, moves, **kwargs):
        recipe = gf_recipe.Recipe()
        with mock.patch.object(gf_moves, "GFCombineMove", FakeCombine):
            StockRecipe(**kwargs).add_steps(recipe, moves)
        return recipe.recipe

    def test_search_then_pe(self):
        moves = {"mbh": BranchMoves([], ["mbh pe"]), "psd": BranchMoves(["psd search"], ["psd pe"])}
        search, pe = self.steps(moves)
        self.assertEqual((search["name"], pe["name"]), ("psd search", "pe"))
        self.assertIsInstance(search["adjust"], gf_recipe.SearchRecipeStep)
        self.assertEqual(search["adjust"].moves, ["psd search"])
        self.assertIsInstance(pe["adjust"], gf_recipe.PERecipeStep)
        (combined,) = pe["adjust"].moves
        self.assertEqual(combined.moves, ["mbh pe", "psd pe"])
        self.assertFalse(combined.share_temperature_control)

    def test_burn_in(self):
        moves = {"psd": BranchMoves(["psd search"], ["psd pe"])}
        search, burn_in, pe = self.steps(moves, burn_in_iterations=50)
        self.assertEqual([s["name"] for s in (search, burn_in, pe)], ["psd search", "burn-in", "pe"])
        self.assertIsInstance(burn_in["adjust"], gf_recipe.IterationCountRecipeStep)
        self.assertEqual(burn_in["adjust"].num_iters, 50)
        self.assertIs(burn_in["adjust"].moves[0], pe["adjust"].moves[0])  # the same moves carry on

    def test_no_search_step_without_search_moves(self):
        (pe,) = self.steps({"psd": BranchMoves([], ["psd pe"])})
        self.assertEqual(pe["name"], "pe")

    def test_nothing_to_sample(self):
        with self.assertRaisesRegex(ValueError, "nothing to sample"):
            self.steps({"psd": BranchMoves(["psd search"], [])})
        with self.assertRaisesRegex(ValueError, "nothing to sample"):
            self.steps({})


class SetupFunctionTest(_Runtime):
    def run_setup(self, cfg, general=None):
        builds = {"mbh": types.SimpleNamespace(waveform_class=FakeWaveform)}
        setup = make_setup_function(cfg, builds)
        curr = types.SimpleNamespace(general_info=general or fake_general(cfg.sampler.ntemps))
        args = ("recipe", "engine_info", curr, "acs", "priors", "state")

        def seed(cfg_, classes, *rest):
            self.events.append(("seed", classes))
            return {"mbh": "generator"}

        def block_moves(ctx, *rest):
            self.events.append(("moves", ctx.wave_gens))
            return {"psd": BranchMoves([], ["pe"])}

        def add_steps(recipe_self, recipe, moves):
            self.events.append(("steps", recipe, moves))

        with mock.patch.object(build_recipe, "seed_branches", seed), \
                mock.patch.object(build_recipe, "build_block_moves", block_moves), \
                mock.patch.object(StockRecipe, "add_steps", add_steps):
            result = setup(*args)
        return args, result

    def test_stock_recipe_seeds_then_builds_moves_then_adds_steps(self):
        self.run_setup(self.config({}, {"class": "StockRecipe", "kwargs": {"burn_in_iterations": 5}}))
        self.assertEqual([e[0] for e in self.events], ["seed", "moves", "steps"])
        self.assertEqual(self.events[0][1], {"mbh": FakeWaveform})
        self.assertEqual(self.events[1][1], {"mbh": "generator"})
        self.assertEqual(self.events[2][1:], ("recipe", {"psd": BranchMoves([], ["pe"])}))

    def test_recipe_file_runs_after_seeding_with_the_engine_arguments(self):
        (self.run_dir / "plain.py").write_text(
            "calls = []\n"
            "def setup_recipe(recipe, engine_info, curr, acs, priors, state):\n"
            "    calls.append((recipe, engine_info, curr, acs, priors, state))\n"
            "    return 'done'\n"
        )
        cfg = self.config({}, {"file": "plain.py"})
        loaded = []
        real_load = build_recipe.load_recipe_function

        def load(path, name):
            fn = real_load(path, name)
            loaded.append(fn)
            return fn

        with mock.patch.object(build_recipe, "load_recipe_function", load):
            args, result = self.run_setup(cfg)
        self.assertEqual(result, "done")
        self.assertEqual([e[0] for e in self.events], ["seed"])  # no library moves, no stock steps
        self.assertEqual(loaded[0].__globals__["calls"], [args])

    def test_recipe_file_with_l2d_gets_the_context(self):
        (self.run_dir / "ctx.py").write_text(
            "seen = []\n"
            "def setup_recipe(recipe, engine_info, curr, acs, priors, state, l2d=None):\n"
            "    seen.append(l2d)\n"
        )
        cfg = self.config({}, {"file": "ctx.py"})
        loaded = []
        real_load = build_recipe.load_recipe_function
        with mock.patch.object(
            build_recipe, "load_recipe_function", lambda p, n: loaded.append(real_load(p, n)) or loaded[-1]
        ):
            self.run_setup(cfg)
        (ctx,) = loaded[0].__globals__["seen"]
        self.assertIsInstance(ctx, RecipeContext)
        self.assertIs(ctx.cfg, cfg)
        self.assertEqual(ctx.wave_gens, {"mbh": "generator"})

    def test_device_is_selected_on_the_gpu_backend_only(self):
        cfg = self.config({})
        fake_cupy = mock.MagicMock()
        with mock.patch.dict(_sys.modules, {"cupy": fake_cupy}):
            self.run_setup(cfg, fake_general(5, force_backend="cpu"))
            fake_cupy.cuda.runtime.setDevice.assert_not_called()
            self.run_setup(cfg, fake_general(5, force_backend="cuda12x"))
            fake_cupy.cuda.runtime.setDevice.assert_called_once_with(3)


# --- the two library changes in recipe.py --- #

class LibraryLadderTest(unittest.TestCase):
    def test_generic_source_builder_runs_on_a_subset_ladder(self):
        ntemps = 5
        info = fake_info([1.0, 0.5], nleaves=3, ndim=4)
        curr = types.SimpleNamespace(general_info=fake_general(ntemps), source_info={"mbh": info})
        state = types.SimpleNamespace(sub_states={"mbh": types.SimpleNamespace()})
        builder = gf_recipe.MBHMoveBuilder(wave_gen=lambda *a, **k: None, betas=np.array([1.0, 0.5]))
        with mock.patch.object(gf_recipe, "ResidualAddOneRemoveOneMove", Recorder):
            _, (move,) = builder.build(None, curr, object(), {}, state)
        self.assertEqual(move.args[1], (2, NWALKERS, 3, 4))  # coords_shape from the ladder
        self.assertEqual(move.kwargs["betas_all"].shape, (3, 2))  # what the move asserts against
        np.testing.assert_array_equal(state.sub_states["mbh"].betas_all, np.tile([1.0, 0.5, 0.5, 0.5, 0.5], (3, 1)))

    def test_generic_source_builder_full_ladder_is_unchanged(self):
        info = fake_info(np.ones(5), nleaves=1, ndim=4)
        curr = types.SimpleNamespace(general_info=fake_general(5), source_info={"mbh": info})
        state = types.SimpleNamespace(sub_states={"mbh": types.SimpleNamespace()})
        builder = gf_recipe.MBHMoveBuilder(wave_gen=lambda *a, **k: None)
        with mock.patch.object(gf_recipe, "ResidualAddOneRemoveOneMove", Recorder):
            _, (move,) = builder.build(None, curr, object(), {}, state)
        self.assertEqual(move.args[1], (5, NWALKERS, 1, 4))
        np.testing.assert_array_equal(state.sub_states["mbh"].betas_all, np.tile(make_ladder(4, ntemps=5), (1, 1)))

    def psd_temperature_control(self, **kwargs):
        info = types.SimpleNamespace(transform=None)
        general = types.SimpleNamespace(nwalkers=NWALKERS, ntemps=5, sensitivity_backend=None)
        curr = types.SimpleNamespace(general_info=general, source_info={"psd": info})
        engine_info = types.SimpleNamespace(ndims={"psd": 2})
        with mock.patch.object(gf_recipe, "PSDMove", Recorder):
            search, pe = gf_recipe.build_psd_moves(engine_info, curr, object(), {}, **kwargs)
        self.assertIs(search.kwargs["temperature_control"], pe.kwargs["temperature_control"])
        return pe.kwargs["temperature_control"]

    def test_build_psd_moves_takes_a_ladder(self):
        ladder = np.array([1.0, 0.4, 0.2, 0.1, 0.05])
        np.testing.assert_array_equal(self.psd_temperature_control(betas=ladder).betas, ladder)

    def test_build_psd_moves_default_is_unchanged(self):
        np.testing.assert_array_equal(
            self.psd_temperature_control().betas, make_ladder(2, ntemps=5, Tmax=1e6)
        )

    def test_psd_builder_passes_its_ladder_on(self):
        ladder = np.array([1.0, 0.4, 0.2, 0.1, 0.05])
        with mock.patch.object(gf_recipe, "build_psd_moves", return_value=("s", "p")) as build:
            gf_recipe.PSDMoveBuilder(betas=ladder).build("ei", "curr", "acs", "priors", "state")
        np.testing.assert_array_equal(build.call_args.kwargs["betas"], ladder)


if __name__ == "__main__":
    unittest.main()
