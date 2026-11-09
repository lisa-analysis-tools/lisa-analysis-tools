"""Recipe stages: per-stage PE-move settings (proposal repeats, inner-move weights) and restart-aware step counts."""

import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import types
import unittest
import warnings

import numpy as np

from lisatools.globalfit import recipe as gf_recipe
from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
from lisatools.globalfit.stock.l2d.build_recipe import BranchMoves, StockRecipe, stage_move_settings
from lisatools.globalfit.stock.l2d.io import RunConfig

from tests.test_l2d_sources import MBH_BLOCK, _Sources, source_raw

SEARCH = {"search_type": "point_estimate", "search_path": ".", "spread": 1e-3}
MBH = {**MBH_BLOCK, "search": SEARCH}  # three inner moves: 0.35, 0.05, 0.05


def fake_move(nmoves=3, has_flow=(False, False, False), flow_burnin=0, branch="mbh"):
    """A ResidualAddOneRemoveOneMove with only what ``apply_settings`` reads (its __init__ needs the engine)."""
    move = object.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name, move.moves = branch, [object()] * nmoves
    move._has_flow, move.flow_burnin, move._iterations = np.array(has_flow), flow_burnin, 0
    move.num_repeats, move.move_weights = 50, np.full(nmoves, 1.0 / nmoves)
    return move


class ApplySettingsTest(unittest.TestCase):
    def test_sets_repeats_and_normalised_weights(self):
        move = fake_move()
        move.apply_settings(num_repeats=15, weights=[2.0, 1.0, 1.0])
        self.assertEqual(move.num_repeats, 15)
        np.testing.assert_allclose(move.move_weights, [0.5, 0.25, 0.25])

    def test_none_keeps_the_current_values(self):
        move = fake_move()
        move.apply_settings()
        self.assertEqual(move.num_repeats, 50)
        np.testing.assert_allclose(move.move_weights, [1 / 3] * 3)

    def test_zero_switches_a_move_off(self):
        move = fake_move()
        move.apply_settings(weights=[0.0, 1.0, 0.0])
        np.testing.assert_allclose(move.move_weights, [0.0, 1.0, 0.0])

    def test_bad_values(self):
        for kwargs, msg in [
            ({"num_repeats": 0}, "positive integer"), ({"num_repeats": 2.5}, "positive integer"),
            ({"num_repeats": True}, "positive integer"), ({"weights": [1.0, 1.0]}, "3 inner moves"),
            ({"weights": [0.0, 0.0, 0.0]}, "not all 0"), ({"weights": [1.0, -1.0, 1.0]}, ">= 0"),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, msg):
                fake_move().apply_settings(**kwargs)

    def test_flow_burnin_needs_a_move_without_a_flow(self):
        move = fake_move(nmoves=2, has_flow=(False, True), flow_burnin=20)
        with self.assertRaisesRegex(ValueError, "during flow_burnin"):
            move.apply_settings(weights=[0.0, 1.0])
        move._iterations = 20  # burn-in over: the flow may be the only move
        move.apply_settings(weights=[0.0, 1.0])
        np.testing.assert_allclose(move._draw_weights(), [0.0, 1.0])


class FakeSampler:
    def __init__(self, iteration=0):
        self.iteration, self.periodic, self.moves, self.weights = iteration, None, None, None


class FakeBackend:
    """Records step starts like ``hdfbackend.HDFBackend.recipe_step_start``."""

    def __init__(self, starts=None):
        self.starts, self.completed = dict(starts or {}), []

    def recipe_step_start(self, name, iteration):
        return self.starts.setdefault(name, iteration)

    def completed_recipe_step(self, name):
        self.completed.append(name)


def staged_recipe(mbh_move, stages, settings):
    """The stock recipe's steps around a real GFCombineMove (the recipe step walks its sub-moves)."""
    recipe = gf_recipe.Recipe()
    StockRecipe(stages=stages).add_steps(recipe, {"mbh": BranchMoves([], [mbh_move])}, settings.get)
    return recipe


class RecipeStepTest(unittest.TestCase):
    STAGES = [{"name": "warm", "iterations": 3}, {"name": "pe"}]
    SETTINGS = {
        "warm": {"mbh": {"num_repeats": 50, "weights": [1.0, 0.0, 0.0]}},
        "pe": {"mbh": {"num_repeats": 10, "weights": [1.0, 1.0, 2.0]}},
    }

    def run_recipe(self, recipe, sampler, until):
        """Call the recipe after each stored iteration, as eryn's stopping_fn, up to iteration ``until``."""
        local = 0
        while sampler.iteration < until:
            sampler.iteration += 1
            if recipe(local, None, sampler):
                return
            local += 1

    def test_each_stage_applies_its_settings(self):
        move, sampler = fake_move(), FakeSampler()
        recipe = staged_recipe(move, self.STAGES, self.SETTINGS)
        recipe.backend = FakeBackend()
        recipe.setup_first_recipe_step(0, None, sampler)
        self.assertEqual(move.num_repeats, 50)
        np.testing.assert_allclose(move.move_weights, [1.0, 0.0, 0.0])

        self.run_recipe(recipe, sampler, until=2)
        self.assertEqual(recipe.current_recipe_step["name"], "warm")
        self.run_recipe(recipe, sampler, until=3)  # exactly `iterations` stored iterations
        self.assertEqual(recipe.current_recipe_step["name"], "pe")
        self.assertEqual(recipe.backend.completed, ["warm"])
        self.assertEqual(move.num_repeats, 10)
        np.testing.assert_allclose(move.move_weights, [0.25, 0.25, 0.5])
        self.assertEqual(recipe.backend.starts, {"warm": 0, "pe": 3})

    def test_restart_inside_a_stage_keeps_its_start(self):
        # stopped at iteration 2 of the 3-iteration stage: eryn's iteration restarts at 0, the stage does not
        move, sampler = fake_move(), FakeSampler(iteration=2)
        recipe = staged_recipe(move, self.STAGES, self.SETTINGS)
        recipe.backend = FakeBackend({"warm": 0})
        recipe.setup_first_recipe_step(sampler.iteration, None, sampler)
        self.run_recipe(recipe, sampler, until=3)
        self.assertEqual(recipe.current_recipe_step["name"], "pe")

    def test_restart_into_a_later_stage_applies_its_settings(self):
        move, sampler = fake_move(), FakeSampler(iteration=7)
        recipe = staged_recipe(move, self.STAGES, self.SETTINGS)
        recipe.recipe[0]["status"] = True  # recorded as completed by the backend (add_recipe)
        recipe.backend = FakeBackend({"warm": 0, "pe": 3})
        recipe.setup_first_recipe_step(sampler.iteration, None, sampler)
        self.assertEqual(move.num_repeats, 10)
        np.testing.assert_allclose(move.move_weights, [0.25, 0.25, 0.5])

    def test_settings_for_a_branch_without_moves(self):
        recipe = staged_recipe(fake_move(), [{"name": "pe"}], {"pe": {"emri": {"num_repeats": 3}}})
        recipe.backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, "emri"):
            recipe.setup_first_recipe_step(0, None, FakeSampler())


class BackendStepStartTest(unittest.TestCase):
    def test_start_is_recorded_once_and_survives_reopening(self):
        import os
        import tempfile

        from lisatools.globalfit.hdfbackend import GFHDFBackend

        def recipe():
            out = gf_recipe.Recipe()
            out.add_recipe_component(gf_recipe.IterationCountRecipeStep(moves=[], num_iters=3), name="warm")
            out.add_recipe_component(gf_recipe.PERecipeStep(moves=[]), name="pe")
            return out

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "backend.h5")
        backend = GFHDFBackend(path)
        backend.reset(4, [3], ntemps=1, nleaves_max=[1], branch_names=["mbh"])
        backend.add_recipe(recipe())
        self.assertEqual(backend.recipe_step_start("warm", 0), 0)
        self.assertEqual(backend.recipe_step_start("warm", 7), 0)  # recorded on the first start only
        backend.completed_recipe_step("warm")

        again, restored = GFHDFBackend(path), recipe()
        again.add_recipe(restored)  # a restart reads the statuses back
        self.assertEqual([bool(s["status"]) for s in restored.recipe], [True, False])
        self.assertEqual(again.recipe_step_start("warm", 99), 0)


class StockRecipeStagesTest(unittest.TestCase):
    def names(self, **kwargs):
        return StockRecipe(**kwargs).stages

    def test_defaults_and_burn_in_shorthand(self):
        self.assertEqual(self.names(), [{"name": "pe", "iterations": None}])
        self.assertEqual(
            self.names(burn_in_iterations=5),
            [{"name": "burn-in", "iterations": 5}, {"name": "pe", "iterations": None}],
        )

    def test_steps(self):
        recipe = staged_recipe(fake_move(), [{"name": "a", "iterations": 2}, {"name": "b", "iterations": 4},
                                             {"name": "c"}], {})
        kinds = [(s["name"], type(s["adjust"]).__name__) for s in recipe.recipe]
        self.assertEqual(
            kinds, [("a", "IterationCountRecipeStep"), ("b", "IterationCountRecipeStep"), ("c", "PERecipeStep")]
        )
        self.assertEqual(len({id(s["adjust"].moves[0]) for s in recipe.recipe}), 1)  # one shared move

    def test_bad_stages(self):
        for kwargs, msg in [
            ({"stages": []}, "non-empty list"),
            ({"stages": [{"name": "pe", "iterations": 3}]}, "leave out its iterations"),
            ({"stages": [{"name": "a"}, {"name": "pe"}]}, "needs iterations"),
            ({"stages": [{"name": "a", "iterations": 0}, {"name": "pe"}]}, "needs iterations"),
            ({"stages": [{"name": "a", "iterations": 2}, {"name": "a"}]}, "unique"),
            ({"stages": [{"name": "psd search"}]}, "other than"),
            ({"stages": [{"name": "pe", "num": 2}]}, "must be"),
            ({"stages": [{"name": "pe"}], "burn_in_iterations": 2}, "not both"),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, msg):
                StockRecipe(**kwargs)


class BlockStagesTest(_Sources):
    def config(self, blocks, recipe=None):
        raw = source_raw(blocks)
        if recipe is not None:
            raw["recipe"] = recipe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return RunConfig.model_validate(raw, context={"run_dir": self.run_dir})

    STAGED = {"class": "StockRecipe", "kwargs": {"stages": [{"name": "warm", "iterations": 5}, {"name": "pe"}]}}

    def test_schema(self):
        block = {**MBH, "stages": {"pe": {"num_prop_repeats": 10, "inner_move_weights": [1, 0, 0]}}}
        stage = self.config({"mbh": block}, self.STAGED).blocks["mbh"].stages["pe"]
        self.assertEqual((stage.num_prop_repeats, stage.inner_move_weights, stage.flow_weight), (10, [1, 0, 0], None))

    def test_errors(self):
        for stages, recipe, msg in [
            ({"later": {}}, self.STAGED, "not recipe stages"),
            ({"pe": {"inner_move_weights": [1, 1]}}, None, "2 weights for 3 inner moves"),
            ({"pe": {"flow_weight": 0.5}}, None, "has no flow"),
            ({"pe": {"num_prop_repeats": 0}}, None, "greater than 0"),
        ]:
            with self.subTest(stages=stages), self.assertRaisesRegex(Exception, msg):
                self.config({"mbh": {**MBH, "stages": stages}}, recipe)
        no_inner = {k: v for k, v in MBH.items() if k != "inner_moves"}
        with self.assertRaisesRegex(Exception, "give inner_moves"):
            self.config({"mbh": {**no_inner, "stages": {"pe": {"inner_move_weights": [1.0]}}}})

    def test_resolved_settings(self):
        block = {**MBH, "stages": {"pe": {"num_prop_repeats": 10, "inner_move_weights": [0.1, 0.2, 0.3]}}}
        cfg = self.config({"mbh": block}, self.STAGED)
        flow_move = object()
        info = types.SimpleNamespace(inner_moves=[("a", 0.35), ("b", 0.05), ("c", 0.05), (flow_move, 0.2)])
        curr = types.SimpleNamespace(source_info={"mbh": info})
        # a stage the block leaves out runs on the block's own values (deterministic on a restart)
        self.assertEqual(stage_move_settings(cfg, curr, "warm"),
                         {"mbh": {"num_repeats": 50, "weights": [0.35, 0.05, 0.05, 0.2]}})
        self.assertEqual(stage_move_settings(cfg, curr, "pe"),
                         {"mbh": {"num_repeats": 10, "weights": [0.1, 0.2, 0.3, 0.2]}})
        # without stages a block is left alone
        self.assertEqual(stage_move_settings(self.config({"mbh": MBH}, self.STAGED), curr, "pe"), {})


if __name__ == "__main__":
    unittest.main()
