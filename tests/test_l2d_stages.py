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
from lisatools.globalfit.stock.l2d.build_recipe import (
    Alternative, BranchMoves, StockRecipe, stage_move_settings,
)
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

    def test_turns_and_settings_together(self):
        # a gb move taking turns in pe only: warm keeps one combined move, pe two, both carrying the mbh move
        from lisatools.globalfit.moves import GFCombineMove

        move, sampler = fake_move(), FakeSampler()
        gb_prior, gb_fstat = types.SimpleNamespace(name="gb prior"), types.SimpleNamespace(name="gb fstat")
        recipe = gf_recipe.Recipe()
        moves = {"gb": BranchMoves([], [gb_prior], (Alternative(gb_fstat, 0.25, {"warm": 0.0}),)),
                 "mbh": BranchMoves([], [move])}
        StockRecipe(stages=self.STAGES).add_steps(recipe, moves, self.SETTINGS.get)
        recipe.backend = FakeBackend()
        recipe.setup_first_recipe_step(0, None, sampler)
        (warm,) = sampler.moves
        self.assertIsInstance(warm, GFCombineMove)
        self.assertEqual(warm.sub_moves, [gb_prior, move])
        self.assertEqual(move.num_repeats, 50)

        self.run_recipe(recipe, sampler, until=3)
        self.assertEqual(recipe.current_recipe_step["name"], "pe")
        self.assertIs(sampler.moves[0], warm)
        self.assertEqual(sampler.moves[1].sub_moves, [gb_fstat, move])
        np.testing.assert_allclose(sampler.weights, [0.75, 0.25])
        self.assertEqual(move.num_repeats, 10)  # the settings reach the move inside both combined moves
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


# --- runtime autocorrelation: the convergence stop and the adaptive proposal repeats --- #

def ar1(nsteps, nwalkers, tau, seed=0, ncols=1):
    """A stationary AR(1) chain ``[nsteps, nwalkers, ncols]`` with integrated ACT ``tau`` (unit variance)."""
    rng = np.random.default_rng(seed)
    phi = (tau - 1.0) / (tau + 1.0)
    x = np.empty((nsteps, nwalkers, ncols))
    x[0] = rng.normal(size=(nwalkers, ncols))
    for i in range(1, nsteps):
        x[i] = phi * x[i - 1] + np.sqrt(1.0 - phi**2) * rng.normal(size=(nwalkers, ncols))
    return x


class ActTest(unittest.TestCase):
    def test_stored_chain(self):
        from eryn.utils import get_integrated_act

        from lisatools.sampling.utility import cold_chain_act

        x = ar1(20000, 16, 20.0)
        tau, reliable = cold_chain_act(x)
        self.assertAlmostEqual(tau[0], 20.0, delta=2.0)
        self.assertTrue(reliable[0])
        # eryn's own estimate, where its fixed 50 lags are enough
        self.assertAlmostEqual(tau[0], get_integrated_act(x[:, None])[0, 0], delta=2.0)
        # ... and where they are not: it cannot say more than about 100
        slow = ar1(60000, 8, 300.0, seed=1)
        self.assertLess(get_integrated_act(slow[:, None])[0, 0], 100.0)
        self.assertAlmostEqual(cold_chain_act(slow)[0][0], 300.0, delta=60.0)

    def test_short_stuck_and_fixed_chains_are_not_reliable(self):
        from lisatools.sampling.utility import cold_chain_act

        tau, reliable = cold_chain_act(ar1(200, 16, 20.0))  # 10 ACTs: an estimate, not a reliable one
        self.assertFalse(reliable[0])
        stuck = np.broadcast_to(np.arange(8.0)[None, :, None], (500, 8, 1))  # each walker at its own value
        tau, reliable = cold_chain_act(stuck)
        self.assertTrue(np.isnan(tau[0]))
        self.assertFalse(reliable[0])
        one_stuck = ar1(5000, 8, 5.0)
        one_stuck[:, 3] = 1.0
        self.assertFalse(cold_chain_act(one_stuck)[1][0])

    def test_periodic_column(self):
        from lisatools.sampling.utility import cold_chain_act

        angle = np.mod(0.3 * ar1(20000, 16, 10.0), 2.0 * np.pi)  # sits on the wrap
        tau, reliable = cold_chain_act(angle, periods={0: 2.0 * np.pi})
        self.assertAlmostEqual(tau[0], 10.0, delta=2.0)
        self.assertLess(cold_chain_act(angle)[0][0], 0.8 * tau[0])  # as it is, the jumps at the wrap read as mixing

    def test_inner_act(self):
        from eryn.utils import get_acf

        from lisatools.sampling.utility import _windowed_act, variogram, variogram_act

        tau, rng = 20.0, np.random.default_rng(5)

        def chain(steps, offset=0.0, seed=0):
            # each walker around its own mean: it carries its own state of the other blocks
            return ar1(steps, 14, tau, seed=seed) + offset * rng.normal(size=(1, 14, 1))

        for offset in (0.0, 5.0):
            for ratio, expected in [(50, tau), (20, tau), (3, np.nan), (1, np.nan)]:  # steps per ACT
                estimates = [variogram_act(variogram(chain(int(ratio * tau), offset, seed))) for seed in range(8)]
                with self.subTest(offset=offset, ratio=ratio):
                    if np.isnan(expected):
                        self.assertGreaterEqual(np.isnan(estimates).mean(), 0.75, msg=f"read as {estimates}")
                    else:
                        self.assertLessEqual(np.isnan(estimates).mean(), 0.25)
                        self.assertAlmostEqual(np.nanmean(estimates), expected, delta=0.2 * tau)
        # each walker's own mean (eryn's get_acf) falls short on a series of 10 ACTs
        own_mean = np.mean([_windowed_act(get_acf(chain(200, seed=s)).mean(axis=1))[0] for s in range(8)])
        self.assertLess(own_mean, 0.7 * tau)

    def test_inner_act_edges(self):
        from lisatools.sampling.utility import variogram, variogram_act

        direct = ar1(300, 5, 10.0, ncols=3).cumsum(axis=0)  # the FFT form against the definition
        expected = [0.5 * ((direct[k:] - direct[: 300 - k]) ** 2).mean(axis=(0, 1)) for k in range(300)]
        np.testing.assert_allclose(variogram(direct), expected, rtol=1e-8, atol=1e-8)

        self.assertTrue(np.isnan(variogram_act(variogram(ar1(5, 32, 3.0)))))  # too few steps to say anything
        self.assertAlmostEqual(variogram_act(variogram(ar1(400, 256, 1.0))), 1.0, delta=0.4)  # independent draws
        drift = ar1(2000, 14, 3.0).cumsum(axis=0)  # a chain that diffuses (burn-in): the variogram never flattens
        self.assertTrue(np.isnan(variogram_act(variogram(drift))))
        fixed = ar1(2000, 32, 5.0, ncols=2)
        fixed[:, :, 1] = 3.0  # a parameter held fixed does not hide the other
        self.assertAlmostEqual(variogram_act(variogram(fixed)), 5.0, delta=1.5)
        stuck = np.broadcast_to(np.arange(32.0)[None, :, None], (200, 32, 1))  # nothing accepted
        self.assertTrue(np.isnan(variogram_act(variogram(stuck))))


class MoveInnerActTest(unittest.TestCase):
    """The move's running inner ACT, fed with leaf visits as its repeat loop feeds it."""

    def build(self, periodic=None):
        move = fake_move()  # the accumulator reads only the branch name and the periodic container
        move._periodic = periodic
        return move

    def test_off_by_default_and_empty_after_a_reset(self):
        move = self.build()
        self.assertFalse(move.record_inner_chain)
        tau, ncalls = move.inner_act()
        self.assertTrue(np.isnan(tau))
        self.assertEqual(ncalls, 0)

    def test_act_is_the_slowest_parameter_of_the_slowest_leaf(self):
        move = self.build()
        slow = np.concatenate([ar1(4000, 14, 5.0, 1), ar1(4000, 14, 20.0, 2)], axis=-1)
        fast = np.concatenate([ar1(4000, 14, 5.0, 3), ar1(4000, 14, 8.0, 4)], axis=-1)
        for call in range(20):  # twenty calls of 200 repeats
            move._add_inner_visit(0, fast[call * 200 : (call + 1) * 200])
            move._add_inner_visit(1, slow[call * 200 : (call + 1) * 200])
        self.assertAlmostEqual(move.inner_act()[0], 20.0, delta=5.0)
        move.reset_inner_act(keep_chain=True)
        self.assertAlmostEqual(move.inner_act()[0], 20.0, delta=5.0)
        move.reset_inner_act()
        self.assertTrue(np.isnan(move.inner_act()[0]))

    def test_the_chain_runs_on_across_calls(self):
        # an ACT of 100 repeats in calls of 50: no call holds it, twenty calls end to end do
        move, chain = self.build(), ar1(4000, 14, 100.0, 7, ncols=2)
        move._add_inner_visit(0, chain[:50])
        self.assertTrue(np.isnan(move.inner_act()[0]))
        for call in range(1, 80):
            move._add_inner_visit(0, chain[call * 50 : (call + 1) * 50])
        self.assertEqual(len(move._inner_chains[0]), move.inner_window)  # the last 20 calls: 1000 repeats
        self.assertAlmostEqual(move.inner_act()[0], 100.0, delta=35.0)
        move.inner_window = 80  # set by the controller (window)
        for call in range(80):
            move._add_inner_visit(0, chain[call * 50 : (call + 1) * 50])
        self.assertAlmostEqual(move.inner_act()[0], 100.0, delta=25.0)

    def test_one_leaf_not_resolved(self):
        move = self.build()
        fast, slow = ar1(4000, 14, 5.0, 1, ncols=2), ar1(4000, 14, 3000.0, 2, ncols=2)  # about one ACT in the window
        for call in range(20):
            move._add_inner_visit(0, fast[call * 200 : (call + 1) * 200])
            move._add_inner_visit(1, slow[call * 200 : (call + 1) * 200])
        self.assertTrue(np.isnan(move.inner_act()[0]))

    def test_periodic_parameter(self):
        from eryn.utils import PeriodicContainer

        # by name, as a prior gives it: the container resolves the index
        move = self.build(PeriodicContainer({"mbh": {"phi": 2.0 * np.pi}}, key_order={"mbh": ["m", "phi"]}))
        chain = ar1(4000, 14, 10.0, 3, ncols=2)
        chain[:, :, 1] = np.mod(0.3 * chain[:, :, 1], 2.0 * np.pi)  # sits on the wrap
        for call in range(20):
            move._add_inner_visit(0, chain[call * 200 : (call + 1) * 200])
        self.assertAlmostEqual(move.inner_act()[0], 10.0, delta=3.0)


class InnerActMove:
    """What a RepeatController needs of a move; its inner ACT is ``true_tau`` repeats, resolved when the window
    (``inner_window`` calls at the current repeats) holds ten ACTs."""

    def __init__(self, true_tau, num_repeats=50):
        self.true_tau, self.num_repeats, self.calls = true_tau, num_repeats, 0
        self.record_inner_chain, self.inner_window = False, None

    def apply_settings(self, num_repeats=None, weights=None):
        self.num_repeats = num_repeats

    def reset_inner_act(self, keep_chain=False):
        self.calls = 0

    def inner_act(self):
        resolved = self.inner_window * self.num_repeats >= 10 * self.true_tau
        return (self.true_tau if resolved else np.nan), self.calls


class RepeatControllerTest(unittest.TestCase):
    def tune(self, move, updates=20, stored=None, **kwargs):
        controller = gf_recipe.RepeatController("mbh", **{"min": 5, "max": 1000, "every": 10, **kwargs})
        controller.attach(move, stored)
        history = []
        for _ in range(updates):
            move.calls = controller.every
            controller.update()
            history.append(move.num_repeats)
        return controller, history

    def test_settles_at_per_act_acts(self):
        for tau, start in [(20.0, 50), (3.0, 400), (60.0, 10)]:
            with self.subTest(tau=tau):
                move = InnerActMove(tau, start)
                controller, history = self.tune(move)
                self.assertEqual(history[-5:], [int(2 * tau)] * 5)  # no oscillation
                self.assertEqual(controller.kept, int(2 * tau))
                self.assertTrue(move.record_inner_chain)
                self.assertEqual(move.inner_window, 20)

    def test_step_limit_and_bounds(self):
        _controller, history = self.tune(InnerActMove(60.0, 10))
        self.assertEqual(history[:4], [20, 40, 80, 120])  # not resolved, then resolved: a factor max_step at a time
        controller, history = self.tune(InnerActMove(5000.0, 50), max=300)
        self.assertEqual(history[-1], 300)
        self.assertEqual(controller.kept, 300)  # never resolved: the maximum
        controller, history = self.tune(InnerActMove(400.0, 50), max=300)
        self.assertEqual((history[-1], controller.kept, controller.tau), (300, 300, 400.0))  # resolved, above max
        controller, history = self.tune(InnerActMove(0.2, 50), min=20)
        self.assertEqual((history[-1], controller.kept), (20, 20))

    def test_waits_for_enough_calls(self):
        move = InnerActMove(20.0, 50)
        controller = gf_recipe.RepeatController("mbh", min=5, max=1000, every=10)
        controller.attach(move)
        move.calls = 9  # e.g. the block's move took turns with another one
        self.assertIsNone(controller.update())
        self.assertEqual(move.num_repeats, 50)
        move.calls = 10
        self.assertEqual(controller.update(), {"tuning": 40, "kept": 40, "tau": 20.0})

    def test_resumes_stored_values(self):
        move = InnerActMove(20.0, 50)
        controller, _history = self.tune(move, updates=0, stored={"tuning": 200.0, "kept": 40.0, "tau": 20.0})
        self.assertEqual((move.num_repeats, controller.kept), (200, 40))


class RepeatsBackend(FakeBackend):
    def __init__(self, starts=None, repeats=None):
        super().__init__(starts)
        self.repeats = dict(repeats or {})

    def recipe_repeats(self, branch, values=None):
        if values is not None:
            self.repeats[branch] = dict(values)
        return dict(self.repeats.get(branch, {}))


class AdaptiveStagesTest(unittest.TestCase):
    STAGES = [{"name": "warm", "iterations": 30}, {"name": "tune", "iterations": 60}, {"name": "pe"}]
    SETTINGS = {name: {"mbh": {"num_repeats": 50}} for name in ("warm", "tune", "pe")}
    ADAPTIVE = {"mbh": ({"min": 5, "max": 1000, "every": 10}, set())}

    def recipe(self, move, backend, adaptive=None, stages=None):
        from lisatools.globalfit.moves import GFCombineMove  # noqa: F401  (the steps walk combined moves)

        recipe = gf_recipe.Recipe()
        StockRecipe(stages=stages or self.STAGES).add_steps(
            recipe, {"mbh": BranchMoves([], [move])}, self.SETTINGS.get, adaptive=adaptive or self.ADAPTIVE
        )
        recipe.backend = backend
        return recipe

    def run_recipe(self, recipe, sampler, move, until):
        local = 0
        while sampler.iteration < until:
            sampler.iteration += 1
            move.calls += 1  # the move ran in this iteration
            if recipe(local, None, sampler):
                return
            local += 1

    def move(self, tau=20.0):
        move = InnerActMove(tau)
        move.branch_name = "mbh"
        return move

    def test_tunes_then_freezes(self):
        move, sampler, backend = self.move(), FakeSampler(), RepeatsBackend()
        recipe = self.recipe(move, backend)
        recipe.setup_first_recipe_step(0, None, sampler)
        self.run_recipe(recipe, sampler, move, until=30)
        self.assertEqual(recipe.current_recipe_step["name"], "tune")
        self.assertEqual(move.num_repeats, 40)  # two inner ACTs; the stage settings (50) did not reset the tuning
        self.run_recipe(recipe, sampler, move, until=89)
        self.assertEqual(move.num_repeats, 40)
        self.assertEqual(backend.repeats["mbh"]["kept"], 40)

        self.run_recipe(recipe, sampler, move, until=90)
        self.assertEqual(recipe.current_recipe_step["name"], "pe")
        self.assertEqual(move.num_repeats, 40)  # as tuned, fixed
        self.assertFalse(move.record_inner_chain)
        move.true_tau = 100.0
        self.run_recipe(recipe, sampler, move, until=150)
        self.assertEqual(move.num_repeats, 40)

    def test_restart_inside_tuning_and_into_the_last_stage(self):
        stored = {"mbh": {"tuning": 200.0, "kept": 40.0, "tau": 20.0}}
        move, sampler = self.move(), FakeSampler(iteration=45)
        recipe = self.recipe(move, RepeatsBackend({"warm": 0, "tune": 30}, stored))
        recipe.recipe[0]["status"] = True
        recipe.setup_first_recipe_step(sampler.iteration, None, sampler)
        self.assertEqual(move.num_repeats, 200)
        self.assertTrue(move.record_inner_chain)

        move, sampler = self.move(), FakeSampler(iteration=120)
        recipe = self.recipe(move, RepeatsBackend({"warm": 0, "tune": 30, "pe": 90}, stored))
        recipe.recipe[0]["status"] = recipe.recipe[1]["status"] = True
        recipe.setup_first_recipe_step(sampler.iteration, None, sampler)
        self.assertEqual(move.num_repeats, 40)

    def test_a_stage_with_its_own_repeats_is_left_alone(self):
        move, sampler = self.move(), FakeSampler()
        recipe = self.recipe(move, RepeatsBackend(), adaptive={"mbh": (self.ADAPTIVE["mbh"][0], {"warm"})})
        recipe.setup_first_recipe_step(0, None, sampler)
        self.run_recipe(recipe, sampler, move, until=29)
        self.assertEqual(move.num_repeats, 50)
        self.assertFalse(move.record_inner_chain)
        self.run_recipe(recipe, sampler, move, until=60)
        self.assertEqual(move.num_repeats, 40)

    def test_backend_stores_the_repeats(self):
        import os
        import tempfile

        from lisatools.globalfit.hdfbackend import GFHDFBackend

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "backend.h5")
        backend = GFHDFBackend(path)
        backend.reset(4, [3], ntemps=1, nleaves_max=[1], branch_names=["mbh"])
        recipe = gf_recipe.Recipe()
        recipe.add_recipe_component(gf_recipe.PERecipeStep(moves=[]), name="pe")
        backend.add_recipe(recipe)
        self.assertEqual(backend.recipe_repeats("mbh"), {})
        backend.recipe_repeats("mbh", {"tuning": 200, "kept": 40, "tau": 20.0})
        backend.recipe_repeats("emri", {"tuning": 30, "kept": 6, "tau": 3.0})
        self.assertEqual(GFHDFBackend(path).recipe_repeats("mbh"), {"tuning": 200.0, "kept": 40.0, "tau": 20.0})


class ChainBackend:
    """A backend holding cold chains ``{branch: [nsteps, nwalkers, nleaves, ndim]}`` (and gb band counts)."""

    def __init__(self, chains, inds=None, bands=None):
        self.chains, self.bands = chains, bands
        self.inds = inds or {name: np.ones(chain.shape[:3], dtype=bool) for name, chain in chains.items()}
        self.iteration = 0
        self.sub_backend = {"gb": self}
        self.starts, self.completed = {}, []

    def get_chain(self, slice_vals, temp_index, branch_names):
        assert temp_index == 0
        return {name: self.chains[name][slice_vals] for name in branch_names}

    def get_inds(self, slice_vals, temp_index, branch_names):
        return {name: self.inds[name][slice_vals] for name in branch_names}

    def get_band_info(self, slice_vals):
        return {"band_num_binaries": self.bands[slice_vals][:, None]}  # a temperature axis, as stored

    def recipe_step_start(self, name, iteration):
        return self.starts.setdefault(name, iteration)

    def completed_recipe_step(self, name):
        self.completed.append(name)


class ConvergenceStepTest(unittest.TestCase):
    NWALKERS = 16

    def chain(self, nsteps, tau, nleaves=1, ndim=2, seed=0):
        return ar1(nsteps, self.NWALKERS, tau, seed, ncols=nleaves * ndim).reshape(nsteps, self.NWALKERS, nleaves, ndim)

    def stops_at(self, backend, monitors, target_ess, until, start=0, check_every=50):
        """The iteration the step stops at (None if it never does up to ``until``), and the step."""
        step = gf_recipe.ConvergenceRecipeStep(
            moves=[], monitors=monitors, target_ess=target_ess, check_every=check_every
        )
        step.start_iteration, step.name = start, "pe"
        sampler = types.SimpleNamespace(backend=backend, periodic=None)
        for iteration in range(start + 1, until + 1):
            backend.iteration = iteration
            if step.stopping_function(iteration, None, sampler):
                return iteration, step
        return None, step

    def test_stops_when_the_slowest_block_has_the_target(self):
        backend = ChainBackend({"mbh": self.chain(6000, 4.0), "emri": self.chain(6000, 20.0, seed=1)})
        monitors = [gf_recipe.ChainMonitor("mbh"), gf_recipe.ChainMonitor("emri")]
        stopped, step = self.stops_at(backend, monitors, target_ess=2000, until=6000)
        # emri gates: 16 walkers / ACT 20 -> 2000 samples after about 2500 iterations
        self.assertIsNotNone(stopped)
        self.assertGreater(stopped, 1800)
        self.assertLess(stopped, 3500)
        report = step.last_report
        self.assertGreater(report["mbh"]["ess"], 3 * report["emri"]["ess"])
        self.assertAlmostEqual(report["emri"]["tau"], 20.0, delta=5.0)

    def test_counts_from_the_stage_start(self):
        chain = self.chain(3000, 4.0)
        chain[:1000] += np.linspace(50.0, 0.0, 1000)[:, None, None, None]  # burn-in: a drift, not part of the stage
        backend = ChainBackend({"mbh": chain})
        stopped, _ = self.stops_at(backend, [gf_recipe.ChainMonitor("mbh")], 2000, until=3000, start=1000)
        self.assertIsNotNone(stopped)
        self.assertLess(stopped - 1000, 1200)
        stopped, _ = self.stops_at(backend, [gf_recipe.ChainMonitor("mbh")], 2000, until=3000, start=0)
        self.assertIsNone(stopped)  # with the drift in, the ACT is long

    def test_does_not_stop_on_a_stuck_or_short_chain(self):
        stuck = self.chain(2000, 4.0)
        stuck[:, 3, 0, 1] = stuck[0, 3, 0, 1]  # one walker never moves in one parameter
        stopped, step = self.stops_at(ChainBackend({"mbh": stuck}), [gf_recipe.ChainMonitor("mbh")], 100, until=2000)
        self.assertIsNone(stopped)
        self.assertTrue(np.isnan(step.last_report["mbh"]["tau"]))
        # 150 iterations of ACT 20: ESS 16 * 150 / 20 = 120 > 100, but the ACT is not reliable yet
        stopped, step = self.stops_at(
            ChainBackend({"mbh": self.chain(150, 20.0)}), [gf_recipe.ChainMonitor("mbh")], 100, until=150
        )
        self.assertIsNone(stopped)
        self.assertFalse(step.last_report["mbh"]["reliable"])

    def test_inactive_leaves_and_fixed_parameters_do_not_hold_the_run(self):
        chain = self.chain(3000, 4.0, nleaves=3, ndim=2)
        chain[:, :, 1] = 0.0  # an inactive leaf slot
        chain[:, :, :, 1] = 7.0  # a parameter held fixed
        inds = {"mbh": np.ones(chain.shape[:3], dtype=bool)}
        inds["mbh"][:, :, 1] = False
        stopped, step = self.stops_at(ChainBackend({"mbh": chain}, inds), [gf_recipe.ChainMonitor("mbh")], 2000, 3000)
        self.assertIsNotNone(stopped)
        self.assertIn("1 inactive leaves", step.last_report["mbh"]["note"])
        self.assertIn("2 fixed parameters", step.last_report["mbh"]["note"])

    def test_gb_gates_on_the_number_of_sources(self):
        nbands = 5
        bands = np.round(10.0 + 2.0 * ar1(4000, self.NWALKERS, 6.0, seed=4, ncols=nbands))
        bands[:, :, 0] = 4.0  # a band whose count never changes
        bands[:, :, 1] = 3.0
        bands[100, :, 1] = 4.0  # ... and one that flips once: would never be reliable
        backend = ChainBackend({}, bands=bands)
        stopped, step = self.stops_at(backend, [gf_recipe.GBCountMonitor("gb")], 2000, until=4000)
        self.assertIsNotNone(stopped)
        self.assertIn("4 of 5 bands change", step.last_report["gb"]["note"])
        # a constant number of sources gates nothing, and with nothing gating the step runs on
        frozen = ChainBackend({}, bands=np.full((1000, self.NWALKERS, nbands), 3.0))
        stopped, step = self.stops_at(frozen, [gf_recipe.GBCountMonitor("gb")], 10, until=1000)
        self.assertIsNone(stopped)
        self.assertFalse(step.last_report["gb"]["gating"])

    def test_stock_recipe_last_stage(self):
        recipe = gf_recipe.Recipe()
        stages = [{"name": "tune", "iterations": 5}, {"name": "pe", "target_ess": 1000, "check_every": 25}]
        monitors = [gf_recipe.ChainMonitor("mbh")]
        StockRecipe(stages=stages).add_steps(recipe, {"mbh": BranchMoves([], [fake_move()])}, monitors=monitors)
        step = recipe.recipe[-1]["adjust"]
        self.assertIsInstance(step, gf_recipe.ConvergenceRecipeStep)
        self.assertEqual((step.target_ess, step.check_every, step.monitors), (1000, 25, monitors))
        for stages, msg in [
            ([{"name": "a", "iterations": 5, "target_ess": 10}, {"name": "pe"}], "must be"),
            ([{"name": "pe", "target_ess": 0}], "positive integer"),
            ([{"name": "pe", "check_every": 10}], "needs target_ess"),
        ]:
            with self.subTest(stages=stages), self.assertRaisesRegex(ValueError, msg):
                StockRecipe(stages=stages)


class AdaptiveRepeatsSchemaTest(BlockStagesTest):
    TUNED = {"class": "StockRecipe", "kwargs": {"stages": [{"name": "tune", "iterations": 5}, {"name": "pe"}]}}

    def test_schema_and_controller_settings(self):
        from lisatools.globalfit.stock.l2d.build_recipe import adaptive_repeats

        block = {**MBH, "adaptive_repeats": {"min": 10, "max": 400}, "stages": {"pe": {"num_prop_repeats": 30}}}
        cfg = self.config({"mbh": block}, self.TUNED)
        kwargs, fixed = adaptive_repeats(cfg)["mbh"]
        self.assertEqual(kwargs, {"min": 10, "max": 400, "per_act": 2.0, "every": 10, "window": 20, "max_step": 2.0})
        self.assertEqual(fixed, {"pe"})
        gf_recipe.RepeatController("mbh", **kwargs)
        self.assertEqual(adaptive_repeats(self.config({"mbh": MBH}, self.TUNED)), {})

    def test_errors(self):
        for adaptive, recipe, extra, msg in [
            ({"min": 60}, self.TUNED, {}, "within"),  # num_prop_repeats is 50
            ({"min": 20, "max": 10}, self.TUNED, {}, "above max"),
            ({"window": 5}, self.TUNED, {}, "below every"),
            ({}, None, {}, "there is none"),  # one stage: nowhere to tune
            ({}, self.TUNED, {"stages": {"tune": {"num_prop_repeats": 20}}}, "there is none"),
        ]:
            with self.subTest(adaptive=adaptive, extra=extra), self.assertRaisesRegex(Exception, msg):
                self.config({"mbh": {**MBH, "adaptive_repeats": adaptive, **extra}}, recipe)


if __name__ == "__main__":
    unittest.main()
