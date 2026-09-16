"""Pure helpers run.py uses to wire the multi-rank roles (no build, no MPI)."""

import unittest

from eryn.moves import CombineMove

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import build_layout
from lisatools.globalfit.moves.functionmove import FunctionMove
from lisatools.globalfit.moves.globalfitmove import GlobalFitMove
from lisatools.globalfit.run import (
    _fanout_unready_moves,
    _leaf_moves,
    _rank_log_filenames,
)


class _Combine(CombineMove):
    """A bare eryn CombineMove holding the given moves (no proposal needed here)."""

    def __init__(self, moves):
        self.moves = moves


class _Served(GlobalFitMove):
    gf_move_name = "served"

    def gf_serve(self, op, payload, clock, model):
        return None


class _Unserved(GlobalFitMove):
    gf_move_name = "unserved"


def _fn(model, state):
    return state, None


class HelpersTest(unittest.TestCase):
    def test_leaf_moves_flattens_combines_and_tuples(self):
        a, b = _Served(name="a"), _Unserved(name="b")
        leaves = _leaf_moves([_Combine([a, (b, 0.5)]), _Combine([_Combine([a])])])
        self.assertEqual(leaves, [a, b, a])

    def test_fanout_unready_names_default_gf_serve_only(self):
        # FunctionMove(Move, GlobalFitMove, ErynMove) lives in moves/functionmove.py; read
        # its constructor (fn first, keyword name) and adapt this line if it differs.
        fn_move = FunctionMove(_fn, name="fn")
        plain_eryn = CombineMove(moves=[])  # a non-GlobalFitMove leaf: head-only, not refused
        names, head_only = _fanout_unready_moves(
            [_Combine([_Served(name="a"), _Unserved(name="b"), fn_move]), plain_eryn]
        )
        self.assertEqual(names, ["unserved"])
        self.assertEqual(head_only, ["CombineMove"])

    def test_rank_log_filenames(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(_rank_log_filenames(lay, 0), ("globalfit_run.log", "global_fit.log"))
        self.assertEqual(
            _rank_log_filenames(lay, 1), ("globalfit_run.rank1.log", "global_fit.rank1.log")
        )
        self.assertEqual(
            _rank_log_filenames(lay, 2), ("globalfit_run.rank2.log", "global_fit.rank2.log")
        )


class MaterializedMovesTest(unittest.TestCase):
    """The flat move list the multi-rank readiness guard is computed from."""

    def test_flattens_step_dicts_and_survives_a_moveless_step(self):
        from lisatools.globalfit.recipe import RecipeStep
        from lisatools.globalfit.run import _materialized_moves

        a, b = _Served(name="a"), _Unserved(name="b")
        recipe = type("R", (), {})()
        recipe.recipe = [
            {"name": "s1", "adjust": type("S", (), {"moves": [a]})(), "status": False},
            # a legacy step that was never given moves: ``moves`` RAISES
            {"name": "s2", "adjust": RecipeStep(), "status": False},
            {"name": "s3", "adjust": type("S", (), {"moves": [b]})(), "status": False},
        ]
        self.assertEqual(_materialized_moves(recipe), [a, b])

    def test_an_unmaterialized_recipe_is_empty(self):
        from lisatools.globalfit.run import _materialized_moves

        self.assertEqual(_materialized_moves(object()), [])


class RebuildViewTest(unittest.TestCase):
    def test_none_block_returns_the_same_state_object(self):
        from lisatools.globalfit.run import _rebuild_state_view
        from tests.test_gf_substate_roundtrip import make_state
        import numpy as np

        state = make_state(np.random.default_rng(3))
        self.assertIs(_rebuild_state_view(state, None), state)

    def test_block_returns_a_walker_slice_without_sub_states(self):
        from lisatools.globalfit.run import _rebuild_state_view
        from tests.test_gf_substate_roundtrip import make_state
        import numpy as np

        state = make_state(np.random.default_rng(3))
        part = _rebuild_state_view(state, (1, 3))
        self.assertEqual(part.branches["gb"].nwalkers, 2)
        np.testing.assert_array_equal(
            part.branches["gb"].coords, state.branches["gb"].coords[:, 1:3]
        )
        self.assertTrue(all(v is None for v in part.sub_states.values()))


class RecipeFanoutHooksTest(unittest.TestCase):
    def test_recipe_forwards_stage_and_iteration_to_the_fanout(self):
        from lisatools.globalfit.recipe import Recipe

        calls = []

        class _Fanout:
            def enter_stage(self, name, kind):
                calls.append(("stage", name, kind))

            def note_iteration(self, i):
                calls.append(("it", i))

        class _Step:
            def __init__(self, kind):
                self.moves = [type("M", (), {"gf_stage_kind": kind})()]
                self.stops = [False, True]

            def setup_run(self, iteration, last_sample, sampler):
                calls.append(("setup_run", iteration))

            def stopping_function(self, iteration, last_sample, sampler):
                return self.stops.pop(0)

        class _Backend:
            def completed_recipe_step(self, name):
                calls.append(("done", name))

        recipe = Recipe()
        recipe._init_runtime()
        recipe.recipe = [
            {"name": "a", "adjust": _Step("search"), "status": False},
            {"name": "b", "adjust": _Step("pe"), "status": False},
        ]
        recipe.backend = _Backend()
        recipe.fanout = _Fanout()
        recipe.setup_first_recipe_step(0, None, None)
        self.assertEqual(calls[-1], ("stage", "a", "search"))
        self.assertFalse(recipe(1, None, None))  # step a not done yet
        self.assertIn(("it", 1), calls)
        self.assertFalse(recipe(2, None, None))  # a done -> b set up
        self.assertEqual(calls[-1], ("stage", "b", "pe"))

    def test_no_fanout_is_a_no_op(self):
        """Single-process: ``Recipe.fanout`` stays ``None`` and nothing is called."""
        from lisatools.globalfit.recipe import Recipe

        calls = []

        class _Step:
            moves = []

            def setup_run(self, iteration, last_sample, sampler):
                calls.append(("setup_run", iteration))

            def stopping_function(self, iteration, last_sample, sampler):
                return False

        recipe = Recipe()
        recipe._init_runtime()
        recipe.recipe = [{"name": "a", "adjust": _Step(), "status": False}]
        self.assertIsNone(recipe.fanout)
        recipe.setup_first_recipe_step(0, None, None)
        self.assertFalse(recipe(1, None, None))
        self.assertEqual(calls, [("setup_run", 0)])


class StageKindOfTest(unittest.TestCase):
    def test_reads_the_combine_moves_stage_kind_through_tuples(self):
        from lisatools.globalfit.recipe import _stage_kind_of

        combined = type("M", (), {"gf_stage_kind": "rj"})()
        self.assertEqual(_stage_kind_of(type("S", (), {"moves": [combined]})()), "rj")
        self.assertEqual(
            _stage_kind_of(type("S", (), {"moves": [(combined, 1.0)]})()), "rj"
        )
        self.assertIsNone(_stage_kind_of(type("S", (), {"moves": []})()))
        self.assertIsNone(_stage_kind_of(object()))

    def test_a_moveless_recipe_step_does_not_raise(self):
        """``RecipeStep.moves`` RAISES when unset -- the helper must swallow that."""
        from lisatools.globalfit.recipe import RecipeStep, _stage_kind_of

        self.assertIsNone(_stage_kind_of(RecipeStep()))


if __name__ == "__main__":
    unittest.main()
