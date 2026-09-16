"""Pure helpers run.py uses to wire the multi-rank roles (no build, no MPI)."""

import unittest
from types import SimpleNamespace

from eryn.moves import CombineMove

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import build_layout
from lisatools.globalfit.moves.functionmove import FunctionMove
from lisatools.globalfit.moves.globalfitmove import GlobalFitMove
from lisatools.globalfit.run import (
    _fanout_unready_moves,
    _leaf_moves,
    _rank_log_filenames,
    _serve_registry,
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


class _NoName(GlobalFitMove):
    """A leaf no Stage stamped (no ``gf_move_name``)."""


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
        # FunctionMove is NAMED head-only (it runs fn on the head's block), in
        # encounter order ahead of the plain eryn leaf.
        self.assertEqual(head_only, ["fn", "CombineMove"])

    def test_rank_log_filenames(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(_rank_log_filenames(lay, 0), ("globalfit_run.log", "global_fit.log"))
        self.assertEqual(
            _rank_log_filenames(lay, 1), ("globalfit_run.rank1.log", "global_fit.rank1.log")
        )
        self.assertEqual(
            _rank_log_filenames(lay, 2), ("globalfit_run.rank2.log", "global_fit.rank2.log")
        )


def _steps(*pairs):
    """A materialized step list: ``(step_name, [moves])`` pairs -> recipe dicts."""
    return SimpleNamespace(
        recipe=[
            {"name": name, "adjust": SimpleNamespace(moves=list(moves)), "status": False}
            for name, moves in pairs
        ]
    )


class ServeRegistryTest(unittest.TestCase):
    """The registry is keyed from the STEP LIST, never from ``gf_stage_name``.

    A stock runtime move is ONE object shared by every stage that lists it
    (``Move.setup`` returns ``ctx.stock_moves[name]``) and ``Stage.setup``
    re-stamps ``gf_stage_name`` on it per stage, so the stamp keeps only the
    LAST stage. Keying off the stamp would leave the earlier stage's commands
    unaddressable (KeyError -> RemoteWorkerError at the stage boundary).
    """

    def _recipe(self):
        # ``shared`` == one stock object listed by BOTH stages (psd_pe);
        # ``ridge_a``/``ridge_b`` == two DIFFERENT objects under one name.
        shared = _Served(name="psd_pe")
        shared.gf_move_name = "psd_pe"
        shared.gf_stage_name = "full_pe"  # the LAST stage's stamp: the defect
        ridge_a, ridge_b = _Served(name="ridge_a"), _Served(name="ridge_b")
        ridge_a.gf_move_name = ridge_b.gf_move_name = "gb_ridge_gibbs"
        ridge_a.gf_stage_name = ridge_b.gf_stage_name = "full_pe"
        recipe = _steps(
            ("noise_search", [shared, ridge_a]),
            ("full_pe", [_Combine([shared, ridge_b])]),
        )
        return recipe, shared, ridge_a, ridge_b

    def test_every_stage_that_lists_a_move_gets_its_tuple_key(self):
        recipe, shared, ridge_a, ridge_b = self._recipe()
        reg = _serve_registry(recipe)
        self.assertIs(reg[("noise_search", "psd_pe")], shared)
        self.assertIs(reg[("full_pe", "psd_pe")], shared)
        self.assertIs(reg[("noise_search", "gb_ridge_gibbs")], ridge_a)
        self.assertIs(reg[("full_pe", "gb_ridge_gibbs")], ridge_b)

    def test_bare_name_only_when_the_name_maps_to_one_object(self):
        recipe, shared, _a, _b = self._recipe()
        reg = _serve_registry(recipe)
        self.assertIs(reg["psd_pe"], shared)
        # two DIFFERENT objects under one name: no unambiguous bare fallback
        self.assertNotIn("gb_ridge_gibbs", reg)

    def test_two_different_objects_in_one_stage_raise(self):
        a, b = _Served(name="a"), _Served(name="b")
        a.gf_move_name = b.gf_move_name = "psd_pe"
        with self.assertRaisesRegex(RuntimeError, "two different moves named 'psd_pe'"):
            _serve_registry(_steps(("pe", [a, b])))

    def test_leaves_without_a_move_name_are_skipped(self):
        unnamed = _NoName(name="x")  # never stamped by a Stage: not served
        self.assertEqual(_serve_registry(_steps(("pe", [unnamed]))), {})

    def test_a_step_whose_moves_raise_is_skipped(self):
        from lisatools.globalfit.recipe import RecipeStep

        named = _Served(name="psd_pe")
        named.gf_move_name = "psd_pe"
        recipe = SimpleNamespace(
            recipe=[
                # a legacy step that was never given moves: ``moves`` RAISES
                {"name": "s1", "adjust": RecipeStep(), "status": False},
                {"name": "s2", "adjust": SimpleNamespace(moves=[named]), "status": False},
            ]
        )
        reg = _serve_registry(recipe)
        self.assertEqual(sorted(k for k in reg if isinstance(k, tuple)), [("s2", "psd_pe")])

    def test_an_unmaterialized_recipe_is_empty(self):
        self.assertEqual(_serve_registry(object()), {})


def _layout(size, gpus=(0, 1)):
    """A resolved WalkerBlockLayout over ``size`` fake ranks (4 walkers)."""
    return FakeWorld(size).run(
        lambda r, c: build_layout(c, 4, list(gpus), legacy=False)
    )[0]


def _gf_skeleton(layout, rank, **attrs):
    """A bare ``GlobalFit`` carrying only the attributes a helper reads."""
    from lisatools.globalfit.run import GlobalFit

    gf = GlobalFit.__new__(GlobalFit)
    gf.layout = layout
    gf.rank = rank
    gf.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    gf.curr = SimpleNamespace(general_info=SimpleNamespace(gpus=[], random_seed=None))
    gf.fanout_comm = None
    for key, value in attrs.items():
        setattr(gf, key, value)
    return gf


class SeedBaseTest(unittest.TestCase):
    """One resolved seed base, drawn on the head and shipped with the state."""

    def test_single_mode_resolves_to_none(self):
        gf = _gf_skeleton(_layout(1, gpus=(0,)), 0)
        gf.curr.general_info.random_seed = 103209
        self.assertIsNone(gf._resolve_seed_base())

    def test_multi_rank_uses_the_configured_seed(self):
        gf = _gf_skeleton(_layout(3), 0)
        gf.curr.general_info.random_seed = 103209
        self.assertEqual(gf._resolve_seed_base(), 103209)

    def test_multi_rank_without_a_seed_draws_a_fresh_one(self):
        gf = _gf_skeleton(_layout(3), 0)
        first, second = gf._resolve_seed_base(), gf._resolve_seed_base()
        for value in (first, second):
            self.assertIsInstance(value, int)
            self.assertTrue(0 <= value < 2**32 - 1)
        # a fixed fallback would hand every run and every resubmit one stream
        self.assertNotEqual(first, second)

    def test_rank_streams_derive_from_the_shipped_base(self):
        import numpy as np

        from lisatools.globalfit.communication.ranks import derive_rank_seed

        layout = _layout(3)
        logged = []
        gf = _gf_skeleton(layout, 1, _seed_base=123)
        gf.logger = SimpleNamespace(info=lambda *a: logged.append(a))
        seed = gf._seed_rank_streams()
        self.assertEqual(seed, derive_rank_seed(123, layout, 1))
        self.assertNotEqual(seed, derive_rank_seed(123, layout, 0))
        self.assertTrue(logged)
        # np.random really was reseeded with it
        self.assertEqual(
            np.random.random_sample(), np.random.RandomState(seed).random_sample()
        )

    def test_single_mode_reseeds_nothing(self):
        gf = _gf_skeleton(_layout(1, gpus=(0,)), 0, _seed_base=None)
        self.assertIsNone(gf._seed_rank_streams())

    def test_the_fanout_clock_carries_the_resolved_base(self):
        gf = _gf_skeleton(_layout(3), 0, _seed_base=123)
        fanout = gf._make_fanout(model=None)
        self.assertEqual(fanout.clock["seed_base"], 123)

    def test_single_mode_has_no_fanout(self):
        gf = _gf_skeleton(_layout(1, gpus=(0,)), 0)
        self.assertIsNone(gf._make_fanout(model=None))


class _LikelihoodAcs:
    """Stands in for the rank's B-row AnalysisContainerArray."""

    def __init__(self, values):
        import numpy as np

        self.values = np.asarray(values, dtype=float)

    def likelihood(self, complex=False):
        return self.values


class GlobalLikelihoodTest(unittest.TestCase):
    """``_global_likelihood`` == the FULL walker vector on every compute rank."""

    def test_single_mode_returns_the_local_vector_as_a_host_array(self):
        import numpy as np

        gf = _gf_skeleton(_layout(1, gpus=(0,)), 0, fanout=None)
        out = gf._global_likelihood(_LikelihoodAcs([1.0, 2.0, 3.0, 4.0]))
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])

    def test_multi_rank_allgathers_the_blocks_in_walker_order(self):
        import numpy as np

        from lisatools.globalfit.communication.fanout import WalkerFanout
        from lisatools.globalfit.communication.ranks import RankRole

        blocks = {0: [10.0, 11.0], 1: [12.0, 13.0]}

        def fn(rank, comm):
            layout = build_layout(comm, 4, [0, 1], legacy=False)
            # Split is a WORLD collective: every rank, saver included, calls it.
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return "saver-idle"
            fanout = WalkerFanout(fcomm, layout, rank, model=None)
            gf = _gf_skeleton(layout, rank, fanout=fanout)
            return gf._global_likelihood(_LikelihoodAcs(blocks[rank]))

        out = FakeWorld(3).run(fn)
        # the SAME full vector on both compute ranks, head's block first
        np.testing.assert_array_equal(out[0], [10.0, 11.0, 12.0, 13.0])
        np.testing.assert_array_equal(out[1], [10.0, 11.0, 12.0, 13.0])


class SubmissionWriterGuardTest(unittest.TestCase):
    """The submission dump reads the head's ACA, which holds ONE walker block.

    ``save_residuals`` writes ``residual_0..residual_{B-1}`` under full-run
    names and ``_prepare_gb_samples`` argmaxes over the block, so a multi-rank
    dump would silently ship a fraction of the run as the whole thing.
    """

    def _gf(self, layout):
        gf = _gf_skeleton(layout, 0)
        gf.curr.general_info.submission_parent_folder = "x"
        gf.run_backend = "BACKEND"
        gf.acs = "ACS"
        gf.warnings = []
        gf.logger = SimpleNamespace(
            info=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            warning=lambda *a: gf.warnings.append(a),
        )
        return gf

    def test_multi_rank_skips_the_dump_and_warns(self):
        from unittest import mock

        gf = self._gf(_layout(3))
        with mock.patch("lisatools.globalfit.run.SubmissionWriter") as writer:
            gf._write_submission()
        writer.assert_not_called()
        self.assertEqual(len(gf.warnings), 1)
        self.assertIn("SKIPPED", gf.warnings[0][0])

    def test_single_rank_writes_it_exactly_as_before(self):
        from unittest import mock

        gf = self._gf(_layout(1, gpus=(0,)))
        with mock.patch("lisatools.globalfit.run.SubmissionWriter") as writer:
            gf._write_submission()
        writer.assert_called_once_with(backend="BACKEND", curr=gf.curr, ess=20_000)
        writer.return_value.write_submission.assert_called_once_with("ACS")
        self.assertEqual(gf.warnings, [])

    def test_no_submission_folder_writes_nothing(self):
        from unittest import mock

        gf = self._gf(_layout(1, gpus=(0,)))
        gf.curr.general_info.submission_parent_folder = None
        with mock.patch("lisatools.globalfit.run.SubmissionWriter") as writer:
            gf._write_submission()
        writer.assert_not_called()


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
        # the clock is seeded BEFORE the first propose, not only after it
        self.assertIn(("it", 0), calls)
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
