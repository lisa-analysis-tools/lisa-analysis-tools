"""MoveBuildContext exposes the rank layout / fan-out; GlobalFitMove.gf_serve
defaults to 'not served'."""

import unittest

from lisatools.globalfit.moves.globalfitmove import GlobalFitMove, MoveBuildContext


class _Curr:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class MoveBuildContextLayoutTest(unittest.TestCase):
    def _ctx(self, curr, **overrides):
        return MoveBuildContext(
            recipe=None, engine_info=None, curr=curr, acs=None, priors={}, state=None, **overrides
        )

    def test_fields_default_to_none_without_a_layout(self):
        ctx = self._ctx(_Curr())
        self.assertIsNone(ctx.layout)
        self.assertIsNone(ctx.fanout)
        self.assertIsNone(ctx.rank)
        self.assertIsNone(ctx.state_local)

    def test_fields_auto_fill_from_curr(self):
        curr = _Curr(rank_layout="LAYOUT", fanout="FANOUT", rank=3)
        ctx = self._ctx(curr)
        self.assertEqual((ctx.layout, ctx.fanout, ctx.rank), ("LAYOUT", "FANOUT", 3))

    def test_explicit_fields_win(self):
        curr = _Curr(rank_layout="LAYOUT", fanout="FANOUT", rank=3)
        ctx = self._ctx(curr, layout="MINE", rank=0, state_local="SLICE")
        self.assertEqual(
            (ctx.layout, ctx.fanout, ctx.rank, ctx.state_local),
            ("MINE", "FANOUT", 0, "SLICE"),
        )


class StateLocalAutoFillTest(unittest.TestCase):
    """``state_local`` is the ctx's OWN walker slice on a multi-rank build.

    The variants build the ctx themselves and never pass ``state_local``, so
    without this fill WP3's documented "act on this rank's rows only"
    mechanism is inert on every rank but the head.
    """

    def _layout(self, size, gpus=(0, 1)):
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import build_layout

        return FakeWorld(size).run(
            lambda r, c: build_layout(c, 4, list(gpus), legacy=False)
        )[0]

    def _state(self):
        import numpy as np

        from tests.test_gf_substate_roundtrip import make_state

        return make_state(np.random.default_rng(7))

    def _ctx(self, state, **curr_attrs):
        return MoveBuildContext(
            recipe=None,
            engine_info=None,
            curr=_Curr(**curr_attrs),
            acs=None,
            priors={},
            state=state,
        )

    def test_multi_rank_ctx_slices_its_own_block(self):
        import numpy as np

        state = self._state()
        layout = self._layout(3)
        ctx = self._ctx(state, rank_layout=layout, rank=1, fanout=None)
        self.assertEqual(layout.block_of(1), (2, 4))
        self.assertIsNotNone(ctx.state_local)
        for name in state.branches:
            self.assertEqual(ctx.state_local.branches[name].nwalkers, 2)
            np.testing.assert_array_equal(
                ctx.state_local.branches[name].coords,
                state.branches[name].coords[:, 2:4],
            )
        # sliced WITHOUT sub-states (the head-side slice in run.py matches)
        self.assertTrue(all(v is None for v in ctx.state_local.sub_states.values()))

    def test_single_layout_leaves_it_none(self):
        state = self._state()
        ctx = self._ctx(state, rank_layout=self._layout(1, gpus=(0,)), rank=0, fanout=None)
        self.assertIsNone(ctx.state_local)

    def test_no_layout_leaves_it_none(self):
        state = self._state()
        ctx = self._ctx(state, rank_layout=None, rank=None, fanout=None)
        self.assertIsNone(ctx.state_local)

    def test_an_explicit_state_local_is_never_overwritten(self):
        state = self._state()
        layout = self._layout(3)
        ctx = MoveBuildContext(
            recipe=None,
            engine_info=None,
            curr=_Curr(rank_layout=layout, rank=1, fanout=None),
            acs=None,
            priors={},
            state=state,
            state_local="MINE",
        )
        self.assertEqual(ctx.state_local, "MINE")


class _PlainMove(GlobalFitMove):
    gf_move_name = "plain"


class GfServeDefaultTest(unittest.TestCase):
    def test_default_gf_serve_raises_naming_the_move(self):
        move = _PlainMove(name="plain")
        with self.assertRaises(NotImplementedError) as cm:
            move.gf_serve("op", None, {}, None)
        self.assertIn("plain", str(cm.exception))
        self.assertIn("op", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
