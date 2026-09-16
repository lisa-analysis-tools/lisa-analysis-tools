"""MoveBuildContext exposes the rank layout / fan-out; GlobalFitMove.gf_serve defaults to 'not served'."""

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
        self.assertEqual((ctx.layout, ctx.fanout, ctx.rank, ctx.state_local), ("MINE", "FANOUT", 0, "SLICE"))


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
