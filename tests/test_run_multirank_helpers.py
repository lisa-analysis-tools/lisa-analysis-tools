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


if __name__ == "__main__":
    unittest.main()
