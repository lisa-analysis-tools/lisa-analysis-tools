"""RowFanout: contiguous row chunks over the compute ranks, gathered in row order."""

import unittest

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.communication.rowfanout import RowFanout


class _StubMove:
    gf_move_name = "stub"

    def __init__(self, tag):
        self.tag = float(tag)
        self.replays = []

    def gf_serve(self, op, payload, clock, model):
        return getattr(self, f"serve_{op}")(payload, clock, model)

    def score(self, rows):
        coords = np.asarray(rows["coords"])
        return {"ll": coords.sum(axis=1) + self.tag, "idx": np.asarray(rows["data_index"])}

    def serve_ll_rows(self, payload, clock, model):
        return self.score(payload["rows"])

    def serve_ar_replay(self, payload, clock, model):
        self.replays.append(payload)
        return None


def _run(head_fn, size=3):
    world = FakeWorld(size, nodes=[0] * size)
    moves = {}

    def fn(rank, comm):
        layout = build_layout(comm, 1, [0, 1], legacy=False)  # nwalkers=1 -> replica mode
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return "saver"
        move = _StubMove(100.0 * rank)
        moves[rank] = move
        fo = WalkerFanout(fcomm, layout, rank, model=None)
        if layout.role_of(rank) == RankRole.HEAD:
            fo.enter_stage("pe", "pe")
            try:
                return head_fn(RowFanout(fo, move), move)
            finally:
                fo.stop()
        return ComputeService(fcomm, layout, rank, registry={("pe", "stub"): move}).serve()

    return world.run(fn), moves


class RowFanoutTest(unittest.TestCase):
    def test_rows_split_contiguously_and_gather_in_order(self):
        coords = np.arange(14.0).reshape(7, 2)
        idx = np.zeros(7, dtype=np.int32)

        def head(rf, move):
            self.assertTrue(rf.active)
            return rf.run("ll_rows", {"coords": coords, "data_index": idx}, local_body=move.score)

        out, _ = _run(head)
        # np.array_split(7, 2) -> rows 0..3 on the head (tag 0), rows 4..6 on rank 1 (tag 100)
        tags = np.array([0, 0, 0, 0, 100, 100, 100], dtype=float)
        np.testing.assert_array_equal(out[0]["ll"], coords.sum(axis=1) + tags)
        np.testing.assert_array_equal(out[0]["idx"], idx)

    def test_fewer_rows_than_ranks_and_zero_rows(self):
        def head(rf, move):
            one = rf.run("ll_rows", {"coords": np.ones((1, 2)), "data_index": np.zeros(1, np.int32)},
                         local_body=move.score)
            zero = rf.run("ll_rows", {"coords": np.ones((0, 2)), "data_index": np.zeros(0, np.int32)},
                          local_body=move.score)
            return one, zero

        out, _ = _run(head)
        one, zero = out[0]
        np.testing.assert_array_equal(one["ll"], [2.0])  # head-only chunk, rank 1 got an empty chunk
        self.assertEqual(zero["ll"].shape, (0,))

    def test_replay_reaches_every_rank_including_head(self):
        seen = []

        def head(rf, move):
            rf.replay("ar_replay", {"kind": "expose", "leaf": 3}, local_body=lambda p: seen.append(p))
            return len(seen)

        out, moves = _run(head)
        self.assertEqual(out[0], 1)
        self.assertEqual(seen, [{"kind": "expose", "leaf": 3}])
        self.assertEqual(moves[1].replays, [{"kind": "expose", "leaf": 3}])

    def test_single_rank_is_a_direct_call(self):
        move = _StubMove(7.0)
        rf = RowFanout(None, move)
        self.assertFalse(rf.active)
        out = rf.run("ll_rows", {"coords": np.ones((3, 2)), "data_index": np.zeros(3, np.int32)},
                     local_body=move.score)
        np.testing.assert_array_equal(out["ll"], [9.0, 9.0, 9.0])
        calls = []
        rf.replay("ar_replay", {"k": 1}, local_body=calls.append)
        self.assertEqual(calls, [{"k": 1}])

    def test_row_length_mismatch_raises(self):
        rf = RowFanout(None, _StubMove(0.0))
        with self.assertRaises(ValueError):
            rf.run("ll_rows", {"coords": np.ones((3, 2)), "data_index": np.zeros(2)}, local_body=lambda r: r)


if __name__ == "__main__":
    unittest.main()
