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


def _run_blocks(head_fn, nwalkers=2, size=5):
    """A world with BOTH axes live: n_blocks x R, not one replicated walker."""
    world = FakeWorld(size, nodes=[0, 1, 0, 1, 0][:size])
    moves = {}

    def fn(rank, comm):
        # this helper exists to make BOTH axes live, which needs the unified
        # routing; it is opt-in at the launcher (ranks.GPU_ROUTING_ENV).
        layout = build_layout(comm, nwalkers, [0, 1], legacy=False, gpu_routing=True)
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return "saver"
        move = _StubMove(100.0 * rank)
        moves[rank] = move
        fo = WalkerFanout(fcomm, layout, rank, model=None,
                          group_comm=layout.make_group_comm(fcomm),
                          reps_comm=layout.make_reps_comm(fcomm))
        if layout.role_of(rank) == RankRole.HEAD:
            fo.enter_stage("pe", "pe")
            try:
                return head_fn(RowFanout(fo, move), move, layout)
            finally:
                fo.stop()
        return ComputeService(fcomm, layout, rank, registry={("pe", "stub"): move}).serve()

    return world.run(fn), moves


class RowFanoutWalkerRoutingTest(unittest.TestCase):
    """Rows go to a rank that HOLDS the walker they score against.

    The original split was contiguous over every compute rank, which is
    correct only when every rank holds the same walker (one-walker replica
    mode). With ``n_blocks > 1`` a row scoring walker w may only be served
    by a rank whose block contains w -- any other rank's ACA has no such
    row. ``walkers=`` (GLOBAL indices) turns the split walker-aware and
    rewrites each shipped row's ``data_index`` to the BLOCK-LOCAL row.
    """

    def test_rows_reach_only_ranks_holding_their_walker(self):
        # 4 compute ranks, 2 walkers -> 2 blocks of 1, R = 2.
        # walker 0 -> ranks (0, 1); walker 1 -> ranks (2, 3).
        coords = np.arange(8.0).reshape(4, 2)
        walkers = np.array([0, 0, 1, 1])

        def head(rf, move, layout):
            self.assertEqual((layout.n_blocks, layout.ranks_per_block), (2, 2))
            return rf.run("ll_rows",
                          {"coords": coords, "data_index": walkers.astype(np.int32)},
                          local_body=move.score, walkers=walkers)

        out, _ = _run_blocks(head)
        tags = np.asarray(out[0]["ll"]) - coords.sum(axis=1)
        # walker-0 rows served by rank 0 or 1; walker-1 rows by rank 2 or 3
        self.assertTrue(set(tags[:2]).issubset({0.0, 100.0}), tags)
        self.assertTrue(set(tags[2:]).issubset({200.0, 300.0}), tags)

    def test_data_index_is_rewritten_to_the_block_local_row(self):
        # Every block here is ONE walker wide, so the local row is always 0
        # even though the global walker index is 0 or 1. Shipping the global
        # index would index past the end of a 1-row ACA.
        coords = np.arange(8.0).reshape(4, 2)
        walkers = np.array([0, 0, 1, 1])

        def head(rf, move, layout):
            return rf.run("ll_rows",
                          {"coords": coords, "data_index": walkers.astype(np.int32)},
                          local_body=move.score, walkers=walkers)

        out, _ = _run_blocks(head)
        np.testing.assert_array_equal(out[0]["idx"], np.zeros(4, dtype=int))

    def test_results_come_back_in_the_original_row_order(self):
        coords = np.arange(12.0).reshape(6, 2)
        walkers = np.array([1, 0, 1, 0, 1, 0])  # deliberately interleaved

        def head(rf, move, layout):
            return rf.run("ll_rows",
                          {"coords": coords, "data_index": np.zeros(6, np.int32)},
                          local_body=move.score, walkers=walkers)

        out, _ = _run_blocks(head)
        tags = np.asarray(out[0]["ll"]) - coords.sum(axis=1)
        for i, w in enumerate(walkers):
            expected = {0.0, 100.0} if w == 0 else {200.0, 300.0}
            self.assertIn(tags[i], expected, msg=f"row {i} (walker {w})")

    def test_a_block_wider_than_one_walker_maps_to_its_local_row(self):
        # 2 compute ranks, 4 walkers -> 2 blocks of 2, R = 1.
        # walker 3 lives in block 1 at LOCAL row 1.
        def head(rf, move, layout):
            self.assertEqual((layout.n_blocks, layout.block), (2, 2))
            w = np.array([0, 1, 2, 3])
            return rf.run("ll_rows",
                          {"coords": np.zeros((4, 2)), "data_index": w.astype(np.int32)},
                          local_body=move.score, walkers=w)

        out, _ = _run_blocks(head, nwalkers=4, size=3)
        np.testing.assert_array_equal(out[0]["idx"], [0, 1, 0, 1])

    def test_without_walkers_the_contiguous_split_is_unchanged(self):
        # the one-walker replica path must stay byte-for-byte what it was
        coords = np.arange(14.0).reshape(7, 2)

        def head(rf, move):
            return rf.run("ll_rows",
                          {"coords": coords, "data_index": np.zeros(7, np.int32)},
                          local_body=move.score)

        out, _ = _run(head)
        tags = np.array([0, 0, 0, 0, 100, 100, 100], dtype=float)
        np.testing.assert_array_equal(out[0]["ll"], coords.sum(axis=1) + tags)


if __name__ == "__main__":
    unittest.main()
