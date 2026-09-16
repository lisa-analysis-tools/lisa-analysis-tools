"""Head-directed commands over the fake communicator: round trip, errors, stop, isolation."""

import time
import unittest

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import (
    ComputeService,
    RemoteWorkerError,
    WalkerFanout,
    concat_blocks,
)
from lisatools.globalfit.communication.ranks import RankRole, build_layout


class _StubMove:
    gf_move_name = "stub"

    def __init__(self):
        self.calls = []
        self.gf_stage_kind = None

    def gf_serve(self, op, payload, clock, model):
        self.calls.append((op, self.gf_stage_kind, clock["call_index"]))
        if op == "boom":
            raise ValueError("kaboom")
        if op == "unpicklable":
            return {"fn": lambda: None}  # pickle.dumps rejects this, like FakeComm.send
        return {"rank_sum": float(np.sum(payload["x"])), "model": model}


def _run_world(size, nodes, nwalkers, head_fn, saver_fn=None):
    world = FakeWorld(size, nodes=nodes)
    stubs = {}

    def fn(rank, comm):
        layout = build_layout(comm, nwalkers, [0, 1], legacy=False)
        fcomm = layout.make_fanout_comm(comm)
        role = layout.role_of(rank)
        if role == RankRole.SAVER:
            return saver_fn(rank, comm) if saver_fn else "saver-idle"
        model = f"model-{rank}"
        if role == RankRole.HEAD:
            fo = WalkerFanout(fcomm, layout, rank, model=model)
            fo.enter_stage("pe", "pe_kind")
            try:
                return head_fn(fo, layout)
            finally:
                fo.stop()
        stubs[rank] = _StubMove()
        service = ComputeService(fcomm, layout, rank, registry={"stub": stubs[rank]}, model=model)
        return service.serve()

    return world.run(fn), stubs


class FanoutFakeCommTest(unittest.TestCase):
    def test_round_trip_in_walker_order_and_stage_stamp(self):
        def head(fo, layout):
            def payload(rank, w0, w1):
                return {"x": np.arange(w0, w1)}

            def body(p, model):
                return {"rank_sum": float(np.sum(p["x"])), "model": model}

            out = fo.run(
                "score", move="stub", per_rank_payload=payload, local_body=body, merge=lambda r: r
            )
            out2 = fo.run(
                "score", move="stub", per_rank_payload=payload, local_body=body, merge=lambda r: r
            )
            return (
                {r: rep["rank_sum"] for r, rep in out.items()},
                out[0]["model"],
                out[1]["model"],
                out2 == out,
            )

        (out, stubs) = _run_world(3, [0, 0, 0], 4, head)
        sums, m0, m1, same = out[0]
        self.assertEqual(sums, {0: 1.0, 1: 5.0})
        self.assertEqual((m0, m1), ("model-0", "model-1"))
        self.assertTrue(same)
        self.assertEqual(out[1], 2)  # the worker served two commands before stop
        self.assertEqual(out[2], "saver-idle")
        self.assertEqual(stubs[1].calls, [("score", "pe_kind", 1), ("score", "pe_kind", 2)])

    def test_worker_error_surfaces_on_the_head_with_traceback(self):
        def head(fo, layout):
            with self.assertRaises(RemoteWorkerError) as cm:
                fo.run(
                    "boom",
                    move="stub",
                    per_rank_payload=lambda r, w0, w1: {"x": np.zeros(1)},
                    local_body=lambda p, m: {},
                    merge=lambda r: r,
                )
            err = cm.exception
            return err.rank, err.op, err.move, "kaboom" in err.remote_traceback

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(out[0], (1, "boom", "stub", True))
        self.assertEqual(out[1], 1)

    def test_unpicklable_result_becomes_a_senderror_reply_not_a_hang(self):
        def head(fo, layout):
            with self.assertRaises(RemoteWorkerError) as cm:
                fo.run(
                    "unpicklable",
                    move="stub",
                    per_rank_payload=lambda r, w0, w1: {"x": np.zeros(1)},
                    local_body=lambda p, m: {},
                    merge=lambda r: r,
                )
            return cm.exception.rank, cm.exception.error["type"]

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(out[0], (1, "SendError"))
        self.assertEqual(out[1], 1)  # the worker still served the command and stopped cleanly

    def test_unknown_move_is_an_error_reply_not_a_hang(self):
        def head(fo, layout):
            with self.assertRaises(RemoteWorkerError) as cm:
                fo.run(
                    "score",
                    move="nope",
                    per_rank_payload=lambda r, w0, w1: {"x": np.zeros(1)},
                    local_body=lambda p, m: {},
                    merge=lambda r: r,
                )
            return cm.exception.error["type"]

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head)
        self.assertEqual(out[0], "KeyError")

    def test_saver_never_sees_fanout_traffic(self):
        def saver(rank, comm):
            time.sleep(0.2)
            return comm.iprobe(source=0)

        def head(fo, layout):
            return fo.run(
                "score",
                move="stub",
                per_rank_payload=lambda r, w0, w1: {"x": np.ones(2)},
                local_body=lambda p, m: {"rank_sum": 2.0},
                merge=lambda r: sorted(r),
            )

        (out, _stubs) = _run_world(3, [0, 0, 0], 4, head, saver_fn=saver)
        self.assertEqual(out[0], [0, 1])
        self.assertFalse(out[2])

    def test_ping_checks_layout_digests(self):
        (out, _stubs) = _run_world(3, [0, 0, 0], 4, lambda fo, layout: fo.ping())
        self.assertEqual(set(out[0].values()), {out[0][0]})
        self.assertEqual(sorted(out[0]), [0, 1])

    def test_allgather_walker_vector_is_a_setup_phase_collective(self):
        world = FakeWorld(3)

        def fn(rank, comm):
            layout = build_layout(comm, 4, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return None
            fo = WalkerFanout(fcomm, layout, rank)
            w0, w1 = layout.block_of(rank)
            return fo.allgather_walker_vector(np.arange(w0, w1) * 10.0)

        out = world.run(fn)
        np.testing.assert_array_equal(out[0], [0.0, 10.0, 20.0, 30.0])
        np.testing.assert_array_equal(out[1], out[0])

    def test_concat_blocks_follows_compute_rank_order(self):
        layout = FakeWorld(5, nodes=[0, 1, 0, 1, 0]).run(
            lambda r, c: build_layout(c, 8, [0, 1], legacy=False)
        )[0]
        results = {
            3: np.array([6, 7]),
            0: np.array([0, 1]),
            2: np.array([4, 5]),
            1: np.array([2, 3]),
        }
        np.testing.assert_array_equal(concat_blocks(results, layout), np.arange(8))

    def test_run_on_a_worker_raises(self):
        layout = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[1]
        fo = WalkerFanout(None, layout, 1)
        with self.assertRaises(RuntimeError):
            fo.run(
                "x",
                per_rank_payload=lambda r, w0, w1: None,
                local_body=lambda p, m: None,
                merge=lambda r: r,
            )


if __name__ == "__main__":
    unittest.main()
