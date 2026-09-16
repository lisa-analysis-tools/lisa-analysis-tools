"""The in-process fake communicator behaves like the mpi4py subset the global fit uses."""

import unittest

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeAbort, FakeWorld


class FakeCommTest(unittest.TestCase):
    def test_send_recv_pickle_copies(self):
        world = FakeWorld(2)

        def fn(rank, comm):
            if rank == 0:
                payload = {"a": np.arange(3)}
                comm.send(payload, dest=1)
                payload["a"][0] = 99  # must NOT reach rank 1
                return comm.recv(source=1)
            got = comm.recv(source=0)
            comm.send(int(got["a"][0]), dest=0)
            return got

        out = world.run(fn)
        self.assertEqual(out[0], 0)
        np.testing.assert_array_equal(out[1]["a"], [0, 1, 2])

    def test_isend_iprobe(self):
        world = FakeWorld(2)

        def fn(rank, comm):
            if rank == 0:
                req = comm.isend("hello", dest=1)
                req.wait()
                return comm.recv(source=1)
            while not comm.iprobe(source=0):
                pass
            msg = comm.recv(source=0)
            comm.send(msg + "!", dest=0)
            return msg

        out = world.run(fn)
        self.assertEqual(out, {0: "hello!", 1: "hello"})

    def test_bcast_and_allgather(self):
        world = FakeWorld(3)

        def fn(rank, comm):
            state = comm.bcast({"x": rank} if rank == 0 else None, root=0)
            return state["x"], comm.allgather(rank * 10)

        out = world.run(fn)
        for r in range(3):
            self.assertEqual(out[r], (0, [0, 10, 20]))

    def test_split_type_by_node_and_split_by_colour(self):
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])

        def fn(rank, comm):
            node = comm.Split_type(comm.COMM_TYPE_SHARED, key=rank)
            color = comm.UNDEFINED if rank == 4 else 0
            sub = comm.Split(color, key=rank)
            sub_info = None if rank == 4 else (sub.Get_rank(), sub.Get_size())
            return (comm.Get_processor_name(), node.Get_rank(), node.Get_size(), sub_info)

        out = world.run(fn)
        self.assertEqual(out[0], ("node0", 0, 3, (0, 4)))
        self.assertEqual(out[2], ("node0", 1, 3, (2, 4)))
        self.assertEqual(out[4], ("node0", 2, 3, None))
        self.assertEqual(out[1], ("node1", 0, 2, (1, 4)))
        self.assertEqual(out[3], ("node1", 1, 2, (3, 4)))

    def test_null_comm_raises(self):
        world = FakeWorld(1)
        with self.assertRaises(RuntimeError):
            world.run(lambda r, c: c.Split(c.UNDEFINED, key=r).Get_rank())

    def test_abort_unblocks_pending_recv_and_reports_root_cause(self):
        world = FakeWorld(2, timeout=5.0)

        def fn(rank, comm):
            if rank == 0:
                raise ValueError("boom")
            return comm.recv(source=0)  # would block forever without the abort

        with self.assertRaises(RuntimeError) as cm:
            world.run(fn)
        self.assertIn("boom", str(cm.exception))
        self.assertIsNotNone(world.aborted)

    def test_explicit_abort(self):
        world = FakeWorld(2, timeout=5.0)

        def fn(rank, comm):
            if rank == 1:
                comm.Abort(3)
            return comm.recv(source=1)

        with self.assertRaises(RuntimeError) as cm:
            world.run(fn)
        self.assertIn("Abort", str(cm.exception))
        self.assertEqual(world.aborted, (3, 1))


if __name__ == "__main__":
    unittest.main()
