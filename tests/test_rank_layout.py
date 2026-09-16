"""Rank roles + walker-block layout resolved identically on every rank (FakeComm)."""

import unittest

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import (
    RankRole,
    build_layout,
    derive_rank_seed,
    resolve_roles,
)


def _layouts(world, nwalkers, pool, **kwargs):
    return world.run(lambda r, c: build_layout(c, nwalkers, pool, legacy=False, **kwargs))


class ResolveRolesTest(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(resolve_roles(1), (0, 0, (0,)))
        self.assertEqual(resolve_roles(2), (0, 0, (0, 1)))
        self.assertEqual(resolve_roles(3), (0, 2, (0, 1)))
        self.assertEqual(resolve_roles(5), (0, 4, (0, 1, 2, 3)))
        with self.assertRaises(ValueError):
            resolve_roles(0)


class BuildLayoutTest(unittest.TestCase):
    def test_two_nodes_one_rank_per_gpu(self):
        # cyclic placement: ranks 0,2,4 on node0, ranks 1,3 on node1; rank 4 = saver
        world = FakeWorld(5, nodes=[0, 1, 0, 1, 0])
        outs = _layouts(world, 8, [0, 1])
        lay = outs[0]
        for r in range(5):
            self.assertEqual(outs[r].describe(), lay.describe())
            self.assertEqual(outs[r].digest(), lay.digest())
        self.assertEqual(lay.compute_ranks, (0, 1, 2, 3))
        self.assertEqual(lay.saver_rank, 4)
        self.assertEqual(lay.worker_ranks, (1, 2, 3))
        self.assertEqual(lay.block, 2)
        self.assertFalse(lay.is_single())
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(lay.placements[2].devices, (1,))
        self.assertEqual(lay.placements[1].devices, (0,))
        self.assertEqual(lay.placements[3].devices, (1,))
        blocks = [lay.block_of(r) for r in lay.compute_ranks]
        self.assertEqual(blocks, [(0, 2), (2, 4), (4, 6), (6, 8)])
        self.assertEqual(lay.role_of(4), RankRole.SAVER)
        self.assertEqual(lay.block_of(4), (0, 0))
        self.assertEqual(lay.local_gpus(3), [1])
        self.assertEqual(lay.fanout_rank(3), 3)
        self.assertEqual(lay.placements[4].local_index, 2)

    def test_ranks_per_gpu_share_one_device(self):
        world = FakeWorld(3)  # head + one compute rank + saver, ONE GPU
        lay = _layouts(world, 6, [0], ranks_per_gpu=2)[0]
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(lay.placements[1].devices, (0,))
        self.assertEqual((lay.placements[0].device_slot, lay.placements[1].device_slot), (0, 1))
        self.assertEqual(lay.ranks_on_device("node0", 0), (0, 1))
        self.assertEqual(lay.block, 3)

    def test_gpus_per_rank_owns_two_devices(self):
        world = FakeWorld(2)  # saver aliased to the head: compute = (0, 1)
        lay = _layouts(world, 4, [0, 1, 2, 3], gpus_per_rank=2)[0]
        self.assertEqual(lay.placements[0].devices, (0, 1))
        self.assertEqual(lay.placements[1].devices, (2, 3))
        self.assertEqual(lay.local_gpus(1), [2, 3])

    def test_errors(self):
        world = FakeWorld(3)
        with self.assertRaises(RuntimeError):
            _layouts(world, 7, [0, 1])  # 7 % 2 != 0
        with self.assertRaises(RuntimeError):
            _layouts(world, 4, [0])  # two compute ranks on one GPU without ranks_per_gpu
        with self.assertRaises(RuntimeError):
            _layouts(world, 4, [0, 1], gpus_per_rank=2, ranks_per_gpu=2)
        with self.assertRaises(RuntimeError):
            _layouts(world, 4, [0, 1], gpus_per_rank=0)

    def test_single_rank_cpu(self):
        lay = FakeWorld(1).run(lambda r, c: build_layout(c, 4, None, legacy=False))[0]
        self.assertTrue(lay.is_single())
        self.assertIsNone(lay.local_gpus(0))
        self.assertEqual(lay.block_of(0), (0, 4))
        self.assertEqual(lay.role_of(0), RankRole.HEAD)
        self.assertEqual(lay.saver_rank, 0)

    def test_legacy_layout_keeps_todays_roles(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 5, [0, 1], legacy=True))[0]
        self.assertTrue(lay.legacy)
        self.assertEqual(lay.compute_ranks, (0,))
        self.assertEqual(lay.block, 5)
        self.assertEqual(lay.placements[0].devices, (0, 1))
        self.assertEqual(lay.role_of(1), RankRole.SPARE)
        self.assertEqual(lay.role_of(2), RankRole.SAVER)


class SeedTest(unittest.TestCase):
    def test_distinct_and_deterministic(self):
        lay = FakeWorld(5, nodes=[0, 1, 0, 1, 0]).run(
            lambda r, c: build_layout(c, 8, [0, 1], legacy=False)
        )[0]
        seeds = [derive_rank_seed(103209, lay, r) for r in lay.compute_ranks]
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(seeds, [derive_rank_seed(103209, lay, r) for r in lay.compute_ranks])
        self.assertNotEqual(seeds, [derive_rank_seed(103210, lay, r) for r in lay.compute_ranks])


if __name__ == "__main__":
    unittest.main()
