"""Rank roles + walker-block layout resolved identically on every rank (FakeComm)."""

import unittest

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import (
    RankRole,
    build_layout,
    derive_rank_seed,
    prepare_rank,
    rank_tag,
    resolve_roles,
    select_rank_device,
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

    def test_legacy_ranks_on_device_excludes_saver_and_spare(self):
        # head owns both devices in legacy mode; the spare (rank 1, device 1)
        # and the saver (rank 2, device 0) must not show up as co-occupants.
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 5, [0, 1], legacy=True))[0]
        self.assertEqual(lay.ranks_on_device("node0", 0), (0,))
        self.assertEqual(lay.ranks_on_device("node0", 1), (0,))

    def test_size2_single_gpu_demotes_rank1_to_saver_with_warning(self):
        import warnings

        world = FakeWorld(2)

        def fn(rank, comm):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lay = build_layout(comm, 4, [0], legacy=False)
            return lay, [str(w.message) for w in caught]

        out = world.run(fn)
        lay, messages = out[0]
        self.assertEqual(lay.compute_ranks, (0,))
        self.assertEqual(lay.saver_rank, 1)
        self.assertTrue(lay.is_single())
        self.assertEqual(lay.role_of(1), RankRole.SAVER)
        self.assertEqual(lay.block_of(0), (0, 4))
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(len(lay.notes), 1)
        self.assertIn("RANKS_PER_GPU=2", lay.notes[0])
        self.assertIn("-n 1", lay.notes[0])
        self.assertTrue(any("dedicated saver" in m for m in messages))
        self.assertIn("dedicated saver", lay.describe())
        self.assertEqual(out[1][0].describe(), lay.describe())

    def test_size2_with_enough_gpus_keeps_two_compute_ranks(self):
        lay = FakeWorld(2).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(lay.compute_ranks, (0, 1))
        self.assertEqual(lay.notes, ())

    def test_size2_ranks_per_gpu_2_on_one_gpu_keeps_two_compute_ranks(self):
        lay = FakeWorld(2).run(
            lambda r, c: build_layout(c, 4, [0], legacy=False, ranks_per_gpu=2)
        )[0]
        self.assertEqual(lay.compute_ranks, (0, 1))
        self.assertEqual(lay.notes, ())

    def test_size3_single_gpu_still_hard_errors(self):
        with self.assertRaises(RuntimeError):
            _layouts(FakeWorld(3), 4, [0])


class SeedTest(unittest.TestCase):
    def test_distinct_and_deterministic(self):
        lay = FakeWorld(5, nodes=[0, 1, 0, 1, 0]).run(
            lambda r, c: build_layout(c, 8, [0, 1], legacy=False)
        )[0]
        seeds = [derive_rank_seed(103209, lay, r) for r in lay.compute_ranks]
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(seeds, [derive_rank_seed(103209, lay, r) for r in lay.compute_ranks])
        self.assertNotEqual(seeds, [derive_rank_seed(103210, lay, r) for r in lay.compute_ranks])


class SelectRankDeviceTest(unittest.TestCase):
    def _layout(self):
        return FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]

    def test_visible_mode_narrows_env_and_renumbers(self):
        env = {}
        gpus, mode = select_rank_device(
            self._layout(), 1, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
        )
        self.assertEqual((gpus, mode), ([0], "visible"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_visible_mode_maps_through_a_prenarrowed_env(self):
        env = {"CUDA_VISIBLE_DEVICES": "2,3"}
        gpus, mode = select_rank_device(
            self._layout(), 1, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
        )
        self.assertEqual((gpus, mode), ([0], "visible"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")

    def test_no_runtime_probe_still_counts_as_visible(self):
        env = {}
        gpus, mode = select_rank_device(
            self._layout(),
            0,
            environ=env,
            device_count_fn=lambda: None,
            set_device_fn=lambda d: None,
        )
        self.assertEqual((gpus, mode), ([0], "visible"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_fallback_setdevice_when_the_runtime_is_already_up(self):
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}
        pinned = []
        gpus, mode = select_rank_device(
            self._layout(), 1, environ=env, device_count_fn=lambda: 2, set_device_fn=pinned.append
        )
        self.assertEqual((gpus, mode), ([1], "setdevice"))
        self.assertEqual(pinned, [1])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")

    def test_cpu_and_legacy(self):
        cpu = FakeWorld(1).run(lambda r, c: build_layout(c, 4, None, legacy=False))[0]
        self.assertEqual(select_rank_device(cpu, 0, environ={}), (None, "cpu"))
        legacy = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=True))[0]
        env = {}
        self.assertEqual(select_rank_device(legacy, 0, environ=env), ([0, 1], "legacy"))
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_pool_index_outside_visible_set_raises(self):
        env = {"CUDA_VISIBLE_DEVICES": "5"}
        with self.assertRaises(ValueError) as cm:
            select_rank_device(
                self._layout(),
                1,
                environ=env,
                device_count_fn=lambda: 1,
                set_device_fn=lambda d: None,
            )
        self.assertIn("gpu-bind", str(cm.exception))

    def test_empty_visible_env_raises_actionable_error(self):
        env = {"CUDA_VISIBLE_DEVICES": ""}
        with self.assertRaises(ValueError) as cm:
            select_rank_device(
                self._layout(),
                1,  # rank 1 has devices in this layout
                environ=env,
                device_count_fn=lambda: 1,
                set_device_fn=lambda d: None,
            )
        msg = str(cm.exception)
        self.assertIn("CUDA_VISIBLE_DEVICES", msg)
        self.assertIn("gpu-bind", msg)


class _General:
    def __init__(self):
        self.nwalkers = 4
        self.gpus = [0, 1]
        self.gpus_per_rank = 1
        self.ranks_per_gpu = 1


class _Fit:
    main_rank = 0
    built = False

    def __init__(self):
        self.general = _General()


class PrepareRankTest(unittest.TestCase):
    def test_sets_local_gpus_layout_and_mode_on_every_rank(self):
        def fn(rank, comm):
            fit = _Fit()
            env = {}
            lay = prepare_rank(
                fit, comm, environ=env, device_count_fn=lambda: 1, set_device_fn=lambda d: None
            )
            again = prepare_rank(fit, comm, environ=env)  # idempotent: no second layout
            return (
                fit.general.gpus,
                fit.rank_device_mode,
                env.get("CUDA_VISIBLE_DEVICES"),
                lay.block_of(rank),
                fit.rank_layout is lay and again is lay,
                rank_tag(lay, rank),
            )

        out = FakeWorld(3).run(fn)
        self.assertEqual(out[0], ([0], "visible", "0", (0, 2), True, "r0/head"))
        self.assertEqual(out[1], ([0], "visible", "1", (2, 4), True, "r1/c1"))
        self.assertEqual(out[2][:3], ([0], "visible", "0"))
        self.assertEqual(out[2][5], "r2/saver")

    def test_refuses_to_run_after_build_with_several_ranks(self):
        def fn(rank, comm):
            fit = _Fit()
            fit.built = True
            return prepare_rank(fit, comm, environ={}, device_count_fn=lambda: 1)

        with self.assertRaises(RuntimeError):
            FakeWorld(2).run(fn)

    def test_cpu_fit_keeps_gpus_none(self):
        def fn(rank, comm):
            fit = _Fit()
            fit.general.gpus = None
            prepare_rank(fit, comm, environ={})
            return fit.general.gpus, fit.rank_device_mode

        self.assertEqual(FakeWorld(2).run(fn)[1], (None, "cpu"))


class AutoGpusPerRankTest(unittest.TestCase):
    """gpus_per_rank=None: a lone compute rank on a node owns the whole pool (today's -n 1)."""

    def test_single_rank_owns_the_whole_pool(self):
        lay = FakeWorld(1).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(lay.local_gpus(0), [0, 1])
        self.assertEqual(lay.n_compute, 1)

    def test_single_rank_explicit_one_narrows_to_the_first_device(self):
        lay = FakeWorld(1).run(
            lambda r, c: build_layout(c, 4, [0, 1], legacy=False, gpus_per_rank=1)
        )[0]
        self.assertEqual(lay.local_gpus(0), [0])

    def test_two_compute_ranks_on_a_two_gpu_node_get_one_device_each(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual((lay.local_gpus(0), lay.local_gpus(1)), ([0], [1]))

    def test_size2_on_one_gpu_still_demotes_rank1_and_head_owns_the_pool(self):
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lay = FakeWorld(2).run(lambda r, c: build_layout(c, 4, [0], legacy=False))[0]
        self.assertEqual(lay.role_of(1), RankRole.SAVER)
        self.assertEqual(lay.local_gpus(0), [0])

    def test_head_plus_saver_on_two_gpus_head_owns_both(self):
        # -n 2 on a 2-GPU pool: rank 1 is a COMPUTE rank (two compute ranks), not a saver
        lay = FakeWorld(2).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(lay.n_compute, 2)
        self.assertEqual((lay.local_gpus(0), lay.local_gpus(1)), ([0], [1]))

    def test_prepare_rank_passes_none_through(self):
        # the fixture's general block leaves gpus_per_rank unset -> AUTO
        fit = _Fit()
        fit.general.gpus_per_rank = None
        lay = FakeWorld(1).run(
            lambda r, c: prepare_rank(fit, c, environ={}, device_count_fn=lambda: 2)
        )[0]
        self.assertEqual(lay.local_gpus(0), [0, 1])
        self.assertEqual(fit.general.gpus, [0, 1])


if __name__ == "__main__":
    unittest.main()
