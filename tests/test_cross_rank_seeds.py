"""Per-rank RNG streams: the warm-start containers and the cupy device states.

9mo job 751 (2026-10-10): the first ``rj_warm_search`` propose logged
IDENTICAL candidate tallies on all four compute ranks (viable 367392 /
prior-gated 38531 / SNR-dropped 74077) because both warm-start containers
were seeded with the bare run seed. The container's ``rvs`` draws every
candidate from that one private Generator, so four walkers proposed the
same sequence. ``recipe._warm_start_seed`` now spawns the seed off the
PER-RANK ``GBSettings.build_seed`` with a domain tag, like the GB priors
and the F-stat birth seed already did.

The second half: ``run.py::_seed_rank_streams`` seeded cupy's RandomState
with device 0 current, so a rank whose walker lives on GPU 1 drew from a
lazily created, entropy-seeded device-1 state. It now seeds every device
the rank owns.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lisatools.globalfit import recipe
from lisatools.globalfit.communication.ranks import derive_rank_seed


class _Info:
    def __init__(self, build_seed):
        self.build_seed = build_seed


class WarmStartSeedTest(unittest.TestCase):
    def test_two_ranks_get_different_seeds(self):
        """Two compute ranks (two build seeds) -> two different container seeds,
        for the search twin and for the PE twin alike."""
        layout = SimpleNamespace(n_compute=4, fanout_rank=lambda r: r)
        base = int(np.random.SeedSequence([103209, 0xB01D]).generate_state(1, dtype=np.uint32)[0])
        seeds = [derive_rank_seed(base, layout, r) for r in range(4)]
        self.assertEqual(len(set(seeds)), 4)
        for tag in (recipe._WARM_SEED_TAG_SEARCH, recipe._WARM_SEED_TAG_PE):
            got = [recipe._warm_start_seed(_Info(s), tag) for s in seeds]
            self.assertEqual(len(set(got)), 4, got)
            self.assertTrue(all(isinstance(g, int) and g >= 0 for g in got))

    def test_same_rank_is_reproducible_and_twins_differ(self):
        a = recipe._warm_start_seed(_Info(12345), recipe._WARM_SEED_TAG_SEARCH)
        b = recipe._warm_start_seed(_Info(12345), recipe._WARM_SEED_TAG_SEARCH)
        c = recipe._warm_start_seed(_Info(12345), recipe._WARM_SEED_TAG_PE)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)              # the search and PE twins never share a stream

    def test_domain_separated_from_the_other_seeds(self):
        """The container seed is neither the bare build seed nor the run's
        temper base nor the global rank stream (all derived from the same
        integers)."""
        build = 777
        s = recipe._warm_start_seed(_Info(build), recipe._WARM_SEED_TAG_SEARCH)
        self.assertNotEqual(s, build)
        temper = int(np.random.SeedSequence([build, 0x7E4B]).generate_state(1, dtype=np.uint32)[0])
        self.assertNotEqual(s, temper)

    def test_no_run_seed_means_entropy(self):
        self.assertIsNone(recipe._warm_start_seed(_Info(None), recipe._WARM_SEED_TAG_SEARCH))
        self.assertIsNone(recipe._warm_start_seed(SimpleNamespace(), recipe._WARM_SEED_TAG_PE))

    def test_the_old_wiring_is_what_the_test_guards(self):
        """Mutation guard: handing both ranks the bare run seed (the pre-fix
        wiring) makes their candidate streams identical -- the defect."""
        rng_a = np.random.default_rng(103209)
        rng_b = np.random.default_rng(103209)
        self.assertTrue(np.array_equal(rng_a.random(8), rng_b.random(8)))
        sa = recipe._warm_start_seed(_Info(1), recipe._WARM_SEED_TAG_SEARCH)
        sb = recipe._warm_start_seed(_Info(2), recipe._WARM_SEED_TAG_SEARCH)
        self.assertFalse(np.array_equal(np.random.default_rng(sa).random(8),
                                        np.random.default_rng(sb).random(8)))


class _FakeCupy:
    """Just what ``_seed_rank_streams`` touches: a per-device RandomState."""

    def __init__(self):
        self.current = 0
        self.seeded = {}            # device -> seed
        self.random = SimpleNamespace(seed=self._seed)
        self.cuda = SimpleNamespace(Device=self._device)

    def _seed(self, s):
        self.seeded[self.current] = int(s)

    @contextmanager
    def _device(self, i):
        prev, self.current = self.current, int(i)
        try:
            yield
        finally:
            self.current = prev


class RankDeviceSeedingTest(unittest.TestCase):
    def _run(self, gpus):
        from lisatools.globalfit import run as run_mod

        fake = _FakeCupy()
        layout = SimpleNamespace(is_single=lambda: False, n_compute=4,
                                 fanout_rank=lambda r: r)
        self_ = SimpleNamespace(
            layout=layout, rank=3, _seed_base=103209,
            curr=SimpleNamespace(general_info=SimpleNamespace(gpus=gpus)),
            logger=mock.Mock(),
        )
        with mock.patch.object(run_mod, "xp", fake), \
                mock.patch.object(run_mod, "_xp_is_cupy", True), \
                mock.patch.object(np.random, "seed"):
            seed = run_mod.GlobalFit._seed_rank_streams(self_)
        return seed, fake

    def test_every_owned_device_is_seeded_with_the_rank_seed(self):
        """A rank whose walker lives on GPU 1 must seed DEVICE 1's state
        (pre-fix: only device 0, whichever was current)."""
        seed, fake = self._run(gpus=[1])
        self.assertEqual(fake.seeded, {1: seed})
        self.assertEqual(fake.current, 0)      # the context was released

    def test_two_devices_both_seeded(self):
        seed, fake = self._run(gpus=[0, 1])
        self.assertEqual(fake.seeded, {0: seed, 1: seed})

    def test_no_gpus_touches_no_device_state(self):
        _, fake = self._run(gpus=[])
        self.assertEqual(fake.seeded, {})


if __name__ == "__main__":
    unittest.main()
