"""Ensemble partition for ONE in-model repeat of the add/remove move.

2026-09-16 (job 508, [SOBBH_LL_TIMING]): the per-leaf scoring window showed
52 ``compute_like`` calls for 25 repeats -- TWO calls per repeat, each at
HALF the ensemble (~54 rows of the 120 = 12 temps x 10 walkers). That is
the red/blue split loop in ``propose`` (``for split in range(self.nsplits)``,
``nsplits=2`` inherited from eryn's ``RedBlueMove`` via ``StretchMove``),
not any re-score: ``prev_logl`` is computed ONCE before the repeat loop and
updated in place on acceptance.

The complementary-ensemble halves are load-bearing ONLY for
:class:`~eryn.moves.StretchMove`, whose proposal reads the OTHER half's
walkers. The production default inner move is
:class:`~eryn.moves.EigenAxisMove` (an ``MHMove``): it proposes from each
row's own coords and a frozen axis table, so the split buys it nothing and
costs one extra half-batch scoring call per repeat. These tests pin the
partition contract -- stretch keeps its halves, MH-style gets ONE
full-ensemble block, i.e. ONE scoring call per repeat.
"""

from __future__ import annotations

import unittest

import numpy as np
from eryn.moves import EigenAxisMove, MHMove, StretchMove

from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove


def _bare_move(ntemps=4, nwalkers=6, nsplits=2, randomize_split=False):
    """Minimal instance: only what the partition helper reads."""
    inst = object.__new__(ResidualAddOneRemoveOneMove)
    inst.ntemps = ntemps
    inst.nwalkers = nwalkers
    inst.nsplits = nsplits
    inst.randomize_split = randomize_split
    return inst


class StretchKeepsRedBlueTest(unittest.TestCase):
    """Stretch is only valid with complementary halves -- do not touch it."""

    def test_stretch_gets_nsplits_blocks(self):
        inst = _bare_move()
        masks = inst._repeat_split_masks(StretchMove())
        self.assertEqual(len(masks), inst.nsplits)

    def test_stretch_blocks_are_disjoint_and_exhaustive(self):
        inst = _bare_move(ntemps=3, nwalkers=8)
        masks = inst._repeat_split_masks(StretchMove())
        total = np.zeros((3, 8), dtype=int)
        for m in masks:
            self.assertEqual(m.shape, (3, 8))
            self.assertEqual(m.dtype, np.dtype(bool))
            total += m.astype(int)
        np.testing.assert_array_equal(total, np.ones((3, 8), dtype=int))

    def test_stretch_blocks_are_equal_sized_per_temperature(self):
        inst = _bare_move(ntemps=3, nwalkers=8, nsplits=2)
        for m in inst._repeat_split_masks(StretchMove()):
            np.testing.assert_array_equal(m.sum(axis=1), np.full(3, 4))

    def test_stretch_honors_randomize_split(self):
        """The shuffle still runs (get_split_inds is the only source)."""
        inst = _bare_move(ntemps=2, nwalkers=8, randomize_split=True)
        np.random.seed(0)
        masks = inst._repeat_split_masks(StretchMove())
        self.assertEqual(len(masks), 2)
        np.testing.assert_array_equal(
            (masks[0] | masks[1]), np.ones((2, 8), dtype=bool)
        )


class MHGetsOneFullBlockTest(unittest.TestCase):
    """An MH-style inner move scores the WHOLE ensemble in one call."""

    def test_eigen_axis_move_gets_exactly_one_block(self):
        inst = _bare_move(ntemps=4, nwalkers=6)
        masks = inst._repeat_split_masks(EigenAxisMove())
        self.assertEqual(
            len(masks), 1,
            "len(masks) IS the number of compute_like calls per repeat",
        )

    def test_the_single_block_is_the_full_ensemble(self):
        inst = _bare_move(ntemps=4, nwalkers=6)
        (mask,) = inst._repeat_split_masks(EigenAxisMove())
        self.assertEqual(mask.shape, (4, 6))
        self.assertTrue(mask.all())

    def test_plain_mh_move_also_gets_one_block(self):
        class _PlainMH(MHMove):
            def get_proposal(self, branches_coords, random, **kwargs):
                raise NotImplementedError

        inst = _bare_move()
        self.assertEqual(len(inst._repeat_split_masks(_PlainMH())), 1)

    def test_mh_partition_ignores_nsplits(self):
        inst = _bare_move(ntemps=2, nwalkers=6, nsplits=3)
        masks = inst._repeat_split_masks(EigenAxisMove())
        self.assertEqual(len(masks), 1)
        self.assertTrue(masks[0].all())

    def test_mh_partition_does_not_shuffle(self):
        """No split -> no need to draw from the RNG at all."""
        inst = _bare_move(ntemps=2, nwalkers=6, randomize_split=True)
        called = []
        inst.get_split_inds = lambda: called.append(1)
        masks = inst._repeat_split_masks(EigenAxisMove())
        self.assertEqual(called, [])
        self.assertTrue(masks[0].all())


class EigenSplitTableAcceptsFullBlockTest(unittest.TestCase):
    """The per-(temp, walker) eigen table slices with an all-True mask."""

    def test_full_mask_slices_the_table_to_the_whole_ensemble(self):
        ntemps, nwalkers, ndim = 3, 4, 2
        inst = _bare_move(ntemps=ntemps, nwalkers=nwalkers)
        inst.branch_name = "sobbh"
        axes = np.tile(np.eye(ndim), (ntemps, nwalkers, 1, 1))
        sigmas = np.ones((ntemps, nwalkers, ndim))
        inst._eigen_tables = {0: (axes, sigmas)}
        move = EigenAxisMove()
        (mask,) = inst._repeat_split_masks(move)
        inst._install_eigen_split_table(move, 0, mask)
        ax, sg = move._tables["sobbh"]
        self.assertEqual(
            tuple(np.shape(ax)), (ntemps, nwalkers, 1, ndim, ndim)
        )
        self.assertEqual(tuple(np.shape(sg)), (ntemps, nwalkers, 1, ndim))


if __name__ == "__main__":
    unittest.main()
