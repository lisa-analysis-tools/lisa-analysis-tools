"""Walker-block slice/merge round trips for the multi-rank fan-out (WP0)."""

import unittest

import numpy as np
from eryn.state import BranchSupplemental

from lisatools.globalfit.state import GBState, GFState, MBHState, ModuleSubState

NTEMPS, NWALKERS, NLEAVES, NDIM = 3, 6, 2, 4


def _fill(sub, rng):
    """Random-fill every tempered array so a lost or misplaced column is detectable."""
    for name in sub.tempered_array_names:
        arr = getattr(sub, name, None)
        if arr is None:
            continue
        if arr.dtype == bool:
            arr[...] = rng.random(arr.shape) > 0.3
        elif np.issubdtype(arr.dtype, np.integer):
            arr[...] = rng.integers(0, 50, size=arr.shape)
        else:
            arr[...] = rng.standard_normal(arr.shape)


def _make_sub(cls, rng, **kwargs):
    sub = cls(None, **kwargs)
    sub.initialize_tempered(NTEMPS, NWALKERS, NLEAVES, NDIM)
    _fill(sub, rng)
    return sub


class SubStateSliceMergeTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def _check_roundtrip(self, sub):
        half = NWALKERS // 2
        left = sub.slice_walkers(0, half)
        right = sub.slice_walkers(half, NWALKERS)

        # slices: right geometry, zeroed delta counters, walker columns copied
        self.assertEqual(left.nwalkers, half)
        self.assertEqual(right.nwalkers, NWALKERS - half)
        for name in sub.delta_counter_names:
            self.assertTrue(np.all(getattr(left, name) == 0), name)
        for name, axis in sub.walker_axes.items():
            src = getattr(sub, name, None)
            if src is None:
                continue
            np.testing.assert_array_equal(
                getattr(left, name), np.take(src, np.arange(0, half), axis=axis), err_msg=name
            )
            np.testing.assert_array_equal(
                getattr(right, name), np.take(src, np.arange(half, NWALKERS), axis=axis), err_msg=name
            )

        # pretend each body counted something, then merge into a zeroed twin
        for name in sub.delta_counter_names:
            getattr(left, name)[...] = 1
            getattr(right, name)[...] = 2
        twin = sub._bare_like()
        twin.initialize_tempered(NTEMPS, NWALKERS, NLEAVES, NDIM)
        for name in sub.delta_counter_names:
            getattr(twin, name)[...] = getattr(sub, name)
        twin.merge_walkers(left, 0, half)
        twin.merge_walkers(right, half, NWALKERS)
        for name, axis in sub.walker_axes.items():
            src = getattr(sub, name, None)
            if src is None:
                continue
            np.testing.assert_array_equal(getattr(twin, name), src, err_msg=name)
        for name in sub.delta_counter_names:
            np.testing.assert_array_equal(getattr(twin, name), getattr(sub, name) + 3, err_msg=name)
        return left, right, twin

    def test_base_substate_flat_ladder_by_value(self):
        sub = _make_sub(ModuleSubState, self.rng)
        sub.betas = np.linspace(1.0, 0.1, NTEMPS)
        left, right, twin = self._check_roundtrip(sub)
        np.testing.assert_array_equal(left.betas, sub.betas)
        self.assertIsNot(left.betas, sub.betas)
        # merge never writes a ladder back
        left.betas[:] = -1.0
        twin.merge_walkers(left, 0, NWALKERS // 2)
        self.assertFalse(np.any(twin.betas == -1.0))

    def test_per_leaf_ladder_walker_axis_last(self):
        betas_all = np.tile(np.linspace(1.0, 0.05, NTEMPS), (NLEAVES, 1))
        sub = _make_sub(MBHState, self.rng, betas_all=betas_all)
        self.assertEqual(sub.log_like.shape, (NLEAVES, NTEMPS, NWALKERS))
        left, right, twin = self._check_roundtrip(sub)
        self.assertEqual(left.log_like.shape, (NLEAVES, NTEMPS, NWALKERS // 2))
        np.testing.assert_array_equal(left.betas_all, betas_all)
        self.assertIsNot(left.betas_all, sub.betas_all)
        self.assertEqual(left.num_mbhs, NLEAVES)

    def test_gb_substate_never_touches_band_info(self):
        sub = _make_sub(GBState, self.rng)
        left, right, twin = self._check_roundtrip(sub)
        self.assertFalse(hasattr(left, "_band_info"))

    def test_bad_blocks_raise(self):
        sub = _make_sub(ModuleSubState, self.rng)
        with self.assertRaises(ValueError):
            sub.slice_walkers(4, 2)
        with self.assertRaises(ValueError):
            sub.slice_walkers(0, NWALKERS + 1)
        with self.assertRaises(ValueError):
            sub.merge_walkers(sub.slice_walkers(0, 2), 0, 3)

    def test_uninitialized_substate_slices_to_bare(self):
        sub = ModuleSubState(None)
        part = sub.slice_walkers(0, 1)
        self.assertFalse(part.tempered_initialized)


if __name__ == "__main__":
    unittest.main()
