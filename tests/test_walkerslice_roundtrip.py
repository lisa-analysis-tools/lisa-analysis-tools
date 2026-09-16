"""Walker-block slice/merge round trips for the multi-rank fan-out (WP0)."""

import unittest

import numpy as np
from eryn.state import BranchSupplemental

from lisatools.globalfit.communication.walkerslice import merge_state, slice_state
from lisatools.globalfit.state import GBState, GFState, MBHState, ModuleSubState
from tests.test_gf_substate_roundtrip import (
    BRANCH_SHAPES,
    NTEMPS as RT_NTEMPS,
    NWALKERS as RT_NWALKERS,
    make_state,
)

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
                getattr(right, name),
                np.take(src, np.arange(half, NWALKERS), axis=axis),
                err_msg=name,
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

    def test_merge_uninitialized_part_raises_but_both_bare_noops(self):
        sub = _make_sub(ModuleSubState, self.rng)
        bare_part = sub._bare_like()
        with self.assertRaises(ValueError):
            sub.merge_walkers(bare_part, 0, NWALKERS // 2)

        bare_self = ModuleSubState(None)
        bare_part2 = bare_self._bare_like()
        # both uninitialized: still a no-op, no exception
        bare_self.merge_walkers(bare_part2, 0, 1)
        self.assertFalse(bare_self.tempered_initialized)


class GFStateSliceMergeTest(unittest.TestCase):
    """Whole-state slice/merge over the roundtrip fixture (4 walkers -> two blocks of 2)."""

    def setUp(self):
        self.rng = np.random.default_rng(99)
        self.state = make_state(self.rng)
        nt, nw = RT_NTEMPS, RT_NWALKERS
        self.state.supplemental = BranchSupplemental(
            {
                "walker_inds": np.tile(np.arange(nw), (nt, 1)),
                "aux": self.rng.standard_normal((nt, nw, 2)),
            },
            base_shape=(nt, nw),
        )
        self.state.branches["psd"].branch_supplemental = BranchSupplemental(
            {"tag": self.rng.integers(0, 9, (nt, nw, 1))}, base_shape=(nt, nw, 1)
        )
        for sub in self.state.sub_states.values():
            for name in ("d_h", "h_h"):
                getattr(sub, name)[...] = self.rng.standard_normal(getattr(sub, name).shape)
            for name in sub.delta_counter_names:
                arr = getattr(sub, name)
                arr[...] = self.rng.integers(0, 5, arr.shape)
        self.ref = GFState(self.state, copy=True)

    def test_slice_geometry_and_walker_inds_remap(self):
        part = slice_state(self.state, 2, 4)
        for name, br in part.branches.items():
            self.assertEqual(br.nwalkers, 2)
            np.testing.assert_array_equal(br.coords, self.ref.branches[name].coords[:, 2:4])
            np.testing.assert_array_equal(br.inds, self.ref.branches[name].inds[:, 2:4])
        np.testing.assert_array_equal(
            part.supplemental.holder["walker_inds"], np.tile(np.arange(2), (RT_NTEMPS, 1))
        )
        np.testing.assert_array_equal(
            part.supplemental.holder["aux"], self.ref.supplemental.holder["aux"][:, 2:4]
        )
        np.testing.assert_array_equal(
            part.branches["psd"].branch_supplemental.holder["tag"],
            self.ref.branches["psd"].branch_supplemental.holder["tag"][:, 2:4],
        )
        np.testing.assert_array_equal(part.log_like, self.ref.log_like[:, 2:4])
        np.testing.assert_array_equal(part.log_prior, self.ref.log_prior[:, 2:4])
        np.testing.assert_array_equal(part.betas, self.ref.betas)
        mbh = part.sub_states["mbh"]
        self.assertEqual(mbh.nwalkers, 2)
        self.assertEqual(mbh.log_like.shape, (BRANCH_SHAPES["mbh"][0], RT_NTEMPS, 2))
        np.testing.assert_array_equal(mbh.d_h, self.ref.sub_states["mbh"].d_h[2:4])
        self.assertEqual(part.sub_state_bases, self.state.sub_state_bases)
        # a slice is a copy: mutating it never reaches the full state
        part.branches["gb"].coords[...] = 123.0
        part.supplemental.holder["aux"][...] = 5.0
        np.testing.assert_array_equal(
            self.state.branches["gb"].coords, self.ref.branches["gb"].coords
        )
        np.testing.assert_array_equal(
            self.state.supplemental.holder["aux"], self.ref.supplemental.holder["aux"]
        )

    def test_sub_state_filter(self):
        part = slice_state(self.state, 0, 2, sub_states=["mbh"])
        self.assertIsNotNone(part.sub_states["mbh"])
        for name in ("gb", "emri", "sobbh", "psd"):
            self.assertIsNone(part.sub_states[name])
        none = slice_state(self.state, 0, 2, sub_states=[])
        self.assertTrue(all(v is None for v in none.sub_states.values()))

    def test_slice_survives_the_gfstate_copy_path(self):
        part = slice_state(self.state, 0, 2)
        twin = GFState(part, copy=True)
        np.testing.assert_array_equal(
            twin.sub_states["mbh"].coords, part.sub_states["mbh"].coords
        )
        self.assertEqual(twin.branches["gb"].nwalkers, 2)

    def test_merge_roundtrip_restores_columns_and_sums_counters(self):
        target = GFState(self.state, copy=True)
        for br in target.branches.values():
            br.coords[...] = 0.0
            br.inds[...] = False
        target.log_like[...] = 0.0
        target.log_prior[...] = 0.0
        target.supplemental.holder["aux"][...] = 0.0
        target.branches["psd"].branch_supplemental.holder["tag"][...] = -1
        for sub in target.sub_states.values():
            sub.coords[...] = 0.0
            sub.d_h[...] = 0.0
        left = slice_state(self.state, 0, 2)
        right = slice_state(self.state, 2, 4)
        for part in (left, right):
            for sub in part.sub_states.values():
                for name in sub.delta_counter_names:
                    getattr(sub, name)[...] = 1
        merge_state(target, left, 0, 2)
        merge_state(target, right, 2, 4)

        for name, br in target.branches.items():
            np.testing.assert_array_equal(br.coords, self.ref.branches[name].coords)
            np.testing.assert_array_equal(br.inds, self.ref.branches[name].inds)
        np.testing.assert_array_equal(target.log_like, self.ref.log_like)
        np.testing.assert_array_equal(target.log_prior, self.ref.log_prior)
        np.testing.assert_array_equal(
            target.supplemental.holder["aux"], self.ref.supplemental.holder["aux"]
        )
        # the head's walker_inds stay GLOBAL ids (a slice's remapped ids never come back)
        np.testing.assert_array_equal(
            target.supplemental.holder["walker_inds"], self.ref.supplemental.holder["walker_inds"]
        )
        np.testing.assert_array_equal(
            target.branches["psd"].branch_supplemental.holder["tag"],
            self.ref.branches["psd"].branch_supplemental.holder["tag"],
        )
        for name, sub in target.sub_states.items():
            ref = self.ref.sub_states[name]
            for aname in sub.walker_axes:
                if getattr(ref, aname, None) is not None:
                    np.testing.assert_array_equal(
                        getattr(sub, aname), getattr(ref, aname), err_msg=f"{name}.{aname}"
                    )
            for cname in sub.delta_counter_names:
                np.testing.assert_array_equal(
                    getattr(sub, cname), getattr(ref, cname) + 2, err_msg=f"{name}.{cname}"
                )
        # ladders untouched
        np.testing.assert_array_equal(
            target.sub_states["mbh"].betas_all, self.ref.sub_states["mbh"].betas_all
        )
        np.testing.assert_array_equal(
            target.sub_states["psd"].betas, self.ref.sub_states["psd"].betas
        )
        np.testing.assert_array_equal(
            target.sub_states["gb"].band_info["band_temps"],
            self.ref.sub_states["gb"].band_info["band_temps"],
        )

    def test_gb_band_info_is_never_sliced(self):
        part = slice_state(self.state, 0, 2)
        self.assertFalse(hasattr(part.sub_states["gb"], "_band_info"))

    def test_bad_blocks_raise(self):
        with self.assertRaises(ValueError):
            slice_state(self.state, 3, 3)
        with self.assertRaises(ValueError):
            merge_state(self.state, slice_state(self.state, 0, 2), 0, 3)

    def test_unknown_sub_state_name_raises(self):
        with self.assertRaises(ValueError) as cm:
            slice_state(self.state, 0, 2, sub_states=["mhb"])
        self.assertIn("mhb", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
