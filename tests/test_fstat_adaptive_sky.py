"""The f0-adaptive stage-B sky grid (2026-09-24).

Stage A already scaled its sky grid as ``f0^2`` while stage B was pinned to a
fixed ``8x8``, so above ~6 mHz the REFINEMENT was up to 8x coarser on the sky
than the scan that produced its own input, and 20 of every 64 stage-B nodes
were duplicates of another node. These tests pin the law, the factorization,
the grouping, the per-group cache keys and the memory step-down.
"""

import os
import unittest
from unittest import mock

import numpy as np

#: These caches are written in the ``Mc`` basis, so the loader's basis check
#: has to be looking for ``Mc`` too -- ``FSTAT_FDOT_AXIS`` defaults to the
#: fdot basis and would otherwise refuse them for a reason unrelated to the
#: sky axes under test.
MC_BASIS = mock.patch.dict(os.environ, {"FSTAT_FDOT_AXIS": "0"})

from lisatools.sampling.fstat_gridfit import (
    enumerate_center_nodes,
    sky_nodes_required,
    stage_b_sky_axes,
    write_stacked_npz,
)
from lisatools.sampling.fstat_proposal import (
    StackedFStatProposal4D,
    stacked_from_cache,
)

TOBS = 1.5552e7  # 180 d, the 6-month production run


def _unit(alpha, sd):
    cd = np.sqrt(np.maximum(1.0 - sd ** 2, 0.0))
    return np.stack([cd * np.cos(alpha), cd * np.sin(alpha), sd], axis=-1)


def _distinct(alpha_ax, sd_ax):
    A, S = np.meshgrid(alpha_ax, sd_ax, indexing="ij")
    v = _unit(A.ravel(), S.ravel())
    return np.unique(np.round(v, 9), axis=0).shape[0]


class SkyLawTest(unittest.TestCase):
    def test_scales_as_f0_squared(self):
        """n ~ (f0 Tobs v/c)^2, snapped up to a power of two."""
        got = {f: int(sky_nodes_required(f, TOBS, vc=1e-4, nmin=1))
               for f in (2.0, 4.0, 8.0, 12.0, 20.0)}
        self.assertEqual(got, {2.0: 16, 4.0: 64, 8.0: 256,
                               12.0: 512, 20.0: 1024})

    def test_floor_and_cap(self):
        self.assertEqual(int(sky_nodes_required(0.5, TOBS, nmin=64)), 64)
        self.assertEqual(int(sky_nodes_required(20.0, TOBS, nmin=64,
                                                nmax=256)), 256)
        # nmax=0 means uncapped, NOT "cap at zero"
        self.assertEqual(int(sky_nodes_required(20.0, TOBS, nmin=64,
                                                nmax=0)), 1024)

    def test_monotone_in_f0(self):
        f = np.linspace(0.5, 22.0, 400)
        n = sky_nodes_required(f, TOBS, nmin=64)
        self.assertTrue(np.all(np.diff(n) >= 0),
                        "a non-monotone requirement would break the "
                        "contiguous-group invariant stage B relies on")

    def test_stage_a_default_floor_is_unchanged(self):
        """Stage A's historical 16..512 window still comes out of the law."""
        f = np.linspace(0.5, 22.0, 200)
        n = sky_nodes_required(f, TOBS, vc=1e-4, nmin=16, nmax=512)
        self.assertEqual((int(n.min()), int(n.max())), (16, 512))


class SkyAxesTest(unittest.TestCase):
    def test_no_duplicate_longitude(self):
        """alpha must be half-open: 0 and 2*pi are the same direction."""
        alpha_ax, _ = stage_b_sky_axes(64)
        self.assertNotAlmostEqual(float(alpha_ax[-1]), 2.0 * np.pi, places=6)
        self.assertAlmostEqual(float(alpha_ax[0]), 0.0)

    def test_poles_kept(self):
        _, sd_ax = stage_b_sky_axes(64)
        self.assertAlmostEqual(float(sd_ax[0]), -1.0)
        self.assertAlmostEqual(float(sd_ax[-1]), 1.0)

    def test_axes_uniformly_spaced(self):
        """StackedFStatProposal4D stores ONE dx per axis."""
        for n in (64, 128, 256, 512, 1024):
            a, s = stage_b_sky_axes(n)
            for ax in (a, s):
                d = np.diff(ax)
                self.assertTrue(np.allclose(d, d[0]), f"n={n}")

    def test_recovers_more_distinct_directions_than_the_old_grid(self):
        old_a = np.linspace(0.0, 2 * np.pi, 8)
        old_s = np.linspace(-1.0, 1.0, 8)
        self.assertEqual(_distinct(old_a, old_s), 44)
        a, s = stage_b_sky_axes(64)
        self.assertGreater(_distinct(a, s), 44)

    def test_node_count_tracks_the_budget(self):
        for n in (64, 128, 256, 512, 1024):
            a, s = stage_b_sky_axes(n)
            self.assertLessEqual(abs(len(a) * len(s) - n) / n, 0.15,
                                 f"n={n} -> {len(a)}x{len(s)}")

    def test_coverage_improves_with_the_budget(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(4000, 3))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        last = np.inf
        for n in (64, 128, 256, 512):
            a, s = stage_b_sky_axes(n)
            A, S = np.meshgrid(a, s, indexing="ij")
            v = _unit(A.ravel(), S.ravel())
            p95 = np.percentile(
                np.arccos(np.clip(z @ v.T, -1, 1).max(axis=1)), 95)
            self.assertLess(p95, last)
            last = p95


class AxisShapeGuardTest(unittest.TestCase):
    """A wrong-length axis used to be silent; it must now raise."""

    def _grid(self, n_mc=2, n_al=4, n_sd=3):
        return np.zeros((2, 3, n_mc, n_al, n_sd))

    def test_mismatched_alpha_axis_raises(self):
        with self.assertRaisesRegex(ValueError, "axes must describe the grid"):
            StackedFStatProposal4D(
                self._grid(), np.array([1.0, 2.0]), np.array([0.1, 0.1]),
                np.linspace(0.0, 1.0, 2), np.linspace(0.0, 6.0, 9),
                np.linspace(-1.0, 1.0, 3))

    def test_matching_axes_accepted(self):
        StackedFStatProposal4D(
            self._grid(), np.array([1.0, 2.0]), np.array([0.1, 0.1]),
            np.linspace(0.0, 1.0, 2), np.linspace(0.0, 6.0, 4),
            np.linspace(-1.0, 1.0, 3))


class PerGroupCacheTest(unittest.TestCase):
    """Per-group sky axes must survive the npz round trip."""

    def _write(self, tmpdir, skies):
        grids, mcs, als, sds, sizes = [], [], [], [], []
        for n_al, n_sd in skies:
            grids.append(np.zeros((2, 3, 2, n_al, n_sd)))
            mcs.append(np.linspace(0.0, 1.0, 2))
            als.append(2 * np.pi * np.arange(n_al) / n_al)
            sds.append(np.linspace(-1.0, 1.0, n_sd))
            sizes.append(2)
        k = 2 * len(skies)
        path = os.path.join(tmpdir, "fstat_grid.npz")
        write_stacked_npz(
            path, grids_g=grids, mc_ax_g=mcs,
            f0_los=np.linspace(1.0, 2.0, k), f0_dxs=np.full(k, 1e-4),
            alpha_ax=als[0], sd_ax=sds[0], alpha_ax_g=als, sd_ax_g=sds,
            grid_basis="Mc", grid_c_t=0.0,
            peaks=np.stack([np.linspace(1.0, 2.0, k), np.full(k, 50.0),
                            np.zeros(k), np.zeros(k)], axis=1),
            band_idx=np.zeros(k, dtype=int),
            band_edges_mHz=np.array([0.5, 3.0]),
            band_edges_hz=np.array([0.5e-3, 3.0e-3]),
            group_sizes=np.asarray(sizes))
        return path

    @MC_BASIS
    def test_round_trip_differing_sky_per_group(self):
        import tempfile
        skies = [(4, 3), (9, 7)]
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, skies)
            with np.load(path, allow_pickle=False) as d:
                for gi, (n_al, n_sd) in enumerate(skies):
                    self.assertEqual(len(d[f"alpha_ax_g{gi}"]), n_al)
                    self.assertEqual(len(d[f"sin_delta_ax_g{gi}"]), n_sd)
                self.assertEqual(len(d["alpha_ax"]), skies[0][0])
                prop = stacked_from_cache(d)
            for gi, (n_al, n_sd) in enumerate(skies):
                got = prop.components[gi]._node_shape[2:]
                self.assertEqual(tuple(got), (n_al, n_sd))

    def test_writer_refuses_axes_that_do_not_match_the_grid(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "never scored at"):
                write_stacked_npz(
                    os.path.join(td, "g.npz"),
                    grids_g=[np.zeros((2, 3, 2, 4, 3))],
                    mc_ax_g=[np.linspace(0, 1, 2)],
                    f0_los=np.array([1.0, 2.0]), f0_dxs=np.full(2, 1e-4),
                    alpha_ax=np.zeros(4), sd_ax=np.zeros(3),
                    alpha_ax_g=[np.zeros(9)], sd_ax_g=[np.zeros(3)],
                    grid_basis="Mc", grid_c_t=0.0,
                    peaks=np.zeros((2, 4)), band_idx=np.zeros(2, dtype=int),
                    band_edges_mHz=np.array([0.5, 3.0]),
                    band_edges_hz=np.array([0.5e-3, 3.0e-3]),
                    group_sizes=np.asarray([2]))


class UniformSkyKeepsTheHistoricalKeySetTest(unittest.TestCase):
    """Passing per-group axes that all AGREE must not change the cache layout.

    Every pinned-sky fit and every pre-adaptive configuration lands here, and
    the stage-B golden files compare key sets, so redundant suffixed copies
    of one shared axis would break them for no gain.
    """

    def test_no_suffixed_keys_when_every_group_agrees(self):
        import tempfile
        al = 2 * np.pi * np.arange(4) / 4
        sd = np.linspace(-1.0, 1.0, 3)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "g.npz")
            write_stacked_npz(
                path, grids_g=[np.zeros((2, 3, 2, 4, 3))] * 2,
                mc_ax_g=[np.linspace(0, 1, 2)] * 2,
                f0_los=np.linspace(1.0, 2.0, 4), f0_dxs=np.full(4, 1e-4),
                alpha_ax=al, sd_ax=sd,
                alpha_ax_g=[al, al], sd_ax_g=[sd, sd],
                grid_basis="Mc", grid_c_t=0.0,
                peaks=np.stack([np.linspace(1.0, 2.0, 4), np.full(4, 50.0),
                                np.zeros(4), np.zeros(4)], axis=1),
                band_idx=np.zeros(4, dtype=int),
                band_edges_mHz=np.array([0.5, 3.0]),
                band_edges_hz=np.array([0.5e-3, 3.0e-3]),
                group_sizes=np.asarray([2, 2]))
            with np.load(path, allow_pickle=False) as d:
                self.assertNotIn("alpha_ax_g0", d.files)
                self.assertNotIn("sin_delta_ax_g1", d.files)


class UnionDiffersPerSlotTest(unittest.TestCase):
    """Slots may have DIFFERENT group counts and different sky grids.

    Each walker selects its own peaks, so the Mc ladder lands differently and
    the group count differs; and a group covering different frequencies
    SHOULD carry a different sky grid. The union keeps every slot's groups as
    separate self-describing components, so neither has to agree -- an
    earlier version of this change required equal group counts and refused
    perfectly good input.
    """

    def _slot(self, path, skies):
        grids, mcs, als, sds, sizes = [], [], [], [], []
        for n_al, n_sd in skies:
            grids.append(np.zeros((2, 3, 2, n_al, n_sd)))
            mcs.append(np.linspace(0.0, 1.0, 2))
            als.append(2 * np.pi * np.arange(n_al) / n_al)
            sds.append(np.linspace(-1.0, 1.0, n_sd))
            sizes.append(2)
        k = 2 * len(skies)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_stacked_npz(
            path, grids_g=grids, mc_ax_g=mcs,
            f0_los=np.linspace(1.0, 2.0, k), f0_dxs=np.full(k, 1e-4),
            alpha_ax=als[0], sd_ax=sds[0], alpha_ax_g=als, sd_ax_g=sds,
            grid_basis="Mc", grid_c_t=0.0,
            peaks=np.stack([np.linspace(1.0, 2.0, k), np.full(k, 50.0),
                            np.zeros(k), np.zeros(k)], axis=1),
            band_idx=np.zeros(k, dtype=int),
            band_edges_mHz=np.array([0.5, 3.0]),
            band_edges_hz=np.array([0.5e-3, 3.0e-3]),
            group_sizes=np.asarray(sizes))

    @MC_BASIS
    def test_union_of_unequal_slots(self):
        import tempfile
        from lisatools.sampling.fstat_gridfit import union_stacked_npz
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "w00", "g.npz")
            b = os.path.join(td, "w01", "g.npz")
            self._slot(a, [(4, 3)])                 # 1 group
            self._slot(b, [(4, 3), (9, 7)])         # 2 groups, coarser+finer
            out = os.path.join(td, "u.npz")
            total, ngroups = union_stacked_npz([a, b], out)
            self.assertEqual((total, ngroups), (6, 3))
            with np.load(out, allow_pickle=False) as d:
                # every output group is self-describing
                self.assertEqual(len(d["alpha_ax_g0"]), 4)
                self.assertEqual(len(d["alpha_ax_g1"]), 4)
                self.assertEqual(len(d["alpha_ax_g2"]), 9)
                prop = stacked_from_cache(d)
            got = [tuple(c._node_shape[2:]) for c in prop.components]
            self.assertEqual(got, [(4, 3), (4, 3), (9, 7)])


class EndToEndStageBTest(unittest.TestCase):
    """Drive the real ``run_stacked_stage_b`` and read the grids it produced.

    Reuses the parallel-fit suite's fixture band grid and scorer so this
    exercises the production code path, not a re-implementation of it.
    """

    def _run(self, tmpdir, **env):
        from tests.test_fstat_parallel_fit import (
            BAND_EDGES, TOBS as T90, _fake_call_fstat, make_peaks,
            stage_b_env,
        )
        from lisatools.sampling.fstat_gridfit import run_stacked_stage_b
        # Unpin the sky: the golden harness pins it to 2x2 on purpose.
        env.setdefault("FSTAT_N_ALPHA", "")
        env.setdefault("FSTAT_N_SINDELTA", "")
        with stage_b_env(**env):
            prop = run_stacked_stage_b(
                _fake_call_fstat(), make_peaks(), xp=np, Tobs=T90,
                band_edges_hz=BAND_EDGES, mc_lims=[0.01, 1.0],
                cache_path=os.path.join(tmpdir, "fstat_grid.npz"),
                fingerprint_extra="|epoch=0", epoch=0)
        return prop

    def _skies(self, prop):
        comps = getattr(prop, "components", [prop])
        return [tuple(c._node_shape[2:]) for c in comps]

    def test_sky_grows_with_frequency(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            skies = self._skies(self._run(td))
        counts = [a * s for a, s in skies]
        self.assertGreater(len(set(counts)), 1,
                           f"sky never adapted across 6-18 mHz: {skies}")
        self.assertEqual(counts, sorted(counts),
                         f"sky must be non-decreasing in f0: {skies}")
        self.assertGreaterEqual(min(counts), 56,
                                f"floor 64 was breached: {skies}")

    def test_pinning_either_axis_restores_a_single_sky_grid(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            skies = self._skies(self._run(td, FSTAT_N_ALPHA="5"))
        self.assertEqual(len(set(skies)), 1, f"pin ignored: {skies}")
        self.assertEqual(skies[0][0], 5)

    def test_sky_adapt_off_restores_the_historical_8x8(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            skies = self._skies(self._run(td, FSTAT_STAGEB_SKY_ADAPT="0"))
        self.assertEqual(set(skies), {(8, 8)}, f"{skies}")

    def test_group_byte_budget_steps_the_sky_back_down(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            big = self._skies(self._run(td, FSTAT_STAGEB_GROUP_MAX_GB="0"))
        with tempfile.TemporaryDirectory() as td:
            small = self._skies(
                self._run(td, FSTAT_STAGEB_GROUP_MAX_GB="1e-6"))
        self.assertGreater(max(a * s for a, s in big),
                           max(a * s for a, s in small),
                           "a tiny per-group budget must reduce the sky")
        self.assertGreaterEqual(min(a * s for a, s in small), 56,
                                "the step-down must stop at the floor, not "
                                "run to zero")

    def test_cap_is_honoured(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            skies = self._skies(self._run(td, FSTAT_STAGEB_NSKY_MAX="64"))
        self.assertLessEqual(max(a * s for a, s in skies), 72, f"{skies}")


class LegacyCacheTest(unittest.TestCase):
    """A cache written before the change has no suffixed keys and must load."""

    @MC_BASIS
    def test_flat_keys_still_load(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "fstat_grid.npz")
            grids = [np.zeros((2, 3, 2, 4, 3)), np.zeros((2, 3, 2, 4, 3))]
            write_stacked_npz(
                path, grids_g=grids,
                mc_ax_g=[np.linspace(0, 1, 2)] * 2,
                f0_los=np.linspace(1.0, 2.0, 4), f0_dxs=np.full(4, 1e-4),
                alpha_ax=2 * np.pi * np.arange(4) / 4,
                sd_ax=np.linspace(-1.0, 1.0, 3),
                grid_basis="Mc", grid_c_t=0.0,
                peaks=np.stack([np.linspace(1.0, 2.0, 4), np.full(4, 50.0),
                                np.zeros(4), np.zeros(4)], axis=1),
                band_idx=np.zeros(4, dtype=int),
                band_edges_mHz=np.array([0.5, 3.0]),
                band_edges_hz=np.array([0.5e-3, 3.0e-3]),
                group_sizes=np.asarray([2, 2]))
            with np.load(path, allow_pickle=False) as d:
                self.assertNotIn("alpha_ax_g0", d.files)
                prop = stacked_from_cache(d)
            self.assertEqual(len(prop.components), 2)


if __name__ == "__main__":
    unittest.main()


class CenterNodesPerGroupSkyTest(unittest.TestCase):
    """REGRESSION: the center-table reader used the FLAT sky axes.

    Job 621 died here, past the comb, on the cheap enumeration that reads
    the already-fitted grids::

        enumerate_center_nodes -> np.unravel_index
        ValueError: index 358 is out of bounds for array with size 189

    ``write_stacked_npz`` emits ``alpha_ax_g{gi}`` / ``sin_delta_ax_g{gi}``
    only when the groups' skies DIFFER, keeping the flat pair for the
    shared case so a pinned-sky cache stays byte-identical. The reader used
    the flat pair unconditionally, which is correct only in that shared
    case.

    ⚠ The raise was the lucky half. A group whose sky is SMALLER than the
    flat axes unravels without error and silently places every center node
    at a sky angle the grid was never scored at, which is a wrong birth
    aim with no traceback -- so the test covers both orderings.
    """

    def _write(self, tmpdir, skies):
        """A stacked cache under the name enumerate_center_nodes reads."""
        grids, mcs, als, sds, sizes = [], [], [], [], []
        for n_al, n_sd in skies:
            g = np.zeros((2, 3, 2, n_al, n_sd))
            # a distinct argmax per (box, f0 node) so the unravel is real
            g[0, 0, 1, n_al - 1, n_sd - 1] = 5.0
            g[1, 2, 0, 0, 0] = 5.0
            grids.append(g)
            mcs.append(np.linspace(0.1, 1.0, 2))
            als.append(2 * np.pi * np.arange(n_al) / n_al)
            sds.append(np.linspace(-1.0, 1.0, n_sd))
            sizes.append(2)
        k = 2 * len(skies)
        path = os.path.join(tmpdir, "fstat_grid_peaks_stacked.npz")
        write_stacked_npz(
            path, grids_g=grids, mc_ax_g=mcs,
            f0_los=np.linspace(1.0, 2.0, k), f0_dxs=np.full(k, 1e-4),
            alpha_ax=als[0], sd_ax=sds[0], alpha_ax_g=als, sd_ax_g=sds,
            grid_basis="Mc", grid_c_t=0.0,
            peaks=np.stack([np.linspace(1.0, 2.0, k), np.full(k, 50.0),
                            np.zeros(k), np.zeros(k)], axis=1),
            band_idx=np.zeros(k, dtype=int),
            band_edges_mHz=np.array([0.5, 3.0]),
            band_edges_hz=np.array([0.5e-3, 3.0e-3]),
            group_sizes=np.asarray(sizes))
        return path

    def _run(self, skies):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self._write(td, skies)
            return enumerate_center_nodes(td, mc_lims=[0.001, 1.0])

    @MC_BASIS
    def test_group_sky_LARGER_than_the_flat_axes(self):
        """The shape job 621 hit: group 1's sky overflows the flat axes."""
        nodes = self._run([(4, 3), (9, 7)])
        self.assertEqual(len(nodes["f0_mHz"]), len(nodes["alpha"]))
        self.assertTrue(np.all(np.isfinite(nodes["alpha"])))

    @MC_BASIS
    def test_group_sky_SMALLER_than_the_flat_axes(self):
        """The SILENT half, and the reason this needs exact values.

        With group 1's sky (4x3) smaller than the flat axes (9x7) the bad
        unravel does NOT overflow -- it returns in-range indices into the
        wrong axis, so the only evidence is the ANGLE itself. Checking
        membership in the union of both groups' axes is not enough: the
        planted argmax at alpha index 3 maps to 2*pi*3/4 correctly and
        2*pi*3/9 incorrectly, and BOTH are members of that union. So the
        expectation is computed from the planted positions.
        """
        nodes = self._run([(9, 7), (4, 3)])
        al = np.asarray(nodes["alpha"])
        # planted: g[0, 0, 1, n_al-1, n_sd-1] and g[1, 2, 0, 0, 0], so each
        # group contributes its own last-alpha and alpha=0.
        want = set(np.round([0.0,
                             2 * np.pi * 8 / 9,      # group 0, n_al=9
                             2 * np.pi * 3 / 4], 9)) # group 1, n_al=4
        got = set(np.round(np.unique(al), 9))
        self.assertTrue(
            got <= want,
            f"emitted alphas {sorted(got)} are not the planted maxima "
            f"{sorted(want)} -- the sky axis used was the wrong group's")
        # the wrong-shape answer for group 1 would be 2*pi*3/9; assert it
        # is absent, which is the whole point of the test.
        self.assertNotIn(round(2 * np.pi * 3 / 9, 9), got)

    @MC_BASIS
    def test_identical_skies_still_work(self):
        """The shared case writes no suffixed keys at all -- the fallback
        to the flat axes has to keep working."""
        nodes = self._run([(5, 4), (5, 4)])
        self.assertGreater(len(nodes["f0_mHz"]), 0)
