"""Seeding the RJ birth container: ``reseed`` / ``reseed_birth_tree``.

Every proposal class in ``fstat_proposal`` builds its own
``np.random.default_rng(seed)`` and the RJ birth assembly
(``fstat_gridfit.build_gb_birth_distribution``) constructs all of them with
``seed=None`` -- OS entropy. ``reseed(seed)`` re-derives the WHOLE tree from
one integer after construction, which is what makes RJ births reproducible
per rank and per F-stat epoch (``GBSpecialRJFStatGridMove._birth_seed``).

The invariant that matters in production is the negative one: with
``general.random_seed`` unset the seed is ``None`` and NOTHING may become
seeded -- pinned by ``test_reseed_none_touches_nothing``.

The extrinsic columns of the container (slot 0 / phi0 / cos_iota / psi, and
the ratio column in the Mc basis) come from eryn's ``UniformDistribution``,
which draws from the MODULE-level ``np.random`` state rather than a private
generator; that stream is seeded per rank by ``run.py::_seed_rank_streams``.
The draws below therefore pin ``np.random.seed`` before every ``rvs`` so the
comparison isolates the private generators this change owns.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from lisatools.sampling.fstat_proposal import (
    CombIntrinsicProposal,
    FdotAxisBirth,
    MixtureProposal,
    RatioTightenedBirth,
    StackedFStatProposal4D,
    UniformFloorMixture,
    make_gb_rj_birth_container,
    reseed_birth_tree,
)

A_LIMS = [1e-23, 1e-20]
MC_LIMS = [0.01, 1.0]
RATIO_MAX = 0.5
TOBS = 3.0 * 24 * 3600.0
F0_LO_MHZ, F0_HI_MHZ = 7.40, 7.70
RATIO_TIGHT = dict(tobs=TOBS, phase_rad=2.0 * np.pi, eps=0.1, w_min=0.05)


def _stack(k=2, seed=3):
    """A tiny 2-box stacked grid (3 nodes per axis, smooth random logp)."""
    rng = np.random.default_rng(seed)
    node = (3, 3, 3, 3)
    grids = rng.normal(size=(k,) + node)
    f0_los = np.linspace(F0_LO_MHZ, F0_HI_MHZ - 0.05, k)
    f0_dxs = np.full(k, 0.01)
    return StackedFStatProposal4D(
        grids, f0_los, f0_dxs,
        mc_ax=np.linspace(MC_LIMS[0], MC_LIMS[1], node[1]),
        alpha_ax=np.linspace(0.0, 2.0 * np.pi, node[2]),
        sin_delta_ax=np.linspace(-1.0, 1.0, node[3]),
    )


def _comb(n=5):
    return CombIntrinsicProposal(
        np.linspace(F0_LO_MHZ, F0_HI_MHZ, n),
        np.linspace(10.0, 40.0, n),
        mc_lims=MC_LIMS,
    )


def _tree():
    """stack + comb -> mixture -> uniform floor -> 9-column birth container."""
    mix = MixtureProposal([_stack(), _comb()], [0.7, 0.3])
    floor = UniformFloorMixture(
        mix, [F0_LO_MHZ, MC_LIMS[0], 0.0, -1.0],
        [F0_HI_MHZ, MC_LIMS[1], 2.0 * np.pi, 1.0], eps=0.1)
    return make_gb_rj_birth_container(
        floor, A_LIMS, fdot_astro_ratio_max=RATIO_MAX,
        ratio_tight=RATIO_TIGHT, tobs=TOBS, mc_lims=MC_LIMS)


def _rngs(obj, out=None, seen=None):
    """Every ``_rng`` object reachable in a birth tree, in walk order."""
    if out is None:
        out, seen = [], set()
    if obj is None or id(obj) in seen:
        return out
    seen.add(id(obj))
    rng = getattr(obj, "_rng", None)
    if rng is not None:
        out.append(rng)
    for name in ("base", "grid4"):
        _rngs(getattr(obj, name, None), out, seen)
    for child in getattr(obj, "components", None) or []:
        _rngs(child, out, seen)
    priors_in = getattr(obj, "priors_in", None)
    if isinstance(priors_in, dict):
        for child in priors_in.values():
            _rngs(child, out, seen)
    return out


def _draw(dist, n=64):
    """``rvs`` with the MODULE-level stream pinned (see the module docstring)."""
    np.random.seed(0)
    return np.asarray(dist.rvs(n))


class ReseedTreeTest(unittest.TestCase):
    """The Mc-basis tree (``FSTAT_FDOT_AXIS=0``): RatioTightenedBirth outside."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"FSTAT_FDOT_AXIS": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_wrapper_is_the_ratio_tightened_one(self):
        self.assertIsInstance(_tree(), RatioTightenedBirth)

    def test_the_same_seed_reproduces_the_draws(self):
        a, b = _tree(), _tree()
        reseed_birth_tree(a, 123)
        reseed_birth_tree(b, 123)
        self.assertTrue(np.array_equal(_draw(a), _draw(b)))

    def test_a_different_seed_does_not(self):
        a, b = _tree(), _tree()
        reseed_birth_tree(a, 123)
        reseed_birth_tree(b, 124)
        self.assertFalse(np.array_equal(_draw(a), _draw(b)))

    def test_reseed_none_touches_nothing(self):
        # the ``general.random_seed is None`` path: every stream stays on the
        # OS entropy it was constructed with, object for object
        tree = _tree()
        before = _rngs(tree)
        self.assertGreaterEqual(len(before), 5)
        reseed_birth_tree(tree, None)
        after = _rngs(tree)
        self.assertEqual(len(before), len(after))
        for old, new in zip(before, after):
            self.assertIs(old, new)

    def test_every_node_gets_its_own_generator(self):
        tree = _tree()
        reseed_birth_tree(tree, 5)
        rngs = _rngs(tree)
        self.assertEqual(len({id(r) for r in rngs}), len(rngs))
        # ... and the streams are independent, not copies of one another
        first = [float(r.random()) for r in rngs]
        self.assertEqual(len(set(first)), len(first))

    def test_a_reseed_is_repeatable_on_the_same_object(self):
        tree = _tree()
        reseed_birth_tree(tree, 77)
        one = _draw(tree)
        reseed_birth_tree(tree, 77)
        self.assertTrue(np.array_equal(one, _draw(tree)))

    def test_a_plain_object_raises(self):
        with self.assertRaises(TypeError) as ctx:
            reseed_birth_tree(object(), 1)
        self.assertIn("object", str(ctx.exception))

        class _NewWrapper:
            pass

        with self.assertRaisesRegex(TypeError, "_NewWrapper"):
            reseed_birth_tree(_NewWrapper(), None)  # even with no seed


class ReseedFdotAxisTreeTest(unittest.TestCase):
    """The fdot-basis tree (default): FdotAxisBirth is the intrinsic block."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"FSTAT_FDOT_AXIS": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fdot_tree(self):
        # axis 1 of the grid is fdot [Hz/s] in this basis
        rng = np.random.default_rng(3)
        node = (3, 3, 3, 3)
        stack = StackedFStatProposal4D(
            rng.normal(size=(2,) + node),
            np.linspace(F0_LO_MHZ, F0_HI_MHZ - 0.05, 2), np.full(2, 0.01),
            mc_ax=np.linspace(-1e-16, 1e-16, node[1]),
            alpha_ax=np.linspace(0.0, 2.0 * np.pi, node[2]),
            sin_delta_ax=np.linspace(-1.0, 1.0, node[3]),
        )
        floor = UniformFloorMixture(
            stack, [F0_LO_MHZ, -1e-16, 0.0, -1.0],
            [F0_HI_MHZ, 1e-16, 2.0 * np.pi, 1.0], eps=0.1)
        return make_gb_rj_birth_container(
            floor, A_LIMS, fdot_astro_ratio_max=RATIO_MAX,
            ratio_tight=RATIO_TIGHT, tobs=TOBS, mc_lims=MC_LIMS)

    def test_the_five_d_block_is_reseeded_too(self):
        tree = self._fdot_tree()
        block = [v for v in tree.priors_in.values()
                 if isinstance(v, FdotAxisBirth)]
        self.assertEqual(len(block), 1)
        a, b = self._fdot_tree(), self._fdot_tree()
        reseed_birth_tree(a, 31)
        reseed_birth_tree(b, 31)
        self.assertTrue(np.array_equal(_draw(a, 32), _draw(b, 32)))
        reseed_birth_tree(b, 32)
        self.assertFalse(np.array_equal(_draw(a, 32), _draw(b, 32)))


class BuildBirthDistributionSeedTest(unittest.TestCase):
    """``build_gb_birth_distribution(seed=...)`` seeds the whole assembly."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fstat_birth_seed_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # A real epoch dir always carries the stage-B npz the live stack was
        # written from; the floor box reads its band edges (with stacked_live
        # the grids themselves are never re-read).
        from lisatools.sampling.fstat_gridfit import GRID_BASENAME

        np.savez(
            os.path.join(self.tmp,
                         GRID_BASENAME.replace(".npz", "_peaks_stacked.npz")),
            band_edges=np.linspace(F0_LO_MHZ - 0.05, F0_HI_MHZ + 0.05, 5) * 1e-3,
        )
        np.savez(
            os.path.join(self.tmp, GRID_BASENAME.replace(".npz", "_comb.npz")),
            f0_nodes_mHz=np.linspace(F0_LO_MHZ, F0_HI_MHZ, 5),
            F_max=np.linspace(10.0, 40.0, 5),
        )
        patcher = mock.patch.dict(os.environ, {"FSTAT_FDOT_AXIS": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build(self, seed):
        from lisatools.sampling.fstat_gridfit import build_gb_birth_distribution

        return build_gb_birth_distribution(
            cache_dir=self.tmp, mc_lims=MC_LIMS, A_lims=A_LIMS,
            fdot_astro_ratio_max=RATIO_MAX, stacked_live=_stack(),
            comb_weight=0.3, ratio_tight=RATIO_TIGHT, tobs=TOBS, seed=seed,
        )

    def test_the_assembled_container_is_reproducible(self):
        a, b = self._build(7), self._build(7)
        # the comb + mixture + floor + wrapper are all in this tree
        self.assertIsInstance(a, RatioTightenedBirth)
        self.assertIsInstance(a.base.priors_in[
            ("f0", "Mc", "alpha", "sin_delta")], UniformFloorMixture)
        self.assertTrue(np.array_equal(_draw(a, 32), _draw(b, 32)))
        self.assertFalse(np.array_equal(_draw(a, 32), _draw(self._build(8), 32)))

    def test_seed_none_leaves_the_streams_alone(self):
        # no assertion of INEQUALITY (two entropy streams may collide in
        # principle): what is pinned is that the build runs and that nothing
        # was replaced by a seeded generator -- a seeded pair would be equal
        # every time, which is exactly what must NOT happen here.
        a, b = self._build(None), self._build(None)
        self.assertIsInstance(a, RatioTightenedBirth)
        self.assertEqual(len(_rngs(a)), len(_rngs(b)))
        self.assertFalse(np.array_equal(_draw(a, 32), _draw(b, 32)))


if __name__ == "__main__":
    unittest.main()
