"""VGB eigen in-model proposal: generic (no-fiber) axes on GB infrastructure.

The VGB move inherits the whole GB info-matrix machinery and historically
ran pure stretch. These tests pin the eigen enablement:

* ``GBSpecialBase._eigen_axes_from_info`` — the axes/width table builder
  factored out of ``_compute_proposal_cholesky``: the GB branch (fiber +
  analytic ridge) is bit-pinned elsewhere; here the GENERIC branch (no
  fiber columns, opt-in via ``eigen_axis_generic_ok``) must produce plain
  whitened-eigen axes with prior-bounded widths, and stay None when not
  opted in.
* ``_vgb_inmodel_defaults`` — the ``VGB_INMODEL_PROPOSAL`` env knob
  (the default arms the info-matrix proposal; ``stretch`` is the escape).
  The default kind itself moved from ``eigen`` to ``observable`` later on
  2026-09-15 — see ``tests/test_vgb_observable_basis.py``, which owns the
  observable composite; this file keeps pinning the ``eigen`` escape it
  was written for.
* The VGB draw branch — one-axis symmetric steps off the (axes * sigma)
  table, graceful fallback to stretch when no table could be built.
* ``PerLeafFillFactorTest`` — the info-matrix path through a transform
  container that carries PER-LEAF fills, which is the shape the real vgb
  container has and the shape the stubs above do NOT. See that class's
  docstring for the production failure it reproduces.
"""

import os
import types
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import gbbands as _gbbands
from lisatools.globalfit.moves.gbspecialstretch import (
    GBSpecialBase,
    VGBSpecialStretchMove,
    _vgb_inmodel_defaults,
)

NDIM = 5


def _spd(rng, n, ndim):
    a = rng.standard_normal((n, ndim, ndim))
    return a @ np.swapaxes(a, -1, -2) + 0.5 * np.eye(ndim)


def _stub(**over):
    s = types.SimpleNamespace(
        xp=np,
        name="vgb",
        branch_name="vgb",
        use_gpu=False,
        _eigen_axis_min_dim=9,
        _last_axis_sigmas=None,
        _last_axis_pick=None,
        _last_im_kind=None,
        jump_factor=1.0,
        use_info_mat_proposal=True,
        stretch_probability=0.0,
        eigen_axis_generic_ok=True,
        _proposal_param_scales=np.ones(NDIM),
    )
    s._eigen_axis_ready = lambda: False
    widths = np.array([2.0, 1.0, 4.0, 0.5, 3.0])
    s._eigen_axis_widths = lambda ndim: widths[:ndim]
    for k, v in over.items():
        setattr(s, k, v)
    return s


class GenericAxesFromInfoTest(unittest.TestCase):
    def test_generic_branch_builds_prior_bounded_eigen_table(self):
        from eryn.moves.eigenaxis import axis_prior_bounds

        rng = np.random.default_rng(83)
        info_y = _spd(rng, 4, NDIM)
        coords = rng.standard_normal((4, NDIM))
        s = _stub()

        out = GBSpecialBase._eigen_axes_from_info(s, info_y, coords, NDIM)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (4, NDIM, NDIM))

        evals, evecs = np.linalg.eigh(info_y)
        widths = s._eigen_axis_widths(NDIM)
        sig = np.minimum(
            1.0 / np.sqrt(evals), axis_prior_bounds(evecs, widths)
        )
        np.testing.assert_allclose(
            np.abs(out), np.abs(evecs * sig[:, None, :]), atol=1e-12
        )
        np.testing.assert_allclose(s._last_axis_sigmas, sig, atol=1e-12)

    def test_not_opted_in_returns_none(self):
        rng = np.random.default_rng(89)
        s = _stub(eigen_axis_generic_ok=False)
        out = GBSpecialBase._eigen_axes_from_info(
            s, _spd(rng, 2, NDIM), rng.standard_normal((2, NDIM)), NDIM
        )
        self.assertIsNone(out)

    def test_vgb_class_opts_in(self):
        self.assertTrue(VGBSpecialStretchMove.eigen_axis_generic_ok)
        self.assertTrue(VGBSpecialStretchMove.infomat_per_block)
        # plain GB moves keep the historical narrow-basis fallback-to-joint
        self.assertFalse(getattr(GBSpecialBase, "eigen_axis_generic_ok"))


class VGBInmodelDefaultsTest(unittest.TestCase):
    def test_default_arms_the_info_matrix_machinery(self):
        # USER RULING 2026-09-15, twice. First "yes make that the default.
        # the eigen proposal"; then, later the same day, "we should be
        # sampling the VGBs just like the GBs now ... still sampling in the
        # observed basis for VGBs (just without f0, sky coords)" -- so the
        # default is now ``observable``. Either way the unset-env default
        # arms the per-block information matrix (the observable composite
        # needs it for the extrinsic widths and for the eigen table in z),
        # which is what this test pins; which DRAW it feeds is pinned in
        # tests/test_vgb_observable_basis.py. The former stretch default
        # was a live-campaign guard for runs whose scripts predate the
        # knob; those runs pin VGB_INMODEL_PROPOSAL=stretch explicitly if
        # they must not change.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VGB_INMODEL_PROPOSAL", None)
            kw = {}
            _vgb_inmodel_defaults(kw)
        self.assertTrue(kw["use_info_mat_proposal"])
        self.assertEqual(kw["stretch_probability"], 0.0)

    def test_env_stretch_escape(self):
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "stretch"}
        ):
            kw = {}
            _vgb_inmodel_defaults(kw)
        self.assertFalse(kw["use_info_mat_proposal"])
        self.assertEqual(kw["stretch_probability"], 1.0)

    def test_explicit_kwargs_win(self):
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "eigen"}
        ):
            kw = {"use_info_mat_proposal": False,
                  "stretch_probability": 0.25}
            _vgb_inmodel_defaults(kw)
        self.assertFalse(kw["use_info_mat_proposal"])
        self.assertEqual(kw["stretch_probability"], 0.25)

    def test_unknown_value_warns_and_falls_to_the_default(self):
        # junk falls to the DEFAULT (eigen since 2026-09-15), loudly
        with mock.patch.dict(
            os.environ, {"VGB_INMODEL_PROPOSAL": "banana"}
        ):
            kw = {}
            with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch",
                level="WARNING",
            ):
                _vgb_inmodel_defaults(kw)
        self.assertTrue(kw["use_info_mat_proposal"])


class VGBEigenDrawTest(unittest.TestCase):
    def _chol(self, n):
        # identity axes with distinct per-column sigmas -> steps are
        # unambiguous single columns
        sig = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        return np.broadcast_to(
            np.eye(NDIM) * sig[None, :], (n, NDIM, NDIM)
        ).copy(), sig

    def test_use_eigen_draw_decision(self):
        s = _stub()
        chol = np.zeros((3, NDIM, NDIM))
        self.assertTrue(
            VGBSpecialStretchMove._vgb_use_eigen_draw(s, chol)
        )
        self.assertFalse(
            VGBSpecialStretchMove._vgb_use_eigen_draw(s, None)
        )
        s2 = _stub(use_info_mat_proposal=False)
        self.assertFalse(
            VGBSpecialStretchMove._vgb_use_eigen_draw(s2, chol)
        )
        s3 = _stub(stretch_probability=1.0)
        self.assertFalse(
            VGBSpecialStretchMove._vgb_use_eigen_draw(s3, chol)
        )

    def test_eigen_draw_moves_one_column_and_is_symmetric(self):
        n = 64
        rng = np.random.default_rng(97)
        coords = rng.standard_normal((n, NDIM))
        chol, sig = self._chol(n)
        s = _stub(jump_factor=2.0,
                  _proposal_param_scales=np.full(NDIM, 0.5))

        np.random.seed(11)
        new, factors = VGBSpecialStretchMove._vgb_eigen_axis_draw(
            s, coords, chol
        )
        np.testing.assert_array_equal(factors, 0.0)
        self.assertEqual(s._last_im_kind, "eigen_axis")
        dy = new - coords
        moved = np.abs(dy) > 0
        # exactly one column moves per source (identity axes)
        np.testing.assert_array_equal(moved.sum(axis=-1), 1)
        # magnitude carries jump_factor * sigma_k * param_scale
        for i in range(n):
            k = int(np.argmax(moved[i]))
            self.assertEqual(k, int(s._last_axis_pick[i]))
            self.assertLessEqual(
                np.abs(dy[i, k]), 2.0 * sig[k] * 0.5 * 6.0
            )  # 6-sigma sanity bound on the normal draw


class VGBCholGracefulTest(unittest.TestCase):
    def test_proposal_cholesky_failure_degrades_to_none(self):
        s = _stub()

        def boom(*args, **kwargs):
            raise RuntimeError("engine cannot serve the vgb basis")

        with mock.patch.object(
            GBSpecialBase, "_proposal_cholesky", side_effect=boom
        ):
            with self.assertLogs(
                "lisatools.globalfit.moves.gbspecialstretch",
                level="WARNING",
            ):
                out = VGBSpecialStretchMove._proposal_cholesky(
                    s, None, None, np.arange(3)
                )
        self.assertIsNone(out)
        # warn ONCE per process-lifetime of the move, not per block
        with mock.patch.object(
            GBSpecialBase, "_proposal_cholesky", side_effect=boom
        ):
            out2 = VGBSpecialStretchMove._proposal_cholesky(
                s, None, None, np.arange(3)
            )
        self.assertIsNone(out2)


class PerBlockTableGateTest(unittest.TestCase):
    def test_instance_flag_skips_the_cold_chain_table(self):
        calls = []
        s = types.SimpleNamespace(
            _prop_timer=None,
            _tables_indexed=False,
            share_proposal_tables=False,
            _build_friend_table=False,
            stretch_probability=0.0,
            use_info_mat_proposal=True,
            infomat_per_block=True,
            name="vgb",
            _infomat_freqs_sorted="stale",
            _infomat_chol_sorted="stale",
        )
        s._refresh_infomat_table = lambda *a: calls.append("refresh")
        bs = types.SimpleNamespace(
            build_infomat_index=lambda *a: calls.append("index"),
            index_friends=lambda *a: calls.append("friends"),
        )
        assert "GB_INFOMAT_PER_BLOCK" not in os.environ
        GBSpecialBase._ensure_proposal_tables(s, None, bs)
        self.assertEqual(calls, [])
        self.assertIsNone(s._infomat_freqs_sorted)
        self.assertIsNone(s._infomat_chol_sorted)


# ---------------------------------------------------------------------------
# Per-leaf fills on the info-matrix path (production failure, job 501)
# ---------------------------------------------------------------------------
#
# The vgb transform container is built with a per-leaf ``fill_dict`` LIST --
# one dict per catalogue source, pinning that leaf's f0 / sky (/ Mc) -- so
# ``fill_values`` selects each row's fill values BY LEAF INDEX and RAISES
# without them (eryn ``src/eryn/utils/transform.py:274-289``:
#
#     if self.n_leaf_fills is not None:
#         if leaf_inds is None:
#             raise ValueError(
#                 "This TransformContainer holds per-leaf fill values; "
#                 "pass leaf_inds (shape params.shape[:-1]) to "
#                 "fill_values/both_transforms.")
#
# and, when supplied, ``leaf_inds_in.shape`` must equal ``params.shape[:-1]``).
#
# On the first cluster run of the eigen default (6mo job 501, 2026-09-15) the
# very first vgb factor build hit exactly that raise and the warn-once wrapper
# degraded the block to stretch for the whole run:
#
#     vgb_pe: info-matrix factor unavailable for the vgb basis
#     (ValueError('This TransformContainer holds per-leaf fill values; ...'));
#     falling back to the stretch proposal
#
# Combined with the exact-truth starts (VGB_START_FACTOR=0 -> every walker
# bit-identical) the stretch fallback then proposed z-scaled ZERO steps off a
# zero-spread ensemble: ~90k "accepted" in-model vgb proposals at 0.537 while
# the stored vgb chain stayed bit-frozen across every row, rung and walker.
#
# The stubs above cannot see this: they hand ``_eigen_axes_from_info`` a
# ready-made ``info_y`` and never touch a transform. These tests drive the
# FACTOR BUILD (``_compute_proposal_cholesky`` -> ``_infomat_jacobian``)
# against a REAL per-leaf container, which is the only thing that exercises
# the contract.

_OUT_BASIS = ["A", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta"]
#: the default vgb sampled basis (``VGB_SAMPLED_BASIS_DIST``)
_IN_BASIS = ["dist", "phi0", "cos_iota", "psi", "fdot_astro_ratio"]
#: sampled/fill names -> their slot in the physical output basis, exactly the
#: aliasing the stock distance-basis gb container uses (``r`` rides the dead
#: fddot slot, ``Mc`` rides the fdot slot until the astro transform fires)
_KEY_MAP = {
    "dist": "A",
    "cos_iota": "iota",
    "fdot_astro_ratio": "fddot",
    "alpha": "lam",
    "sin_delta": "beta",
    "Mc": "fdot",
}

#: two leaves = two DIFFERENT physical sources, which is the whole reason the
#: vgb container holds per-leaf fills at all
_LEAF_FILLS = [
    {"f0": 2.0, "alpha": 0.7, "sin_delta": 0.2, "Mc": 0.30},
    {"f0": 9.0, "alpha": 2.4, "sin_delta": -0.6, "Mc": 0.55},
]


def _astro_quad(a_slot, f0_slot, fdot_slot, fddot_slot):
    """``(dist, f0, Mc, r) -> (A, f0, fdot, fddot=0)``.

    Structurally the stock distance-basis map: the amplitude and the
    frequency derivative BOTH mix a sampled column with the per-leaf fills,
    so a Jacobian built with the wrong leaf's fills is wrong by a finite,
    measurable amount rather than by roundoff.
    """
    dist, f0, mc, r = a_slot, f0_slot, fdot_slot, fddot_slot
    amp = mc ** (5.0 / 3.0) * f0 ** (2.0 / 3.0) / dist
    fdot = f0 ** (11.0 / 3.0) * mc ** (5.0 / 3.0) * (1.0 + r)
    return amp, f0, fdot, np.zeros_like(r)


def _per_leaf_container():
    from eryn.utils import TransformContainer

    return TransformContainer(
        input_basis=list(_IN_BASIS),
        output_basis=list(_OUT_BASIS),
        key_map=dict(_KEY_MAP),
        fill_dict=[dict(d) for d in _LEAF_FILLS],
        parameter_transforms={
            "iota": np.arccos,
            "beta": np.arcsin,
            ("A", "f0", "fdot", "fddot"): _astro_quad,
        },
    )


#: one coordinate row per source; rows 0/1 are the SAME coordinates so the
#: only thing that can separate their Jacobians is the per-leaf fills
_COORDS = np.array([
    [4.0, 1.1, 0.30, 0.8, 0.02],
    [4.0, 1.1, 0.30, 0.8, 0.02],
    [6.5, 2.0, -0.40, 1.9, -0.05],
    [6.5, 2.0, -0.40, 1.9, -0.05],
])
#: leaf identity per row: leaf 0, leaf 1, leaf 0, leaf 1
_LEAVES = np.array([0, 1, 0, 1])


def _info_phys_stub(n, ndim=NDIM, seed=5):
    """One SPD physical matrix, IDENTICAL for every source.

    Identical on purpose: with the physical curvature held fixed, every
    per-source difference in the mapped matrix is attributable to the
    Jacobian, hence to the fills the Jacobian was built with.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((ndim, ndim))
    m = a @ a.T + ndim * np.eye(ndim)
    return np.broadcast_to(m, (n, ndim, ndim)).copy()


def _factor_stub(container, **over):
    """A ``self`` carrying the REAL ``_compute_proposal_cholesky`` chain."""
    s = _stub(
        transform_fn=container,
        parameter_transforms=container,
        _prop_timer=None,
        _basis_settings=None,                    # not FDSettings -> wdm comp
        gb_wdm_comp=object(),                    # no ``chunked`` -> no slots
        gb_fd_comp=object(),
        _fdot_col=None,
        _fdot_scale=1e-16,
        _per_leaf_fill=True,
        mempool=None,
    )
    s._observable_basis_ready = lambda: False
    s._infomat_route_check = lambda *a, **k: None
    for name in ("_compute_proposal_cholesky", "_infomat_jacobian",
                 "_infomat_phys_inds", "_obs_eigen_mode",
                 "_eigen_axes_from_info"):
        setattr(s, name, types.MethodType(getattr(GBSpecialBase, name), s))
    for k, v in over.items():
        setattr(s, k, v)
    return s


def _sorter_stub(coords=_COORDS, leaves=_LEAVES):
    n = coords.shape[0]
    return types.SimpleNamespace(
        coords=coords.copy(),
        leaf_inds=leaves.copy(),
        walker_inds=np.arange(n),
        inds=np.ones(n, dtype=bool),
        infomat_take_inds=None,                  # -> direct per-block branch
    )


class PerLeafFillContractTest(unittest.TestCase):
    """The container contract the fix has to satisfy (eryn, not edited)."""

    def test_raises_without_leaf_inds_and_selects_with_them(self):
        tc = _per_leaf_container()
        self.assertEqual(tc.n_leaf_fills, len(_LEAF_FILLS))
        with self.assertRaises(ValueError) as cm:
            tc.both_transforms(_COORDS.copy(), xp=np)
        self.assertIn("per-leaf fill values", str(cm.exception))
        self.assertIn("leaf_inds", str(cm.exception))

        # shape contract: leaf_inds must be params.shape[:-1]
        with self.assertRaises(ValueError):
            tc.both_transforms(_COORDS.copy(), xp=np,
                               leaf_inds=np.zeros((2, 2), dtype=int))

        phys = tc.both_transforms(_COORDS.copy(), xp=np, leaf_inds=_LEAVES)
        f0_slot = _OUT_BASIS.index("f0")
        np.testing.assert_allclose(
            phys[:, f0_slot],
            [_LEAF_FILLS[i]["f0"] for i in _LEAVES])
        # identical coords, different leaf -> different physical amplitude
        self.assertNotAlmostEqual(float(phys[0, 0]), float(phys[1, 0]))


class PerLeafFillFactorTest(unittest.TestCase):
    """The vgb factor build against a real per-leaf container.

    RED before the fix: ``_compute_proposal_cholesky`` calls
    ``both_transforms`` with no ``leaf_inds``, the container raises, and
    ``VGBSpecialStretchMove._proposal_cholesky`` warns once and returns
    ``None`` -- i.e. every block degrades to stretch, which is production.
    """

    def _run_factor(self, s=None, sorter=None, ids=None):
        s = s if s is not None else _factor_stub(_per_leaf_container())
        sorter = sorter if sorter is not None else _sorter_stub()
        ids = np.arange(sorter.coords.shape[0]) if ids is None else ids
        with mock.patch.object(
            _gbbands._RoutedBandEngine, "route_information_matrix",
            side_effect=lambda comp, holder, params_phys, **kw:
                _info_phys_stub(int(np.shape(params_phys)[0])),
        ):
            return s, sorter, ids, s._compute_proposal_cholesky(
                types.SimpleNamespace(analysis_container_arr=None),
                sorter, ids)

    # -- the reproduction --------------------------------------------------

    def test_factor_is_built_not_degraded_to_stretch(self):
        s = _factor_stub(_per_leaf_container())
        sorter = _sorter_stub()
        with mock.patch.object(
            _gbbands._RoutedBandEngine, "route_information_matrix",
            side_effect=lambda comp, holder, params_phys, **kw:
                _info_phys_stub(int(np.shape(params_phys)[0])),
        ):
            # the warn-once fallback must NOT fire; assertNoLogs reports the
            # swallowed exception text when it does, which is the RED message
            with self.assertNoLogs(
                "lisatools.globalfit.moves.gbspecialstretch", level="WARNING",
            ):
                chol = VGBSpecialStretchMove._proposal_cholesky(
                    s,
                    types.SimpleNamespace(analysis_container_arr=None),
                    sorter, np.arange(4))
        self.assertIsNotNone(
            chol,
            "the vgb info-matrix factor fell back to stretch -- the per-leaf "
            "fill indices are not threaded through the info-matrix path "
            "(6mo job 501)")
        self.assertEqual(np.asarray(chol).shape, (4, NDIM, NDIM))
        self.assertTrue(np.all(np.isfinite(np.asarray(chol))))
        # the eigen draw is armed off this table
        self.assertTrue(VGBSpecialStretchMove._vgb_use_eigen_draw(s, chol))

    # -- the discrimination the fix exists for -----------------------------

    def test_jacobian_uses_each_row_s_own_leaf_fills(self):
        tc = _per_leaf_container()
        s = _factor_stub(tc)
        scales = np.ones(NDIM)
        test_inds = np.asarray(tc.fill_dict["test_inds"])

        j = s._infomat_jacobian(_COORDS.copy(), test_inds, scales,
                                leaf_inds=_LEAVES)
        self.assertEqual(j.shape, (4, NDIM, NDIM))
        self.assertTrue(np.all(np.isfinite(j)))

        # rows 0 and 1 hold IDENTICAL coordinates and differ only by leaf
        self.assertFalse(
            np.allclose(j[0], j[1]),
            "leaves 0 and 1 got the same Jacobian from identical coords -- "
            "the per-leaf fills are not reaching the central differences")

        # ... and each row matches a single-leaf build of its OWN leaf
        j0 = s._infomat_jacobian(_COORDS.copy(), test_inds, scales,
                                 leaf_inds=np.zeros(4, dtype=int))
        j1 = s._infomat_jacobian(_COORDS.copy(), test_inds, scales,
                                 leaf_inds=np.ones(4, dtype=int))
        np.testing.assert_allclose(j[0], j0[0], rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(j[1], j1[1], rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(j[2], j0[2], rtol=1e-9, atol=0.0)
        np.testing.assert_allclose(j[3], j1[3], rtol=1e-9, atol=0.0)

        # the amplitude derivative is the analytic one for THAT leaf's fills
        a_row = list(test_inds).index(_OUT_BASIS.index("A"))
        for row, leaf in enumerate(_LEAVES):
            f0 = _LEAF_FILLS[leaf]["f0"]
            mc = _LEAF_FILLS[leaf]["Mc"]
            dist = _COORDS[row, 0]
            want = -(mc ** (5.0 / 3.0)) * f0 ** (2.0 / 3.0) / dist ** 2
            self.assertAlmostEqual(
                float(j[row, a_row, 0]) / want, 1.0, delta=1e-4,
                msg=f"row {row} (leaf {leaf}) d(A)/d(dist)")

    def test_factor_table_separates_the_two_leaves(self):
        chol = np.asarray(self._run_factor()[3])
        # rows (0, 1) and rows (2, 3) are each one coordinate row visited by
        # BOTH leaves, and the physical matrix is identical everywhere, so
        # any difference within a pair IS the per-leaf fills
        for a, b in ((0, 1), (2, 3)):
            self.assertFalse(
                np.allclose(chol[a], chol[b]),
                f"rows {a}/{b} hold identical coords with different leaves "
                "and got the same proposal factor -- the info matrix was "
                "mapped with the wrong fills")

        # ... and each row equals the table built with ITS OWN leaf forced
        # everywhere, so no row is picking up a neighbour's fills
        per_leaf = [
            np.asarray(self._run_factor(
                sorter=_sorter_stub(leaves=np.full(4, leaf)))[3])
            for leaf in (0, 1)
        ]
        for row, leaf in enumerate(_LEAVES):
            np.testing.assert_allclose(
                chol[row], per_leaf[leaf][row], rtol=1e-9, atol=0.0,
                err_msg=f"row {row} was not built with leaf {leaf}'s fills")
            other = per_leaf[1 - leaf][row]
            self.assertFalse(
                np.allclose(chol[row], other),
                f"row {row} is indistinguishable from leaf {1 - leaf}'s "
                "factor -- the cross-leaf discrimination is not real")

    # -- the frozen-chain interaction --------------------------------------

    def test_zero_spread_ensemble_still_moves(self):
        """VGB_START_FACTOR=0 gives every walker bit-identical coords.

        Stretch off that ensemble is the identity (the frozen vgb chain of
        job 501); the eigen draw must not be, because its step comes from
        the information matrix rather than from the ensemble spread.
        """
        n = 8
        coords = np.tile(_COORDS[0], (n, 1))
        sorter = _sorter_stub(coords=coords,
                              leaves=np.tile(_LEAVES[:2], n // 2))
        s, sorter, ids, chol = self._run_factor(sorter=sorter)
        self.assertIsNotNone(chol)

        # the zero-spread signature: stretch cannot move this ensemble
        self.assertTrue(np.allclose(coords - coords[0][None, :], 0.0))

        np.random.seed(3)
        new, factors = VGBSpecialStretchMove._vgb_eigen_axis_draw(
            s, coords.copy(), chol)
        dy = np.asarray(new) - coords
        np.testing.assert_array_equal(np.asarray(factors), 0.0)
        self.assertEqual(s._last_im_kind, "eigen_axis")
        self.assertTrue(
            np.all(np.abs(dy).max(axis=-1) > 0.0),
            "the eigen draw proposed a zero move from a zero-spread "
            "ensemble -- the bit-frozen vgb chain signature")


if __name__ == "__main__":
    unittest.main()
