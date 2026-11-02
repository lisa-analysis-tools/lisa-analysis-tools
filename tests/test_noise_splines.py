"""Spline noise model on MOSA combinations (lisatools.noise_splines).

* the combination matrix and its inverse (mixing) are consistent;
* layout: components, edges, constants on the TM arm differences, reference levels
  prepended by ``flatten``, Gibbs blocks;
* knot building: edges, sorted interior knots, inactive knots and NaN padding;
* regression: the symmetric model equals the legacy symmetric spline path, and the
  asymmetric backend with only the common curves equals the symmetric one;
* aliasing with splines: a constant spline deviation scales the whole covariance,
  including the alias terms;
* the covariance equals the parametric one at the reference levels (times a constant
  deviation, also on the TM arm differences);
* knot prior: minimum separation and its normalisation ``P_ok(n)``, prior draws;
* reversible jump: a prior-only chain with birth/death and Gaussian moves recovers the
  uniform prior on the number of knots and the knot prior.

CPU backend only; no data files required.
"""

# Pre-existing env shims (see tests/test_quintic_response.py).
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU backend

import unittest

import numpy as np

from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import FDSettings
from lisatools.noise_splines import (
    COMBINATIONS,
    SplineComponent,
    SplineNoiseModel,
    combination_matrix,
    default_spline_components,
    make_spline_noise_model,
    mixing_vectors,
)
from lisatools.sensitivity import XYZSensitivityBackend

F_MIN, F_MAX = 1e-4, 2e-2
LEVELS = np.r_[np.full(6, 15e-12), np.full(6, 3e-15)]
LEVELS_SYM = [15e-12, 3e-15]


def _settings():
    return FDSettings(N=4096, df=5e-6, min_freq=F_MIN, max_freq=F_MAX, force_backend="cpu")


def _elements(b, Soms, Sa, kp=None, ka=None):
    return [np.asarray(c) for c in b._compute_matrix_elements(b.f_arr, Soms, Sa, knots_position_all=kp, knots_amplitude_all=ka)]


def _symmetric(n_interior=2, **kw):
    return SplineNoiseModel(default_spline_components("symmetric", n_interior), F_MIN, F_MAX, LEVELS_SYM, "symmetric", **kw)


class CombinationTest(unittest.TestCase):
    def test_mixing_inverts_combinations(self):
        R = combination_matrix()
        M = np.column_stack([mixing_vectors()[c] for c in COMBINATIONS])
        np.testing.assert_allclose(R @ M, np.eye(6), atol=1e-14)

    def test_common_is_uniform_and_armsum_is_arm_symmetric(self):
        vec = mixing_vectors()
        np.testing.assert_allclose(vec["common"], np.ones(6), atol=1e-14)
        for c in ("armsum_12_23", "armsum_23_31"):
            v = vec[c]
            for cw, ccw in [(0, 5), (1, 4), (2, 3)]:
                self.assertAlmostEqual(v[cw], v[ccw], places=14)
            self.assertAlmostEqual(v.sum(), 0.0, places=14)

    def test_tm_cw_ccw_rejected(self):
        with self.assertRaises(ValueError):
            SplineNoiseModel([SplineComponent("tm", "cw_ccw")], F_MIN, F_MAX, LEVELS)

    def test_symmetric_only_common(self):
        with self.assertRaises(ValueError):
            SplineNoiseModel([SplineComponent("oms", "armsum_12_23")], F_MIN, F_MAX, LEVELS_SYM, "symmetric")


class LayoutTest(unittest.TestCase):
    """Levels fixed at a reference spectrum, curves with edges + free knots, constants on
    the TM arm differences."""

    def setUp(self):
        self.model = SplineNoiseModel(default_spline_components("asymmetric", 3), F_MIN, F_MAX, LEVELS)

    def _branches(self, num, rng):
        return {b: self.model.draw_interior_knots(b, (num,), rng=rng)[0] for b in self.model.branch_names}

    def test_components(self):
        comps = default_spline_components()
        names = [c.name for c in comps]
        self.assertEqual(len(names), 11)
        self.assertNotIn("tm_cw_ccw", names)
        self.assertEqual([c.name for c in comps if c.constant], ["tm_armdiff_12_23", "tm_armdiff_23_31"])
        self.assertEqual([c.name for c in default_spline_components("symmetric")], ["oms_common", "tm_common"])
        self.assertTrue(all(c.knot_sigma == 0.3 and c.edge_sigma == 0.3 and c.amplitude_bound is None for c in comps))
        self.assertFalse(self.model.rj)
        self.assertEqual(self.model.n_knots_max, 5)
        self.assertEqual(self.model.n_edge_params, 9 * 2 + 2)
        self.assertEqual(self.model.n_spline_params, 20 + 9 * 3 * 2)
        self.assertEqual(len(self.model.branch_names), 9)
        self.assertEqual(self.model.edge_param_keys[-3:], ["tm_armsum_23_31_edge_max", "tm_armdiff_12_23_constant",
                                                           "tm_armdiff_23_31_constant"])
        with self.assertRaises(ValueError):  # a constant has no interior knots
            SplineNoiseModel([SplineComponent("tm", "armdiff_12_23", n_interior=1, constant=True)], F_MIN, F_MAX, LEVELS)
        with self.assertRaises(ValueError):
            SplineNoiseModel(default_spline_components(), F_MIN, F_MAX, LEVELS[:2])
        with self.assertRaises(ValueError):  # 14 knots 0.2 dex apart do not fit in 2.3 dex
            SplineNoiseModel(default_spline_components("symmetric", 14), F_MIN, F_MAX, LEVELS_SYM, "symmetric")

    def test_knots(self):
        model = _symmetric(1)
        sp = np.array([0.1, -0.2, 0.3, 0.05, -2.5, 0.4, -3.5, -0.1])  # edges (oms, tm), knots (oms, tm)
        pos, amp = model.knots(sp)
        self.assertEqual(pos.shape, (2, 1, 3))
        lo, hi = np.log10(F_MIN), np.log10(F_MAX)
        np.testing.assert_allclose(pos[0, 0], [lo, -2.5, hi])
        np.testing.assert_allclose(amp[0, 0], [0.1, 0.4, -0.2])
        np.testing.assert_allclose(pos[1, 0], [lo, -3.5, hi])
        np.testing.assert_allclose(amp[1, 0], [0.3, -0.1, 0.05])
        # a constant curve: one parameter at both edges, then NaN padding
        pos, amp = self.model.knots(np.arange(self.model.n_spline_params, dtype=float) / 100)
        j = [c.name for c in self.model.components].index("tm_armdiff_23_31")
        np.testing.assert_allclose(pos[j, 0, :2], [lo, hi])
        np.testing.assert_allclose(amp[j, 0, :2], [0.19, 0.19])  # last edge parameter
        self.assertTrue(np.all(np.isnan(pos[j, 0, 2:])))

    def test_inactive_knots(self):
        """Inactive knots are dropped: the knots equal those of a curve without them."""
        model = _symmetric(3)
        edges = np.array([[0.1, -0.2, 0.3, 0.05]])
        oms = np.array([[[-3.5, 0.4], [-2.0, -0.3], [-2.8, 0.2]]])
        tm = np.array([[[-3.0, 0.1], [-2.5, 0.2], [-2.2, -0.1]]])
        inds = {"psd_oms_common": np.array([[True, False, True]]), "psd_tm_common": np.array([[False, False, False]])}
        rows = model.flatten(edges, {"psd_oms_common": oms, "psd_tm_common": tm}, inds)
        self.assertTrue(np.all(np.isnan(rows[0, 2 + 4 + 2 : 2 + 4 + 4])))  # second OMS knot
        pos, amp = model.knots(rows[:, 2:])
        lo, hi = np.log10(F_MIN), np.log10(F_MAX)
        np.testing.assert_allclose(pos[0, 0, :4], [lo, -3.5, -2.8, hi])
        np.testing.assert_allclose(amp[0, 0, :4], [0.1, 0.4, 0.2, -0.2])
        np.testing.assert_allclose(pos[1, 0, :2], [lo, hi])
        np.testing.assert_allclose(amp[1, 0, :2], [0.3, 0.05])
        self.assertTrue(np.all(np.isnan(pos[0, 0, 4:])) and np.all(np.isnan(pos[1, 0, 2:])))
        self.assertTrue(np.all(np.isnan(amp[1, 0, 2:])))
        self.assertFalse(model.invalid_knots(pos).any())

    def test_flatten_prepends_levels(self):
        rng = np.random.default_rng(2)
        edges = rng.normal(0, 0.3, (4, self.model.n_edge_params))
        rows = self.model.flatten(edges, self._branches(4, rng))
        self.assertEqual(rows.shape, (4, 12 + self.model.n_spline_params))
        np.testing.assert_array_equal(rows[:, :12], np.broadcast_to(LEVELS, (4, 12)))
        np.testing.assert_array_equal(rows[:, 12 : 12 + self.model.n_edge_params], edges)
        with self.assertRaises(ValueError):  # the psd branch holds only the edges
            self.model.flatten(np.zeros((4, 12 + self.model.n_edge_params)), self._branches(4, rng))

    def test_priors(self):
        edge = list(self.model.edge_priors().values())[0]
        np.testing.assert_allclose(edge.logpdf(np.array([0.3])), -0.5 - np.log(0.3 * np.sqrt(2 * np.pi)))
        knots = self.model.branch_priors()["psd_oms_common"]
        lp = knots.logpdf(np.array([[-3.0, 0.0], [-0.5, 0.0]]))  # in / out of the band
        self.assertTrue(np.isfinite(lp[0]) and np.isinf(lp[1]))
        bounded = SplineNoiseModel(default_spline_components("asymmetric", 3, amplitude_bound=1.0), F_MIN, F_MAX, LEVELS)
        edge = list(bounded.edge_priors().values())[0]
        self.assertTrue(np.isfinite(edge.logpdf(np.array([0.99]))).all())
        self.assertTrue(np.isinf(edge.logpdf(np.array([1.01]))).all())

    def test_gibbs_blocks(self):
        blocks = self.model.gibbs_sampling_setup(self.model.n_edge_params)
        psd_blocks = [m for b, m in blocks[:2]]
        self.assertEqual([b for b, _ in blocks[:2]], ["psd", "psd"])  # OMS edges, TM edges
        np.testing.assert_array_equal(psd_blocks[0][0] | psd_blocks[1][0], True)
        self.assertEqual(psd_blocks[0][0].sum(), 12)  # 6 OMS curves x 2 edges
        self.assertEqual(psd_blocks[1][0].sum(), 8)   # 3 TM curves x 2 edges + 2 constants
        self.assertEqual(blocks[2:], self.model.branch_names)
        with self.assertRaises(ValueError):
            self.model.gibbs_sampling_setup(12 + self.model.n_edge_params)


class KnotPriorTest(unittest.TestCase):
    def setUp(self):
        self.model = _symmetric(6)
        self.width = self.model.log10_f_max - self.model.log10_f_min

    def test_separation_probability_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        s = self.model.min_log10_separation
        for n in range(7):
            x = np.sort(rng.uniform(0, self.width, (200000, n)), axis=1)
            gaps = np.diff(np.c_[np.zeros(len(x)), x, np.full(len(x), self.width)], axis=1)
            mc = np.mean(np.all(gaps >= s, axis=1))
            self.assertAlmostEqual(np.exp(self.model.log_separation_probability(n)), mc, delta=4e-3)
        self.assertEqual(self.model.log_separation_probability(0), 0.0)
        self.assertEqual(self.model.log_separation_probability(12), -np.inf)  # 13 gaps > 2.3 dex

    def test_knot_log_prior(self):
        lo = self.model.log10_f_min
        coords = np.zeros((4, 6, 2))
        coords[..., 0] = lo + 0.3 * (1 + np.arange(6))  # 0.3 dex apart, inside the band
        inds = np.ones((4, 6), dtype=bool)
        inds[1, 3:] = False  # 3 knots
        coords[2, 1, 0] = coords[2, 0, 0] + 0.1  # too close to its neighbour
        inds[3, 1] = False
        coords[3, 1, 0] = coords[3, 0, 0] + 0.1  # too close, but inactive
        coords[3, 5, 0] = lo + 0.1  # too close to the f_min edge
        lp = self.model.knot_log_prior("psd_oms_common", coords, inds)
        np.testing.assert_allclose(lp[:2], -self.model.log_separation_probability([6, 3]))
        self.assertEqual(lp[2], -np.inf)
        self.assertEqual(lp[3], -np.inf)
        inds[3, 5] = False
        self.assertAlmostEqual(
            self.model.knot_log_prior("psd_oms_common", coords[3], inds[3]), -self.model.log_separation_probability(4)
        )

    def test_exact_draws_match_rejection_sampling(self):
        """The spacing-based draw equals uniform knots conditioned on the separation."""
        rng = np.random.default_rng(3)
        lo, hi = self.model.log10_f_min, self.model.log10_f_max
        for n in (2, 4):
            x = rng.uniform(lo, hi, (400000, n))
            keep = np.isfinite(self.model.knot_log_prior("psd_oms_common", np.stack([x, 0 * x], -1)))
            ref = np.sort(x[keep], axis=1)
            m = SplineNoiseModel(default_spline_components("symmetric", n), F_MIN, F_MAX, LEVELS_SYM, "symmetric")
            got = np.sort(m.draw_interior_knots("psd_oms_common", (len(ref),), rng=rng)[0][..., 0], axis=1)
            np.testing.assert_allclose(got.mean(0), ref.mean(0), atol=0.01)
            np.testing.assert_allclose(got.std(0), ref.std(0), atol=0.01)
            self.assertGreaterEqual(np.diff(np.c_[np.full(len(got), lo), got, np.full(len(got), hi)], axis=1).min(),
                                    self.model.min_log10_separation)

    def test_draws(self):
        rj = SplineNoiseModel(default_spline_components("symmetric", 6, n_interior_min=0), F_MIN, F_MAX, LEVELS_SYM, "symmetric")
        self.assertTrue(rj.rj)
        self.assertEqual(rj.nleaves_min, {"psd_oms_common": 0, "psd_tm_common": 0})
        self.assertEqual(rj.nleaves_max, {"psd_oms_common": 6, "psd_tm_common": 6})
        rng = np.random.default_rng(1)
        n = rj.draw_n_active("psd_oms_common", (3, 50), rng)
        self.assertEqual(set(np.unique(n)), set(range(7)))
        coords, inds = rj.draw_interior_knots("psd_oms_common", (3, 50), n_active=n, rng=rng)
        self.assertEqual(coords.shape, (3, 50, 6, 2))
        np.testing.assert_array_equal(inds.sum(axis=-1), n)
        self.assertTrue(np.all(np.isfinite(rj.knot_log_prior("psd_oms_common", coords, inds))))
        # fixed number of knots: all active
        coords, inds = self.model.draw_interior_knots("psd_tm_common", (40,), rng=rng)
        self.assertTrue(inds.all())
        self.assertTrue(np.all(np.isfinite(self.model.knot_log_prior("psd_tm_common", coords, inds))))
        with self.assertRaises(ValueError):
            self.model.draw_interior_knots("psd_tm_common", (2,), n_active=3)  # below the fixed number


class FactoryTest(unittest.TestCase):
    def test_make_spline_noise_model(self):
        m = make_spline_noise_model(F_MIN, F_MAX, np.full(6, 15e-12), 3e-15)
        self.assertEqual(m.noise_symmetry, "asymmetric")
        self.assertFalse(m.rj)
        np.testing.assert_array_equal(m.reference_levels, LEVELS)
        self.assertEqual(m.n_sampled_params, 20 + 9 * 3 * 2)
        self.assertEqual(m.psd_priors().ndim, m.n_edge_params)
        np.testing.assert_array_equal(m.injection, np.zeros(20))
        self.assertEqual(len(m.gibbs_sampling_setup()), 2 + 9)
        md = m.metadata()
        self.assertEqual((md["num_params"], md["reversible_jump"], md["min_log10_separation"]), (74, False, 0.2))
        rj = make_spline_noise_model(F_MIN, F_MAX, np.full(6, 15e-12), np.full(6, 3e-15), "symmetric", rj=True,
                                     rj_n_min=1, rj_n_max=5, sigma=0.2, min_separation=0.1)
        self.assertTrue(rj.rj)
        np.testing.assert_array_equal(rj.reference_levels, LEVELS_SYM)
        self.assertEqual(rj.nleaves_min, {"psd_oms_common": 1, "psd_tm_common": 1})
        self.assertEqual(rj.nleaves_max, {"psd_oms_common": 5, "psd_tm_common": 5})
        self.assertEqual(rj.components[0].knot_sigma, 0.2)
        back = SplineNoiseModel.from_dict(__import__("json").loads(__import__("json").dumps(rj.to_dict())))
        self.assertEqual(back.describe(), rj.describe())
        self.assertEqual(back.components, rj.components)
        with self.assertRaises(ValueError):  # unequal levels need the asymmetric model
            make_spline_noise_model(F_MIN, F_MAX, np.linspace(1, 2, 6), 3e-15, "symmetric")

    def test_zero_separation_adds_nothing(self):
        m = make_spline_noise_model(F_MIN, F_MAX, 15e-12, 3e-15, "symmetric", rj=True, min_separation=0.0)
        np.testing.assert_array_equal(m.log_separation_probability(np.arange(9)), 0.0)
        coords, inds = m.draw_interior_knots("psd_oms_common", (50,), n_active=m.draw_n_active("psd_oms_common", (50,)))
        coords[:, 1, 0] = coords[:, 0, 0]  # coincident knots are allowed
        np.testing.assert_array_equal(m.knot_log_prior("psd_oms_common", coords, inds), 0.0)


class EqualNoiseRegressionTest(unittest.TestCase):
    """The symmetric model equals the legacy symmetric splines."""

    @classmethod
    def setUpClass(cls):
        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")
        rng = np.random.default_rng(3)
        # cubic, to compare with the (cubic Akima) legacy layout
        cls.model_sym = _symmetric(2, interpolation="cubic")
        cls.model_asym = SplineNoiseModel(
            default_spline_components("symmetric", 2), F_MIN, F_MAX, LEVELS, "asymmetric", interpolation="cubic"
        )
        # edges then interior knots (oms, tm)
        cls.sp = np.r_[rng.normal(0, 0.3, 4), [-3.3, 0.2, -2.4, -0.1], [-3.6, 0.15, -2.2, 0.1]]

    def _legacy_params(self, model):
        """Same knots in the legacy interleaved (amplitude, position) layout."""
        pos, amp = model.knots(self.sp)
        out = []
        for c in range(2):
            ok = ~np.isnan(pos[c, 0])
            for a, p in zip(amp[c, 0][ok], pos[c, 0][ok]):
                out += [a, p]
        return np.array(out)

    def test_symmetric_model_matches_legacy(self):
        legacy = XYZSensitivityBackend(orbits=self.orbits, settings=_settings(), force_backend="cpu", use_splines=True)
        new = XYZSensitivityBackend(orbits=self.orbits, settings=_settings(), force_backend="cpu", spline_model=self.model_sym)
        kp_l, ka_l = legacy.build_spline_arrays(self._legacy_params(self.model_sym))
        kp_n, ka_n = new.build_spline_arrays(self.sp)
        a = _elements(legacy, 15e-12, 3e-15, kp_l, ka_l)
        b = _elements(new, 15e-12, 3e-15, kp_n, ka_n)
        for x, y in zip(a, b):
            np.testing.assert_allclose(y, x, rtol=1e-13)

    def test_asymmetric_common_only_matches_symmetric(self):
        sym = XYZSensitivityBackend(orbits=self.orbits, settings=_settings(), force_backend="cpu", spline_model=self.model_sym)
        asym = XYZSensitivityBackend(
            orbits=self.orbits, settings=_settings(), force_backend="cpu", noise_symmetry="asymmetric", spline_model=self.model_asym
        )
        kp, ka = sym.build_spline_arrays(self.sp)
        a = _elements(sym, 15e-12, 3e-15, kp, ka)
        b = _elements(asym, np.full(6, 15e-12), np.full(6, 3e-15), kp, ka)
        for x, y in zip(a, b):
            np.testing.assert_allclose(y, x, rtol=1e-12, atol=1e-12 * np.abs(x).max())

    def test_unequal_model_runs_with_inactive_knots(self):
        model = SplineNoiseModel(default_spline_components("asymmetric", 3, n_interior_min=0), F_MIN, F_MAX, LEVELS)
        b = XYZSensitivityBackend(
            orbits=self.orbits, settings=_settings(), force_backend="cpu", noise_symmetry="asymmetric", spline_model=model
        )
        rng = np.random.default_rng(5)
        names = model.branch_names
        n = {k: rng.integers(0, 4) for k in names}
        drawn = {k: model.draw_interior_knots(k, (1,), n_active=n[k], rng=rng) for k in names}
        rows = model.flatten(
            rng.normal(0, 0.2, (1, model.n_edge_params)), {k: v[0] for k, v in drawn.items()}, {k: v[1] for k, v in drawn.items()}
        )
        sens = b("w0", rows[0])  # __call__ path: split, build knots, set the matrix
        self.assertTrue(np.all(np.isfinite(np.asarray(sens.sens_mat))))


class LinearInterpolationTest(unittest.TestCase):
    def test_default_linear_matches_np_interp(self):
        model = _symmetric(2)
        self.assertEqual(model.interpolation, "linear")
        b = XYZSensitivityBackend(orbits=EqualArmlengthOrbits(force_backend="cpu"), settings=_settings(),
                                  force_backend="cpu", spline_model=model)
        self.assertEqual(b.spline_order, "linear")
        sp = np.array([0.2, -0.1, 0.05, 0.3, -3.3, 0.4, -2.4, -0.2, -3.6, -0.3, -2.2, 0.1])
        kp, ka = b.build_spline_arrays(sp)
        oms, tm = b._spline_deviations(kp, ka, 1, b.f_arr)
        lf = np.clip(np.log10(np.asarray(b.f_arr)), model.log10_f_min, model.log10_f_max)
        for c, dev in enumerate((oms, tm)):
            ok = ~np.isnan(kp[c, 0])
            np.testing.assert_allclose(dev, np.interp(lf, kp[c, 0][ok], ka[c, 0][ok]), atol=1e-14)


class AliasSplineTest(unittest.TestCase):
    def test_constant_deviation_scales_alias_terms(self):
        orbits = EqualArmlengthOrbits(force_backend="cpu")
        st = _settings()
        f = np.asarray(st.f_arr)
        fs_new = 2 * F_MAX * 1.2
        alias = dict(alias_frequencies=-(fs_new - f)[None], alias_filters_response=np.full((1, f.size), 0.5))
        model = _symmetric(0)
        b = XYZSensitivityBackend(orbits=orbits, settings=st, force_backend="cpu", spline_model=model, **alias)
        b0 = XYZSensitivityBackend(orbits=orbits, settings=st, force_backend="cpu", **alias)
        # a constant curve (both edges at c) multiplies every term by 10^c
        for Soms, Sa, c in [(15e-12, 0.0, 0.3), (0.0, 3e-15, -0.2)]:
            kp, ka = b.build_spline_arrays(np.zeros(model.n_spline_params))
            ka = np.where(np.isnan(kp), np.nan, c)
            got = _elements(b, Soms, Sa, kp, ka)
            ref = _elements(b0, Soms, Sa)
            for x, y in zip(ref, got):
                np.testing.assert_allclose(y, 10**c * x, rtol=1e-13, atol=1e-13 * np.abs(x).max())


class CovarianceTest(unittest.TestCase):
    def test_covariance_matches_parametric(self):
        model = SplineNoiseModel(default_spline_components("asymmetric", 3), F_MIN, F_MAX, LEVELS)
        orbits = EqualArmlengthOrbits(force_backend="cpu")
        kw = dict(orbits=orbits, settings=_settings(), force_backend="cpu", noise_symmetry="asymmetric")
        b = XYZSensitivityBackend(spline_model=model, **kw)
        b0 = XYZSensitivityBackend(**kw)
        rng = np.random.default_rng(4)
        br = {k: model.draw_interior_knots(k, (1,), rng=rng)[0] for k in model.branch_names}
        # zero deviation: the parametric model at the reference levels
        edges = np.zeros((1, model.n_edge_params))
        for k in br:
            br[k][..., 1] = 0.0
        got = np.asarray(b("w0", model.flatten(edges, br)[0]).sens_mat)
        ref = np.asarray(b0("w1", LEVELS).sens_mat)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
        # constant c on the common curves (edges and knots) = levels scaled by 10^(c/2)
        c_oms, c_tm = 0.4, -0.3
        names = [c.name for c in model.components]
        for name, c in (("oms_common", c_oms), ("tm_common", c_tm)):
            j = names.index(name)
            edges[0, [model.edge_columns[0][j], model.edge_columns[1][j]]] = c
            br[f"psd_{name}"][..., 1] = c
        got = np.asarray(b("w0", model.flatten(edges, br)[0]).sens_mat)
        scaled = LEVELS * np.r_[np.full(6, 10 ** (c_oms / 2)), np.full(6, 10 ** (c_tm / 2))]
        ref = np.asarray(b0("w1", scaled).sens_mat)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())
        # plus a constant on a TM arm difference: per-MOSA TM levels scaled by its mixing pattern
        c_ad = 0.5
        edges[0, model.edge_param_keys.index("tm_armdiff_12_23_constant")] = c_ad
        got = np.asarray(b("w0", model.flatten(edges, br)[0]).sens_mat)
        scaled[6:] *= 10 ** (mixing_vectors()["armdiff_12_23"] * c_ad / 2)
        ref = np.asarray(b0("w1", scaled).sens_mat)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


class ReversibleJumpPriorTest(unittest.TestCase):
    """Prior-only chain with the PSD move's birth/death step and Gaussian in-model moves:
    the number of knots must follow the uniform prior and the knots their prior."""

    def test_prior_recovery(self):
        from eryn.model import Model
        from eryn.moves import GaussianMove, TemperatureControl
        from eryn.state import State

        from lisatools.globalfit.moves.psdmove import SplineKnotRJMove

        model = SplineNoiseModel(default_spline_components("symmetric", 6, n_interior_min=0), F_MIN, F_MAX, LEVELS_SYM, "symmetric")
        branch = "psd_oms_common"
        L, nw = 6, 64
        priors = model.branch_priors()

        def log_prior(coords, inds=None, **kwargs):
            x, act = coords[branch], inds[branch]
            lp = priors[branch].logpdf(x.reshape(-1, 2)).reshape(act.shape)
            return np.where(act, lp, 0.0).sum(-1) + model.knot_log_prior(branch, x, act)

        def log_like(coords, inds=None, logp=None, **kwargs):
            return np.where(np.isfinite(logp), 0.0, -1e300), None

        rng = np.random.default_rng(7)
        coords, inds = model.draw_interior_knots(branch, (1, nw), n_active=model.draw_n_active(branch, (1, nw), rng), rng=rng)
        state = State({branch: coords}, inds={branch: inds})
        state.log_prior = log_prior(state.branches_coords, state.branches_inds)
        state.log_like = np.zeros((1, nw))

        tc = TemperatureControl(2 * L, nw, ntemps=1)
        random = np.random.RandomState(8)
        mdl = Model(None, log_like, log_prior, tc, None, random)
        rj = SplineKnotRJMove(
            priors, nleaves_max={branch: L}, nleaves_min={branch: 0}, temperature_control=tc, prevent_swaps=True
        )
        gm = GaussianMove({branch: np.diag([0.3, 0.1]) ** 2}, temperature_control=tc, prevent_swaps=True)
        for m in (rj, gm):
            m.accepted = np.zeros((1, nw))

        n_samples, pos, amp = [], [], []
        for it in range(3000):
            state, _ = rj.propose(mdl, state)
            state, _ = gm.propose(mdl, state)
            if it >= 500 and it % 5 == 0:
                act = state.branches_inds[branch][0]
                n_samples.append(act.sum(-1))
                x = state.branches_coords[branch][0][act]
                pos.append(x[:, 0])
                amp.append(x[:, 1])
        n_samples = np.concatenate(n_samples)
        freq = np.bincount(n_samples, minlength=L + 1) / n_samples.size
        np.testing.assert_allclose(freq, 1 / (L + 1), atol=0.03)
        amp = np.concatenate(amp)
        self.assertAlmostEqual(amp.mean(), 0.0, delta=0.02)
        self.assertAlmostEqual(amp.std(), 0.3, delta=0.02)
        pos = np.concatenate(pos)
        self.assertTrue(np.all((pos > model.log10_f_min) & (pos < model.log10_f_max)))
        self.assertGreater(rj.accepted.sum() / (rj.num_proposals * nw), 0.2)


class FancySwapTest(unittest.TestCase):
    """The PSD move's temperature swap carries the active knots: equal to eryn's without
    ``inds``, and to an explicit swap of coordinates and ``inds`` with them."""

    def test_fancy_swap_with_inds(self):
        import types

        from eryn.moves import TemperatureControl

        from lisatools.globalfit.moves.psdmove import _fancy_swap_with_inds

        rng = np.random.default_rng(0)
        nt, nw = 3, 8
        x = {"a": rng.normal(size=(nt, nw, 1, 2)), "b": rng.normal(size=(nt, nw, 3, 2))}
        bs = {"a": None, "b": None}

        def log_like(coords, inds=None, supps=None, branch_supps=None, logp=None):
            # depends on the walker index, like the per-walker data streams
            act = np.ones(coords["b"].shape[:3], bool) if inds is None else inds["b"]
            v = coords["a"][..., 0, 0] ** 2 + (np.where(act, coords["b"][..., 0], 0) ** 2).sum(-1)
            return -v * (1 + 0.1 * np.arange(coords["a"].shape[1])[None]), None

        tc = TemperatureControl(4, nw, ntemps=nt)
        swap = types.MethodType(_fancy_swap_with_inds, tc)
        iperm, i1perm = rng.permutation(nw), rng.permutation(nw)
        logl = log_like(x)[0]
        ref = tc.perform_fancy_swap_acceptance_fraction(0.1, 1, iperm, i1perm, x, logl, branch_supps=bs, compute_log_like=log_like)
        np.testing.assert_allclose(swap(0.1, 1, iperm, i1perm, x, logl, branch_supps=bs, compute_log_like=log_like), ref)

        inds = {"a": np.ones((nt, nw, 1), bool), "b": rng.random((nt, nw, 3)) > 0.4}
        logl = log_like(x, inds)[0]
        got = swap(0.1, 1, iperm, i1perm, x, logl, inds=inds, branch_supps=bs, compute_log_like=log_like)
        xs = {k: v[:2].copy() for k, v in x.items()}
        ins = {k: v[:2].copy() for k, v in inds.items()}
        for d in (xs, ins):
            for k in d:
                src = d[k].copy()
                d[k][1, iperm], d[k][0, i1perm] = src[0, i1perm], src[1, iperm]
        new = log_like(xs, ins)[0]
        np.testing.assert_allclose(got, tc.betas[1] * (new[1] - logl[1]) + tc.betas[0] * (new[0] - logl[0]))


if __name__ == "__main__":
    unittest.main()
