"""Eigen-table stencils must stay inside the prior box.

The 2026-10-09 defect (9mo job 751): EMRI leaf 3 on walker 0 sat at
``cos(qK) = -0.9999770663`` with the stock prior ``uniform(-0.99999,
0.99999)`` (``stock/erebor/emri.py``), 1.3e-5 inside the lower bound. Every
refresh then built finite-difference stencils ``x0 +/- s e_i`` with ``s``
(2e-4 and up) larger than that margin, so one stencil row left the box:

* the Gram route (``_gram_info``) got a NaN template (the cos -> arccos
  transform) and failed with "non-finite information matrix" every refresh;
* the likelihood fallback (``eigen_table_from_ll``) scored the same corner --
  the EMRI scorer turns a NaN template into "no signal", so the second
  differences came out finite but absurd and nothing warned.

The fix: :func:`eigen_refresh.prior_box_bounds` reads the box,
:func:`eigen_refresh.nudge_inside` moves the expansion point one step in from
a bound, and both builders use it. The fakes here return NaN for any row
outside the box, exactly what the transform did.
"""

import types
import unittest
from unittest import mock

import numpy as np

from eryn.moves import EigenAxisMove
from eryn.prior import ProbDistContainer, uniform_dist

from lisatools.globalfit.moves import eigen_refresh
from lisatools.globalfit.moves.eigen_refresh import (
    eigen_table_from_ll,
    eigen_tables_from_ll_batch,
    prior_box_widths,
)

#: the stock EMRI cos(qS) / cos(qK) box (stock/erebor/emri.py)
LO, HI = -0.99999, 0.99999
#: job 751, EMRI leaf 3, walker 0: 1.3e-5 inside LO
X_EDGE = -0.9999770663


def _in_box(rows, lo, hi):
    rows = np.atleast_2d(rows)
    return bool(np.all((rows >= lo) & (rows <= hi)))


# ---------------------------------------------------------------------------
# prior_box_bounds (and prior_box_widths unchanged by the shared reader)
# ---------------------------------------------------------------------------


class _Parsed:
    """A container that carries only eryn's parsed ``priors`` list."""

    def __init__(self, entries):
        self.priors = entries


class _Dist:
    def __init__(self, mn, mx):
        self.minimum = mn
        self.maximum = mx


def _mixed_container():
    """Every reader case on one container (ndim = 9)."""
    return _Parsed([
        [np.array([0]), _Dist(0.0, 4.0)],                    # plain box
        [np.array([1, 3]), _Dist(-1.0, 1.0)],                # tuple key, scalar box
        [np.array([2]), _Dist(-np.inf, np.inf)],             # eryn's default box
        [np.array([4]), _Dist(2.0, 2.0)],                    # zero width
        [np.array([6]), object()],                           # no minimum / maximum
        [np.array([7, 9]),                                   # column 9 is out of range
         _Dist(np.array([0.0, 10.0]), np.array([3.0, 20.0]))],
        [np.array([8]), _Dist(3.0, 1.0)],                    # reversed box
        # column 5: not covered at all
    ])


class PriorBoxBoundsTest(unittest.TestCase):
    def test_covered_columns_give_the_prior_box_and_uncovered_are_infinite(self):
        from lisatools.globalfit.moves.eigen_refresh import prior_box_bounds

        pri = ProbDistContainer({
            0: uniform_dist(0.1, 30.0),
            1: uniform_dist(LO, HI),
            2: uniform_dist(1e-3, 1.0),
        })
        lo, hi = prior_box_bounds(pri, 5)
        np.testing.assert_array_equal(lo[:3], [0.1, LO, 1e-3])
        np.testing.assert_array_equal(hi[:3], [30.0, HI, 1.0])
        np.testing.assert_array_equal(lo[3:], -np.inf)
        np.testing.assert_array_equal(hi[3:], np.inf)

    def test_label_keyed_dict_resolves_like_prior_box_widths(self):
        # the stock EMRI / MBH / SOBBH / psd dicts are keyed by LABEL; only
        # the parsed priors list knows their columns (_prior_entries)
        from lisatools.globalfit.moves.eigen_refresh import prior_box_bounds

        pri = ProbDistContainer({
            "logm1": uniform_dist(np.log(3e5), np.log(1.5e7)),
            "m2": uniform_dist(1.0, 200.0),
            "qK": uniform_dist(LO, HI),
        })
        lo, hi = prior_box_bounds(pri, 4)
        np.testing.assert_array_equal(lo[:3], [np.log(3e5), 1.0, LO])
        np.testing.assert_array_equal(hi[:3], [np.log(1.5e7), 200.0, HI])
        self.assertEqual((lo[3], hi[3]), (-np.inf, np.inf))
        np.testing.assert_allclose(prior_box_widths(pri, 4)[:3], (hi - lo)[:3],
                                   rtol=1e-15)

    def test_every_reader_case_on_one_container(self):
        from lisatools.globalfit.moves.eigen_refresh import prior_box_bounds

        lo, hi = prior_box_bounds(_mixed_container(), 9)
        inf = np.inf
        np.testing.assert_array_equal(
            lo, [0.0, -1.0, -inf, -1.0, -inf, -inf, -inf, 0.0, -inf])
        np.testing.assert_array_equal(
            hi, [4.0, 1.0, inf, 1.0, inf, inf, inf, 3.0, inf])

    def test_prior_box_widths_values_and_warning_unchanged(self):
        # pinned against the pre-refactor reader (2026-10-09): the shared
        # one-pass reader must not move a single width or warned column
        eigen_refresh._UNREAD_WARNED.clear()
        with self.assertLogs(eigen_refresh.logger, level="WARNING") as cm:
            w = prior_box_widths(_mixed_container(), 9)
        np.testing.assert_array_equal(w, [4.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 3.0, 2.0])
        self.assertEqual(len(cm.output), 1, cm.output)
        self.assertIn("[2, 4, 6, 8]", cm.output[0])

    def test_a_reader_failure_keeps_the_columns_read_so_far_quietly(self):
        # prior_box_widths reports a broken prior on the same refresh; the
        # bounds degrade to "no nudge" on the unread columns without a
        # second log line
        from lisatools.globalfit.moves.eigen_refresh import prior_box_bounds

        class Raising:
            maximum = 1.0

            @property
            def minimum(self):
                raise RuntimeError("exotic")

        pri = _Parsed([
            [np.array([0]), _Dist(0.0, 5.0)],
            [np.array([1]), Raising()],
        ])
        with self.assertNoLogs(eigen_refresh.logger, level="WARNING"):
            lo, hi = prior_box_bounds(pri, 2)
        np.testing.assert_array_equal(lo, [0.0, -np.inf])
        np.testing.assert_array_equal(hi, [5.0, np.inf])

        class Boom:
            @property
            def priors_in(self):
                raise RuntimeError("no")

        lo, hi = prior_box_bounds(Boom(), 3)
        np.testing.assert_array_equal(lo, -np.inf)
        np.testing.assert_array_equal(hi, np.inf)


# ---------------------------------------------------------------------------
# nudge_inside
# ---------------------------------------------------------------------------


class NudgeInsideTest(unittest.TestCase):
    def setUp(self):
        from lisatools.globalfit.moves.eigen_refresh import nudge_inside

        self.nudge = nudge_inside

    def test_the_job751_point_moves_one_step_in(self):
        x = np.array([X_EDGE])
        out = self.nudge(x, np.array([2e-4]), np.array([LO]), np.array([HI]))
        np.testing.assert_allclose(out, [LO + 2e-4], rtol=0, atol=1e-15)
        self.assertGreaterEqual((out - 2e-4)[0], LO)
        self.assertEqual(x[0], X_EDGE)  # the input is never mutated

    def test_the_upper_edge_moves_one_step_in(self):
        out = self.nudge(np.array([-X_EDGE]), np.array([2e-4]),
                         np.array([LO]), np.array([HI]))
        np.testing.assert_allclose(out, [HI - 2e-4], rtol=0, atol=1e-15)

    def test_interior_points_are_untouched_and_a_copy_is_returned(self):
        x = np.array([0.3, -0.5, 2.0])
        out = self.nudge(x, np.full(3, 2e-4), np.array([LO, LO, 0.0]),
                         np.array([HI, HI, 2 * np.pi]))
        np.testing.assert_array_equal(out, x)
        self.assertIsNot(out, x)
        out[0] = 99.0
        self.assertEqual(x[0], 0.3)

    def test_infinite_bounds_are_untouched(self):
        x = np.array([-1e30, 1e30, 5.0])
        out = self.nudge(x, np.full(3, 1.0), np.full(3, -np.inf), np.full(3, np.inf))
        np.testing.assert_array_equal(out, x)

    def test_a_box_narrower_than_two_steps_centres_the_point(self):
        out = self.nudge(np.array([1e-5]), np.array([1e-3]),
                         np.array([0.0]), np.array([1e-4]))
        np.testing.assert_allclose(out, [5e-5], rtol=1e-15)

    def test_a_frozen_column_is_untouched(self):
        # step <= 0 freezes the column in information_matrix_from_ll: no
        # stencil there, nothing to keep inside
        out = self.nudge(np.array([X_EDGE, X_EDGE]), np.array([0.0, 2e-4]),
                         np.full(2, LO), np.full(2, HI))
        self.assertEqual(out[0], X_EDGE)
        self.assertGreater(out[1], X_EDGE)

    def test_rows_are_nudged_independently(self):
        x = np.array([[X_EDGE, 0.0], [0.0, -X_EDGE], [0.1, 0.2]])
        out = self.nudge(x, np.array([2e-4, 3e-4]), np.full(2, LO), np.full(2, HI))
        np.testing.assert_allclose(out[0], [LO + 2e-4, 0.0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(out[1], [0.0, HI - 3e-4], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(out[2], x[2])

    def test_the_stencil_survives_floating_point_rounding(self):
        # fl(fl(lo + s) - s) lands one ulp below lo for these steps (and
        # fl(fl(hi - s) + s) one ulp above hi): the stencil row computed the
        # way the builders compute it must still be inside the box
        s_lo, s_hi = 0.08374305481129224, 0.07242207701407438
        self.assertLess((LO + s_lo) - s_lo, LO)      # the trap is real
        self.assertGreater((HI - s_hi) + s_hi, HI)
        out = self.nudge(np.array([X_EDGE, -X_EDGE]), np.array([s_lo, s_hi]),
                         np.full(2, LO), np.full(2, HI))
        self.assertGreaterEqual(out[0] - s_lo, LO)
        self.assertLessEqual(out[0] + s_lo, HI)
        self.assertLessEqual(out[1] + s_hi, HI)
        self.assertGreaterEqual(out[1] - s_hi, LO)


# ---------------------------------------------------------------------------
# the likelihood route: eigen_table_from_ll / eigen_tables_from_ll_batch
# ---------------------------------------------------------------------------

#: a 3-column box with the job-751 cos column first
BOX_LO = np.array([LO, 0.0, 0.01])
BOX_HI = np.array([HI, 2 * np.pi, 100.0])
BOX_W = BOX_HI - BOX_LO


def _correlated_cov(seed=83):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((3, 3))
    s = np.diag(0.02 * BOX_W)  # widths far inside the box: the cap cannot bind
    return s @ (a @ a.T + 0.5 * np.eye(3)) @ s


def _box_ll(cov, outside, seen):
    """Gaussian lnL inside the box; ``outside(n_bad)`` values beyond it."""
    inv = np.linalg.inv(cov)
    mu = np.array([0.0, 3.0, 50.0])

    def call_ll(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        seen.append(x.copy())
        c = x - mu
        out = -0.5 * np.einsum("ni,ij,nj->n", c, inv, c)
        bad = ~np.all((x >= BOX_LO) & (x <= BOX_HI), axis=1)
        out[bad] = outside(int(bad.sum()))
        return out

    return call_ll


def _axis_aligned(axes):
    a = np.abs(np.asarray(axes))
    return np.allclose(np.sort(a, axis=-2)[..., :-1, :], 0.0, atol=1e-9)


class EigenTableFromLLBoundsTest(unittest.TestCase):
    X0 = np.array([X_EDGE, 3.0, 50.0])

    def _analytic_ok(self, axes, sigmas, cov):
        F = np.linalg.inv(cov)
        want = np.array([1.0 / np.sqrt(axes[:, k] @ F @ axes[:, k]) for k in range(3)])
        return np.allclose(sigmas, want, rtol=1e-4), want

    def test_nan_corners_old_path_kept_without_bounds_fixed_with_them(self):
        cov = _correlated_cov()
        nan = lambda n: np.nan  # noqa: E731 - the arccos NaN template

        # without bounds: the pre-fix path, identity rows + the infomat warning
        seen = []
        with self.assertLogs("lisatools.info_matrix_ll", level="WARNING") as cm:
            axes, sigmas = eigen_table_from_ll(_box_ll(cov, nan, seen), self.X0, BOX_W)
        self.assertTrue(any("non-finite" in m for m in cm.output), cm.output)
        self.assertFalse(_in_box(np.concatenate(seen), BOX_LO, BOX_HI))
        self.assertTrue(_axis_aligned(axes))   # no curvature information

        # with bounds: every corner inside, the analytic curvature recovered
        from lisatools.globalfit.moves.eigen_refresh import prior_box_bounds

        pri = ProbDistContainer({i: uniform_dist(BOX_LO[i], BOX_HI[i]) for i in range(3)})
        bounds = prior_box_bounds(pri, 3)
        seen = []
        with self.assertNoLogs("lisatools.info_matrix_ll", level="WARNING"), \
                self.assertNoLogs(eigen_refresh.logger, level="WARNING"):
            axes, sigmas = eigen_table_from_ll(
                _box_ll(cov, nan, seen), self.X0, BOX_W, bounds=bounds)
        self.assertTrue(_in_box(np.concatenate(seen), BOX_LO, BOX_HI))
        ok, want = self._analytic_ok(axes, sigmas, cov)
        self.assertTrue(ok, (sigmas, want))
        self.assertFalse(_axis_aligned(axes))

    def test_a_finite_no_signal_corner_is_silent_garbage_unless_bounded(self):
        # the EMRI scorer turns the NaN template into "no signal": a FINITE
        # lnL, so nothing warns and the table is silently wrong (job 751)
        cov = _correlated_cov()
        no_signal = lambda n: -50.0  # noqa: E731
        axes, sigmas = eigen_table_from_ll(_box_ll(cov, no_signal, []), self.X0, BOX_W)
        self.assertFalse(self._analytic_ok(axes, sigmas, cov)[0])

        bounds = (BOX_LO.copy(), BOX_HI.copy())
        axes, sigmas = eigen_table_from_ll(
            _box_ll(cov, no_signal, []), self.X0, BOX_W, bounds=bounds)
        ok, want = self._analytic_ok(axes, sigmas, cov)
        self.assertTrue(ok, (sigmas, want))

    def test_the_batch_nudges_each_row(self):
        cov = _correlated_cov()
        x0s = np.array([self.X0, [0.2, 3.0, 50.0], [-X_EDGE, 1.0, 20.0]])
        n = x0s.shape[0]
        seen = []
        base = _box_ll(cov, lambda k: np.nan, seen)
        with self.assertNoLogs("lisatools.info_matrix_ll", level="WARNING"):
            axes, sigmas = eigen_tables_from_ll_batch(
                base, x0s, BOX_W, bounds=(BOX_LO.copy(), BOX_HI.copy()))
        rows = np.concatenate(seen)
        self.assertTrue(_in_box(rows, BOX_LO, BOX_HI))
        # the central block (the first n rows) is the nudged points; the
        # interior row is exactly where it was
        np.testing.assert_array_equal(rows[1], x0s[1])
        np.testing.assert_allclose(rows[0, 0], LO + 1e-4 * BOX_W[0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(rows[2, 0], HI - 1e-4 * BOX_W[0], rtol=0, atol=1e-15)
        for k in range(n):
            ok, want = self._analytic_ok(axes[k], sigmas[k], cov)
            self.assertTrue(ok, (k, sigmas[k], want))


# ---------------------------------------------------------------------------
# the move: ll routes (walker_max / per_walker) record the nudged point
# ---------------------------------------------------------------------------

NT, NW, ND = 2, 4, 3


def _emri_like_priors():
    # label-keyed like the stock EMRI dict: (qS, qK, phiK)
    return ProbDistContainer({
        "qS": uniform_dist(LO, HI),
        "qK": uniform_dist(LO, HI),
        "phiK": uniform_dist(0.0, 2 * np.pi),
    })


MV_LO = np.array([LO, LO, 0.0])
MV_HI = np.array([HI, HI, 2 * np.pi])


def _ll_stub(scope, best=1):
    from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

    move = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name = "emri"
    move.ndim, move.ntemps, move.nwalkers = ND, NT, NW
    move.moves = [EigenAxisMove()]
    move.priors = {"emri": _emri_like_priors()}
    move.eigen_refresh_every = 10
    move.eigen_eps_rel = 1e-4
    move.eigen_table_scope = scope
    move.eigen_info = "ll"
    move._to_phys = lambda x: np.atleast_2d(np.asarray(x, dtype=float))
    inv = np.linalg.inv(np.diag([0.05, 0.05, 0.3]) ** 2)
    move.seen_rows = []

    def compute_like(x, data_index=None):
        x = np.atleast_2d(x)
        if x.shape[0] == NW:  # the max-lnL walker selection
            out = np.zeros(NW)
            out[best] = 5.0
            return out
        move.seen_rows.append(x.copy())
        c = x - np.array([0.0, 0.0, 3.0])
        out = -0.5 * np.einsum("ni,ij,nj->n", c, inv, c)
        out[~np.all((x >= MV_LO) & (x <= MV_HI), axis=1)] = np.nan
        return out

    move.compute_like = compute_like
    return move


def _work(best=1):
    rng = np.random.default_rng(89)
    coords = 0.3 * rng.standard_normal((NT, NW, 2, ND))
    coords[..., 2] += 3.0
    coords[0, best, 0, 1] = X_EDGE        # the cold max-lnL walker at the edge
    coords[1, 2, 0, 0] = -X_EDGE          # a hot point at the upper edge
    return types.SimpleNamespace(coords=coords)


class MoveLLRouteTest(unittest.TestCase):
    def _widths(self, move):
        return prior_box_widths(move.priors["emri"], ND)

    def test_walker_max_corners_stay_inside_and_the_nudged_point_is_recorded(self):
        from lisatools.globalfit.moves.eigen_refresh import nudge_inside

        move = _ll_stub("walker_max")
        work = _work()
        x0 = work.coords[0, 1, 0].copy()
        with mock.patch.object(eigen_refresh, "eigen_table_from_ll",
                               wraps=eigen_refresh.eigen_table_from_ll) as spy, \
                self.assertNoLogs("lisatools.info_matrix_ll", level="WARNING"):
            axes, sigmas = move._build_eigen_table(0, work)
        self.assertTrue(_in_box(np.concatenate(move.seen_rows), MV_LO, MV_HI))
        lo, hi = spy.call_args.kwargs["bounds"]
        np.testing.assert_array_equal(lo, MV_LO)
        np.testing.assert_array_equal(hi, MV_HI)
        want = nudge_inside(x0, 1e-4 * self._widths(move), MV_LO, MV_HI)
        np.testing.assert_array_equal(move._eigen_x0[0], want)
        self.assertGreater(move._eigen_x0[0][1], X_EDGE)
        self.assertTrue(np.all(np.isfinite(axes)) and np.all(np.isfinite(sigmas)))

    def test_per_walker_corners_stay_inside_and_the_nudged_points_are_recorded(self):
        from lisatools.globalfit.moves.eigen_refresh import nudge_inside

        move = _ll_stub("per_walker")
        work = _work()
        pts = work.coords[:NT, :, 0].copy()
        with self.assertNoLogs("lisatools.info_matrix_ll", level="WARNING"):
            axes, sigmas = move._build_eigen_table(0, work)
        self.assertTrue(_in_box(np.concatenate(move.seen_rows), MV_LO, MV_HI))
        rec = move._eigen_x0[0]
        self.assertEqual(rec.shape, (NT, NW, ND))
        want = nudge_inside(pts, 1e-4 * self._widths(move), MV_LO, MV_HI)
        np.testing.assert_array_equal(rec, want)
        self.assertGreater(rec[0, 1, 1], X_EDGE)
        self.assertLess(rec[1, 2, 0], -X_EDGE)
        self.assertEqual(axes.shape, (NT, NW, ND, ND))
        self.assertTrue(np.all(np.isfinite(sigmas)))


# ---------------------------------------------------------------------------
# the move: the Gram route (_gram_info)
# ---------------------------------------------------------------------------

_NPIX = (8, 10)
_RNG = np.random.default_rng(97)
_A = _RNG.uniform(0.5, 2.0, (ND,) + _NPIX)    # linear phase per column
_C = _RNG.uniform(0.5, 2.0, _NPIX)            # quadratic phase in the edge column


def _phase(x):
    return _A[0] * x[0] + _A[1] * x[1] + _A[2] * x[2] + _C * x[1] ** 2


def _template(x):
    return np.sin(_phase(x))[None]          # (1 channel, Nf, Nt)


def _jacobian(x):
    c = np.cos(_phase(x))
    return [c * _A[0], c * (_A[1] + 2.0 * _C * x[1]), c * _A[2]]


class _FlatAC:
    def __init__(self, box):
        from lisatools.sensitivity import SensitivityMatrix

        self.psd = SensitivityMatrix(box, np.ones((1,) + _NPIX))

    def _slice_to_template(self, h):
        return None, None, self.psd


def _gram_stub(nan_if=None):
    """A move whose ``_gram_templates`` is the quadratic-phase toy, NaN for
    any row outside the prior box (the cos -> arccos transform) or where
    ``nan_if(row)`` holds."""
    from lisatools.domains import WDMSettings
    from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

    box = WDMSettings(Nf=_NPIX[0], Nt=_NPIX[1], dt=2.0, force_backend="cpu")
    move = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    move.branch_name = "emri"
    move.ndim, move.ntemps, move.nwalkers = ND, NT, NW
    move.use_gpu = False
    move.priors = {"emri": _emri_like_priors()}
    move.eigen_eps_rel = 1e-4
    move.eigen_gram_eps_rel = 1e-4
    move.eigen_gram_target = 1e-3
    move.waveform_like_kwargs = {}
    move._to_phys = lambda x: np.atleast_2d(np.asarray(x, dtype=float))
    ac = _FlatAC(box)
    acs_arr = np.empty(NW, dtype=object)
    acs_arr[:] = [ac] * NW
    move.acs = types.SimpleNamespace(acs=acs_arr)
    move.gram_rows = []

    def gram_templates(rows, walker):
        rows = np.atleast_2d(rows)
        move.gram_rows.append(rows.copy())
        out = []
        for r in rows:
            bad = not _in_box(r, MV_LO, MV_HI) or (nan_if is not None and nan_if(r))
            out.append(np.full((1,) + _NPIX, np.nan) if bad else _template(r))
        return np.stack(out), box

    move._gram_templates = gram_templates
    return move, ac


def _gram_truth(ac, x):
    from lisatools.diagnostic import inner_product
    from lisatools.domains import WDMSignal

    box = ac.psd.basis_settings
    J = [WDMSignal(j[None], box) for j in _jacobian(x)]
    return np.array([[float(inner_product(a, b, psd=ac.psd)) for b in J] for a in J])


class GramAtTheBoundTest(unittest.TestCase):
    X0 = np.array([0.3, X_EDGE, 2.0])

    def test_the_fake_template_is_nan_outside_the_box(self):
        # paired control: the fixture really does punish a leaking stencil
        move, _ = _gram_stub()
        arr, _ = move._gram_templates(np.array([[0.3, X_EDGE - 2e-4, 2.0],
                                                [0.3, X_EDGE, 2.0]]), 0)
        self.assertTrue(np.all(np.isnan(arr[0])))
        self.assertTrue(np.all(np.isfinite(arr[1])))

    def test_gram_at_the_edge_is_finite_and_every_stencil_row_stayed_inside(self):
        from lisatools.globalfit.moves.eigen_refresh import nudge_inside

        move, ac = _gram_stub()
        widths = prior_box_widths(move.priors["emri"], ND)
        G, steps = move._gram_info(self.X0.copy(), 0, widths, return_steps=True)
        self.assertTrue(np.all(np.isfinite(G)), G)
        self.assertTrue(_in_box(np.concatenate(move.gram_rows), MV_LO, MV_HI))
        # the matrix is the Gram at the nudged centre
        xc = nudge_inside(self.X0, steps, MV_LO, MV_HI)
        self.assertGreater(xc[1], X_EDGE)
        T = _gram_truth(ac, xc)
        d = np.sqrt(np.diag(T))
        self.assertLess(np.max(np.abs(G - T) / np.outer(d, d)), 1e-4)

    def test_the_gram_table_builds_and_records_the_nudged_centre(self):
        # pre-fix: _table_from_info raised "non-finite information matrix"
        from lisatools.globalfit.moves.eigen_refresh import nudge_inside

        move, _ = _gram_stub()
        widths = prior_box_widths(move.priors["emri"], ND)
        cold = np.tile(self.X0, (NW, 1))
        with mock.patch.object(move, "_eigen_best_walker", return_value=(0, cold)):
            axes, sigmas = move._build_eigen_table_gram(3, None, widths)
        self.assertTrue(np.all(np.isfinite(axes)) and np.all(sigmas > 0))
        _, steps = move._gram_info(self.X0.copy(), 0, widths, return_steps=True)
        np.testing.assert_array_equal(
            move._eigen_x0[3], nudge_inside(self.X0, steps, MV_LO, MV_HI))

    def test_an_interior_nan_names_the_offending_column(self):
        x0 = np.array([0.3, 0.2, 2.0])
        # column 2's +step row breaks for a reason that is not the box
        move, _ = _gram_stub(nan_if=lambda r: r[2] > x0[2])
        widths = prior_box_widths(move.priors["emri"], ND)
        with self.assertRaisesRegex(ValueError, r"non-finite information matrix") as cm:
            move._gram_info(x0.copy(), 0, widths)
        msg = str(cm.exception)
        self.assertIn("columns [2]", msg)

        # ... and it reaches the refresh's log line before the ll fallback
        move.eigen_info = "gram"
        move.eigen_table_scope = "walker_max"
        cold = np.tile(x0, (NW, 1))
        with mock.patch.object(move, "_eigen_best_walker", return_value=(0, cold)), \
                mock.patch.object(eigen_refresh, "eigen_table_from_ll",
                                  return_value=(np.eye(ND), np.ones(ND))) as ll_route, \
                self.assertLogs(eigen_refresh.logger, level="WARNING") as logs:
            move._build_eigen_table(3, None)
        ll_route.assert_called_once()
        self.assertTrue(any("Gram build failed" in m and "columns [2]" in m
                            for m in logs.output), logs.output)


if __name__ == "__main__":
    unittest.main()
