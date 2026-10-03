"""EMRIDirectLikeMove: direct-template scoring against per-walker residuals AND
per-walker PSDs equals the container path; the fill runs through the containers' installed
generator; refusals, fallbacks (on the move's production generator), chunking, routing and
the tolerance checks.

The direct adapter is faked by the containers' own toy generator (the MBH toy of
``tests/test_mbh_batched_move.py``), so "direct == production" holds exactly and
every difference these tests see comes from the move."""
from __future__ import annotations

import logging
import os
import unittest
from unittest import mock

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.domains import WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix

try:
    from tests.test_mbh_batched_move import BASE_ROW, LAYER, NT, _slow_gen, _toy_wdm
except ImportError:  # pragma: no cover - run from tests/
    from test_mbh_batched_move import BASE_ROW, LAYER, NT, _slow_gen, _toy_wdm

NWALKERS, NTEMPS = 3, 2
NDIM = BASE_ROW.size
_MOD = "lisatools.globalfit.moves.emridirectmove"


class _FakeDirect:
    """The adapter contract: ``templates(rows, **kw) -> (arr (n, 3, Nf_a, Nt_a), ok)``.

    ``scale`` multiplies every template (a direct/production template difference);
    ``refuse_dist`` marks rows whose column 4 equals it as FEW-refused."""

    def __init__(self, wdm, scale=1.0):
        self.domain_settings = wdm
        self.scale = float(scale)
        self.fail_with = None
        self.refuse_dist = None
        self.calls = []

    def templates(self, rows, **kwargs):
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        self.calls.append((rows.copy(), dict(kwargs)))
        if self.fail_with is not None:
            raise self.fail_with
        arr = np.stack([np.asarray(_slow_gen(*r).arr) for r in rows]) * self.scale
        ok = np.ones(rows.shape[0], dtype=bool)
        if self.refuse_dist is not None:
            ok = rows[:, 4] != self.refuse_dist
            arr[~ok] = 0.0
        return arr, ok


def _build(scale=1.0):
    wdm = _toy_wdm(0.0)
    _slow_gen.wdm = wdm
    _slow_gen.t0_stock = 0.0
    _slow_gen.data_t0 = 0.0
    rng = np.random.default_rng(7)
    acs_list = []
    models = ["scirdv1", "mrdv1", "scirdv1"]      # walker 1 has a DIFFERENT PSD
    for w in range(NWALKERS):
        noise = 1e-23 * rng.normal(size=(3, wdm.Nf_active, NT))
        ac = AnalysisContainer(WDMSignal(noise, wdm), XYZ2SensitivityMatrix(wdm, model=models[w]))
        ac.signal_gen = {"emri": _slow_gen}
        acs_list.append(ac)
    return AnalysisContainerArray(acs_list), _FakeDirect(wdm, scale=scale), wdm


def _build_move(acs, direct, batch_max_size=2, gen_kwargs=None, ntemps=NTEMPS):
    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, uniform_dist

    from lisatools.globalfit.moves import EMRIDirectLikeMove

    betas = 1 / 1.2 ** np.arange(ntemps)
    priors = {"emri": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(NDIM)})}
    move = EMRIDirectLikeMove(
        "emri", (ntemps, NWALKERS, 1, NDIM), None, dict(gen_kwargs or {}), {}, acs, 1, None,
        priors, [(StretchMove(), 1.0)], betas_all=np.tile(betas, (1, 1)),
        direct_gen=direct, batch_max_size=batch_max_size, name="emri direct test",
    )
    move._current_leaf = 0
    return move


def _cold_rows(seed=3):
    rng = np.random.default_rng(seed)
    rows = np.tile(BASE_ROW, (NWALKERS, 1))
    rows[:, 4] *= rng.uniform(0.8, 1.2, NWALKERS)
    rows[:, 10] += rng.uniform(-0.2, 0.2, NWALKERS) * LAYER
    return rows


def _proposal_rows(n, seed=11):
    rng = np.random.default_rng(seed)
    rows = np.tile(BASE_ROW, (n, 1))
    rows[:, 4] *= rng.uniform(0.7, 1.3, n)
    rows[:, 10] += rng.uniform(-0.3, 0.3, n) * LAYER
    rows[:, 5] += rng.uniform(-1, 1, n)
    return rows


class _Armed(unittest.TestCase):
    SCALE = 1.0

    def setUp(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("EMRI_") and k not in ("ADDREMOVE_CHECK_LL",)}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.acs, self.direct, self.wdm = _build(self.SCALE)
        self.move = _build_move(self.acs, self.direct)
        self.cold = _cold_rows()
        self.move.remove_cold_chain_sources(self.cold)     # expose (production fill)
        self.move.setup_likelihood_here(self.cold)


class EMRIDirectParityTest(_Armed):
    def test_direct_matches_container_path_per_walker_psd(self):
        rows = _proposal_rows(6)
        idx = np.array([0, 1, 2, 1, 0, 1])
        fast = self.move.compute_like(rows, idx)
        slow = self.move.compute_acs_like(rows, idx)
        print(f"[emri direct parity] max |fast-slow| = {np.abs(fast - slow).max():.3e} "
              f"on lnL ~ {np.abs(slow).max():.3e}")
        np.testing.assert_allclose(fast, slow, rtol=1e-11, atol=0)
        self.assertEqual(self.move.n_batch_fallbacks, 0)
        self.assertTrue(np.all(np.isfinite(self.move._last_d_h)))
        self.assertTrue(np.all(self.move._last_h_h > 0))
        # the rows really went through the direct adapter, in chunks of 2
        self.assertEqual([c[0].shape[0] for c in self.direct.calls], [2, 2, 2])
        # walker 1's PSD and residual differ: the same row scores differently
        same = np.tile(rows[:1], (2, 1))
        vals = self.move.compute_like(same, np.array([0, 1]))
        self.assertNotAlmostEqual(vals[0] - self.move._exposed_offset[0],
                                  vals[1] - self.move._exposed_offset[1], places=3)

    def test_fill_stays_on_the_production_generator(self):
        """Expose/fold build their templates through the containers' installed
        generator; the direct adapter is never asked for a fill template."""
        acs, direct, _ = _build()
        move = _build_move(acs, direct)
        before = [np.array(ac.data.arr, copy=True) for ac in acs.acs.flatten()]
        cold = _cold_rows()
        move.remove_cold_chain_sources(cold)
        move.add_back_in_cold_chain_sources(cold)
        self.assertEqual(direct.calls, [])
        for ac, b in zip(acs.acs.flatten(), before):
            np.testing.assert_allclose(np.asarray(ac.data.arr), b, rtol=0, atol=1e-12 * np.abs(b).max())

    def test_chunk_size_does_not_change_results(self):
        rows = _proposal_rows(5)
        idx = np.array([0, 1, 2, 0, 1])
        a = self.move.compute_like(rows, idx)
        big = _build_move(self.acs, self.direct, batch_max_size=16)
        big._exposed_offset = self.move._exposed_offset
        b = big.compute_like(rows, idx)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_inner_products_batched_per_walker_not_per_row(self):
        import lisatools.globalfit.moves.emridirectmove as mod

        big = _build_move(self.acs, self.direct, batch_max_size=4)
        big._exposed_offset = self.move._exposed_offset
        rows = _proposal_rows(4)
        idx = np.array([0, 1, 0, 1])
        with mock.patch.object(mod, "inner_product", wraps=mod.inner_product) as spy:
            out = big.compute_like(rows, idx)
        self.assertLessEqual(spy.call_count, 2 * len(np.unique(idx)))
        np.testing.assert_allclose(out, self.move.compute_like(rows, idx), rtol=1e-12, atol=0)

    def test_generator_kwargs_reach_the_adapter(self):
        move = _build_move(self.acs, self.direct, gen_kwargs={"mode_selection_threshold": 1e-4})
        move._exposed_offset = self.move._exposed_offset
        move.compute_like(_proposal_rows(1), np.array([0]))
        self.assertEqual(self.direct.calls[-1][1], {"mode_selection_threshold": 1e-4})

    def test_non_finite_row_gets_sentinel_and_is_never_generated(self):
        rows = _proposal_rows(2)
        rows[1, 0] = np.nan
        out = self.move.compute_like(rows, np.array([0, 1]))
        self.assertTrue(np.isfinite(out[0]))
        self.assertEqual(out[1], -1e300)
        self.assertEqual(sum(c[0].shape[0] for c in self.direct.calls), 1)

    def test_refused_row_scores_the_floor_others_unchanged(self):
        rows = _proposal_rows(4)
        idx = np.array([0, 1, 2, 0])
        slow = self.move.compute_acs_like(rows, idx)
        self.direct.refuse_dist = rows[1, 4]
        out = self.move.compute_like(rows, idx)
        self.assertEqual(out[1], -1e300)
        keep = np.array([0, 2, 3])
        np.testing.assert_allclose(out[keep], slow[keep], rtol=1e-11, atol=0)
        self.assertTrue(np.isnan(self.move._last_d_h[1]))
        self.assertEqual(self.move._stats["refused"], 1)
        self.assertEqual(self.move.n_batch_fallbacks, 0)

    def test_failed_batch_falls_back_to_container_path_loudly(self):
        rows = _proposal_rows(3)
        idx = np.array([0, 1, 2])
        slow = self.move.compute_acs_like(rows, idx)
        self.direct.fail_with = RuntimeError("cupy OutOfMemoryError stand-in")
        with self.assertLogs(_MOD, logging.WARNING) as cm:
            out = self.move.compute_like(rows, idx)
        np.testing.assert_allclose(out, slow, rtol=0, atol=0)
        self.assertEqual(self.move.n_batch_fallbacks, 2)        # one per chunk of 2
        self.assertEqual(len(cm.output), 1)                      # warned once per leaf
        self.assertIn("OutOfMemoryError", cm.output[0])
        self.assertIsInstance(self.move.last_batch_error, RuntimeError)

    def test_fallback_scores_with_the_production_generator(self):
        """In the fit the containers' installed generator IS the direct template (the one that
        just failed); the fallback must use the move's own production ``waveform_gen``."""
        from lisatools.domains import WDMSignal

        def production(*p, **kw):                      # differs from the installed generator
            h = _slow_gen(*p, **kw)
            return WDMSignal(1.01 * np.asarray(h.arr), _slow_gen.wdm)

        move = _build_move(self.acs, self.direct)
        move.waveform_gen = production
        move.remove_cold_chain_sources(self.cold)
        move.setup_likelihood_here(self.cold)
        rows = _proposal_rows(3)
        idx = np.array([0, 1, 2])
        want = move.compute_acs_like(rows, idx, signal_gen=production)
        installed = move.compute_acs_like(rows, idx)
        self.assertGreater(np.abs(want - installed).max(), 1e-6 * np.abs(installed).max())
        self.direct.fail_with = RuntimeError("kernel missing")
        with self.assertLogs(_MOD, logging.WARNING):
            out = move.compute_like(rows, idx)
        np.testing.assert_allclose(out, want, rtol=1e-12, atol=0)
        move.add_back_in_cold_chain_sources(self.cold)

    def test_compute_like_requires_armed_offset(self):
        move = _build_move(self.acs, self.direct)
        with self.assertRaises(RuntimeError):
            move.compute_like(_proposal_rows(1), np.array([0]))

    def test_record_inner_products_from_the_direct_scorer(self):
        from types import SimpleNamespace

        sub = SimpleNamespace(d_h=np.full((NWALKERS, 1), np.nan), h_h=np.full((NWALKERS, 1), np.nan))
        state = SimpleNamespace(sub_states={"emri": sub})
        self.assertTrue(self.move.record_inner_products)       # default ON for this move
        n_calls = len(self.direct.calls)
        with mock.patch.object(type(self.move), "compute_acs_like",
                               side_effect=AssertionError("container path used")):
            self.move._record_leaf_inner_products(state, self.cold, 0)
        self.assertGreater(len(self.direct.calls), n_calls)
        self.assertTrue(np.all(np.isfinite(sub.d_h)) and np.all(sub.h_h > 0))


class EMRIDirectBoxTest(unittest.TestCase):
    def test_active_box_mismatch_raises_not_falls_back(self):
        from lisatools.domains import WDMSettings

        acs, direct, wdm = _build()
        direct.domain_settings = WDMSettings(wdm.Nf, wdm.Nt, wdm.data_dt, min_freq=6e-3,
                                             max_freq=2e-2, force_backend="cpu")
        move = _build_move(acs, direct)
        cold = _cold_rows()
        move.remove_cold_chain_sources(cold)
        move.setup_likelihood_here(cold)
        with self.assertRaisesRegex(ValueError, "active box"):
            move.compute_like(_proposal_rows(1), np.array([0]))
        self.assertEqual(move.n_batch_fallbacks, 0)

    def test_ctor_guards(self):
        from eryn.moves import StretchMove
        from eryn.prior import ProbDistContainer, uniform_dist

        from lisatools.globalfit.moves import EMRIDirectLikeMove

        acs, direct, _ = _build()
        priors = {"emri": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(NDIM)})}
        args = ("emri", (NTEMPS, NWALKERS, 1, NDIM), None, {}, {}, acs, 1, None, priors,
                [(StretchMove(), 1.0)])
        with self.assertRaisesRegex(ValueError, "direct_gen"):
            EMRIDirectLikeMove(*args)
        with self.assertRaisesRegex(ValueError, "DCGA"):
            EMRIDirectLikeMove(*args, direct_gen=direct, dcga=object())

    def test_non_wdm_domain_refused(self):
        from eryn.moves import StretchMove
        from eryn.prior import ProbDistContainer, uniform_dist

        from lisatools.globalfit.moves import EMRIDirectLikeMove

        acs, direct, _ = _build()
        fake = mock.MagicMock()
        fake.acs.flatten.return_value = [mock.MagicMock()]
        priors = {"emri": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(NDIM)})}
        with mock.patch.object(EMRIDirectLikeMove.__mro__[1], "__init__", lambda self, *a, **k: None):
            move = EMRIDirectLikeMove.__new__(EMRIDirectLikeMove)
            move.acs = fake
            with self.assertRaisesRegex(ValueError, "WDM run domain"):
                EMRIDirectLikeMove.__init__(
                    move, "emri", (NTEMPS, NWALKERS, 1, NDIM), None, {}, {}, fake, 1, None,
                    priors, [(StretchMove(), 1.0)], direct_gen=direct)


class EMRIDirectChecksTest(_Armed):
    """The cross-check and the expose invariant gate the COLD rung on a TOLERANCE."""

    def _prev(self, move=None):
        move = move or self.move
        rows = _proposal_rows(NWALKERS * NTEMPS)
        idx = np.tile(np.arange(NWALKERS), NTEMPS)
        return rows, idx, move.compute_like(rows, idx).reshape(NTEMPS, NWALKERS)

    def test_defaults_tolerance_one_nat_and_every_10th_visit(self):
        from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

        self.assertEqual((self.move.check_ll_tol, self.move.check_ll_mm), (1.0, 3e-4))
        self.assertEqual(self.move.check_ll_every, 10)
        with mock.patch.dict(os.environ, {"EMRI_CHECK_LL_TOL": "0.25", "EMRI_CHECK_LL_EVERY": "3",
                                          "EMRI_CHECK_LL_MM": "0"}):
            m = _build_move(self.acs, self.direct)
        self.assertEqual((m.check_ll_tol, m.check_ll_every, m.check_ll_mm), (0.25, 3, 0.0))
        self.assertEqual(ResidualAddOneRemoveOneMove._check_ll_every_default, "1")

    def test_verify_prev_logl_silent_when_templates_agree(self):
        rows, idx, prev = self._prev()
        with self.assertNoLogs(_MOD, level="WARNING"):
            self.move._verify_prev_logl(prev, rows, idx, 0)

    def test_verify_prev_logl_gates_on_the_cold_rung_only(self):
        rows, idx, prev = self._prev()
        self.move.check_ll_mm = 0.0            # the plain 1-nat tolerance
        hot_bad = prev.copy()
        hot_bad[1:] += 7.0
        self.move.check_ll_mode = "strict"
        with self.assertNoLogs(_MOD, level="WARNING"):
            self.move._verify_prev_logl(hot_bad, rows, idx, 0)
        within = hot_bad.copy()
        within[0, 1] += 0.9                    # below the 1-nat tolerance
        self.move._verify_prev_logl(within, rows, idx, 0)
        both_bad = hot_bad.copy()
        both_bad[0, 1] += 3.0
        with self.assertRaises(ValueError) as cm:
            self.move._verify_prev_logl(both_bad, rows, idx, 0)
        msg = str(cm.exception)
        self.assertIn("COLD", msg)
        self.assertRegex(msg, r"max\|diff\|=3\.0\d*e\+00")
        self.assertRegex(msg, r"hot rungs, not gating: max\|diff\|=7\.0\d*e\+00")

    def test_entry_invariant_with_the_template_tolerance(self):
        # cold_ref = the pre-expose full lnL; exact templates reproduce it
        acs, direct, _ = _build()
        move = _build_move(acs, direct)
        cold = _cold_rows()
        cold_ref = np.asarray(acs.likelihood())      # residual = noise - h(cold): pre-expose
        move.remove_cold_chain_sources(cold)
        move.setup_likelihood_here(cold)
        prev = move.compute_like(cold, np.arange(NWALKERS)).reshape(1, NWALKERS)
        np.testing.assert_allclose(prev[0], cold_ref, rtol=1e-11, atol=0)
        move.check_ll_mode = "strict"
        move.check_ll_mm = 0.0                 # the plain 1-nat tolerance
        move._verify_entry_vs_acs(prev, cold_ref, 0)                  # silent
        move._verify_entry_vs_acs(prev + 0.9, cold_ref, 0)            # inside the tolerance
        with self.assertRaisesRegex(ValueError, "EXPOSE INVARIANT"):
            move._verify_entry_vs_acs(prev + 2.0, cold_ref, 0)


class EMRIDirectToleranceScalingTest(_Armed):
    """The tolerance grows with each point's <h|h>: a 3-nat gap at SNR 70 is template
    accuracy, the same gap at SNR 10 is not (paired with the plain tolerance)."""

    def _gate(self, mm, diff, hh):
        self.move.check_ll_tol, self.move.check_ll_mm = 1.0, mm
        self.move.check_ll_mode = "strict"
        self.move._last_h_h = np.asarray(hh, dtype=float)
        prev = np.zeros((1, len(diff)))
        self.move._verify_entry_vs_acs(prev, -np.asarray(diff, dtype=float), 0)

    def test_point_tolerance_values(self):
        self.move.check_ll_tol, self.move.check_ll_mm = 1.0, 1e-4
        self.move._last_h_h = np.array([4900.0, 100.0, np.nan, -5.0])
        tol = self.move._point_tolerance((1, 4))[0]
        np.testing.assert_allclose(tol[:2], [1 + 0.49 + 3 * np.sqrt(0.98), 1 + 0.01 + 3 * np.sqrt(0.02)])
        np.testing.assert_allclose(tol[2:], [1.0, 1.0])          # no usable <h|h>: the plain tolerance
        self.move._last_h_h = np.ones(7)                          # another call's rows: never misapplied
        np.testing.assert_allclose(self.move._point_tolerance((1, 4)), 1.0)

    def test_high_snr_gap_passes_low_snr_gap_raises(self):
        self._gate(1e-4, [3.0, 1.2], [4900.0, 100.0])            # inside 4.46 and 1.43
        with self.assertRaisesRegex(ValueError, "EXPOSE INVARIANT"):
            self._gate(1e-4, [3.0, 1.6], [4900.0, 100.0])        # 1.6 > 1.43 at SNR 10
        with self.assertRaisesRegex(ValueError, "EXPOSE INVARIANT"):
            self._gate(0.0, [3.0, 1.2], [4900.0, 100.0])         # plain 1-nat tolerance


class EMRIDirectMismatchedTemplateTest(_Armed):
    """A direct template 2% louder than production: the move scores the DIRECT
    likelihood (consistent with itself), and the cross-check reports the gap."""

    SCALE = 1.02

    def test_scores_the_direct_template_and_the_check_sees_it(self):
        rows = _proposal_rows(NWALKERS * NTEMPS)
        idx = np.tile(np.arange(NWALKERS), NTEMPS)
        fast = self.move.compute_like(rows, idx)
        slow = self.move.compute_acs_like(rows, idx)
        self.assertGreater(np.abs(fast - slow).max(), 1.0)
        # == the container path's likelihood of the SCALED template (exposed residual)
        wdm = self.wdm

        def scaled(*p, **kw):
            return WDMSignal(1.02 * np.asarray(_slow_gen(*p).arr), wdm)

        want = self.move.compute_acs_like(rows, idx, signal_gen=scaled)
        np.testing.assert_allclose(fast, want, rtol=1e-11, atol=0)
        self.move.check_ll_mode = "strict"
        self.move.check_ll_mm = 0.0
        with self.assertRaisesRegex(ValueError, "disagree beyond the template tolerance"):
            self.move._verify_prev_logl(fast.reshape(NTEMPS, NWALKERS), rows, idx, 0)


class EMRIDirectDevicePinningTest(unittest.TestCase):
    """Multi-GPU: every shard's rows are generated under THAT shard's cupy device."""

    def test_each_shard_generates_under_its_device(self):
        from lisatools.analysiscontainer import AnalysisContainerArray as ACA

        try:
            from tests._multishard import RecordingXp
        except ImportError:  # pragma: no cover - run from tests/
            from _multishard import RecordingXp

        acs, direct, _ = _build()
        move = _build_move(acs, direct, batch_max_size=2)
        rx = RecordingXp()
        owner = {0: 1, 1: 2, 2: 1}

        def fake_split(idx):
            idx = np.asarray(idx).reshape(-1)
            out = []
            for dev in (1, 2):
                pos = np.where([owner[int(w)] == dev for w in idx])[0]
                if pos.size:
                    out.append((dev, pos))
            return out

        seen = []
        real = direct.templates

        def recording(rows, **kw):
            seen.append((rx.current_device, [int(round(v)) for v in np.atleast_2d(rows)[:, 1]]))
            return real(rows, **kw)

        direct.templates = recording
        cold = _cold_rows()
        rows = np.tile(BASE_ROW, (6, 1))
        idx = np.array([0, 1, 2, 1, 0, 2])
        rows[:, 1] = idx                       # column 1 is unused by the toy: tags the walker
        with mock.patch.object(ACA, "xp", new=property(lambda self: rx)), \
                mock.patch.object(move, "_split_rows", fake_split):
            move.remove_cold_chain_sources(cold)
            move.setup_likelihood_here(cold)
            move.compute_like(rows, idx)
        self.assertTrue(seen)
        for dev, walkers in seen:
            self.assertEqual({owner[w] for w in walkers}, {dev}, seen)
        self.assertEqual({d for d, _ in seen}, {1, 2})


if __name__ == "__main__":
    unittest.main()
