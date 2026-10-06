# tests/test_sobbh_lookup_move.py
"""SOBBHChunkedLikeMove with the LOOKUP comp: fast-vs-slow parity through the move's own seams.

Same toy choreography as tests/test_sobbh_chunked_move.py (slow side = the stock
``get_sobbh_tdionfly_gen`` + ``SOBBHTDIonFlyWaveWrap`` + the container likelihood), with
``chunked_comp=SOBBHLookupComputations`` on a 3600-s-layer grid (Nf=180, dt=20) and a tiny
table built at (Nf=64, dt=56.25) -- the table-portability property in use.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

from lisatools.utils.constants import YRSID_SI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table  # noqa: E402

DT = 20.0
NF, NT = 180, 256
NOBS = NF * NT
T_START = int(0.5 * YRSID_SI / DT) * DT
F_LOW = 6.0e-3
NWALKERS = 3
NTEMPS = 2

#: stock SOBBH waveform basis (what compute_like receives):
#: (m1, m2, s1, s2, dist[Gpc], inc, f_low, lam, beta, psi, phi0)
#: dist=0.05 Gpc -> SNR ~ 3 (SNR^2 ~ 11) against walker 0's data slab, so the parity
#: suite actually has power: a sign-flipped/doubled/decorrelated template would fail it.
REF_STOCK = np.array(
    [60.0, 55.0, 0.1, 0.2, 0.05, np.arccos(0.3), F_LOW, 3.1, np.arcsin(0.2), 0.7, 1.1]
)


def _build_toy(tmpdir):
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.domains import TDSettings, WDMSettings, WDMSignal
    from lisatools.globalfit.stock.erebor.wrappers import SOBBHTDIonFlyWaveWrap
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.sources.sobbh.response import get_sobbh_tdionfly_gen
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

    backend = "cpu"
    orbits = EqualArmlengthOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    td_set = TDSettings(NOBS, DT, force_backend=backend)
    wdm = WDMSettings(
        NF,
        NT,
        DT,
        t0=T_START,
        min_freq=2e-3,
        max_freq=2e-2,
        is_complex=False,
        force_backend=backend,
    )
    gen = get_sobbh_tdionfly_gen(
        Tobs=NOBS * DT,
        dt=DT,
        t_start=T_START,
        tdi_config=tdi_config,
        reference_time=float(T_START),
        orbits=orbits,
        n_grid=1024,
        buffer_time=5000.0,
        force_backend=backend,
    )
    t_arr = np.arange(NOBS) * DT + T_START
    wrap = SOBBHTDIonFlyWaveWrap(gen, t_arr, td_set, wdm, td_window=None, nchannels=3)

    def sobbh_gen(*params, apply_transform=False, leaf_inds=None, **kwargs):
        return wrap(*params)

    inj = wrap(*REF_STOCK)
    acs_list = []
    for w in range(NWALKERS):
        # distinct per-walker data slabs: w=0 holds exactly the injection (template == data ->
        # SNR^2 of the REF row is directly readable from acs.likelihood()); w=1,2 are scaled
        # copies so routing/per-walker scoring is actually observable (identical slabs would
        # make a cross-walker indexing bug invisible).
        ac = AnalysisContainer(
            WDMSignal(np.array(np.asarray(inj.arr), copy=True) * (1.0 + 0.5 * w), wdm),
            XYZ2SensitivityMatrix(wdm, model="scirdv1"),
        )
        ac.signal_gen = {"sobbh": sobbh_gen}
        acs_list.append(ac)
    acs = AnalysisContainerArray(acs_list)

    _, table = build_tiny_table(tmpdir)
    comp = SOBBHLookupComputations(
        wdm,
        float(T_START),
        table,
        orbits=orbits,
        tdi_config="2nd generation",
        tdi_type="XYZ",
        n_grid=1024,
        buffer_time=5000.0,
        # production defaults: 12-h response grid, the table's B-spline, the fused kernel
        # when the backend module has it (else the Python lookup)
        num_m_layers=2,
        row_batch=4,
        force_backend=backend,
        d_d=0.0,
    )
    return acs, sobbh_gen, comp, wdm, td_set


def _build_move(acs, comp):
    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, uniform_dist

    from lisatools.globalfit.moves import SOBBHChunkedLikeMove

    betas = 1 / 1.2 ** np.arange(NTEMPS)
    priors = {"sobbh": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(11)})}
    return SOBBHChunkedLikeMove(
        "sobbh",
        (NTEMPS, NWALKERS, 1, 11),
        None,
        {},
        {},
        acs,
        1,
        None,
        priors,
        [(StretchMove(), 1.0)],
        betas_all=np.tile(betas, (1, 1)),
        chunked_comp=comp,
        m_band_half_width=2,
        name="sobbh lookup test",
    )


class SOBBHLookupParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        try:
            cls.acs, cls.gen, cls.comp, cls.wdm, cls.td_set = _build_toy(cls.tmp.name)
        except Exception as exc:  # missing compiled deps etc.
            raise unittest.SkipTest(f"toy setup unavailable: {exc}")
        cls.move = _build_move(cls.acs, cls.comp)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _rows(self, n_pert=8, seed=5):
        rng = np.random.default_rng(seed)
        rows = np.tile(REF_STOCK, (n_pert + 1, 1))
        layer_df = float(self.wdm.layer_df)
        rows[1:, 6] += rng.uniform(-0.2, 0.2, n_pert) * layer_df
        rows[1:, 10] = rng.uniform(0, 2 * np.pi, n_pert)
        rows[1:, 5] = np.arccos(rng.uniform(-0.9, 0.9, n_pert))
        rows[1:, 4] *= rng.uniform(0.7, 1.5, n_pert)
        idx = np.arange(rows.shape[0], dtype=np.int32) % NWALKERS
        return rows, idx

    def test_fast_vs_slow_parity_through_move(self):
        move = self.move
        move.setup_likelihood_here(None)
        rows, idx = self._rows()
        fast = move.compute_like(rows, idx)
        slow = move.compute_acs_like(rows, idx, **move.waveform_like_kwargs)
        self.assertTrue(np.all(np.isfinite(fast)))
        diff = np.abs(fast - slow)
        self.assertLess(
            float(diff.max()),
            move.check_ll_tol,
            msg=f"lookup vs slow lnL diff {diff.max():.3e} exceeds tol "
            f"{move.check_ll_tol} (median {np.median(diff):.3e})",
        )
        # relative bound: the suite must have POWER at the toy's SNR, not just pass an
        # absolute tolerance that a sign-flipped/doubled/decorrelated template could also
        # pass. row 0 is the unperturbed REF row scored against data_index=0 (walker 0 holds
        # exactly the injection), so slow[0] - offset[0] = 0.5 * <h|h> = 0.5 * SNR^2.
        snr2 = 2.0 * float(slow[0] - move._exposed_offset[idx[0]])
        self.assertLess(
            float(diff.max()),
            0.05 * snr2,
            msg=f"lookup vs slow lnL diff {diff.max():.3e} exceeds 5% of SNR^2={snr2:.3e}",
        )
        move.check_ll_mode = "strict"
        move._verify_prev_logl(fast.reshape(1, -1), rows, idx, leaf=0)
        self.assertIn("launch", self.comp.last_call_spans)
        self.assertEqual(self.comp.last_call_spans["num_bin"], rows.shape[0])

    def test_parity_control_no_parity_turn(self):
        # paired negative control for the relative-bound assertion above: with the lookup
        # table's parity-turn step disabled, the template is wrong enough (sign/parity error)
        # that it must FAIL the same relative bound the real path comfortably passes.
        move = self.move
        move.setup_likelihood_here(None)
        rows, idx = self._rows()
        slow = move.compute_acs_like(rows, idx, **move.waveform_like_kwargs)
        snr2 = 2.0 * float(slow[0] - move._exposed_offset[idx[0]])
        # the switch lives on the Python evaluator: route this call through the Python lookup
        # (the fused kernel always applies the quarter turn; its own controls and mutations
        # are in tests/test_sobbh_lookup_kernel.py)
        uses_kernel = self.comp.uses_kernel
        self.comp.uses_kernel = False
        self.comp.direct.ev.basis_cycle = "no_parity_turn"
        try:
            fast = move.compute_like(rows, idx)
        finally:
            self.comp.direct.ev.basis_cycle = "quarter_turn"
            self.comp.uses_kernel = uses_kernel
        diff = np.abs(fast - slow)
        self.assertGreater(
            float(diff.max()),
            0.05 * snr2,
            msg=f"no_parity_turn control diff {diff.max():.3e} did not exceed 5% of "
            f"SNR^2={snr2:.3e} -- the parity suite has no power at this SNR",
        )

    def test_expose_invariant_via_offset(self):
        move = self.move
        move.setup_likelihood_here(None)
        rows = np.tile(REF_STOCK, (NWALKERS, 1))
        idx = np.arange(NWALKERS, dtype=np.int32)
        fast = move.compute_like(rows, idx)
        slow = move.compute_acs_like(rows, idx, **move.waveform_like_kwargs)
        np.testing.assert_allclose(fast, slow, atol=move.check_ll_tol)

    def test_out_of_band_sentinel(self):
        move = self.move
        move.setup_likelihood_here(None)
        bad = REF_STOCK.copy()
        bad[6] = 0.5
        self.assertEqual(
            float(move.compute_like(bad.reshape(1, -1), np.zeros(1, dtype=np.int32))[0]), -1e300
        )
        nanrow = REF_STOCK.copy()
        nanrow[0] = np.nan
        self.assertEqual(
            float(move.compute_like(nanrow.reshape(1, -1), np.zeros(1, dtype=np.int32))[0]), -1e300
        )

    def test_record_leaf_inner_products_from_comp(self):
        from types import SimpleNamespace

        move = self.move
        move.setup_likelihood_here(None)
        sub = SimpleNamespace(
            d_h=np.full((NWALKERS, 1), np.nan), h_h=np.full((NWALKERS, 1), np.nan)
        )
        state = SimpleNamespace(sub_states={"sobbh": sub})
        move._record_leaf_inner_products(state, np.tile(REF_STOCK, (NWALKERS, 1)), 0)
        self.assertTrue(np.all(np.isfinite(sub.d_h)))
        self.assertTrue(np.all(sub.h_h > 0))

    def test_fill_through_move_round_trip(self):
        move = self.move
        buf = self.acs.linear_data_arr[0]
        before = np.array(buf, copy=True)
        coords = np.tile(REF_STOCK, (NWALKERS, 1))
        move._apply_cold_chain_sources(coords, +1)
        self.assertGreater(float(np.abs(buf - before).max()), 0.0)
        move._apply_cold_chain_sources(coords, -1)
        np.testing.assert_allclose(buf, before, rtol=0, atol=1e-12 * np.abs(before).max())

    def test_merged_before_window_is_zero_template(self):
        # the production case: the catalogue epoch (t_ref) lies BEFORE the data window and the
        # source merges in between -> every pixel is dead -> zero template, finite lnL
        from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

        kw = self.comp.kwargs
        comp2 = SOBBHLookupComputations(
            self.wdm, float(T_START) - 30.0 * 86400.0, self.comp.args[2], **kw
        )
        row = SOBBHLookupParityTestHelper.chunked(REF_STOCK)
        row[0, 0] = row[0, 1] = 2000.0  # tc ~ 2 days after the reference epoch
        row[0, 5] = 1.5e-2
        ll = comp2.get_ll_wdm(
            row,
            self.acs,
            data_index=np.zeros(1, dtype=np.int32),
            noise_index=np.zeros(1, dtype=np.int32),
        )
        self.assertTrue(np.isfinite(float(ll[0])))
        self.assertEqual(float(ll[0]), 0.0)  # d_d = 0, d_h = h_h = 0
        self.assertEqual(float(comp2.h_h_out[0]), 0.0)
        self.assertEqual(float(comp2.d_h_out[0]), 0.0)
        self.assertEqual(comp2.last_stats["merged_rows"], 1)
        self.assertEqual(comp2.last_stats["lookup_pixels"], 0)

    def test_fill_into_lone_container_writes_it(self):
        # fill_global_wdm against a LONE AnalysisContainer must write into the container's OWN
        # residual buffer, not a disposable shallow-copy wrapper (as_single_shard_holder's
        # single-AC path copies the data/sens_mat holders -- correct for scoring, silently
        # inert for a fill, since nothing downstream reads the fresh copy's buffer).
        ac = self.acs[0]
        before = np.array(ac.data_res_arr.arr, copy=True)
        self.comp.fill_global_wdm(
            SOBBHLookupParityTestHelper.chunked(REF_STOCK), ac, factors=np.array([1.0])
        )
        after = np.asarray(ac.data_res_arr.arr)
        self.assertGreater(float(np.abs(after - before).max()), 0.0)
        self.comp.fill_global_wdm(
            SOBBHLookupParityTestHelper.chunked(REF_STOCK), ac, factors=np.array([-1.0])
        )
        np.testing.assert_allclose(
            np.asarray(ac.data_res_arr.arr), before, rtol=0, atol=1e-12 * np.abs(before).max()
        )

    def test_bad_data_index_raises(self):
        with self.assertRaises(IndexError):
            self.comp.get_ll_wdm(
                SOBBHLookupParityTestHelper.chunked(REF_STOCK),
                self.acs,
                data_index=np.array([NWALKERS]),
                noise_index=np.array([0]),
            )

    def test_comp_surface_for_the_move(self):
        comp = self.comp
        self.assertEqual(comp.d_d, 0.0)
        self.assertIs(comp.wdm_settings, self.wdm)
        self.assertIsNone(comp.d_h_im_out)
        self.assertEqual(comp.args[1], float(T_START))
        self.assertEqual(comp.kwargs["row_batch"], 4)


class SOBBHLookupGramTest(unittest.TestCase):
    """``SOBBH_EIGEN_INFO=gram`` with the LOOKUP comp (production:
    ``SOBBH_LIKELIHOOD=lookup`` hands the lookup router to the move as its
    ``comp``, so the Gram templates are lookup fills like every other SOBBH
    template). Against an independent Gram of the dense stock templates at the
    move's own steps. Spins barely move the template: judged apart."""

    KEEP = [0, 1, 4, 5, 6, 7, 8, 9, 10]
    WIDTHS = np.array([40.0, 40.0, 1.0, 1.0, 0.1, 1.0, 1e-3, 1.0, 1.0, 1.0, 1.0])

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        try:
            cls.acs, cls.gen, cls.comp, cls.wdm, cls.td_set = _build_toy(cls.tmp.name)
        except Exception as exc:  # missing compiled deps etc.
            raise unittest.SkipTest(f"toy setup unavailable: {exc}")
        cls.move = _build_move(cls.acs, cls.comp)
        cls.move._to_phys = lambda x: np.atleast_2d(np.asarray(x, dtype=float))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _dense(self, steps, walker):
        from lisatools.diagnostic import inner_product

        gen = type(self).__dict__["gen"]
        dh = []
        for i in self.KEEP:
            e = np.zeros(11)
            e[i] = steps[i]
            hp, hm = gen(*(REF_STOCK + e)), gen(*(REF_STOCK - e))
            dh.append(type(hp)((np.asarray(hp.arr) - np.asarray(hm.arr)) / (2 * steps[i]),
                               hp.settings))
        _, _, psd = self.acs.acs.flatten()[walker]._slice_to_template(dh[0])
        k = len(self.KEEP)
        G = np.zeros((k, k))
        for a in range(k):
            for b in range(a, k):
                G[a, b] = G[b, a] = float(np.real(inner_product(dh[a], dh[b], psd=psd)))
        return G

    def test_lookup_gram_matches_the_dense_template_gram(self):
        from unittest import mock

        with mock.patch.object(self.comp, "fill_global_wdm",
                               wraps=self.comp.fill_global_wdm) as fill:
            G, steps = self.move._gram_info(REF_STOCK.copy(), 0, self.WIDTHS,
                                            return_steps=True)
        self.assertGreater(fill.call_count, 0)        # the lookup comp made the templates
        G = G[np.ix_(self.KEEP, self.KEEP)]
        T = self._dense(steps, 0)
        d = np.sqrt(np.abs(np.diag(T)))
        rel = np.abs(G - T) / np.outer(d, d)
        print(f"[sobbh lookup gram] max correlation-normalized diff {rel.max():.3e}, "
              f"diag ratio {np.round(np.diag(G) / np.diag(T), 4)}")
        self.assertLess(rel.max(), self.TOL)

    TOL = 3e-2  # measured 1.5e-2 (lookup tiny table vs dense)


class SOBBHLookupParityTestHelper:
    @staticmethod
    def chunked(row):
        from lisatools.globalfit.moves import SOBBHChunkedLikeMove

        return SOBBHChunkedLikeMove.to_chunked_basis(row)


if __name__ == "__main__":
    unittest.main()
