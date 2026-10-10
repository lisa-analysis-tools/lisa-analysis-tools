"""The per-(walker, band) RJ valve judges the band's GAIN, not its residual.

USER RULING 2026-10-10 (9mo gb_search_2, where psd + galfor are re-fitted at
the end of every cycle). The valve used to judge the band's cold-walker
residual term ``-1/2 <r|r>``. A band holds ~2,780 pixels, so that term is
~-1,400 nats and a relative sensitivity change ``eps`` moves it by
~-1,400 * eps with NO GB change: when the noise block LOWERS the foreground
every pair falls below its stale best and shuts as "converged" within three
cycles, whatever the GB moves do. The statistic is now the band's likelihood
GAIN over the empty model under the walker's CURRENT noise,

    G = 1/2 <d|d> - 1/2 <r|r>      (G >= 0 when the templates explain power).

At a fixed noise the data term is a constant, so G has exactly the residual
term's increments and the valve's decisions do not change. When the noise
drops G moves UP by ``eps * G`` and sets a fresh best by itself; when it
rises G moves down and the band may shut (accepted).

The fixtures are the light fakes of ``test_gb_search_stage_schedule`` plus a
one-shard CPU ACA with a diagonal (AE-like) FD noise model, built so every
number is exact: the noise ``n`` is real and the sources ``h`` imaginary, so
``<n|h> = 0`` and, with invC = 1, a band's two terms are

    1/2 <n|n> = 1000 per band,   1/2 <h|h> = 500 per band.
"""

import inspect
import logging
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import lisatools.globalfit.moves.gbspecialstretch as gbs
from lisatools.globalfit.moves.gbbands import SubBandBuffer

from tests.test_gb_cap_cell_grid import BAND_EDGES, NUM_BANDS
from tests.test_gb_search_stage_schedule import (_band_info, _counts,
                                                 _stage_move, _state)

DF = 1e-4                  # FD bin width: 10 bins per 1 mHz band
START = 50                 # first stored bin = BAND_EDGES[0] / DF
NF = 40                    # stored bins = the four bands, exactly
NC = 2                     # diagonal channels
NORM = 4.0 * DF            # <a|b> = 4 df sum Re(a* b) invC
A_NOISE = 500.0            # 1/2 <n|n> = 1/2 * NORM * 20 * 500**2 = 1000 / band
A_SRC = np.sqrt(1.25e5)    # 1/2 <h|h> = 500 / band
HALF_NN = 1000.0
HALF_HH = 500.0
TOL = 4.0                  # D/2 with leaf_cap_ndim = 8 (the fixture's)
CONV = 3


def _band_bins(b):
    return slice(10 * b, 10 * (b + 1))


def _data(nw, src_bands=(1, 2, 3)):
    """``(nw, NC, NF)`` data: real noise everywhere, imaginary sources in
    ``src_bands`` (band 0 holds no source by default)."""
    d = np.full((nw, NC, NF), A_NOISE, dtype=complex)
    for b in src_bands:
        d[:, :, _band_bins(b)] += 1j * A_SRC
    return d


def _residual(fit, src_bands=(1, 2, 3)):
    """Residual for per-(walker, band) fit fractions ``fit`` (nw, nbands):
    ``r = n + (1 - f) h`` in a source band, ``n`` elsewhere."""
    fit = np.asarray(fit, dtype=float)
    nw = fit.shape[0]
    r = np.full((nw, NC, NF), A_NOISE, dtype=complex)
    for w in range(nw):
        for b in src_bands:
            r[w, :, _band_bins(b)] += 1j * A_SRC * (1.0 - fit[w, b])
    return r


def _gain(f, scale=1.0):
    """Exact G for a source band at fit fraction ``f``, invC = ``scale``."""
    return scale * HALF_HH * (1.0 - (1.0 - f) ** 2)


def _resid(f, scale=1.0):
    """Exact -1/2 <r|r> for a source band at fit fraction ``f``."""
    return -scale * (HALF_NN + HALF_HH * (1.0 - f) ** 2)


class _FakeACA:
    """What the window reductions read: ONE CPU shard, diagonal noise.

    ``input_data_residual_array`` mirrors the production chain
    ``DataResidualArray -> .data_res_arr (DomainBase) -> .arr``.
    """

    gpus = None
    nchannels = NC
    shape_sens = (NC,)
    start_freq_ind = np.array([START])
    data_length = NF

    def __init__(self, data, residual, inv_psd=1.0, with_data=True):
        self.residual = np.asarray(residual, dtype=complex)
        nw = self.residual.shape[0]
        self.inv_psd = np.full((nw, NC, NF), 1.0) * np.asarray(inv_psd)
        self.acs_total_entries = nw
        self.gpu_splits = [np.arange(nw)]
        self.psd_version = 0
        if with_data:
            self.input_data_residual_array = SimpleNamespace(
                data_res_arr=SimpleNamespace(arr=np.asarray(data[0], complex)))

    def scale_noise(self, factor):
        """A noise move: every walker's inverse PSD x ``factor``, and the
        plane's version bumped, as ``reset_linear_psd_arr`` does."""
        self.inv_psd = self.inv_psd * float(factor)
        self.psd_version += 1

    @property
    def data_shaped(self):
        return [self.residual]

    @property
    def psd_shaped(self):
        return [self.inv_psd]


def _armed_move(nwalkers):
    """The valve fixture of ``test_gb_search_stage_schedule``, plus the FD
    basis the window reductions read."""
    m = _stage_move(shutoff=True, conv_iter=CONV, nwalkers=nwalkers)
    m._basis_settings = SimpleNamespace(df=DF, differential_component=DF)
    bi = _band_info(nwalkers=nwalkers, shutoff=True)
    m._rj_band_shutoff_w = bi["band_rj_shutoff_w"]
    m._bind_shutoff_window(bi, np.shape(bi["band_rj_shutoff_w"]))
    return m, bi


def _judge(m, bi, aca, occ):
    """One iteration's judge with NO stash: the valve forms its statistic
    from ``_cap_stats_local`` on the model's ACA (the single-process path)."""
    m.num_proposals += 1
    m._update_search_band_shutoff(
        SimpleNamespace(analysis_container_arr=aca), _state(bi), _counts(occ))


def _no_tol_env():
    env = dict(os.environ)
    env.pop("GB_SEARCH_BAND_SHUTOFF_LL_TOL", None)
    env.pop("GB_SEARCH_BAND_SHUTOFF_REQUIRE_OCC", None)
    return mock.patch.dict(os.environ, env, clear=True)


# (walker, band) fit-fraction schedule, t = 0..7: births, polish, plateaus.
# Band 0 holds no source at all; walker 0's band 3 is never fitted.
_FIT = np.array([
    # w0: b0  b1    b2   b3      w1: b0  b1   b2   b3
    [[0, 0.00, 0.0, 0], [0, 1.0, 0.0, 0.0]],
    [[0, 0.90, 0.0, 0], [0, 1.0, 0.5, 0.0]],
    [[0, 0.95, 0.0, 0], [0, 1.0, 0.5, 0.3]],
    [[0, 0.99, 0.8, 0], [0, 1.0, 0.5, 0.3]],
    [[0, 1.00, 1.0, 0], [0, 1.0, 0.5, 0.6]],
    [[0, 1.00, 1.0, 0], [0, 1.0, 0.5, 0.6]],
    [[0, 1.00, 1.0, 0], [0, 1.0, 0.5, 0.6]],
    [[0, 1.00, 1.0, 0], [0, 1.0, 0.5, 0.6]],
], dtype=float)

# THE OLD DECISIONS, pinned by hand from the residual-term judge (conv_iter 3,
# tol 4): the step at which each (walker, band) shut, -1 = never.
_SHUT_AT = np.array([[-1, 4, 7, -1],
                     [-1, 3, 4, 7]])
_STREAK_END = np.array([[7, 6, 3, 7],
                        [7, 7, 6, 3]])
_RESETS_END = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 2]])


class FixedNoiseSameDecisionsTest(unittest.TestCase):
    """At a fixed noise the gain judges exactly as the residual term did."""

    def _run(self, feed_residual):
        nw = _FIT.shape[1]
        m, bi = _armed_move(nw)
        d = _data(nw)
        shut_steps, values = [], []
        for t in range(_FIT.shape[0]):
            fit = _FIT[t]
            occ = (fit > 0).astype(np.int64)
            aca = _FakeACA(d, _residual(fit))
            if feed_residual:
                # the OLD statistic, -1/2<r|r>, fed straight to the judge
                m.num_proposals += 1
                m._stage_band_gain = m._band_residual_lls(aca)
                m._stage_band_lls = m._stage_band_gain
                m._stage_band_lls_stamp = m.num_proposals
                m._update_search_band_shutoff(None, _state(bi), _counts(occ))
            else:
                m._stage_band_gain = m._stage_band_lls = None
                _judge(m, bi, aca, occ)
            shut_steps.append(bi["band_rj_shutoff_w"].copy())
            values.append(bi["band_cold_logl_w"].copy())
        return shut_steps, values, bi

    def _assert_pinned(self, shut_steps, bi):
        for t, shut in enumerate(shut_steps):
            want = (_SHUT_AT >= 0) & (_SHUT_AT <= t)
            np.testing.assert_array_equal(shut, want, err_msg=f"step {t}")
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"], _STREAK_END)
        np.testing.assert_array_equal(bi["band_shutoff_reset_w"], _RESETS_END)

    def test_the_residual_term_reproduces_the_pinned_decisions(self):
        """The control: the pins ARE the old statistic's decisions."""
        with _no_tol_env():
            shut_steps, values, bi = self._run(feed_residual=True)
        self._assert_pinned(shut_steps, bi)
        self.assertLess(float(np.max(values[-1])), -HALF_NN + 1e-6)

    def test_the_gain_makes_identical_decisions_and_is_what_is_judged(self):
        with _no_tol_env():
            shut_steps, values, bi = self._run(feed_residual=False)
        self._assert_pinned(shut_steps, bi)
        # ...and the judged value is the GAIN: 1/2<d|d> - 1/2<r|r>, exactly
        # 0 for an empty band and 500 (1 - (1 - f)^2) for a source band
        for t in range(_FIT.shape[0]):
            want = _gain(_FIT[t])
            want[:, 0] = 0.0                 # band 0 holds no source
            np.testing.assert_allclose(values[t], want, rtol=0, atol=1e-8,
                                       err_msg=f"step {t}")


class NoiseMoveTest(unittest.TestCase):
    """A noise move between judges: the gain follows the fit, not the noise."""

    EPS = 0.05

    def _mature(self):
        """Band 1 fitted on both walkers, judged twice at a plateau: streak
        2 of 3, one quiet iteration from shutting."""
        m, bi = _armed_move(2)
        fit = np.zeros((2, NUM_BANDS))
        fit[:, 1] = 1.0
        occ = (fit > 0).astype(np.int64)
        aca = _FakeACA(_data(2), _residual(fit))
        for _ in range(CONV):
            _judge(m, bi, aca, occ)
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"][:, 1], 2)
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        return m, bi, aca, occ

    def test_a_noise_DROP_raises_the_gain_and_resets_a_mature_streak(self):
        with _no_tol_env():
            m, bi, aca, occ = self._mature()
            r_before = m._band_residual_lls(aca)[:, 1].copy()
            aca.scale_noise(1.0 + self.EPS)          # foreground lowered
            r_after = m._band_residual_lls(aca)[:, 1]
            _judge(m, bi, aca, occ)
        # the residual term FELL -- judged on it, the pair would have shut
        np.testing.assert_allclose(r_before, _resid(1.0))
        np.testing.assert_allclose(r_after, _resid(1.0, 1.0 + self.EPS))
        self.assertTrue(np.all(r_after < r_before - TOL))
        # the gain ROSE by eps * G and set a fresh best
        np.testing.assert_allclose(bi["band_cold_logl_w"][:, 1],
                                   _gain(1.0, 1.0 + self.EPS))
        np.testing.assert_allclose(bi["band_cold_logl_max_w"][:, 1],
                                   _gain(1.0, 1.0 + self.EPS))
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"][:, 1], 0)
        np.testing.assert_array_equal(bi["band_shutoff_reset_w"][:, 1], 1)
        self.assertFalse(bi["band_rj_shutoff_w"][:, 1].any())

    def test_a_noise_RISE_lowers_the_gain_and_does_not_reset(self):
        with _no_tol_env():
            m, bi, aca, occ = self._mature()
            r_before = m._band_residual_lls(aca)[:, 1].copy()
            aca.scale_noise(1.0 - self.EPS)
            r_after = m._band_residual_lls(aca)[:, 1]
            _judge(m, bi, aca, occ)
        # the residual term ROSE past the tolerance -- judged on it, this
        # would have been read as an improvement and reset the streak
        self.assertTrue(np.all(r_after > r_before + TOL))
        np.testing.assert_allclose(bi["band_cold_logl_w"][:, 1],
                                   _gain(1.0, 1.0 - self.EPS))
        # the gain fell: no improvement, the streak runs out, the pair shuts
        np.testing.assert_array_equal(bi["band_shutoff_streak_w"][:, 1], 3)
        np.testing.assert_array_equal(bi["band_shutoff_reset_w"][:, 1], 0)
        self.assertTrue(bi["band_rj_shutoff_w"][:, 1].all())


class EmptyBandTest(unittest.TestCase):
    """A band with no template: ``r = d`` there, so G = 0 under ANY noise."""

    def test_the_gain_of_an_empty_band_is_zero_and_ignores_the_noise(self):
        m, _bi = _armed_move(2)
        fit = np.zeros((2, NUM_BANDS))
        fit[:, 1] = 1.0                     # band 1 fitted; band 0 has nothing
        aca = _FakeACA(_data(2), _residual(fit))
        model = SimpleNamespace(analysis_container_arr=aca)
        for factor in (1.0, 1.0 + 0.05, 1.0 - 0.05):
            if factor != 1.0:
                aca.scale_noise(factor)
            stats = m._cap_stats_local(model, None)
            np.testing.assert_allclose(stats["band_gain"][:, 0], 0.0,
                                       rtol=0, atol=1e-9)
            # the residual term of the same band is ~-1000 and moves with it
            self.assertTrue(np.all(stats["band_lls"][:, 0] < -900.0))


class StaleUnitsTransitionTest(unittest.TestCase):
    """A store written before the ruling holds its best in RESIDUAL units.

    ~-1,500 nats where the gain is ~+500: the first gain judge beats it
    once (one reset per pair), and judging then proceeds normally. No
    migration is needed.
    """

    def test_an_old_unit_best_is_beaten_once_then_judging_is_normal(self):
        with _no_tol_env():
            m, bi = _armed_move(2)
            fit = np.zeros((2, NUM_BANDS))
            fit[:, 1] = 1.0
            occ = (fit > 0).astype(np.int64)
            aca = _FakeACA(_data(2), _residual(fit))
            # the stored, old-unit record: best = the residual term, streak 2
            bi["band_cold_logl_max_w"][:] = _resid(1.0)
            bi["band_shutoff_streak_w"][:] = 2
            _judge(m, bi, aca, occ)
            np.testing.assert_allclose(bi["band_cold_logl_max_w"][:, 1],
                                       _gain(1.0))
            np.testing.assert_array_equal(bi["band_shutoff_streak_w"][:, 1], 0)
            np.testing.assert_array_equal(bi["band_shutoff_reset_w"][:, 1], 1)
            self.assertFalse(bi["band_rj_shutoff_w"].any())
            for k in range(1, CONV + 1):
                _judge(m, bi, aca, occ)
                np.testing.assert_array_equal(
                    bi["band_shutoff_streak_w"][:, 1], k)
            self.assertTrue(bi["band_rj_shutoff_w"][:, 1].all())
            np.testing.assert_array_equal(bi["band_shutoff_reset_w"][:, 1], 1)


class _FakeBuf:
    """A SubBandBuffer stand-in that runs the REAL slot likelihood code.

    One slot per (walker, band) cell, each a ``2 * PAD``-bin-wider window
    than its band (a slab), cut out of the ACA's residual and inverse PSD
    exactly as ``fill_buffer_residual_and_psd_from_acs`` does.
    """

    likelihood = SubBandBuffer.likelihood
    band_likelihoods = SubBandBuffer.likelihood
    _likelihood_slots = SubBandBuffer._likelihood_slots
    nchannels = NC
    tdi_channel_setup = "AE"
    use_template_arr = False
    _psd_shared_mirror = False
    xp = np
    PAD = 3

    def __init__(self, aca, cells):
        L = 10 + 2 * self.PAD
        self.settings = SimpleNamespace(differential_component=DF)
        self._per_band_data_shape = (NC, L)
        origins = np.array([min(max(10 * b - self.PAD, 0), NF - L)
                            for _, b in cells])
        self.buffer_start_index = START + origins
        self.band_buffer = np.stack([aca.residual[w, :, o:o + L]
                                     for (w, _), o in zip(cells, origins)])
        self.psd_buffer = np.stack([aca.inv_psd[w, :, o:o + L]
                                    for (w, _), o in zip(cells, origins)])


class RankObservationUnitsTest(unittest.TestCase):
    """The cold peak a rank observes is in the SAME units as the head's judge.

    Under the fan-out every rank folds in-model observations into its block
    peak, and the head folds the blocks with ``np.maximum`` next to the
    judge-time statistic. A rank reading the slot's slab residual while the
    head judged the band's gain would make the maximum meaningless.
    """

    def _setup(self):
        m, _bi = _armed_move(2)
        fit = np.array([[0, 0.9, 0.5, 0.0],
                        [0, 1.0, 0.0, 0.3]])
        aca = _FakeACA(_data(2), _residual(fit))
        model = SimpleNamespace(analysis_container_arr=aca)
        cells = [(w, b) for w in range(2) for b in range(NUM_BANDS)]
        return m, model, aca, _FakeBuf(aca, cells), cells

    def test_the_observed_gain_equals_the_judge_time_gain(self):
        m, model, _aca, buf, cells = self._setup()
        w = np.array([c[0] for c in cells])
        b = np.array([c[1] for c in cells])
        slots = np.arange(len(cells))
        obs = m._observed_band_gain(model, buf, slots, w, b)
        head = m._cap_stats_local(model, None)["band_gain"]
        np.testing.assert_allclose(obs, head[w, b], rtol=1e-12, atol=1e-9)

    def test_the_slab_residual_is_NOT_the_band_value(self):
        """Why the window matters: the slab carries its neighbours' pixels,
        so its residual term sits far below the band's."""
        m, model, _aca, buf, cells = self._setup()
        slab = np.asarray(buf.band_likelihoods(
            source_only=True, slots=np.arange(len(cells))))
        band = m._band_residual_lls(model.analysis_container_arr)
        w = np.array([c[0] for c in cells])
        b = np.array([c[1] for c in cells])
        self.assertTrue(np.all(slab < band[w, b] - 100.0))

    def test_a_noise_move_keeps_rank_and_head_in_step(self):
        m, model, aca, _buf, cells = self._setup()
        aca.scale_noise(1.03)
        buf = _FakeBuf(aca, cells)          # the next unit's fill
        w = np.array([c[0] for c in cells])
        b = np.array([c[1] for c in cells])
        obs = m._observed_band_gain(model, buf, np.arange(len(cells)), w, b)
        head = m._cap_stats_local(model, None)["band_gain"]
        np.testing.assert_allclose(obs, head[w, b], rtol=1e-12, atol=1e-9)

    def test_the_in_model_site_observes_the_gain(self):
        """The observation site goes through the gain helper, not the bare
        slab residual it used to fold into the peak."""
        src = inspect.getsource(gbs.GBSpecialBase._run_in_model_repeats)
        i = src.index("COLD-CHAIN BAND")
        blk = src[i:src.index("np.maximum.at(_cold_peak", i)]
        self.assertIn("self._observed_band_gain(", blk)
        self.assertNotIn("band_likelihoods(", blk)


class DataTermReductionTest(unittest.TestCase):
    """The data term reduces ONE shared data plane with the same windows
    and per-walker inverse PSDs as the residual term -- on every einsum
    branch (WDM x {XYZ, diagonal}; the FD diagonal one is above). With every
    walker's residual EQUAL to the data the two must agree to rounding."""

    NW, NFA, NT = 3, 6, 5

    def _move(self):
        from lisatools.domains import WDMSettings

        class _WDM(WDMSettings):
            layer_df = 1.0
            ind_min_f = 0
            Nf_active = self.NFA
            Nt_active = self.NT
            differential_component = 0.5

        m = gbs.GBSpecialStretchMove.__new__(gbs.GBSpecialStretchMove)
        m._backend_name = "lisatools_cpu"
        m._basis_settings = object.__new__(_WDM)
        return m

    def _check(self, xyz):
        rng = np.random.default_rng(3 if xyz else 4)
        nc = 3
        d = rng.standard_normal((nc, self.NFA, self.NT))
        shape_sens = (nc, nc) if xyz else (nc,)
        ic = rng.standard_normal((self.NW,) + shape_sens + (self.NFA, self.NT))
        aca = SimpleNamespace(
            gpus=None, nchannels=nc, shape_sens=shape_sens,
            acs_total_entries=self.NW, gpu_splits=[np.arange(self.NW)],
            data_shaped=[np.repeat(d[None], self.NW, axis=0)],
            psd_shaped=[ic],
            input_data_residual_array=SimpleNamespace(
                data_res_arr=SimpleNamespace(arr=d)))
        edges = np.array([0.0, 2.0, 4.0, 6.0])
        m = self._move()
        resid, dof = m._window_residual_lls(aca, edges)
        data = m._window_data_lls(aca, edges)
        np.testing.assert_allclose(data, -resid, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.all(np.abs(data) > 0))
        # the walkers' own inverse PSDs: rows differ
        self.assertGreater(float(np.ptp(data[:, 0])), 0.0)

    def test_wdm_xyz(self):
        self._check(xyz=True)

    def test_wdm_diagonal(self):
        self._check(xyz=False)


class StatisticPlumbingTest(unittest.TestCase):
    """Who reads what: the valve the gain, the leaf caps the residual."""

    def _aca(self):
        fit = np.zeros((2, NUM_BANDS))
        fit[:, 2] = 1.0
        return _FakeACA(_data(2), _residual(fit))

    def test_cap_stats_carry_both_and_the_cap_gate_keeps_the_residual(self):
        m, _bi = _armed_move(2)
        stats = m._cap_stats_local(
            SimpleNamespace(analysis_container_arr=self._aca()), None)
        np.testing.assert_allclose(stats["band_lls"][:, 2], _resid(1.0))
        # the leaf-cap gate's own input is untouched by the ruling
        np.testing.assert_allclose(stats["lls"], stats["band_lls"])
        np.testing.assert_allclose(stats["band_gain"][:, 2], _gain(1.0))
        np.testing.assert_allclose(
            stats["band_gain"] - stats["band_lls"],
            m._window_data_lls(self._aca(), m.band_edges))

    def test_the_cap_gate_stashes_the_gain_and_keeps_the_residual(self):
        """The production stash site with caps ON (``_update_band_leaf_caps``
        with the head-assembled statistic): the valve gets the gain, the
        gate's own record stays the residual term."""
        from lisatools.globalfit.state import ensure_leaf_cap_fields
        from tests.test_gb_cap_cell_grid import _gate_move

        m = _gate_move(cap_divisor=1, nwalkers=2)
        m._cold_occupancy = lambda bc, ns: np.ones((2, NUM_BANDS), dtype=int)
        bi = {"num_bands": NUM_BANDS}
        ensure_leaf_cap_fields(bi, NUM_BANDS)
        bi["band_leaf_cap"][:] = 1
        bi["band_cold_ll"] = np.zeros((2, NUM_BANDS))
        state = _state(bi)
        resid = np.full((2, NUM_BANDS), -1500.0)
        gain = np.full((2, NUM_BANDS), 250.0)
        m.num_proposals = 3
        m._update_band_leaf_caps(
            SimpleNamespace(analysis_container_arr=None), state, None,
            precomputed={"band_lls": resid, "band_gain": gain, "lls": resid,
                         "dof": np.full(NUM_BANDS, 20.0),
                         "band_dof": np.full(NUM_BANDS, 20.0),
                         "is_cells": False})
        self.assertEqual(m._stage_band_lls_stamp, 3)
        np.testing.assert_allclose(m._stage_band_lls, resid)
        np.testing.assert_allclose(m._stage_band_gain, gain)
        np.testing.assert_allclose(m._shutoff_band_lls(None, state), gain)
        np.testing.assert_allclose(bi["band_cold_ll"], resid)

    def test_the_valve_reads_the_gain_stash_never_the_residual_one(self):
        m, bi = _armed_move(2)
        m.num_proposals = 4
        m._stage_band_lls = np.full((2, NUM_BANDS), -1500.0)
        m._stage_band_gain = np.full((2, NUM_BANDS), 250.0)
        m._stage_band_lls_stamp = 4
        np.testing.assert_allclose(m._shutoff_band_lls(None, _state(bi)), 250.0)

    def test_no_data_term_leaves_the_valve_inert_never_the_residual(self):
        """Without the data the gain cannot be formed. Falling back to the
        residual term would bring back exactly the failure the ruling
        removes, so the valve does nothing (the permissive direction)."""
        m, bi = _armed_move(2)
        fit = np.zeros((2, NUM_BANDS))
        fit[:, 1] = 1.0
        occ = (fit > 0).astype(np.int64)
        aca = _FakeACA(_data(2), _residual(fit), with_data=False)
        with self.assertLogs(gbs.logger, level=logging.WARNING):
            for _ in range(CONV + 2):
                _judge(m, bi, aca, occ)
        self.assertFalse(bi["band_rj_shutoff_w"].any())
        self.assertTrue(np.all(np.isneginf(bi["band_cold_logl_max_w"])))


if __name__ == "__main__":
    unittest.main()
