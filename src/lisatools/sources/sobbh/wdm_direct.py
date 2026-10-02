# src/lisatools/sources/sobbh/wdm_direct.py
"""SOBBH templates built directly in the WDM domain (batched TDI-on-the-fly + lookup table).

The SOBBH twin of :mod:`lisatools.sources.emri.wdm_direct`, vectorized over parameter rows so
one call scores a whole proposal batch (spec:
docs/superpowers/specs/2026-09-30-sobbh-wdm-lookup-design.md). Pieces, in data-flow order:

* :func:`sobbh_amp_phase_batch` -- the 3.5PN amplitude / GW phase of ``N`` rows on one time
  grid (the core of :mod:`lisatools.sources.sobbh.waveform`, row-vectorized);
* :class:`SOBBHBatchedTOF` -- ONE :class:`~lisatools.response.tdionfly.TDTDIonTheFly` for the
  batch, built exactly as :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly` builds it for one row but
  evaluated on a coarse grid (the dense data grid is never built);
* :func:`sobbh_tracer` -- per-channel ``(amp, phase, f, fdot)`` at the WDM pixel centres from
  the response splines;
* :class:`SOBBHDirectWDM` -- the sparse template (5 layers per pixel per channel) through
  :class:`~lisatools.wdm_lookup_eval.WDMLookupEvaluator`;
* :func:`sparse_inner_products` / :func:`scatter_add` -- the chunked kernel's ``<d|h>`` /
  ``<h|h>`` accumulation and the residual fill, gathered at the sparse support;
* :class:`SOBBHLookupComputations` -- the comp that slots into
  :class:`~lisatools.globalfit.moves.sobbhspecialmove.SOBBHChunkedLikeMove`.

Conventions pinned by the tests: chunked-basis rows ``(m1, m2, s1, s2, dist[pc], f_low, phi_c,
inc, psi, lam, beta)``; the response feed ``phase = gw_phase + pi`` with the intrinsic
amplitude (``SOBBHTDIonFly``); pixel centres ``t_n = t_obs_start + n * layer_dt`` with ``n`` the
absolute grid index; the inner products carry NO ``4 * differential_component`` factor (the
kernel convention: ``-0.5 (h_h - 2 d_h)`` is the container's source term).
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from ...utils.device import current_device, synchronize
from ...utils.utility import asnumpy, get_array_module

logger = logging.getLogger(__name__)


def sobbh_amp_phase_batch(params, times, reference_time, t_shift=0.0):
    """3.5PN ``(amp, gw_phase, tc_abs)`` of ``N`` rows at absolute ``times`` [s].

    ``params`` is ``(N, >= 7)`` in the chunked basis ``(m1 [Msun], m2 [Msun], s1, s2, dist [pc],
    f_low [Hz], phi_c [rad], ...)``; ``f_low`` and ``phi_c`` are defined at ``reference_time``
    (the PN time is ``times - reference_time - t_shift``). Row-wise identical to
    :meth:`lisatools.sources.sobbh.waveform.SOBBHWaveform.compute_amp_phase`: ``amp = 2 M eta x
    / D`` (intrinsic, no inclination), ``gw_phase = 2 Phi`` anchored at the reference epoch,
    the same Newton-refined ``tc``. Samples at or past merger get ``amp = 0`` and the phase
    frozen at the last live sample. Rows whose merger time is not finite and positive return a
    zero template. Returns ``amp (N, T)``, ``gw_phase (N, T)``, ``tc_abs (N,)``.
    """
    from .waveform import MTsun, c, pc
    from .waveform import phase as _phase
    from .waveform import tau_to_x, time_to_merger

    p = np.atleast_2d(np.asarray(params, dtype=float))
    if p.ndim != 2 or p.shape[1] < 7:
        raise ValueError(f"params must be (N, >= 7) chunked-basis rows, got shape {p.shape}")
    t = np.asarray(times, dtype=float).reshape(-1)
    m1 = p[:, 0:1] * MTsun
    m2 = p[:, 1:2] * MTsun
    s1 = p[:, 2:3]
    s2 = p[:, 3:4]
    D = p[:, 4:5] * pc / c
    f_low = p[:, 5:6]
    phi_c = p[:, 6:7]
    M = m1 + m2
    eta = m1 * m2 / M**2
    sigma = (m2 * s2 - m1 * s1) / M
    s = (m1**2 * s1 + m2**2 * s2) / M**2
    delta = (m1 - m2) / M
    # NOTE: v0**2 (v0 = (pi M f_low)**(1/3)) matches _pn_amp_phase_core's exact
    # op order bit-for-bit. The equivalent one-shot `**(2/3)` differs by 1 ULP
    # in x0; that error survives -- via incomplete cancellation, not via any
    # steepness in phase(x) itself -- in `-phase(x) + phase(x_ref)`
    # (phase(x_ref) ~ 2e7 rad here, so ~8 ULP ~ 6e-8 rad leaks through). The
    # row match against the jnp reference is therefore bit-tight only on a host
    # where both paths hit the same libm `pow`; where JAX evaluates on a GPU the
    # two differ by that ~1e-8 rad floor (tests allow 1e-6 rad absolute).
    v0 = (np.pi * M * f_low) ** (1.0 / 3.0)
    x0 = v0**2

    # tc consistent with the tau_to_x series: Newton on tau_to_x(tau_ref) == x0 (waveform.py)
    tc = np.asarray(time_to_merger(x0, sigma, delta, eta, s), dtype=float) * M
    for _ in range(4):
        tau_ref = eta * tc / (5.0 * M)
        x_ref = np.asarray(tau_to_x(tau_ref, sigma, delta, eta, s), dtype=float)
        h = tau_ref * 1e-6
        dx = (np.asarray(tau_to_x(tau_ref + h, sigma, delta, eta, s), dtype=float) - x_ref) / h
        tau_ref = tau_ref - (x_ref - x0) / dx
        tc = 5.0 * M * tau_ref / eta

    # Rows whose Newton-refined tc isn't finite and positive (pathological params) would
    # otherwise poison tau/x/phase with NaNs for the whole row; sidestep that with a dummy
    # finite tc for the PN evaluation and zero the row's template afterwards.
    bad = ~np.isfinite(tc) | (tc <= 0)
    tc_safe = np.where(bad, 1.0, tc)

    pn_t = (t - float(reference_time) - float(t_shift))[None, :]
    live = pn_t < tc_safe  # (N, T)
    tau = eta * (tc_safe - np.where(live, pn_t, 0.0)) / (5.0 * M)  # dead: tau at pn_t = 0
    x = np.asarray(tau_to_x(tau, sigma, delta, eta, s), dtype=float)
    tau_ref = eta * tc_safe / (5.0 * M)
    x_ref = np.asarray(tau_to_x(tau_ref, sigma, delta, eta, s), dtype=float)
    Phi = (
        phi_c
        - np.asarray(_phase(x, sigma, delta, eta, s), dtype=float)
        + np.asarray(_phase(x_ref, sigma, delta, eta, s), dtype=float)
    )
    amp = np.where(live, 2.0 * M * eta * x / D, 0.0)
    gw_phase = 2.0 * Phi
    amp = np.where(bad, 0.0, amp)
    gw_phase = np.where(bad, 0.0, gw_phase)
    n_live = live.sum(axis=1)  # bad rows: gw_phase is already all-zero, so the freeze is a no-op
    for i in np.flatnonzero(n_live < t.size):
        gw_phase[i, n_live[i] :] = gw_phase[i, max(int(n_live[i]) - 1, 0)]
    return amp, gw_phase, tc.ravel() + float(reference_time) + float(t_shift)


class SOBBHBatchedTOF:
    """ONE TDI-on-the-fly response for a batch of SOBBH rows on a coarse evaluation grid.

    Mirrors :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly` (node grid ``linspace(lo - buffer_time,
    hi + buffer_time, n_grid)``, feed ``phase = gw_phase + pi`` with the intrinsic amplitude,
    the real ``inc`` / ``psi``, ``(lam, beta)`` consumed in the orbits frame) but for ``N`` rows
    at once (``num_sub = N``) and evaluated on a SPARSE grid -- at most every ``eval_dt``
    seconds, ending exactly at the window -- instead of on the dense data grid: the lookup only
    needs the per-channel amplitude / phase splines at the WDM pixel centres, and the whole
    channel phase of an in-band SOBBH (the slow PN carrier plus the orbital Doppler term) has a
    fourth derivative small enough that a cubic spline on 12-h nodes is ~3e-7 rad at worst
    (the 6-month gate reproduced every digit of the 600-s result up to 1-day nodes). For a row
    merging inside the window the shared node grid continues past ``tc`` with zero amplitude
    (``SOBBHTDIonFly`` truncates its node grid at ``tc`` instead, and so zeros the last
    ``buffer_time`` before merger).

    The evaluation grid is kept ``DELAY_MARGIN`` inside the orbit tables' coverage
    (``orbits.t_base``): the C++ response zeroes any time it cannot serve, and a half-day
    spline across that zero edge rings back into the window.

    Args:
        orbits: :class:`~lisatools.detector.Orbits` (frame = the frame of ``lam``/``beta``).
        tdi_config: :class:`~lisatools.response.tdiconfig.TDIConfig`.
        reference_time: epoch [s] where ``f_low`` / ``phi_c`` are defined.
        n_grid, buffer_time: node grid size / padding [s] (production: 2048 / 5000);
            ``buffer_time`` must cover the TDI delays (``>= DELAY_MARGIN``).
        eval_dt: response evaluation step [s] (default 43200 = 12 h; 600 reproduces
            ``SOBBHTDIonFly`` to ~1e-16 on the module test grid and 1800 .. 86400 reproduce the
            600-s gate to every printed digit at 6 months).
        force_backend: backend name (``"cpu"``, ``"cuda12x"``, ...).
    """

    #: TDI delay margin [s]: the response at ``t`` reads the source and the orbits up to this
    #: much away from ``t`` (the SSB projection ``|k . x| / c`` is < 500 s, the arms ~8 s each)
    DELAY_MARGIN = 600.0

    def __init__(
        self,
        orbits,
        tdi_config,
        reference_time,
        *,
        n_grid=2048,
        buffer_time=5000.0,
        eval_dt=43200.0,
        force_backend="cpu",
    ):
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.reference_time = float(reference_time)
        self.n_grid = int(n_grid)
        self.buffer_time = float(buffer_time)
        self.eval_dt = float(eval_dt)
        if self.n_grid < 16 or self.eval_dt <= 0.0:
            raise ValueError("n_grid must be >= 16 and eval_dt > 0")
        if self.buffer_time < self.DELAY_MARGIN:
            raise ValueError(
                f"buffer_time ({self.buffer_time}) must be >= DELAY_MARGIN "
                f"({self.DELAY_MARGIN}): the kernel reads the source spline up to the TDI "
                "delays away from each evaluation time, and the padded node grid must cover "
                "that."
            )
        self.force_backend = (
            force_backend if isinstance(force_backend, str) else force_backend.name.split("_")[-1]
        )
        self._orbit_span_cache = None

    def orbit_span(self):
        """Absolute times ``(t_lo, t_hi)`` the orbit tables cover (``orbits.t_base``), or
        ``(-inf, inf)`` for orbits without a base grid."""
        if self._orbit_span_cache is None:
            t_base = getattr(self.orbits, "t_base", None)
            if t_base is None:
                self._orbit_span_cache = (-np.inf, np.inf)
            else:
                t_base = np.asarray(t_base, dtype=float)
                self._orbit_span_cache = (float(t_base[0]), float(t_base[-1]))
        return self._orbit_span_cache

    def eval_grid(self, t_lo, t_hi):
        """The response evaluation times over ``[t_lo, t_hi]`` [absolute s]: uniform, at most
        ``eval_dt`` apart, at least 4 points, ending exactly at the window ends after clipping
        them ``DELAY_MARGIN`` inside the orbit span."""
        t_lo, t_hi = float(t_lo), float(t_hi)
        if t_hi <= t_lo:
            raise ValueError(f"t_hi ({t_hi}) must be > t_lo ({t_lo})")
        o_lo, o_hi = self.orbit_span()
        lo = max(t_lo, o_lo + self.DELAY_MARGIN)
        hi = min(t_hi, o_hi - self.DELAY_MARGIN)
        if hi <= lo:
            raise ValueError(
                f"window [{t_lo}, {t_hi}] lies outside the orbit tables' coverage "
                f"[{o_lo}, {o_hi}] (minus the {self.DELAY_MARGIN} s delay margin)"
            )
        n_eval = max(4, int(np.ceil((hi - lo) / self.eval_dt)) + 1)
        return np.linspace(lo, hi, n_eval)

    @property
    def backend(self):
        import lisatools

        return lisatools.get_backend(self.force_backend)

    @property
    def xp(self):
        return self.backend.xp

    def build(self, params, t_lo, t_hi):
        """Response of every row over ``[t_lo, t_hi]`` [absolute s] -> ``TDTDIOutput`` with splines.

        ``out.tc`` holds the rows' absolute merger times, as a host ``numpy`` array (it comes
        from :func:`sobbh_amp_phase_batch`, which is numpy-only) regardless of ``force_backend``.
        ``params`` may be a cupy array (pulled to host via :func:`~lisatools.utils.utility.asnumpy`
        before the numpy-only PN math runs).
        """
        from ...response.tdionfly import TDTDIonTheFly

        eval_t = self.eval_grid(t_lo, t_hi)
        p = np.atleast_2d(np.asarray(asnumpy(params), dtype=float))
        if p.shape[1] < 11:
            raise ValueError(f"params must be (N, 11) chunked-basis rows, got {p.shape}")
        N = p.shape[0]
        n_eval = eval_t.size
        lo, hi = float(eval_t[0]), float(eval_t[-1])
        node_t = np.linspace(lo - self.buffer_time, hi + self.buffer_time, self.n_grid)
        amp, gw_phase, tc = sobbh_amp_phase_batch(p, node_t, self.reference_time)
        xp = self.xp
        gen = TDTDIonTheFly(
            xp.asarray(np.ascontiguousarray(np.broadcast_to(eval_t, (N, n_eval)))),
            xp.asarray(amp),
            xp.asarray(gw_phase + np.pi),
            sampling_frequency=1.0 / float(eval_t[1] - eval_t[0]),
            num_sub=N,
            t_input=xp.asarray(np.ascontiguousarray(np.broadcast_to(node_t, (N, node_t.size)))),
            tdi_config=self.tdi_config,
            orbits=self.orbits,
            force_backend=self.force_backend,
        )
        out = gen(
            xp.asarray(p[:, 7]),
            xp.asarray(p[:, 8]),
            xp.asarray(p[:, 9]),
            xp.asarray(p[:, 10]),
            return_spline=True,
        )
        out.tc = tc
        return out


def sobbh_tracer(out, t_pixels):
    """Per-row, per-channel ``(amp, phase, f, fdot)`` at ``t_pixels`` [absolute s].

    The channel signal is ``Re[amp exp(-i phase)] = amp cos(phase)`` with ``phase = tdi_phase +
    phase_ref`` (``TDTDIOutput.eval_tdi``); ``f`` and ``fdot`` are the first and second
    derivatives of that spline phase over ``2 pi`` (``CubicSplineInterpolant(...,
    derivative=)``), so they carry the channel's Doppler shift. A negative-frequency row is
    mirrored (``cos`` is even): ``phase -> -phase``, ``f -> -f``, ``fdot -> -fdot``. Shapes
    ``(N, nch, P)`` on ``out.xp``.

    If ``out`` carries a ``tc`` attribute (the rows' absolute merger times, as set by
    :meth:`SOBBHBatchedTOF.build`), ``amp`` is forced to exactly ``0.0`` at or past each row's
    ``tc``: :class:`~lisatools.response.tdionfly.TDTDIonTheFly` fits one GLOBAL cubic spline
    through the amplitude nodes, which does not itself go to zero past merger -- it RINGS
    (alternating-sign lobes decaying roughly geometrically per node interval) because the
    underlying node values drop sharply to zero at ``tc`` instead of smoothly. Without this
    mask, ``amp`` past merger is small but nonzero and can even be negative, breaking any
    downstream ``live = amp > 0``-style logic. Rows with a non-finite ``tc`` (degenerate
    params) are treated as never merging. ``out`` objects without a ``tc`` attribute (e.g. the
    test double in this module's tests) skip the masking entirely.
    """
    xp = out.xp
    t = xp.asarray(t_pixels, dtype=float).reshape(-1)
    nb = int(out.num_bin)
    nch = int(out.tdi_amp.shape[1])
    t3 = xp.ascontiguousarray(xp.broadcast_to(t, (nb, nch, t.size)))
    t2 = xp.ascontiguousarray(xp.broadcast_to(t, (nb, t.size)))
    amp = xp.asarray(out.tdi_amp_spl(t3))
    tc_attr = getattr(out, "tc", None)
    if tc_attr is not None:
        tc = xp.asarray(np.asarray(tc_attr, dtype=float))
        tc = xp.where(xp.isfinite(tc), tc, xp.inf)
        amp = xp.where(t[None, None, :] >= tc[:, None, None], 0.0, amp)
    ph = xp.asarray(out.tdi_phase_spl(t3)) + xp.asarray(out.phase_ref_spl(t2))[:, None, :]
    d1 = (
        xp.asarray(out.tdi_phase_spl(t3, derivative=1))
        + xp.asarray(out.phase_ref_spl(t2, derivative=1))[:, None, :]
    )
    d2 = (
        xp.asarray(out.tdi_phase_spl(t3, derivative=2))
        + xp.asarray(out.phase_ref_spl(t2, derivative=2))[:, None, :]
    )
    f = d1 / (2.0 * np.pi)
    fdot = d2 / (2.0 * np.pi)
    neg = f < 0
    return amp, xp.where(neg, -ph, ph), xp.where(neg, -f, f), xp.where(neg, -fdot, fdot)


# ----------------------------------------------------------------------
# Sparse template, inner products and fill
# ----------------------------------------------------------------------


@dataclasses.dataclass
class SparseWDMTemplate:
    """A batch of lookup templates on their sparse support.

    ``w`` ``(N, nch, L, P)`` coefficients (zero where ``valid`` is False); ``m_act`` ``(N, L, P)``
    layer index into the ACTIVE band (clipped into range where invalid); ``n_act`` ``(P,)`` time
    index into the active band; ``valid`` ``(N, nch, L, P)``; ``stats`` the accounting dict (``rows``, ``pixels``,
    ``lookup_pixels``, ``dropped_pixels``, ``merged_rows``, ``dropped_layers`` -- live layers
    outside the active band, a normal band-edge effect that is counted but never warned).
    """

    w: object
    m_act: object
    n_act: object
    valid: object
    stats: dict


def as_single_shard_holder(holder):
    """Normalise ``holder`` to a single-shard :class:`AnalysisContainerArray`.

    Same contract as ``WDMComputationsBase._as_wdm_holder``: an ACA passes through (multi-shard
    raises: route per shard), a lone :class:`AnalysisContainer` is wrapped in a 1-element ACA
    built from SHALLOW copies of its data / sens_mat holders (wrapping the container itself
    would rebind it away from its parent's buffers).
    """
    if hasattr(holder, "linear_data_arr"):
        if len(holder.linear_data_arr) != 1:
            raise NotImplementedError(
                "the lookup comp is single-shard (it consumes linear_data_arr[0]); route a "
                "multi-shard ACA per split (SOBBHChunkedLikeMove already does)."
            )
        return holder
    import copy as _copy

    from ...analysiscontainer import AnalysisContainer, AnalysisContainerArray

    if isinstance(holder, AnalysisContainer):
        arr = getattr(getattr(holder, "data_res_arr", None), "arr", None)
        dev = getattr(getattr(arr, "device", None), "id", None)
        gpus = None if dev is None else [int(dev)]
        ac = AnalysisContainer(_copy.copy(holder.data), _copy.copy(holder.sens_mat))
        return AnalysisContainerArray(ac, gpus=gpus)
    raise TypeError(
        f"holder must be an AnalysisContainerArray or AnalysisContainer, "
        f"got {type(holder).__name__}"
    )


class SOBBHDirectWDM:
    """SOBBH template directly in the WDM domain: batched response -> tracer -> lookup.

    Args:
        wdm_settings: the run's :class:`~lisatools.domains.WDMSettings` (active band + grid).
        table: :class:`~lisatools.domains.WDMLookupTable` with the SAME layer duration
            (``Nf * dt``); any sampling with that layer duration works.
        orbits, tdi_config, reference_time, n_grid, buffer_time, eval_dt, force_backend: see
            :class:`SOBBHBatchedTOF`.
        t_obs_start: absolute time [s] of grid pixel 0 (default ``wdm_settings.t0``; the stock
            domains carry an array-space ``t0 = 0`` while the data starts at ``data_t0``).
        num_m_layers: layers each side of the carrier layer per pixel (5 layers for 2).
        interp: table interpolation (``"cubic"`` / ``"linear"``).
    """

    def __init__(
        self,
        wdm_settings,
        table,
        *,
        orbits,
        tdi_config,
        reference_time,
        t_obs_start=None,
        n_grid=2048,
        buffer_time=5000.0,
        eval_dt=43200.0,
        num_m_layers=2,
        interp="cubic",
        force_backend="cpu",
    ):
        from ...wdm_lookup_eval import WDMLookupEvaluator

        if not np.isclose(float(table.layer_dt), float(wdm_settings.layer_dt), rtol=1e-9, atol=0):
            raise ValueError(
                f"lookup table layer_dt = {float(table.layer_dt):g} s but the grid's layer_dt = "
                f"{float(wdm_settings.layer_dt):g} s; the table must be built at the grid's "
                "layer duration (any Nf * dt with that product)"
            )
        self.wdm = wdm_settings
        self.ev = WDMLookupEvaluator(table, interp=interp, force_backend=force_backend)
        self.tof = SOBBHBatchedTOF(
            orbits,
            tdi_config,
            reference_time,
            n_grid=n_grid,
            buffer_time=buffer_time,
            eval_dt=eval_dt,
            force_backend=force_backend,
        )
        self.t_obs_start = float(wdm_settings.t0 if t_obs_start is None else t_obs_start)
        self.num_m_layers = int(num_m_layers)
        self.nchannels = int(tdi_config.nchannels)
        n_lo, n_hi = int(wdm_settings.ind_min_t), int(wdm_settings.ind_max_t)
        if int(wdm_settings.Nt_active) != n_hi - n_lo + 1:
            raise ValueError("WDMSettings.Nt_active != ind_max_t - ind_min_t + 1")
        self.n_pixels = np.arange(n_lo, n_hi + 1)
        self.t_pixels = self.t_obs_start + self.n_pixels * float(wdm_settings.layer_dt)
        self.last_stats = {}
        self.last_spans = {}

    def sparse(self, params, *, warn=True):
        """Lookup templates of the rows on their sparse support -> :class:`SparseWDMTemplate`.

        ``warn=False`` suppresses the per-call ``dropped_pixels`` warning (used by callers that
        batch many ``sparse`` calls and aggregate the stats themselves, e.g.
        :class:`SOBBHLookupComputations`, to avoid one warning per sub-batch)."""
        xp = self.ev.xp
        ws = self.wdm
        p = np.atleast_2d(np.asarray(asnumpy(params), dtype=float))
        N = p.shape[0]
        pad = 2.0 * self.tof.eval_dt
        t_a = time.perf_counter()
        out = self.tof.build(p, float(self.t_pixels[0]) - pad, float(self.t_pixels[-1]) + pad)
        synchronize(xp)
        t_b = time.perf_counter()
        amp, phase, f, fdot = sobbh_tracer(out, self.t_pixels)  # (N, nch, P)
        synchronize(xp)
        t_c = time.perf_counter()
        # Amendment 1: the TDI-on-the-fly amplitude is SIGNED (measured negative on most
        # channels of the merging test row), so liveness is amp != 0, never amp > 0.
        live = amp != 0.0
        n_live = xp.maximum(live.sum(axis=1), 1)
        f_ref = xp.where(live, f, 0.0).sum(axis=1) / n_live  # channel mean (N, P)
        m = xp.moveaxis(self.ev.layers_for(f_ref, self.num_m_layers), -1, 1)  # (N, L, P)
        n = xp.asarray(self.n_pixels)
        w, ok = self.ev.coeffs(
            amp[:, :, None, :],
            phase[:, :, None, :],
            f[:, :, None, :],
            fdot[:, :, None, :],
            n[None, None, None, :],
            m[:, None, :, :],
        )
        in_band = (m >= int(ws.ind_min_f)) & (m <= int(ws.ind_max_f))
        valid = ok & in_band[:, None, :, :] & live[:, :, None, :]
        w = xp.where(valid, w, 0.0)
        m_act = xp.clip(m - int(ws.ind_min_f), 0, int(ws.Nf_active) - 1)
        n_act = n - int(ws.ind_min_t)
        # A single-fdot table (nfdot == 1) has no chirp-rate axis at all -- every chirping pixel
        # would otherwise be miscounted as "dropped" although `coeffs` used it (the 1-point axis
        # is not a real bound); and the axis need not be symmetric, so an asymmetric lower bound
        # (fdot_min) must be checked too, not just fdot_max.
        if self.ev.nfdot > 1:
            fdot_bad = live & ((fdot < self.ev.fdot_min) | (fdot > self.ev.fdot_max))
        else:
            fdot_bad = xp.zeros_like(live)
        stats = dict(
            rows=int(N),
            pixels=int(self.n_pixels.size),
            lookup_pixels=int(asnumpy(live.any(axis=1).sum())),
            dropped_pixels=int(asnumpy(fdot_bad.sum())),
            merged_rows=int(asnumpy((~live.all(axis=(1, 2))).sum())),
            # live, in-table layers that fall outside the active band (a normal band-edge effect)
            dropped_layers=int(asnumpy((live[:, :, None, :] & ~in_band[:, None, :, :] & ok).sum())),
        )
        if warn and stats["dropped_pixels"] > 0:
            logger.warning(
                "SOBBHDirectWDM: %d channel-pixels of %d rows have |fdot| beyond the table's "
                "axis (%.3g Hz/s) and were dropped (sources near merger).",
                stats["dropped_pixels"],
                N,
                self.ev.fdot_max,
            )
        self.last_stats = stats
        synchronize(xp)
        self.last_spans = {
            "response": t_b - t_a,
            "tracer": t_c - t_b,
            "lookup": time.perf_counter() - t_c,
        }
        return SparseWDMTemplate(w, m_act, n_act, valid, stats)

    def dense(self, params):
        """Active-band :class:`~lisatools.domains.WDMSignal` per row (gates and tests)."""
        from ...domains import WDMSignal

        tpl = self.sparse(params)
        xp = self.ev.xp
        N = int(tpl.w.shape[0])
        nfa, nta = int(self.wdm.Nf_active), int(self.wdm.Nt_active)
        buf = xp.zeros(N * self.nchannels * nfa * nta)
        scatter_add(
            tpl,
            buf,
            xp.arange(N),
            xp.ones(N),
            nchannels=self.nchannels,
            Nf_active=nfa,
            Nt_active=nta,
        )
        buf = buf.reshape(N, self.nchannels, nfa, nta)
        return [WDMSignal(buf[i], self.wdm) for i in range(N)]


def sparse_inner_products(
    tpl,
    data_flat,
    invC_flat,
    data_index,
    noise_index,
    *,
    nchannels,
    Nf_active,
    Nt_active,
    tdi_type="XYZ",
):
    """``<d|h>`` and ``<h|h>`` per row on the template's sparse support.

    The chunked kernel's accumulation (``lat_chunked_het_kernels.hh``, ``wdm_het_get_ll_kernel``
    steps 7-8): ``d_h = sum_pix sum_{c,c'} d_c invC_{cc'} h_{c'}``, ``h_h = sum h_c invC_{cc'}
    h_{c'}`` with the full channel matrix for XYZ and the diagonal for AET / AE -- no
    ``4 * differential_component`` factor. ``data_flat`` / ``invC_flat`` are the ACA's flat
    per-walker slabs ``(nch, Nf_active, Nt_active)`` / ``(nch[, nch], Nf_active, Nt_active)``;
    ``data_index`` / ``noise_index`` ``(N,)`` pick the slabs. Returns ``(d_h, h_h)`` ``(N,)``.
    """
    if tdi_type not in ("XYZ", "AET", "AE"):
        raise ValueError(f"tdi_type must be one of ('XYZ', 'AET', 'AE'), got {tdi_type!r}")
    xp = get_array_module(tpl.w)
    nch, nfa, nta = int(nchannels), int(Nf_active), int(Nt_active)
    per_d = nch * nfa * nta
    if data_flat.size % per_d:
        raise ValueError(f"data buffer size {data_flat.size} is not a multiple of {per_d}")
    d = data_flat.reshape(data_flat.size // per_d, nch, nfa, nta)
    full = tdi_type == "XYZ"
    per_c = (nch * nch if full else nch) * nfa * nta
    if invC_flat.size % per_c:
        raise ValueError(f"invC buffer size {invC_flat.size} is not a multiple of {per_c}")
    nw_c = invC_flat.size // per_c
    inv = invC_flat.reshape((nw_c, nch, nch, nfa, nta) if full else (nw_c, nch, nfa, nta))
    r_d = xp.asarray(data_index, dtype=xp.int64)[:, None, None]
    r_c = xp.asarray(noise_index, dtype=xp.int64)[:, None, None]
    m = tpl.m_act
    n = xp.asarray(tpl.n_act, dtype=xp.int64)[None, None, :]
    w = tpl.w
    d_h = h_h = None
    for c in range(nch):
        dc = d[r_d, c, m, n]  # (N, L, P)
        for c2 in range(nch):
            if full:
                icc = inv[r_c, c, c2, m, n]
            elif c2 == c:
                icc = inv[r_c, c, m, n]
            else:
                continue
            dh = (dc * icc * w[:, c2]).sum(axis=(1, 2))
            hh = (w[:, c] * icc * w[:, c2]).sum(axis=(1, 2))
            d_h = dh if d_h is None else d_h + dh
            h_h = hh if h_h is None else h_h + hh
    return d_h, h_h


def scatter_add(tpl, buf_flat, data_index, factors, *, nchannels, Nf_active, Nt_active):
    """Accumulate ``factors[row] * w`` into the flat active-band buffer (per-walker slabs).

    ``buf_flat`` must be 1-D and C-contiguous (a view of the ACA's ``linear_data_arr[0]`` or a
    freshly allocated buffer); ``data_index`` ``(N,)`` picks each row's slab.
    """
    xp = get_array_module(buf_flat)
    nch, nfa, nta = int(nchannels), int(Nf_active), int(Nt_active)
    if buf_flat.ndim != 1 or not buf_flat.flags.c_contiguous:
        raise ValueError("fill target must be a 1-D C-contiguous buffer (writes go in place)")
    per = nch * nfa * nta
    if buf_flat.size % per:
        raise ValueError(f"fill target size {buf_flat.size} is not a multiple of {per}")
    buf = buf_flat.reshape(buf_flat.size // per, nch, nfa, nta)
    idx = (
        xp.asarray(data_index, dtype=xp.int64)[:, None, None, None],
        xp.arange(nch, dtype=xp.int64)[None, :, None, None],
        tpl.m_act[:, None, :, :],
        xp.asarray(tpl.n_act, dtype=xp.int64)[None, None, None, :],
    )
    vals = xp.asarray(factors, dtype=float)[:, None, None, None] * tpl.w
    if xp is np:
        np.add.at(buf, idx, vals)
    else:
        import cupyx

        cupyx.scatter_add(buf, idx, vals)


# ----------------------------------------------------------------------
# The comp that slots into SOBBHChunkedLikeMove
# ----------------------------------------------------------------------


class SOBBHLookupComputations:
    """Lookup-table SOBBH scorer with the comp surface ``SOBBHChunkedLikeMove`` consumes.

    Duck-types the parts of :class:`bbhx.sobbhcomps.SOBBHWDMComputations` the move and the
    engine signal generator use (``get_ll_wdm``, ``fill_global_wdm``, ``d_d``, ``wdm_settings``,
    ``xp`` / ``backend``, ``args`` / ``kwargs``, the ``d_h_out`` / ``h_h_out`` /
    ``last_call_spans`` stashes). Scoring convention identical to the kernel: returns
    ``-0.5 * (d_d + h_h - 2 d_h)`` with ``d_d = 0`` by default (the move adds the per-walker
    exposed-residual offset). ``m_band_half_width`` is accepted for signature compatibility and
    ignored: the lookup band is ``num_m_layers`` layers each side of the carrier layer.

    Args:
        wdm_settings: the run's :class:`~lisatools.domains.WDMSettings`.
        t_ref: reference epoch [s] of ``f_low`` / ``phi_c``.
        table: :class:`~lisatools.domains.WDMLookupTable` with the grid's layer duration.
        orbits, tdi_config, tdi_type, t_obs_start: as on ``WDMComputationsBase`` (``tdi_config``
            may be a string; ``None`` orbits -> ``EqualArmlengthOrbits``).
        n_grid, buffer_time, eval_dt, num_m_layers, interp: see :class:`SOBBHDirectWDM`.
        row_batch: rows per response build (bounds the spline memory).
        force_backend: backend name at construction (never per call).
        d_d: constant folded into the returned likelihood (default 0 = source-only).
    """

    _NPARAMS = 11

    def __init__(
        self,
        wdm_settings,
        t_ref,
        table,
        *,
        orbits=None,
        tdi_config=None,
        tdi_type="XYZ",
        t_obs_start=None,
        n_grid=2048,
        buffer_time=5000.0,
        eval_dt=43200.0,
        num_m_layers=2,
        interp="cubic",
        row_batch=32,
        force_backend="cpu",
        d_d=0.0,
    ):
        from ...detector import EqualArmlengthOrbits, Orbits
        from ...domains import WDMSettings
        from ...response.tdiconfig import TDIConfig

        if not isinstance(wdm_settings, WDMSettings):
            raise TypeError("wdm_settings must be a lisatools.domains.WDMSettings instance")
        if tdi_type not in ("XYZ", "AET", "AE"):
            raise ValueError(f"tdi_type must be one of 'XYZ', 'AET', 'AE'; got {tdi_type!r}")
        self._ctor_args = (wdm_settings, float(t_ref), table)
        self._ctor_kwargs = dict(
            orbits=orbits,
            tdi_config=tdi_config,
            tdi_type=tdi_type,
            t_obs_start=t_obs_start,
            n_grid=n_grid,
            buffer_time=buffer_time,
            eval_dt=eval_dt,
            num_m_layers=num_m_layers,
            interp=interp,
            row_batch=row_batch,
            force_backend=force_backend,
            d_d=d_d,
        )
        self.force_backend = (
            force_backend if isinstance(force_backend, str) else force_backend.name.split("_")[-1]
        )
        if orbits is None:
            orbits = EqualArmlengthOrbits(force_backend=self.force_backend)
        elif not isinstance(orbits, Orbits) and issubclass(orbits, Orbits):
            orbits = orbits()
        if tdi_config is None:
            tdi_config = TDIConfig("1st generation", force_backend=self.force_backend)
        elif isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config, force_backend=self.force_backend)
        self.wdm_settings = wdm_settings
        self.t_ref = float(t_ref)
        self.tdi_type = tdi_type
        self.t_obs_start = float(wdm_settings.t0 if t_obs_start is None else t_obs_start)
        self.d_d = float(d_d)
        self.row_batch = max(int(row_batch), 1)
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.nchannels = int(tdi_config.nchannels)
        self.direct = SOBBHDirectWDM(
            wdm_settings,
            table,
            orbits=orbits,
            tdi_config=tdi_config,
            reference_time=self.t_ref,
            t_obs_start=self.t_obs_start,
            n_grid=n_grid,
            buffer_time=buffer_time,
            eval_dt=eval_dt,
            num_m_layers=num_m_layers,
            interp=interp,
            force_backend=self.force_backend,
        )
        self.d_h_out = None
        self.h_h_out = None
        self.d_h_im_out = None
        self.last_call_spans = None
        self.last_stats = {}
        self._band_noted = False
        self._build_device = current_device(self.xp)

    @property
    def backend(self):
        import lisatools

        return lisatools.get_backend(self.force_backend)

    @property
    def xp(self):
        return self.direct.ev.xp

    @property
    def args(self):
        return self._ctor_args

    @property
    def kwargs(self):
        return dict(self._ctor_kwargs)

    def _note_band(self, m_band_half_width):
        if m_band_half_width is not None and not self._band_noted:
            logger.info(
                "SOBBHLookupComputations: m_band_half_width=%s ignored (the lookup "
                "band is num_m_layers=%d layers each side of the carrier layer)",
                m_band_half_width,
                self.direct.num_m_layers,
            )
            self._band_noted = True

    def _prep(self, params, data_index, noise_index, n_slots, convert_to_ra_dec):
        from ...response.directresponse import ecliptic_to_icrs, warn_deprecated_frame_conversion

        p = np.atleast_2d(np.asarray(asnumpy(params), dtype=float))
        if p.ndim != 2 or p.shape[1] != self._NPARAMS:
            raise ValueError(f"params must be (N, {self._NPARAMS}); got {p.shape}")
        N = p.shape[0]
        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            p = p.copy()
            p[:, -2], p[:, -1] = ecliptic_to_icrs(p[:, -2], p[:, -1])
        di = (
            np.zeros(N, dtype=np.int64)
            if data_index is None
            else np.asarray(asnumpy(data_index), dtype=np.int64).reshape(-1)
        )
        ni = (
            di
            if noise_index is None
            else np.asarray(asnumpy(noise_index), dtype=np.int64).reshape(-1)
        )
        if di.shape != (N,) or ni.shape != (N,):
            raise ValueError("data_index / noise_index must have one entry per row")
        if N and (di.min() < 0 or di.max() >= n_slots or ni.min() < 0 or ni.max() >= n_slots):
            raise IndexError(f"data_index / noise_index outside the holder's {n_slots} slots")
        return p, di, ni

    def get_ll_wdm(
        self,
        params,
        wdm_holder,
        data_index=None,
        noise_index=None,
        convert_to_ra_dec=None,
        grid_dim=0,
        use_layer_groups=True,
        group_band_layers=5,
        margin_layers=0,
        m_band_half_width=None,
    ):
        """Per-row ``-0.5 * (d_d + h_h - 2 d_h)`` against the holder's residual slabs.

        Stashes ``d_h_out`` / ``h_h_out`` (``d_h_im_out`` stays ``None``: no fused quadrature)
        and a ``last_call_spans`` dict (``stage``, ``geom``, ``wrap``, ``launch``, ``response``,
        ``tracer``, ``lookup`` (the three template stages, summed over the row batches),
        ``template`` (their sum), ``inner``, ``total``, ``num_bin``, ``n_groups``); the spans are
        device-synchronised.
        """
        t_entry = time.perf_counter()
        dev = current_device(self.xp)
        if dev != self._build_device:
            raise RuntimeError(
                f"SOBBHLookupComputations was built on device {self._build_device} but is being "
                f"called on device {dev}: the lookup comp is single-device (per-device replicas "
                "are a follow-up; use SOBBH_LIKELIHOOD=chunked for multi-GPU walker shards)"
            )
        self._note_band(m_band_half_width)
        holder = as_single_shard_holder(wdm_holder)
        data_flat = holder.linear_data_arr[0]
        invC_flat = holder.linear_psd_arr[0]
        p, di, ni = self._prep(params, data_index, noise_index, len(holder), convert_to_ra_dec)
        xp = self.xp
        ws = self.wdm_settings
        N = p.shape[0]
        t_stage = time.perf_counter() - t_entry
        d_h = xp.zeros(N)
        h_h = xp.zeros(N)
        totals = dict(lookup_pixels=0, dropped_pixels=0, merged_rows=0)
        t_build = t_inner = 0.0
        sp = dict(response=0.0, tracer=0.0, lookup=0.0)
        for lo in range(0, N, self.row_batch):
            hi = min(N, lo + self.row_batch)
            t0 = time.perf_counter()
            tpl = self.direct.sparse(p[lo:hi], warn=False)
            t_build += time.perf_counter() - t0
            for key in sp:
                sp[key] += self.direct.last_spans[key]
            t0 = time.perf_counter()
            a, b = sparse_inner_products(
                tpl,
                data_flat,
                invC_flat,
                di[lo:hi],
                ni[lo:hi],
                nchannels=self.nchannels,
                Nf_active=int(ws.Nf_active),
                Nt_active=int(ws.Nt_active),
                tdi_type=self.tdi_type,
            )
            d_h[lo:hi] = a
            h_h[lo:hi] = b
            synchronize(self.xp)
            t_inner += time.perf_counter() - t0
            for key in totals:
                totals[key] += int(self.direct.last_stats.get(key, 0))
        if totals["dropped_pixels"] > 0:
            logger.warning(
                "SOBBHDirectWDM: %d channel-pixels of %d rows have |fdot| beyond the table's "
                "axis (%.3g Hz/s) and were dropped (sources near merger).",
                totals["dropped_pixels"],
                N,
                self.direct.ev.fdot_max,
            )
        self.d_h_out = d_h
        self.h_h_out = h_h
        self.d_h_im_out = None
        self.last_stats = totals
        self.last_call_spans = {
            "stage": t_stage,
            "geom": 0.0,
            "wrap": 0.0,
            "launch": t_build + t_inner,
            "response": sp["response"],
            "tracer": sp["tracer"],
            "lookup": sp["lookup"],
            "template": t_build,
            "inner": t_inner,
            "total": time.perf_counter() - t_entry,
            "num_bin": int(N),
            "n_groups": 0,
        }
        return -0.5 * (self.d_d + h_h - 2.0 * d_h)

    def _fill_target(self, templates):
        from ...domains import DomainBase

        if hasattr(templates, "linear_data_arr"):
            if len(templates.linear_data_arr) != 1:
                raise NotImplementedError(
                    "fill_global_wdm writes into linear_data_arr[0] and is single-shard; route a "
                    "multi-shard ACA per split."
                )
            arr = templates.linear_data_arr[0]
        elif isinstance(templates, DomainBase):
            arr = templates.arr
        else:
            from ...analysiscontainer import AnalysisContainer

            if isinstance(templates, AnalysisContainer):
                # the container's OWN residual array object -- NOT as_single_shard_holder,
                # which (for a lone AnalysisContainer) wraps shallow COPIES of the data/sens_mat
                # holders so scoring stays safe; a fill through that wrapper would silently
                # write into a disposable copy and leave the real residual untouched.
                arr = templates.data_res_arr.arr
            else:
                arr = templates
        if not arr.flags.c_contiguous:
            raise ValueError("fill target must be C-contiguous (the fill writes in place)")
        ws = self.wdm_settings
        nfa, nta = int(ws.Nf_active), int(ws.Nt_active)
        if arr.ndim >= 3:
            expect = (self.nchannels, nfa, nta)
            got = tuple(arr.shape[-3:])
            if got != expect:
                raise ValueError(
                    f"fill target of shape {tuple(arr.shape)} must have trailing shape "
                    f"(nchannels, Nf_active, Nt_active) = {expect}; got {got} -- a dense "
                    "full-grid buffer (Nf, Nt instead of Nf_active, Nt_active) is rejected "
                    "rather than silently mis-sliced"
                )
        per = self.nchannels * nfa * nta
        flat = arr.reshape(-1)
        if flat.size % per:
            raise ValueError(
                f"fill target of shape {tuple(arr.shape)} is not a stack of ACTIVE-band "
                f"({self.nchannels}, {nfa}, {nta}) templates"
            )
        return flat, flat.size // per

    def fill_global_wdm(
        self,
        params,
        templates,
        convert_to_ra_dec=None,
        data_index=None,
        factors=None,
        grid_dim=0,
        m_band_half_width=None,
        band_slab_Nf=None,
        slab_min_f=None,
    ):
        """Accumulate ``factors * h`` into ``templates`` (ACA holder, ``WDMSignal``, or an
        active-band array ``(nch, Nf_active, Nt_active)`` / ``(num, nch, Nf_active, Nt_active)``
        / flat), slab per row by ``data_index`` (default 0)."""
        dev = current_device(self.xp)
        if dev != self._build_device:
            raise RuntimeError(
                f"SOBBHLookupComputations was built on device {self._build_device} but is being "
                f"called on device {dev}: the lookup comp is single-device (per-device replicas "
                "are a follow-up; use SOBBH_LIKELIHOOD=chunked for multi-GPU walker shards)"
            )
        self._note_band(m_band_half_width)
        if band_slab_Nf is not None or slab_min_f is not None:
            raise NotImplementedError("per-band slabs are not supported by the lookup fill")
        flat, n_slots = self._fill_target(templates)
        p, di, _ = self._prep(params, data_index, None, n_slots, convert_to_ra_dec)
        N = p.shape[0]
        if N == 0:
            return
        fac = (
            np.ones(N) if factors is None else np.asarray(asnumpy(factors), dtype=float).reshape(-1)
        )
        if fac.shape != (N,):
            raise ValueError(f"factors must have shape ({N},), got {fac.shape}")
        ws = self.wdm_settings
        dropped_total = 0
        t_entry = time.perf_counter()
        sp = dict(response=0.0, tracer=0.0, lookup=0.0)
        for lo in range(0, N, self.row_batch):
            hi = min(N, lo + self.row_batch)
            tpl = self.direct.sparse(p[lo:hi], warn=False)
            for key in sp:
                sp[key] += self.direct.last_spans[key]
            dropped_total += int(self.direct.last_stats.get("dropped_pixels", 0))
            scatter_add(
                tpl,
                flat,
                di[lo:hi],
                fac[lo:hi],
                nchannels=self.nchannels,
                Nf_active=int(ws.Nf_active),
                Nt_active=int(ws.Nt_active),
            )
            synchronize(self.xp)
        self.last_call_spans = {
            **sp,
            "template": sum(sp.values()),
            "total": time.perf_counter() - t_entry,
            "num_bin": int(N),
        }
        if dropped_total > 0:
            logger.warning(
                "SOBBHDirectWDM: %d channel-pixels of %d rows have |fdot| beyond the table's "
                "axis (%.3g Hz/s) and were dropped (sources near merger).",
                dropped_total,
                N,
                self.direct.ev.fdot_max,
            )


__all__ = [
    "SOBBHBatchedTOF",
    "SOBBHDirectWDM",
    "SOBBHLookupComputations",
    "SparseWDMTemplate",
    "as_single_shard_holder",
    "scatter_add",
    "sobbh_amp_phase_batch",
    "sobbh_tracer",
    "sparse_inner_products",
]
