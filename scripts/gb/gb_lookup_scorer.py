"""GB direct-to-WDM lookup template (prototype, xp-vectorised: numpy or cupy).

Reference-free GB scoring: the WDM coefficients of each source are read
straight from the n_ref lookup table at every (pixel, layer), so there is no
heterodyne reference, no anchor offset, no stash and no refresh. Design
(measured by lisa-sprint-2026-1e, 10-03, against a dense TD->WDM truth):

* Front end: the GB TDI-on-the-fly response at ``n_nodes`` control points
  per source (``GBTDIonTheFly``); each channel's envelope demodulated by the
  COMMON reference phase, ``w_c = amp_c e^{i tdi_phase_c}``, is splined in
  Re/Im (linear in the channels: the low-f X+Y+Z null survives), and the
  common phase ``phi_ref - 2 pi f0g t`` is splined once.
* ONE table read per (source, pixel, layer) at the common carrier
  ``(f_ref, fdot_ref) = (phi_ref', phi_ref'') / 2 pi`` -- the coefficient of
  channel c is ``cc Re(W_c) - ss Im(W_c)`` with ``W_c = w_c e^{i phi_ref}``,
  linear in the channels.
* First-order amplitude-slope term (the per-channel ``K_1`` term the
  per-channel polar lookup drops, which is what broke the null):
  ``(1/2 pi) d/df [cc Re(V_c) - ss Im(V_c)]`` with
  ``V_c = -i wdot_c e^{i phi_ref}`` and ``d/df`` the derivative of the same
  B-spline along the table's offset axis (no new table).

Conventions follow :class:`lisatools.wdm_lookup_eval.WDMLookupEvaluator`
(quarter-turn rule, seam-unbaked sin table, prefiltered cubic B-spline,
pixel centres ``t_n = t_obs_start + n * layer_dt`` with ``n`` absolute).
Inner products use the kernel convention (no 4*diff factor).
"""

from __future__ import annotations

import numpy as np


def _bspline_dweights(t):
    """d/dt of the uniform cubic B-spline weights (offsets -1, 0, 1, 2)."""
    s = 1.0 - t
    return (-0.5 * s * s, 1.5 * t * t - 2.0 * t, -1.5 * t * t + t + 0.5, 0.5 * t * t)


class GBLookupTemplate:
    """Lookup WDM templates of GB rows on a WDM grid's active band.

    Args:
        ev: a :class:`~lisatools.wdm_lookup_eval.WDMLookupEvaluator` built with
            ``interp="spline"`` from a table at the grid's layer duration.
        wdm: the grid's :class:`~lisatools.domains.WDMSettings`.
        orbits, tdi_config, t_ref: the GB response setup (as for
            :class:`~lisatools.response.tdionfly.GBTDIonTheFly`).
        t_obs_start: absolute time of pixel 0 (default ``wdm.t0``).
        n_nodes: control points over the analysed time span.
        num_m_layers: layers each side of the carrier layer (2 -> 5 layers).
        k1: include the amplitude-slope term.
        force_backend: "cpu" / "cuda12x" ... (the evaluator's backend wins).
    """

    def __init__(self, ev, wdm, *, orbits, tdi_config, t_ref, t_obs_start=None,
                 n_nodes=64, num_m_layers=2, k1=True, force_backend="cpu"):
        if ev.interp != "spline":
            raise ValueError("GBLookupTemplate needs the B-spline evaluator (interp='spline')")
        self.ev = ev
        self.wdm = wdm
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.t_ref = float(t_ref)
        self.t0 = float(wdm.t0 if t_obs_start is None else t_obs_start)
        self.n_nodes = int(n_nodes)
        self.L = int(num_m_layers)
        self.k1 = bool(k1)
        self.force_backend = force_backend
        n_lo, n_hi = int(wdm.ind_min_t), int(wdm.ind_max_t)
        self.n_pix = np.arange(n_lo, n_hi + 1)
        self.t_pix = self.t0 + self.n_pix * float(wdm.layer_dt)
        # nodes span the analysed pixels plus one layer each side (complete
        # response everywhere the splines are evaluated)
        self.t_nodes = np.linspace(self.t_pix[0] - wdm.layer_dt,
                                   self.t_pix[-1] + wdm.layer_dt, self.n_nodes)

    @property
    def xp(self):
        return self.ev.xp

    # ---- response at the nodes -> splines -> pixel quantities -------------
    def _front_end(self, params):
        from scipy.interpolate import CubicSpline
        from lisatools.response.tdionfly import GBTDIonTheFly

        p = np.atleast_2d(np.asarray(params, dtype=float))
        N = p.shape[0]
        Tspan = float(self.t_nodes[-1] - self.t_nodes[0])
        gen = GBTDIonTheFly(self.t_nodes, Tspan, self.t_ref, 1.0, N,
                            tdi_config=self.tdi_config, orbits=self.orbits,
                            force_backend="cpu")
        out = gen(*[p[:, j] for j in range(9)], convert_to_ra_dec=False)
        amp = np.asarray(out.tdi_amp)          # (N, 3, n)
        tph = np.asarray(out.tdi_phase)        # (N, 3, n)
        pref = np.asarray(out.phase_ref)       # (N, n)
        f0g = p[:, 1]
        tn, tp = self.t_nodes, self.t_pix
        w = amp * np.exp(1j * tph)
        dp = CubicSpline(tn, pref - 2 * np.pi * f0g[:, None] * tn[None, :], axis=1)
        phi = dp(tp) + 2 * np.pi * f0g[:, None] * tp[None, :]
        f_ref = (dp(tp, 1) + 2 * np.pi * f0g[:, None]) / (2 * np.pi)
        fdot_ref = dp(tp, 2) / (2 * np.pi)
        sr = CubicSpline(tn, w.real, axis=2)
        si = CubicSpline(tn, w.imag, axis=2)
        W = (sr(tp) + 1j * si(tp)) * np.exp(1j * phi)[:, None, :]
        V = -1j * (sr(tp, 1) + 1j * si(tp, 1)) * np.exp(1j * phi)[:, None, :]
        return W, V, f_ref, fdot_ref

    # ---- table read (+ offset-axis derivative) ----------------------------
    def _table(self, f, fdot, m, n):
        """(cc, ss, dcc/df, dss/df) after the quarter-turn rule; shapes of f."""
        ev, xp = self.ev, self.xp
        from lisatools.wdm_lookup_eval import _bspline_weights, _mirror_index

        delta = f - m * ev.layer_df
        ok = (delta >= ev.f_min) & (delta <= ev.f_max)
        u = (delta - ev.f_min) / ev.df
        if ev.nfdot > 1:
            ok = ok & (fdot >= ev.fdot_min) & (fdot <= ev.fdot_max)
            v = (fdot - ev.fdot_min) / ev.dfdot
        else:
            v = xp.zeros_like(u)
        i0 = xp.floor(u).astype(xp.int64)
        j0 = xp.floor(v).astype(xp.int64)
        wu, dwu = _bspline_weights(u - i0), _bspline_dweights(u - i0)
        wv = _bspline_weights(v - j0)
        out = {}
        for name, tab in (("c", ev.tab_cos), ("s", ev.tab_sin)):
            val = dval = None
            for a in range(4):
                ja = _mirror_index(xp, j0 + a - 1, ev.nfdot)
                row = drow = None
                for b in range(4):
                    ib = _mirror_index(xp, i0 + b - 1, ev.nf)
                    t = tab[ja, ib]
                    row = wu[b] * t if row is None else row + wu[b] * t
                    drow = dwu[b] * t if drow is None else drow + dwu[b] * t
                val = wv[a] * row if val is None else val + wv[a] * row
                dval = wv[a] * drow if dval is None else dval + wv[a] * drow
            out[name], out["d" + name] = val, dval / ev.df      # d/d(delta) = d/df
        c, s, dc, ds = out["c"], out["s"], out["dc"], out["ds"]
        if (ev.m_ref + ev.n_ref) % 2:
            c, dc = -c, -dc
        odd = ((m + n) % 2) != 0
        cc = xp.where(odd, s, c)
        ss = xp.where(odd, -c, s)
        dcc = xp.where(odd, ds, dc)
        dss = xp.where(odd, -dc, ds)
        z = xp.zeros_like(cc)
        return (xp.where(ok, cc, z), xp.where(ok, ss, z),
                xp.where(ok, dcc, z), xp.where(ok, dss, z))

    def coeffs(self, params):
        """``(w, m_abs)``: coefficients ``(N, 3, 2L+1, P)`` and their absolute
        layers ``(N, 2L+1, P)`` at the analysed pixels."""
        xp = self.xp
        W, V, f_ref, fdot_ref = (xp.asarray(a) for a in self._front_end(params))
        m = xp.floor(f_ref / self.ev.layer_df).astype(xp.int64)[:, None, :] + \
            xp.arange(-self.L, self.L + 1)[None, :, None]                 # (N, L, P)
        n = xp.asarray(self.n_pix)[None, None, :]
        cc, ss, dcc, dss = self._table(f_ref[:, None, :], fdot_ref[:, None, :], m, n)
        w = cc[:, None] * W.real[:, :, None] - ss[:, None] * W.imag[:, :, None]
        if self.k1:
            w = w + (dcc[:, None] * V.real[:, :, None]
                     - dss[:, None] * V.imag[:, :, None]) / (2 * np.pi)
        return w, m

    def slab(self, params, slab_lo, W_slab):
        """Dense per-row slabs ``(N, 3, W_slab, P)`` starting at absolute layer
        ``slab_lo[i]`` (layers outside a row's slab are dropped)."""
        xp = self.xp
        w, m = self.coeffs(params)
        N, nch, L, P = w.shape
        out = xp.zeros((N, nch, int(W_slab), P))
        rel = m - xp.asarray(np.asarray(slab_lo, dtype=np.int64))[:, None, None]
        for i in range(N):
            for l in range(L):
                idx = xp.flatnonzero((rel[i, l] >= 0) & (rel[i, l] < W_slab))
                for c in range(nch):
                    out[i, c, rel[i, l, idx], idx] += w[i, c, l, idx]
        return out
