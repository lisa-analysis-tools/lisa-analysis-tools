"""EMRI templates built directly in the WDM domain (lookup table + plunge chunk).

Harmonic tracks follow FEW's own conventions (few/waveform/base.py:357-364 and
few/summation/directmodesum.py):

* ``xI0 < 0`` is first mapped to ``(a, xI0) = (-a, +1)`` as FEW does; then for ``a > 0``
  the azimuthal phase is multiplied by ``sign(xI0)`` (a no-op after that mapping);
* for ``integrate_backwards`` the knot-end offset ``Phi[-1] + Phi[0]`` is added;
* the ``+m`` harmonic is ``Y_lm A e^{-i Phi}``; its ``-m`` partner (holder built with
  ``include_minus_mkn=True``, ``ylms`` of length ``2 * nmodes``) is
  ``(-1)^l Y_{l,-m} conj(A) e^{+i Phi}``, i.e. a track with ``amp = (-1)^l Y_{l,-m}
  conj(A)`` and ``phase, f, fdot, fddot`` all negated.

Frequencies and their derivatives come ANALYTICALLY from the integrator's derivative
spline (orders 1, 2, 3), never from differencing an unwrapped phase.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
from scipy.interpolate import CubicSpline


@dataclasses.dataclass
class HarmonicTrack:
    """One harmonic at pixel-centre times: h = amp * exp(-1j * phase)."""

    lmkn: tuple
    t: np.ndarray
    amp: np.ndarray
    phase: np.ndarray
    f: np.ndarray
    fdot: np.ndarray
    fddot: np.ndarray


def _phase_columns(integrator, t, order):
    if order == 0:
        cols = np.asarray(integrator.eval_integrator_spline(t))[:, 3:6]
    else:
        cols = np.asarray(integrator.eval_integrator_derivative_spline(t, order=order))[:, 3:6]
    return np.array(cols, dtype=float, copy=True)


def harmonic_tracks_from_holder(holder, integrator, t_pixels, *, a, xI0, phase_only=False):
    """Per-harmonic tracks at ``t_pixels`` [s, trajectory clock] from a FEW sparse holder.

    Args:
        holder: FEW ``SparseInfoHolder`` (``return_sparse_holder=True``).
        integrator: the trajectory integrator (``...inspiral_generator.inspiral_generator``)
            exposing ``eval_integrator_spline`` / ``eval_integrator_derivative_spline``.
        t_pixels: evaluation times, same clock as ``holder.t_arr``.
        a, xI0: spin and inclination cosine (retrograde convention).
        phase_only: skip the frequency derivatives (``f``, ``fdot``, ``fddot`` are None): the
            response feed needs amplitude and phase only.
    Vectorised over modes (one matrix product per derivative order).
    """
    t_pixels = np.asarray(t_pixels, dtype=float)
    if xI0 < 0:   # FEW's internal convention first (few/waveform/base.py:208-212)
        a, xI0 = -a, -xI0
    orders = (0,) if phase_only else (0, 1, 2, 3)
    P = [_phase_columns(integrator, t_pixels, k) for k in orders]
    if a > 0:
        for arr in P:
            arr[:, 0] *= np.sign(xI0)
    if holder.integrate_backwards:
        knots = _phase_columns(integrator, np.asarray(holder.t_arr, dtype=float), 0)
        if a > 0:
            knots[:, 0] *= np.sign(xI0)
        P[0] = P[0] + (knots[-1] + knots[0])[None, :]

    t_k = np.asarray(holder.t_arr, dtype=float)
    teuk = np.asarray(holder.teuk_modes)
    nm = teuk.shape[1]
    A = CubicSpline(t_k, teuk.real, axis=0)(t_pixels) + 1j * CubicSpline(t_k, teuk.imag, axis=0)(t_pixels)
    ylms = np.asarray(holder.ylms)
    with_minus = ylms.shape[0] == 2 * nm
    ls, ms, ks, ns = (np.asarray(x) for x in (holder.ls, holder.ms, holder.ks, holder.ns))
    Wm = np.stack([ms, ks, ns]).astype(float)                       # (3, nm)
    D = [Pk @ Wm for Pk in P]                                       # each (T, nm)
    ph = D[0]
    f, fd, fdd = ((D[k] / (2 * np.pi)) for k in (1, 2, 3)) if not phase_only else (None, None, None)
    amp = A * ylms[None, :nm]
    tracks = []
    for j in range(nm):
        lmkn = (int(ls[j]), int(ms[j]), int(ks[j]), int(ns[j]))
        tracks.append(HarmonicTrack(lmkn, t_pixels, amp[:, j], ph[:, j],
                                    None if f is None else f[:, j], None if fd is None else fd[:, j],
                                    None if fdd is None else fdd[:, j]))
    minus = []
    if with_minus:
        jm = np.flatnonzero(ms != 0)
        amp_m = ((-1.0) ** ls[jm])[None, :] * ylms[None, nm + jm] * np.conj(A[:, jm])
        for c, j in enumerate(jm):
            minus.append(HarmonicTrack((int(ls[j]), -int(ms[j]), -int(ks[j]), -int(ns[j])), t_pixels,
                                       amp_m[:, c], -ph[:, j], None if f is None else -f[:, j],
                                       None if fd is None else -fd[:, j], None if fdd is None else -fdd[:, j]))
    return tracks + minus


# ----------------------------------------------------------------------
# Handoff from the lookup to the plunge chunk
# ----------------------------------------------------------------------

#: Half-support of the WDM wavelet in units of layer_dt used by the curvature trigger.
#: Calibrated in Task A6 (scripts/emri/emri_single_harmonic_gate.py): the exact local-chirp
#: model error (1.9e-4 median over the inspiral) passes the table-interpolation error
#: (~3-5e-3) near 0.9 t_plunge, where (pi/3)|fddot| layer_dt^3 ~ 1.2e-3 rad; a 0.1 rad
#: trigger there needs tau ~ 4 layer_dt (the WDM wavelet's tails are wide).
WDM_HALF_SUPPORT_LAYERS = 4.0
def handoff_pixel(track, layer_dt, layer_df, fdot_axis_max, tol_rad=0.1):
    """First pixel index where the local-quadratic lookup stops being valid.

    Trips on the cubic phase term ``(pi/3) |fddot| tau^3 > tol_rad`` (tau = the wavelet
    half-support) OR on ``|fdot|`` leaving the table's fdot axis, whichever comes
    first. Never on fdot alone: the dominant harmonic can stay inside the fdot axis
    and still fail on curvature. Returns ``len(track.t)`` if it never trips.
    """
    tau = WDM_HALF_SUPPORT_LAYERS * layer_dt
    cubic = (np.pi / 3.0) * np.abs(np.asarray(track.fddot)) * tau ** 3
    trip = (cubic > tol_rad) | (np.abs(np.asarray(track.fdot)) > fdot_axis_max)
    idx = np.flatnonzero(trip)
    return int(idx[0]) if idx.size else int(np.asarray(track.t).size)


def plunge_chunk_wdm(td_tail_fn, n_h, Nt, Nf, dt, Nt_sub=128, n_end=None, ind_max_t=None, backend="cpu"):
    """WDM of the plunge tail ``[n_h, n_end)`` from ONE even-start chunk.

    Args:
        td_tail_fn: ``(start_sample, n_samples) -> (nch, n_samples)`` dense TD of the
            signal over the chunk window only (no window/taper).
        n_h: first pixel handed off to the chunk; n_end: one past the last pixel to
            fill (default ``Nt``).
        ind_max_t: the production time crop's last kept pixel + 1; a chunk reaching
            past it would carry a tapered tail, so this raises instead.
    Returns ``(chunk, n0, keep_lo, keep_hi)`` for :func:`lisatools.wdm_het.splice_chunk`.
    """
    from ...wdm_het import chunk_start_for_pixels, wdm_chunk_of_td

    n_end = int(Nt if n_end is None else n_end)
    # Edge contamination decays algebraically from each chunk edge (Nt_sub=128: 0.2 at
    # the edge, 5e-5 at 16 px, 6e-6 at 24 px; a Tukey does not help) -> discard a
    # quarter of the chunk on each side (tests/test_wdm_chunk_splice.py measures it).
    n_pad = Nt_sub // 4
    n0 = chunk_start_for_pixels(n_h, n_end, Nt, Nt_sub, n_pad)
    if ind_max_t is not None and n0 + Nt_sub > int(ind_max_t):
        raise ValueError(
            f"plunge chunk [{n0}, {n0 + Nt_sub}) runs into the production time crop "
            f"(ind_max_t={ind_max_t}); it would include tapered/cropped samples")
    td = td_tail_fn(n0 * int(Nf), int(Nf) * int(Nt_sub))
    chunk = wdm_chunk_of_td(td, 0, Nf, Nt_sub, dt, backend=backend)
    return chunk, n0, int(n_h) - n0, n_end - n0


# ----------------------------------------------------------------------
# Per-channel tracer from the TDI-on-the-fly output, and the full assembly
# ----------------------------------------------------------------------

def tracer_from_tof_output(out, t_pixels, h=30.0, h_fdot=None):
    """Per-sub, per-channel ``(amp, phase, f, fdot)`` at ``t_pixels`` [absolute s].

    The channel signal is ``Re[amp exp(-i phase)]`` with ``phase = tdi_phase + phase_ref``
    (``TDTDIOutput.eval_tdi``). ``f`` and ``fdot`` are central differences (step ``h``)
    of that CONTINUOUS spline phase, so they carry the Doppler shift of the channel
    (the source-frame f is off by ~1e-4 f, a ~1% pixel error). A negative-frequency
    sub (a -m partner) is mirrored to positive frequency: cos is even, so
    ``phase -> -phase``, ``f -> -f``, ``fdot -> -fdot``.
    Returns arrays of shape ``(num_sub, nch, P)``.
    """
    xp = getattr(out, "xp", np)
    t = xp.asarray(t_pixels, dtype=float)
    hd = float(os.environ.get("EMRI_TRACER_H_FDOT", "300")) if h_fdot is None else float(h_fdot)
    P = t.size
    # ONE spline evaluation at all five stencils (t, t +- h, t +- hd)
    amp_all, tph, pref = out.eval_spline_vals(xp.concatenate([t, t + h, t - h, t + hd, t - hd]))
    ph = xp.asarray(tph) + xp.asarray(pref)[:, None, :]
    ph0, php, phm, phpd, phmd = (ph[..., k * P:(k + 1) * P] for k in range(5))
    amp = xp.asarray(amp_all)[..., :P]
    f = (php - phm) / (2 * h) / (2 * np.pi)
    fdot = (phpd - 2 * ph0 + phmd) / hd ** 2 / (2 * np.pi)   # phase ~1e5 rad: a short
    #                                                   step turns roundoff into fdot noise
    neg = f < 0
    return amp, xp.where(neg, -ph0, ph0), xp.where(neg, -f, f), xp.where(neg, -fdot, fdot)


def _scatter_add(xp, acc, idx, vals):
    """acc[idx] += vals with repeated indices accumulated (numpy add.at / cupyx scatter_add)."""
    if xp is np:
        np.add.at(acc, idx, vals)
    else:
        import cupyx
        cupyx.scatter_add(acc, idx, vals)


def accumulate_harmonic_batch(acc, table, tracks, tracer, n_ok, tail_td, *, Nf, Nt, dt, layer_dt,
                              layer_df, t0, Nt_sub=128, num_m_layers=2, fdot_axis_max=np.inf,
                              pixel_edge=8, backend="cpu", sub_row=None):
    """Vectorised :func:`_accumulate_harmonic_batch_loop`: ONE table evaluation for every
    (sub, channel, pixel) of the batch and one scatter-add, on the array module of ``acc``
    (numpy or cupy). Same arguments, same result up to summation order.

    ``sub_row`` (optional, length num_sub): the template each sub belongs to; ``acc`` is then
    ``(n_templates, nch, Nf, Nt)`` and many templates accumulate in one call (no plunge
    chunks allowed in that mode)."""
    from ...wdm_het import tail_chunk_plan, wdm_chunk_of_td
    from ...utils.utility import get_array_module

    xp = get_array_module(acc)
    amp, phase, f, fdot = (xp.asarray(a) for a in tracer)
    S, nch, P = amp.shape
    n_ok_x = xp.asarray(n_ok)
    stats = dict(lookup_pixels=0, chunk_pixels=0, dropped_pixels=0)
    k_h = np.array([handoff_pixel(tr, layer_dt, layer_df, fdot_axis_max) for tr in tracks])
    before = xp.arange(P)[None, :] < xp.asarray(k_h)[:, None]                       # (S, P)
    sel = before[:, None, :] & (xp.abs(fdot) <= fdot_axis_max) & (f > 2 * layer_df)  # (S, C, P)
    stats["dropped_pixels"] = int(nch * int(before.sum()) - int(sel.sum()))
    s_i, c_i, p_i = xp.nonzero(sel)
    if s_i.size:
        co, mm = table.get_wdm_coeffs(amp[sel], phase[sel], f[sel], fdot[sel], n_ok_x[p_i],
                                      num_m_layers=num_m_layers, out_of_support="zero")
        co, mm = xp.asarray(co), xp.asarray(mm)
        n_e = n_ok_x[p_i]
        r_i = None if sub_row is None else xp.asarray(sub_row)[s_i]
        for c in range(co.shape[1]):
            good = mm[:, c] >= 0
            idx = (c_i[good], mm[good, c], n_e[good])
            if r_i is not None:
                idx = (r_i[good],) + idx
            _scatter_add(xp, acc, idx, co[good, c])
        stats["lookup_pixels"] = int(s_i.size)
    windows = {}
    n_ok_h = np.asarray(n_ok.get() if hasattr(n_ok, "get") else n_ok)
    for s in range(S):
        if k_h[s] < n_ok_h.size:
            n_h = int(n_ok_h[k_h[s]])
            n_end = min(int(n_ok_h[-1]) + 1 + Nt_sub // 4, Nt - pixel_edge)
            for n0, klo, khi in tail_chunk_plan(n_h, n_end, Nt, Nt_sub):
                windows.setdefault(n0, []).append((s, klo, khi))
    if windows and sub_row is not None:
        raise NotImplementedError("plunge chunks are not batched across templates; build that template alone")
    for n0, items in windows.items():
        ts = t0 + (n0 * Nf + xp.arange(Nf * Nt_sub)) * dt
        td_all = xp.asarray(tail_td(ts))
        for s, klo, khi in items:
            chunk = xp.asarray(wdm_chunk_of_td(td_all[s], 0, Nf, Nt_sub, dt, backend=backend))
            acc[:, :chunk.shape[-2], n0 + klo:n0 + khi] += chunk[:, :, klo:khi]
            stats["chunk_pixels"] += khi - klo
    return stats


def _accumulate_harmonic_batch_loop(acc, table, tracks, tracer, n_ok, tail_td, *, Nf, Nt, dt, layer_dt,
                                    layer_df, t0, Nt_sub=128, num_m_layers=2, fdot_axis_max=np.inf,
                                    pixel_edge=8, backend="cpu"):
    """Add one batch of harmonics to ``acc`` (``(nch, Nf, Nt)``, global layers x pixels).

    Per sub ``s`` (``tracks[s]``, tracer row ``s``): the n_ref lookup on pixels before its
    handoff pixel, then even-start chunks of its own dense TD over ``[n_h, n_end)``
    (``n_end`` = a quarter chunk past the last tracked pixel: the abrupt stop at plunge
    rings ~24 px). The tail TD is evaluated ONCE per chunk window for all subs that need it
    (``tail_td(ts) -> (num_sub, nch, ts.size)``, zero outside the response support) and
    chunks are added in place over their kept slice (no full-grid temporaries).

    Args:
        tracer: ``(amp, phase, f, fdot)`` from :func:`tracer_from_tof_output`,
            each ``(num_sub, nch, n_ok.size)``.
        n_ok: pixel indices of the tracer/track samples (ascending).
        t0: absolute time of pixel 0.
    Returns a stats dict (lookup pixels, chunk pixels, pixels dropped before the handoff
    because the channel fdot left the table or f was below 2 layers).
    """
    from ...wdm_het import tail_chunk_plan, wdm_chunk_of_td

    amp, phase, f, fdot = tracer
    nch = amp.shape[1]
    n_ok = np.asarray(n_ok)
    stats = dict(lookup_pixels=0, chunk_pixels=0, dropped_pixels=0)
    windows = {}
    for s, tr in enumerate(tracks):
        k_h = handoff_pixel(tr, layer_dt, layer_df, fdot_axis_max)
        before = np.arange(n_ok.size) < k_h
        for ch in range(nch):
            sel = before & (np.abs(fdot[s, ch]) <= fdot_axis_max) & (f[s, ch] > 2 * layer_df)
            stats["dropped_pixels"] += int(np.sum(before & ~sel))
            if np.any(sel):
                co, mm = table.get_wdm_coeffs(amp[s, ch, sel], phase[s, ch, sel], f[s, ch, sel],
                                              fdot[s, ch, sel], n_ok[sel], num_m_layers=num_m_layers,
                                              out_of_support="zero")
                co, mm = np.asarray(co), np.asarray(mm)
                for c in range(co.shape[1]):
                    good = mm[:, c] >= 0
                    np.add.at(acc[ch], (mm[good, c], n_ok[sel][good]), co[good, c])
                stats["lookup_pixels"] += int(sel.sum())
        if k_h < n_ok.size:
            n_h = int(n_ok[k_h])
            n_end = min(int(n_ok[-1]) + 1 + Nt_sub // 4, Nt - pixel_edge)
            for n0, klo, khi in tail_chunk_plan(n_h, n_end, Nt, Nt_sub):
                windows.setdefault(n0, []).append((s, klo, khi))
    for n0, items in windows.items():
        ts = t0 + (n0 * Nf + np.arange(Nf * Nt_sub)) * dt
        td_all = np.asarray(tail_td(ts))                     # one evaluation per window
        for s, klo, khi in items:
            chunk = np.asarray(wdm_chunk_of_td(td_all[s], 0, Nf, Nt_sub, dt, backend=backend))
            acc[:, :chunk.shape[-2], n0 + klo:n0 + khi] += chunk[:, :, klo:khi]   # chunk rows = global layers
            stats["chunk_pixels"] += khi - klo
    return stats


def dense_inputs_from_holder(holder, integrator, *, a, xI0):
    """Inputs of :class:`lisatools.response.tdionfly.TDDenseTDIonTheFly` for one template.

    Returns ``(t_k (K,), phase_coeffs (K-1, 3, 8), mkn (S, 3), amp_re (S, K-1, 4),
    amp_im (S, K-1, 4))`` with the harmonics in :func:`harmonic_tracks_from_holder`'s order
    (m >= 0 modes, then the -m partners). The DOPR853 dense-output coefficients carry every
    convention of ``harmonic_tracks_from_holder`` (massratio scaling, the integrator's backwards
    adjustment, sign(xI0) on Phi_phi for a > 0, the holder's backwards offset), so the dense
    polynomial IS the track phase; the amplitude splines are the tracks' (scipy CubicSpline over
    the knots, monomial coefficients in ``t - t_knot``).
    """
    if xI0 < 0:
        a, xI0 = -a, -xI0
    t_int = np.asarray(integrator.integrator_t_cache, dtype=float).copy()
    t_k = np.asarray(holder.t_arr, dtype=float).copy()     # amplitude knots (FEW cuts the last at T)
    C = np.array(integrator.integrator_spline_coeff, dtype=float)[:, 3:6, :] / integrator.massratio
    if integrator.integrate_backwards:              # eval_integrator_spline's adjustment
        traj = np.asarray(integrator.trajectory)
        C[:, :, 0] += (traj[0, 4:7] + traj[-1, 4:7])[None, :]
    if a > 0:
        C[:, 0, :] *= np.sign(xI0)
    if holder.integrate_backwards:                  # base.py:361-364 (as harmonic_tracks_from_holder)
        knots = _phase_columns(integrator, t_k, 0)
        if a > 0:
            knots[:, 0] *= np.sign(xI0)
        C[:, :, 0] += (knots[-1] + knots[0])[None, :]
    C = _dense_on_grid(t_int, C, t_k)                        # phases re-expressed on the amplitude knots
    teuk = np.asarray(holder.teuk_modes)
    nm = teuk.shape[1]
    cr = CubicSpline(t_k, teuk.real, axis=0).c[::-1]         # (4, K-1, nm), a0..a3
    ci = CubicSpline(t_k, teuk.imag, axis=0).c[::-1]
    ylms = np.asarray(holder.ylms)
    ls, ms, ks, ns = (np.asarray(x) for x in (holder.ls, holder.ms, holder.ks, holder.ns))
    mkn, are, aim = [], [], []
    for j in range(nm):                                      # A * Ylm
        y = ylms[j]
        are.append((y.real * cr[:, :, j] - y.imag * ci[:, :, j]).T)
        aim.append((y.real * ci[:, :, j] + y.imag * cr[:, :, j]).T)
        mkn.append((ms[j], ks[j], ns[j]))
    if ylms.shape[0] == 2 * nm:                              # (-1)^l Y_{l,-m} conj(A), phase -Phi
        for j in np.flatnonzero(ms != 0):
            y = ((-1.0) ** ls[j]) * ylms[nm + j]
            are.append((y.real * cr[:, :, j] + y.imag * ci[:, :, j]).T)
            aim.append((-y.real * ci[:, :, j] + y.imag * cr[:, :, j]).T)
            mkn.append((-ms[j], -ks[j], -ns[j]))
    return t_k, C, np.array(mkn, dtype=np.int32), np.array(are), np.array(aim)


def _dense_basis(s):
    """The DOPR853 nested dense-output basis at s (n,) -> (n, 8): the polynomial is basis @ r."""
    s1 = 1.0 - s
    return np.stack([np.ones_like(s), s, s * s1, s ** 2 * s1, s ** 2 * s1 ** 2, s ** 3 * s1 ** 2,
                     s ** 3 * s1 ** 3, s ** 4 * s1 ** 3], axis=-1)


def _dense_on_grid(t_int, C, t_new):
    """DOPR853 coefficients ``C`` (n_int - 1, P, 8) on knots ``t_int`` re-expressed on knots
    ``t_new`` (a subset of ``t_int`` plus a cut inside a segment, e.g. FEW's final point at T).

    Segments shared by both grids keep their coefficients; any other is refit EXACTLY (the 8
    nested basis functions span all degree-7 polynomials) from 16 Chebyshev points of the
    original polynomial."""
    t_new = np.asarray(t_new, dtype=float)
    out = np.empty((t_new.size - 1,) + C.shape[1:])
    j = np.clip(np.searchsorted(t_int, t_new[:-1], side="right") - 1, 0, t_int.size - 2)
    same = (np.abs(t_int[j] - t_new[:-1]) <= 1e-9 * np.abs(t_new[:-1]).max()) & \
           (np.abs(t_int[j + 1] - t_new[1:]) <= 1e-9 * np.abs(t_new[1:]).max())
    out[same] = C[j[same]]
    x = 0.5 * (1 - np.cos(np.pi * (np.arange(16) + 0.5) / 16))
    B = _dense_basis(x)
    for q in np.flatnonzero(~same):
        ts = t_new[q] + x * (t_new[q + 1] - t_new[q])
        vals = dense_phase_eval(t_int, C, ts)                # (16, P)
        out[q] = np.linalg.lstsq(B, vals, rcond=None)[0].T
    return out


def dense_phase_eval(t_k, C, t):
    """Evaluate DOPR853 dense-output coefficients ``C`` (K-1, P, 8) at ``t`` (reference for tests)."""
    t = np.asarray(t, dtype=float)
    seg = np.clip(np.searchsorted(t_k, t, side="right") - 1, 0, t_k.size - 2)
    s = ((t - t_k[seg]) / np.diff(t_k)[seg])[:, None]
    s1 = 1.0 - s
    c = C[seg]
    return c[..., 0] + s * (c[..., 1] + s1 * (c[..., 2] + s * (c[..., 3] + s1 * (c[..., 4] + s * (c[..., 5] + s1 * (c[..., 6] + s * c[..., 7]))))))


def feed_from_tracks(tracks, amp_factor):
    """Per-sub (amp, phase) with h = sum amp exp(-i phase) from harmonic tracks on the feed grid.

    A track term is ``track.amp * exp(-i track.phase)`` (FEW's mode sum, -m partners included),
    so amp = |track.amp| * amp_factor and phase = track.phase - unwrap(arg(track.amp))."""
    amp = np.stack([np.abs(tr.amp) for tr in tracks]) * amp_factor
    phase = np.stack([tr.phase - np.unwrap(np.angle(tr.amp)) for tr in tracks])
    return amp, phase


def slice_holder(H, idx):
    """The host holder restricted to modes ``idx`` (m >= 0 rows; the -m partners' Ylm follow)."""
    import types

    idx = np.asarray(idx)
    nm = len(H.ls)
    ylms = np.asarray(H.ylms)
    yl = np.concatenate([ylms[idx], ylms[nm + idx]]) if ylms.shape[0] == 2 * nm else ylms[idx]
    return types.SimpleNamespace(t_arr=H.t_arr, teuk_modes=np.asarray(H.teuk_modes)[:, idx], phases=H.phases,
                                 freqs=H.freqs, ylms=yl, ls=np.asarray(H.ls)[idx], ms=np.asarray(H.ms)[idx],
                                 ks=np.asarray(H.ks)[idx], ns=np.asarray(H.ns)[idx],
                                 integrate_backwards=H.integrate_backwards)


def track_rows(H, idx):
    """Rows of ``harmonic_tracks_from_holder(H, ...)`` for modes ``idx``, in the order the same
    function (and ``EMRITDIonFly.mode_amp_phase``) gives for ``slice_holder(H, idx)``: the m >= 0
    modes, then the -m partners of those with m != 0."""
    ms = np.asarray(H.ms)
    nm = ms.size
    with_minus = np.asarray(H.ylms).shape[0] == 2 * nm
    minus_row = np.full(nm, -1)
    minus_row[ms != 0] = nm + np.arange(int(np.sum(ms != 0)))
    rows = list(np.asarray(idx))
    if with_minus:
        rows += [int(minus_row[i]) for i in idx if ms[i] != 0]
    return rows


def subset_track(tr, pos):
    """A HarmonicTrack restricted to sample positions ``pos``."""
    return HarmonicTrack(tr.lmkn, tr.t[pos], tr.amp[pos], tr.phase[pos], tr.f[pos], tr.fdot[pos], tr.fddot[pos])


class EMRIDirectWDM:
    """EMRI template built directly in the WDM domain.

    Per harmonic: the n_ref lookup up to its handoff pixel (curvature or fdot range),
    then even-start chunks of its own dense TDI-on-the-fly TD over the plunge tail.
    Harmonics are processed in batches of ``mode_batch`` so the per-harmonic response
    splines never all live at once (the scale-up constraint).

    Args:
        few_gen: FEW generator (the production one; its inspiral options are restored
            after every call by EMRITDIonFly).
        table: :class:`lisatools.domains.WDMLookupTable` built on the same Nf, dt.
        wdm_set: target :class:`WDMSettings` (grid starts at ``data_t0``).
        orbits, tdi_config: as for :class:`EMRITDIonFly` (ICRS orbits, special frame).
        t_start: FEW reference epoch (trajectory clock origin), e.g. MOJITO_REFERENCE_TIME.
        data_t0: absolute time of WDM pixel 0.
        n_fine: fixed number of fine trajectory points over the window. Default ``None``:
            one per ``fine_dt`` (3600 s: direct-vs-production unchanged from 80 s to 3600 s; 6 months: 1800 and 3600 s identical, mm 1e-7..4e-7
            on CD1L EMRI 1, 16 d), or one per ``fine_dt_plunge`` (80 s) when the
            trajectory ends inside the window or the modes are given explicitly (the
            plunge chunk reads the dense response from these splines).
        pixel_edge: pixels dropped at each grid end (response spline support).
        interp: table interpolation (``"spline"`` default: uniform cubic B-spline, CPU and GPU;
            ``"cubic"`` scipy-only; ``None`` keeps the table's).
    """

    def __init__(self, few_gen, table, wdm_set, *, orbits, tdi_config, t_start, data_t0,
                 Nt_sub=128, n_fine=None, fine_dt=3600.0, fine_dt_plunge=80.0, mode_batch=None,
                 pixel_edge=8, num_m_layers=2, interp="spline", force_backend="cpu", feed="knots",
                 response="spline"):
        self.few_gen, self.table, self.wdm = few_gen, table, wdm_set
        self.orbits, self.tdi_config = orbits, tdi_config
        self.t_start, self.data_t0 = float(t_start), float(data_t0)
        self.Nt_sub, self.pixel_edge = int(Nt_sub), int(pixel_edge)
        self.mode_batch = None if mode_batch is None else int(mode_batch)   # None: all modes in one batch
        # layers m-num_m_layers..m+num_m_layers per pixel: a chirping carrier near a layer edge
        # puts ~2.5e-4 of its energy two layers away (A8 gate); needs table support [-2, 3) df
        self.num_m_layers = int(os.environ.get("EMRI_DIRECT_NUM_M_LAYERS", num_m_layers))
        span = wdm_set.Nt * wdm_set.layer_dt
        if feed not in ("knots", "fine"):
            raise ValueError("feed must be 'knots' or 'fine'")
        # "knots": FEW amplitudes at the integrator knots, splined to the response grid, phases
        # from the integrator's dense output (production's construction; the cheap FEW call).
        # "fine": FEW evaluates amplitudes at every fine point (the earlier feed).
        self.feed = feed
        if response not in ("spline", "dense"):
            raise ValueError("response must be 'spline' or 'dense'")
        # "dense": TDDenseTDIonTheFly (exact dense-output phases, geometry shared across the
        # harmonics of a template, many templates per launch); needs feed="knots" and a backend
        # module built with TDDenseTDIWaveformWrap. "spline": EMRITDIonFly -> TDTDIonTheFly.
        self.response = response
        if response == "dense" and feed != "knots":
            raise ValueError("response='dense' needs feed='knots'")
        self.n_fine_fixed = int(n_fine) if n_fine is not None else None
        self.fine_dt, self.fine_dt_plunge = float(fine_dt), float(fine_dt_plunge)
        self.n_fine = self.n_fine_fixed or max(64, int(span / self.fine_dt_plunge))   # set per call
        self.force_backend = force_backend
        from ...utils.utility import asnumpy
        from ... import get_backend
        self.xp = get_backend(force_backend).xp
        self.fdot_axis_max = float(np.max(np.abs(asnumpy(table.fdot_vals))))
        # cubic table interpolation: linear left a ~2.5e-4 amplitude deficit (A9, EMRI 1).
        # NOTE: this switches the passed table's interpolators (set_interp_method).
        if interp is not None and getattr(table, "INTERP_METHOD", None) != interp:
            table.set_interp_method(interp)
        self.last_stats = {}

    def _few_holder(self, few_args, few_kwargs, new_t, mode_selection=None):
        """ONE FEW call on the fine grid ``new_t`` (FEW clock) -> host sparse holder.

        Mode selection by ``few_kwargs['mode_selection_threshold']`` unless ``mode_selection``
        is given. The generator's inspiral_kwargs are restored afterwards (FEW merges
        call-time ones permanently)."""
        from few.utils.utility import get_viewing_angles

        from .emritdionfly import host_holder

        m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = few_args[:14]
        th, ph = get_viewing_angles(qS, phiS, qK, phiK)
        kw = {k: v for k, v in few_kwargs.items() if k != "inspiral_kwargs"}
        if mode_selection is not None:
            kw.pop("mode_selection_threshold", None)
            kw["mode_selection"] = [tuple(int(v) for v in md) for md in mode_selection]
        new_t = np.asarray(new_t, dtype=float)
        saved = dict(self.few_gen.inspiral_kwargs)
        try:
            H = self.few_gen(m1, m2, a, p0, e0, x0, th, ph, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
                             Phi_r0=Pr, T=float(new_t[-1]) / 3.15581497635456e7, dt=self.wdm.data_dt,
                             return_sparse_holder=True, include_minus_mkn=True,
                             inspiral_kwargs={"upsample": True, "fix_t": True, "new_t": new_t}, **kw)
        finally:
            self.few_gen.inspiral_kwargs.clear()
            self.few_gen.inspiral_kwargs.update(saved)
        return host_holder(H)                               # GPU generator: cupy -> host

    def _few_knots(self, few_args, few_kwargs, mode_selection=None):
        """ONE FEW call on the integrator's own knots (no upsampling: amplitudes at ~10^2-10^3
        knots, as the production mode sum) -> host sparse holder; mode selection by the call's
        threshold unless ``mode_selection`` is given."""
        from few.utils.utility import get_viewing_angles

        from .emritdionfly import host_holder

        m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = few_args[:14]
        th, ph = get_viewing_angles(qS, phiS, qK, phiK)
        kw = {k: v for k, v in few_kwargs.items() if k != "inspiral_kwargs"}
        if mode_selection is not None:
            kw.pop("mode_selection_threshold", None)
            kw["mode_selection"] = [tuple(int(v) for v in md) for md in mode_selection]
        span = self.wdm.Nt * self.wdm.layer_dt
        T = (self.data_t0 - self.t_start + span + 2000.0) / 3.15581497635456e7
        saved = dict(self.few_gen.inspiral_kwargs)
        try:
            H = self.few_gen(m1, m2, a, p0, e0, x0, th, ph, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
                             Phi_r0=Pr, T=T, dt=self.wdm.data_dt, return_sparse_holder=True,
                             include_minus_mkn=True, **kw)
        finally:
            self.few_gen.inspiral_kwargs.clear()
            self.few_gen.inspiral_kwargs.update(saved)
        return host_holder(H)

    def _pixel_times(self, H):
        """Pixel indices (and times, FEW clock) inside the holder's trajectory span."""
        t_arr = np.asarray(H.t_arr)
        lo = self.data_t0 - self.t_start
        n_all = np.arange(self.pixel_edge, self.wdm.Nt - self.pixel_edge)
        t_rel = lo + n_all * self.wdm.layer_dt
        inside = (t_rel >= t_arr[0]) & (t_rel <= t_arr[-1])
        return n_all[inside], t_rel[inside]

    def _mode_list(self, few_args, few_kwargs):
        """Modes kept at the call's threshold and the earliest in-window handoff time.

        One FEW call on the coarse fine grid; the holder and its harmonic tracks are kept
        (``_last_holder``, ``_last_tracks``) so the template reuses them."""
        if self.feed == "knots":
            H = self._few_knots(few_args, few_kwargs, mode_selection=few_kwargs.get("mode_selection"))
        else:
            H = self._few_holder(few_args, few_kwargs, self._fine_grid(None))
        modes = [(int(l), int(m), int(k), int(n)) for l, m, k, n in zip(H.ls, H.ms, H.ks, H.ns)]
        n_in, t_rel = self._pixel_times(H)
        integ = self.few_gen.inspiral_generator.inspiral_generator
        tracks = harmonic_tracks_from_holder(H, integ, t_rel, a=few_args[2], xI0=few_args[5]) if t_rel.size else []
        self._last_holder, self._last_tracks, self._last_track_n = H, tracks, n_in
        return modes, self._chunk_start(H, few_args, tracks=tracks, t_rel=t_rel)

    def _chunk_start(self, H, few_args, tracks=None, t_rel=None):
        """Earliest time (FEW clock) at which any harmonic of the holder ``H`` hands off to the
        plunge chunk inside the window, or ``None`` if none does. The chunk reads the dense
        response, so the fine grid must be dense from there on."""
        t_arr = np.asarray(H.t_arr)
        lo = self.data_t0 - self.t_start
        if tracks is None:
            _, t_rel = self._pixel_times(H)
            integ = self.few_gen.inspiral_generator.inspiral_generator
            tracks = harmonic_tracks_from_holder(H, integ, t_rel, a=few_args[2], xI0=few_args[5]) if t_rel.size else []
        first = np.inf
        for tr in tracks:
            k = handoff_pixel(tr, self.wdm.layer_dt, self.wdm.layer_df, self.fdot_axis_max)
            if k < t_rel.size:
                first = min(first, float(t_rel[k]))
        span = self.wdm.Nt * self.wdm.layer_dt
        if t_arr[-1] < lo + span - 1.0:                      # stops in the window
            first = min(first, float(t_arr[-1]) - self.Nt_sub * self.wdm.layer_dt)
        return None if not np.isfinite(first) else first

    def _fine_grid(self, chunk_start):
        """Fine trajectory times (FEW clock): ``fine_dt`` over the window, ``fine_dt_plunge``
        from a chunk length before ``chunk_start`` to the end (non-uniform is fine: the
        response splines auto-detect general spacing)."""
        lo = self.data_t0 - self.t_start
        span = self.wdm.Nt * self.wdm.layer_dt
        pad = 600.0 + 2.0 * self.fine_dt                     # delay margin + spline support
        a = max(0.0, lo - pad)
        b = lo + span + 2000.0                                # = the integration span (T_traj)
        coarse = np.append(np.arange(a, b, self.fine_dt), b)
        if chunk_start is None:
            return coarse
        d0 = max(a, chunk_start - self.Nt_sub * self.wdm.layer_dt)
        dense = np.append(np.arange(d0, b, self.fine_dt_plunge), b)
        return np.union1d(coarse[coarse < d0], dense)

    def _knots_feed(self, H, few_args, t_fine, fly):
        """Response feed on ``t_fine`` from the knots holder ``H``: amplitudes splined over the
        knots, phases from the integrator's dense output (call right after H's FEW call: the
        integrator holds H's trajectory). Returns ``fly.prepare_feed_arrays(...)``."""
        from .emritdionfly import EMRITDIonFly

        t_end = float(np.asarray(H.t_arr)[-1])
        t_src = t_fine[t_fine <= t_end]
        if t_src[-1] < t_end - 1e-6:
            t_src = np.append(t_src, t_end)
        integ = self.few_gen.inspiral_generator.inspiral_generator
        tr = harmonic_tracks_from_holder(H, integ, t_src, a=few_args[2], xI0=few_args[5], phase_only=True)
        amp, ph = feed_from_tracks(tr, EMRITDIonFly.AMP_FACTOR)
        return fly.prepare_feed_arrays(t_src, amp, ph)

    def _dense_response(self, items, t_fine):
        """ONE TDDenseTDIonTheFly call for ``items`` = [(dense inputs, (psi, lam, beta)), ...]."""
        from ...response.tdionfly import TDDenseTDIonTheFly
        from .emritdionfly import EMRITDIonFly

        K = max(it[0][0].size for it in items)
        n_temp = len(items)
        t_k = np.empty((n_temp, K))
        C = np.zeros((n_temp, K - 1, 3, 8))
        n_k = np.zeros(n_temp, dtype=np.int32)
        mkn, are, aim, offs, par = [], [], [], [0], []
        for b, ((tk, Cb, mk, ar, ai), (psi, lam, beta)) in enumerate(items):
            nk = tk.size
            tk = self.t_start + tk                                 # FEW clock -> absolute (the response's clock)
            t_k[b, :nk], t_k[b, nk:] = tk, tk[-1]
            C[b, :nk - 1] = Cb
            n_k[b] = nk
            pad = ((0, 0), (0, K - nk), (0, 0))
            mkn.append(mk)
            are.append(np.pad(ar, pad))
            aim.append(np.pad(ai, pad))
            offs.append(offs[-1] + mk.shape[0])
            par.append((0.0, psi, lam, beta))
        dense = TDDenseTDIonTheFly(
            np.tile(self.t_start + np.asarray(t_fine), (n_temp, 1)), np.array(offs), np.concatenate(mkn),
            t_k, n_k, C, np.concatenate(are), np.concatenate(aim), amp_factor=EMRITDIonFly.AMP_FACTOR,
            tdi_config=self.tdi_config, orbits=self.orbits, force_backend=self.force_backend)
        return dense(np.array(par))

    def _call_knots(self, few_args, few_kwargs, modes):
        from ...domains import WDMSignal
        from .emritdionfly import EMRITDIonFly

        wdm = self.wdm
        Nf, Nt, dt, ldt, ldf = wdm.Nf, wdm.Nt, wdm.data_dt, wdm.layer_dt, wdm.layer_df
        span = Nt * ldt
        n_all = np.arange(self.pixel_edge, Nt - self.pixel_edge)
        t_pix = self.data_t0 + n_all * ldt
        T_traj = self.data_t0 - self.t_start + span + 2000.0
        xp = self.xp
        acc = xp.zeros((self.tdi_config.nchannels, Nf, Nt))
        kw = dict(few_kwargs)
        if modes is not None:
            kw["mode_selection"] = modes
        modes, chunk_start = self._mode_list(few_args, kw)
        H, tracks_pix, n_tr = self._last_holder, self._last_tracks, self._last_track_n
        t_fine = self._fine_grid(chunk_start)
        self.n_fine, self.last_t_fine = int(t_fine.size), t_fine
        fly = EMRITDIonFly(self.few_gen, self.orbits, self.tdi_config, dt, T_traj, self.t_start,
                           frame="icrs_special", t_fine_window=(self.data_t0, self.data_t0 + span), t_fine=t_fine)
        _, _, psi, lam, beta = fly.sky(few_args[7], few_args[8], few_args[9], few_args[10])
        if self.response == "dense":
            integ = self.few_gen.inspiral_generator.inspiral_generator
            din = dense_inputs_from_holder(H, integ, a=few_args[2], xI0=few_args[5])
            t_end = float(np.asarray(H.t_arr)[-1])
            t_resp = t_fine
            if t_end < t_fine[-1] - 1.0:
                # stops in the window: end the response grid two delay margins after the stop
                # (as the spline feed's zero-amplitude continuation does); beyond it the channel
                # is exactly zero and its extracted phase meaningless, so it must not be splined
                t_resp = t_fine[t_fine <= t_end + 2 * 600.0 + 2 * self.fine_dt_plunge]
            out = self._dense_response([(din, (psi, lam, beta))], t_resp)
        else:
            feed = self._knots_feed(H, few_args, t_fine, fly)
            out = fly.run_response([tuple(feed) + (psi, lam, beta)])
        x = np.asarray(out.x.get() if hasattr(out.x, "get") else out.x)
        t_traj_end = float(np.asarray(H.t_arr)[-1])
        ok_t = (t_pix > x[:, 0].max()) & (t_pix < x[:, -1].min()) & (t_pix - self.t_start <= t_traj_end)
        n_ok, tt = n_all[ok_t], t_pix[ok_t]
        totals = dict(lookup_pixels=0, chunk_pixels=0, dropped_pixels=0)
        if n_ok.size:
            pos = np.searchsorted(n_tr, n_ok)
            assert np.array_equal(n_tr[pos], n_ok)
            tracks = [subset_track(t, pos) for t in tracks_pix]
            tracer = tracer_from_tof_output(out, xp.asarray(tt))
            assert tracer[0].shape[0] == len(tracks), (tracer[0].shape, len(tracks))

            def tail_td(ts, out=out, x_lo=float(x[:, 0].max()), x_hi=float(x[:, -1].min())):
                live = (ts > x_lo) & (ts < x_hi)
                td = xp.zeros((x.shape[0], tracer[0].shape[1], ts.size))
                if bool(xp.any(live)):
                    td[:, :, live] = xp.asarray(out.eval_tdi(ts[live]))
                return td

            totals = accumulate_harmonic_batch(
                acc, self.table, tracks, tracer, n_ok, tail_td, Nf=Nf, Nt=Nt, dt=dt, layer_dt=ldt,
                layer_df=ldf, t0=self.data_t0, Nt_sub=self.Nt_sub, num_m_layers=self.num_m_layers,
                fdot_axis_max=self.fdot_axis_max, pixel_edge=self.pixel_edge, backend=self.force_backend)
        self.last_stats = dict(modes=len(modes), n_fine=self.n_fine, feed="knots", chunk_start=chunk_start, **totals)
        return WDMSignal(acc, wdm)

    def __call__(self, *few_args, **few_kwargs):
        from ...domains import WDMSignal
        from .emritdionfly import EMRITDIonFly

        if self.feed == "knots" and self.n_fine_fixed is None:
            kw = dict(few_kwargs)
            return self._call_knots(few_args, kw, kw.pop("mode_selection", None))

        wdm = self.wdm
        Nf, Nt, dt, ldt, ldf = wdm.Nf, wdm.Nt, wdm.data_dt, wdm.layer_dt, wdm.layer_df
        span = Nt * ldt
        n_all = np.arange(self.pixel_edge, Nt - self.pixel_edge)
        t_pix = self.data_t0 + n_all * ldt
        T_traj = self.data_t0 - self.t_start + span + 2000.0
        xp = self.xp
        acc = xp.zeros((self.tdi_config.nchannels, Nf, Nt))
        few_kwargs = dict(few_kwargs)
        modes = few_kwargs.pop("mode_selection", None)      # explicit (l, m, k, n) list, e.g. gates
        totals = dict(lookup_pixels=0, chunk_pixels=0, dropped_pixels=0)
        H = tracks_all = n_tr = None
        t_fine, chunk_start = None, "unknown"
        if modes is None and self.n_fine_fixed is None:
            # ONE FEW call on the coarse grid gives the modes, the holder the response is fed
            # from and the harmonic tracks; a second (explicit-mode) call only when a harmonic
            # hands off to the plunge chunk inside the window and the grid must be densified
            modes, chunk_start = self._mode_list(few_args, few_kwargs)
            H, tracks_all, n_tr = self._last_holder, self._last_tracks, self._last_track_n
            t_fine = self._fine_grid(chunk_start)
            if chunk_start is not None:
                H = self._few_holder(few_args, few_kwargs, t_fine, mode_selection=modes)
                n_tr, t_rel = self._pixel_times(H)
                integ = self.few_gen.inspiral_generator.inspiral_generator
                tracks_all = harmonic_tracks_from_holder(H, integ, t_rel, a=few_args[2], xI0=few_args[5])
            self.n_fine = int(t_fine.size)
        else:                                               # explicit modes / fixed n_fine: FEW per batch
            if modes is None:
                modes, chunk_start = self._mode_list(few_args, few_kwargs)
            self.n_fine = self.n_fine_fixed if self.n_fine_fixed is not None else max(64, int(span / self.fine_dt_plunge))
        self.last_t_fine = t_fine                             # None: uniform n_fine over the window
        few_kwargs.pop("mode_selection_threshold", None)

        nm = len(modes)
        batch = nm if (self.mode_batch is None or self.mode_batch <= 0) else int(self.mode_batch)
        for j in range(0, nm, batch):
            idx = np.arange(j, min(j + batch, nm))
            fly = EMRITDIonFly(self.few_gen, self.orbits, self.tdi_config, dt, T_traj, self.t_start,
                               frame="icrs_special", n_fine=None if t_fine is not None else self.n_fine,
                               t_fine_window=(self.data_t0, self.data_t0 + span), t_fine=t_fine)
            if H is not None:
                Hb = H if idx.size == nm else slice_holder(H, idx)
                out = fly(*few_args, holder=Hb, **few_kwargs)
            else:
                out = fly(*few_args, mode_selection=[modes[i] for i in idx], **few_kwargs)
                Hb = fly.last_holder
            x = np.asarray(out.x.get() if hasattr(out.x, "get") else out.x)
            t_traj_end = float(np.asarray(Hb.t_arr)[-1])
            ok_t = (t_pix > x[:, 0].max()) & (t_pix < x[:, -1].min()) & (t_pix - self.t_start <= t_traj_end)
            n_ok, tt = n_all[ok_t], t_pix[ok_t]
            if n_ok.size == 0:
                continue
            if tracks_all is not None:
                pos = np.searchsorted(n_tr, n_ok)
                assert np.array_equal(n_tr[pos], n_ok)
                tracks = [subset_track(tracks_all[i], pos) for i in track_rows(H, idx)]
            else:
                integ = self.few_gen.inspiral_generator.inspiral_generator
                tracks = harmonic_tracks_from_holder(Hb, integ, tt - self.t_start, a=few_args[2], xI0=few_args[5])
            tracer = tracer_from_tof_output(out, xp.asarray(tt))
            assert tracer[0].shape[0] == len(tracks), (tracer[0].shape, len(tracks))
            x_lo, x_hi = x[:, 0][:, None, None], x[:, -1][:, None, None]

            def tail_td(ts, out=out, x_lo=float(x_lo.max()), x_hi=float(x_hi.min())):
                live = (ts > x_lo) & (ts < x_hi)
                td = xp.zeros((x.shape[0], tracer[0].shape[1], ts.size))
                if bool(xp.any(live)):
                    td[:, :, live] = xp.asarray(out.eval_tdi(ts[live]))
                return td

            st = accumulate_harmonic_batch(
                acc, self.table, tracks, tracer, n_ok, tail_td, Nf=Nf, Nt=Nt, dt=dt, layer_dt=ldt,
                layer_df=ldf, t0=self.data_t0, Nt_sub=self.Nt_sub, num_m_layers=self.num_m_layers,
                fdot_axis_max=self.fdot_axis_max, pixel_edge=self.pixel_edge, backend=self.force_backend)
            for key in totals:
                totals[key] += st[key]
            del out, fly
        self.last_stats = dict(modes=nm, n_fine=self.n_fine,
                               chunk_start=None if chunk_start == "unknown" else chunk_start, **totals)
        return WDMSignal(acc, wdm)

    def batch(self, rows, chunk_rows=16, consume=None, **few_kwargs):
        """Templates for many parameter rows, ``chunk_rows`` per response call (bounds GPU memory:
        one 6-month production-grid template is ~150 MB of WDM coefficients).

        ``consume(row_indices, arr)`` (optional) receives each chunk's ``(n, nch, Nf, Nt)`` array
        and nothing is kept; otherwise the full ``(n_rows, nch, Nf, Nt)`` array is returned.
        ``last_stats`` sums the chunks' stats."""
        rows = list(rows)
        parts, tot = [], {}
        for i in range(0, len(rows), max(1, int(chunk_rows))):
            idx = list(range(i, min(i + int(chunk_rows), len(rows))))
            arr = self._batch_chunk([rows[k] for k in idx], **few_kwargs)
            for k, v in self.last_stats.items():
                tot[k] = tot.get(k, 0) + v if isinstance(v, (int, float)) else v
            if consume is not None:
                consume(idx, arr)
                del arr
            else:
                parts.append(arr)
        self.last_stats = tot
        if consume is not None:
            return None
        return parts[0] if len(parts) == 1 else self.xp.concatenate(parts, axis=0)

    def _batch_chunk(self, rows, **few_kwargs):
        """Many templates with ONE TDI-on-the-fly response call, ONE tracer and ONE lookup.

        ``rows``: parameter rows (as for ``__call__``). FEW runs once per row (it is a
        single-template generator); every row whose harmonics stay on the lookup feeds one
        response kernel launch (num_sub x n_rows blocks fill the GPU), one tracer evaluation
        and one table call + scatter-add into ``(n_rows, nch, Nf, Nt)``. A row that hands off
        to the plunge chunk is built alone with ``__call__``. Returns the array (backend xp).
        """
        from .emritdionfly import EMRITDIonFly

        wdm = self.wdm
        Nf, Nt, dt, ldt, ldf = wdm.Nf, wdm.Nt, wdm.data_dt, wdm.layer_dt, wdm.layer_df
        span = Nt * ldt
        xp = self.xp
        n_all = np.arange(self.pixel_edge, Nt - self.pixel_edge)
        t_pix = self.data_t0 + n_all * ldt
        T_traj = self.data_t0 - self.t_start + span + 2000.0
        out_arr = xp.zeros((len(rows), self.tdi_config.nchannels, Nf, Nt))
        few_kwargs = dict(few_kwargs)
        t_fine = self._fine_grid(None)
        fly = EMRITDIonFly(self.few_gen, self.orbits, self.tdi_config, dt, T_traj, self.t_start,
                           frame="icrs_special", t_fine_window=(self.data_t0, self.data_t0 + span), t_fine=t_fine)
        feeds, tracks, rows_in, n_trs, t_ends = [], [], [], [], []
        stats = dict(rows=len(rows), alone=0, subs=0)
        for r, p in enumerate(rows):
            modes, chunk_start = self._mode_list(p, few_kwargs)
            if chunk_start is not None:                      # plunge chunk: build this one alone
                out_arr[r] = xp.asarray(self(*p, **few_kwargs).arr)
                stats["alone"] += 1
                continue
            H = self._last_holder
            _, _, psi, lam, beta = fly.sky(p[7], p[8], p[9], p[10])
            if self.response == "dense":
                integ = self.few_gen.inspiral_generator.inspiral_generator
                feeds.append((dense_inputs_from_holder(H, integ, a=p[2], xI0=p[5]), (psi, lam, beta)))
            else:
                if self.feed == "knots":
                    t_in, amp, ph, t_tdi = self._knots_feed(H, p, t_fine, fly)
                else:
                    t_in, amp, ph, t_tdi = fly.prepare_feed(H, True)
                feeds.append((t_in, amp, ph, t_tdi, psi, lam, beta))
            tracks.append(self._last_tracks)
            n_trs.append(self._last_track_n)
            t_ends.append(float(np.asarray(H.t_arr)[-1]))
            rows_in.append(r)
        if not feeds:
            self.last_stats = stats
            return out_arr
        out = self._dense_response(feeds, t_fine) if self.response == "dense" else fly.run_response(feeds)
        x = np.asarray(out.x.get() if hasattr(out.x, "get") else out.x)
        ok_t = (t_pix > x[:, 0].max()) & (t_pix < x[:, -1].min()) & (t_pix - self.t_start <= min(t_ends))
        n_ok, tt = n_all[ok_t], t_pix[ok_t]
        sub_row, trk = [], []
        for k, r in enumerate(rows_in):
            pos = np.searchsorted(n_trs[k], n_ok)
            assert np.array_equal(n_trs[k][pos], n_ok)
            trk += [subset_track(tr, pos) for tr in tracks[k]]
            sub_row += [r] * len(tracks[k])
        tracer = tracer_from_tof_output(out, xp.asarray(tt))
        assert tracer[0].shape[0] == len(trk), (tracer[0].shape, len(trk))
        st = accumulate_harmonic_batch(
            out_arr, self.table, trk, tracer, n_ok, None, Nf=Nf, Nt=Nt, dt=dt, layer_dt=ldt,
            layer_df=ldf, t0=self.data_t0, Nt_sub=self.Nt_sub, num_m_layers=self.num_m_layers,
            fdot_axis_max=self.fdot_axis_max, pixel_edge=self.pixel_edge, backend=self.force_backend,
            sub_row=np.asarray(sub_row))
        stats.update(subs=len(trk), **st)
        self.last_stats = stats
        return out_arr
