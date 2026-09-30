"""EMRI templates built directly in the WDM domain (lookup table + plunge chunk).

Harmonic tracks follow FEW's own conventions (few/waveform/base.py:357-364 and
few/summation/directmodesum.py):

* for ``a > 0`` the azimuthal phase is multiplied by ``sign(xI0)``;
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


def harmonic_tracks_from_holder(holder, integrator, t_pixels, *, a, xI0):
    """Per-harmonic tracks at ``t_pixels`` [s, trajectory clock] from a FEW sparse holder.

    Args:
        holder: FEW ``SparseInfoHolder`` (``return_sparse_holder=True``).
        integrator: the trajectory integrator (``...inspiral_generator.inspiral_generator``)
            exposing ``eval_integrator_spline`` / ``eval_integrator_derivative_spline``.
        t_pixels: evaluation times, same clock as ``holder.t_arr``.
        a, xI0: spin and inclination cosine (retrograde convention).
    """
    t_pixels = np.asarray(t_pixels, dtype=float)
    P = [_phase_columns(integrator, t_pixels, k) for k in (0, 1, 2, 3)]
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

    tracks = []
    minus = []
    for j, (l, m, k, n) in enumerate(zip(holder.ls, holder.ms, holder.ks, holder.ns)):
        w = np.array([m, k, n], dtype=float)
        ph = P[0] @ w
        f = (P[1] @ w) / (2 * np.pi)
        fd = (P[2] @ w) / (2 * np.pi)
        fdd = (P[3] @ w) / (2 * np.pi)
        lmkn = (int(l), int(m), int(k), int(n))
        tracks.append(HarmonicTrack(lmkn, t_pixels, A[:, j] * ylms[j], ph, f, fd, fdd))
        if with_minus and m != 0:
            amp_m = ((-1.0) ** int(l)) * ylms[nm + j] * np.conj(A[:, j])
            minus.append(HarmonicTrack((int(l), -int(m), -int(k), -int(n)), t_pixels, amp_m,
                                       -ph, -f, -fd, -fdd))
    return tracks + minus


# ----------------------------------------------------------------------
# Handoff from the lookup to the plunge chunk
# ----------------------------------------------------------------------

#: Half-support of the WDM wavelet in units of layer_dt used by the curvature trigger.
#: Calibrated against the plunging gate source in Task A6 (see the ledger).
WDM_HALF_SUPPORT_LAYERS = 1.0
#: Intra-chunk sweep [layers] beyond which the chunk sheds chirped power SILENTLY
#: (SOBBH measurement: optimum ~3.5, tolerable to 7, collapse by 14;
#: globalfit/stock/erebor/source_runtime.py resolve_sobbh_nt_sub).
CHUNK_KAPPA_MAX = 7.0


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


def chunk_kappa(fdot_max, fddot_max, Nt_sub, layer_dt, layer_df):
    """Carrier sweep across one chunk, in layers: (|fdot| T + |fddot| T^2 / 2) / layer_df."""
    T = Nt_sub * layer_dt
    return (abs(fdot_max) * T + 0.5 * abs(fddot_max) * T ** 2) / layer_df


def assert_chunk_kappa(fdot_max, fddot_max, Nt_sub, layer_dt, layer_df):
    """Raise instead of letting a chunk shed chirped power silently."""
    kappa = chunk_kappa(fdot_max, fddot_max, Nt_sub, layer_dt, layer_df)
    if kappa > CHUNK_KAPPA_MAX:
        raise ValueError(
            f"plunge chunk sweep kappa={kappa:.1f} layers > {CHUNK_KAPPA_MAX}: the chunk would "
            "shed chirped power silently; shrink Nt_sub or hand off earlier")
    return kappa


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
