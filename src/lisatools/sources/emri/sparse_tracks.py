"""Sparse EMRI harmonic tracks: FEW sparse holder -> sparse TDI response -> per-harmonic tracks.

The domain-independent front half of the sparse EMRI template. Nothing here knows about the
output domain (WDM, STFT, ...): it ends at per-(harmonic, channel) ``(amp, phase, f, fdot)``
evaluated at arbitrary times (:meth:`SplinedTDIOutput.eval_tracer`) and at the exact channel time
series for the stretches a slowly-chirping model cannot represent
(:meth:`EMRISparseResponse.td_channels`).

1. ONE FEW call per template on the integrator's own knots gives the kept modes and the sparse
   holder (``return_sparse_holder=True``, FEW ``gpu_backend``); the trajectory's DOPR853 dense
   output gives exact harmonic phases (:func:`dense_inputs_from_holder`).
2. :class:`~lisatools.response.tdionfly.TDDenseTDIonTheFly` evaluates the TDI response on a SPARSE
   time grid (:func:`sparse_response_grid`: the knots plus a point every ``sparse_dt``).
3. :class:`ExactPhaseTDIOutput` keeps the carrier exact and splines only the amplitude and the
   slow residual phase.

Provenance: the helper functions and the two output classes are lifted verbatim from
``dev:src/lisatools/sources/emri/wdm_direct.py`` (e25ff72f), where they live next to the
WDM-specific lookup assembly; :class:`EMRISparseResponse` re-packages the domain-independent
methods of ``EMRIDirectWDM`` (``_few_knots``, ``_sparse_grid``, ``_dense_kernel``,
``_dense_response``, ``_dense_td``). Docstrings of the lifted code still say "pixel" for an
evaluation time.

Harmonic tracks follow FEW's own conventions (few/waveform/base.py and
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

Environment knobs read here: ``EMRI_TRACER`` (``analytic``), ``EMRI_TRACER_H_FDOT`` (300 s),
``EMRI_TRACER_MAX_ELEMS`` (6e7), ``EMRI_DIRECT_TAIL_BLOCK_ELEMS`` (2e7).
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
from scipy.interpolate import CubicSpline

from ...utils.constants import YRSID_SI


# ======================================================================
# Lifted verbatim from dev:src/lisatools/sources/emri/wdm_direct.py
# ======================================================================

@dataclasses.dataclass
class HarmonicTrack:
    """One harmonic at pixel-centre times: h = amp * exp(-1j * phase).

    Attributes:
        lmkn (tuple): ``(l, m, k, n)`` mode indices (negated ``m, k, n`` for a -m partner).
        t (np.ndarray): Times [s], FEW clock, ``(P,)``.
        amp (np.ndarray): Complex amplitude ``Y_lm A`` (or the -m partner's), ``(P,)``.
        phase (np.ndarray): ``m Phi_phi + k Phi_theta + n Phi_r`` [rad], ``(P,)``.
        f (np.ndarray | None): Source-frame harmonic frequency [Hz] (``None`` with
            ``phase_only``).
        fdot (np.ndarray | None): Its first time derivative [Hz/s].
        fddot (np.ndarray | None): Its second time derivative [Hz/s^2].
    """

    lmkn: tuple
    t: np.ndarray
    amp: np.ndarray
    phase: np.ndarray
    f: np.ndarray
    fdot: np.ndarray
    fddot: np.ndarray


def _phase_columns(integrator, t, order):
    """The three fundamental phases ``(Phi_phi, Phi_theta, Phi_r)`` (``order = 0``) or their
    ``order``-th time derivative at times ``t`` from the integrator splines: ``(len(t), 3)``."""
    if order == 0:
        cols = np.asarray(integrator.eval_integrator_spline(t))[:, 3:6]
    else:
        cols = np.asarray(integrator.eval_integrator_derivative_spline(t, order=order))[:, 3:6]
    return np.array(cols, dtype=float, copy=True)


def harmonic_tracks_from_holder(holder, integrator, t_pixels, *, a, xI0, phase_only=False):
    """Per-harmonic tracks at ``t_pixels`` [s, trajectory clock] from a FEW sparse holder.

    Vectorised over modes (one matrix product per derivative order). The amplitudes are cubic
    splines of the holder's Teukolsky amplitudes at its knots times ``Y_lm``; phases and their
    derivatives come from the integrator (the conventions in the module docstring).

    Args:
        holder: FEW ``SparseInfoHolder`` (``return_sparse_holder=True``), host copy.
        integrator: the trajectory integrator (``...inspiral_generator.inspiral_generator``)
            exposing ``eval_integrator_spline`` / ``eval_integrator_derivative_spline``; it
            must still hold ``holder``'s trajectory.
        t_pixels (np.ndarray): evaluation times, same clock as ``holder.t_arr``.
        a (float): spin (retrograde convention: ``xI0 < 0`` maps to ``(-a, +1)``).
        xI0 (float): inclination cosine.
        phase_only (bool, optional): skip the frequency derivatives (``f``, ``fdot``,
            ``fddot`` are None): the response feed needs amplitude and phase only.

    Returns:
        list[HarmonicTrack]: The ``m >= 0`` modes in holder order, then the -m partners of the
        ``m != 0`` modes (when the holder carries ``2 * nmodes`` Ylms).
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
# Per-channel tracer from the TDI-on-the-fly output, and the full assembly
# ----------------------------------------------------------------------

def _spline_vals(out, t, subs):
    """``out.eval_spline_vals(t)`` (``(amp, tdi_phase, phase_ref)``), for the subset ``subs``
    when given."""
    if subs is None:
        return out.eval_spline_vals(t)
    return out.eval_spline_vals(t, subs=subs)


def _total_phase(out, t, subs=None):
    """Channel phase ``(S, nch, len(t))`` of a response output at times ``t``."""
    if hasattr(out, "eval_phase"):
        return out.eval_phase(t) if subs is None else out.eval_phase(t, subs=subs)
    _, tph, pref = _spline_vals(out, t, subs)
    xp = getattr(out, "xp", np)
    return xp.asarray(tph) + xp.asarray(pref)[:, None, :]


def tracer_from_tof_output(out, t_pixels, h=30.0, h_fdot=None, max_elems=None, subs=None, method=None):
    """Per-sub, per-channel ``(amp, phase, f, fdot)`` at ``t_pixels`` [absolute s].

    The channel signal is ``Re[amp exp(-i phase)]`` with ``phase = tdi_phase + phase_ref``
    (``TDTDIOutput.eval_tdi``; an :class:`ExactPhaseTDIOutput` adds its exact carrier instead).
    ``f`` and ``fdot`` are the first two derivatives of that CONTINUOUS phase over 2 pi, so they
    carry the Doppler shift of the channel (the source-frame f is off by ~1e-4 f, a ~1% pixel
    error). ``method``: ``"analytic"`` (default for an output with ``eval_tracer``, i.e.
    :class:`SplinedTDIOutput` / :class:`ExactPhaseTDIOutput`; env ``EMRI_TRACER``) differentiates
    the splines and the dense-output carrier exactly, one evaluation per pixel; ``"stencil"``
    takes central differences (steps ``h``, ``h_fdot``) of the phase (any output). A
    negative-frequency sub (a -m partner) is mirrored to positive frequency: cos is even, so
    ``phase -> -phase``, ``f -> -f``, ``fdot -> -fdot``.

    ``subs`` (host integer array, optional): only these harmonics (rows of the output), for a
    mode-batched caller (needs an output with subset evaluation, :class:`SplinedTDIOutput`).

    Memory: evaluated in blocks of pixels so no temporary exceeds ``max_elems`` (env
    ``EMRI_TRACER_MAX_ELEMS``, default 6e7) elements per ``(S, nch, block)`` array, and the
    amplitude is read at the pixel centres only.

    Args:
        out: Response output (:class:`SplinedTDIOutput`, :class:`ExactPhaseTDIOutput` or a
            ``TDTDIOutput`` with splines).
        t_pixels: Pixel-centre times [s, absolute], ``(P,)``.
        h (float, optional): Stencil step [s] for ``f``. Default ``30.0``.
        h_fdot (float, optional): Stencil step [s] for ``fdot``; ``None`` reads
            ``EMRI_TRACER_H_FDOT`` (300).
        max_elems (float, optional): Block budget; ``None`` reads ``EMRI_TRACER_MAX_ELEMS``.
        subs (np.ndarray, optional): Rows (harmonics) to evaluate; ``None`` = all.
        method (str, optional): ``"analytic"`` or ``"stencil"``; ``None`` reads
            ``EMRI_TRACER`` (``"analytic"``).

    Returns:
        tuple: ``(amp, phase, f, fdot)``, each ``(num_sub, nch, P)`` on ``out.xp``.

    Raises:
        ValueError: For an unknown ``method``.
    """
    xp = getattr(out, "xp", np)
    t = xp.asarray(t_pixels, dtype=float)
    hd = float(os.environ.get("EMRI_TRACER_H_FDOT", "300")) if h_fdot is None else float(h_fdot)
    P = int(t.size)
    try:
        S = int(np.size(subs)) if subs is not None else int(np.shape(out.x)[0])
        nch = int(getattr(out, "nch", 0) or np.shape(out.tdi_amp)[1])
    except AttributeError:                                  # a minimal output: one pass
        S = nch = 1
    budget = float(os.environ.get("EMRI_TRACER_MAX_ELEMS", 6e7)) if max_elems is None else float(max_elems)
    block = max(1, int(budget // max(1, S * nch)))
    if method is None:
        method = os.environ.get("EMRI_TRACER", "analytic")
    if method not in ("analytic", "stencil"):
        raise ValueError("tracer method must be 'analytic' or 'stencil'")
    analytic = method == "analytic" and hasattr(out, "eval_tracer")
    parts = []
    for p0 in range(0, P, block):
        tb = t[p0:p0 + block]
        if analytic:
            parts.append(out.eval_tracer(tb, subs=subs))
            continue
        amp, tph, pref = _spline_vals(out, tb, subs)
        amp = xp.asarray(amp)
        ph0 = xp.asarray(tph) + xp.asarray(pref)[:, None, :]
        del tph, pref
        f = (_total_phase(out, tb + h, subs) - _total_phase(out, tb - h, subs)) / (2 * h) / (2 * np.pi)
        fdot = (_total_phase(out, tb + hd, subs) - 2 * ph0 + _total_phase(out, tb - hd, subs)) / hd ** 2 / (2 * np.pi)
        #                                     phase ~1e5 rad: a short step turns roundoff into fdot noise
        neg = f < 0
        parts.append((amp, xp.where(neg, -ph0, ph0), xp.where(neg, -f, f), xp.where(neg, -fdot, fdot)))
        del ph0, f, fdot, neg
    if len(parts) == 1:
        return parts[0]
    return tuple(xp.concatenate([p[k] for p in parts], axis=-1) for k in range(4))


def dense_inputs_from_holder(holder, integrator, *, a, xI0):
    """Inputs of :class:`lisatools.response.tdionfly.TDDenseTDIonTheFly` for one template.

    Returns ``(t_k (K,), phase_coeffs (K-1, 3, 8), mkn (S, 3), amp_re (S, K-1, 4),
    amp_im (S, K-1, 4))`` with the harmonics in :func:`harmonic_tracks_from_holder`'s order
    (m >= 0 modes, then the -m partners). The DOPR853 dense-output coefficients carry every
    convention of ``harmonic_tracks_from_holder`` (massratio scaling, the integrator's backwards
    adjustment, sign(xI0) on Phi_phi for a > 0, the holder's backwards offset), so the dense
    polynomial IS the track phase; the amplitude splines are the tracks' (scipy CubicSpline over
    the knots, monomial coefficients in ``t - t_knot``).

    Args:
        holder: Host FEW sparse holder of the template (knots feed).
        integrator: The trajectory integrator, still holding ``holder``'s trajectory.
        a (float): Spin.
        xI0 (float): Inclination cosine (``< 0`` maps to ``(-a, +1)``).

    Returns:
        tuple: ``(t_k, phase_coeffs, mkn, amp_re, amp_im)``; ``t_k`` on the FEW clock.
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


def dense_eval_derivs(c, s):
    """The nested DOPR853 polynomial and its first two ``s``-derivatives.

    ``c`` (..., 8) coefficients, ``s`` broadcastable to ``c[..., 0]``. Horner with forward-mode
    derivatives through ``r1 + s(r2 + s1(r3 + s(r4 + s1(r5 + s(r6 + s1(r7 + s r8))))))``,
    ``s1 = 1 - s``: a step ``q = c + s p`` gives ``q' = p + s p'``, ``q'' = 2 p' + s p''``; a step
    ``q = c + s1 p`` gives ``q' = -p + s1 p'``, ``q'' = -2 p' + s1 p''``. Returns ``(P, dP/ds,
    d2P/ds2)``; divide by ``h`` and ``h**2`` for time derivatives. The C++ lookup-sum kernel runs
    the same recursion (cutils/wdm_lookup_kernels.hh: wdm_dense_phase_derivs)."""
    s1 = 1.0 - s
    v = c[..., 7]
    d1 = 0.0 * v
    d2 = 0.0 * v
    for j, use_s in ((6, True), (5, False), (4, True), (3, False), (2, True), (1, False), (0, True)):
        if use_s:
            d2, d1, v = 2.0 * d1 + s * d2, v + s * d1, c[..., j] + s * v
        else:
            d2, d1, v = -2.0 * d1 + s1 * d2, -v + s1 * d1, c[..., j] + s1 * v
    return v, d1, d2


def dense_phase_eval(t_k, C, t):
    """Evaluate DOPR853 dense-output coefficients ``C`` (K-1, P, 8) on knots ``t_k`` at ``t``.

    Returns ``(len(t), P)``; times outside the knots use the end segments. Used by
    :func:`_dense_on_grid` and as the reference in tests.
    """
    t = np.asarray(t, dtype=float)
    seg = np.clip(np.searchsorted(t_k, t, side="right") - 1, 0, t_k.size - 2)
    s = ((t - t_k[seg]) / np.diff(t_k)[seg])[:, None]
    s1 = 1.0 - s
    c = C[seg]
    return c[..., 0] + s * (c[..., 1] + s1 * (c[..., 2] + s * (c[..., 3] + s1 * (c[..., 4] + s * (c[..., 5] + s1 * (c[..., 6] + s * c[..., 7]))))))


#: Fewest points a template's response grid may have (:meth:`EMRIDirectWDM._dense_response`
#: refuses fewer, as a domain error). The response splines are fitted on the run's backend;
#: on the GPU that is GBT's ``CubicSplineInterpolant`` -> cuSPARSE ``gtsv2StridedBatch``,
#: which rejects a system of fewer than 3 points, and GBT then calls ``exit(-1)``: no
#: traceback, the MPI job torn down (9mo jobs 795, 798 and 808, 2026-10-20/22, a hot-rung row
#: whose trajectory ended within the delay margins of its start, so
#: :func:`sparse_response_grid` returned its 2-point fallback).
MIN_RESPONSE_POINTS = 4


def sparse_response_grid(knots, a, b, sparse_dt, margin):
    """Sparse response times over ``[a, b]``, kept where the response is COMPLETE.

    The knots inside, plus a point every ``sparse_dt`` at most, restricted to ``[knots[0] +
    margin, knots[-1] - margin]``: at a time within ``margin`` (the TDI delay margin) of a
    trajectory end some delayed source times fall outside the trajectory and the channel
    amplitude drops to zero there. A cubic spline over half-day intervals across that drop
    rings back into the interior (2 % in amplitude, 0.3 rad in phase two intervals in), so the
    sparse grid must never contain such a point; an in-window plunge is resolved by the plunge
    chunk's exact dense time series instead.

    Args:
        knots (np.ndarray): Trajectory knot times [s, FEW clock].
        a (float): Window start [s, FEW clock].
        b (float): Window end [s, FEW clock].
        sparse_dt (float): Largest spacing [s].
        margin (float): TDI delay margin [s].

    Returns:
        np.ndarray: Sorted grid times; ``[lo, lo + 1]`` when the restricted window is empty
        (a trajectory shorter than about two margins: too few points for the response
        splines, refused by :meth:`EMRIDirectWDM._dense_response`).
    """
    knots = np.asarray(knots, dtype=float)
    lo = max(float(a), float(knots[0]) + margin)
    hi = min(float(b), float(knots[-1]) - margin)
    if hi <= lo:
        return np.array([lo, lo + 1.0])
    inner = knots[(knots > lo) & (knots < hi)]
    return np.union1d(np.append(np.arange(lo, hi, sparse_dt), hi), inner)


def pad_grid(g, N):
    """``g`` grown to ``N`` points by halving its longest intervals (stays inside ``g``'s span:
    a template's grid is padded to the batch's common length without leaving the region where
    its response is complete). Returns an ``np.ndarray`` (sorted when grown; the input as a
    float array when it already has ``>= N`` points)."""
    g = np.asarray(g, dtype=float)
    while g.size < N:
        d = np.diff(g)
        k = min(N - g.size, d.size)
        widest = np.argsort(d)[::-1][:k]
        g = np.sort(np.concatenate([g, g[widest] + 0.5 * d[widest]]))
    return g


class SplinedTDIOutput:
    """A dense TDI response's amplitude and phase splined over its output grid, evaluable for
    any SUBSET of harmonics (a mode-batched caller keeps every temporary at a block's size).

    The channel signal of harmonic ``s`` is ``amp(t) exp(-i (carrier_s(t) + r(t)))``; here the
    carrier is the kernel's ``phase_ref`` and ``r`` its ``tdi_phase``, both splined (the
    original construction: the grid must resolve the whole phase). Duck-types what
    :func:`tracer_from_tof_output` and the plunge tail read from a ``TDTDIOutput``: ``x``,
    ``xp``, ``nch``, ``eval_spline_vals(t, subs=None)`` (``(amp, r, carrier)``),
    ``eval_phase(t, subs=None)`` and ``eval_tdi(t)``. Used with ``response_grid="pixels"``.

    Args:
        out (TDTDIOutput): Raw dense-kernel output (``return_spline=False``): ``x``
            ``(num_sub, N)``, ``tdi_amp`` / ``tdi_phase`` ``(num_sub, nch, N)``, ``phase_ref``
            ``(num_sub, N)``, ``build_spline``.
        sub_mkn (np.ndarray): ``(num_sub, 3)`` harmonic ``(m, k, n)``.
        sub_temp (np.ndarray): ``(num_sub,)`` template index of each harmonic.

    Attributes:
        x: Response times ``(num_sub, N)`` [s, absolute].
        nch (int): Channels.
        num_sub (int): Harmonics (all templates).
    """

    def __init__(self, out, sub_mkn, sub_temp):
        self.xp = out.xp
        self.x = out.x
        self.nch = int(out.tdi_amp.shape[1])
        self.num_sub = int(out.tdi_amp.shape[0])
        self._temp = np.asarray(sub_temp, dtype=np.int64).reshape(-1)
        self._mkn_host = np.asarray(sub_mkn, dtype=float).reshape(-1, 3)
        self._amp_spl = out.build_spline(out.x, out.tdi_amp)
        self._res_spl = out.build_spline(out.x, self._residual(out))
        self._car_spl = self._carrier_spline(out)

    def _residual(self, out):
        """The splined channel phase: ``tdi_phase`` here (the whole phase minus the carrier)."""
        return out.tdi_phase

    def _carrier_spline(self, out):
        """Spline of the kernel's ``phase_ref`` (the carrier)."""
        return out.build_spline(out.x, out.phase_ref)

    def _subs(self, subs):
        """``subs`` as a flat int64 host array (all harmonics for ``None``)."""
        return np.arange(self.num_sub) if subs is None else np.asarray(subs, dtype=np.int64).reshape(-1)

    def _eval_chan(self, spl, t, subs, derivative=0):
        """A per-(harmonic, channel) spline at the shared times ``t``: ``(len(subs), nch, M)``."""
        xp = self.xp
        t = xp.asarray(t, dtype=float)
        n = subs.size
        ind = xp.asarray(subs[:, None] * self.nch + np.arange(self.nch)[None, :])
        return spl(xp.tile(t, (n, self.nch, 1)), ind_interps=ind, derivative=derivative)

    def carrier(self, t, subs=None):
        """Carrier phase ``(len(subs), len(t))``."""
        xp = self.xp
        subs = self._subs(subs)
        t = xp.asarray(t, dtype=float)
        return self._car_spl(xp.tile(t, (subs.size, 1)), ind_interps=xp.asarray(subs))

    def carrier_derivs(self, t, subs=None):
        """Carrier phase and its first two time derivatives, each ``(len(subs), len(t))``."""
        xp = self.xp
        subs = self._subs(subs)
        tt = xp.tile(xp.asarray(t, dtype=float), (subs.size, 1))
        ind = xp.asarray(subs)
        return tuple(self._car_spl(tt, ind_interps=ind, derivative=d) for d in (0, 1, 2))

    def eval_tracer(self, t, subs=None):
        """``(amp, phase, f, fdot)`` at times ``t`` with ANALYTIC derivatives, one evaluation
        per time: ``f = (carrier' + r') / 2 pi`` and ``fdot = (carrier'' + r'') / 2 pi`` (the
        residual spline's own derivatives, the carrier's exact ones). A negative-frequency sub
        (a -m partner) is mirrored as in :func:`tracer_from_tof_output`. Each ``(len(subs),
        nch, len(t))``."""
        xp = self.xp
        subs = self._subs(subs)
        car, car1, car2 = self.carrier_derivs(t, subs)
        amp = self._eval_chan(self._amp_spl, t, subs)
        ph = self._eval_chan(self._res_spl, t, subs) + car[:, None, :]
        f = (self._eval_chan(self._res_spl, t, subs, derivative=1) + car1[:, None, :]) / (2 * np.pi)
        fdot = (self._eval_chan(self._res_spl, t, subs, derivative=2) + car2[:, None, :]) / (2 * np.pi)
        neg = f < 0
        return amp, xp.where(neg, -ph, ph), xp.where(neg, -f, f), xp.where(neg, -fdot, fdot)

    def eval_spline_vals(self, t, subs=None):
        """``(amp, r, carrier)`` at the shared times ``t``: ``(S, nch, M)``, ``(S, nch, M)``,
        ``(S, M)`` for ``S = len(subs)``."""
        subs = self._subs(subs)
        return (self._eval_chan(self._amp_spl, t, subs), self._eval_chan(self._res_spl, t, subs),
                self.carrier(t, subs))

    def eval_phase(self, t, subs=None):
        """Channel phase ``r + carrier`` only (no amplitude): ``(len(subs), nch, len(t))``."""
        subs = self._subs(subs)
        return self._eval_chan(self._res_spl, t, subs) + self.carrier(t, subs)[:, None, :]

    def eval_tdi(self, t, subs=None):
        """Channel time series ``Re[amp exp(-i (r + carrier))]``, ``(len(subs), nch, len(t))``."""
        amp, res, car = self.eval_spline_vals(t, subs)
        return self.xp.real(amp * self.xp.exp(-1j * (res + car[:, None, :])))

    def kernel_inputs(self):
        """Arrays of the fused lookup-sum kernel (:func:`lookup_sum_kernel`): the amplitude and
        phase splines as GBT CubicSplineInterpolant flats (ninterps ``num_sub * nch``, row
        ``s * nch + ch``) and the carrier. Here the carrier spline is folded into the phase
        spline (a cubic spline is linear in its data on fixed knots: the spline of
        ``tdi_phase + phase_ref`` has the summed coefficients) and the kernel runs without one."""
        xp = self.xp
        S, nch = self.num_sub, self.nch
        res = []
        for name in ("y_flat", "c1_flat", "c2_flat", "c3_flat"):
            r = getattr(self._res_spl, name).reshape(S, nch, -1)
            c = getattr(self._car_spl, name).reshape(S, 1, -1)
            res.append(xp.ascontiguousarray(r + c).reshape(-1))
        return self._kernel_common(dict(carrier=0, res=res))

    def _kernel_common(self, d):
        """Add the amplitude spline flats and the sizes (``N``, ``nch``, ``num_sub``, ``x``) to
        the kernel-input dict ``d`` and return it."""
        xp = self.xp
        a = self._amp_spl
        d.update(N=int(a.length), nch=self.nch, num_sub=self.num_sub,
                 x=xp.ascontiguousarray(a.x_flat, dtype=xp.float64),
                 amp=[xp.ascontiguousarray(getattr(a, k), dtype=xp.float64) for k in ("y_flat", "c1_flat", "c2_flat", "c3_flat")])
        return d


class ExactPhaseTDIOutput(SplinedTDIOutput):
    """A dense TDI response computed on a SPARSE time grid, evaluated anywhere with the
    harmonic's EXACT carrier phase.

    The channel signal of harmonic ``s`` is ``amp(t) exp(-i (Phi_s(t) + r(t)))``. ``Phi_s =
    m Phi_phi + k Phi_theta + n Phi_r`` is the DOPR853 dense-output carrier, exact at any time
    from the same coefficients the kernel uses (held at the trajectory end outside it, as the
    kernel holds its reference phase). Everything else is slow -- the TDI amplitude and the
    residual ``r = tdi_phase + phase_ref - Phi_s`` (the Doppler delay to spacecraft 1, the
    polarisation and transfer-function phase, the mode amplitude's own phase) -- so only
    ``amp`` and ``r`` are splined over the response grid. The response therefore needs only
    the trajectory's own knots plus a spacing cap, not one sample per pixel. The carrier is
    evaluated on the output's array module (the GPU on a GPU run), per template. This is the
    production response output (``response="dense"``, ``response_grid="sparse"``).

    Args:
        out (TDTDIOutput): Raw dense-kernel output on the sparse grid (see
            :class:`SplinedTDIOutput`).
        t_knots (list[np.ndarray]): Per template, the absolute knot times [s] of its DOPR853
            dense output.
        coeffs (list[np.ndarray]): Per template, the dense-output coefficients
            ``(K - 1, 3, 8)``.
        sub_mkn (np.ndarray): ``(num_sub, 3)`` harmonic ``(m, k, n)``.
        sub_temp (np.ndarray): ``(num_sub,)`` template index of each harmonic.
    """

    def __init__(self, out, t_knots, coeffs, sub_mkn, sub_temp):
        xp = out.xp
        self._tk = [xp.asarray(np.asarray(tk, dtype=float)) for tk in t_knots]   # absolute, per template
        self._C = [xp.asarray(np.asarray(c, dtype=float)) for c in coeffs]
        self._mkn = xp.asarray(np.asarray(sub_mkn, dtype=float).reshape(-1, 3))
        super().__init__(out, sub_mkn, sub_temp)

    def _residual(self, out):
        """The slow residual ``tdi_phase + phase_ref - Phi_s`` on the response grid."""
        xp = self.xp
        carrier = xp.empty(out.phase_ref.shape)
        x = out.x
        for b in np.unique(self._temp):
            rows = np.flatnonzero(self._temp == b)
            carrier[xp.asarray(rows)] = self._mkn[xp.asarray(rows)] @ self._P3(int(b), x[int(rows[0])]).T
        return out.tdi_phase + (out.phase_ref - carrier)[:, None, :]

    def _carrier_spline(self, out):
        """No carrier spline: the carrier is exact (:meth:`carrier`)."""
        return None

    def _P3(self, b, t):
        """Template ``b``'s three fundamental phases at the 1-D times ``t``: ``(len(t), 3)``."""
        xp = self.xp
        tk, C = self._tk[b], self._C[b]
        t = xp.clip(xp.asarray(t, dtype=float), tk[0], tk[-1])
        seg = xp.clip(xp.searchsorted(tk, t, side="right") - 1, 0, tk.size - 2)
        s = ((t - tk[seg]) / (tk[seg + 1] - tk[seg]))[:, None]
        s1 = 1.0 - s
        c = C[seg]
        return c[..., 0] + s * (c[..., 1] + s1 * (c[..., 2] + s * (c[..., 3] + s1 * (c[..., 4] + s * (
            c[..., 5] + s1 * (c[..., 6] + s * c[..., 7]))))))

    def _P3_derivs(self, b, t):
        """``_P3`` and its first two time derivatives, each ``(len(t), 3)``; outside the
        trajectory the phase is held at the nearest end, so both derivatives are zero there."""
        xp = self.xp
        tk, C = self._tk[b], self._C[b]
        t_in = xp.asarray(t, dtype=float)
        t = xp.clip(t_in, tk[0], tk[-1])
        held = ((t_in < tk[0]) | (t_in > tk[-1]))[:, None]
        seg = xp.clip(xp.searchsorted(tk, t, side="right") - 1, 0, tk.size - 2)
        h = (tk[seg + 1] - tk[seg])[:, None]
        v, d1, d2 = dense_eval_derivs(C[seg], ((t - tk[seg]) / (tk[seg + 1] - tk[seg]))[:, None])
        return v, xp.where(held, 0.0, d1 / h), xp.where(held, 0.0, d2 / h ** 2)

    def carrier_derivs(self, t, subs=None):
        """Exact carrier ``Phi_s`` and its first two time derivatives at ``t``, each
        ``(len(subs), len(t))`` (zero derivatives outside the trajectory)."""
        xp = self.xp
        subs = self._subs(subs)
        t = xp.asarray(t, dtype=float)
        outs = [xp.empty((subs.size, t.size)) for _ in range(3)]
        temp = self._temp[subs]
        for b in np.unique(temp):
            pos = np.flatnonzero(temp == b)
            mk = self._mkn[xp.asarray(subs[pos])]
            for o, P in zip(outs, self._P3_derivs(int(b), t)):
                o[xp.asarray(pos)] = mk @ P.T
        return tuple(outs)

    def kernel_inputs(self):
        """Kernel arrays (see :meth:`SplinedTDIOutput.kernel_inputs`) with the EXACT carrier: the
        templates' knots padded to a common stride ``K`` (``(n_temp, K)``, the last knot
        repeated), ``n_knots``, the DOPR853 coefficients ``(n_temp, K - 1, 3, 8)``, ``sub_temp``
        and ``sub_mkn``; the phase spline is the residual."""
        xp = self.xp
        n_temp = len(self._tk)
        K = max(int(tk.size) for tk in self._tk)
        t_knots = xp.empty((n_temp, K))
        C = xp.zeros((n_temp, K - 1, 3, 8))
        for b, (tk, Cb) in enumerate(zip(self._tk, self._C)):
            t_knots[b, :tk.size] = tk
            t_knots[b, tk.size:] = tk[-1]
            C[b, :tk.size - 1] = Cb
        r = self._res_spl
        return self._kernel_common(dict(
            carrier=1, n_temp=n_temp, K=K, t_knots=t_knots.reshape(-1), C=C.reshape(-1),
            n_knots=xp.asarray(np.array([int(tk.size) for tk in self._tk], dtype=np.int32)),
            sub_temp=xp.asarray(self._temp.astype(np.int32)),
            sub_mkn=xp.asarray(np.rint(self._mkn_host).astype(np.int32).reshape(-1)),
            res=[xp.ascontiguousarray(getattr(r, k), dtype=xp.float64) for k in ("y_flat", "c1_flat", "c2_flat", "c3_flat")]))

    def carrier(self, t, subs=None):
        """``Phi_s(t)`` for harmonics ``subs`` at the shared times ``t``: ``(len(subs), len(t))``."""
        xp = self.xp
        subs = self._subs(subs)
        t = xp.asarray(t, dtype=float)
        out = xp.empty((subs.size, t.size))
        temp = self._temp[subs]
        for b in np.unique(temp):
            pos = np.flatnonzero(temp == b)
            out[xp.asarray(pos)] = self._mkn[xp.asarray(subs[pos])] @ self._P3(int(b), t).T
        return out


def feed_from_tracks(tracks, amp_factor):
    """Per-sub (amp, phase) with h = sum amp exp(-i phase) from harmonic tracks on the feed grid.

    A track term is ``track.amp * exp(-i track.phase)`` (FEW's mode sum, -m partners included),
    so ``amp = |track.amp| * amp_factor`` and ``phase = track.phase - unwrap(arg(track.amp))``.
    Returns two ``(num_sub, n)`` ``np.ndarray`` (the spline-response knots feed)."""
    amp = np.stack([np.abs(tr.amp) for tr in tracks]) * amp_factor
    phase = np.stack([tr.phase - np.unwrap(np.angle(tr.amp)) for tr in tracks])
    return amp, phase


def slice_holder(H, idx):
    """The host holder restricted to modes ``idx`` (m >= 0 rows; the -m partners' Ylm follow).

    Returns a ``types.SimpleNamespace`` with the holder fields (``t_arr``, ``phases`` and
    ``freqs`` shared, not copied). Used for the mode batches of :meth:`EMRIDirectWDM.__call__`.
    """
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
    """A HarmonicTrack restricted to sample positions ``pos`` (needs ``f``, ``fdot``, ``fddot``)."""
    return HarmonicTrack(tr.lmkn, tr.t[pos], tr.amp[pos], tr.phase[pos], tr.f[pos], tr.fdot[pos], tr.fddot[pos])



# ======================================================================
# Domain-independent driver (re-packaged from dev's EMRIDirectWDM)
# ======================================================================

class EMRISparseResponse:
    """FEW sparse holder -> sparse dense-kernel TDI response, for any output domain.

    Args:
        few_gen: FEW waveform generator accepting ``return_sparse_holder=True`` (the inner
            ``FastKerrEccentricEquatorialFlux`` of FEW ``gpu_backend``, NOT
            ``GenerateEMRIWaveform``, which divides the output by the distance). Its
            ``inspiral_kwargs`` are restored after every call.
        orbits (Orbits): response orbits; the sky handed to :meth:`template_inputs` must be in
            their frame.
        tdi_config (TDIConfig): TDI configuration (``nchannels`` output channels).
        t_start (float): FEW reference epoch [s, absolute]: FEW's clock is ``t - t_start``.
        t_window (tuple): ``(t_lo, t_hi)`` absolute times the templates must cover.
        data_dt (float): data cadence [s] (FEW's ``dt`` argument; no dense series is built).
        sparse_dt (float, optional): largest spacing of the response grid [s]. Default 43200.
        traj_pad (float, optional): trajectory span past ``t_hi`` [s]. Default 2000.
        force_backend (str, optional): ``lisatools`` backend of the response. Default ``"cpu"``.

    A template "item" is ``(dense inputs, (psi, lam, beta))`` as built by
    :meth:`template_inputs`; items are what :meth:`tracks` and :meth:`td_channels` consume.
    """

    #: TDI delay margin [s]: the response grid stays this far inside the trajectory and the
    #: orbit tables (the kernel zeroes a channel whose delayed times leave either).
    DELAY_MARGIN = 600.0

    def __init__(self, few_gen, *, orbits, tdi_config, t_start, t_window, data_dt,
                 sparse_dt=43200.0, traj_pad=2000.0, force_backend="cpu"):
        from ... import get_backend

        self.few_gen = few_gen
        self.orbits, self.tdi_config = orbits, tdi_config
        self.t_start = float(t_start)
        self.t_lo, self.t_hi = float(t_window[0]), float(t_window[1])
        if not self.t_hi > self.t_lo:
            raise ValueError("t_window must be (t_lo, t_hi) with t_hi > t_lo")
        self.data_dt = float(data_dt)
        self.sparse_dt = float(sparse_dt)
        self.traj_pad = float(traj_pad)
        self.force_backend = force_backend
        self.xp = get_backend(force_backend).xp
        self._orbit_span_cache = None

    # ------------------------------------------------------------------
    # FEW
    # ------------------------------------------------------------------
    def few_holder(self, few_args, mode_selection=None, **few_kwargs):
        """ONE FEW call on the integrator's own knots -> host sparse holder.

        Args:
            few_args: ``(m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0,
                Phi_theta0, Phi_r0)`` with FEW's (ecliptic) sky / spin angles.
            mode_selection (optional): explicit ``(l, m, k, n)`` list; otherwise the call's
                ``mode_selection_threshold``.
            **few_kwargs: forwarded to FEW (``inspiral_kwargs`` is ignored).

        Raises:
            WaveformDomainError: parameters outside FEW's domain of validity.
        """
        from few.utils.utility import get_viewing_angles

        from .domain import few_domain_guard
        from .emritdionfly import host_holder

        m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = few_args[:14]
        th, ph = get_viewing_angles(qS, phiS, qK, phiK)
        kw = {k: v for k, v in few_kwargs.items() if k != "inspiral_kwargs"}
        if mode_selection is not None:
            kw.pop("mode_selection_threshold", None)
            kw["mode_selection"] = [tuple(int(v) for v in md) for md in mode_selection]
        T = (self.t_hi - self.t_start + self.traj_pad) / YRSID_SI
        saved = dict(self.few_gen.inspiral_kwargs)
        try:
            with few_domain_guard():
                H = self.few_gen(m1, m2, a, p0, e0, x0, th, ph, dist=dist, Phi_phi0=Pp,
                                 Phi_theta0=Pt, Phi_r0=Pr, T=T, dt=self.data_dt,
                                 return_sparse_holder=True, include_minus_mkn=True, **kw)
        finally:
            self.few_gen.inspiral_kwargs.clear()
            self.few_gen.inspiral_kwargs.update(saved)
        return host_holder(H)

    def template_inputs(self, few_args, lam, beta, mode_selection=None, **few_kwargs):
        """FEW call + dense-kernel inputs of one template.

        The dense inputs are read from the integrator right after the FEW call (the next FEW
        call overwrites its trajectory), so the two steps are one method.

        Args:
            few_args: as :meth:`few_holder`.
            lam (float): sky longitude in the ORBITS' frame [rad] (RA for ``frame="icrs"``).
            beta (float): sky latitude in the orbits' frame [rad] (Dec for ``frame="icrs"``).

        Returns:
            tuple: ``(item, H)``: the template item ``(dense inputs, (psi, lam, beta))`` and the
            host holder. ``psi`` is FEW's polarization angle of the (ecliptic) ``few_args``
            angles: with ICRS orbits and ``(lam, beta)`` the catalogue's own (RA, Dec) this is
            the mixed-frame recipe validated against mojito (the same as the dense
            ``EMRITDIWaveform`` path, without an ecliptic -> ICRS round trip of the sky).
        """
        from few.utils.utility import get_polarization_angle

        H = self.few_holder(few_args, mode_selection=mode_selection, **few_kwargs)
        integ = self.few_gen.inspiral_generator.inspiral_generator
        dense_in = dense_inputs_from_holder(H, integ, a=few_args[2], xI0=few_args[5])
        psi = get_polarization_angle(few_args[7], few_args[8], few_args[9], few_args[10])
        return (dense_in, (float(psi), float(lam), float(beta))), H

    def ends_before_window(self, H):
        """True when the inspiral ends before the window (minus the delay margin): no sample of
        the window sees it, the template is exactly zero and the response grid would be empty."""
        return float(np.asarray(H.t_arr)[-1]) < self.t_lo - self.t_start - self.DELAY_MARGIN

    # ------------------------------------------------------------------
    # Sparse response grid
    # ------------------------------------------------------------------
    def _orbit_span(self):
        """Absolute times ``(t_lo, t_hi)`` the orbit tables cover (spacecraft positions AND
        light travel times: the C++ response zeroes any time outside either)."""
        span = self._orbit_span_cache
        if span is None:
            args = getattr(self.orbits, "pycppdetector_args", None) if self.orbits is not None else None
            if args is None:
                span = (-np.inf, np.inf)
            else:
                sc_t0, sc_dt, sc_N, ltt_t0, ltt_dt, ltt_N = (float(v) for v in args[:6])
                span = (max(sc_t0, ltt_t0), min(sc_t0 + (sc_N - 1) * sc_dt, ltt_t0 + (ltt_N - 1) * ltt_dt))
            self._orbit_span_cache = span
        return span

    def sparse_grid(self, H):
        """Response times (FEW clock) of the holder ``H``: its integrator knots over the window
        (plus margins) and every ``sparse_dt`` at most in between, ending a delay margin inside
        the trajectory and inside the orbit tables."""
        lo = self.t_lo - self.t_start
        a = max(0.0, lo - (self.DELAY_MARGIN + 2.0 * self.sparse_dt))
        b = self.t_hi - self.t_start + self.traj_pad            # = the integration span
        o_lo, o_hi = self._orbit_span()
        a = max(a, o_lo - self.t_start + self.DELAY_MARGIN)
        b = min(b, o_hi - self.t_start - self.DELAY_MARGIN)
        return sparse_response_grid(H.t_arr, a, b, self.sparse_dt, self.DELAY_MARGIN)

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------
    def _dense_kernel(self, items, t_abs):
        """The TDDenseTDIonTheFly of ``items`` at the absolute times ``t_abs`` ``(n_temp, N)``;
        returns ``(dense, params, sub_mkn, t_knots_abs, coeffs)``."""
        from ...response.tdionfly import TDDenseTDIonTheFly
        from .emritdionfly import EMRITDIonFly

        K = max(it[0][0].size for it in items)
        n_temp = len(items)
        t_k = np.empty((n_temp, K))
        C = np.zeros((n_temp, K - 1, 3, 8))
        n_k = np.zeros(n_temp, dtype=np.int32)
        mkn, are, aim, offs, par, tk_abs, Cs = [], [], [], [0], [], [], []
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
            tk_abs.append(tk)
            Cs.append(Cb)
        dense = TDDenseTDIonTheFly(
            t_abs, np.array(offs), np.concatenate(mkn),
            t_k, n_k, C, np.concatenate(are), np.concatenate(aim), amp_factor=EMRITDIonFly.AMP_FACTOR,
            tdi_config=self.tdi_config, orbits=self.orbits, force_backend=self.force_backend)
        return dense, np.array(par), np.concatenate(mkn), tk_abs, Cs

    def tracks(self, items, t_grid):
        """ONE dense-kernel call for ``items`` on the sparse grid(s) -> :class:`ExactPhaseTDIOutput`.

        Args:
            items (list): template items (:meth:`template_inputs`).
            t_grid: response times (FEW clock): one array shared by every template, or one
                array per template (:meth:`sparse_grid`; padded to a common length).

        Returns:
            ExactPhaseTDIOutput: ``eval_tracer(t_abs, subs)`` gives ``(amp, phase, f, fdot)``
            per (harmonic, channel) at absolute times; harmonics of template ``b`` are the
            rows ``sub_temp == b``.

        Raises:
            WaveformDomainError: fewer than :data:`MIN_RESPONSE_POINTS` grid points.
        """
        n_temp = len(items)
        if isinstance(t_grid, (list, tuple)):
            N = max(int(np.size(g)) for g in t_grid)
        else:
            N = int(np.size(t_grid))
        if N < MIN_RESPONSE_POINTS:
            from ...utils.exceptions import WaveformDomainError

            raise WaveformDomainError(
                f"EMRI sparse response grid of {N} point(s) (fewer than {MIN_RESPONSE_POINTS}): "
                "the trajectory is too short against the TDI delay margin for the response "
                "splines")
        if isinstance(t_grid, (list, tuple)):
            t_eval = np.stack([pad_grid(g, N) for g in t_grid])
        else:
            t_eval = np.tile(np.asarray(t_grid, dtype=float), (n_temp, 1))
        dense, par, mkn, tk_abs, Cs = self._dense_kernel(items, self.t_start + t_eval)
        # outside the trajectory the kernel holds the reference phase at the trajectory end (no jump)
        out = dense(par, return_spline=False)
        return ExactPhaseTDIOutput(out, tk_abs, Cs, mkn, dense.sub_temp_host)

    def td_channels(self, item, ts, sum_subs=False, stop_taper=0.0):
        """EXACT channel time series of one template at the absolute sample times ``ts``.

        The real part of the dense kernel's raw TDI channels: ``(num_sub, nch, ts.size)``, or
        ``(nch, ts.size)`` with ``sum_subs`` (summed inside the kernel). For the stretches an
        amplitude/phase track cannot represent (fast chirp, a signal that stops inside the
        window). ``stop_taper`` [s] > 0 tapers the source over that span of EMISSION time before
        the trajectory's end (``TDDenseTDIonTheFly.channels``). Evaluated in blocks of at most
        ``EMRI_DIRECT_TAIL_BLOCK_ELEMS`` (sub, channel, sample) entries (2e7).
        """
        xp = self.xp
        nch = self.tdi_config.nchannels
        S = 1 if sum_subs else int(item[0][2].shape[0])
        ts = xp.asarray(ts, dtype=float)
        out = xp.empty((S, nch, int(ts.size)))
        blk = max(1, int(float(os.environ.get("EMRI_DIRECT_TAIL_BLOCK_ELEMS", 2e7)) // (S * nch)))
        for b0 in range(0, int(ts.size), blk):
            tb = ts[b0:b0 + blk]
            dense, par = self._dense_kernel([item], tb[None, :])[:2]
            out[:, :, b0:b0 + tb.size] = xp.real(dense.channels(par, sum_subs=sum_subs, stop_taper=stop_taper))
        return out[0] if sum_subs else out
