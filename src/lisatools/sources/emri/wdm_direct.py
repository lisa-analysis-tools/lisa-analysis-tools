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
#: Calibrated in Task A6 (scripts/emri/emri_single_harmonic_gate.py): the exact local-chirp
#: model error (1.9e-4 median over the inspiral) passes the table-interpolation error
#: (~3-5e-3) near 0.9 t_plunge, where (pi/3)|fddot| layer_dt^3 ~ 1.2e-3 rad; a 0.1 rad
#: trigger there needs tau ~ 4 layer_dt (the WDM wavelet's tails are wide).
WDM_HALF_SUPPORT_LAYERS = 4.0
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
    t = np.asarray(t_pixels, dtype=float)

    def _ph(tt):
        _, tph, pref = out.eval_spline_vals(tt)
        return np.asarray(tph) + np.asarray(pref)[:, None, :]

    amp, tph, pref = out.eval_spline_vals(t)
    amp = np.asarray(amp)
    ph0 = np.asarray(tph) + np.asarray(pref)[:, None, :]
    php, phm = _ph(t + h), _ph(t - h)
    f = (php - phm) / (2 * h) / (2 * np.pi)
    hd = float(os.environ.get("EMRI_TRACER_H_FDOT", "300")) if h_fdot is None else float(h_fdot)
    fdot = (_ph(t + hd) - 2 * ph0 + _ph(t - hd)) / hd ** 2 / (2 * np.pi)   # phase ~1e5 rad: a short
    #                                                   step turns roundoff into fdot noise
    neg = f < 0
    return amp, np.where(neg, -ph0, ph0), np.where(neg, -f, f), np.where(neg, -fdot, fdot)


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
        n_fine: fine trajectory points over the window (default: one per 80 s).
        pixel_edge: pixels dropped at each grid end (response spline support).
    """

    def __init__(self, few_gen, table, wdm_set, *, orbits, tdi_config, t_start, data_t0,
                 Nt_sub=128, n_fine=None, mode_batch=64, pixel_edge=8, num_m_layers=2,
                 force_backend="cpu"):
        self.few_gen, self.table, self.wdm = few_gen, table, wdm_set
        self.orbits, self.tdi_config = orbits, tdi_config
        self.t_start, self.data_t0 = float(t_start), float(data_t0)
        self.Nt_sub, self.mode_batch, self.pixel_edge = int(Nt_sub), int(mode_batch), int(pixel_edge)
        # layers m-num_m_layers..m+num_m_layers per pixel: a chirping carrier near a layer edge
        # puts ~2.5e-4 of its energy two layers away (A8 gate); needs table support [-2, 3) df
        self.num_m_layers = int(os.environ.get("EMRI_DIRECT_NUM_M_LAYERS", num_m_layers))
        span = wdm_set.Nt * wdm_set.layer_dt
        self.n_fine = int(n_fine) if n_fine is not None else max(1024, int(span / 80.0))
        self.force_backend = force_backend
        self.fdot_axis_max = float(np.max(np.abs(np.asarray(table.fdot_vals))))
        self.last_stats = {}

    def _mode_list(self, few_args, few_kwargs):
        from few.utils.utility import get_viewing_angles

        m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Pp, Pt, Pr = few_args[:14]
        th, ph = get_viewing_angles(qS, phiS, qK, phiK)
        saved = dict(self.few_gen.inspiral_kwargs)
        span = self.wdm.Nt * self.wdm.layer_dt
        lo = self.data_t0 - self.t_start
        try:
            H = self.few_gen(m1, m2, a, p0, e0, x0, th, ph, dist=dist, Phi_phi0=Pp, Phi_theta0=Pt,
                             Phi_r0=Pr, T=(lo + span + 2000.0) / 3.15581497635456e7, dt=self.wdm.data_dt,
                             return_sparse_holder=True, include_minus_mkn=True,
                             inspiral_kwargs={"upsample": True, "fix_t": True,
                                              "new_t": np.linspace(max(lo, 0.0), lo + span, 256)},
                             **{k: v for k, v in few_kwargs.items() if k != "inspiral_kwargs"})
        finally:
            self.few_gen.inspiral_kwargs.clear()
            self.few_gen.inspiral_kwargs.update(saved)
        return [(int(l), int(m), int(k), int(n)) for l, m, k, n in zip(H.ls, H.ms, H.ks, H.ns)]

    def __call__(self, *few_args, **few_kwargs):
        from ...domains import WDMSignal
        from ...wdm_het import splice_chunk, tail_chunk_plan, wdm_chunk_of_td
        from .emritdionfly import EMRITDIonFly

        wdm = self.wdm
        Nf, Nt, dt, ldt, ldf = wdm.Nf, wdm.Nt, wdm.data_dt, wdm.layer_dt, wdm.layer_df
        span = Nt * ldt
        n_all = np.arange(self.pixel_edge, Nt - self.pixel_edge)
        t_pix = self.data_t0 + n_all * ldt
        T_traj = self.data_t0 - self.t_start + span + 2000.0
        acc = np.zeros((3, Nf, Nt))
        n_pad = self.Nt_sub // 4
        few_kwargs = dict(few_kwargs)
        modes = few_kwargs.pop("mode_selection", None)      # explicit (l, m, k, n) list, e.g. gates
        if modes is None:
            modes = self._mode_list(few_args, few_kwargs)
        n_lookup = n_chunk_px = 0
        few_kwargs.pop("mode_selection_threshold", None)

        for j in range(0, len(modes), self.mode_batch):
            fly = EMRITDIonFly(self.few_gen, self.orbits, self.tdi_config, dt, T_traj, self.t_start,
                               frame="icrs_special", n_fine=self.n_fine,
                               t_fine_window=(self.data_t0, self.data_t0 + span))
            out = fly(*few_args, mode_selection=modes[j:j + self.mode_batch], **few_kwargs)
            x = np.asarray(out.x)
            integ = self.few_gen.inspiral_generator.inspiral_generator
            H = fly.last_holder
            t_traj_end = float(np.asarray(H.t_arr)[-1])
            ok_t = (t_pix > x[:, 0].max()) & (t_pix < x[:, -1].min()) & (t_pix - self.t_start <= t_traj_end)
            n_ok, tt = n_all[ok_t], t_pix[ok_t]
            if n_ok.size == 0:
                continue
            tracks = harmonic_tracks_from_holder(H, integ, tt - self.t_start, a=few_args[2], xI0=few_args[5])
            amp, phase, f, fdot = tracer_from_tof_output(out, tt)
            assert amp.shape[0] == len(tracks), (amp.shape, len(tracks))
            # dense TD of every sub over the tail region (chunks), evaluated lazily below
            for s, tr in enumerate(tracks):
                k_h = handoff_pixel(tr, ldt, ldf, self.fdot_axis_max)
                for ch in range(3):
                    sel = np.arange(n_ok.size) < k_h
                    sel &= np.abs(fdot[s, ch]) <= self.fdot_axis_max
                    sel &= f[s, ch] > 2 * ldf
                    if np.any(sel):
                        co, mm = self.table.get_wdm_coeffs(amp[s, ch, sel], phase[s, ch, sel], f[s, ch, sel],
                                                           fdot[s, ch, sel], n_ok[sel], num_m_layers=self.num_m_layers,
                                                           out_of_support="zero")
                        co, mm = np.asarray(co), np.asarray(mm)
                        for c in range(co.shape[1]):
                            good = mm[:, c] >= 0
                            np.add.at(acc[ch], (mm[good, c], n_ok[sel][good]), co[good, c])
                        n_lookup += int(sel.sum())
                if k_h < n_ok.size:
                    # the waveform stops abruptly at plunge: the transform of that stop rings
                    # ~24 px past it (production has the same stop) -> run the chunk tiling a
                    # quarter chunk beyond the last trajectory pixel (td is 0 there)
                    n_h = int(n_ok[k_h])
                    n_end = min(int(n_ok[-1]) + 1 + self.Nt_sub // 4, Nt - self.pixel_edge)
                    for n0, klo, khi in tail_chunk_plan(n_h, n_end, Nt, self.Nt_sub):
                        ts = self.data_t0 + (n0 * Nf + np.arange(Nf * self.Nt_sub)) * dt
                        live = (ts > x[s, 0]) & (ts < x[s, -1])
                        td = np.zeros((3, ts.size))
                        if np.any(live):
                            td[:, live] = np.asarray(out.eval_tdi(ts[live]))[s]
                        chunk = np.asarray(wdm_chunk_of_td(td, 0, Nf, self.Nt_sub, dt, backend=self.force_backend))
                        full_chunk = np.zeros((3, Nf, self.Nt_sub))
                        full_chunk[:, :chunk.shape[-2], :] = chunk   # chunk grid is full-band: rows = global layers
                        tmp = np.zeros_like(acc)
                        splice_chunk(tmp, full_chunk, n0, klo, khi)
                        acc += tmp
                        n_chunk_px += khi - klo
            del out, fly
        self.last_stats = dict(modes=len(modes), lookup_pixels=n_lookup, chunk_pixels=n_chunk_px)
        return WDMSignal(acc, wdm)
