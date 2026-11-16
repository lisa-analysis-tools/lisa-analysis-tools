"""Prototype: sparse EMRI harmonic tracks -> STFT Fresnel kernel (array entry point) -> template.

Accuracy study for the sparse-EMRI STFT likelihood (plan stage S2). The reference is the dense
window + rfft of the EXACT TDI channels from the same dense kernel
(EMRISparseResponse.td_channels), so the mismatch isolates the per-segment linear-chirp
(Fresnel) model and the side-bin truncation; it does NOT compare against the Lagrange dense
response of EMRITDIWaveform. White-noise mismatch over (channel, segment, bin).

Usage (FEW gpu_backend + its data files needed):
    python sparse_stft_fresnel_prototype.py [span_days=24] [p0=9.5] [m2=10] [e0=0.3]
e.g. the non-plunging case 24 and dev's plunging gate source 36 7.0 100 0.4.

Findings so far (CPU, 2026-11-16) are recorded in the branch HANDOFF.md.
"""
import sys, time, numpy as np
from scipy.signal.windows import tukey
import lisatools
from few.waveform import FastKerrEccentricEquatorialFlux
from lisatools.detector import EqualArmlengthOrbits
from lisatools.response.tdiconfig import TDIConfig
from lisatools.sources.emri.sparse_tracks import EMRISparseResponse

be = lisatools.get_backend("cpu")
DAY = 86400.0

def fresnel_rows(T_seg, alpha, amp, ph, f, fd, t_start, freqs, midpoint=True):
    """0.5 * STFT of amp*exp(+i Phi) per row; freqs (rows, nf)."""
    fr = be.STFTFresnelWrap(1, 1, 3, 0.0, 0.0, 1.0, T_seg, 1.0 / T_seg, alpha, midpoint, False)
    rows, nf = freqs.shape
    out = np.zeros(rows * nf, dtype=complex)
    c = np.ascontiguousarray
    fr.compute_fourier_values(out, c(amp), c(ph), c(f), c(fd), c(t_start), c(freqs.ravel()), 1.0, rows, nf)
    return 0.5 * out.reshape(rows, nf)

def dense_stft(x, dt, nper, alpha):
    """x (nch, NT*nper) -> dt * rfft(w x) per segment, (nch, NT, nper//2+1)."""
    w = tukey(nper, alpha) if alpha > 0 else np.ones(nper)
    return dt * np.fft.rfft(x.reshape(x.shape[0], -1, nper) * w, axis=-1)

def mismatch(a, b):
    ab = np.vdot(a, b).real
    return 1.0 - ab / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)

# ---- step 0: convention check on one analytic linear chirp --------------------------------
T_seg, dt = 21600.0, 5.0
nper = int(T_seg / dt)
tt = np.arange(nper) * dt
A, phi_m, f_m, fd_m = 1.3, 0.7, 3.0e-3 + 0.37 / T_seg, 4e-10
tau = tt - T_seg / 2
x = A * np.cos(phi_m + 2 * np.pi * (f_m * tau + 0.5 * fd_m * tau**2))
for alpha in (0.0, 0.1):
    ref = dense_stft(x[None], dt, nper, alpha)[0, 0]
    k0 = int(round(f_m * T_seg))
    ks = np.arange(k0 - 40, k0 + 41)
    val = fresnel_rows(T_seg, alpha, np.array([A]), np.array([phi_m]), np.array([f_m]), np.array([fd_m]),
                       np.array([0.0]), (ks / T_seg)[None])[0]
    print(f"[convention] alpha={alpha}: max |fresnel - rfft| / max|rfft| = {np.abs(val - ref[ks]).max() / np.abs(ref).max():.2e}")

# ---- EMRI ------------------------------------------------------------------------------------
gen = FastKerrEccentricEquatorialFlux(inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4)},
                                      sum_kwargs={"pad_output": True}, force_backend="cpu")
T0 = 2.0e7
span = float(sys.argv[1]) * DAY if len(sys.argv) > 1 else 24 * DAY
dt = 10.0
few_args = (1e6, float(sys.argv[3]) if len(sys.argv) > 3 else 10., 0.9, float(sys.argv[2]) if len(sys.argv) > 2 else 9.5, float(sys.argv[4]) if len(sys.argv) > 4 else 0.3, 1., 1.0, 0.8, 1.1, 0.5, 2.0, 0.3, 0.0, 1.2)
eps = 1e-3
d = EMRISparseResponse(gen, orbits=EqualArmlengthOrbits(force_backend="cpu"),
                       tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
                       t_start=T0 - 2 * DAY, t_window=(T0, T0 + span), data_dt=dt)
d.template_inputs(few_args, lam=1.1, beta=np.pi / 2 - 0.8, mode_selection_threshold=eps)   # warm-up
t1 = time.perf_counter()
item, H = d.template_inputs(few_args, lam=1.1, beta=np.pi / 2 - 0.8, mode_selection_threshold=eps)
t2 = time.perf_counter()
out = d.tracks([item], d.sparse_grid(H))
t3 = time.perf_counter()
S_ = item[0][2].shape[0]
t_end_traj = d.t_start + H.t_arr[-1]
print(f"\nEMRI p0={few_args[3]}: {S_} harmonics, traj ends {(t_end_traj - T0) / DAY:.1f} d after window start (window {span / DAY:.0f} d); "
      f"FEW+inputs {1e3 * (t2 - t1):.0f} ms, sparse response {1e3 * (t3 - t2):.0f} ms")
N = int(span / dt)
ts = T0 + np.arange(N) * dt
t4 = time.perf_counter()
x = np.asarray(d.td_channels(item, ts, sum_subs=True))                    # (3, N) exact reference
print(f"dense exact reference: {N} samples in {time.perf_counter() - t4:.1f} s")

for T_seg in (21600.0, 86400.0):
    nper = int(T_seg / dt); NT = N // nper; df = 1.0 / T_seg; NF = nper // 2 + 1
    tm = T0 + (np.arange(NT) + 0.5) * T_seg
    tstart = T0 + np.arange(NT) * T_seg
    t5 = time.perf_counter()
    x_hi = float(np.asarray(out.x).max())
    valid = (tstart + T_seg) <= x_hi                                       # whole segment inside the track span
    amp = np.zeros((S_, 3, NT)); ph = np.zeros_like(amp); f = np.full_like(amp, 1e-3); fd = np.full_like(amp, 1e-12)
    if valid.any():
        a_, p_, f_, d_ = (np.asarray(v) for v in out.eval_tracer(tm[valid]))
        amp[..., valid], ph[..., valid], f[..., valid], fd[..., valid] = a_, p_, f_, d_
    t_tr = time.perf_counter() - t5
    print(f"   fdot: min |fdot| {np.abs(fd[..., valid]).min():.1e}, negative fraction {np.mean(fd[..., valid] < 0):.3f}; ref finite: {np.isfinite(x).all()}")
    print(f"   negative-amplitude fraction {np.mean(amp[..., valid] < 0):.3f}")
    ph = np.where(amp < 0, ph + np.pi, ph); amp = np.abs(amp)             # signed amplitude -> (|A|, phase + pi)
    FLOOR = 1e-22
    fd = np.where(np.abs(fd) < FLOOR, np.where(fd < 0, -FLOOR, FLOOR), fd)
    sweep = np.abs(fd) * T_seg**2
    # cubic phase from differencing fdot across anchors
    fdd = np.zeros_like(fd); fdd[..., valid] = np.gradient(fd[..., valid], T_seg, axis=-1)
    cub = (np.pi / 3) * np.abs(fdd) * (T_seg / 2) ** 3
    print(f"\n== T_seg = {T_seg / 3600:.0f} h: NT={NT}, df={df:.2e}; tracer {1e3 * t_tr:.1f} ms; "
          f"max sweep {sweep.max():.2f} bins, max cubic phase {cub.max():.2e} rad")
    for alpha in (0.1, 0.5):
        ref = dense_stft(x[:, :NT * nper], dt, nper, alpha)                # (3, NT, NF)
        for n_side in (10, 30, 60):
            t6 = time.perf_counter()
            k0 = np.rint(f / df).astype(int)                               # (S, 3, NT)
            ks = k0[..., None] + np.arange(-n_side, n_side + 1)            # (S, 3, NT, nf)
            rows = ks.reshape(-1, ks.shape[-1])
            vals = fresnel_rows(T_seg, alpha, amp.ravel(), ph.ravel(), f.ravel(), fd.ravel(),
                                np.broadcast_to(tstart, amp.shape).ravel() - T0, rows * df)
            if n_side == 10 and alpha == 0.5:
                bad = ~np.isfinite(vals).all(axis=1)
                if bad.any():
                    i = np.flatnonzero(bad)[0]
                    print(f"      NaN rows: {bad.sum()}/{bad.size}; first: amp={amp.ravel()[i]:.2e} f={f.ravel()[i]:.3e} fdot={fd.ravel()[i]:.2e} sweep={abs(fd.ravel()[i])*T_seg**2:.2e} bins")
            tmpl = np.zeros((3, NT, NF), dtype=complex)
            ci = np.broadcast_to(np.arange(3)[None, :, None, None], ks.shape)
            ji = np.broadcast_to(np.arange(NT)[None, None, :, None], ks.shape)
            ok = (ks >= 0) & (ks < NF)
            np.add.at(tmpl, (ci[ok], ji[ok], ks[ok]), vals.reshape(ks.shape)[ok])
            t_acc = time.perf_counter() - t6
            tmpl[:, ~valid] = ref[:, ~valid]                               # dense fallback stand-in for the tail
            mm = mismatch(ref[:, valid], tmpl[:, valid])
            # per-segment mismatch to localise
            mseg = np.array([mismatch(ref[:, j], tmpl[:, j]) if valid[j] else 0.0 for j in range(NT)])
            amp_ratio = np.sqrt(np.vdot(tmpl, tmpl).real / np.vdot(ref, ref).real)
            if n_side == 60 and alpha == 0.5:
                j_end = int((t_end_traj - T0) // T_seg)
                lo = max(0, min(NT, j_end + 1) - 8)
                print("      last segments before the trajectory end (seg: mismatch, max sweep bins, max cubic rad):")
                for j in range(lo, min(NT, j_end + 1)):
                    if not valid[j]:
                        print(f'        seg {j}: dense fallback (track ends inside)'); continue
                    print(f"        seg {j} ({(t_end_traj - tm[j]) / DAY:+.2f} d to end): {mseg[j]:.1e}, {sweep[..., j].max():.2f}, {cub[..., j].max():.1e}")
                good = (np.arange(NT) < lo) & valid
                if good.any():
                    print(f"      mismatch excluding those: {mismatch(ref[:, good], tmpl[:, good]):.2e}")
            print(f"   alpha={alpha:<4} n_side={n_side:<3} mismatch {mm:.2e}  |h|/|ref| {amp_ratio:.6f}  "
                  f"worst seg {mseg.max():.1e} (seg {mseg.argmax()}/{NT})  median seg {np.median(mseg):.1e}  [{1e3 * t_acc:.0f} ms python]")
