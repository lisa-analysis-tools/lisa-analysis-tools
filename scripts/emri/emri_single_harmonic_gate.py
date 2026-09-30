"""Single-harmonic lookup gate for EMRIs (Task A6, intrinsic variant).

Truth: ONE EMRI harmonic Re[Y_lm A(t) exp(-i Phi(t))] on the dense grid, with the
EXACT phase from the integrator's dense output (8th order) and the amplitude from
the cubic spline of the Teukolsky knots, transformed TD->WDM with lisatools.
Model: WDMLookupTable.get_wdm_coeffs fed the harmonic track at pixel times
t_n = n * layer_dt (the builder's convention): amplitude |Y A|, carrier phase
Phi - arg(Y A), f, fdot from the derivative spline (never from phase differencing).

Reported per region of the inspiral (t / t_plunge < 0.9, 0.9-0.99, > 0.99): the
per-pixel relative error on the carrier layer (median / max) and the rel L2 error
over the three lit layers. Controls: --no-fdot feeds fdot = 0. --calibrate prints
pixel error vs the cubic phase proxy (pi/3)|fddot| (k layer_dt)^3 for the handoff.

Grid: Nf=180, dt=20 s (layer_dt 3600 s, the production value), Nt=1024 (42.7 d);
the table must be built on the same Nf/dt/layer_dt.
"""

import argparse
import os

import numpy as np

from lisatools.utils.constants import YRSID_SI

NF, DT = 180, 20.0
NT = int(os.environ.get("GATE_NT", "1024"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--p0", type=float, default=float(os.environ.get("GATE_P0", "7.0")))
    ap.add_argument("--m1", type=float, default=1e6)
    ap.add_argument("--m2", type=float, default=1e2)
    ap.add_argument("--a", type=float, default=0.9)
    ap.add_argument("--e0", type=float, default=0.4)
    ap.add_argument("--mode", default="2,2,0,0")
    ap.add_argument("--no-fdot", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    from scipy.interpolate import CubicSpline

    from few.waveform import FastKerrEccentricEquatorialFlux

    from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings
    from lisatools.sources.emri.wdm_direct import harmonic_tracks_from_holder

    wdm = WDMSettings(Nf=NF, Nt=NT, dt=DT, force_backend="cpu")
    ldt, ldf = wdm.layer_dt, wdm.layer_df
    table = WDMLookupTable.from_file(args.table, force_backend="cpu")
    assert (table.Nf, float(table.data_dt)) == (NF, DT), (table.Nf, table.data_dt)
    fdot_axis_max = float(np.max(np.abs(table.fdot_vals)))

    mode = tuple(int(x) for x in args.mode.split(","))
    gen = FastKerrEccentricEquatorialFlux(
        force_backend="cpu", inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e4)},
        sum_kwargs={"pad_output": True})
    T_s = NT * ldt
    H = gen(args.m1, args.m2, args.a, args.p0, args.e0, 1.0, 1.0, 0.3, dist=1.0,
            Phi_phi0=0.2, Phi_theta0=0.0, Phi_r0=0.7, T=T_s / YRSID_SI, dt=DT,
            return_sparse_holder=True, include_minus_mkn=False, mode_selection=[mode])
    integ = gen.inspiral_generator.inspiral_generator
    t_k = np.asarray(H.t_arr, dtype=float)
    t_end = float(t_k[-1])
    plunges = t_end < 0.999 * T_s
    print(f"mode {mode}: knots={t_k.size}, trajectory ends {t_end / 86400:.2f} d of {T_s / 86400:.1f} d "
          f"({'PLUNGES' if plunges else 'no plunge in window'})", flush=True)

    # --- truth: dense harmonic, exact phase ------------------------------------------------
    N = NF * NT
    t = np.arange(N) * DT
    live = t <= t_end
    ph_dense = np.zeros(N)
    ph_dense[live] = np.asarray(integ.eval_integrator_spline(t[live]))[:, 3:6] @ np.array(mode[1:], float)
    teuk = np.asarray(H.teuk_modes)[:, 0]
    A = np.zeros(N, dtype=complex)
    A[live] = (CubicSpline(t_k, teuk.real)(t[live]) + 1j * CubicSpline(t_k, teuk.imag)(t[live])) * H.ylms[0]
    const_amp = bool(os.environ.get("GATE_CONST_AMP"))
    if const_amp:   # diagnostic: freeze the amplitude so only the phase model is tested
        A0 = A[live][np.argmin(np.abs(t[live] - 0.5 * t_end))]
        A = np.where(live, A0, 0.0)
    y = np.where(live, np.real(A * np.exp(-1j * ph_dense)), 0.0)[None, :]
    truth = np.asarray(TDSignal(y, TDSettings(N, DT, force_backend="cpu")).transform(wdm).arr)[0]

    # --- model: lookup from the harmonic track at pixel times ---------------------------------
    # the synthetic truth switches on abruptly at t=0: its WDM edge contamination decays
    # algebraically (>1e-5 within ~24 px, tests/test_wdm_chunk_splice.py) -> skip 32 px
    n = np.arange(int(os.environ.get("GATE_START_SKIP", "32")), NT - 8)
    n = n[n * ldt <= t_end]
    tr = harmonic_tracks_from_holder(H, integ, n * ldt, a=args.a, xI0=1.0)[0]
    tr_amp = np.full_like(tr.amp, A0) if const_amp else tr.amp
    amp = np.abs(tr_amp)
    phi = tr.phase - np.angle(tr_amp)                          # Re[amp e^{-i Phi}] = |amp| cos(Phi - arg amp)
    fdot = np.zeros_like(tr.fdot) if args.no_fdot else tr.fdot
    inside = np.abs(fdot) <= fdot_axis_max
    print(f"harmonic f {tr.f.min():.3e}..{tr.f.max():.3e} Hz, max|fdot| {np.abs(tr.fdot).max():.3e} "
          f"(table axis {fdot_axis_max:.3e}), max|fddot| {np.abs(tr.fddot).max():.3e}; "
          f"{(~inside).sum()} pixels beyond the fdot axis (excluded)", flush=True)
    n, amp, phi, f, fdot, fddot = n[inside], amp[inside], phi[inside], tr.f[inside], fdot[inside], tr.fddot[inside]
    coeffs, m_map = table.get_wdm_coeffs(amp, phi, f, fdot, n)
    coeffs, m_map = np.asarray(coeffs), np.asarray(m_map)

    # --- errors per region -------------------------------------------------------------------------
    frac = n * ldt / t_end
    m_car = (f / ldf).astype(int)
    tv_car = truth[m_car - int(wdm.ind_min_f), n]
    j_car = np.argmax(m_map == m_car[:, None], axis=1)
    lk_car = coeffs[np.arange(n.size), j_car]
    pix_rel = np.abs(lk_car - tv_car) / np.maximum(np.abs(tv_car), 1e-300)
    print(f"\n{'region':>14} {'pixels':>6} {'carrier median':>15} {'carrier max':>12} {'rel L2 (3 layers)':>18}")
    for lo, hi, name in ((0.0, 0.9, "t<0.90 tp"), (0.9, 0.99, "0.90-0.99"), (0.99, 1.01, ">0.99 (plunge)")):
        sel = (frac >= lo) & (frac < hi)
        if not np.any(sel):
            print(f"{name:>14} {0:>6}")
            continue
        num = den = 0.0
        for j in range(coeffs.shape[1]):
            ok = sel & (m_map[:, j] >= 0)
            tv = truth[m_map[ok, j] - int(wdm.ind_min_f), n[ok]]
            num += np.sum((coeffs[ok, j] - tv) ** 2)
            den += np.sum(tv ** 2)
        print(f"{name:>14} {int(sel.sum()):>6} {np.median(pix_rel[sel]):>15.2e} {pix_rel[sel].max():>12.2e} "
              f"{np.sqrt(num / den):>18.2e}", flush=True)

    n_exact = int(os.environ.get("GATE_EXACT_LOCAL", "0"))
    if n_exact:
        # the premise method (no table): an exact local chirp from (A, Phi, f, fdot) at the
        # pixel centre, extended over the grid and TD->WDM; splits model vs table error
        idx = np.unique(np.linspace(0, n.size - 1, n_exact).astype(int))
        tds = TDSettings(N, DT, force_backend="cpu")
        rows = []
        for i in idx:
            tau = t - n[i] * ldt
            yl = amp[i] * np.cos(2 * np.pi * (f[i] * tau + 0.5 * fdot[i] * tau ** 2) + phi[i])[None, :]
            wl = np.asarray(TDSignal(yl, tds).transform(wdm).arr)[0]
            ok = m_map[i] >= 0
            ms = m_map[i, ok] - int(wdm.ind_min_f)
            loc, tv, lk = wl[ms, n[i]], truth[ms, n[i]], coeffs[i, ok]
            rows.append((frac[i], np.linalg.norm(loc - tv) / np.linalg.norm(tv), np.linalg.norm(lk - loc) / np.linalg.norm(loc)))
        rows = np.array(rows)
        print("\nexact local chirp (premise method), 3-layer rel err per pixel:")
        for lo, hi, name in ((0.0, 0.9, "t<0.90 tp"), (0.9, 0.99, "0.90-0.99"), (0.99, 1.01, ">0.99")):
            sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
            if np.any(sel):
                print(f"  {name:>10}: model (local vs truth) median {np.median(rows[sel, 1]):.2e} max {rows[sel, 1].max():.2e} | "
                      f"table (lookup vs local) median {np.median(rows[sel, 2]):.2e} max {rows[sel, 2].max():.2e}  [{int(sel.sum())} px]")

    if os.environ.get("GATE_PROFILE"):
        dlnA = np.gradient(np.log(np.abs(tr.amp)), tr.t) * ldt      # relative amp change per layer
        dargA = np.gradient(np.unwrap(np.angle(tr.amp)), tr.t) * ldt   # amp phase change per layer [rad]
        print("\nprofile: n, t/tp, carrier rel err, |dlnA|/layer, |d argA|/layer [rad], fdot [units]")
        for i in list(range(0, 40, 4)) + list(range(40, n.size, max(1, n.size // 12))):
            k = int(np.searchsorted(tr.t, n[i] * ldt))
            print(f"  n={n[i]:4d} t/tp={frac[i]:.3f} err={pix_rel[i]:.2e} dlnA={abs(dlnA[k]):.2e} dargA={abs(dargA[k]):.2e} "
                  f"fdot={fdot[i] / (ldf / ldt):+.3f}")

    if args.calibrate:
        print("\ncalibration: carrier-pixel rel err vs cubic proxy (pi/3)|fddot|(k ldt)^3, k=1")
        cub = (np.pi / 3) * np.abs(fddot) * ldt ** 3
        for q in (0.5, 0.9, 0.99, 0.999):
            idx = int(np.searchsorted(np.sort(frac), q))
            idx = min(idx, n.size - 1)
            print(f"  t/tp={frac[idx]:.4f}: cubic(k=1)={cub[idx]:.2e} rad  carrier rel err={pix_rel[idx]:.2e}")
        for thr in (1e-4, 1e-3, 1e-2):
            bad = np.flatnonzero(pix_rel > thr)
            first = bad[np.searchsorted(bad, np.argmax(frac > 0.5))] if bad.size and np.any(bad >= np.argmax(frac > 0.5)) else None
            if first is not None:
                print(f"  carrier err first > {thr:g} after mid-inspiral at t/tp={frac[first]:.4f}: cubic(k=1)={cub[first]:.2e} rad "
                      f"-> k for 0.1 rad = {(0.1 / max(cub[first], 1e-300)) ** (1 / 3):.2f}")


if __name__ == "__main__":
    main()
