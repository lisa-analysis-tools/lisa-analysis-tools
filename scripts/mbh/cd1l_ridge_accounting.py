#!/usr/bin/env python3
"""Gate 2 for the CD1-L ridge plan: who owns the sky/time excess?

The mixing surveys say the Fisher matches the posterior width for the
intrinsic block but the posterior is 2-600x wider than the Fisher in
(ra, sin_dec, t_plunge). This script asks, on the EXISTING post1d_mc chains
(truth sky mode, second half), how much of the ra / sin_dec / t_plunge
variance is explained by

  B  the LISA Doppler delay:  t depends on the sky only through n_hat . r_LISA / c,
     so regress t_plunge on the three ecliptic components of n_hat (exact for
     any orbital phase, no ephemeris needed) -- what sampling t_LISA instead
     of t_SSB would remove;
  A  the extrinsic block (psi, phi_ref, dist): linear terms plus the harmonics
     the response is built from (sin/cos 2psi, 4psi; sin/cos m phi, m=1..4) --
     what marginalising them analytically would remove;
  L  everything linear in the other eight coordinates (the Fisher's world);
  L+A+B  all of the above.

Residual sd / raw sd per coordinate, out of sample (fit on odd walkers, score
on even), so a large basis cannot flatter itself. Login node, no GPU.

    python cd1l_ridge_accounting.py [--runs roulet_spins|runs_dt25_o8] [--ids 0 2 14 19]
"""
import os, sys, json, argparse
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cd1l_sampling import _to_ecliptic, stock_to_mc, BASIS_MC, I

ROOT = "/data/nbody/majoburo/cd1l_pe"
PER = {I["phi_ref"]: 2 * np.pi, I["psi"]: np.pi, I["ra"]: 2 * np.pi}


def load_chain(path):
    with h5py.File(path, "r") as f:
        g = f["mcmc"]; it = int(g.attrs.get("iteration", 0))
        ch = g["chain/mbh"][:it]          # (it, nsamplers, ntemps, nw, nleaves, ndim)
    return ch[:, 0, 0, :, 0, :], it       # COLD chain only (ntemps=5 in these runs)


def injection_mc(sid, runs="runs"):
    # Each campaign saves its OWN injection: t_plunge carries that run's
    # waveform_t0 lattice snap (0 to +-5 s between dt=10 and dt=2.5), which is
    # up to 9 sigma for the loud sources.  Score a chain against its own runs dir.
    p = f"{ROOT}/{runs}/roulet_spins/injection_id{sid}_mc.npy"
    if os.path.exists(p):
        return np.load(p)
    return stock_to_mc(np.load(f"{ROOT}/runs/injection_id{sid}.npy"))


def truth_mode(x, inj):
    lam, beta, _ = _to_ecliptic(x[:, I["ra"]], np.arcsin(np.clip(x[:, I["sin_dec"]], -1, 1)))
    lam0, beta0, _ = _to_ecliptic(inj[I["ra"]], np.arcsin(inj[I["sin_dec"]]))
    k = np.round(((lam - lam0[0]) % (2 * np.pi)) / (np.pi / 2)) % 4
    return (k == 0) & (np.sign(beta) == np.sign(beta0[0])), lam, beta


def unwrap(x, inj):
    x = x.copy()
    for j, per in PER.items():
        x[:, j] = inj[j] + (x[:, j] - inj[j] + per / 2) % per - per / 2
    return x


def recentre_periodic(x):
    """Re-centre every periodic coordinate on its OWN circular mean.

    unwrap() puts the branch cut half a period from the injection, which lands
    inside the lobe whenever the posterior sits far from truth (id19: phi_ref
    lobe at 0.10, injection at 3.84 -> the lobe is split and its sd inflated
    0.52 -> 1.77).  Any covariance built from such samples is meaningless in
    that coordinate and in every correlation with it.  The centres themselves
    are never returned: only second moments are affected.
    """
    x = x.copy()
    for j, per in PER.items():
        c = np.angle(np.mean(np.exp(2j * np.pi * x[:, j] / per))) * per / (2 * np.pi)
        x[:, j] = c + (x[:, j] - c + per / 2) % per - per / 2
    return x


def resid_sd(y_tr, X_tr, y_te, X_te):
    """out-of-sample residual sd after least-squares on X (with intercept)."""
    A = np.column_stack([np.ones(len(X_tr)), X_tr]); B = np.column_stack([np.ones(len(X_te)), X_te])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return float((y_te - B @ coef).std())


def bases(x, lam, beta):
    psi, phi, d = x[:, I["psi"]], x[:, I["phi_ref"]], x[:, I["dist"]]
    n_hat = np.column_stack([np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)])
    harm = [np.sin(2 * psi), np.cos(2 * psi), np.sin(4 * psi), np.cos(4 * psi)]
    harm += [f(m * phi) for m in (1, 2, 3, 4) for f in (np.sin, np.cos)]
    harm = np.column_stack(harm)
    cross = np.column_stack([harm[:, :4][:, i] * harm[:, 4:][:, j] for i in range(4) for j in range(8)])
    A_lin = np.column_stack([psi, phi, d])
    A_full = np.column_stack([A_lin, harm, cross, d[:, None] * harm])
    others = [I[k] for k in ("lnMc", "q", "s1z", "s2z", "dist", "phi_ref", "cos_iota", "psi")]
    L = x[:, others]
    return dict(B=n_hat, A=A_full, L=L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="runs (old campaign) or runs_dt25_o8")
    ap.add_argument("--ids", nargs="*", type=int, default=list(range(20)))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    survey = {r["id"]: r for r in json.load(open(f"{ROOT}/" + ("mixing_survey_dt25_o8.json" if "dt25" in args.runs else "mixing_survey.json")))}
    rows = []
    hdr = f"{'id':>3} {'snr':>5} {'n_tm':>6} {'tm%':>4} | {'coord':>8} {'sd_raw':>8} {'post/F':>7} | {'B':>5} {'A':>5} {'L':>5} {'L+A':>5} {'L+A+B':>6} {'all+skyX':>8}"
    print(hdr)
    for sid in args.ids:
        path = f"{ROOT}/{args.runs}/roulet_spins/cd1l_mbh_id{sid}_post1d_mc.h5"
        if not os.path.exists(path):
            print(f"{sid:3d} missing {path}"); continue
        ch, it = load_chain(path)
        inj = injection_mc(sid, args.runs)
        half = ch[it // 2:]                                     # (n, nw, 11)
        nw = half.shape[1]
        tr = half[:, 0::2].reshape(-1, 11); te = half[:, 1::2].reshape(-1, 11)
        out_src = dict(id=sid, snr=survey[sid]["snr"], coords={})
        for tag, x in (("tr", tr), ("te", te)):
            x = unwrap(x, inj)
            m, lam, beta = truth_mode(x, inj)
            out_src[tag] = (x[m], lam[m], beta[m])
        xtr, ltr, btr = out_src.pop("tr"); xte, lte, bte = out_src.pop("te")
        if len(xtr) < 200 or len(xte) < 200:
            print(f"{sid:3d} too few truth-mode samples ({len(xtr)}, {len(xte)})"); continue
        Btr, Bte = bases(xtr, ltr, btr), bases(xte, lte, bte)
        tm_frac = (len(xtr) + len(xte)) / (2 * len(tr))
        for coord in ("ra", "sin_dec", "t_plunge"):
            j = I[coord]; y_tr, y_te = xtr[:, j], xte[:, j]
            sd = float(y_te.std())
            # sky cross terms: the other sky coordinate, linear (for t: both)
            sky_others = [I[c] for c in ("ra", "sin_dec", "t_plunge") if c != coord]
            r = dict(sd_raw=sd, post_over_fisher=survey[sid]["sig_ratio_post_fisher"][coord])
            r["B"] = resid_sd(y_tr, Btr["B"], y_te, Bte["B"]) / sd if coord == "t_plunge" else \
                     resid_sd(y_tr, xtr[:, [I["t_plunge"]]], y_te, xte[:, [I["t_plunge"]]]) / sd
            r["A"] = resid_sd(y_tr, Btr["A"], y_te, Bte["A"]) / sd
            r["L"] = resid_sd(y_tr, Btr["L"], y_te, Bte["L"]) / sd
            r["L+A"] = resid_sd(y_tr, np.column_stack([Btr["L"], Btr["A"]]), y_te, np.column_stack([Bte["L"], Bte["A"]])) / sd
            XB_tr = Btr["B"] if coord == "t_plunge" else xtr[:, [I["t_plunge"]]]
            XB_te = Bte["B"] if coord == "t_plunge" else xte[:, [I["t_plunge"]]]
            r["L+A+B"] = resid_sd(y_tr, np.column_stack([Btr["L"], Btr["A"], XB_tr]), y_te,
                                  np.column_stack([Bte["L"], Bte["A"], XB_te])) / sd
            r["all+skyX"] = resid_sd(y_tr, np.column_stack([Btr["L"], Btr["A"], XB_tr, xtr[:, sky_others]]),
                                     y_te, np.column_stack([Bte["L"], Bte["A"], XB_te, xte[:, sky_others]])) / sd
            out_src["coords"][coord] = r
            print(f"{sid:3d} {survey[sid]['snr']:5.0f} {len(xte):6d} {100*tm_frac:4.0f} | {coord:>8} {sd:8.3g} {r['post_over_fisher']:7.1f} | "
                  f"{r['B']:5.2f} {r['A']:5.2f} {r['L']:5.2f} {r['L+A']:5.2f} {r['L+A+B']:6.2f} {r['all+skyX']:8.2f}")
        rows.append(out_src)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1, default=float)
        print("saved", args.out)


if __name__ == "__main__":
    main()
