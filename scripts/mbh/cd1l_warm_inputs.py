#!/usr/bin/env python3
"""Warm-restart inputs for cd1l_pe.py: the truth-mode, cold, second-half
covariance of a finished chain (unwrapped around the injection), saved as the
CD1L_PROPOSAL_COV file. The chain itself is the CD1L_INIT_CHAIN pool.

    python cd1l_warm_inputs.py --runs runs --ids 0 2 14 19 --out /data/nbody/majoburo/cd1l_pe/runs_warmcov
"""
import os, sys, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cd1l_ridge_accounting import (load_chain, injection_mc, truth_mode, unwrap,
                                   recentre_periodic, ROOT, BASIS_MC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--ids", nargs="*", type=int, default=[0, 2, 14, 19])
    ap.add_argument("--out", default=f"{ROOT}/runs_warmcov")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for sid in args.ids:
        path = f"{ROOT}/{args.runs}/roulet_spins/cd1l_mbh_id{sid}_post1d_mc.h5"
        ch, it = load_chain(path); inj = injection_mc(sid, args.runs)
        x = unwrap(ch[it // 2:].reshape(-1, 11), inj)
        m, _, _ = truth_mode(x, inj); x = x[m]
        # Unwrapping around the INJECTION drops the branch cut INSIDE the lobe
        # whenever the posterior sits far from truth: id19's phi_ref lobe is at
        # 0.10 while the injection is at 3.84, so the lobe was split and its sd
        # inflated 0.52 -> 1.77, which took RouletSpinGaussianMove's acceptance
        # to 0.00.  (No-op for id2/id14, whose lobes clear the cut.)
        x = recentre_periodic(x)
        C = np.cov(x.T); s = np.sqrt(np.diag(C))
        w = np.linalg.eigvalsh(C / np.outer(s, s))
        fn = f"{args.out}/chaincov_id{sid}.npy"; np.save(fn, C)
        json.dump(dict(source_chain=path, iteration=it, n_truth_mode=int(len(x)),
                       sigma=dict(zip(BASIS_MC, s.tolist())), corr_eig_min=float(w.min()),
                       corr_eig_max=float(w.max())), open(fn.replace(".npy", ".json"), "w"), indent=1)
        print(f"id{sid}: {len(x)} samples, corr eig [{w.min():.2e}, {w.max():.2f}], sigma "
              + " ".join(f"{k}={v:.2g}" for k, v in zip(BASIS_MC, s)) + f"\n   -> {fn}")


if __name__ == "__main__":
    main()
