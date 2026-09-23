#!/usr/bin/env python
"""Pull the per-band temperature-ladder history out of a global-fit store.

WHY THIS EXISTS. The reduced ``*_extract.h5`` carries ``band_temps`` only on
its LAST stored row -- every earlier row reads zero -- so the question "is the
ladder adapting, and is it adapting usefully?" cannot be answered from the
snapshot tar. It has to come off the full store on the cluster. This reads
that, reduces it to a few MB, and prints enough that the answer is legible
without opening the npz.

WHAT IT MEASURES. ``_adapt_band_temps`` is the ptemcee hyperbolic-decay rule:
it nudges adjacent swap-acceptance ratios TOWARD EACH OTHER, with the cold
(beta=1) and hot (beta~0) ends pinned. It has no acceptance target, so a flat
profile is a fixed point no matter what level it is flat AT. The diagnostics
below are therefore:

  * the ladder itself over time -- is it still moving, or converged?
  * the acceptance profile per rung pair -- flat (adaptation done) or sloped
    (still working)?
  * the LEVEL of that profile -- ~0.25 is the textbook efficient value;
    0.7-0.8 means adjacent rungs are nearly the same distribution and the
    ladder is spending rungs it could use elsewhere;
  * the top gap, beta[-2] -> beta[-1]. Measured 0.0153 -> 0.0001 at
    iteration 588 in bands 396/405, a 153x jump against ~1.2x everywhere
    else, accepting 0.39 and 0.05. That is the step a discovery made at
    beta~0 has to take to start coming down the ladder.

USAGE (on the cluster, where the full store lives)::

    export HDF5_USE_FILE_LOCKING=FALSE
    python scripts/diagnostics/gb_band_temp_history.py \\
        /shared/data/global_fit_output/gf_prod_6mo_v8_4gpu/gf_prod_6mo_testing.h5 \\
        --bands 396,405 --out band_temp_hist.npz

Bands 396 and 405 are the 7.444 mHz and 7.599 mHz sub-bands of the 6mo run.
``--bands`` also accepts frequencies in mHz (anything with a dot, e.g.
``7.444,7.599``), resolved through the store's own ``band_edges``.

Reads only; nothing is written to the store. Iterations are streamed in
chunks so the full (n_it, 1232, 24) block is never held at once.
"""

import argparse
import os

import h5py
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("--bands", default="396,405",
                    help="comma list of band INDICES, or frequencies in mHz "
                         "(any entry containing '.' is read as mHz)")
    ap.add_argument("--out", default="band_temp_hist.npz")
    ap.add_argument("--chunk", type=int, default=64,
                    help="iterations per read (memory knob)")
    ap.add_argument("--print-every", type=int, default=0,
                    help="print the summary every N iterations (0 = ~12 rows)")
    a = ap.parse_args(argv)

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    with h5py.File(a.store, "r") as f:
        g = f["global_fit"]
        gb = g["sub_backend/gb"]
        nit = int(g.attrs.get("iteration", gb["band_temps"].shape[0]))
        bt_d = gb["band_temps"]                      # (rows, nbands, ntemps)
        sa_d = gb["band_swaps_accepted"]             # (rows, nbands, ntemps-1)
        sp_d = gb["band_swaps_proposed"]
        nb_d = gb["band_num_binaries"]               # (rows, ntemps, nwalk, nbands)
        edges = gb["band_edges"][:]
        nrow, nband, ntemp = bt_d.shape
        nit = min(nit, nrow)

        sel = []
        for tok in a.bands.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "." in tok:
                b = int(np.searchsorted(edges, float(tok) * 1e-3, side="right") - 1)
                print(f"  {tok} mHz -> band {b}")
            else:
                b = int(tok)
            if 0 <= b < nband:
                sel.append(b)
        sel = sorted(set(sel))

        print(f"store {a.store}")
        print(f"{nit} live iterations, {nband} bands, {ntemp} rungs; "
              f"tracking bands {sel}")

        # ARE THE COUNTERS PER-ITERATION OR RUNNING TOTALS? Decide once, from a
        # strided probe of one pair summed over bands, before any chunk is
        # read. A running total is non-decreasing and ends far above where it
        # starts; a per-iteration count fluctuates about a level.
        probe_rows = np.unique(np.linspace(0, nit - 1, min(nit, 200)).astype(int))
        probe = np.asarray(sp_d[probe_rows, :, 0], float).sum(1)
        nzp = probe[probe > 0]
        cumulative = bool(nzp.size > 2 and np.all(np.diff(nzp) >= 0)
                          and nzp[0] < 0.2 * nzp[-1])
        print(f"  swap counters: "
              f"{'RUNNING TOTALS (differencing)' if cumulative else 'PER-ITERATION'}")

        med = np.zeros((nit, ntemp))          # median ladder over ACTIVE bands
        p10 = np.zeros((nit, ntemp)); p90 = np.zeros((nit, ntemp))
        acc_med = np.zeros((nit, ntemp - 1))  # median swap acceptance per pair
        acc_p10 = np.zeros((nit, ntemp - 1))
        n_active = np.zeros(nit, int)
        n_armed = np.zeros(nit, int)
        bt_sel = np.zeros((nit, len(sel), ntemp))
        acc_sel = np.zeros((nit, len(sel), ntemp - 1))

        for lo in range(0, nit, a.chunk):
            hi = min(lo + a.chunk, nit)
            bt = np.asarray(bt_d[lo:hi], float)
            sa = np.asarray(sa_d[lo:hi], float)
            sp = np.asarray(sp_d[lo:hi], float)
            # cold-rung occupancy, any walker: (chunk, nwalk, nbands) -> (chunk, nbands)
            occ = np.asarray(nb_d[lo:hi, 0], float).max(axis=1) > 0
            if cumulative:
                pre = np.asarray(sa_d[max(lo - 1, 0):lo + 1], float)[:1] if lo else sa[:1]
                sa = np.diff(sa, axis=0, prepend=pre)
                pre = np.asarray(sp_d[max(lo - 1, 0):lo + 1], float)[:1] if lo else sp[:1]
                sp = np.diff(sp, axis=0, prepend=pre)
            # AN EMPTY SUB-BAND ACCEPTS EVERY SWAP -- there is nothing in it to
            # score, so both rungs have the same (zero) likelihood. Including
            # empty bands drags every median to exactly 1.00 and hides the
            # real ladder entirely (~740 of 1232 bands are empty at 6mo).
            act = (bt[:, :, 0] > 0) & occ
            for k in range(hi - lo):
                m = act[k]
                n_active[lo + k] = int(m.sum())
                n_armed[lo + k] = int((bt[k, :, 0] > 0).sum())
                if m.any():
                    med[lo + k] = np.median(bt[k][m], axis=0)
                    p10[lo + k] = np.percentile(bt[k][m], 10, axis=0)
                    p90[lo + k] = np.percentile(bt[k][m], 90, axis=0)
                    r = sa[k][m] / np.maximum(sp[k][m], 1)
                    r[sp[k][m] <= 0] = np.nan
                    with np.errstate(invalid="ignore"):
                        acc_med[lo + k] = np.nanmedian(r, axis=0)
                        acc_p10[lo + k] = np.nanpercentile(r, 10, axis=0)
            for j, b in enumerate(sel):
                bt_sel[lo:hi, j] = bt[:, b]
                acc_sel[lo:hi, j] = sa[:, b] / np.maximum(sp[:, b], 1)
            print(f"  read {hi}/{nit}", end="\r", flush=True)

    live = n_active > 0
    if not live.any():
        print("\nNO iteration has an armed ladder -- band_temps is all zero. "
              "This is the reduced extract, not the full store.")
        return 1
    first = int(np.argmax(live))
    print(f"\nladder armed from iteration {first}; "
          f"{int(live.sum())} iterations carry one")
    print(f"bands used: {n_active[nit-1]} OCCUPIED of {n_armed[nit-1]} armed "
          f"(empty bands accept every swap and are excluded)")
    if int(live.sum()) < 20:
        print("!! Only a handful of rows carry a ladder. This is the REDUCED "
              "*_extract.h5, which keeps band_temps and the swap counters on "
              "its last rows only. Run this against the full "
              "*_testing.h5 on the cluster.")

    rows = ([first] + list(range(first, nit, a.print_every))
            if a.print_every else
            sorted(set(np.linspace(first, nit - 1, 12).astype(int))))
    print("\nMEDIAN LADDER over active bands (beta per rung)")
    print(f"{'iter':>6} " + " ".join(f"T{t:<5}" for t in range(0, ntemp, 3)))
    for i in rows:
        print(f"{i:>6} " + " ".join(f"{med[i, t]:<6.4f}" for t in range(0, ntemp, 3)))

    print("\nMEDIAN SWAP ACCEPTANCE per rung pair, OCCUPIED bands only")
    print("  (~0.25 is the efficient value; 0.7-0.8 means adjacent rungs are "
          "nearly the same distribution)")
    print(f"{'iter':>6} " + " ".join(f"{t}-{t+1:<3}" for t in range(0, ntemp - 1, 3)))
    for i in rows:
        print(f"{i:>6} " + " ".join(f"{acc_med[i, t]:<6.2f}"
                                   for t in range(0, ntemp - 1, 3)))
    print(f"{'p10':>6} " + " ".join(f"{acc_p10[nit-1, t]:<6.2f}"
                                    for t in range(0, ntemp - 1, 3))
          + "   <- worst decile of bands, last iteration")

    print("\nIS IT STILL MOVING?  |d beta| per iteration, summed over rungs")
    d = np.abs(np.diff(med[first:nit], axis=0)).sum(1)
    if len(d) >= 6:
        third = max(len(d) // 3, 1)
        for lab, sl in (("first", slice(0, third)),
                        ("middle", slice(third, 2 * third)),
                        ("last", slice(2 * third, None))):
            print(f"  {lab:>8} third: median {np.median(d[sl]):.3e}  "
                  f"max {d[sl].max():.3e}")
        print("  (falling by orders of magnitude = converged/frozen; flat = "
              "still adapting)")
    else:
        print("  too few armed rows to say")

    print("\nTOP GAP (the step a beta~0 discovery must take to come down)")
    print(f"{'iter':>6} {'beta[-2]':>10} {'beta[-1]':>10} {'ratio':>9} "
          f"{'accept':>8}")
    for i in rows:
        b2, b1 = med[i, -2], med[i, -1]
        print(f"{i:>6} {b2:>10.5f} {b1:>10.5f} {b2 / max(b1, 1e-12):>9.1f} "
              f"{acc_med[i, -1]:>8.3f}")

    for j, b in enumerate(sel):
        lo_e, hi_e = edges[b] * 1e3, edges[b + 1] * 1e3
        print(f"\n=== BAND {b}  [{lo_e:.5f}, {hi_e:.5f}] mHz ===")
        print(f"{'iter':>6} {'b[1]':>8} {'b[6]':>8} {'b[12]':>8} {'b[-2]':>9} "
              f"{'acc 0-1':>8} {'acc mid':>8} {'acc top':>8}")
        for i in rows:
            print(f"{i:>6} {bt_sel[i, j, 1]:>8.4f} {bt_sel[i, j, 6]:>8.4f} "
                  f"{bt_sel[i, j, 12]:>8.4f} {bt_sel[i, j, -2]:>9.5f} "
                  f"{acc_sel[i, j, 0]:>8.2f} "
                  f"{np.nanmedian(acc_sel[i, j, 1:-1]):>8.2f} "
                  f"{acc_sel[i, j, -1]:>8.2f}")

    np.savez_compressed(a.out, med=med, p10=p10, p90=p90, acc_med=acc_med,
                        acc_p10=acc_p10, n_armed=n_armed,
                        n_active=n_active, bands=np.asarray(sel),
                        band_temps_sel=bt_sel, band_acc_sel=acc_sel,
                        band_edges=edges, first_armed=np.array(first),
                        n_iter=np.array(nit))
    print(f"\nwrote {a.out}  "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB) -- send this back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
