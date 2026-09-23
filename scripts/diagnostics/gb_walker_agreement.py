#!/usr/bin/env python
"""How often do the cold walkers disagree about a galactic binary?

The monitor renders ONE walker (the max-lnL one), so a source that only some
walkers hold is invisible on the page, and a leaf that sits in the wrong local
maximum is counted as a miss for that walker alone. Two dives at 7.444 and
7.599 mHz (2026-09-23) found exactly that in both cases. This scans the whole
band for it.

METHOD, and its one assumption. No injections are used -- the truth set is a
cluster-scale build and its ephemeris was wrong until today. Instead the cold
walkers are compared against EACH OTHER at the waveform level: two leaves are
"the same source" when their phase-maximised, noise-weighted overlap exceeds
``--agree`` (0.9), and a connected component of that relation is one source
candidate. The assumption is that walkers agreeing to overlap 0.9 have found
the same real thing; it says nothing about whether they found it correctly.

Each component is then labelled by how many of the walkers hold it, and every
leaf that joins NO multi-walker component is tested against the components
near it:

  * overlap >= ``--agree``           : it belongs there (handled above)
  * ``--trapped-lo`` .. ``--agree``  : TRAPPED -- partially on a source the
    other walkers place elsewhere. For these the barrier is reported:

        death of the trapped leaf   = -(rho_c rho_m O - 0.5 rho_m^2)
        birth of a companion        = +(rho_c^2 (1 - O) ... ) - 0.5 rho_c^2
        replace (death + birth)     = the sum

    where ``rho_c`` is the consensus template's SNR and ``O`` the overlap.
    A large negative death and a small positive companion-birth is the
    signature of a leaf only a REPLACE move can fix -- and no replace move
    runs in the gb_search stage.
  * below ``--trapped-lo``           : unrelated (a genuine separate leaf)

Waveforms use the injected mojito orbits (``build_truth.l1_orbits``); the
analytic ephemeris changes overlaps at 7.5 mHz from 0.05 to 0.58 and would
make every number here meaningless.

    python scripts/diagnostics/gb_walker_agreement.py STORE [--link-bins 20]
"""

import argparse
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from build_truth import (fitted_noise, sens_grids, nw_for,  # noqa: E402
                         store_tobs, l1_orbits, FLO, FHI)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("--iteration", type=int, default=-1)
    ap.add_argument("--link-bins", type=float, default=20.0,
                    help="pair up leaves within this many FD bins (default 20 "
                         "~ the annual-Doppler sideband half-width at 7 mHz)")
    ap.add_argument("--agree", type=float, default=0.9)
    ap.add_argument("--trapped-lo", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--analytic-orbits", action="store_true")
    ap.add_argument("--out", default=None, help="npz with the per-leaf labels")
    a = ap.parse_args(argv)

    tobs = store_tobs(a.store)
    df, nw = 1.0 / tobs, nw_for(tobs)
    with h5py.File(a.store, "r") as f:
        g = f["global_fit"]
        nit = int(g.attrs["iteration"])
        it = (nit - 1) if a.iteration < 0 else a.iteration
        gb = g["sub_backend/gb"]
        coords = gb["chain"][it, 0]
        alive = gb["inds"][it, 0]
        ll = g["log_like"][it, 0, 0]
        psd_p, gal_p = fitted_noise(a.store, it)
    nwk = coords.shape[0]
    WB = int(np.argmax(ll))
    print(f"iteration {it} of {nit}; {nwk} cold walkers; the page renders "
          f"walker {WB} (max lnL)")

    sa, se = sens_grids(psd_p, gal_p, df)
    from gbgpu.gbgpu import GBGPU
    from lisatools import detector as lisa_models
    from lisatools.globalfit.stock.erebor.variants.gb_no_fg import GB_MOJITO_T_REF
    from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container
    orb = None
    if not a.analytic_orbits:
        orb, src = l1_orbits()
        print(f"orbits: {'mojito L1 ' + src if orb else 'FALLBACK ' + src}")
    if orb is None:
        orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
    gbw = GBGPU(force_backend="cpu", orbits=orb, t0=float(GB_MOJITO_T_REF))
    tc = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                     use_distance=True, mc_lims=(0.001, 1.0))

    rows, who = [], []
    for w in range(nwk):
        r = coords[w][alive[w]]
        k = (r[:, 1] * 1e-3 >= FLO) & (r[:, 1] * 1e-3 <= FHI)
        rows.append(r[k]); who.append(np.full(int(k.sum()), w))
    phys = tc.both_transforms(np.concatenate(rows).astype(float))
    who = np.concatenate(who)
    o = np.argsort(phys[:, 1]); phys, who = phys[o], who[o]
    n = len(phys)
    print(f"{n} cold leaves in [{FLO*1e3:.3g}, {FHI*1e3:.4g}] mHz "
          f"({[int((who == w).sum()) for w in range(nwk)]} per walker)")

    A = np.zeros((n, nw), complex); E = np.zeros((n, nw), complex)
    S = np.zeros(n, int)
    for lo in range(0, n, a.batch):
        p = phys[lo:lo + a.batch]
        gbw.run_wave(*[np.ascontiguousarray(p[:, k]) for k in range(9)], N=nw,
                     T=tobs, dt=2.5, tdi2=True, tdi_channel_setup="AE")
        A[lo:lo + len(p)] = gbw.A; E[lo:lo + len(p)] = gbw.E
        S[lo:lo + len(p)] = np.asarray(gbw.start_inds).astype(int)
        print(f"  waveforms {min(lo + a.batch, n)}/{n}", end="\r", flush=True)
    good = (S >= 0) & (S + nw <= sa.size)
    rho = np.zeros(n)
    for i in np.nonzero(good)[0]:
        rho[i] = np.sqrt(max(4 * df * float(np.sum(
            np.abs(A[i]) ** 2 / sa[S[i]:S[i] + nw]
            + np.abs(E[i]) ** 2 / se[S[i]:S[i] + nw])), 0.0))
    print(f"\nSNR: median {np.median(rho[good]):.1f}, "
          f"{int((rho > 7).sum())} above 7, {int((~good).sum())} off-grid")

    def ovl(i, j):
        lo_ = min(S[i], S[j]); sp = max(S[i], S[j]) + nw - lo_
        if lo_ < 0 or lo_ + sp > sa.size:
            return 0.0
        _sa, _se = sa[lo_:lo_ + sp], se[lo_:lo_ + sp]
        a1 = np.zeros(sp, complex); e1 = np.zeros(sp, complex)
        a2 = np.zeros(sp, complex); e2 = np.zeros(sp, complex)
        k = S[i] - lo_; a1[k:k + nw] = A[i]; e1[k:k + nw] = E[i]
        k = S[j] - lo_; a2[k:k + nw] = A[j]; e2[k:k + nw] = E[j]
        num = abs(np.sum(np.conj(a1) * a2 / _sa) + np.sum(np.conj(e1) * e2 / _se))
        d1 = np.sum(np.abs(a1) ** 2 / _sa) + np.sum(np.abs(e1) ** 2 / _se)
        d2 = np.sum(np.abs(a2) ** 2 / _sa) + np.sum(np.abs(e2) ** 2 / _se)
        return float(num / np.sqrt(max(d1.real * d2.real, 1e-300)))

    # ---- pair every leaf with its neighbours in frequency -------------------
    tol = a.link_bins * df
    f0 = phys[:, 1]
    hi = np.searchsorted(f0, f0 + tol, side="right")
    pairs, ov = [], []
    for i in range(n):
        for j in range(i + 1, int(hi[i])):
            v = ovl(i, j)
            if v > a.trapped_lo:
                pairs.append((i, j)); ov.append(v)
        if i % 500 == 0:
            print(f"  overlaps {i}/{n}", end="\r", flush=True)
    pairs = np.asarray(pairs).reshape(-1, 2); ov = np.asarray(ov)
    print(f"\n{len(pairs)} neighbour pairs above {a.trapped_lo} within "
          f"{a.link_bins:.0f} bins")

    # ---- components of the "same source" relation --------------------------
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for (i, j), v in zip(pairs, ov):
        if v >= a.agree:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    comp = np.array([find(i) for i in range(n)])
    _, comp = np.unique(comp, return_inverse=True)
    ncomp = comp.max() + 1
    cover = np.zeros((ncomp, nwk), bool)
    cover[comp, who] = True
    nw_per = cover.sum(1)
    print(f"\n{ncomp} source candidates (components of overlap >= {a.agree})")
    for k in range(1, nwk + 1):
        m = nw_per == k
        print(f"  held by {k} walker(s): {int(m.sum()):5d} ({100*m.mean():5.1f}%)"
              f"   leaves {int(np.isin(comp, np.nonzero(m)[0]).sum()):6d}")
    shared = nw_per >= 2
    print(f"\n  components held by >=2 walkers (the consensus set): {int(shared.sum())}")
    for w in range(nwk):
        has = cover[:, w] & shared
        print(f"    walker {w}: holds {int(has.sum())} of {int(shared.sum())} "
              f"({100*has.sum()/max(shared.sum(),1):.1f}%), MISSING "
              f"{int((shared & ~cover[:, w]).sum())}"
              + ("   <-- rendered by the page" if w == WB else ""))

    # ---- trapped leaves: in no consensus component, but near one -----------
    in_shared = shared[comp]
    trapped = []
    for (i, j), v in zip(pairs, ov):
        if v >= a.agree:
            continue
        for x, y in ((i, j), (j, i)):
            if (not in_shared[x]) and in_shared[y] and who[x] != who[y]:
                trapped.append((x, comp[y], v, y))
    best = {}
    for x, cy, v, y in trapped:
        if x not in best or v > best[x][1]:
            best[x] = (cy, v, y)
    print(f"\nTRAPPED leaves (overlap {a.trapped_lo}-{a.agree} with a source "
          f"{'>=2'} walkers agree on, but not part of it): {len(best)}")
    if best:
        idx = np.array(sorted(best))
        vs = np.array([best[i][1] for i in idx])
        rc = np.array([rho[best[i][2]] for i in idx])
        rm = rho[idx]
        death = -(rc * rm * vs - 0.5 * rm ** 2)
        companion = rc ** 2 * (1.0 - vs * rm / np.maximum(rc, 1e-30)) - 0.5 * rc ** 2
        replace = death + 0.5 * rc ** 2   # kill it, then birth the true source
        dfb = np.array([(f0[i] - f0[best[i][2]]) / df for i in idx])
        print(f"  per walker: " + "  ".join(
            f"w{w}: {int((who[idx] == w).sum())}" for w in range(nwk)))
        print(f"  |df0| from the consensus leaf [bins]: median "
              f"{np.median(np.abs(dfb)):.1f}, p90 {np.percentile(np.abs(dfb), 90):.1f}")
        print(f"  overlap with it: median {np.median(vs):.3f}")
        print(f"  death of the trapped leaf   [nats]: median {np.median(death):8.1f}"
              f"   p10 {np.percentile(death, 10):8.1f}")
        print(f"  companion birth beside it   [nats]: median {np.median(companion):8.1f}"
              f"   p90 {np.percentile(companion, 90):8.1f}")
        print(f"  replace (death + true birth)[nats]: median {np.median(replace):8.1f}"
              f"   p90 {np.percentile(replace, 90):8.1f}")
        print(f"  trapped leaves whose death costs > 10 nats (cold chain cannot "
              f"kill them): {int((death < -10).sum())} of {len(idx)}")
    if a.out:
        np.savez_compressed(a.out, f0=f0, walker=who, comp=comp, rho=rho,
                            n_walkers_of_comp=nw_per[comp],
                            trapped=np.isin(np.arange(n), list(best)))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
