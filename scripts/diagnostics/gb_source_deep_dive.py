#!/usr/bin/env python
"""Why did THIS galactic binary miss? A waveform-level dive on one frequency.

The monitor page reports completeness against a ONE-TO-ONE, 2-frequency-bin
match plus a phase-maximised overlap threshold. A source can fail that test
for reasons that are not "the search never found it":

  * a model leaf IS sitting on it but lands outside the 2-bin window
    (frequency bias -- an epoch/Doppler problem, or an unconverged leaf);
  * a leaf sits on it at high overlap but the globally-greedy one-to-one
    assignment handed that leaf to a NEIGHBOURING injection instead;
  * two model leaves split it, so each one alone overlaps poorly;
  * the cap cell it lives in is full, so no birth can land there;
  * it is genuinely absent from the model.

This separates those. For one frequency window it prints, side by side, every
catalogue injection and every cold-chain model leaf, their optimal SNRs under
the run's OWN fitted noise, the full overlap matrix, the implied
log-likelihood cost of each mismatch, and the band / cap-cell geometry the
sampler was working against.

DELTA lnL. With the data written as ``d = h_inj + (everything else)``, the
log-likelihood difference between carrying a model template ``h_mod`` and
carrying the true ``h_inj`` is, dropping the cross-terms with other sources,

    dlnL = -0.5 <h_inj - h_mod | h_inj - h_mod>
         = -0.5 (rho_inj^2 + rho_mod^2 - 2 rho_inj rho_mod * overlap)

so a MISSING source costs ``0.5 rho^2`` and a source recovered at overlap
``O`` with matched SNR costs ``rho^2 (1 - O)``. That is the number that says
whether a miss is a real likelihood failure or a bookkeeping one: the sampler
cannot be blamed for not finding something worth 2 nats, and cannot be
excused for leaving 200 on the table.

Everything -- waveform, noise, transform, catalogue conversion -- comes from
the run's own machinery via ``build_truth`` and the erebor transforms; the
only arithmetic written here is the inner product, copied from the monitor's
recovery section so the numbers are directly comparable to the page's.

    python scripts/diagnostics/gb_source_deep_dive.py STORE --freq 7.444
    python scripts/diagnostics/gb_source_deep_dive.py STORE --freq 7.599 \
        --window-bins 400 --iteration 587

Waveforms use the INJECTED (mojito L1) orbits -- see ``build_truth.l1_orbits``
for why that is not optional above a few mHz. ``--analytic-orbits`` forces the
old analytic ephemeris for debugging only.
"""

import argparse
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# The injected-orbit loader lives in build_truth (the shared helper this
# script already draws its noise, catalogue and grid code from), so the
# truth builder and every diagnostic use ONE definition.
from build_truth import find_l1_brick, l1_orbits  # noqa: E402,F401


def _inner(a1, e1, s1, a2, e2, s2, sa, se, nw):
    """Noise-weighted <1|2>, phase-UNmaximised, on GBGPU's sparse output.

    Copied from ``gf_monitor_gen``'s recovery block (the ``MM`` loop) so this
    script's overlaps are the same statistic the page thresholds at 0.80.
    The modulus of the returned value IS the phase-maximised inner product.
    """
    o = min(s1, s2)
    sp = max(s1, s2) + nw - o
    if o < 0 or o + sp > sa.size:
        return np.nan
    _sa, _se = sa[o:o + sp], se[o:o + sp]
    A1 = np.zeros(sp, complex); E1 = np.zeros(sp, complex)
    A2 = np.zeros(sp, complex); E2 = np.zeros(sp, complex)
    k = s1 - o; A1[k:k + nw] = a1; E1[k:k + nw] = e1
    k = s2 - o; A2[k:k + nw] = a2; E2[k:k + nw] = e2
    return complex(np.sum(np.conj(A1) * A2 / _sa) + np.sum(np.conj(E1) * E2 / _se))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="the run's *_extract.h5 (or full store)")
    ap.add_argument("--freq", type=float, required=True, help="centre [mHz]")
    ap.add_argument("--window-bins", type=float, default=300.0,
                    help="half-width in FD bins (default 300 ~ one sub-band)")
    ap.add_argument("--iteration", type=int, default=-1,
                    help="stored iteration; -1 = last live one")
    ap.add_argument("--tobs", type=float, default=None)
    ap.add_argument("--nw", type=int, default=None)
    ap.add_argument("--catalogue", default=None)
    ap.add_argument("--analytic-orbits", action="store_true",
                    help="use DefaultOrbits instead of the mojito L1 table. "
                         "MEASURED 2026-09-23 at 7.44 mHz: the analytic "
                         "ephemeris turns a 0.580 overlap into 0.052 and a "
                         "0.982 into 0.368. Debug only.")
    ap.add_argument("--hot-rungs", type=int, default=0,
                    help="also census this many temperature rungs (0 = cold only)")
    a = ap.parse_args(argv)

    from build_truth import (fitted_noise, sens_grids, catalogue_phys, nw_for,
                             store_tobs)

    f_c = a.freq * 1e-3
    tobs = float(a.tobs) if a.tobs else store_tobs(a.store)
    df = 1.0 / tobs
    nw = int(a.nw) if a.nw else nw_for(tobs)
    half = a.window_bins * df
    flo, fhi = f_c - half, f_c + half
    print("=" * 78)
    print(f"DEEP DIVE @ {a.freq:.5f} mHz   window +-{a.window_bins:.0f} bins "
          f"= [{flo * 1e3:.5f}, {fhi * 1e3:.5f}] mHz")
    print(f"Tobs {tobs:.0f} s ({tobs / 86400:.1f} d)  df {df:.5g} Hz "
          f"({df * 1e3:.6f} mHz)  N/waveform {nw}")
    print("=" * 78)

    # ---- the store: geometry, caps, and the cold-chain leaves --------------
    with h5py.File(a.store, "r") as f:
        g = f["global_fit"]
        nit = int(g.attrs["iteration"])
        it = (nit - 1) if a.iteration < 0 else a.iteration
        gb = g["sub_backend/gb"]
        be = gb["band_edges"][:]
        ce = gb["cap_edges"][:] if "cap_edges" in gb else be
        cap = gb["cap_cell_leaf_cap"][it] if "cap_cell_leaf_cap" in gb \
            else gb["band_leaf_cap"][it]
        nt_, nwk = gb["chain"].shape[1], gb["chain"].shape[2]
        n_rung = 1 + max(0, min(a.hot_rungs, nt_ - 1))
        coords = gb["chain"][it, :n_rung]
        alive = gb["inds"][it, :n_rung]
        ll = g["log_like"][it, 0, 0]
        psd_p, gal_p = fitted_noise(a.store, it)
    print(f"stored iteration {it} of {nit}; cold lnL per walker "
          f"{np.array2string(ll, precision=1)}")
    print(f"fitted noise: Soms_d {psd_p[0]:.6e}  Sa_a {psd_p[1]:.6e}  "
          f"galfor {np.array2string(np.asarray(gal_p), precision=3)}")

    # geometry around the target
    ib = int(np.searchsorted(be, f_c, side="right") - 1)
    ic = int(np.searchsorted(ce, f_c, side="right") - 1)
    print(f"\nGEOMETRY")
    print(f"  sub-band   {ib}: [{be[ib] * 1e3:.5f}, {be[ib + 1] * 1e3:.5f}] mHz "
          f"(width {(be[ib + 1] - be[ib]) * 1e3:.5f} mHz = "
          f"{(be[ib + 1] - be[ib]) / df:.0f} bins); target sits "
          f"{(f_c - be[ib]) / (be[ib + 1] - be[ib]) * 100:.1f}% across it")
    print(f"  cap cell   {ic}: [{ce[ic] * 1e3:.5f}, {ce[ic + 1] * 1e3:.5f}] mHz, "
          f"cap = {int(cap[ic])}; target sits "
          f"{(f_c - ce[ic]) / (ce[ic + 1] - ce[ic]) * 100:.1f}% across it")
    print(f"  nearest band edge  {min(abs(f_c - be[ib]), abs(be[ib + 1] - f_c)) / df:7.1f} bins away")
    print(f"  nearest cell edge  {min(abs(f_c - ce[ic]), abs(ce[ic + 1] - f_c)) / df:7.1f} bins away")

    # ---- noise, orbits, waveform engine -----------------------------------
    sa, se = sens_grids(psd_p, gal_p, df)
    from gbgpu.gbgpu import GBGPU
    from lisatools import detector as lisa_models
    from lisatools.globalfit.stock.erebor.variants.gb_no_fg import GB_MOJITO_T_REF
    from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container
    orb = None
    if not a.analytic_orbits:
        orb, why = l1_orbits()
        print(f"\norbits: {'mojito L1 ' + why if orb is not None else 'FALLBACK to DefaultOrbits, ' + why}")
    if orb is None:
        orb = lisa_models.DefaultOrbits(force_backend="cpu", frame="icrs")
        if a.analytic_orbits:
            print("\norbits: analytic DefaultOrbits BY REQUEST -- overlaps "
                  "above ~5 mHz are not trustworthy (see --analytic-orbits)")
    gbw = GBGPU(force_backend="cpu", orbits=orb, t0=float(GB_MOJITO_T_REF))

    def wave(phys):
        phys = np.atleast_2d(np.asarray(phys, float))
        gbw.run_wave(*[np.ascontiguousarray(phys[:, k]) for k in range(9)],
                     N=nw, T=tobs, dt=2.5, tdi2=True, tdi_channel_setup="AE")
        return (np.asarray(gbw.A).copy(), np.asarray(gbw.E).copy(),
                np.asarray(gbw.start_inds).astype(int).copy())

    def snr_of(A, E, s):
        out = np.zeros(len(s))
        for i in range(len(s)):
            if s[i] < 0 or s[i] + nw > sa.size:
                continue
            out[i] = np.sqrt(max(4 * df * float(np.sum(
                np.abs(A[i]) ** 2 / sa[s[i]:s[i] + nw]
                + np.abs(E[i]) ** 2 / se[s[i]:s[i] + nw])), 0.0))
        return out

    # ---- injections in the window -----------------------------------------
    inj = catalogue_phys(GB_MOJITO_T_REF, flo, fhi, a.catalogue)
    if not len(inj):
        print("\nNO catalogue sources in this window.")
        return 0
    Ai, Ei, si = wave(inj)
    rho_i = snr_of(Ai, Ei, si)
    o = np.argsort(inj[:, 1])
    inj, Ai, Ei, si, rho_i = inj[o], Ai[o], Ei[o], si[o], rho_i[o]

    # ---- model leaves in the window ---------------------------------------
    tc = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                     use_distance=True, mc_lims=(0.001, 1.0))
    rows, tags = [], []
    for t in range(n_rung):
        for w in range(nwk):
            m = alive[t, w]
            c9 = coords[t, w][m]
            k = (c9[:, 1] * 1e-3 >= flo) & (c9[:, 1] * 1e-3 <= fhi)
            for r in c9[k]:
                rows.append(r); tags.append(f"T{t}w{w}")
    print(f"\ncatalogue injections in window: {len(inj)}   "
          f"model leaves in window: {len(rows)} "
          f"(rungs 0..{n_rung - 1}, {nwk} walkers)")
    if rows:
        mod = tc.both_transforms(np.asarray(rows, float).copy())
        Am, Em, sm = wave(mod)
        rho_m = snr_of(Am, Em, sm)
        o = np.argsort(mod[:, 1])
        mod, Am, Em, sm, rho_m = mod[o], Am[o], Em[o], sm[o], rho_m[o]
        tags = [tags[i] for i in o]
    else:
        mod = np.zeros((0, 9)); rho_m = np.zeros(0)

    hdr = (f"{'':>4} {'f0 [mHz]':>12} {'d bins':>8} {'SNR':>7} {'amp':>10} "
           f"{'fdot':>11} {'iota':>7} {'psi':>7} {'lam':>7} {'beta':>7}")
    print("\nINJECTIONS (catalogue, this run's epoch/frame)")
    print(hdr)
    for i in range(len(inj)):
        p = inj[i]
        print(f"I{i:<3} {p[1] * 1e3:12.7f} {(p[1] - f_c) / df:8.2f} "
              f"{rho_i[i]:7.2f} {p[0]:10.3e} {p[2]:11.3e} {p[5]:7.3f} "
              f"{p[6]:7.3f} {p[7]:7.3f} {p[8]:7.3f}")
    if len(mod):
        print("\nMODEL LEAVES")
        print(hdr + "   tag")
        for i in range(len(mod)):
            p = mod[i]
            print(f"M{i:<3} {p[1] * 1e3:12.7f} {(p[1] - f_c) / df:8.2f} "
                  f"{rho_m[i]:7.2f} {p[0]:10.3e} {p[2]:11.3e} {p[5]:7.3f} "
                  f"{p[6]:7.3f} {p[7]:7.3f} {p[8]:7.3f}   {tags[i]}")

    # ---- overlaps ----------------------------------------------------------
    def ov(A1, E1, s1, A2, E2, s2):
        n12 = _inner(A1, E1, s1, A2, E2, s2, sa, se, nw)
        n11 = _inner(A1, E1, s1, A1, E1, s1, sa, se, nw)
        n22 = _inner(A2, E2, s2, A2, E2, s2, sa, se, nw)
        if not np.isfinite(n12) or not np.isfinite(n11) or not np.isfinite(n22):
            return np.nan
        return float(np.abs(n12) / np.sqrt(max(n11.real * n22.real, 1e-300)))

    if len(mod):
        print("\nOVERLAP  model (rows) x injection (cols), phase-maximised, "
              "noise-weighted")
        print("     " + " ".join(f"I{j:<5}" for j in range(len(inj))))
        O = np.zeros((len(mod), len(inj)))
        for i in range(len(mod)):
            for j in range(len(inj)):
                O[i, j] = ov(Am[i], Em[i], sm[i], Ai[j], Ei[j], si[j])
            print(f"M{i:<3} " + " ".join(f"{O[i, j]:.3f} " for j in range(len(inj))))

        print("\nPER INJECTION: best model leaf, and the lnL cost of the gap")
        print(f"{'inj':>5} {'SNR':>7} {'best':>6} {'overlap':>8} "
              f"{'d f0 [bins]':>12} {'dlnL vs perfect':>16}  verdict")
        for j in range(len(inj)):
            i = int(np.argmax(O[:, j])) if len(mod) else -1
            best = O[i, j] if i >= 0 else 0.0
            # -0.5 |h_inj - h_mod|^2 in the matched-template approximation
            dl = -0.5 * (rho_i[j] ** 2 + rho_m[i] ** 2
                         - 2 * rho_i[j] * rho_m[i] * best) if i >= 0 else \
                -0.5 * rho_i[j] ** 2
            dfb = (mod[i, 1] - inj[j, 1]) / df if i >= 0 else np.nan
            if best >= 0.8 and abs(dfb) <= 2:
                v = "RECOVERED (passes the page's test)"
            elif best >= 0.8:
                v = f"ON IT at overlap {best:.2f} but {abs(dfb):.1f} bins out -> page calls it a MISS"
            elif best >= 0.5:
                v = "partially absorbed"
            else:
                v = "NOT in the model"
            print(f"I{j:<4} {rho_i[j]:7.2f} M{i:<5} {best:8.3f} {dfb:12.2f} "
                  f"{dl:16.1f}  {v}")

        if len(mod) > 1:
            print("\nOVERLAP  model x model (are two leaves splitting one source?)")
            print("     " + " ".join(f"M{j:<5}" for j in range(len(mod))))
            for i in range(len(mod)):
                r = []
                for j in range(len(mod)):
                    r.append("  --  " if i == j else
                             f"{ov(Am[i], Em[i], sm[i], Am[j], Em[j], sm[j]):.3f} ")
                print(f"M{i:<3} " + " ".join(r))

    # ---- occupancy / cap pressure in the window ---------------------------
    print("\nCAP PRESSURE in the window")
    c_lo = int(np.searchsorted(ce, flo, side="right") - 1)
    c_hi = int(np.searchsorted(ce, fhi, side="right") - 1)
    for c in range(max(c_lo, 0), min(c_hi + 1, len(cap))):
        occ = []
        for w in range(nwk):
            m = alive[0, w]
            fv = coords[0, w][m][:, 1] * 1e-3
            occ.append(int(((fv >= ce[c]) & (fv < ce[c + 1])).sum()))
        ninj = int(((inj[:, 1] >= ce[c]) & (inj[:, 1] < ce[c + 1])).sum())
        star = "  <-- target" if c == ic else ""
        print(f"  cell {c}: [{ce[c] * 1e3:.5f},{ce[c + 1] * 1e3:.5f}] mHz "
              f"cap {int(cap[c]):3d}  cold occupancy {occ}  "
              f"catalogue rows here {ninj}"
              + ("  AT CAP" if max(occ) >= cap[c] else "") + star)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
