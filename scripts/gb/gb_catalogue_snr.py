"""Top-SNR sources of the mojito GB (wdwd galaxy) and VGB catalogues + their inclination census.

1. A fast SNR proxy for every catalogue row:
       snr_proxy = A sqrt(Tobs / S_n(f)) sqrt(((1 + c^2) / 2)^2 + c^2),   c = cos(iota)
   with S_n the sky-averaged LISA sensitivity (scirdv1 + fitted tanh foreground at Tobs).
2. The top ``--refine`` rows by proxy are re-scored EXACTLY: chunked-het template on the run box
   (Nf 180 / dt 20, 3600-s layers, ``--days``), full XYZ 3x3 scirdv1 + foreground noise on the
   source's 5-layer slab -> snr.
3. Writes ``--out`` (json): the top ``--top`` rows by exact SNR (catalogue index, ID, params at
   REF, snr, snr_proxy, cos_iota) for both catalogues, and the |cos iota| census of the rows above
   ``--snr-min`` (proxy, rescaled by the refine set's median exact/proxy ratio).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gb_testbox as tb  # noqa: E402

import gbgpu  # noqa: E402,F401


def proxy_snr(cat, tobs):
    import h5py
    from lisatools.sensitivity import LISASens
    from lisatools.stochastic import FittedHyperbolicTangentGalacticForeground as FG

    with h5py.File(cat, "r") as f:
        b = f["Binaries"]
        A = np.asarray(b["Amplitude"][:])
        f0 = np.asarray(b["GW22FrequencySSBFrame"][:])
        c = np.cos(np.asarray(b["InclinationAngle"][:]))
    ok = (f0 >= tb.MIN_FREQ) & (f0 <= tb.MAX_FREQ)
    snr = np.zeros_like(A)
    fs = np.clip(f0[ok], tb.MIN_FREQ, tb.MAX_FREQ)
    # S_n on a log grid, interpolated (15.5M rows)
    grid = np.logspace(np.log10(tb.MIN_FREQ), np.log10(tb.MAX_FREQ), 4000)
    sn = np.asarray(LISASens.get_Sn(grid, model="scirdv1", stochastic_params=(float(tobs),),
                                    stochastic_function=FG))
    S = np.exp(np.interp(np.log(fs), np.log(grid), np.log(sn)))
    snr[ok] = A[ok] * np.sqrt(tobs / S) * np.sqrt(((1 + c[ok] ** 2) / 2) ** 2 + c[ok] ** 2)
    return snr, c, f0


def exact_snr(params, days):
    from lisatools.detector import DefaultOrbits

    nf, nt, dt = tb.grid_args(days, laptop=True)
    t0 = 0.5 * 365.25 * 86400.0 + tb.REF
    wdm = tb.run_box(nf, nt, dt, t0)
    eng = tb.build_engines(wdm, DefaultOrbits(force_backend="cpu", frame="icrs"),
                           names=["chunked"])["chunked"]
    sens, _ = tb.noise(wdm, wdm.Tobs, "on")
    T = int(wdm.ind_max_t - wdm.ind_min_t + 1)
    out = np.zeros(len(params))
    for s in range(0, len(params), 64):
        p = params[s:s + 64]
        lo = tb.slab_lo_for(p[:, 1], wdm)
        invc = tb.slab_invc(sens, wdm, lo)
        h = tb.SlabHolder(np.zeros((len(p), 3, tb.SLAB_W, T)), invc, lo)
        eng.fill_template(h, p, np.arange(len(p)), np.full(len(p), 1024), factor=+1,
                          waveform_kwargs={}, band_slab_Nf=tb.SLAB_W, slab_min_f=lo)
        hs = h.linear_data_arr[0].reshape(len(p), 3, tb.SLAB_W, T)
        out[s:s + len(p)] = np.sqrt(np.einsum("ncwt,ncdwt,ndwt->n", hs, invc, hs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=180.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--refine", type=int, default=300)
    ap.add_argument("--snr-min", type=float, default=7.0)
    ap.add_argument("--l1-dir", default=None)
    ap.add_argument("--catalogue", default=None, help="GB (wdwd) catalogue path")
    ap.add_argument("--vgb-catalogue", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tobs = a.days * 86400.0
    result = dict(days=a.days, snr_min=a.snr_min)
    for kind in ("GB", "VGB"):
        cat = a.catalogue if kind == "GB" else a.vgb_catalogue
        if cat is None:
            brick = tb.find_brick(kind, a.l1_dir)
            if brick is None:
                print(f"[{kind}] no brick / catalogue found -- skipped", flush=True)
                continue
            cat = tb.find_catalogue(kind, brick)
        snr_p, c, f0 = proxy_snr(cat, tobs)
        order = np.argsort(snr_p)[::-1][: max(a.refine, a.top)]
        params, ids, idx = tb.catalogue_params(cat, idx=order)
        # catalogue_params sorts idx; keep the proxy alongside
        snr_e = exact_snr(params, a.days)
        p_of = dict(zip(order.tolist(), snr_p[order].tolist()))
        best = np.argsort(snr_e)[::-1][: a.top]
        ratio = float(np.median(snr_e / np.array([p_of[int(i)] for i in idx])))
        sel = snr_p * ratio >= a.snr_min
        ac = np.abs(c[sel])
        census = {f"|cos i|<{x}": float(np.mean(ac < x)) for x in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)}
        census["n_above_snr_min"] = int(sel.sum())
        result[kind] = dict(
            catalogue=cat, exact_over_proxy_median=ratio, census=census,
            top=[dict(index=int(idx[k]), id=ids[k], snr=float(snr_e[k]),
                      snr_proxy=float(p_of[int(idx[k])]), f0=float(params[k, 1]),
                      cos_iota=float(np.cos(params[k, 5])), params=params[k].tolist())
                 for k in best])
        print(f"\n== {kind}: {sel.sum()} rows above SNR {a.snr_min} (proxy x {ratio:.3f}); "
              f"|cos i| census: " + ", ".join(f"{k} {v:.1%}" for k, v in census.items()
                                              if k.startswith("|")), flush=True)
        for t in result[kind]["top"]:
            print(f"   {t['id']:>12} f0 {1e3 * t['f0']:8.4f} mHz  SNR {t['snr']:8.1f}  "
                  f"cos i {t['cos_iota']:+.3f}", flush=True)
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
