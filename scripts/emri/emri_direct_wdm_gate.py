"""EMRIDirectWDM gate (Task A8): direct-to-WDM vs the TOF's own dense WDM and vs production.

Grid Nf=180, dt=20 s (layer_dt 3600 s), Nt=1024 (42.7 d) starting START_OFFSET_S after the
reference epoch; ICRS orbits from the local mojito L1 file (cropped); a synthetic plunging
source (p0=7.0 plunges ~34 d after REF). References, all on the same WDM grid:

* ``tof``: the SAME EMRITDIonFly response evaluated densely and TD->WDM (isolates the
  lookup + plunge chunk from the response);
* ``prod``: the production legacy response (get_emri_response_wrapper) -> WDM.

Reports per channel 1 - Re(O) (flat, over active pixels) for direct-vs-tof,
direct-vs-prod and tof-vs-prod, the same per region (pixel time / plunge time
< 0.9, 0.9-0.99, >= 0.99), EMRIDirectWDM stats, wall times.
Controls: --no-handoff (lookup everywhere; the plunge must fail).
"""
import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_tof_xyz_threeway as W  # noqa: E402  (orbits/params loading, REF)

NF, DT, NT = 180, 20.0, 1024
START_OFFSET_S = 5e4


def mm(a, b):
    num = float(np.sum(a * b))
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return 1.0 - num / den if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--modes", default="2,2,0,0", help="';'-separated l,m,k,n or 'all'")
    ap.add_argument("--thresh", type=float, default=1e-5)
    ap.add_argument("--p0", type=float, default=7.0)
    ap.add_argument("--no-handoff", action="store_true")
    args = ap.parse_args()

    from few.waveform import FastKerrEccentricEquatorialFlux  # noqa: F401

    from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri import EMRITDIonFly
    from lisatools.sources.emri import wdm_direct as WD

    W.N_WIN = NF * NT * int(DT / DT)   # not used for data here
    _, _, _, orb = W.load(1)            # ICRS orbits of the local L1 file (params unused)
    REF = W.REF
    data_t0 = REF + START_OFFSET_S
    span = NT * NF * DT
    # synthetic plunging source in the SPECIAL recipe basis (ecliptic-polar sky, raw spin)
    params = [1e6, 1e2, 0.9, args.p0, 0.4, 1.0, 1.0, 1.1, 2.3, 0.7, 4.0, 0.2, 0.0, 0.7]

    wdm = WDMSettings(Nf=NF, Nt=NT, dt=DT, force_backend="cpu")
    table = WDMLookupTable.from_file(args.table, force_backend="cpu")
    tdi = TDIConfig("2nd generation", force_backend="cpu")
    wg, offset_int, gen = None, None, None

    # production legacy response (one FEW construction; reused below)
    from lisatools.sources.emri.response import get_emri_response_wrapper

    off = data_t0 - REF
    offset_int = int(round(off / DT))
    from lisatools.globalfit.stock import erebor

    fit = erebor.get_stock("all_sources")        # the production knobs (see threeway.legacy_wrapper)
    tdi = TDIConfig(fit.general.tdi_gen_str, force_backend="cpu")
    wg = get_emri_response_wrapper(Tobs=(NF * NT + offset_int) * DT + 4e4, dt=DT, t_start=REF,
                                   t0_shift_to_data=off - offset_int * DT, tdi_config=tdi,
                                   tdi_chan=fit.general.tdi_chan, order=fit.emri.response_order,
                                   force_backend="cpu", orbits=orb)
    gen = wg.waveform_gen.waveform_generator
    if args.modes == "all":
        modes = None
    else:
        modes = [tuple(int(x) for x in m.split(",")) for m in args.modes.split(";")]
    kw = dict(mode_selection_threshold=args.thresh)
    if modes is not None:
        kw["mode_selection"] = modes

    if args.no_handoff:
        WD.WDM_HALF_SUPPORT_LAYERS = 0.0          # curvature never trips
    fdot_save = None
    direct = WD.EMRIDirectWDM(gen, table, wdm, orbits=orb, tdi_config=tdi, t_start=REF, data_t0=data_t0,
                              mode_batch=16, response=os.environ.get("GATE_RESPONSE", "spline"))
    if args.no_handoff:
        fdot_save, direct.fdot_axis_max = direct.fdot_axis_max, np.inf
    t0 = time.perf_counter()
    h_dir = np.asarray(direct(*params, **kw).arr)
    t_dir = time.perf_counter() - t0
    print(f"EMRIDirectWDM: {direct.last_stats}, {t_dir:.1f} s", flush=True)
    modes_used = modes if modes is not None else direct._mode_list(params, {"mode_selection_threshold": args.thresh})[0]

    # tof reference: same response, dense TD -> WDM (modes in batches)
    t0 = time.perf_counter()
    tg = data_t0 + np.arange(NF * NT) * DT
    td = np.zeros((3, tg.size))
    for j in range(0, len(modes_used), 16):
        tf = getattr(direct, "last_t_fine", None)          # the SAME fine grid as the direct template
        fly = EMRITDIonFly(gen, orb, tdi, DT, off + span + 2000.0, REF, frame="icrs_special",
                           n_fine=None if tf is not None else direct.n_fine,
                           t_fine_window=(data_t0, data_t0 + span), t_fine=tf)
        out = fly(*params, mode_selection=modes_used[j:j + 16])
        x = np.asarray(out.x)
        ins = np.flatnonzero((tg > x[:, 0].max()) & (tg < x[:, -1].min()))
        for i in range(0, ins.size, 4096):
            sl = ins[i:i + 4096]
            td[:, sl] += np.real(np.sum(np.asarray(out.eval_tdi(tg[sl])), axis=0))
        t_end_traj = REF + float(np.asarray(fly.last_holder.t_arr)[-1])
        del out, fly
        gc.collect()
    t_tof = time.perf_counter() - t0
    tds = TDSettings(tg.size, DT, force_backend="cpu")
    h_tof = np.asarray(TDSignal(td, tds).transform(wdm).arr)

    # production reference
    t0 = time.perf_counter()
    prod_kw = dict(mode_selection_threshold=args.thresh)
    if modes is not None:
        prod_kw["mode_selection"] = modes
    elif os.environ.get("GATE_PROD_SAME_MODES"):
        # production selects modes over ITS span (from the reference epoch), the direct
        # template over the window: give production the direct template's modes
        prod_kw["mode_selection"] = [tuple(m) for m in modes_used]
    hp = np.atleast_2d(np.asarray(wg(*params, **prod_kw)))[:3, offset_int:offset_int + tg.size]
    t_prod = time.perf_counter() - t0
    h_prod = np.asarray(TDSignal(hp, tds).transform(wdm).arr)

    # regions by pixel time relative to the plunge
    n = np.arange(NT)
    frac = (data_t0 + n * wdm.layer_dt - REF) / (t_end_traj - REF)
    # the references switch on abruptly at the window start: their WDM edge contamination
    # (>1e-5 within ~24 px) is excluded, as production's edge_crop (>= 20 wavelets) does
    edge = int(os.environ.get("GATE_EDGE", "32"))
    act = (n >= edge) & (n < NT - edge)
    print(f"plunge at {(t_end_traj - data_t0) / 86400:.2f} d into the window; modes={len(modes_used)}; "
          f"wall direct {t_dir:.1f}s tof {t_tof:.1f}s prod {t_prod:.1f}s", flush=True)
    regions = [("all", act), ("<0.90", act & (frac < 0.9)), ("0.90-0.99", act & (frac >= 0.9) & (frac < 0.99)),
               (">=0.99", act & (frac >= 0.99) & (frac <= 1.02))]
    print(f"{'pair':>12} {'ch':>2} " + " ".join(f"{r[0]:>11}" for r in regions))
    if os.environ.get("GATE_LOCALIZE"):
        c = 0
        sel = regions[1][1]
        a, b = h_dir[c][:, sel], h_tof[c][:, sel]
        res = (a - b) ** 2
        mcar = np.argmax(np.abs(b), axis=0)                      # carrier layer per column (truth)
        tot = np.sum(b ** 2)
        print("  residual energy / signal energy in region <0.90 (X), by layer offset from the carrier:")
        for d in range(-4, 5):
            rows = mcar + d
            ok = (rows >= 0) & (rows < b.shape[0])
            e = np.sum(res[rows[ok], np.flatnonzero(ok)])
            sig = np.sum(b[rows[ok], np.flatnonzero(ok)] ** 2)
            print(f"    d={d:+d}: residual {e / tot:.2e}  (signal in that layer {sig / tot:.2e})")
        cols = np.flatnonzero(sel)
        colres = res.sum(axis=0) / np.maximum((b ** 2).sum(axis=0), 1e-300)
        print("  per-column residual/signal at deciles:", " ".join(f"{np.quantile(colres, q):.1e}" for q in (0.1, 0.5, 0.9, 0.99)))
        worst = np.argsort(colres)[-5:]
        print("  worst columns n:", cols[worst], "values", np.round(colres[worst], 4))
    for tag, a, b in (("direct-tof", h_dir, h_tof), ("direct-prod", h_dir, h_prod), ("tof-prod", h_tof, h_prod)):
        for c, ch in enumerate("XYZ"):
            vals = [mm(a[c][:, sel], b[c][:, sel]) if np.any(sel) else np.nan for _, sel in regions]
            amps = [np.linalg.norm(a[c][:, sel]) / np.linalg.norm(b[c][:, sel]) if np.any(sel) else np.nan
                    for _, sel in regions]
            print(f"{tag:>12} {ch:>2} " + " ".join(f"{v:11.3e}" for v in vals)
                  + "   amp " + " ".join(f"{r:.6f}" for r in amps), flush=True)


if __name__ == "__main__":
    main()
