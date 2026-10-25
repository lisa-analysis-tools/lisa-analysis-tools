"""EMRIDirectWDM gate (Task A8): direct-to-WDM vs the TOF's own dense WDM and vs production.

Grid Nf = 3600 s / dt (``GATE_DT``, default 20 s; a plunge inside the window needs 5 s, the
NF720_DT5 table: 20 s aliases the high harmonics near plunge), layer_dt 3600 s, Nt=1024 (42.7 d)
starting START_OFFSET_S after the reference epoch; ICRS orbits from the local mojito L1 file
(cropped); a synthetic plunging source (p0=7.0 plunges ~34 d after REF). References, all on the
same WDM grid:

* ``tof``: the SAME EMRITDIonFly response evaluated densely and TD->WDM (isolates the
  lookup + plunge chunk from the response);
* ``prod``: the production legacy response (get_emri_response_wrapper) -> WDM.

Reports per channel 1 - Re(O) (flat, over active pixels) for direct-vs-tof,
direct-vs-prod and tof-vs-prod, the same per region (pixel time / plunge time
< 0.9, 0.9-0.99, >= 0.99), EMRIDirectWDM stats, wall times.
Controls: --no-handoff (lookup everywhere; the plunge must fail).

Plunge-chunk length (``--nt-sub``, a comma list of time layers per chunk, default 128): the direct
template is built once per value (``EMRIDirectWDM(Nt_sub=v)``; FEW, the table and the production
reference load once) and the region table is printed per value with an ``nt_sub`` column, plus
``direct-d<first>`` (vs the first value's direct template), the template's 0.25-1 mHz SNR (the
plunge-leak metric of docs/emri-direct-wdm.md "In-window plunge") under scirdv1 XYZ + the fitted
tanh galactic foreground at this window's Tobs with the same GATE_EDGE pixels cropped (the TOF
and production references' band SNRs alongside, and the band SNR of the residuals direct - tof
and direct - first value: the method's own leak), where the direct-tof residual sits relative
to the trajectory's last pixel, the chunk split (lookup / chunk pixels, windows, tail) and a
closing summary (worst channel per region). The TOF reference follows each template's own fine
grid (recomputed only when the grid changes). The fine grid's 80 s segment starts ``Nt_sub``
layers before the chunk start, so its points fall differently relative to the trajectory's end
for different values, and the SPLINE response at the stop (read by both the TOF reference and
the spline path's chunk) changes with it; ``--fixed-fine-grid`` builds every value on the first
value's grid (list the largest value first) so that only the chunk length and the lookup's
reach differ. Production is not affected: its chunk is the exact dense kernel and its sparse
grid ignores the chunk start.

``GATE_RESPONSE`` picks the direct template's response: ``spline`` (default; the TOF reference
is the same response, so direct-vs-tof is the lookup + chunk alone) or ``dense`` (production;
needs a backend module built with TDDenseTDIWaveformWrap; its plunge tail carries the 120 s
source taper the spline TOF does not, so read direct-vs-prod there). The table is found or
built at ``--table`` (``wdm_lookup_store.ensure_lookup_table``; a table on another grid
raises).

    GATE_DT=5 python scripts/emri/emri_direct_wdm_gate.py \\
        --table wdm_lookup_emri_cx_NF720_DT5_TL32_fd8x0p01_nld2.h5 --modes 2,2,0,0 --nt-sub 128,96,64
"""
import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_batch_speed as B  # noqa: E402  (RunBox: the scirdv1 + foreground SNR box)
import emri_tof_xyz_threeway as W  # noqa: E402  (orbits/params loading, REF)

DT = float(os.environ.get("GATE_DT", "20"))   # 20 s aliases the high harmonics near plunge: use 5 s for those
NF, NT = int(round(3600.0 / DT)), 1024
START_OFFSET_S = 5e4
LEAK_BAND = (2.5e-4, 1e-3)                    # Hz: the plunge-leak band of docs/emri-direct-wdm.md


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
    ap.add_argument("--nt-sub", default="128", help="comma list of plunge-chunk lengths in layers")
    ap.add_argument("--fixed-fine-grid", action="store_true",
                    help="build every --nt-sub value on the first value's fine trajectory grid (list the "
                         "largest first): the spline response, which the spline path's chunk reads across "
                         "the stop, is then the same for every value")
    args = ap.parse_args()
    nt_subs = [int(v) for v in args.nt_sub.split(",") if v.strip()]

    from few.waveform import FastKerrEccentricEquatorialFlux  # noqa: F401

    from lisatools import wdm_lookup_store
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
    status = wdm_lookup_store.ensure_lookup_table(args.table, Nf=NF, dt=DT)   # checks the grid
    print(f"lookup table {args.table}: {status} (Nf={NF}, dt={DT:g} s)", flush=True)
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
    tg = data_t0 + np.arange(NF * NT) * DT
    tds = TDSettings(tg.size, DT, force_backend="cpu")
    # the references switch on abruptly at the window start: their WDM edge contamination
    # (>1e-5 within ~24 px) is excluded, as production's edge_crop (>= 20 wavelets) does
    edge = int(os.environ.get("GATE_EDGE", "32"))
    n = np.arange(NT)
    act = (n >= edge) & (n < NT - edge)
    # the plunge-leak band SNR: scirdv1 XYZ + the fitted tanh foreground at this window's Tobs
    leak_box = B.RunBox(NF, NT, DT, "cpu", edge=edge, min_freq=LEAK_BAND[0], max_freq=LEAK_BAND[1])
    band = f"{LEAK_BAND[0] * 1e3:g}-{LEAK_BAND[1] * 1e3:g} mHz"

    def tof_reference(direct, modes_used):
        """The same response, dense TD -> WDM (modes in batches) on the direct template's own fine
        grid; returns (h_tof, t_end_traj [absolute s])."""
        td = np.zeros((3, tg.size))
        t_end_traj = None
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
        return np.asarray(TDSignal(td, tds).transform(wdm).arr), t_end_traj

    tof_cache = {}                 # fine-grid bytes -> (h_tof, t_end_traj, wall)
    h_prod = t_prod = modes_used = h_first = None
    d0_ref = None                  # --fixed-fine-grid: the first value's dense-segment start (FEW clock)
    summary = []
    for v in nt_subs:
        direct = WD.EMRIDirectWDM(gen, table, wdm, orbits=orb, tdi_config=tdi, t_start=REF, data_t0=data_t0,
                                  mode_batch=16, response=os.environ.get("GATE_RESPONSE", "spline"), Nt_sub=v)
        if args.no_handoff:
            direct.fdot_axis_max = np.inf
        if args.fixed_fine_grid and d0_ref is not None:
            # _fine_grid(cs) starts its fine_dt_plunge segment at cs - Nt_sub layer_dt: hand it the
            # chunk start that puts that segment where the first value's began (or earlier, never
            # later than this value's own: its chunk must stay on the dense segment). Where the
            # 80 s points fall relative to the trajectory's end changes the spline response at the
            # stop (both the TOF reference and the spline path's chunk read it); production's
            # chunk is the exact dense kernel and its sparse grid ignores the chunk start.
            def _fixed(cs, _orig=direct._fine_grid, _v=v):
                return _orig(None if cs is None else min(cs, d0_ref + _v * wdm.layer_dt))
            direct._fine_grid = _fixed
        t0 = time.perf_counter()
        h_dir = np.asarray(direct(*params, **kw).arr)
        t_dir = time.perf_counter() - t0
        st = dict(direct.last_stats)
        print(f"\nnt_sub={v}: EMRIDirectWDM: {st}, {t_dir:.1f} s", flush=True)
        if args.fixed_fine_grid and d0_ref is None and st.get("chunk_start") is not None:
            d0_ref = float(st["chunk_start"]) - v * wdm.layer_dt
        if modes_used is None:
            modes_used = modes if modes is not None else direct._mode_list(
                params, {"mode_selection_threshold": args.thresh})[0]

        tf = getattr(direct, "last_t_fine", None)
        key = None if tf is None else np.asarray(tf, dtype=float).tobytes()
        if key not in tof_cache:
            t0 = time.perf_counter()
            h_tof, t_end_traj = tof_reference(direct, modes_used)
            tof_cache[key] = (h_tof, t_end_traj, time.perf_counter() - t0)
        h_tof, t_end_traj, t_tof = tof_cache[key]

        if h_prod is None:                       # production reference (independent of nt_sub)
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
            del hp
        if h_first is None:
            h_first, v_first = h_dir, v

        # regions by pixel time relative to the plunge
        frac = (data_t0 + n * wdm.layer_dt - REF) / (t_end_traj - REF)
        print(f"plunge at {(t_end_traj - data_t0) / 86400:.2f} d into the window; modes={len(modes_used)}; "
              f"wall direct {t_dir:.1f}s tof {t_tof:.1f}s prod {t_prod:.1f}s", flush=True)
        regions = [("all", act), ("<0.90", act & (frac < 0.9)), ("0.90-0.99", act & (frac >= 0.9) & (frac < 0.99)),
                   (">=0.99", act & (frac >= 0.99) & (frac <= 1.02))]
        print(f"{'nt_sub':>6} {'pair':>12} {'ch':>2} " + " ".join(f"{r[0]:>11}" for r in regions))
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
        pairs = [("direct-tof", h_dir, h_tof), ("direct-prod", h_dir, h_prod), ("tof-prod", h_tof, h_prod)]
        if v != v_first:
            pairs.append((f"direct-d{v_first}", h_dir, h_first))
        worst = {}
        for tag, a, b in pairs:
            worst[tag] = []
            for c, ch in enumerate("XYZ"):
                vals = [mm(a[c][:, sel], b[c][:, sel]) if np.any(sel) else np.nan for _, sel in regions]
                amps = [np.linalg.norm(a[c][:, sel]) / np.linalg.norm(b[c][:, sel]) if np.any(sel) else np.nan
                        for _, sel in regions]
                worst[tag].append(vals)
                print(f"{v:>6} {tag:>12} {ch:>2} " + " ".join(f"{x:11.3e}" for x in vals)
                      + "   amp " + " ".join(f"{r:.6f}" for r in amps), flush=True)
            worst[tag] = np.nanmax(np.asarray(worst[tag]), axis=0)          # worst channel per region
        snr_dir = leak_box.snr(h_dir)
        snr_tof, snr_prod = leak_box.snr(h_tof), leak_box.snr(h_prod)
        res_tof = leak_box.snr(h_dir - h_tof)                 # the lookup + chunk's own band error
        res_first = leak_box.snr(h_dir - h_first)
        n_pad = v // 4
        tail = st.get("tail_layers", 0)
        print(f"{v:>6} {band} SNR ({leak_box.describe()}): direct {snr_dir:.3e}  tof {snr_tof:.3e}  "
              f"prod {snr_prod:.3e}; of the residual direct-tof {res_tof:.3e}, direct-d{v_first} "
              f"{res_first:.3e}", flush=True)
        # where the direct-tof residual sits: share of its (flat, all-channel, active-pixel) energy
        # by pixel offset from the trajectory's last pixel
        n_stop = int(np.floor((t_end_traj - data_t0) / wdm.layer_dt))
        e_col = np.sum((h_dir - h_tof) ** 2, axis=(0, 1)) * act
        bins = [(-NT, -64), (-64, -32), (-32, -16), (-16, 0), (0, 16), (16, 32), (32, NT)]
        shares = [np.sum(e_col[max(0, n_stop + a):max(0, n_stop + b)]) / max(np.sum(e_col), 1e-300)
                  for a, b in bins]
        print(f"{v:>6} direct-tof residual energy by pixel offset from the last pixel ({n_stop}): "
              + " ".join(f"[{a if a > -NT else '':>3},{b if b < NT else '':>3}) {x:.2f}"
                         for (a, b), x in zip(bins, shares)), flush=True)
        print(f"{v:>6} chunk: n_pad {n_pad} px, kept interior {v - 2 * n_pad} px; chunk start "
              f"{st.get('chunk_start')}; lookup px {st.get('lookup_pixels')} chunk px {st.get('chunk_pixels')} "
              f"dropped px {st.get('dropped_pixels')}; {st.get('windows')} windows, tail {tail} px", flush=True)
        summary.append((v, worst, snr_dir, snr_tof, snr_prod, v - 2 * n_pad, st, t_dir, res_tof, res_first))
        del direct
        gc.collect()

    print(f"\n[nt_sub summary] worst channel per region (1 - Re(O), {edge} px edges cropped); {band} SNR; "
          f"modes={len(modes_used)}; response={os.environ.get('GATE_RESPONSE', 'spline')}; dt={DT:g} s", flush=True)
    names = [r[0] for r in regions][1:]
    print(f"{'nt_sub':>6} {'pair':>12} " + " ".join(f"{r:>10}" for r in names)
          + f" {'band SNR':>10} {'res SNR':>10} {'res d1st':>10} {'kept px':>7} {'chunk px':>8} {'windows':>7}"
          f" {'wall s':>7}")
    for v, worst, snr_dir, snr_tof, snr_prod, kept, st, t_dir, res_tof, res_first in summary:
        for tag in ("direct-tof", "direct-prod"):
            print(f"{v:>6} {tag:>12} " + " ".join(f"{x:10.2e}" for x in worst[tag][1:])
                  + f" {snr_dir:10.3e} {res_tof:10.3e} {res_first:10.3e} {kept:7d} {st.get('chunk_pixels', 0):8d} "
                  f"{st.get('windows', 0):7d} {t_dir:7.1f}", flush=True)
    print(f"  band SNR = the direct template's {band} SNR, res SNR = that of direct - tof, res d1st = that of "
          f"direct - the nt_sub={summary[0][0]} template (the chunk length's own leak on either response); references: "
          f"prod {summary[0][4]:.3e}, tof (each template's own fine grid) "
          + ", ".join(f"{row[0]}: {row[3]:.3e}" for row in summary), flush=True)


if __name__ == "__main__":
    main()
