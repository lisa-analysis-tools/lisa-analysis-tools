#!/usr/bin/env python
"""SOBBH lookup scorer vs the mojito L1 SOBHB bricks: match, SNR, logL and scoring time per duration.

The SOBBH member of the aligned EMRI / MBH / SOBBH 1-GPU test setup (``_sobbh_testbox.py``):
for every SOBHB source with an L1 brick (``find_sobhb_bricks``) and every duration in ``--days``
(Nt = days * 24 one-hour layers, Nf 1440 at dt 2.5 s, the window starting ``START_OFFSET_S``
after the brick start):

* the data: the source's own brick (noiseless, that source only; ``MojitoL1File.tdis.xyz_doppler``),
  WDM-transformed on the run box (0.25-25 mHz, EDGE_CROP_WAVELETS = 60 layers cropped at each
  end; the box end is pulled in to the brick's orbit tables when the window outlasts them);
* the template parameters: the catalogue row through the STOCK mapping
  (``sobbh_catalogue_to_waveform_basis`` -> ``SOBBHChunkedLikeMove.to_chunked_basis``), the
  brick's ``L1Orbits`` (ICRS), ``MOJITO_REFERENCE_TIME``, 2nd-generation XYZ; a window the
  brick's orbit tables (REF + 0.01 .. 730.5 d) or data do not hold is refused with the numbers;
* the noise for every SNR / score: scirdv1 + the fitted tanh galactic foreground at the window's
  Tobs (``--foreground off`` = instrument only);
* the LOOKUP comp (``SOBBHLookupComputations``; the fused kernel by default) scores the row:
  ``snr = sqrt(<h|h>)``, ``data_snr = sqrt(<d|d>)``, ``mm = 1 - <d|h>/sqrt(<d|d><h|h>)``
  (noise-weighted, no maximisation), ``logL = -1/2 <d - h|d - h>``,
  ``snr_ratio = sqrt(<h|h>/<d|d>)``, the warm ``get_ll`` time;
* the PRODUCTION template (the stock dense TD TDI-on-the-fly wave wrap) the same way, and
  ``mm_vs_production`` (lookup vs production template), so a mismatch is attributed to the model
  (production vs mojito) or to the lookup (lookup vs production).

Run from the repo root::

    python scripts/sobbh/sobbh_lookup_mojito.py --days 180,360,720 --out mojito.jsonl

JSON (one line per source and duration; the field names shared with the EMRI / MBH setups):
``source, src, brick, days, nt, nf, dt, tobs_s, box_days, edge, start_offset, foreground, noise,
backend, tag, data_snr, snr_catalogue, lookup {path, snr, data {mm, logL, snr_ratio, snr, snr_det},
t_get_ll_ms}, production {snr, data {...}, t_s}, mm_vs_production, stats``; ``data.snr`` is the
optimal SNR ``sqrt(<h|h>)``, ``data.snr_det`` the detected SNR ``<d|h>/sqrt(<h|h>)``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sobbh_testbox as tb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1-dir", default=None, help="directory of SOBHB L1 bricks (searched first)")
    ap.add_argument("--catalog", default=None, help="SOBHB catalogue (default: next to the bricks)")
    ap.add_argument("--sources", default="all", help="comma list of SOBHB ids, or 'all' found")
    ap.add_argument("--days", default="180,360,720", help="comma list of window lengths [days]")
    ap.add_argument("--backend", default=None, help="cpu / cuda13x / ... (default: first CUDA)")
    ap.add_argument("--nf", type=int, default=1440)
    ap.add_argument("--dt", type=float, default=2.5, help="grid step [s]; the bricks are 2.5 s")
    ap.add_argument("--edge", type=int, default=tb.EDGE_CROP_WAVELETS)
    ap.add_argument("--start-offset", type=float, default=tb.START_OFFSET_S)
    ap.add_argument("--foreground", default="on", choices=("on", "off"))
    ap.add_argument("--table", default=os.environ.get("SOBBH_LOOKUP_TABLE_PATH", ""))
    ap.add_argument("--kernel", default="auto", choices=("auto", "kernel", "python"))
    ap.add_argument("--eval-dt", type=float, default=43200.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-production", action="store_true", help="skip the dense TD template")
    ap.add_argument("--out", default="sobbh_lookup_mojito.jsonl")
    args = ap.parse_args()

    import lisatools

    backend = args.backend or ("cuda" if lisatools.has_backend("cuda") else "cpu")
    if not lisatools.has_backend(backend):
        raise SystemExit(f"backend {backend!r} unavailable on this host")
    bricks = tb.find_sobhb_bricks(args.l1_dir)
    if not bricks:
        print(
            "[mojito] no SOBHB L1 bricks found (--l1-dir, MOJITO_LIGHT_PATH, MOJITO_DATA_PATH, "
            "MOJITO_INFO_PATH): nothing to compare"
        )
        return 0
    ids = sorted(bricks) if args.sources == "all" else [int(s) for s in args.sources.split(",")]
    missing = [i for i in ids if i not in bricks]
    if missing:
        print(f"[mojito] no L1 brick for SOBHB ids {missing}; skipping them")
    ids = [i for i in ids if i in bricks]
    if not (args.table and os.path.exists(args.table)):
        raise SystemExit("--table / SOBBH_LOOKUP_TABLE_PATH: an n_ref table with 3600-s layers")
    days_list = sorted(int(d) for d in args.days.split(","))

    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import L1Orbits
    from lisatools.diagnostic import inner_product
    from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSignal
    from lisatools.globalfit.moves.sobbhspecialmove import SOBBHChunkedLikeMove
    from lisatools.globalfit.recipe import MOJITO_REFERENCE_TIME
    from lisatools.globalfit.stock.erebor.injections import sobbh_catalogue_to_waveform_basis
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations
    from lisatools.utils.utility import asnumpy

    tdi = TDIConfig("2nd generation", force_backend=backend)
    table = WDMLookupTable.from_file(args.table, force_backend=backend)
    xp = lisatools.get_backend(backend).xp
    nf, dt = int(args.nf), float(args.dt)
    layer_dt = nf * dt
    tag = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(
        f"[mojito] SOBHB bricks for ids {ids}  days {days_list}  backend {backend}  grid Nf={nf} "
        f"dt={dt}  box [{tb.MIN_FREQ:g}, {tb.MAX_FREQ:g}] Hz, edge {args.edge} layers  "
        f"foreground {args.foreground}  table {os.path.basename(args.table)}"
    )
    summary = []
    for src in ids:
        brick = bricks[src]
        row = tb.catalogue_row(tb.find_catalogue(brick, args.catalog), src)
        wb = sobbh_catalogue_to_waveform_basis(row)
        p = SOBBHChunkedLikeMove.to_chunked_basis(wb)[0]
        rows = p[None, :]
        orbits = L1Orbits(brick, force_backend=backend, frame="icrs")
        # the response's coverage: the CONFIGURED sc / ltt tables (not orbits.t_base)
        sc_t0, sc_dt, sc_N, ltt_t0, ltt_dt, ltt_N = (
            float(v) for v in tuple(orbits.pycppdetector_args)[:6]
        )
        o_lo = max(sc_t0, ltt_t0)
        o_hi = min(sc_t0 + (sc_N - 1) * sc_dt, ltt_t0 + (ltt_N - 1) * ltt_dt)
        print(
            f"[mojito] src {src}: m1 {p[0]:.1f} m2 {p[1]:.1f} f_low {1e3 * p[5]:.3f} mHz D "
            f"{p[4] / 1e6:.0f} Mpc catalogue SNR {row.get('EstimatedSNR', float('nan')):.3g}  "
            f"brick {os.path.basename(brick)}"
        )
        for days in days_list:
            nt = days * 24
            nobs = nf * nt
            t_a = time.perf_counter()
            try:
                data, t0, _ = tb.load_l1_window(brick, nobs, dt, start_offset=args.start_offset)
            except ValueError as exc:
                print(f"[mojito] src {src} {days} d: {exc}; skipping")
                continue
            t_read = time.perf_counter() - t_a
            # the window must sit inside the orbit tables, with the production response's
            # 4e4-s read-ahead past its end (the EMRI check_window rule)
            t_end = t0 + nobs * dt
            if t0 < o_lo or t_end + 4.0e4 > o_hi:
                print(
                    f"[mojito] src {src} {days} d: window [{t0:.0f}, {t_end:.0f}] s (+4e4 s) does "
                    f"not fit the brick's orbit tables [{o_lo:.0f}, {o_hi:.0f}] s "
                    f"({(o_hi - t0) / 86400.0:.1f} d from the window start); skipping"
                )
                continue
            # a source chirping out of the box (or merging) inside the window: stop the box
            # BAND_EXIT_MARGIN_LAYERS layers before it leaves (power above the band is not scored)
            from lisatools.sources.sobbh.wdm_direct import SOBBHBatchedTOF

            t_exit = tb.band_exit_time(
                SOBBHBatchedTOF(orbits, tdi, MOJITO_REFERENCE_TIME, force_backend=backend),
                p,
                t0,
                t_end,
                layer_dt,
                tb.MAX_FREQ - 2.0 / (2.0 * layer_dt),
            )
            max_time = None
            if t_exit is not None:
                max_time = t_exit - t0 - tb.BAND_EXIT_MARGIN_LAYERS * layer_dt
            try:
                wdm = tb.run_box(
                    nf, nt, dt, t0, edge=args.edge, force_backend=backend, max_time=max_time
                )
            except ValueError as exc:
                print(f"[mojito] src {src} {days} d: {exc}; skipping")
                continue
            if max_time is not None and max_time < (nt - args.edge) * layer_dt:
                print(
                    f"[mojito] src {src} {days} d: the source leaves the {tb.MAX_FREQ * 1e3:g}-mHz "
                    f"box (or merges) {(t_exit - t0) / 86400.0:.0f} d into the window; box stops "
                    f"{tb.BAND_EXIT_MARGIN_LAYERS} layers before"
                )
            box_days = (float(wdm.ind_max_t) - float(wdm.ind_min_t) + 1) * layer_dt / 86400.0
            tobs = nobs * dt
            sens, noise_label = tb.noise(wdm, tobs, args.foreground)
            td_set = TDSettings(nobs, dt, t0=t0, force_backend=backend)
            d_sig = TDSignal(xp.asarray(data), td_set).transform(wdm)
            ac = AnalysisContainer(d_sig, sens)
            d_d = float(np.real(asnumpy(ac.inner_product())))
            dev = getattr(getattr(d_sig.arr, "device", None), "id", None)
            gpus = None if dev is None else [int(dev)]
            aca = AnalysisContainerArray([ac], gpus=gpus)
            comp = SOBBHLookupComputations(
                wdm,
                MOJITO_REFERENCE_TIME,
                table,
                orbits=orbits,
                tdi_config=tdi,
                tdi_type="XYZ",
                t_obs_start=t0,
                eval_dt=args.eval_dt,
                kernel=args.kernel,
                force_backend=backend,
            )
            idx = np.zeros(1, dtype=np.int32)
            comp.get_ll_wdm(rows, aca, data_index=idx, noise_index=idx)  # warm
            t_a = time.perf_counter()
            for _ in range(max(args.repeats, 1)):
                comp.get_ll_wdm(rows, aca, data_index=idx, noise_index=idx)
            t_ll = (time.perf_counter() - t_a) / max(args.repeats, 1)
            d_h = float(np.asarray(asnumpy(comp.d_h_out), dtype=float)[0])
            h_h = float(np.asarray(asnumpy(comp.h_h_out), dtype=float)[0])

            def scores(dh, hh):
                """the shared per-template data scores against the brick"""
                return dict(
                    mm=float(1.0 - dh / np.sqrt(d_d * hh)),
                    logL=float(-0.5 * (d_d + hh - 2.0 * dh)),
                    snr_ratio=float(np.sqrt(hh / d_d)),
                    snr=float(np.sqrt(hh)),
                    snr_det=float(dh / np.sqrt(hh)),
                )

            rec = dict(
                source="sobbh",
                src=src,
                brick=os.path.basename(brick),
                days=days,
                nt=nt,
                nf=nf,
                dt=dt,
                tobs_s=tobs,
                box_days=box_days,
                edge=args.edge,
                start_offset=args.start_offset,
                foreground=args.foreground,
                noise=noise_label,
                band_exit_days=None if t_exit is None else (t_exit - t0) / 86400.0,
                backend=backend,
                tag=tag,
                data_snr=float(np.sqrt(d_d)),
                snr_catalogue=row.get("EstimatedSNR", float("nan")),
                lookup=dict(
                    path="kernel" if comp.uses_kernel else "python",
                    snr=float(np.sqrt(h_h)),
                    data=scores(d_h, h_h),
                    t_get_ll_ms=1e3 * t_ll,
                ),
                t_read_s=t_read,
                stats=dict(comp.last_stats),
            )
            if not args.no_production:
                from lisatools.globalfit.stock.erebor.wrappers import SOBBHTDIonFlyWaveWrap
                from lisatools.sources.sobbh.response import get_sobbh_tdionfly_gen

                t_a = time.perf_counter()
                gen = get_sobbh_tdionfly_gen(
                    Tobs=tobs,
                    dt=dt,
                    t_start=t0,
                    tdi_config=tdi,
                    reference_time=MOJITO_REFERENCE_TIME,
                    orbits=orbits,
                    force_backend=backend,
                )
                t_arr = xp.asarray(t0 + np.arange(nobs) * dt)
                wrap = SOBBHTDIonFlyWaveWrap(gen, t_arr, td_set, wdm, td_window=None, nchannels=3)
                hp = wrap(*wb)
                t_prod = time.perf_counter() - t_a
                hh_p = float(np.real(asnumpy(inner_product(hp, hp, psd=sens))))
                dh_p = float(np.real(asnumpy(ac.template_inner_product(hp))))
                # the lookup scored against the production template
                acp = AnalysisContainer(WDMSignal(xp.asarray(hp.arr).copy(), wdm), sens)
                comp.get_ll_wdm(
                    rows[:1],
                    AnalysisContainerArray([acp], gpus=gpus),
                    data_index=idx[:1],
                    noise_index=idx[:1],
                )
                dh_lp = float(np.asarray(asnumpy(comp.d_h_out))[0])
                hh_l = float(np.asarray(asnumpy(comp.h_h_out))[0])
                rec.update(
                    production=dict(snr=float(np.sqrt(hh_p)), data=scores(dh_p, hh_p), t_s=t_prod),
                    mm_vs_production=float(1.0 - dh_lp / np.sqrt(hh_p * hh_l)),
                )
            summary.append(rec)
            lk = rec["lookup"]
            line = (
                f"[mojito] src {src} {days:4d} d  data_snr {rec['data_snr']:.3g} (catalogue "
                f"{rec['snr_catalogue']:.3g})  lookup[{lk['path']}] snr {lk['snr']:.3g} mm "
                f"{lk['data']['mm']:.2e} snr_ratio {lk['data']['snr_ratio']:.5f} logL "
                f"{lk['data']['logL']:.3e} get_ll {lk['t_get_ll_ms']:.1f} ms"
            )
            if "production" in rec:
                pr = rec["production"]
                line += (
                    f" | production mm {pr['data']['mm']:.2e} logL {pr['data']['logL']:.3e}"
                    f" | lookup vs production mm {rec['mm_vs_production']:.2e}"
                )
            print(line, flush=True)
            with open(args.out, "a") as fp:
                fp.write(json.dumps(rec) + "\n")
    print(
        f"[mojito] data SNR per source vs duration ({'with' if args.foreground == 'on' else 'without'}"
        " the fitted tanh foreground):"
    )
    print("  src  " + "  ".join(f"{d:6d} d" for d in days_list))
    for src in ids:
        vals = {r["days"]: r["data_snr"] for r in summary if r["src"] == src}
        print(f"  {src:3d}  " + "  ".join(f"{vals.get(d, float('nan')):8.3g}" for d in days_list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
