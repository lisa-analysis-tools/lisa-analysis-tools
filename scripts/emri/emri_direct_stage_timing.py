"""Stage timing of the direct-to-WDM EMRI template (GPU or CPU): where do the milliseconds go?

Wraps the stages of one EMRIDirectWDM call with timers (a device synchronise before each
timer stops, so GPU time lands on the stage that launched it) and prints per-template totals
next to the production template's wall time on the same grid.

    python scripts/emri/emri_direct_stage_timing.py --backend cuda13x --dt 2.5 --reps 5 \
        --catalog /path/emri_cat_mojito_lite_processed_MT.hdf5 --orbits equal-arm \
        --direct-table wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5 --thresh 1e-3,1e-5

Sweeps in ONE process (FEW and the table load once): ``--response-grid sparse,pixels`` (the
dense response's grid) and ``--chunk-rows 1,2,4,8,16,32`` (rows per response call, timed over
``--batch-rows`` rows each). Each (threshold, grid, chunk) prints its stage split and, on a GPU,
the cupy memory-pool footprint of that batch; ``--out`` appends one JSON line per run and a
summary table (ms per template vs rows per call) closes each threshold.
"""
import argparse
import collections
import functools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_batch_speed as B  # noqa: E402  (source loader, grid conventions)
import emri_tof_xyz_threeway as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=int, default=1)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--dt", type=float, default=2.5)
    ap.add_argument("--days", type=float, default=180.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--thresh", default="1e-3,1e-5")
    ap.add_argument("--mode-batch", type=int, default=64)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--l1-dir", default=None)
    ap.add_argument("--orbits", choices=("auto", "l1", "equal-arm"), default="auto")
    ap.add_argument("--direct-table", required=True)
    ap.add_argument("--batch-rows", type=int, default=0,
                    help="also time EMRIDirectWDM.batch over this many parameter rows")
    ap.add_argument("--chunk-rows", default="16",
                    help="rows per response call in the batch; a comma list sweeps them (e.g. 1,2,4,8,16,32)")
    ap.add_argument("--response-grid", default="sparse",
                    help="dense response grid(s): sparse, pixels, or a comma list of both")
    ap.add_argument("--out", default=None, help="append one JSON line per (threshold, grid, chunk) here")
    ap.add_argument("--response", choices=("spline", "dense"), default="dense",
                    help="TDI-on-the-fly response: 'dense' (TDDenseTDIonTheFly: exact phases, geometry shared "
                         "across harmonics; needs a backend built with it) or 'spline' (TDTDIonTheFly)")
    args = ap.parse_args()

    from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings
    from lisatools.globalfit.stock import erebor
    from lisatools.response import tdionfly as TF
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri import emritdionfly as EF
    from lisatools.sources.emri import wdm_direct as WD
    from lisatools.sources.emri.response import get_emri_response_wrapper
    import lisatools.wdm_het as WH

    gpu = args.backend != "cpu"
    if gpu:
        import cupy as cp

    def sync():
        if gpu:
            cp.cuda.Device().synchronize()

    T = collections.defaultdict(float)
    C = collections.Counter()
    LABELS = []

    def wrap(owner, name, label):
        f = getattr(owner, name)
        LABELS.append(label)

        @functools.wraps(f)
        def g(*a, **k):
            sync()
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                sync()
                T[label] += time.perf_counter() - t0
                C[label] += 1
        import inspect
        static = isinstance(owner, type) and isinstance(inspect.getattr_static(owner, name), staticmethod)
        setattr(owner, name, staticmethod(g) if static else g)

    nf = int(round(B.LAYER_DT / args.dt))
    nt = int(round(args.days * 86400.0 / B.LAYER_DT))
    n = nf * nt
    params, data_t0, orb = B.load_source(args.src, args.backend, n * args.dt, catalog=args.catalog,
                                         l1_dir=args.l1_dir, orbits=args.orbits)
    fit = erebor.get_stock("all_sources")
    off = data_t0 - W.REF
    oi = int(round(off / args.dt))
    tdi = TDIConfig(fit.general.tdi_gen_str, force_backend=args.backend)
    wg = get_emri_response_wrapper(Tobs=(n + oi) * args.dt + 4e4, dt=args.dt, t_start=W.REF,
                                   t0_shift_to_data=off - oi * args.dt, tdi_config=tdi,
                                   tdi_chan=fit.general.tdi_chan, order=fit.emri.response_order,
                                   force_backend=args.backend, orbits=orb)
    gen = wg.waveform_gen.waveform_generator
    tds = TDSettings(n, args.dt, t0=0.0, force_backend=args.backend)
    wdm = WDMSettings(nf, nt, args.dt, force_backend=args.backend)
    table = WDMLookupTable.from_file(args.direct_table, force_backend=args.backend)
    grids = [g.strip() for g in args.response_grid.split(",") if g.strip()]
    chunks = [int(c) for c in str(args.chunk_rows).split(",") if c.strip()]
    directs = {g: WD.EMRIDirectWDM(gen, table, wdm, orbits=orb, tdi_config=tdi, t_start=W.REF, data_t0=data_t0,
                                   mode_batch=args.mode_batch, force_backend=args.backend, response=args.response,
                                   response_grid=g) for g in grids}
    pool = cp.get_default_memory_pool() if gpu else None

    wrap(WD.EMRIDirectWDM, "_mode_list", "1 mode list (FEW call #1 + handoff check)")
    wrap(type(gen), "__call__", "  FEW generator calls (both)")
    wrap(EF.EMRITDIonFly, "__call__", "2 TOF total (FEW call #2 + feed + response)")
    wrap(EF.EMRITDIonFly, "mode_amp_phase", "  TOF feed: mode amp/phase")
    wrap(EF.EMRITDIonFly, "run_response", "  TOF run_response (input splines + kernel + output)")
    wrap(WD.EMRIDirectWDM, "_dense_response", "  DENSE response (kernel + output splines)")
    wrap(TF.TDTDIonTheFly, "__init__", "  TOF input splines")
    wrap(TF.TDTDIonTheFly, "__call__", "  TOF response kernel + output splines")
    wrap(WD, "harmonic_tracks_from_holder", "3 harmonic tracks")
    wrap(WD, "tracer_from_tof_output", "4 tracer (output spline evals)")
    wrap(WD, "accumulate_harmonic_batch", "5 lookup + scatter-add")
    wrap(type(table), "get_wdm_coeffs", "  table get_wdm_coeffs")
    wrap(WH, "wdm_chunk_of_td", "6 plunge chunks")

    def prod(thr):
        h = wg(*params, mode_selection_threshold=thr)
        xp = cp if gpu else np
        h = xp.stack([xp.asarray(c) for c in h]) if isinstance(h, (list, tuple)) else xp.atleast_2d(h)
        return TDSignal(h[:3, oi:oi + n], tds).transform(wdm).arr

    def record(**rec):
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    base = dict(src=args.src, days=args.days, dt=args.dt, nf=nf, nt=nt, backend=args.backend,
                response=args.response)
    for thr in [float(x) for x in args.thresh.split(",")]:
        prod(thr)                                                        # warm-up
        sync()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            prod(thr)
        sync()
        t_prod = (time.perf_counter() - t0) / args.reps
        summary = {}
        for grid, direct in directs.items():
            direct(*params, mode_selection_threshold=thr)               # warm-up
            sync()
            T.clear()
            C.clear()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                direct(*params, mode_selection_threshold=thr)
            sync()
            t_dir = (time.perf_counter() - t0) / args.reps
            print(f"\n[stages] response={args.response} grid={grid} thr={thr:g} modes={direct.last_stats.get('modes')} "
                  f"n_fine={direct.last_stats.get('n_fine')} backend={args.backend} grid Nf={nf} Nt={nt} dt={args.dt} "
                  f"({args.days:g} d): direct {t_dir * 1e3:.0f} ms, production {t_prod * 1e3:.0f} ms (per template)",
                  flush=True)
            for k in [lab for lab in LABELS if lab in T]:
                print(f"  {k:48s} {T[k] / args.reps * 1e3:8.1f} ms  ({C[k] // args.reps} calls)", flush=True)
            stages = {k.strip(): T[k] / args.reps * 1e3 for k in LABELS if k in T}
            record(**base, thr=thr, grid=grid, chunk=0, rows=1, modes=direct.last_stats.get("modes"),
                   n_response=direct.last_stats.get("n_fine"), ms_per_template=t_dir * 1e3,
                   production_ms=t_prod * 1e3, stages_ms=stages)
            summary[(grid, 0)] = t_dir * 1e3
            if args.batch_rows <= 0:
                continue
            rows = B.batch_rows(params, max(args.batch_rows, max(chunks)))
            sink = lambda idx, arr: None                                 # noqa: E731 (timing only)
            for chunk in chunks:
                direct.batch(rows[:min(2, chunk)], chunk_rows=chunk, consume=sink, mode_selection_threshold=thr)
                sync()
                if pool is not None:
                    pool.free_all_blocks()
                T.clear()
                C.clear()
                t0 = time.perf_counter()
                direct.batch(rows, chunk_rows=chunk, consume=sink, mode_selection_threshold=thr)
                sync()
                t_b = time.perf_counter() - t0
                mem = pool.total_bytes() / 1e9 if pool is not None else float("nan")
                print(f"[stages] grid={grid} thr={thr:g} BATCH of {len(rows)} ({chunk} per call): {t_b * 1e3:.0f} ms "
                      f"total = {t_b / len(rows) * 1e3:.0f} ms per template (production {t_prod * 1e3:.0f}); "
                      f"gpu pool {mem:.1f} GB {direct.last_stats}", flush=True)
                for k in [lab for lab in LABELS if lab in T]:
                    print(f"  {k:48s} {T[k] / len(rows) * 1e3:8.1f} ms per template  ({C[k]} calls)", flush=True)
                stages = {k.strip(): T[k] / len(rows) * 1e3 for k in LABELS if k in T}
                record(**base, thr=thr, grid=grid, chunk=chunk, rows=len(rows),
                       modes=direct.last_stats.get("modes"), n_response=direct.last_stats.get("n_response"),
                       ms_per_template=t_b / len(rows) * 1e3, production_ms=t_prod * 1e3, gpu_pool_gb=mem,
                       stages_ms=stages)
                summary[(grid, chunk)] = t_b / len(rows) * 1e3
        cols = [0] + (chunks if args.batch_rows > 0 else [])
        head = "  ".join(f"{('single' if c == 0 else f'{c}/call'):>8s}" for c in cols)
        print(f"\n[summary] {args.days:g} d thr={thr:g}: ms per template (production {t_prod * 1e3:.0f})\n"
              f"  {'grid':8s}{head}", flush=True)
        for grid in directs:
            print(f"  {grid:8s}" + "  ".join(f"{summary.get((grid, c), float('nan')):8.0f}" for c in cols), flush=True)


if __name__ == "__main__":
    main()
