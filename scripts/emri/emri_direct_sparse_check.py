"""EMRI direct-to-WDM: the response on the SPARSE grid (integrator knots + a spacing cap, exact
dense-output carrier at the pixels) vs the response on the PIXEL grid (one sample per pixel,
whole phase splined) vs production, on a CD1L source.

Reports per channel the mismatch sparse-vs-pixels and both vs production (active box), the
response points, and the wall time per template single and batched.

laptop (CPU, the 20 s table)::

    python scripts/emri/emri_direct_sparse_check.py --dt 20 --days 16 \
        --direct-table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5

cluster (GPU, 6-month grid)::

    python scripts/emri/emri_direct_sparse_check.py --backend cuda13x --days 180 --rows 16 \
        --direct-table wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5 \
        --catalog /shared/data/mojito_cache/catalogues/emri_cat_mojito_lite_processed_MT.hdf5 --orbits equal-arm
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_batch_speed as S  # noqa: E402
import emri_tof_xyz_threeway as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--direct-table", required=True)
    ap.add_argument("--src", type=int, default=1)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--l1-dir", default=None)
    ap.add_argument("--orbits", default="auto", choices=("auto", "l1", "equal-arm"))
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--dt", type=float, default=2.5)
    ap.add_argument("--days", type=float, default=180.0)
    ap.add_argument("--thresh", default="1e-3")
    ap.add_argument("--rows", type=int, default=4, help="templates timed (batched in one call)")
    ap.add_argument("--sparse-dt", type=float, default=43200.0)
    args = ap.parse_args()

    from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings
    from lisatools.globalfit.stock import erebor
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri.response import get_emri_response_wrapper
    from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

    nf = int(round(3600.0 / args.dt))
    nt = int(round(args.days * 86400.0 / 3600.0))
    nt += nt % 2
    n = nf * nt
    params, data_t0, orb = S.load_source(args.src, args.backend, n * args.dt, catalog=args.catalog,
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
    gpu = args.backend != "cpu"
    if gpu:
        import cupy as cp

    def sync():
        if gpu:
            cp.cuda.Device().synchronize()

    def host(x):
        return x.get() if hasattr(x, "get") else np.asarray(x)

    mk = dict(orbits=orb, tdi_config=tdi, t_start=W.REF, data_t0=data_t0, force_backend=args.backend,
              response="dense")
    gens = {g: EMRIDirectWDM(gen, table, wdm, response_grid=g, sparse_dt=args.sparse_dt, **mk)
            for g in ("pixels", "sparse")}
    act = slice(20, nt - 20)

    def mm(a, b):
        out = []
        for c in range(3):
            x, y = a[c][:, act], b[c][:, act]
            out.append(float(1 - np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y))))
        return out

    rows = S.batch_rows(params, args.rows)
    for thr in [float(x) for x in args.thresh.split(",")]:
        h = wg(*params, mode_selection_threshold=thr)
        xp = cp if gpu else np
        h = xp.stack([xp.asarray(c) for c in h]) if isinstance(h, (list, tuple)) else xp.atleast_2d(h)
        h_prod = host(TDSignal(h[:3, oi:oi + n], tds).transform(wdm).arr)
        rec = dict(src=args.src, days=args.days, dt=args.dt, thr=thr, backend=args.backend)
        tmpl = {}
        for g, d in gens.items():
            d(*params, mode_selection_threshold=thr)                     # warm-up
            sync()
            t0 = time.perf_counter()
            tmpl[g] = host(d(*params, mode_selection_threshold=thr).arr)
            sync()
            rec[f"{g}_single_ms"] = 1e3 * (time.perf_counter() - t0)
            rec[f"{g}_n_response"] = int(d.n_fine)
            rec[f"{g}_mm_vs_prod"] = mm(tmpl[g], h_prod)
            sync()
            t0 = time.perf_counter()
            d.batch(rows, chunk_rows=len(rows), mode_selection_threshold=thr)
            sync()
            rec[f"{g}_batch_ms_per_template"] = 1e3 * (time.perf_counter() - t0) / len(rows)
            rec[f"{g}_batch_n_response"] = int(d.last_stats.get("n_response", -1))
        rec["mm_sparse_vs_pixels"] = mm(tmpl["sparse"], tmpl["pixels"])
        print(f"[sparse] thr={thr:g} response points pixels {rec['pixels_n_response']} -> sparse "
              f"{rec['sparse_n_response']} (batch {rec['sparse_batch_n_response']}); "
              f"mm sparse vs pixels {'/'.join(f'{x:.1e}' for x in rec['mm_sparse_vs_pixels'])}; vs production: "
              f"pixels {'/'.join(f'{x:.1e}' for x in rec['pixels_mm_vs_prod'])}, sparse "
              f"{'/'.join(f'{x:.1e}' for x in rec['sparse_mm_vs_prod'])}", flush=True)
        print(f"[sparse] thr={thr:g} ms/template single: pixels {rec['pixels_single_ms']:.0f}, sparse "
              f"{rec['sparse_single_ms']:.0f} | batch of {len(rows)}: pixels "
              f"{rec['pixels_batch_ms_per_template']:.0f}, sparse {rec['sparse_batch_ms_per_template']:.0f}",
              flush=True)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
