#!/usr/bin/env python
"""One-GPU speed test: the SOBBH lookup comp vs the chunked-heterodyne comp, head to head.

Production 6-month grid by default (Nf=1440, Nt=4320, dt=2.5 s: layer 3600 s, band
2.5e-4..2.5e-2 Hz), synthetic epoch 0.5 yr, EqualArmlengthOrbits, 2nd-generation TDI, scirdv1
XYZ, ONE walker slab. For every batch size in ``--rows``: lookup ``get_ll_wdm`` (first call,
then the median of ``--repeats`` warm calls; the lookup comp was already exercised by the setup
fill, so its first call is not a from-scratch cold start) and ``fill_global_wdm``; chunked
``get_ll_wdm`` (``--m-band``) and ``fill_global_wdm`` (``--fill-band``); a device sync around
every call; the lookup's response/tracer/lookup/inner split; the GPU memory pool (in-use and total, per comp,
freed between the comps); and the per-row lnL difference of the two comps on the same residual
(a consistency figure, not an accuracy gate: the accuracy gates are docs/sobbh-wdm-lookup.md).

Cluster (one GPU; the stack's python, the branch checked out):
    SOBBH_LOOKUP_TABLE_PATH=/path/to/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5 \\
    python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda12x --rows 4,8,32,96,288 \\
        --out sobbh_lookup_speed_gpu.jsonl
Laptop smoke (CPU, small preset):
    .wtenv/run.sh scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cpu --laptop --nt 256 \\
        --rows 2,4 --repeats 1 --out /tmp/speed_smoke.jsonl
The table depends on the layer duration only, so the 3600-s laptop table serves the production
grid (TablePortabilityTest).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

TABLE_DEFAULT = (
    "/Users/mkatz/Research/lisa_sprint_2026/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5"
)
# (m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta): the gate's catalogue-like rows
SOURCES = np.array(
    [
        [36.0, 29.0, 0.1, -0.2, 0.6e9, 4.5e-3, 0.3, 0.8, 1.1, 2.0, 0.4],
        [50.0, 40.0, 0.3, 0.3, 1.0e9, 8.0e-3, 2.2, 2.1, 0.4, 4.4, -0.7],
        [25.0, 20.0, -0.4, 0.0, 0.3e9, 1.2e-2, 4.0, 1.4, 2.5, 0.7, 1.1],
        [60.0, 55.0, 0.1, 0.2, 0.8e9, 1.5e-2, 1.1, 1.2, 0.7, 3.1, 0.2],
        [20.0, 10.0, 0.0, 0.5, 0.5e9, 6.0e-3, 5.5, 0.3, 1.8, 5.9, -1.2],
        [80.0, 75.0, 0.6, 0.6, 2.0e9, 1.8e-2, 0.9, 1.9, 0.1, 1.5, 0.9],
    ]
)


def sync(xp):
    if hasattr(xp, "cuda"):
        xp.cuda.Device().synchronize()


def timed(fn, xp, repeats):
    """``(cold, warm_median)`` seconds of ``fn()``: one cold call, then ``repeats`` warm calls."""
    sync(xp)
    t = time.perf_counter()
    fn()
    sync(xp)
    cold = time.perf_counter() - t
    warm = []
    for _ in range(int(repeats)):
        sync(xp)
        t = time.perf_counter()
        fn()
        sync(xp)
        warm.append(time.perf_counter() - t)
    return cold, (float(np.median(warm)) if warm else cold)


def pool_reset(xp):
    """Release the cupy pool's cached blocks (no-op on numpy)."""
    if hasattr(xp, "cuda"):
        xp.get_default_memory_pool().free_all_blocks()


def pool_gb(xp):
    """``(used, total)`` GB of the cupy default pool (nan on numpy)."""
    if not hasattr(xp, "cuda"):
        return float("nan"), float("nan")
    pool = xp.get_default_memory_pool()
    return pool.used_bytes() / 1e9, pool.total_bytes() / 1e9


def clean(rec):
    """NaN -> None so the JSON line is strict."""
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--backend",
        default=None,
        help="cpu / cuda12x / ... (default: the first CUDA backend, else cpu)",
    )
    ap.add_argument("--laptop", action="store_true", help="Nf=180, dt=20 (layer 3600 s) preset")
    ap.add_argument("--nf", type=int, default=1440)
    ap.add_argument("--nt", type=int, default=4320)
    ap.add_argument("--dt", type=float, default=2.5)
    ap.add_argument("--rows", default="4,8,32,96")
    ap.add_argument("--row-batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--eval-dt", type=float, default=43200.0)
    ap.add_argument("--interp", default="spline", choices=("spline", "cubic", "linear"))
    ap.add_argument(
        "--kernel",
        default="auto",
        choices=("auto", "kernel", "python"),
        help="fused C++/CUDA lookup (auto: when the backend module has it)",
    )
    ap.add_argument("--table", default=os.environ.get("SOBBH_LOOKUP_TABLE_PATH", TABLE_DEFAULT))
    ap.add_argument("--out", default="sobbh_lookup_speed_gpu.jsonl")
    ap.add_argument("--no-chunked", action="store_true")
    ap.add_argument("--m-band", type=int, default=3, help="chunked scoring band half-width")
    ap.add_argument("--fill-band", type=int, default=8, help="chunked fill band half-width")
    ap.add_argument("--nt-sub", type=int, default=32)
    args = ap.parse_args()

    import lisatools

    backend = args.backend
    if backend is None:
        backend = "cuda" if lisatools.has_backend("cuda") else "cpu"
    if not lisatools.has_backend(backend):
        raise SystemExit(f"backend {backend!r} unavailable on this host")
    if args.laptop:
        args.nf, args.dt = 180, 20.0
    nf, nt, dt = int(args.nf), int(args.nt), float(args.dt)

    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.domains import WDMLookupTable, WDMSettings, WDMSignal
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations
    from lisatools.utils.constants import YRSID_SI
    from lisatools.utils.utility import asnumpy

    t0 = int(0.5 * YRSID_SI / dt) * dt
    ref = float(t0)
    orbits = EqualArmlengthOrbits(force_backend=backend)
    tdi = TDIConfig("2nd generation", force_backend=backend)
    wdm = WDMSettings(nf, nt, dt, t0=t0, min_freq=2.5e-4, max_freq=2.5e-2, force_backend=backend)
    t_build = time.perf_counter()
    table = WDMLookupTable.from_file(args.table, force_backend=backend)
    comp = SOBBHLookupComputations(
        wdm,
        ref,
        table,
        orbits=orbits,
        tdi_config=tdi,
        tdi_type="XYZ",
        n_grid=2048,
        buffer_time=5000.0,
        eval_dt=args.eval_dt,
        num_m_layers=2,
        interp=args.interp,
        kernel=args.kernel,
        row_batch=args.row_batch,
        force_backend=backend,
        d_d=0.0,
    )
    xp = comp.xp
    t_build = time.perf_counter() - t_build
    nch, nfa, nta = 3, int(wdm.Nf_active), int(wdm.Nt_active)
    print(
        f"backend {backend}  grid Nf={nf} Nt={nt} dt={dt} layer_dt={float(wdm.layer_dt):g} s  "
        f"active {nfa} x {nta}  table {os.path.basename(args.table)}  comp build {t_build:.1f} s  "
        f"lookup={'fused kernel' if comp.uses_kernel else 'python'} interp={args.interp}"
    )

    # one residual slab: the lookup fill of two sources (no dense transform on any backend)
    data = xp.zeros(nch * nfa * nta)
    comp.fill_global_wdm(
        SOURCES[:2], data, data_index=np.zeros(2, dtype=np.int32), factors=np.ones(2)
    )
    sync(xp)
    ac = AnalysisContainer(
        WDMSignal(data.reshape(nch, nfa, nta), wdm), XYZ2SensitivityMatrix(wdm, model="scirdv1")
    )
    dev = getattr(getattr(data, "device", None), "id", None)
    aca = AnalysisContainerArray([ac], gpus=None if dev is None else [int(dev)])

    ch = None
    if not args.no_chunked:
        from bbhx.sobbhcomps import SOBBHWDMComputations

        t_build = time.perf_counter()
        ch = SOBBHWDMComputations(
            wdm,
            t_ref=ref,
            Nt_sub=args.nt_sub,
            n_pad=4,
            N_sparse=256,
            N_cp_sig=0,
            N_cp_orbit=0,
            orbits=orbits,
            tdi_config="2nd generation",
            force_backend=backend,
            d_d=0.0,
            tdi_type="XYZ",
        )
        print(
            f"chunked comp: Nt_sub={args.nt_sub} n_chunks={ch.n_chunks} "
            f"build {time.perf_counter() - t_build:.1f} s"
        )

    rng = np.random.default_rng(1)
    tag = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows_list = [int(r) for r in args.rows.split(",") if r.strip()]
    header = (
        f"{'rows':>5} {'look_first':>11} {'look_warm':>10} {'resp':>8} {'pn':>8} {'trac':>8} {'look':>8} {'inner':>8} "
        f"{'look_fill':>10} {'ch_first':>9} {'ch_warm':>9} {'ch_fill':>9} {'ch/look':>8} "
        f"{'max|dll|':>10} {'lk_pool':>8} {'ch_pool':>8} {'snr':>8}"
    )
    print(header)
    with open(args.out, "a") as fp:
        for n in rows_list:
            batch = np.tile(SOURCES[0], (n, 1))
            batch[:, 5] += rng.uniform(-0.3, 0.3, n) * float(wdm.layer_df)
            batch[:, 6] = rng.uniform(0.0, 2.0 * np.pi, n)
            idx = np.zeros(n, dtype=np.int32)
            fac = np.ones(n)
            rec = dict(
                rows=n,
                backend=backend,
                nf=nf,
                nt=nt,
                dt=dt,
                row_batch=args.row_batch,
                eval_dt=args.eval_dt,
                m_band=args.m_band,
                fill_band=args.fill_band,
                num_m_layers=2,
                nt_sub=args.nt_sub,
                interp=args.interp,
                lookup="kernel" if comp.uses_kernel else "python",
                tag=tag,
            )
            pool_reset(xp)
            spans = {}

            def look_call():
                out = comp.get_ll_wdm(batch, aca, data_index=idx, noise_index=idx)
                spans.update(comp.last_call_spans or {})
                return out

            look_first, look_warm = timed(look_call, xp, args.repeats)
            ll_look = np.asarray(asnumpy(look_call()), dtype=float)
            # the rows' optimal SNR against this grid's noise (scirdv1, instrument only):
            # sqrt(<h|h>) of the lookup template
            snr = float(np.median(np.sqrt(np.asarray(asnumpy(comp.h_h_out), dtype=float))))
            buf = xp.zeros(nch * nfa * nta)
            _, look_fill = timed(
                lambda: comp.fill_global_wdm(batch, buf, data_index=idx, factors=fac),
                xp,
                args.repeats,
            )
            lk_used, lk_total = pool_gb(xp)
            pool_reset(xp)
            rec.update(
                look_first=look_first,
                look_warm=look_warm,
                look_fill=look_fill,
                snr=snr,
                look_template=float(spans.get("template", float("nan"))),
                look_response=float(spans.get("response", float("nan"))),
                look_pn=float(spans.get("pn", float("nan"))),
                look_tracer=float(spans.get("tracer", float("nan"))),
                look_lookup=float(spans.get("lookup", float("nan"))),
                look_inner=float(spans.get("inner", float("nan"))),
                lookup_stats=dict(comp.last_stats),
                look_pool_used_gb=lk_used,
                look_pool_total_gb=lk_total,
            )

            ch_first = ch_warm = ch_fill = dll = float("nan")
            if ch is not None:
                ch_first, ch_warm = timed(
                    lambda: ch.get_ll_wdm(
                        batch, aca, data_index=idx, noise_index=idx, m_band_half_width=args.m_band
                    ),
                    xp,
                    args.repeats,
                )
                ll_ch = np.asarray(
                    asnumpy(
                        ch.get_ll_wdm(
                            batch,
                            aca,
                            data_index=idx,
                            noise_index=idx,
                            m_band_half_width=args.m_band,
                        )
                    ),
                    dtype=float,
                )
                buf2 = xp.zeros(nch * nfa * nta)
                _, ch_fill = timed(
                    lambda: ch.fill_global_wdm(
                        batch, buf2, data_index=idx, factors=fac, m_band_half_width=args.fill_band
                    ),
                    xp,
                    args.repeats,
                )
                dll = float(np.abs(ll_look - ll_ch).max())
                rec.update(
                    ch_first=ch_first,
                    ch_warm=ch_warm,
                    ch_fill=ch_fill,
                    max_abs_dll=dll,
                    median_abs_dll=float(np.median(np.abs(ll_look - ll_ch))),
                )
            ch_used, ch_total = pool_gb(xp)
            rec.update(ch_pool_used_gb=ch_used, ch_pool_total_gb=ch_total)
            ratio = ch_warm / look_warm if look_warm > 0 else float("nan")
            print(
                f"{n:5d} {look_first:11.3f} {look_warm:10.3f} {rec['look_response']:8.3f} {rec['look_pn']:8.3f} "
                f"{rec['look_tracer']:8.3f} {rec['look_lookup']:8.3f} {rec['look_inner']:8.3f} {look_fill:10.3f} {ch_first:9.3f} {ch_warm:9.3f} "
                f"{ch_fill:9.3f} {ratio:8.2f} {dll:10.3e} {lk_total:8.2f} {ch_total:8.2f} {snr:8.3g}"
            )
            fp.write(json.dumps(clean(rec)) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
