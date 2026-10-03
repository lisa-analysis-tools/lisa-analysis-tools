"""Fit-wiring check of ``EMRI_LIKELIHOOD=direct``: the REAL getters and the REAL move.

Builds the production EMRI wrap (``get_emri_wave_wrap``) and the direct-to-WDM adapter
(``get_emri_direct_gen``) exactly as the global fit does (cfg from ``source_signal_cfg``
with the EMRI direct resolver), installs the fit's engine template generator
(``SourceSignalGen``: with ``--fill direct``, the default and the fit's setup since 10-03, the
direct template; ``--fill production`` the production wrap) on one container per walker
holding ``production(truth) - fill(cold)`` on a fit-style WDM run domain (active band + edge
crop), then:

* template mismatch, direct vs production, at the truth;
* ``EMRIDirectLikeMove.compute_like`` (direct, batched) vs ``compute_acs_like`` (the per-row
  container path with the INSTALLED generator) and vs ``compute_check_like`` (the move's
  cross-check: the production generator), row by row;
* wall time per row of the paths (``--backend`` GPU: synchronised).

laptop (CPU, the 20 s table, short window)::

    python scripts/emri/emri_direct_fit_wiring_check.py --dt 20 --days 16 \
        --table wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5 --response spline

cluster (GPU, the 6-month grid)::

    python scripts/emri/emri_direct_fit_wiring_check.py --backend cuda12x --days 180 \
        --table-dir /path/to/tables --rows 8 \
        --catalog /path/to/emri_cat_mojito_lite_processed_MT.hdf5 --l1-dir /path/to/EMRI/L1

Prints one ``[fitwire]`` line per row and a JSON summary (``--out`` appends it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_batch_speed as S  # noqa: E402  (load_source, batch_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default=None,
                    help="lookup table; default: the canonical table in --table-dir (built if missing)")
    ap.add_argument("--table-dir", default=os.getcwd())
    ap.add_argument("--src", type=int, default=1)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--l1-dir", default=None)
    ap.add_argument("--orbits", default="auto", choices=("auto", "l1", "equal-arm"))
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--dt", type=float, default=2.5)
    ap.add_argument("--days", type=float, default=180.0)
    ap.add_argument("--edge", type=int, default=60, help="EDGE_CROP_WAVELETS of the run domain")
    ap.add_argument("--eps", type=float, default=1e-3, help="EMRI_EPS")
    ap.add_argument("--response", default="dense", choices=("dense", "spline"))
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--fill", choices=("direct", "production"), default="direct",
                    help="the containers' installed (engine) generator; direct = the fit's setup")
    ap.add_argument("--walkers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, uniform_dist

    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.domains import TDSettings, WDMSettings, WDMSignal
    from lisatools.globalfit.moves import EMRIDirectLikeMove
    from lisatools.globalfit.stock.erebor import source_runtime as sr
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.utils.utility import asnumpy

    layer = 3600.0
    nf = int(round(layer / args.dt))
    nt = int(round(args.days * 86400.0 / layer))
    nt += nt % 2
    n = nf * nt
    params, data_t0, orb = S.load_source(args.src, args.backend, n * args.dt, catalog=args.catalog,
                                         l1_dir=args.l1_dir, orbits=args.orbits)
    params = np.asarray(params, dtype=float)
    gpu = args.backend != "cpu"
    if gpu:
        import cupy as cp

    def sync():
        if gpu:
            cp.cuda.Device().synchronize()

    dom = WDMSettings(nf, nt, args.dt, min_freq=2.5e-4, max_freq=2.5e-2, min_time=args.edge * layer,
                      max_time=(nt - args.edge) * layer, force_backend=args.backend)
    tds = TDSettings(n, args.dt, t0=0.0, force_backend=args.backend)
    gi = SimpleNamespace(gpus=[0] if gpu else None, orbits=orb, gpu_orbits=orb,
                         force_backend=args.backend, dt=args.dt, data_t0=data_t0, Tobs=n * args.dt,
                         data_td_settings=tds, domain_settings=dom, file_store_dir=args.table_dir)
    gs = SimpleNamespace(tdi_chan="XYZ", tdi_gen_str="2nd generation", nchannels=3, data_mode="mojito",
                         sobbh_reference_time=None, mbh_waveform_t0=0.0, min_freq=2.5e-4, max_freq=2.5e-2)
    emri = sr.SourceEMRISettings(likelihood="direct", direct_table=args.table, eps=args.eps,
                                 direct_response=args.response, traj_workers=0, waveform_kwargs={})
    cfg = sr.source_signal_cfg(gs, sr.SourceMBHSettings(likelihood="full"), sr.SourceSOBBHSettings(),
                               emri, domain_settings=dom)
    t0 = time.perf_counter()
    prod = sr.get_emri_wave_wrap(gi, cfg)
    direct = sr.get_emri_direct_gen(gi, cfg)
    build_s = time.perf_counter() - t0
    print(f"[fitwire] grid Nf={nf} Nt={nt} dt={args.dt} active f[{dom.ind_min_f}:{dom.ind_max_f}] "
          f"t[{dom.ind_min_t}:{dom.ind_max_t}] src={args.src} eps={args.eps} response={args.response} "
          f"build {build_s:.1f} s", flush=True)

    sens = XYZ2SensitivityMatrix(dom, model="scirdv1")
    rows = np.asarray(S.batch_rows(params, args.rows + args.walkers), dtype=float)
    cold, rows = rows[: args.walkers], rows[args.walkers:]
    truth = np.asarray(asnumpy(prod(*params).arr))

    if args.fill == "direct":
        fill_cfg = dict(cfg)
    else:
        fill_cfg = dict(cfg, emri_likelihood="full")
    gen = sr.SourceSignalGen("emri", None, gi, fill_cfg)            # the fit's engine generator

    acs_list = []
    for w in range(args.walkers):
        res = truth - np.asarray(asnumpy(gen(*cold[w], apply_transform=False).arr))
        xp_res = cp.asarray(res) if gpu else res
        ac = AnalysisContainer(WDMSignal(xp_res, dom), sens)
        ac.signal_gen = {"emri": gen}
        acs_list.append(ac)
    acs = AnalysisContainerArray(acs_list)
    ndim = params.size
    priors = {"emri": ProbDistContainer({i: uniform_dist(-1e30, 1e30) for i in range(ndim)})}
    move = EMRIDirectLikeMove(
        "emri", (1, args.walkers, 1, ndim), prod, {}, {}, acs, 1, None, priors, [(StretchMove(), 1.0)],
        betas_all=np.ones((1, 1)), direct_gen=direct, batch_max_size=8, name="emri fit wiring check")
    move._current_leaf = 0
    move.remove_cold_chain_sources(cold)
    move.setup_likelihood_here(cold)
    idx = np.arange(rows.shape[0]) % args.walkers

    sync()
    t0 = time.perf_counter()
    fast = move.compute_like(rows, idx)
    sync()
    fast_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    slow = move.compute_acs_like(rows, idx)
    sync()
    slow_s = time.perf_counter() - t0
    check = np.real(np.asarray(move.compute_check_like(rows, idx), dtype=float))

    def mm(a, b):
        ac = AnalysisContainer(WDMSignal(b, dom), sens)
        bb = float(np.real(ac.inner_product()))
        ll = float(np.real(ac.template_likelihood(WDMSignal(a, dom))))
        aa = float(np.real(ac.template_snr(WDMSignal(a, dom))[0])) ** 2
        return 1.0 - 0.5 * (bb + aa + 2 * ll) / np.sqrt(aa * bb), np.sqrt(aa / bb)

    arr, ok = direct.templates(params[None])
    mm_truth, amp_truth = mm(arr[0], prod(*params).arr)
    print(f"[fitwire] truth: mismatch(direct, production)={mm_truth:.3e} amp ratio={amp_truth:.6f} "
          f"ok={bool(ok[0])} stats={direct.last_stats}", flush=True)
    for k in range(rows.shape[0]):
        print(f"[fitwire] row {k} walker {idx[k]}: lnL direct={fast[k]:.6f} container({args.fill})="
              f"{slow[k]:.6f} dlogL={fast[k] - slow[k]:+.4e}; production check={check[k]:.6f} "
              f"dlogL={fast[k] - check[k]:+.4e}", flush=True)
    summary = dict(
        src=args.src, nf=nf, nt=nt, dt=args.dt, days=args.days, eps=args.eps, response=args.response,
        backend=args.backend, mm_truth=mm_truth, amp_truth=amp_truth,
        fill=args.fill, dlogL=[float(x) for x in fast - slow], max_abs_dlogL=float(np.abs(fast - slow).max()),
        max_abs_dlogL_vs_production=float(np.abs(fast - check).max()),
        direct_s_per_row=fast_s / rows.shape[0], production_s_per_row=slow_s / rows.shape[0],
        fallbacks=int(move.n_batch_fallbacks), build_s=build_s,
    )
    print(f"[fitwire] fill={args.fill}: max|dlogL| vs the container path {summary['max_abs_dlogL']:.4e}, vs "
          f"production {summary['max_abs_dlogL_vs_production']:.4e}; direct {summary['direct_s_per_row']:.3f} "
          f"s/row vs container {summary['production_s_per_row']:.3f} s/row, fallbacks={summary['fallbacks']}",
          flush=True)
    print(json.dumps(summary), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
