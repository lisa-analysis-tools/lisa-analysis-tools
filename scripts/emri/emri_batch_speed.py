"""Batched EMRI template speed test: GPU (or CPU) production templates + CPU trajectory fan-out.

For ONE CD1L EMRI, build a batch of template parameter rows the way the sampler uses them
(information-matrix style: +/- steps in each of the 14 parameters, plus jittered walker
rows), and time, per mode threshold:

* ``serial``: each template integrates its own trajectory in the parent process;
* ``pooled``: ``few.trajectory.pool``: every row's inspiral call is captured, the unique
  trajectories are integrated on a pool of CPU worker processes, and the templates are
  then built with every trajectory served from the cache.

Template = the production EMRI response (get_emri_response_wrapper: SPECIAL frame, ICRS L1
orbits, Lagrange order 40, REF-anchored) sliced onto the data grid and transformed to WDM,
on ``--backend`` (cuda12x on the cluster). Reports wall per template, the trajectory-only
share, the precompute cost, the speedup, and a bitwise check that pooled templates equal
serial ones. Appends one JSON line per threshold to ``--out``.

Env: MOJITO_LIGHT_PATH (mojito light v1.0.0 dir), START_OFFSET_S (default 5e4).
Example (cluster GPU, 6mo production grid):
    python scripts/emri/emri_batch_speed.py --src 1 --backend cuda12x --workers 8 --rows 64
Laptop smoke:
    python scripts/emri/emri_batch_speed.py --src 1 --backend cpu --days 4 --dt 20 --rows 8 --workers 2
"""
import argparse
import json
import os
import socket
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emri_tof_xyz_threeway as W  # noqa: E402  (catalogue reader constants: PATH, REF)

LAYER_DT = 3600.0


def load_source(src, backend, span):
    """Catalogue params (FEW order, special frame), data start, and ICRS L1 orbits."""
    from mojito import MojitoL1File

    from lisatools.detector import L1Orbits
    from lisatools.globalfit.preprocessing import find_file
    from lisatools.sources.utils import icrs_to_ecliptic

    path = os.environ.get("MOJITO_LIGHT_PATH", W.PATH)
    with h5py.File(os.path.join(path, "catalogues", "emri_cat_mojito_lite_processed_MT.hdf5"), "r") as f:
        b = f["Binaries"]
        g = lambda k: float(b[k][src])  # noqa: E731
        lam, beta = icrs_to_ecliptic(g("RightAscension") % (2 * np.pi), g("Declination"))
        params = [g("PrimaryMassSSBFrame"), g("SecondaryMassSSBFrame"), g("PrimarySpinParameter"),
                  g("SemiLatusRectum"), g("Eccentricity"), 1.0, g("LuminosityDistance") / 1e3,
                  float(np.pi / 2 - beta), float(lam) % (2 * np.pi),
                  g("PolarAnglePrimarySpin"), g("AzimuthalAnglePrimarySpin"),
                  g("AzimuthalPhase"), g("PolarPhase"), g("RadialPhase")]
    fp = find_file(os.path.join(path, "data", "EMRI", "L1"), "EMRI", src)
    ts = MojitoL1File(fp).tdis.time_sampling
    data_t0 = float(ts.t0) + float(os.environ.get("START_OFFSET_S", "5e4"))
    orb = L1Orbits(fp, force_backend=backend, frame="icrs")
    if backend != "cpu":
        return params, data_t0, orb          # GPU nodes: keep the full table (no host-side trim)
    # keep the light-travel-time table to [REF - pad, window end + pad] (laptop memory), as the campaign does
    pad = 1e5
    lo = max(W.REF - pad, float(orb.sc_t0))
    hi = min(data_t0 + span + pad, float(orb._sc_t_base[-1]))
    lt = np.asarray(orb.ltt_t)
    m = (lt >= lo) & (lt <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy()
    orb.ltt_t = lt[m].copy()
    orb.ltt_t0 = float(orb.ltt_t[0])
    orb.configure(linear_interp_setup=True)
    return params, data_t0, orb


def batch_rows(params, n_rows, seed=11):
    """Information-matrix rows (+/- a step per parameter; x0 held) then jittered walker rows."""
    p = np.asarray(params, dtype=float)
    step = np.array([1e-6 * p[0], 1e-6 * p[1], 1e-6, 1e-6, 1e-6, 0.0, 1e-4 * p[6],
                     1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
    rows = []
    for i in range(len(p)):
        if step[i] == 0.0:
            continue
        for s in (+1, -1):
            q = p.copy()
            q[i] += s * step[i]
            rows.append(q)
    rng = np.random.default_rng(seed)
    while len(rows) < n_rows:
        rows.append(p + rng.normal(size=p.size) * step * 10)
    return [list(map(float, r)) for r in rows[:n_rows]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=int, default=1)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--dt", type=float, default=2.5, help="sample spacing [s] (production 2.5)")
    ap.add_argument("--days", type=float, default=180.0, help="observation span (production 180)")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--thresh", default="1e-3,1e-4,1e-5")
    ap.add_argument("--out", default="emri_batch_speed.jsonl")
    args = ap.parse_args()

    from few.trajectory.pool import TrajectoryCache, TrajectoryPool, inspiral_init_kwargs_from, reset_stepper

    from lisatools.domains import TDSettings, TDSignal, WDMSettings
    from lisatools.globalfit.stock import erebor
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri.response import get_emri_response_wrapper

    nf = int(round(LAYER_DT / args.dt))
    nt = int(round(args.days * 86400.0 / LAYER_DT))
    n = nf * nt
    params, data_t0, orb = load_source(args.src, args.backend, n * args.dt)
    fit = erebor.get_stock("all_sources")
    off = data_t0 - W.REF
    offset_int = int(round(off / args.dt))
    wg = get_emri_response_wrapper(
        Tobs=(n + offset_int) * args.dt + 4e4, dt=args.dt, t_start=W.REF, t0_shift_to_data=off - offset_int * args.dt,
        tdi_config=TDIConfig(fit.general.tdi_gen_str, force_backend=args.backend), tdi_chan=fit.general.tdi_chan,
        order=fit.emri.response_order, force_backend=args.backend, orbits=orb)
    gen = wg.waveform_gen.waveform_generator
    tds = TDSettings(n, args.dt, t0=0.0, force_backend=args.backend)
    wdm = WDMSettings(nf, nt, args.dt, force_backend=args.backend)
    gpu = args.backend not in ("cpu",)
    if gpu:
        import cupy as cp

    def sync():
        if gpu:
            cp.cuda.Device().synchronize()

    thr_now = {"v": None}

    def template(*p):
        h = wg(*p, mode_selection_threshold=thr_now["v"])
        xp = cp if gpu else np
        h = xp.stack([xp.asarray(c) for c in h]) if isinstance(h, (list, tuple)) else xp.atleast_2d(h)
        h = h[:3, offset_int:offset_int + n]
        return TDSignal(h, tds).transform(wdm).arr

    rows = batch_rows(params, args.rows)
    t0 = time.perf_counter()
    pool = TrajectoryPool(n_workers=args.workers, inspiral_init_kwargs=inspiral_init_kwargs_from(gen))
    cache = TrajectoryCache.install(gen)
    thr_now["v"] = 1e-3
    cache.precompute(template, rows[:args.workers], pool)      # spawn + warm every worker
    pool_start = time.perf_counter() - t0
    cache.clear()
    print(f"[speed] src={args.src} backend={args.backend} grid Nf={nf} Nt={nt} dt={args.dt} rows={len(rows)} "
          f"workers={args.workers} pool start-up {pool_start:.1f}s", flush=True)

    for thr in [float(x) for x in args.thresh.split(",")]:
        thr_now["v"] = thr
        rec = dict(src=args.src, backend=args.backend, host=socket.gethostname(), nf=nf, nt=nt, dt=args.dt,
                   rows=len(rows), workers=args.workers, thresh=thr, pool_startup_s=pool_start)
        TrajectoryCache.uninstall(gen)
        reset_stepper(gen.inspiral_generator)
        template(*rows[0])                                       # warm-up (FEW/GPU kernels)
        sync()

        # serial: every template integrates its own trajectory
        t_ser = time.perf_counter()
        for r in rows:
            reset_stepper(gen.inspiral_generator)
            template(*r)
        sync()
        t_ser = time.perf_counter() - t_ser

        # the trajectory-only share of the serial path
        cache = TrajectoryCache.install(gen)
        calls = cache.capture(template, rows)
        TrajectoryCache.uninstall(gen)
        uniq = {}
        from few.trajectory.pool import call_key
        for c in calls:
            if c is not None:
                uniq.setdefault(call_key(*c), c)
        t_traj = time.perf_counter()
        for c in calls:
            if c is not None:
                reset_stepper(gen.inspiral_generator)
                gen.inspiral_generator(*c[0], **c[1])
        t_traj = time.perf_counter() - t_traj

        # pooled: capture + fan-out + collect, then templates from the cache
        cache = TrajectoryCache.install(gen)
        cache.clear()
        t_pre = time.perf_counter()
        info = cache.precompute(template, rows, pool)
        t_pre = time.perf_counter() - t_pre
        t_pool = time.perf_counter()
        for r in rows:
            template(*r)
        sync()
        t_pool = time.perf_counter() - t_pool
        hits, misses = cache.hits, cache.misses

        # bitwise check on a few rows: pooled (cache hit) vs serial (fresh stepper)
        maxdiff = 0.0
        for r in rows[:3]:
            a = template(*r)
            TrajectoryCache.uninstall(gen)
            reset_stepper(gen.inspiral_generator)
            b = template(*r)
            cache = TrajectoryCache.install(gen)
            a, b = (x.get() if hasattr(x, "get") else np.asarray(x) for x in (a, b))
            maxdiff = max(maxdiff, float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)))
        cache.clear()

        rec.update(nmodes=int(getattr(gen, "num_modes_kept", -1)), unique_trajectories=len(uniq),
                   serial_s=t_ser, serial_per_template_s=t_ser / len(rows), traj_only_s=t_traj,
                   traj_share_serial=t_traj / t_ser, precompute_s=t_pre, pooled_templates_s=t_pool,
                   pooled_total_s=t_pre + t_pool, speedup=t_ser / (t_pre + t_pool),
                   cache_hits=hits, cache_misses=misses, precompute_info=info,
                   max_rel_diff_pooled_vs_serial=maxdiff)
        print(f"[speed] thr={thr:g} modes={rec['nmodes']} serial {t_ser:.2f}s ({t_ser / len(rows) * 1e3:.0f} ms/tmpl, "
              f"traj share {t_traj / t_ser:.0%}) | pooled precompute {t_pre:.2f}s + templates {t_pool:.2f}s = "
              f"{t_pre + t_pool:.2f}s  speedup {rec['speedup']:.2f}x  hits {hits}/{len(rows)}  "
              f"unique traj {len(uniq)}  maxdiff {maxdiff:.1e}", flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    pool.close()


if __name__ == "__main__":
    main()
