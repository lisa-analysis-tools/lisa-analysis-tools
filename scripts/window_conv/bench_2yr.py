"""Benchmark ``XYZSensitivityBackend.compute_log_like`` with/without window convolution.

Measures, per (num_walkers, convolve_window) case on a CUDA backend:
* wall time (median / min over ``--repeats`` timed calls after ``--warmup`` calls,
  ``Device.synchronize()`` around each call);
* peak cupy memory-pool usage DURING the call (a tracking allocator records
  ``pool.used_bytes()`` after every allocation), minus the resident baseline
  (data + params + backend arrays), and the pool's ``total_bytes()`` afterwards;
* the device free-memory drop seen by ``runtime.memGetInfo`` (catches allocations
  outside the cupy pool, e.g. a cuFFT workspace not routed through it).
Pool blocks and the cuFFT plan cache are freed between cases.

Defaults reproduce the production-like 2-yr setup (galaxy on, splines off, aliasing on,
dt = 5 s, Tukey window). DO NOT run the full 2-yr size on a shared GPU without checking
free memory first: the data array alone is ``walkers * 3 * nf * 16 B`` (the binding
requires one data block per walker), i.e. ~0.3 GB/walker at 2 yr, and the legacy
convolution path needs several GB/walker on top.

    CUDA_VISIBLE_DEVICES=5 .venv/bin/python bench_2yr.py --T-days 30 --walkers 8
    ... --T-days 730 --walkers 8 32 --modes 0 1 --kw window_convolution_mode=exact

``--kw key=value`` is forwarded to ``XYZSensitivityBackend`` (values parsed as Python
literals when possible) so new kwargs can be benchmarked without editing this file.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import time

import numpy as np

import _use_worktree  # noqa: F401
from fixtures import CONFIGS, FIDUCIAL, make_backend, window_alpha, window_taper_s


def _parse_kw(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def make_bench_inputs(sm, num_walkers: int, seed: int = 0):
    """Cheap physical-ish inputs on the device: per-walker params with +-20 % scatter and
    data = sqrt(diag C_fid / (2 df)) * CN(0, 1) per channel (diagonal approximation;
    the timing does not depend on the data values)."""
    xp = sm.xp
    cfg = sm._fixture_config
    rng = np.random.default_rng(seed)
    n_par = sm.n_noise_par
    shape = (num_walkers, n_par) if n_par > 1 else (num_walkers,)
    p = dict(
        Soms_in_all=FIDUCIAL["Soms"] * rng.uniform(0.8, 1.2, shape),
        Sa_in_all=FIDUCIAL["Sa"] * rng.uniform(0.8, 1.2, shape),
    )
    gal_keys = ("Amp", "alpha", "f_1", "kn", "f_2")
    for k in gal_keys:
        v = FIDUCIAL[k] * rng.uniform(0.9, 1.1, num_walkers) if cfg["galaxy"] else np.zeros(num_walkers)
        p[f"{k}_in_all"] = v
    if cfg["use_splines"]:
        f = sm.f_arr[sm.f_arr > 0]
        lo, hi = float(xp.log10(f[0])) - 0.1, float(xp.log10(f[-1])) + 0.1
        k = cfg["n_knots"]
        p["knots_position_all"] = np.broadcast_to(np.linspace(lo, hi, k), (2, num_walkers, k)).copy()
        p["knots_amplitude_all"] = 0.1 * rng.standard_normal((2, num_walkers, k))
    p = {k: xp.asarray(v) for k, v in p.items()}

    gal = [FIDUCIAL[k] for k in gal_keys] if cfg["galaxy"] else [0.0] * 5
    Soms = np.full(n_par, FIDUCIAL["Soms"]) if n_par > 1 else FIDUCIAL["Soms"]
    Sa = np.full(n_par, FIDUCIAL["Sa"]) if n_par > 1 else FIDUCIAL["Sa"]
    kp = ka = None
    if cfg["use_splines"]:
        kp, ka = p["knots_position_all"][:, :1].copy(), xp.zeros_like(p["knots_amplitude_all"][:, :1])
    c = sm._compute_matrix_elements(sm.f_arr, Soms, Sa, *gal, kp, ka, convolve_window=False)
    diag = xp.stack([xp.abs(xp.nan_to_num(c[i].ravel())) for i in range(3)])  # (3, nf)
    del c
    scale = xp.sqrt(diag / (2.0 * sm.basis_settings.differential_component))
    del diag
    nf = sm.num_freqs
    data = xp.empty((num_walkers, 3, nf), dtype=xp.complex128)
    g = xp.random.default_rng(seed)
    for w in range(num_walkers):  # walker by walker: keeps the transient small
        data[w] = (g.standard_normal((3, nf)) + 1j * g.standard_normal((3, nf))) * (scale / np.sqrt(2))
    p["data_in_all"] = data.ravel()
    p["data_index_all"] = xp.arange(num_walkers, dtype=xp.int32)
    return p


class PeakTracker:
    """cupy allocator wrapper recording the pool's peak ``used_bytes``."""

    def __init__(self, cp):
        self.cp = cp
        self.pool = cp.get_default_memory_pool()
        self.peak = 0

    def __call__(self, size):
        mem = self.pool.malloc(size)
        u = self.pool.used_bytes()
        if u > self.peak:
            self.peak = u
        return mem

    def reset(self):
        self.peak = self.pool.used_bytes()


def bench_case(sm, inputs, convolve, repeats, warmup, cp, tracker):
    from cupy.cuda import runtime  # NOT `from cupy.cuda.runtime import ...` (see memory note)

    dev = cp.cuda.Device()
    pool = tracker.pool
    cp.fft.config.get_plan_cache().clear()
    pool.free_all_blocks()
    dev.synchronize()
    base_used = pool.used_bytes()
    free0, _ = runtime.memGetInfo()
    tracker.reset()
    times = []
    ll = None
    for i in range(warmup + repeats):
        dev.synchronize()
        t0 = time.perf_counter()
        ll = sm.compute_log_like(**inputs, convolve_window=convolve)
        dev.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    free1, _ = runtime.memGetInfo()
    res = dict(
        convolve_window=convolve,
        t_median_s=float(np.median(times)),
        t_min_s=float(np.min(times)),
        peak_pool_used_GB=tracker.peak / 1e9,
        peak_over_baseline_GB=(tracker.peak - base_used) / 1e9,
        pool_total_after_GB=pool.total_bytes() / 1e9,
        device_free_drop_GB=(free0 - free1) / 1e9,
        ll_finite=bool(cp.isfinite(ll).all()),
        ll_first=float(ll[0]),
    )
    del ll
    cp.fft.config.get_plan_cache().clear()
    pool.free_all_blocks()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="gal1_spl0_sym", choices=list(CONFIGS))
    ap.add_argument("--T-days", type=float, default=730.0)
    ap.add_argument("--dt", type=float, default=5.0)
    ap.add_argument("--min-freq", type=float, default=None, help="default: config's")
    ap.add_argument("--max-freq", type=float, default=None, help="default: config's")
    ap.add_argument("--taper-s", type=float, default=None,
                    help="Tukey taper duration [s]; default = production rule 1 / min_freq")
    ap.add_argument("--walkers", type=int, nargs="+", default=[8])
    ap.add_argument("--modes", type=int, nargs="+", default=[0, 1], help="convolve_window values (0/1)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--backend", default="cuda12x")
    ap.add_argument("--kw", nargs="*", default=[], help="extra XYZSensitivityBackend kwargs key=value")
    ap.add_argument("--json", default=None, help="append results as JSON lines to this file")
    a = ap.parse_args(argv)

    import cupy as cp

    cfg = copy.deepcopy(CONFIGS[a.config])
    cfg.update(T_days=a.T_days, dt=a.dt)
    if a.min_freq is not None:
        cfg["min_freq"] = a.min_freq
    if a.max_freq is not None:
        cfg["max_freq"] = a.max_freq
    cfg["window_taper_s"] = a.taper_s

    tracker = PeakTracker(cp)
    cp.cuda.set_allocator(tracker)

    t0 = time.perf_counter()
    sm = make_backend(cfg, a.backend, **_parse_kw(a.kw))
    print(
        f"config={a.config} T={a.T_days} d dt={a.dt} N={len(sm.window_values)} nf_active={sm.num_freqs} "
        f"n_alias={sm.n_alias} taper={window_taper_s(cfg):.4g}s (alpha={window_alpha(cfg):.3g}) extra_kw={_parse_kw(a.kw)} "
        f"init={time.perf_counter() - t0:.1f}s pool_after_init={tracker.pool.used_bytes() / 1e9:.2f} GB"
    )
    for nw in a.walkers:
        inputs = make_bench_inputs(sm, nw)
        cp.cuda.Device().synchronize()
        print(f"  walkers={nw}: resident inputs {tracker.pool.used_bytes() / 1e9:.2f} GB (pool used)")
        for m in a.modes:
            try:
                r = bench_case(sm, inputs, bool(m), a.repeats, a.warmup, cp, tracker)
            except cp.cuda.memory.OutOfMemoryError as e:
                r = dict(convolve_window=bool(m), error=f"OOM: {e}")
                cp.get_default_memory_pool().free_all_blocks()
            r.update(walkers=nw, T_days=a.T_days, dt=a.dt, config=a.config, kw=a.kw)
            if "error" in r:
                print(f"    convolve={bool(m)}: {r['error']}")
            else:
                print(
                    f"    convolve={bool(m)!s:5s} t_med={r['t_median_s']:.4f}s t_min={r['t_min_s']:.4f}s "
                    f"peak_pool={r['peak_pool_used_GB']:.3f} GB (+{r['peak_over_baseline_GB']:.3f} over baseline) "
                    f"pool_total_after={r['pool_total_after_GB']:.3f} GB dev_free_drop={r['device_free_drop_GB']:.3f} GB "
                    f"finite={r['ll_finite']}"
                )
            if a.json:
                with open(a.json, "a") as f:
                    f.write(json.dumps(r) + "\n")
        del inputs
        tracker.pool.free_all_blocks()


if __name__ == "__main__":
    main()
