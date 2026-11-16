"""Benchmark the exact galaxy convolution (agent 3) vs the legacy 6-entry convolution.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/window_conv/bench_agent3_galaxy.py \
        --T-days 730 --walkers 32 --legacy-walkers 8 --fmax None 0.03 0.02

Times (median of --repeats after 1 warm-up, device-synchronised) and peak cupy-pool usage
over the resident baseline for: ``galaxy_convolved_rows`` (per ``galaxy_fmax``) and the
legacy ``_compute_matrix_elements(..., convolve_window=True)`` (what ``compute_log_like``
runs today; galaxy-only inputs, same cost as the total).
"""

from __future__ import annotations

import argparse
import ast
import copy
import time

import numpy as np

import _use_worktree  # noqa: F401
from bench_2yr import PeakTracker
from fixtures import CONFIGS, FIDUCIAL, make_backend

from lisatools import convolution_helpers as sw


def timed(fn, cp, tracker, repeats):
    dev = cp.cuda.Device()
    cp.fft.config.get_plan_cache().clear()
    tracker.pool.free_all_blocks()
    dev.synchronize()
    base = tracker.pool.used_bytes()
    tracker.reset()
    ts = []
    for i in range(1 + repeats):
        dev.synchronize()
        t0 = time.perf_counter()
        out = fn()
        dev.synchronize()
        if i:
            ts.append(time.perf_counter() - t0)
        del out
    peak = (tracker.peak - base) / 1e9
    cp.fft.config.get_plan_cache().clear()
    tracker.pool.free_all_blocks()
    return float(np.median(ts)), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-days", type=float, default=730.0)
    ap.add_argument("--walkers", type=int, default=32)
    ap.add_argument("--legacy-walkers", type=int, nargs="*", default=[8])
    ap.add_argument("--fmax", nargs="*", default=["None", "0.03"])
    ap.add_argument("--batch", type=int, nargs="*", default=[8])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    import cupy as cp

    tracker = PeakTracker(cp)
    cp.cuda.set_allocator(tracker)
    cfg = copy.deepcopy(CONFIGS["gal1_spl0_sym"])
    cfg["T_days"] = a.T_days
    t0 = time.perf_counter()
    sm = make_backend(cfg, "cuda12x")
    print(f"T={a.T_days} d N={len(sm.window_values)} nf_active={sm.num_freqs} init {time.perf_counter() - t0:.1f}s")
    rng = np.random.default_rng(0)

    def params(n):
        return [cp.asarray(FIDUCIAL[k] * rng.uniform(0.9, 1.1, n)) for k in ("Amp", "alpha", "f_1", "kn", "f_2")]

    for fm in a.fmax:
        fmax = ast.literal_eval(fm)
        for b in a.batch:
            t0 = time.perf_counter()
            st = sw.get_galaxy_setup(sm, galaxy_fmax=fmax, batch_size=b)
            cp.cuda.Device().synchronize()
            ts = time.perf_counter() - t0
            p = params(a.walkers)
            t, peak = timed(lambda: sw.galaxy_convolved_rows(sm, *p, setup=st), cp, tracker, a.repeats)
            pl = st["plan"]
            print(f"  ours fmax={fmax} batch={b}: K={st['K']} n_sup={pl['n_sup']} M={pl['M']} n_gal={pl['n_gal']} "
                  f"setup {ts:.2f}s | {a.walkers} walkers: {t:.3f}s, peak {peak:.2f} GB "
                  f"(output {a.walkers * pl['n_gal'] * 8 / 1e9:.2f} GB)")
    z = None
    for nw in a.legacy_walkers:
        p = params(nw)
        z = cp.zeros(nw)
        try:
            t, peak = timed(lambda: sm._compute_matrix_elements(sm.f_arr, z, z, *p, convolve_window=True),
                            cp, tracker, max(1, a.repeats - 2))
            print(f"  legacy {nw} walkers: {t:.3f}s, peak {peak:.2f} GB")
        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"  legacy {nw} walkers: OOM ({e})")


if __name__ == "__main__":
    main()
